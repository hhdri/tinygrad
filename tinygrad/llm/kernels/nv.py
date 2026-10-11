from __future__ import annotations
import functools
from tinygrad import Tensor, UOp, Device, Context, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops
from tinygrad.renderer.cstyle import CUDARenderer

# NVIDIA (sm_61+, dp4a) kernels over packed ggml weights, rendered by the CUDA C renderer for the NV and CUDA devices.
# Decode quantizes the activations once to q8 (per-32 scale, per-16 sums of the quantized ints), then one warp per output row
# walks the whole row: 8 lanes per 256-weight super-block, 16-byte weight loads and dp4a, like llama.cpp's mmvq.
Q4_K, Q6_K = 12, 14
NV_QUANTS = {Q4_K: 144, Q6_K: 210}  # bytes per 256 weights
ROWS_PER_BLOCK = 4

@functools.cache
def nv_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if getenv("DISABLE_NV_KERNELS"): return False
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] not in ("NV", "CUDA"): return False
  with Context(ALLOW_DEVICE_USAGE=1):
    return isinstance(r:=Device[device].renderer, CUDARenderer) and int(r.target.arch[3:]) >= 61

def _custom(fmt:str, dtype, *src:UOp) -> UOp: return UOp(Ops.CUSTOM, src=src, arg=(fmt, dtype))
def _shfl_xor(val:UOp, offset:int) -> UOp: return _custom(f"__shfl_xor_sync(0xffffffffu, {{0}}, {offset})", val.dtype, val)
def warp_reduce(val:UOp, maximum:bool=False, width:int=32) -> UOp:
  offset = width//2
  while offset:
    val = val.maximum(_shfl_xor(val, offset)) if maximum else val + _shfl_xor(val, offset)
    offset //= 2
  return val
# CUSTOM, not CUSTOMI: an inline CUSTOMI is re-rendered at every use
def _dp4a(a:UOp, b:UOp, c:UOp) -> UOp: return _custom("__dp4a({0}, {1}, {2})", dtypes.int32, a.bitcast(dtypes.int32), b.bitcast(dtypes.int32), c)
def _vload(buf:UOp, idx:UOp, lanes:int) -> list[UOp]:
  vec = UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(lanes))).load()
  return [vec[i] for i in range(lanes)]
def _half(bits:UOp) -> UOp: return bits.cast(dtypes.uint16).bitcast(dtypes.float16).float()
def _u32(raw:UOp, idx:UOp) -> UOp: return raw[idx].load().cast(dtypes.uint32) | (raw[idx+1].load().cast(dtypes.uint32) << 16)

# ******** q8 activations ********

@functools.cache
def _q8_quantize_kernel(xq:UOp, xd:UOp, xs:UOp, x:UOp, tokens:int, in_features:int) -> UOp:
  groups = in_features//32
  token_group, lane = UOp.range(tokens*groups, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  token, group = token_group//groups, token_group%groups
  value = x.reshape(tokens, groups, 32)[token, group, lane].load().float()
  # keep the division: reciprocal multiplication can move values across rounding ties
  d = _custom("({0}/127.0f)", dtypes.float, warp_reduce(value.abs(), maximum=True)).maximum(1e-30)
  quant = _custom("rintf({0}/{1})", dtypes.float, value, d).clip(-127, 127)
  word = quant.cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << ((lane%4)*8).cast(dtypes.uint32)
  for offset in (1, 2): word = word | _shfl_xor(word, offset)
  stores = (xq[token, group*8 + (lane//4).valid((lane%4).eq(0))].store(word), xd[token, group.valid(lane.eq(0))].store(d),
            xs[token, group*2 + (lane//16).valid((lane%16).eq(0))].store(warp_reduce(quant, width=16)))
  return UOp.group(*stores).end(token_group, lane).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int) -> tuple[Tensor, Tensor, Tensor]:
  xq = Tensor.empty(tokens, in_features//4, dtype=dtypes.uint32, device=x.device)
  xd = Tensor.empty(tokens, in_features//32, dtype=dtypes.float32, device=x.device)
  xs = Tensor.empty(tokens, in_features//16, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features)
  xq, xd, xs = Tensor.custom_kernel(xq, xd, xs, x.contiguous(), fxn=fxn)[:3]
  return xq, xd, xs

# ******** decode: dp4a gemv ********

def _gemv(out:UOp, tokens:int, out_features:int, in_features:int, lane_dot, name:str) -> UOp:
  row_block, lane, warp = UOp.range(tokens*out_features//ROWS_PER_BLOCK, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), \
    UOp.range(ROWS_PER_BLOCK, 2, AxisType.LOCAL)
  token_row = row_block*ROWS_PER_BLOCK + warp
  token, row = token_row//out_features, token_row%out_features
  acc = UOp.const(0, dtypes.float32)
  for step in range(in_features//1024): acc = acc + lane_dot(token, row, step*4 + lane//8, lane%8)
  return out[token, row.valid(lane.eq(0))].store(warp_reduce(acc)).end(row_block, lane, warp).sink(arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q4k_gemv_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, tokens:int, out_features:int, in_features:int) -> UOp:
  def lane_dot(token:UOp, row:UOp, block:UOp, t:UOp) -> UOp:
    # 16 qs bytes per lane: the low nibbles are sub-block 2j, the high nibbles 2j+1
    j, h = t//2, t%2
    base = (row*(in_features//256) + block) * 36
    head, s0, s1, s2 = _vload(raw, base, 4)
    qs = _vload(raw, base + 4 + j*8 + h*4, 4)
    y = block*256 + j*64 + h*16
    xlo, xhi = _vload(xq, token*xq.shape[1] + y//4, 4), _vload(xq, token*xq.shape[1] + (y+32)//4, 4)
    dlo = dhi = UOp.const(0, dtypes.int32)
    for i in range(4): dlo, dhi = _dp4a(qs[i] & 0x0f0f0f0f, xlo[i], dlo), _dp4a((qs[i] >> 4) & 0x0f0f0f0f, xhi[i], dhi)
    # get_scale_min_k4 for sub-blocks 2j and 2j+1
    def scale_min(shift:UOp) -> tuple[UOp, UOp]:
      a, b, c = (s0 >> shift) & 255, (s1 >> shift) & 255, (s2 >> shift) & 255
      return (j < 2).where(a & 63, (c & 15) | ((a >> 6) << 4)).float(), (j < 2).where(b & 63, (c >> 4) | ((b >> 6) << 4)).float()
    (sclo, mnlo), (schi, mnhi) = scale_min(((j%2)*16).cast(dtypes.uint32)), scale_min(((j%2)*16+8).cast(dtypes.uint32))
    xdlo, xdhi, xslo, xshi = xd[token, y//32].load(), xd[token, y//32+1].load(), xs[token, y//16].load(), xs[token, y//16+2].load()
    return _half(head & 0xffff) * (sclo*dlo.float()*xdlo + schi*dhi.float()*xdhi) - _half(head >> 16) * (mnlo*xslo*xdlo + mnhi*xshi*xdhi)
  return _gemv(out, tokens, out_features, in_features, lane_dot, "q4k_gemv")

@functools.cache
def _q6k_gemv_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, tokens:int, out_features:int, in_features:int) -> UOp:
  def lane_dot(token:UOp, row:UOp, block:UOp, t:UOp) -> UOp:
    # 210-byte blocks are only halfword aligned. Lane t covers 16 ql bytes of half n, quarter c:
    # the low nibbles are y[128n+32c+l], the high nibbles y[128n+64+32c+l], qh adds bits 4-5
    n, c, col = t//4, (t//2)%2, t%2
    base = (row*(in_features//256) + block) * 105
    ql = [_u32(raw, base + n*32 + c*16 + col*8 + i*2) for i in range(4)]
    qh = [_u32(raw, base + 64 + n*16 + col*8 + i*2) for i in range(4)]
    shift = (c*2).cast(dtypes.uint32)
    y = block*256 + n*128 + c*32 + col*16
    xlo, xhi = _vload(xq, token*xq.shape[1] + y//4, 4), _vload(xq, token*xq.shape[1] + (y+64)//4, 4)
    dlo = dhi = UOp.const(0, dtypes.int32)
    for i in range(4):
      dlo = _dp4a((ql[i] & 0x0f0f0f0f) | (((qh[i] >> shift) & 0x03030303) << 4), xlo[i], dlo)
      dhi = _dp4a(((ql[i] >> 4) & 0x0f0f0f0f) | (((qh[i] >> (shift+4)) & 0x03030303) << 4), xhi[i], dhi)
    s = n*8 + c*2 + col  # scale of y//16; the high nibbles use s+4
    scales = raw[base + 96 + s//2].load(), raw[base + 98 + s//2].load()
    sclo, schi = ((sc >> ((s%2)*8).cast(dtypes.uint16)).cast(dtypes.uint8).bitcast(dtypes.int8).float() for sc in scales)
    xdlo, xdhi, xslo, xshi = xd[token, y//32].load(), xd[token, y//32+2].load(), xs[token, y//16].load(), xs[token, y//16+4].load()
    # the -32 offset of the 6-bit quants is folded into the integer dot through the activation sums
    return _half(raw[base + 104].load()) * (sclo*(dlo.float() - 32*xslo)*xdlo + schi*(dhi.float() - 32*xshi)*xdhi)
  return _gemv(out, tokens, out_features, in_features, lane_dot, "q6k_gemv")

def nv_linear_supported(ggml_type:int|None, in_features:int, out_features:int) -> bool:
  return ggml_type in NV_QUANTS and in_features % 1024 == 0 and out_features % ROWS_PER_BLOCK == 0

def nv_q8_linear(weight:Tensor, ggml_type:int, x:Tensor, in_features:int, out_features:int) -> Tensor:
  tokens = int(x.numel()) // in_features
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_q4k_gemv_kernel if ggml_type == Q4_K else _q6k_gemv_kernel, tokens=tokens, out_features=out_features,
                          in_features=in_features)
  return Tensor.custom_kernel(out, weight, *q8_quantize(x, tokens, in_features), fxn=fxn)[0].reshape(*x.shape[:-1], out_features)
