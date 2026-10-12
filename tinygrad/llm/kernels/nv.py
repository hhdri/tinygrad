from __future__ import annotations
import functools, math
from tinygrad import Tensor, UOp, Device, Context, dtypes
from tinygrad.helpers import getenv, prod
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, AddrSpace
from tinygrad.renderer.cstyle import CUDARenderer

# NVIDIA (sm_61+, dp4a) kernels over packed ggml weights, rendered by the CUDA C renderer for the NV and CUDA devices.
# Decode quantizes the activations once to q8 (per-32 scale, per-16 sums of the quantized ints), then one warp per output row
# walks the whole row: 8 lanes per 256-weight super-block, 16-byte weight loads and dp4a, like llama.cpp's mmvq.
Q4_K, Q6_K = 12, 14
NV_QUANTS = {Q4_K: 144, Q6_K: 210}  # bytes per 256 weights
ROWS_PER_BLOCK = 4
MAGIC = 12582912.0  # 1.5*2**23

@functools.cache
def nv_custom_kernels_supported(device:str|tuple[str, ...]|None) -> bool:
  if getenv("DISABLE_NV_KERNELS"): return False
  if isinstance(device, tuple): device = device[0]
  if device is None or device.split(":")[0] not in ("NV", "CUDA"): return False
  with Context(ALLOW_DEVICE_USAGE=1):
    return isinstance(r:=Device[device].renderer, CUDARenderer) and int(r.target.arch[3:]) >= 61

@functools.cache
def nv_mmq_supported(device:str|tuple[str, ...]|None) -> bool:
  if not isinstance(device, str) or not nv_custom_kernels_supported(device): return False
  with Context(ALLOW_DEVICE_USAGE=1): return int(Device[device].renderer.target.arch[3:]) >= 80

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
def pad_unwritten(t:Tensor, shape:tuple[int, ...]) -> Tensor:
  # t (of symbolic size) as the leading slice of an uninitialized buffer of shape: the producing kernel runs over t only, where pad_to
  # would also write the zeros, for consumers that skip the rows past the live ones
  buf = Tensor.empty(*shape, dtype=t.dtype, device=t.device)
  return Tensor(buf.uop.after(buf.uop.shrink(tuple((0, s) for s in t.shape)).store(t.uop)))

def _fma(a:UOp, b:UOp, c:UOp) -> UOp: return _custom("fmaf({0}, {1}, {2})", dtypes.float32, a, b, c)
def _half(bits:UOp) -> UOp: return bits.cast(dtypes.uint16).bitcast(dtypes.float16).float()
def _u32(raw:UOp, idx:UOp) -> UOp: return raw[idx].load().cast(dtypes.uint32) | (raw[idx+1].load().cast(dtypes.uint32) << 16)

def _tile(n:int, *sizes:int) -> int: return next(s for s in sizes if n % s == 0)

# ******** q8 activations ********

@functools.cache
def _q8_quantize_kernel(xq:UOp, xd:UOp, xs:UOp, x:UOp, *up:UOp, tokens:int, in_features:int, live:UOp|None=None,
                        group_major:bool=True) -> UOp:
  # each lane loads four consecutive values and owns one packed word: a warp covers four groups of 32, the group maximum
  # reduces over 8 lanes and the per-16 sums over 4. Several warps per block keep the occupancy up.
  chunks = in_features//128
  warps = _tile(tokens*chunks, 8, 4, 2, 1)
  # a padded symbolic chunk quantizes only its live tokens (rounded up to whole blocks); the rows past them stay unwritten
  blocks = tokens*chunks//warps if live is None else (live.unbind_all()[0]*chunks+warps-1)//warps
  block, lane, warp = UOp.range(blocks, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), \
    UOp.range(warps, 1, AxisType.LOCAL)
  token_chunk = block*warps + warp
  token, chunk = token_chunk//chunks, token_chunk%chunks
  values = [v.float() for v in _vload(x, token*in_features + chunk*128 + lane*4, 4)]
  # with up, x is a SwiGLU gate: quantize silu(x)*up
  if up: values = [v.silu()*u.float() for v, u in zip(values, _vload(up[0], token*in_features + chunk*128 + lane*4, 4))]
  amax = functools.reduce(UOp.maximum, (v.abs() for v in values))
  # keep the division: reciprocal multiplication can move values across rounding ties
  d = _custom("({0}/127.0f)", dtypes.float, warp_reduce(amax, maximum=True, width=8)).maximum(1e-30)
  quants = [_custom("rintf({0}/{1})", dtypes.float, v, d).clip(-127, 127) for v in values]
  word = functools.reduce(UOp.bitwise_or, (q.cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i*8) for i,q in enumerate(quants)))
  group, w = chunk*4 + lane//8, lane%8
  # group major scales and sums: a Q4_K MMQ token block reads a group's values contiguously (Q6_K's staging prefers token major)
  def at(buf:UOp, g:UOp) -> UOp: return buf[g, token] if group_major else buf[token, g]
  stores = (xq[token, chunk*32 + lane].store(word), at(xd, group.valid(w.eq(0))).store(d),
            at(xs, group*2 + (w//4).valid((w%4).eq(0))).store(warp_reduce(sum(quants[1:], quants[0]), width=4)))
  return UOp.group(*stores).end(block, lane, warp).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int, cache:dict|None=None, up:Tensor|None=None,
                live:UOp|None=None, group_major:bool=True) -> tuple[Tensor, Tensor, Tensor]:
  # cache: shared by projections of the same input (Linear.project), which then quantize it once
  # up: x is a SwiGLU gate, the quantized input is silu(x)*up (never written out)
  if cache is not None and up is None and (hit:=cache.get((tokens, in_features, group_major))) is not None: return hit
  xq = Tensor.empty(tokens, in_features//4, dtype=dtypes.uint32, device=x.device)
  xd = Tensor.empty(*((in_features//32, tokens) if group_major else (tokens, in_features//32)), dtype=dtypes.float32, device=x.device)
  xs = Tensor.empty(*((in_features//16, tokens) if group_major else (tokens, in_features//16)), dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features, live=live, group_major=group_major)
  if live is not None: xq = Tensor(xq.uop.after(live))  # carries the live-token binding into the kernel
  xq, xd, xs = Tensor.custom_kernel(xq, xd, xs, x.contiguous(), *(() if up is None else (up.contiguous(),)), fxn=fxn)[:3]
  if cache is not None and up is None: cache[(tokens, in_features, group_major)] = (xq, xd, xs)
  return xq, xd, xs

# ******** decode: dp4a gemv ********

def _gemv(out:UOp, tokens:int, out_features:int, in_features:int, lane_dot, name:str, res:tuple[UOp, ...]=()) -> UOp:
  # res: a residual (tokens, out_features) added to the product in the store
  row_block, lane, warp = UOp.range(tokens*out_features//ROWS_PER_BLOCK, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), \
    UOp.range(ROWS_PER_BLOCK, 2, AxisType.LOCAL)
  token_row = row_block*ROWS_PER_BLOCK + warp
  token, row = token_row//out_features, token_row%out_features
  acc = UOp.const(0, dtypes.float32)
  for step in range(in_features//1024): acc = acc + lane_dot(token, row, step*4 + lane//8, lane%8)
  val = warp_reduce(acc) + res[0][token, row].load() if res else warp_reduce(acc)
  return out[token, row.valid(lane.eq(0))].store(val).end(row_block, lane, warp).sink(arg=KernelInfo(name=name, opts_to_apply=()))

@functools.cache
def _q4k_gemv_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *res:UOp, tokens:int, out_features:int, in_features:int) -> UOp:
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
    xdlo, xdhi, xslo, xshi = xd[y//32, token].load(), xd[y//32+1, token].load(), xs[y//16, token].load(), xs[y//16+2, token].load()
    return _half(head & 0xffff) * (sclo*dlo.float()*xdlo + schi*dhi.float()*xdhi) - _half(head >> 16) * (mnlo*xslo*xdlo + mnhi*xshi*xdhi)
  return _gemv(out, tokens, out_features, in_features, lane_dot, "q4k_gemv", res)

@functools.cache
def _q4k_multi_gemv_kernel(*bufs:UOp, n:int, rows:tuple[int, ...], in_features:int) -> UOp:
  # several Q4_K projections of one token in one launch (bufs: n outputs, n packed weights, xq, xd, xs): rows are numbered across
  # the matrices, each warp takes one row and reads its matrix through gated loads (the other matrices' loads are predicated off)
  outs, raws, (xq, xd, xs) = bufs[:n], bufs[n:2*n], bufs[2*n:]
  starts = [sum(rows[:m]) for m in range(n)]
  def sel(row:UOp, m:int) -> UOp: return (row >= starts[m]) & (row < starts[m] + rows[m]) if n > 1 else UOp.const(True, dtypes.bool)
  def lane_dot(token:UOp, row:UOp, block:UOp, t:UOp) -> UOp:
    j, h = t//2, t%2
    def words(off:int, cnt:int) -> list[UOp]:
      vals = [[raws[m][(((row-starts[m])*(in_features//256) + block)*36 + off + i).valid(sel(row, m))].load() for i in range(cnt)]
              for m in range(n)]
      return [functools.reduce(lambda acc, m: sel(row, m).where(vals[m][i], acc), range(n-2, -1, -1), vals[n-1][i]) for i in range(cnt)]
    head, s0, s1, s2 = words(0, 4)
    qv = [[raws[m][(((row-starts[m])*(in_features//256) + block)*36 + 4 + j*8 + h*4 + i).valid(sel(row, m))].load() for i in range(4)]
          for m in range(n)]
    qs = [functools.reduce(lambda acc, m: sel(row, m).where(qv[m][i], acc), range(n-2, -1, -1), qv[n-1][i]) for i in range(4)]
    y = block*256 + j*64 + h*16
    xlo, xhi = _vload(xq, token*xq.shape[1] + y//4, 4), _vload(xq, token*xq.shape[1] + (y+32)//4, 4)
    dlo = dhi = UOp.const(0, dtypes.int32)
    for i in range(4): dlo, dhi = _dp4a(qs[i] & 0x0f0f0f0f, xlo[i], dlo), _dp4a((qs[i] >> 4) & 0x0f0f0f0f, xhi[i], dhi)
    def scale_min(shift:UOp) -> tuple[UOp, UOp]:
      a, b, c = (s0 >> shift) & 255, (s1 >> shift) & 255, (s2 >> shift) & 255
      return (j < 2).where(a & 63, (c & 15) | ((a >> 6) << 4)).float(), (j < 2).where(b & 63, (c >> 4) | ((b >> 6) << 4)).float()
    (sclo, mnlo), (schi, mnhi) = scale_min(((j%2)*16).cast(dtypes.uint32)), scale_min(((j%2)*16+8).cast(dtypes.uint32))
    xdlo, xdhi, xslo, xshi = xd[y//32, token].load(), xd[y//32+1, token].load(), xs[y//16, token].load(), xs[y//16+2, token].load()
    return _half(head & 0xffff) * (sclo*dlo.float()*xdlo + schi*dhi.float()*xdhi) - _half(head >> 16) * (mnlo*xslo*xdlo + mnhi*xshi*xdhi)
  total = sum(rows)
  row_block, lane, warp = UOp.range(total//ROWS_PER_BLOCK, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), \
    UOp.range(ROWS_PER_BLOCK, 2, AxisType.LOCAL)
  row = row_block*ROWS_PER_BLOCK + warp
  token = UOp.const(0, dtypes.weakint)
  acc = UOp.const(0, dtypes.float32)
  for step in range(in_features//1024): acc = acc + lane_dot(token, row, step*4 + lane//8, lane%8)
  val = warp_reduce(acc)
  stores = [outs[m][0, (row-starts[m]).valid(sel(row, m) & lane.eq(0))].store(val) for m in range(n)]
  return UOp.group(*stores).end(row_block, lane, warp).sink(arg=KernelInfo(name="q4k_gemv", opts_to_apply=()))

def nv_q8_multi_linear(weights:list[Tensor], x:Tensor, in_features:int, out_features:list[int], q8:dict|None=None) -> list[Tensor]:
  # one decode token through several packed Q4_K weights in one launch (see _q4k_multi_gemv_kernel)
  outs = [Tensor.empty(1, o, dtype=dtypes.float32, device=x.device) for o in out_features]
  fxn = functools.partial(_q4k_multi_gemv_kernel, n=len(weights), rows=tuple(out_features), in_features=in_features)
  res = Tensor.custom_kernel(*outs, *weights, *q8_quantize(x, 1, in_features, q8), fxn=fxn)[:len(weights)]
  return [r.reshape(*x.shape[:-1], o) for r, o in zip(res, out_features)]

@functools.cache
def _q6k_gemv_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, *res:UOp, tokens:int, out_features:int, in_features:int) -> UOp:
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
    xdlo, xdhi, xslo, xshi = xd[y//32, token].load(), xd[y//32+2, token].load(), xs[y//16, token].load(), xs[y//16+4, token].load()
    # the -32 offset of the 6-bit quants is folded into the integer dot through the activation sums
    return _half(raw[base + 104].load()) * (sclo*(dlo.float() - 32*xslo)*xdlo + schi*(dhi.float() - 32*xshi)*xdhi)
  return _gemv(out, tokens, out_features, in_features, lane_dot, "q6k_gemv", res)

def nv_linear_supported(ggml_type:int|None, in_features:int, out_features:int) -> bool:
  return ggml_type in NV_QUANTS and in_features % 1024 == 0 and out_features % ROWS_PER_BLOCK == 0

def nv_q8_linear(weight:Tensor, ggml_type:int, x:Tensor, in_features:int, out_features:int, q8:dict|None=None,
                 up:Tensor|None=None, residual:Tensor|None=None) -> Tensor:
  # residual: added to the product by the kernel's store
  tokens = int(x.numel()) // in_features
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_q4k_gemv_kernel if ggml_type == Q4_K else _q6k_gemv_kernel, tokens=tokens, out_features=out_features,
                          in_features=in_features)
  res = () if residual is None else (residual.reshape(tokens, out_features).float().contiguous(),)
  return Tensor.custom_kernel(out, weight, *q8_quantize(x, tokens, in_features, q8, up), *res, fxn=fxn)[0].reshape(*x.shape[:-1], out_features)

# ******** dense decode ********

@functools.cache
def _f16_gemv_kernel(out:UOp, weight:UOp, x:UOp, in_features:int, out_features:int) -> UOp:
  row_block, lane, warp = UOp.range(out_features//ROWS_PER_BLOCK, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), \
    UOp.range(ROWS_PER_BLOCK, 2, AxisType.LOCAL)
  row = row_block*ROWS_PER_BLOCK + warp
  acc = UOp.const(0, dtypes.float32)
  for step in range(in_features//256):
    idx = step*256 + lane*8
    w, v = _vload(weight, row*in_features + idx, 8), _vload(x, idx, 8)
    for j in range(8): acc = acc + w[j].float()*v[j].float()
  return out[row.valid(lane.eq(0))].store(warp_reduce(acc)).end(row_block, lane, warp).sink(
    arg=KernelInfo(name="nv_f16_gemv", opts_to_apply=()))

def nv_f16_gemv(weight:Tensor, x:Tensor, in_features:int, out_features:int) -> Tensor:
  out = Tensor.empty(out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_f16_gemv_kernel, in_features=in_features, out_features=out_features)
  return Tensor.custom_kernel(out, weight.contiguous(), x.contiguous(), fxn=fxn)[0].reshape(*x.shape[:-1], out_features)

# ******** prefill: dense F16 weights ********

GEMM_TN, GEMM_TM, GEMM_TK = 64, 64, 64

@functools.cache
def _f16_gemm_kernel(out:UOp, w:UOp, x:UOp, tokens:int, out_features:int, in_features:int, live:UOp|None=None) -> UOp:
  # out[t, n] = sum_k x[t, k] w[n, k] on m16n8k16 half MMAs with float accumulation. A block owns 64 rows x 64 tokens, 4 warps of
  # 32 x 32; K goes in 64-half stages through two shared buffers: the next stage's cp.async copies are issued before this stage's
  # MMAs (16-byte chunks XOR-swizzled by row, ldmatrix fragments). A padded symbolic chunk launches its live token blocks only.
  TN, TM, TK, threads = GEMM_TN, GEMM_TM, GEMM_TK, 128
  n = None if live is None else live.unbind_all()[0]
  token_block = UOp.range(tokens//TM if n is None else (n+TM-1)//TM, 1, AxisType.GLOBAL)
  row_block, lane, warp = UOp.range(out_features//TN, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), UOp.range(4, 2, AxisType.LOCAL)
  tid, g, t = warp*32 + lane, lane//4, lane%4
  row0, tok0, wr, tc = row_block*TN, token_block*TM, (warp//2)*32, (warp%2)*32
  words, chunks, stages = TK//2, TK//8, in_features//TK
  bufs = [(UOp.alloc((TN, words), dtypes.uint32, addrspace=AddrSpace.LOCAL), UOp.alloc((TM, words), dtypes.uint32, addrspace=AddrSpace.LOCAL))
          for _ in range(2)]
  def swz(r:UOp, c:UOp) -> UOp: return r*words + (c ^ (r%8))*4
  def copies(stage:UOp, ws:UOp, xs:UOp) -> list[UOp]:
    ops = []
    for i in range((TN+TM)*chunks//threads):
      idx = tid + i*threads
      r, c = idx//chunks % TN, idx%chunks
      dst, src, base = (ws, w, row0) if i < TN*chunks//threads else (xs, x, tok0)
      ops.append(_cp_async(dst.flatten()[swz(r, c)], src.flatten()[(base+r)*in_features + stage*TK + c*8], UOp.const(16, dtypes.int)))
    return ops
  def compute(ws:UOp, xs:UOp, cur:UOp, axis:int, pre:list[UOp]) -> UOp:
    # pre: the next stage's copies, issued before this loop (which waits on them through the AFTER)
    ks = UOp.range(TK//16, axis, AxisType.LOOP)
    c = cur.after(ks, *pre)
    vals = [c[i].load() for i in range(32)]
    af = [_ldm(ws, swz(wr + mt*16 + lane%8 + (lane//8%2)*8, ks*2 + lane//16), 4) for mt in range(2)]
    bf = [_ldm(xs, swz(tc + np_*16 + lane%8 + (lane//16)*8, ks*2 + lane//8%2), 4) for np_ in range(2)]
    for nt in range(4):
      b = UOp.stack(bf[nt//2][(nt%2)*2], bf[nt//2][(nt%2)*2+1]).bitcast(dtypes.half)
      for mt in range(2):
        o = (mt*4+nt)*4
        res = UOp.wmma(UOp.stack(*(af[mt][j] for j in range(4))).bitcast(dtypes.half), b, UOp.stack(*vals[o:o+4]), (8,16,16), 32)
        vals[o:o+4] = [res[i] for i in range(4)]
    return cur.after(c.store(UOp.stack(*vals)).end(ks))
  acc = UOp.alloc((32,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  zero = UOp.const(0, dtypes.weakint)
  b0 = _cp_wait(*copies(zero, *bufs[0])).barrier()
  it = UOp.range(stages//2, 3, AxisType.LOOP)
  (w0, x0), (w1, x1) = [(a.after(b0, it), b.after(b0, it)) for a, b in bufs]
  cur = acc.after(it)
  # stage 2it from buffer 0 while stage 2it+1 copies into buffer 1, then the reverse (the last copy re-reads stage 0, never used)
  c1 = copies(it*2+1, w1, x1)
  st0 = compute(w0, x0, cur, 10, c1)
  b1 = _cp_wait(*c1, st0[0].load()).barrier()
  w0b, x0b, w1b, x1b = (u.after(b1) for u in (w0, x0, w1, x1))
  c2 = copies(((it+1)*2) % stages, w0b, x0b)
  st1 = compute(w1b, x1b, st0.after(b1), 11, c2)
  b2 = _cp_wait(*c2, st1[0].load()).barrier()
  acc = acc.after(b2.end(it))
  def orow(tk:UOp) -> UOp: return tk if n is None else tk.valid(tk < tokens)
  outs = [out[orow(tok0+tc+nt*8+2*t+i%2), row0+wr+mt*16+g+(i//2)*8].store(acc[(mt*4+nt)*4+i].load())
          for mt in range(2) for nt in range(4) for i in range(4)]
  return UOp.group(*outs).end(token_block, row_block, lane, warp).sink(arg=KernelInfo(name="f16_gemm", opts_to_apply=()))

def nv_f16_gemm_supported(weight:Tensor, x:Tensor, in_features:int, out_features:int) -> bool:
  return nv_mmq_supported(weight.device) and in_features % (2*GEMM_TK) == 0 and out_features % GEMM_TN == 0 and \
    prod(x.max_shape)//in_features % GEMM_TM == 0

def nv_f16_gemm(weight:Tensor, x:Tensor, in_features:int, out_features:int, live:UOp|None=None) -> Tensor:
  # x: (tokens, in_features) padded to whole 64-token blocks (a symbolic chunk to its max); weight: (out_features, in_features) half
  tokens = int(x.numel())//in_features
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
  xh = x.reshape(tokens, in_features).cast(dtypes.half).contiguous()  # a half input (e.g. written for its live rows only) passes as is
  if live is not None: xh = Tensor(xh.uop.after(live))  # carries the live-token binding into the kernel
  fxn = functools.partial(_f16_gemm_kernel, tokens=tokens, out_features=out_features, in_features=in_features, live=live)
  return Tensor.custom_kernel(out, weight.contiguous(), xh, fxn=fxn)[0]

# ******** prefill: integer tensor cores ********

@functools.cache
def _mmq_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, tokens:int, out_features:int, in_features:int, ggml_type:int,
                live:UOp|None=None, small_tail:int|None=None) -> UOp:
  # Each warp owns row_tiles 16-row x token_tiles 8-token MMA tiles, so every weight fragment feeds token_tiles MMAs and every
  # activation fragment row_tiles MMAs. The block stages 64 weights per row at a time: it unpacks its rows to int8 words in
  # shared memory, decodes each row's group coefficients once, and stages the activations next to them.
  # Q4_K groups (32) match m16n8k32; Q6_K scales every 16 weights, so it uses m16n8k16.
  q4 = ggml_type == Q4_K
  group_size, words = (32, 36) if q4 else (16, 105)
  # weights per row per stage: 128-weight stages halve the barriers per K and their 90 KB need dynamic shared memory. Measured on an
  # RTX 4060 at 512 tokens, they speed up Q4_K with out_features <= in_features (attention 4096x4096 -7%, FFN down -6%) but slow
  # the wide gate/up shapes (+5%) and Q6_K (+11%)
  SK = 128 if q4 and out_features <= in_features else 64
  parts, g64 = SK//64, 64//group_size  # 64-weight parts per stage, groups per part
  groups = SK//group_size
  # Q6_K's k16 fragments and per-16 coefficients need more registers per MMA; with a long K (e.g. FFN down, 12288 -> 4096) 32 rows
  # per warp leave room to hide latency (10% faster there), while short-K shapes keep the larger tile
  narrow = not q4 and in_features > out_features
  row_tiles = _tile(out_features//16, *((2, 1) if narrow else (4, 2, 1)))
  row_warps = _tile(out_features//(16*row_tiles), 2, 1)
  token_tiles = _tile(tokens//8, 4, 2, 1)
  token_warps = _tile(tokens//(8*token_tiles), 4, 2, 1)
  TN, TM, threads = 16*row_tiles*row_warps, 8*token_tiles*token_warps, 32*row_warps*token_warps
  # a padded symbolic chunk launches only the token blocks holding its live tokens. Token blocks vary fastest (the last global
  # axis is the launch's x), so the blocks sharing a weight tile run together and read it from L2: with the row blocks fastest,
  # each weight tile came from DRAM once per token block (gate/up at 512 tokens read 116 MB instead of 31)
  token_block = UOp.range(tokens//TM if live is None else _live_blocks(live.unbind_all()[0], TM, small_tail), 1, AxisType.GLOBAL)
  row_block, lane = UOp.range(out_features//TN, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  warp = UOp.range(row_warps*token_warps, 2, AxisType.LOCAL)
  tid, g, t = warp*32+lane, lane//4, lane%4
  row0, tok0 = row_block*TN, token_block*TM
  wrow, wtok = (warp//token_warps)*(16*row_tiles), (warp%token_warps)*(8*token_tiles)
  # 20-word rows: the fragment reads of 8 rows x 4 words fall in distinct banks, and 16-byte stores stay aligned
  # double buffered: iteration s computes from buffer s%2 while it stages s+1 into the other one; the barrier that ends each
  # iteration orders both the stores before the next iteration's reads and the reads before the stores that overwrite them
  w_s = UOp.alloc((2, TN, SK//4+4), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  x_s = UOp.alloc((2, TM, SK//4+4), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  # Q4_K: (scale*d, min*dmin) per row and (d, -(sum of quants)*d) per token; Q6_K: scale*d and d
  c_s = UOp.alloc((2, groups, TN, 2 if q4 else 1), dtypes.float32, addrspace=AddrSpace.LOCAL)
  t_s = UOp.alloc((2, groups, TM, 2 if q4 else 1), dtypes.float32, addrspace=AddrSpace.LOCAL)
  acc = UOp.alloc((row_tiles*token_tiles*4,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  stages = in_features//SK

  def stage_stores(stage:UOp, buf:UOp|int) -> list[UOp]:
    stores:list[UOp] = []
    for p in range(parts): stage_part(stage*parts+p, p, buf, stores)
    return stores

  def stage_part(part64:UOp, p:int, buf:UOp|int, stores:list[UOp]) -> None:
    # one 64-weight part: words p*16.. and groups p*g64.. of the stage
    sb, part = part64//4, part64%4
    def coop(n:int, fn) -> None:
      # spread n items over the block's threads; a partial last round masks its stores
      for i in range((n+threads-1)//threads):
        idx = tid+i*threads
        stores.extend(fn(idx.minimum(n-1) if n % threads else idx, idx < n if n % threads else None))
    def masked(i:UOp, valid:UOp|None) -> UOp: return i if valid is None else i.valid(valid)
    def row_base(r:UOp) -> UOp: return ((row0+r)*(in_features//256)+sb)*words

    if q4:
      def weights(i, v):
        r, quarter = i//2, i%2
        qs = _vload(raw, row_base(r)+4+part*8+quarter*4, 4)
        return [w_s[buf, masked(r, v), p*16+h*8+quarter*4+k].store((qs[k] >> (h*4)) & 0x0f0f0f0f) for h in range(2) for k in range(4)]
      coop(TN*2, weights)
      def coefficients(i, v):
        sub, r = i//TN, i%TN
        head, s0, s1, s2 = _vload(raw, row_base(r), 4)
        a, b, c = ((x >> (((part*2+sub)%4)*8).cast(dtypes.uint32)) & 255 for x in (s0, s1, s2))
        low = part < 2
        scale, minimum = low.where(a & 63, (c & 15) | ((a >> 6) << 4)), low.where(b & 63, (c >> 4) | ((b >> 6) << 4))
        return [c_s[buf, p*g64+sub, masked(r, v), 0].store(_half(head & 0xffff)*scale.float()),
                c_s[buf, p*g64+sub, masked(r, v), 1].store(_half(head >> 16)*minimum.float())]
      coop(TN*g64, coefficients)
    else:
      def weights(i, v):
        r, w = i//16, i%16
        n, q, l = part//2, (part%2)*2+w//8, (w%8)*4
        ql, qh = _u32(raw, row_base(r)+(n*64+(q%2)*32+l)//2), _u32(raw, row_base(r)+64+(n*32+l)//2)
        word = ((ql >> ((q//2)*4).cast(dtypes.uint32)) & 0x0f0f0f0f) | (((qh >> (q*2).cast(dtypes.uint32)) & 0x03030303) << 4)
        # bytewise q-32 without borrows: clearing bit 5 subtracts 32 from q >= 32, q < 32 also gets the sign bits 0xc0
        return [w_s[buf, masked(r, v), p*16+w].store((word ^ 0x20202020) | (((word ^ 0xffffffff) & 0x20202020) * 6))]
      coop(TN*16, weights)
      def coefficients(i, v):
        sub, r = i//TN, i%TN
        s16 = part*4+sub
        scale = (raw[row_base(r)+96+s16//2].load() >> ((s16%2)*8).cast(dtypes.uint16)).cast(dtypes.uint8).bitcast(dtypes.int8).float()
        return [c_s[buf, p*g64+sub, masked(r, v), 0].store(_half(raw[row_base(r)+104].load())*scale)]
      coop(TN*g64, coefficients)
    def activations(i, v):
      tk, quarter = i//4, i%4
      x = _vload(xq, (tok0+tk)*(in_features//4)+part64*16+quarter*4, 4)
      return [x_s[buf, masked(tk, v), p*16+quarter*4+k].store(x[k]) for k in range(4)]
    coop(TM*4, activations)
    def token_terms(i, v):
      sub, tk = i//TM, i%TM
      tok, grp = tok0+tk, part64*g64+sub
      if not q4: return [t_s[buf, p*g64+sub, masked(tk, v), 0].store(xd[tok, grp//2].load())]
      d = xd[grp, tok].load()
      total = xs[grp*2, tok].load()+xs[grp*2+1, tok].load()
      return [t_s[buf, p*g64+sub, masked(tk, v), 0].store(d), t_s[buf, p*g64+sub, masked(tk, v), 1].store(-total*d)]
    coop(TM*g64, token_terms)

  prologue = stage_stores(UOp.const(0, dtypes.weakint), 0)
  stage = UOp.range(stages, 3, AxisType.LOOP)
  cur = stage%2
  # the last iteration restages its own (clamped) stage into the idle buffer, which nothing reads
  stores = stage_stores((stage+1).minimum(stages-1), 1-cur)
  w_s, x_s, c_s, t_s = (b.after(*prologue) for b in (w_s, x_s, c_s, t_s))
  values = [acc.after(stage)[i].load() for i in range(row_tiles*token_tiles*4)]
  wpg = group_size//4  # words per group
  for sub in range(groups):
    a_frags = [UOp.stack(*(w_s[cur, wrow+r*16+g+(i%2)*8, sub*wpg+t+(i//2)*4].load() for i in range(group_size//8))).bitcast(dtypes.int8)
               for r in range(row_tiles)]
    b_words = [[x_s[cur, wtok+j*8+g, sub*wpg+t+i*4].load() for i in range(group_size//16)] for j in range(token_tiles)]
    # a scalar word bitcasts to a scalar, so m16n8k16's single B word is split into bytes
    b_frags = [UOp.stack(*w).bitcast(dtypes.int8) if len(w) > 1 else
               UOp.stack(*((w[0] >> (i*8)).cast(dtypes.uint8).bitcast(dtypes.int8) for i in range(4))) for w in b_words]
    coeffs = [[c_s[cur, sub, wrow+r*16+g+h*8, k].load() for k in range(2 if q4 else 1)] for r in range(row_tiles) for h in range(2)]
    terms = [[t_s[cur, sub, wtok+j*8+t*2+i, k].load() for k in range(2 if q4 else 1)] for j in range(token_tiles) for i in range(2)]
    # int->float conversion is quarter rate, so the MMA accumulator starts at the bits of MAGIC, its int32 result reads as MAGIC+dot
    # in float (exact for |dot| < 2**22), and fma(MAGIC+dot, c, -MAGIC*c) == dot*c exactly (c is an F16 scale times a small int).
    # Q6_K's wide tile has no registers left for the biases (up to 4% slower there), so it keeps the conversion.
    magic = q4 or narrow
    seed = UOp.stack(*(UOp.const(0x4b400000 if magic else 0, dtypes.int32) for _ in range(4)))
    biases = [c[0]*-MAGIC for c in coeffs]
    for r, a in enumerate(a_frags):
      for j, b in enumerate(b_frags):
        dot = UOp.wmma(a, b, seed, (8,16,group_size), 32)
        for i in range(4):
          c, tt, v = coeffs[r*2+i//2], terms[j*2+i%2], (r*token_tiles+j)*4+i
          # explicit fmafs: the symbolic rewriter may otherwise distribute the magic product and lose the exact cancellation, and
          # the accumulation is an explicit fmaf chain too: as a plain sum the rewriter reorders the terms of a stage and puts the
          # accumulator last, which costs an extra FMUL and FADD per output
          if q4:
            scaled = _fma(dot[i].bitcast(dtypes.float32), c[0], biases[r*2+i//2])
            values[v] = _fma(scaled, tt[0], _fma(c[1], tt[1], values[v]))
          elif magic: values[v] = values[v] + _fma(dot[i].bitcast(dtypes.float32), c[0], biases[r*2+i//2])*tt[0]
          else: values[v] = values[v] + c[0]*dot[i].float()*tt[0]
  update = UOp.group(acc.store(UOp.stack(*values)), *stores).end(stage)
  outs = [out[tok0+wtok+j*8+t*2+i%2, row0+wrow+r*16+g+(i//2)*8].store(acc.after(update)[(r*token_tiles+j)*4+i].load())
          for r in range(row_tiles) for j in range(token_tiles) for i in range(4)]
  return UOp.group(*outs).end(token_block, row_block, lane, warp).sink(arg=KernelInfo(name="q4k_mmq" if q4 else "q6k_mmq", opts_to_apply=()))

def _live_blocks(n:UOp, TM:int, small_tail:int|None) -> UOp:
  # token blocks over the live tokens. A small_tail kernel pair splits by live count: the 64-token-block kernel (TM=64) runs only when the
  # live tokens end in a 1-64 token tail past the last full 128-token block and number fewer than small_tail, the 128-token one otherwise
  blocks = (n+TM-1)//TM
  if small_tail is None: return blocks
  tail = n % 128
  cond = (tail > 0) & (tail < 65) & (n < small_tail)
  return cond.where(blocks, 0) if TM == 64 else cond.where(0, blocks)

def _cp_async(dst:UOp, src:UOp, nbytes:UOp) -> UOp:
  # 16-byte global -> shared copy that holds no registers; bytes past nbytes are zero filled
  return _custom('asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"((unsigned)__cvta_generic_to_shared({0})), "l"({1}), '
                 '"r"((unsigned)({2})));', dtypes.void, dst, src, nbytes)
def _cp_wait(*copies:UOp) -> UOp: return _custom('asm volatile("cp.async.wait_all;" ::: "memory");', dtypes.void, *copies)
def _ldm(buf:UOp, idx:UOp, n:int, trans:bool=False) -> UOp:
  return UOp(Ops.SHRINK, src=(buf.flatten(), idx, UOp.const(n))).load(arg=f"ldmatrix_x{n}{'_trans' if trans else ''}")
def _bytes(word:UOp) -> UOp: return UOp.stack(*((word >> (i*8)).cast(dtypes.uint8).bitcast(dtypes.int8) for i in range(4)))

@functools.cache
def _q6k_mmq_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, tokens:int, out_features:int, in_features:int, live:UOp|None=None,
                    TN:int=64, TM:int=128, small_tail:int|None=None) -> UOp:
  # Q6_K MMQ that stages without registers, so two blocks fit per SM: each row's 210-byte super-block arrives raw through cp.async
  # (its 16-byte aligned envelope, once per super-block) and is unpacked to signed int8 in shared memory per 128-weight half; the
  # q8 activations are prefetched into registers. Single shared buffers, two barriers per half. Warps own 2x4 m16n8k16 tiles; each
  # 32-group runs two magic-seeded k16 MMAs (one per 16-weight scale). xd is group major; xs is unused (Q6_K has no mins).
  threads, nsb, MT, NT = 256, in_features//256, TN//32, TM//32  # 8 warps (2 row x 4 token) of TN/2 rows x TM/4 tokens
  n = None if live is None else live.unbind_all()[0]
  token_block = UOp.range(tokens//TM if n is None else _live_blocks(n, TM, small_tail), 1, AxisType.GLOBAL)
  row_block, lane = UOp.range(out_features//TN, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  warp = UOp.range(8, 2, AxisType.LOCAL)
  tid, g8, t4 = warp*32+lane, lane//4, lane%4
  row0, tok0, wr, tc = row_block*TN, token_block*TM, (warp//4)*(TN//2), (warp%4)*(TM//4)
  # 16-byte chunks of the weight and activation rows are XOR-swizzled by row, so ldmatrix and the unpack stores avoid bank conflicts
  wq = UOp.alloc((TN, 32), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  aq = UOp.alloc((TM, 32), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  ws = UOp.alloc((4, TN, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
  ts = UOp.alloc((4, TM), dtypes.float32, addrspace=AddrSpace.LOCAL)
  rw = UOp.alloc((TN, 56), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  def start(r:UOp, sb:UOp) -> UOp: return ((row0+r)*nsb+sb)*210  # byte offset of row r's super-block (the tensor starts 32B aligned)
  def raw_copies(sb:UOp, rwb:UOp) -> list[UOp]:  # rwb orders the copies after the last unpack of the previous super-block
    ops = []
    for i in range((TN*14+threads-1)//threads):
      idx = (tid+i*threads).minimum(TN*14-1)  # the last round's spare threads repeat a copy
      r, vec = idx//14, idx%14
      src = start(r, sb)//16*16 + vec*16
      ops.append(_cp_async(rwb[r, vec*4], raw[src//2], (start(r, sb)+210-src).maximum(0).minimum(16)))
    return ops
  def prefetch(stage:UOp) -> tuple[list, list]:
    acts = [(it, _vload(xq, (tok0+it//8)*(in_features//4) + stage*32 + (it%8)*4, 4)) for it in (tid+i*threads for i in range(TM*8//threads))]
    scales = [(it, xd[stage*4+it//TM, tok0+it%TM].load()) for it in (tid+i*threads for i in range(4*TM//threads))]
    return acts, scales
  def stage_stores(sb:UOp, h:int, regs:tuple[list, list], rwb:UOp, wqb:UOp, aqb:UOp, wsb:UOp, tsb:UOp) -> list[UOp]:
    acts, scales = regs
    st = [aqb[it//8, ((it%8) ^ (it//8%8))*4+j].store(v[j]) for it, v in acts for j in range(4)]
    st += [tsb[it//TM, it%TM].store(v) for it, v in scales]
    flat = rwb.flatten()
    def word(p:UOp, j:int) -> UOp:  # the 4 bytes at byte p+4j of the raw rows, p two-byte aligned
      return _custom("__funnelshift_r({0}, {1}, {2})", dtypes.uint32, flat[p//4+j].load(), flat[p//4+j+1].load(), ((p%4)*8).cast(dtypes.uint32))
    # unpack: item (row, 16-weight half hcol, group pair gp): ql bytes l and l+32 of this half, then its qh bytes
    for it in (tid+i*threads for i in range(TN*4//threads)):
      r, hcol, gp = it//2%TN, it%2, it//(2*TN)
      base = r*224 + start(r, sb)%16
      wreg = [word(base+off, j) for off in (h*64+hcol*16, h*64+32+hcol*16, 128+h*32+hcol*16) for j in range(4)]
      for k in range(2):
        grp = gp*2+k  # groups 0, 1 take the low nibbles of ql[l], ql[l+32]; 2, 3 the high ones
        vals = []
        for w in range(4):
          code = ((wreg[4*k+w] >> (gp*4).cast(dtypes.uint32)) & 0x0f0f0f0f) | (((wreg[8+w] >> (grp*2).cast(dtypes.uint32)) & 0x03030303) << 4)
          vals.append(((code | 0x80808080) - 0x20202020) ^ 0x80808080)  # bytewise code-32
        st += [wqb[r, ((grp*2+hcol) ^ (r%8))*4+w].store(v) for w, v in enumerate(vals)]
    # coefficients: item (group, row): d times the int8 scales of the group's two 16-weight halves
    def byte(p:UOp) -> UOp: return (flat[p//4].load() >> ((p%4)*8).cast(dtypes.uint32)) & 0xff
    for it in (tid+i*threads for i in range(4*TN//threads)):
      gi, rr = it//TN, it%TN
      hb = rr*224 + start(rr, sb)%16
      d = _half((flat[(hb+208)//4].load() >> (((hb+208)%4)*8).cast(dtypes.uint32)) & 0xffff)
      for e in range(2):
        st.append(wsb[gi, rr, e].store(d*byte(hb+192+h*8+gi*2+e).cast(dtypes.uint8).bitcast(dtypes.int8).float()))
    return st
  seed = UOp.stack(*(UOp.const(0x4b400000, dtypes.int32) for _ in range(4)))
  def compute(wqb:UOp, aqb:UOp, wsb:UOp, tsb:UOp, cur:UOp, axis:int, copies:list[UOp]) -> UOp:
    # the 4 groups of a half are a real loop (as in llama.cpp / Rust): unrolled, every group's fragments were live at once
    gi = UOp.range(4, axis, AxisType.LOOP)
    c = cur.after(gi, *copies)  # the next super-block's copies are issued before this loop and overlap it
    values = [c[i].load() for i in range(MT*NT*4)]
    afr = [_ldm(wqb, (row:=wr+mt*16+lane%8+(lane//8%2)*8)*32 + ((gi*2+lane//16) ^ (row%8))*4, 4) for mt in range(MT)]
    bfr = [_ldm(aqb, (tok:=tc+npair*16+lane%8+(lane//16)*8)*32 + ((gi*2+lane//8%2) ^ (tok%8))*4, 4) for npair in range(NT//2)]
    wl = [wsb[gi, wr+(r//2)*16+g8+(r%2)*8, e].load() for r in range(2*MT) for e in range(2)]
    bias = [w*-MAGIC for w in wl]
    for nt in range(NT):
      tok = tc+nt*8+2*t4
      xs0, xs1 = tsb[gi, tok].load(), tsb[gi, tok+1].load()
      blo, bhi = bfr[nt//2][(nt%2)*2], bfr[nt//2][(nt%2)*2+1]
      for mt in range(MT):
        lo = UOp.wmma(UOp.stack(afr[mt][0], afr[mt][1]).bitcast(dtypes.int8), _bytes(blo), seed, (8,16,16), 32)
        hi = UOp.wmma(UOp.stack(afr[mt][2], afr[mt][3]).bitcast(dtypes.int8), _bytes(bhi), seed, (8,16,16), 32)
        for i in range(4):
          k = (mt*2+i//2)*2
          v = _fma(lo[i].bitcast(dtypes.float32), wl[k], bias[k]) + _fma(hi[i].bitcast(dtypes.float32), wl[k+1], bias[k+1])
          o = (mt*NT+nt)*4+i
          values[o] = _fma(v, xs0 if i%2 == 0 else xs1, values[o])
    return cur.after(c.store(UOp.stack(*values)).end(gi))
  acc = UOp.alloc((MT*NT*4,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  zero = UOp.const(0, dtypes.weakint)
  # prologue: super-block 0 arrives, half 0 is staged
  pro = _cp_wait(*raw_copies(zero, rw))
  b0 = UOp.group(pro).barrier()
  bp = UOp.group(*stage_stores(zero, 0, prefetch(zero), rw.after(b0), wq, aq, ws, ts)).barrier()
  sbr = UOp.range(nsb, 3, AxisType.LOOP)
  bufs = tuple(b.after(bp, sbr) for b in (rw, wq, aq, ws, ts))
  cur = acc.after(sbr)
  for h in range(2):
    rwb, wqb, aqb, wsb, tsb = bufs
    nsb_next, nh = (sbr, 1) if h == 0 else ((sbr+1).minimum(nsb-1), 0)
    regs = prefetch(nsb_next*2+nh)
    copies = raw_copies(nsb_next, rwb) if h == 1 else []
    done = compute(wqb, aqb, wsb, tsb, cur, 10+h, copies)
    # the barrier ending the compute hangs directly on the copies' wait (an AFTER would drop a void CUSTOM); the wait reads the
    # accumulators, so it follows every shared read of this half
    b1 = _cp_wait(*copies, done[0].load()).barrier() if copies else UOp.group(done).barrier()
    nxt = tuple(b.after(b1) for b in bufs)
    b2 = UOp.group(*stage_stores(nsb_next, nh, regs, *nxt)).barrier()
    cur = done
    bufs, cur = tuple(b.after(b2) for b in bufs), cur.after(b2)
  acc = acc.after(b2.end(sbr))
  outs = [out[tok0+tc+nt*8+2*t4+i%2, row0+wr+mt*16+g8+(i//2)*8].store(acc[(mt*NT+nt)*4+i].load())
          for mt in range(MT) for nt in range(NT) for i in range(4)]
  return UOp.group(*outs).end(token_block, row_block, lane, warp).sink(arg=KernelInfo(name="q6k_mmq", opts_to_apply=(), min_blocks=2))

@functools.cache
def _q4k_mmq_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, tokens:int, out_features:int, in_features:int, live:UOp|None=None,
                    TN:int=64, TM:int=128, small_tail:int|None=None, WR:int=2) -> UOp:
  # Q4_K counterpart of _q6k_mmq_kernel: 144-byte super-blocks are 16-byte aligned, so cp.async copies them directly; the nibbles are
  # unpacked per 128-weight half into a swizzled int8 tile; m16n8k32 MMAs, scale/min coefficients and token sums as in _mmq_kernel.
  # 8 warps: WR row x 8//WR token warps of TN/WR rows x TM*WR/8 tokens.
  WT = 8//WR
  threads, nsb, MT, NT = 256, in_features//256, TN//(WR*16), TM//(WT*8)
  n = None if live is None else live.unbind_all()[0]
  token_block = UOp.range(tokens//TM if n is None else _live_blocks(n, TM, small_tail), 1, AxisType.GLOBAL)
  row_block, lane = UOp.range(out_features//TN, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  warp = UOp.range(8, 2, AxisType.LOCAL)
  tid, g8, t4 = warp*32+lane, lane//4, lane%4
  row0, tok0, wr, tc = row_block*TN, token_block*TM, (warp//WT)*(TN//WR), (warp%WT)*(TM//WT)
  wq = UOp.alloc((TN, 32), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  aq = UOp.alloc((TM, 32), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  ws = UOp.alloc((4, TN, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
  ts = UOp.alloc((4, TM, 2), dtypes.float32, addrspace=AddrSpace.LOCAL)
  rw = UOp.alloc((TN, 36), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  def items(n_items:int): return [(tid+i*threads).minimum(n_items-1) for i in range((n_items+threads-1)//threads)]
  def word0(r:UOp, sb:UOp) -> UOp: return ((row0+r)*nsb+sb)*36  # first word of row r's super-block (raw is a uint32 view)
  def raw_copies(sb:UOp, rwb:UOp) -> list[UOp]:
    return [_cp_async(rwb[idx//9, idx%9*4], raw[word0(idx//9, sb)+idx%9*4], UOp.const(16, dtypes.int)) for idx in items(TN*9)]
  def prefetch(stage:UOp) -> tuple[list, list]:
    acts = [(it, _vload(xq, (tok0+it//8)*(in_features//4) + stage*32 + (it%8)*4, 4)) for it in items(TM*8)]
    terms = []
    for it in items(4*TM):
      g, tk = stage*4+it//TM, tok0+it%TM
      d = xd[g, tk].load()
      terms.append((it, d, -(xs[g*2, tk].load()+xs[g*2+1, tk].load())*d))
    return acts, terms
  def stage_stores(sb:UOp, h:int, regs:tuple[list, list], rwb:UOp, wqb:UOp, aqb:UOp, wsb:UOp, tsb:UOp) -> list[UOp]:
    acts, terms = regs
    st = [aqb[it//8, ((it%8) ^ (it//8%8))*4+j].store(v[j]) for it, v in acts for j in range(4)]
    st += [x for it, d, m in terms for x in (tsb[it//TM, it%TM, 0].store(d), tsb[it//TM, it%TM, 1].store(m))]
    # unpack: item (row, 32-byte qs chunk of this half, 16-byte half of it): low nibbles are group 2*chunk, high ones 2*chunk+1
    for it in items(TN*4):
      r, ch, hcol = it//4, it//2%2, it%2
      qs = [rwb[r, 4 + (h*2+ch)*8 + hcol*4 + j].load() for j in range(4)]
      for nib in range(2):
        grp = ch*2+nib
        st += [wqb[r, ((grp*2+hcol) ^ (r%8))*4+j].store((q >> (nib*4)) & 0x0f0f0f0f) for j, q in enumerate(qs)]
    # coefficients: item (group, row): d*scale and dmin*min of super-block group 4h+gi
    for it in items(4*TN):
      gi, rr = it//TN, it%TN
      head, s0, s1, s2 = (rwb[rr, j].load() for j in range(4))
      g = 4*h+gi
      a, b, c = ((x >> ((g%4)*8).cast(dtypes.uint32)) & 255 for x in (s0, s1, s2))
      low = g < 4
      scale, mn = low.where(a & 63, (c & 15) | ((a >> 6) << 4)), low.where(b & 63, (c >> 4) | ((b >> 6) << 4))
      st += [wsb[gi, rr, 0].store(_half(head & 0xffff)*scale.float()), wsb[gi, rr, 1].store(_half(head >> 16)*mn.float())]
    return st
  seed = UOp.stack(*(UOp.const(0x4b400000, dtypes.int32) for _ in range(4)))
  def compute(wqb:UOp, aqb:UOp, wsb:UOp, tsb:UOp, cur:UOp, axis:int, copies:list[UOp]) -> UOp:
    gi = UOp.range(4, axis, AxisType.LOOP)
    c = cur.after(gi, *copies)  # the next super-block's copies are issued before this loop and overlap it
    values = [c[i].load() for i in range(MT*NT*4)]
    afr = [_ldm(wqb, (row:=wr+mt*16+lane%8+(lane//8%2)*8)*32 + ((gi*2+lane//16) ^ (row%8))*4, 4) for mt in range(MT)]
    bfr = [_ldm(aqb, (tok:=tc+npair*16+lane%8+(lane//16)*8)*32 + ((gi*2+lane//8%2) ^ (tok%8))*4, 4) for npair in range(NT//2)]
    co = [[wsb[gi, wr+(r//2)*16+g8+(r%2)*8, e].load() for e in range(2)] for r in range(2*MT)]
    bias = [cc[0]*-MAGIC for cc in co]
    for nt in range(NT):
      tok = tc+nt*8+2*t4
      tt = [[tsb[gi, tok+e, k].load() for k in range(2)] for e in range(2)]
      b = UOp.stack(bfr[nt//2][(nt%2)*2], bfr[nt//2][(nt%2)*2+1]).bitcast(dtypes.int8)
      for mt in range(MT):
        dot = UOp.wmma(UOp.stack(*(afr[mt][j] for j in range(4))).bitcast(dtypes.int8), b, seed, (8,16,32), 32)
        for i in range(4):
          r, o = mt*2+i//2, (mt*NT+nt)*4+i
          scaled = _fma(dot[i].bitcast(dtypes.float32), co[r][0], bias[r])
          values[o] = _fma(scaled, tt[i%2][0], _fma(co[r][1], tt[i%2][1], values[o]))
    return cur.after(c.store(UOp.stack(*values)).end(gi))
  acc = UOp.alloc((MT*NT*4,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  zero = UOp.const(0, dtypes.weakint)
  b0 = _cp_wait(*raw_copies(zero, rw)).barrier()
  bp = UOp.group(*stage_stores(zero, 0, prefetch(zero), rw.after(b0), wq, aq, ws, ts)).barrier()
  sbr = UOp.range(nsb, 3, AxisType.LOOP)
  bufs = tuple(b.after(bp, sbr) for b in (rw, wq, aq, ws, ts))
  cur = acc.after(sbr)
  for h in range(2):
    rwb, wqb, aqb, wsb, tsb = bufs
    nsb_next, nh = (sbr, 1) if h == 0 else ((sbr+1).minimum(nsb-1), 0)
    regs = prefetch(nsb_next*2+nh)
    copies = raw_copies(nsb_next, rwb) if h == 1 else []
    done = compute(wqb, aqb, wsb, tsb, cur, 10+h, copies)
    b1 = _cp_wait(*copies, done[0].load()).barrier() if copies else UOp.group(done).barrier()
    nxt = tuple(b.after(b1) for b in bufs)
    b2 = UOp.group(*stage_stores(nsb_next, nh, regs, *nxt)).barrier()
    bufs, cur = tuple(b.after(b2) for b in bufs), done.after(b2)
  acc = acc.after(b2.end(sbr))
  outs = [out[tok0+tc+nt*8+2*t4+i%2, row0+wr+mt*16+g8+(i//2)*8].store(acc[(mt*NT+nt)*4+i].load())
          for mt in range(MT) for nt in range(NT) for i in range(4)]
  return UOp.group(*outs).end(token_block, row_block, lane, warp).sink(arg=KernelInfo(name="q4k_mmq", opts_to_apply=(), min_blocks=2))

def nv_mmq(weight:Tensor, x:Tensor, in_features:int, out_features:int, ggml_type:int, live:UOp|None=None, q8:dict|None=None,
           up:Tensor|None=None) -> Tensor:
  # live: the symbolic token count of a chunk padded to its max size; the rows past it are left unwritten
  shape = x.shape[:-1]
  tokens = int(x.numel())//in_features
  padded_tokens = (tokens+7)//8*8
  x = x.reshape(tokens, in_features).pad(((0, padded_tokens-tokens), (0, 0)))
  if up is not None: up = up.reshape(tokens, in_features).pad(((0, padded_tokens-tokens), (0, 0)))
  out = Tensor.empty(padded_tokens, out_features, dtype=dtypes.float32, device=x.device)
  # Q6_K stages raw super-blocks with cp.async where its 64x128 tile divides the shape (two blocks per SM); other shapes share the
  # register-staged kernel with Q4_K
  v2 = ggml_type == Q6_K and out_features % 64 == 0 and padded_tokens % 128 == 0
  fxn = functools.partial(_q6k_mmq_kernel, tokens=padded_tokens, out_features=out_features, in_features=in_features, live=live) if v2 else \
        functools.partial(_mmq_kernel, tokens=padded_tokens, out_features=out_features, in_features=in_features, ggml_type=ggml_type, live=live)
  xq, xd, xs = q8_quantize(x, padded_tokens, in_features, q8, up, live, group_major=ggml_type == Q4_K or v2)
  if live is not None: xq = Tensor(xq.uop.after(live))
  # a padded symbolic chunk whose live tokens end in a 1-64 token tail can run a 64-token-block cp.async kernel instead (no nearly empty
  # 128-token block): 128 rows per block for Q4_K (4 row x 2 token warps: 1.5% faster than 2 x 4 here), 64 for Q6_K, whose raw tile is
  # larger (two blocks per SM either way). Both kernels are
  # launched, and the one that doesn't apply gets no blocks. A 64-token block costs ~0.56x a 128-token one for Q4_K, so its 2k+1 blocks
  # win over k+1 for any chunk; ~0.6-0.65x for Q6_K, so only with at most one full block before the tail (live < 256)
  tail = {Q4_K: (_q4k_mmq_kernel, 128, padded_tokens+1), Q6_K: (_q6k_mmq_kernel, 64, 256)}.get(ggml_type) \
    if live is not None and padded_tokens % 128 == 0 else None
  if tail is not None and out_features % tail[1] == 0:
    out = Tensor.custom_kernel(out, weight, xq, xd, xs, fxn=functools.partial(fxn, small_tail=tail[2]))[0]
    fxn = functools.partial(tail[0], tokens=padded_tokens, out_features=out_features, in_features=in_features, live=live, TN=tail[1], TM=64,
                            small_tail=tail[2], **({"WR": 4} if ggml_type == Q4_K else {}))
  return Tensor.custom_kernel(out, weight, xq, xd, xs, fxn=fxn)[0][:tokens].reshape(*shape, out_features)

# ******** decode: residual sum, RMS norm and q8 quantization in one kernel ********

@functools.cache
def _rmsnorm_q8_kernel(xq:UOp, xd:UOp, xs:UOp, *rest:UOp, n_parts:int, dim:int, eps:float, write_sum:bool, write_normed:bool, tokens:int=1,
                       live:UOp|None=None) -> UOp:
  # per token: x = sum of the parts (the residual stream), its RMS norm with the weight, then the q8 quantization of q8_quantize
  # (group-major scales and sums). One block of up to 32 warps per token (a padded chunk launches its live tokens only), warp w owns
  # 128-value chunks w, w+32, ...; the sum of squares reduces over each warp and then across the warps through shared memory.
  outs = list(rest[:int(write_sum)+int(write_normed)])
  parts, weight = rest[len(outs):len(outs)+n_parts], rest[-1]
  chunks = dim//128
  warps = min(chunks, 32)  # one warp per chunk: one token's work is latency bound
  token = UOp.range(tokens if live is None else live.unbind_all()[0], 0, AxisType.GLOBAL)
  lane, warp = UOp.range(32, -1, AxisType.WARP), UOp.range(warps, 1, AxisType.LOCAL)
  idx = [(warp + warps*c)*128 + lane*4 + i for c in range(chunks//warps) for i in range(4)]
  vals = [functools.reduce(UOp.__add__, (p[token*dim + j].load() for p in parts)) for j in idx]
  red = UOp.alloc((warps,), dtypes.float32, addrspace=AddrSpace.LOCAL)
  red = red.after(red[warp.valid(lane.eq(0))].store(warp_reduce(sum((v*v for v in vals[1:]), vals[0]*vals[0]))).barrier())
  rms = (sum((red[w].load() for w in range(1, warps)), red[0].load())/dim + eps).rsqrt()
  stores = []
  for c in range(chunks//warps):
    chunk = warp + warps*c
    v = vals[c*4:c*4+4]
    normed = [x*rms*weight[j].load().float() for x, j in zip(v, idx[c*4:c*4+4])]
    if write_sum: stores += [outs[0][token*dim + j].store(x) for x, j in zip(v, idx[c*4:c*4+4])]
    if write_normed: stores += [outs[-1][token*dim + j].store(x.cast(outs[-1].dtype)) for x, j in zip(normed, idx[c*4:c*4+4])]
    amax = functools.reduce(UOp.maximum, (x.abs() for x in normed))
    d = _custom("({0}/127.0f)", dtypes.float, warp_reduce(amax, maximum=True, width=8)).maximum(1e-30)
    quants = [_custom("rintf({0}/{1})", dtypes.float, x, d).clip(-127, 127) for x in normed]
    word = functools.reduce(UOp.bitwise_or, (q.cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i*8) for i,q in enumerate(quants)))
    group, w = chunk*4 + lane//8, lane%8
    stores += [xq[token, chunk*32 + lane].store(word), xd[group.valid(w.eq(0)), token].store(d),
               xs[group*2 + (w//4).valid((w%4).eq(0)), token].store(warp_reduce(sum(quants[1:], quants[0]), width=4))]
  return UOp.group(*stores).end(token, lane, warp).sink(arg=KernelInfo(name="rmsnorm_q8", opts_to_apply=()))

def nv_rmsnorm_q8_supported(x:Tensor, dim:int) -> bool:
  # one decode token, or a symbolic prefill chunk (the MMQ path)
  if not (nv_custom_kernels_supported(x.device) and isinstance(x.device, str) and dim % 1024 == 0): return False
  return x.numel() == dim if isinstance(x.numel(), int) else nv_mmq_supported(x.device) and prod(x.max_shape) % (8*dim) == 0

def nv_rmsnorm_q8(parts:list[Tensor], weight:Tensor, eps:float, write_sum:bool=False, write_normed:bool=False) -> tuple[Tensor|None,
                  Tensor|None, dict]:
  # one decode token or a symbolic prefill chunk: returns the sum of the parts (if write_sum), the normed input (if write_normed) and
  # a q8 cache for Linear.project, whose packed projections then skip their own quantization
  shape, dim, dev = parts[0].shape, int(parts[0].shape[-1]), parts[0].device
  numel = parts[0].numel()
  live = None if isinstance(numel, int) else numel//dim
  tokens = prod(parts[0].max_shape)//dim
  xq = Tensor.empty(tokens, dim//4, dtype=dtypes.uint32, device=dev)
  xd = Tensor.empty(dim//32, tokens, dtype=dtypes.float32, device=dev)
  xs = Tensor.empty(dim//16, tokens, dtype=dtypes.float32, device=dev)
  if live is not None: xq = Tensor(xq.uop.after(live))  # carries the live-token binding into the kernel
  # the normed input of a prefill chunk is written as half (its consumer, the F16 GEMM, casts to half anyway); decode keeps float
  extra = [Tensor.empty(tokens*dim, dtype=dtypes.float32, device=dev)] if write_sum else []
  if write_normed: extra.append(Tensor.empty(tokens*dim, dtype=dtypes.float32 if live is None else dtypes.half, device=dev))
  fxn = functools.partial(_rmsnorm_q8_kernel, n_parts=len(parts), dim=dim, eps=eps, write_sum=write_sum, write_normed=write_normed,
                          tokens=tokens, live=live)
  ins = [(p if live is None else pad_to_max(p)).reshape(tokens*dim).float().contiguous() for p in parts]
  # the weight goes in as stored (a float cast here would be a kernel per call), the kernel converts it
  outs = Tensor.custom_kernel(xq, xd, xs, *extra, *ins, weight.contiguous(), fxn=fxn)
  xq, xd, xs, extra = outs[0], outs[1], outs[2], outs[3:3+len(extra)]
  def view(t:Tensor) -> Tensor:
    t = t.reshape(*parts[0].max_shape)
    return t if live is None else t.shrink(tuple((0, s) for s in shape))
  h = view(extra[0]) if write_sum else None
  xn = view(extra[-1]) if write_normed else None
  return h, xn, {(tokens, dim, True): (xq, xd, xs)}

def pad_to_max(t:Tensor) -> Tensor:
  # a symbolic slice of a max-size buffer (a custom kernel's padded output) passes on the whole buffer: the padded rows only feed
  # outputs past the live tokens, which the kernels skip. Padding the slice instead would copy it; anything else is written into an
  # uninitialized max-size buffer, live rows only
  if (u:=t.uop).op is Ops.SHRINK and u.src[0].shape == t.max_shape and u.src[0].base.has_buffer_identity(after_ok=True): return Tensor(u.src[0])
  return pad_unwritten(t, t.max_shape)

# ******** decode: flash-decoding combine with the o-proj input's q8 quantization ********

@functools.cache
def _combine_q8_kernel(xq:UOp, xd:UOp, xs:UOp, partial:UOp, stats:UOp, live:UOp|int) -> UOp:
  # one warp per head: the flash-decoding combine (max over the live chunks, then their weighted partial sums) of the head's D=128
  # outputs, four per lane, then their q8 quantization as in q8_quantize: a head is one 128-value chunk of the o-proj input
  live = live.unbind_all()[0] if isinstance(live, UOp) else live
  H = int(partial.shape[1])
  warps = _tile(H, 8, 4, 2, 1)
  block, lane, warp = UOp.range(H//warps, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), UOp.range(warps, 1, AxisType.LOCAL)
  h = block*warps + warp
  def reg(n:int, val:float, dep:UOp|None=None) -> UOp:
    r = UOp.alloc((n,), dtypes.float32, addrspace=AddrSpace.REG)
    return r.after((r if dep is None else r.after(dep)).store(UOp.stack(*(UOp.const(val, dtypes.float32) for _ in range(n)))))
  c1 = UOp.range(live, 100, AxisType.LOOP)
  mx = reg(1, -math.inf)
  mx = mx.after(mx[0].store(mx.after(c1)[0].load().maximum(stats[0, h, c1, 0].load())).end(c1))
  c2 = UOp.range(live, 101, AxisType.LOOP)
  acc = reg(5, 0.0, mx)
  w = ((stats[0, h, c2, 0].load() - mx[0].load())*LOG2E).exp2()
  cur = acc.after(c2)
  acc = acc.after(acc.store(UOp.stack(*(cur[i].load() + w*partial[0, h, c2, lane*4 + i].load() for i in range(4)),
                                      cur[4].load() + w*stats[0, h, c2, 1].load())).end(c2))
  inv = 1/acc[4].load()
  vals = [acc[i].load()*inv for i in range(4)]
  amax = functools.reduce(UOp.maximum, (x.abs() for x in vals))
  d = _custom("({0}/127.0f)", dtypes.float, warp_reduce(amax, maximum=True, width=8)).maximum(1e-30)
  quants = [_custom("rintf({0}/{1})", dtypes.float, x, d).clip(-127, 127) for x in vals]
  word = functools.reduce(UOp.bitwise_or, (q.cast(dtypes.int8).cast(dtypes.uint8).cast(dtypes.uint32) << (i*8) for i,q in enumerate(quants)))
  group, wl = h*4 + lane//8, lane%8
  stores = (xq[0, h*32 + lane].store(word), xd[group.valid(wl.eq(0)), 0].store(d),
            xs[group*2 + (wl//4).valid((wl%4).eq(0)), 0].store(warp_reduce(sum(quants[1:], quants[0]), width=4)))
  return UOp.group(*stores).end(block, lane, warp).sink(arg=KernelInfo(name="flash_decode_combine_q8", opts_to_apply=()))

def nv_flash_decode_q8(q:Tensor, assigned_kv:Tensor, valid_end:UOp, max_kv_len:int) -> tuple[Tensor, dict]:
  # single-token attention through the shared flash-decoding kernel, combined straight into the o-proj input's q8 (Linear.project's
  # shared quantized input); also returns the (lazy, unrealized unless used) float output for other consumers
  from tinygrad.llm.kernels.amd import amd_flash_attention_decode
  H, D = q.shape[1], q.shape[3]
  def combine(partial:Tensor, stats:Tensor, live) -> tuple[Tensor, Tensor, Tensor]:
    xq = Tensor.empty(1, H*D//4, dtype=dtypes.uint32, device=q.device)
    xd, xs = Tensor.empty(H*D//32, 1, dtype=dtypes.float32, device=q.device), Tensor.empty(H*D//16, 1, dtype=dtypes.float32, device=q.device)
    if isinstance(live, UOp): xq = Tensor(xq.uop.after(live))  # carries the binding into the kernel
    xq, xd, xs = Tensor.custom_kernel(xq, xd, xs, partial, stats, fxn=functools.partial(_combine_q8_kernel, live=live))[:3]
    return xq, xd, xs
  xq, xd, xs = amd_flash_attention_decode(q, assigned_kv, valid_end, max_kv_len, combine=combine)
  return amd_flash_attention_decode(q, assigned_kv, valid_end, max_kv_len), {(1, H*D, True): (xq, xd, xs)}

# ******** decode: attention prologue ********

@functools.cache
def _qk_prep_kernel(q_out:UOp, cache:UOp, q:UOp, k:UOp, v:UOp, q_norm:UOp, k_norm:UOp, freqs:UOp, start_pos:UOp,
                    heads:int, kv_heads:int, head_dim:int, eps:float) -> UOp:
  # one decode token's attention prologue, one warp per head row (rows < heads are queries, the rest keys): RMS norm over the head
  # with its weight, rotary embedding (element i pairs with i + D/2, which is 16 lanes away), then q goes out as half and the key and
  # its value row into the KV cache at start_pos. Replaces the norm reduce, norm apply/rope and cache store kernels per tensor.
  D, per, rows = head_dim, head_dim//32, heads+kv_heads
  sp = start_pos.unbind_all()[0]
  warps = _tile(rows, 8, 4, 2, 1)
  block, lane, warp = UOp.range(rows//warps, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), UOp.range(warps, 1, AxisType.LOCAL)
  row = block*warps + warp
  is_q = row < heads
  kv_row = (row - heads).maximum(0)
  e0 = lane*per  # first of this lane's head elements
  vals = [is_q.where(q[row.minimum(heads-1)*D + e0 + i].load(), k[kv_row*D + e0 + i].load()) for i in range(per)]
  rms = (warp_reduce(sum((x*x for x in vals[1:]), vals[0]*vals[0]))/D + eps).rsqrt()
  normed = [x*rms*is_q.where(q_norm[e0+i].load().float(), k_norm[e0+i].load().float()) for i, x in enumerate(vals)]
  partner = [_shfl_xor(x, 16) for x in normed]
  first = lane < 16
  out = []
  for i in range(per):
    f = (e0 + i) % (D//2)
    cos, sin = freqs[sp, f].load(), freqs[sp, D//2 + f].load()
    out.append(first.where(normed[i]*cos - partner[i]*sin, normed[i]*cos + partner[i]*sin))
  cw = cache.reshape(2, kv_heads, cache.shape[-2], D)
  stores = [q_out[row.valid(is_q)*D + e0 + i].store(out[i].cast(dtypes.half)) for i in range(per)]
  stores += [cw[0, kv_row.valid(is_q.logical_not()), sp, e0 + i].store(out[i].cast(dtypes.half)) for i in range(per)]
  stores += [cw[1, kv_row.valid(is_q.logical_not()), sp, e0 + i].store(v[kv_row*D + e0 + i].load().cast(dtypes.half)) for i in range(per)]
  return UOp.group(*stores).end(block, lane, warp).sink(arg=KernelInfo(name="qk_prep", opts_to_apply=()))

def nv_qk_prep_supported(q:Tensor, cache_kv:Tensor, head_dim:int) -> bool:
  return nv_custom_kernels_supported(q.device) and isinstance(q.device, str) and head_dim % 32 == 0 and cache_kv.shape[1] == 1

def nv_qk_prep(q:Tensor, k:Tensor, v:Tensor, q_norm:Tensor, k_norm:Tensor, freqs:Tensor, cache_kv:Tensor, start_pos:UOp,
               heads:int, kv_heads:int, head_dim:int, eps:float) -> tuple[Tensor, Tensor]:
  # decode (one token, batch 1): returns the normed, rotated queries (1, heads, 1, head_dim) as half and the cache with k and v written
  q_out = Tensor.empty(heads*head_dim, dtype=dtypes.half, device=q.device)
  fxn = functools.partial(_qk_prep_kernel, start_pos=start_pos, heads=heads, kv_heads=kv_heads, head_dim=head_dim, eps=eps)
  freqs = Tensor(freqs.float().contiguous().uop.after(start_pos))  # an input carries the start_pos binding into the kernel
  # the norm weights go in as stored (a float cast here would be a kernel per call), the kernel converts them
  q_out, kv = Tensor.custom_kernel(q_out, cache_kv, *(t.reshape(-1).float().contiguous() for t in (q, k, v)), q_norm.contiguous(),
                                   k_norm.contiguous(), freqs, fxn=fxn)[:2]
  return q_out.reshape(1, heads, 1, head_dim), kv

# ******** prefill: flash attention ********

FA_BLOCK_M, FA_BLOCK_N, FA_WARPS, LOG2E = 64, 64, 4, math.log2(math.e)

@functools.cache
def _flash_attention_kernel(o:UOp, q:UOp, cache:UOp, valid_kv_len:UOp|int, q_start:UOp|int) -> UOp:
  # FlashAttention-2 on m16n8k16 half MMAs: each warp owns 16 query rows of a 64-row block and keeps them in registers. Per
  # 64-key tile the block stages K and V^T in shared memory; the S accumulators are reused as the A fragments of P@V.
  # q and the cache are half; the staging works on 32-bit words of two halves
  BH, M, D = q.shape
  _, B, H_KV, N, _ = cache.shape
  group, words = BH//(B*H_KV), D//2
  assert M % FA_BLOCK_M == 0 and D % 16 == 0 and N % FA_BLOCK_N == 0
  valid_kv_len = valid_kv_len.unbind_all()[0] if isinstance(valid_kv_len, UOp) else valid_kv_len
  q_start = q_start.unbind_all()[0] if isinstance(q_start, UOp) else q_start
  # a padded symbolic chunk launches only the query blocks holding its valid_kv_len - q_start live rows
  live = M if isinstance(valid_kv_len, int) and isinstance(q_start, int) else valid_kv_len - q_start
  block_bh, block_m = UOp.range(BH, 0, AxisType.GLOBAL), UOp.range((live + FA_BLOCK_M - 1)//FA_BLOCK_M, 1, AxisType.GLOBAL)
  lane, warp = UOp.range(32, -1, AxisType.WARP), UOp.range(FA_WARPS, 2, AxisType.LOCAL)
  tid, g, t = warp*32 + lane, lane//4, lane%4
  kv = block_bh//group  # (batch, kv head)
  cw = cache.reshape(2, B*H_KV, N, D)
  # K and V tiles stay row major (key, d): their 16-byte chunks are XOR-swizzled by key, so the ldmatrix rows (8 keys per matrix)
  # fall in distinct banks; V's fragments are loaded transposed
  K_s = UOp.alloc((FA_BLOCK_N, words), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  V_s = UOp.alloc((FA_BLOCK_N, words), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  chunks = words//4
  def swz(key:UOp, chunk:UOp) -> UOp: return key*words + (chunk ^ (key%8))*4
  rows = [block_m*FA_BLOCK_M + warp*16 + g + h*8 for h in range(2)]
  q_frags = [UOp.stack(*(q[block_bh, rows[i%2], (ks*8 + t + (i//2)*4)*2 + e].load() for i in range(4) for e in range(2))) for ks in range(D//16)]
  n_tiles = D//8  # output column tiles of P@V
  acc = UOp.alloc((n_tiles*4,), dtypes.float32, addrspace=AddrSpace.REG)
  stats = UOp.alloc((4,), dtypes.float32, addrspace=AddrSpace.REG)  # (row max, partial row sum) for rows g and g+8
  acc = acc.after(acc.store(acc.const_like(0)))
  stats = stats.after(stats.store(UOp.stack(*(UOp.const(x, dtypes.float32) for x in (-1e30, -1e30, 0, 0)))))
  last_key = (q_start + (block_m+1)*FA_BLOCK_M).minimum(valid_kv_len)
  tile = UOp.range((last_key + FA_BLOCK_N - 1)//FA_BLOCK_N, 3, AxisType.LOOP)
  threads = 32*FA_WARPS
  copies = []
  for sel, buf in enumerate((K_s, V_s)):  # the tile's K and V rows: 16-byte chunks from the cache through cp.async (no registers)
    for i in range(FA_BLOCK_N*chunks//threads):
      idx = tid + i*threads
      key, chunk = idx//chunks, idx%chunks
      copies.append(_cp_async(buf.flatten()[swz(key, chunk)], cw.flatten()[((sel*B*H_KV + kv)*N + tile*FA_BLOCK_N + key)*D + chunk*8],
                              UOp.const(16, dtypes.int)))
  ready = _cp_wait(*copies).barrier()
  K_r, V_r = K_s.after(ready), V_s.after(ready)
  zero = UOp.stack(*(UOp.const(0.0, dtypes.float32) for _ in range(4)))
  # S = Q K^T for 8 key tiles of 8
  s = []
  for j in range(FA_BLOCK_N//8):
    c = zero
    for kp in range(D//32):  # one ldmatrix.x4 holds the B fragments of two k steps: matrices (keys j*8.., d 32kp + 8m..)
      fr = _ldm(K_r, swz(j*8 + lane%8, kp*4 + lane//8), 4)
      for e in range(2): c = UOp.wmma(q_frags[kp*2 + e], UOp.stack(fr[e*2], fr[e*2+1]).bitcast(dtypes.half), c, (8,16,16), 32)
    s.append(c)
  scale = LOG2E/math.sqrt(D)
  old = [stats.after(tile)[i].load() for i in range(4)]
  new_max:list[UOp] = []
  probs:list[list[UOp]] = [[] for _ in range(FA_BLOCK_N//8)]
  for h in range(2):
    vals = []
    for j in range(FA_BLOCK_N//8):
      for e in range(2):
        key, qpos = tile*FA_BLOCK_N + j*8 + t*2 + e, q_start + rows[h]
        vals.append(((key <= qpos) & (key < valid_kv_len)).where(s[j][h*2 + e]*scale, UOp.const(-math.inf, dtypes.float32)))
    m = warp_reduce(functools.reduce(UOp.maximum, vals), maximum=True, width=4).maximum(old[h])
    new_max.append(m)
    for j in range(FA_BLOCK_N//8): probs[j] += [(vals[j*2 + e] - m).exp2() for e in range(2)]
  alpha = [(old[h] - new_max[h]).exp2() for h in range(2)]
  sums = [old[2+h]*alpha[h] + sum((probs[j][h*2 + e] for j in range(FA_BLOCK_N//8) for e in range(2)), UOp.const(0.0, dtypes.float32))
          for h in range(2)]
  # O = O*alpha + P V: two adjacent S tiles (16 keys) are the A fragment of one k step
  outs = [acc.after(tile)[i].load()*alpha[(i%4)//2] for i in range(n_tiles*4)]
  for kstep in range(FA_BLOCK_N//16):
    a = UOp.stack(*(probs[kstep*2 + i//2][(i%2)*2 + e].cast(dtypes.half) for i in range(4) for e in range(2)))
    for np_ in range(n_tiles//2):  # one ldmatrix.x4.trans holds the B fragments of two d tiles: matrices (keys +0/+8, d tile 2np/2np+1)
      fr = _ldm(V_r, swz(kstep*16 + lane%8 + (lane//8%2)*8, np_*2 + lane//16), 4, trans=True)
      for e in range(2):
        n = np_*2 + e
        res = UOp.wmma(a, UOp.stack(fr[e*2], fr[e*2+1]).bitcast(dtypes.half), UOp.stack(*outs[n*4:n*4+4]), (8,16,16), 32)
        outs[n*4:n*4+4] = [res[i] for i in range(4)]
  # the barrier ends the tile, so the next tile's copies don't overwrite K and V while a warp still reads them
  update = UOp.group(acc.store(UOp.stack(*outs)), stats.store(UOp.stack(*new_max, *sums))).barrier().end(tile)
  acc, stats = acc.after(update), stats.after(update)
  totals = [warp_reduce(stats[2+h].load(), width=4) for h in range(2)]
  ow = o.reshape(BH, M, D)
  final = [ow[block_bh, rows[i//2], n*8 + t*2 + i%2].store(acc[n*4 + i].load()/totals[i//2]) for n in range(n_tiles) for i in range(4)]
  return UOp.group(*final).end(block_bh, block_m, lane, warp).sink(arg=KernelInfo(name="nv_flash_attention", opts_to_apply=()))

def nv_flash_attention_supported(q:Tensor, assigned_kv:Tensor) -> bool:
  return nv_mmq_supported(q.device) and q.shape[3] % 16 == 0 and assigned_kv.shape[3] % FA_BLOCK_N == 0

def nv_flash_attention(q:Tensor, assigned_kv:Tensor, valid_end:int|UOp) -> Tensor:
  # causal prefill attention on the half KV cache (already written through assigned_kv); a symbolic chunk is padded to its max
  # queries are padded to whole 64-row blocks (a symbolic chunk to its max first); the padded rows are sliced off
  B, H, T, D = q.shape
  T_pad = (q.max_shape[2] + FA_BLOCK_M - 1)//FA_BLOCK_M*FA_BLOCK_M
  q_start = valid_end - T
  padded = T_pad != q.max_shape[2] or not isinstance(T, int)
  out = Tensor.empty(B*H, T_pad, D, dtype=dtypes.float32, device=q.device)
  if isinstance(valid_end, UOp): assigned_kv = Tensor(assigned_kv.uop.after(valid_end))
  fxn = functools.partial(_flash_attention_kernel, valid_kv_len=valid_end, q_start=q_start)
  qh = q.half().reshape(B*H, T, D)
  qh = pad_unwritten(qh, (q.max_shape[0]*q.max_shape[1], T_pad, q.max_shape[3])) if padded else qh.contiguous()
  out = Tensor.custom_kernel(out, qh, assigned_kv, fxn=fxn)[0].reshape(B, H, T_pad, D)
  return out[:, :, :T] if padded else out
