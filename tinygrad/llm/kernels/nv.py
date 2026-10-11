from __future__ import annotations
import functools
from tinygrad import Tensor, UOp, Device, Context, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, AddrSpace
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

# ******** prefill: integer tensor cores ********

def _tile(n:int, *sizes:int) -> int: return next(s for s in sizes if n % s == 0)

@functools.cache
def _mmq_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, tokens:int, out_features:int, in_features:int, ggml_type:int,
                live:UOp|None=None) -> UOp:
  # Each warp owns row_tiles 16-row x token_tiles 8-token MMA tiles, so every weight fragment feeds token_tiles MMAs and every
  # activation fragment row_tiles MMAs. The block stages 64 weights per row at a time: it unpacks its rows to int8 words in
  # shared memory, decodes each row's group coefficients once, and stages the activations next to them.
  # Q4_K groups (32) match m16n8k32; Q6_K scales every 16 weights, so it uses m16n8k16.
  q4 = ggml_type == Q4_K
  group_size, words = (32, 36) if q4 else (16, 105)
  groups = 64//group_size
  row_tiles = _tile(out_features//16, 4, 2, 1)
  row_warps = _tile(out_features//(16*row_tiles), 2, 1)
  token_tiles = _tile(tokens//8, 4, 2, 1)
  token_warps = _tile(tokens//(8*token_tiles), 4, 2, 1)
  TN, TM, threads = 16*row_tiles*row_warps, 8*token_tiles*token_warps, 32*row_warps*token_warps
  # a padded symbolic chunk launches only the token blocks holding its live tokens; token blocks vary fastest, so blocks
  # sharing a weight tile run together and the weights come from L2
  token_block = UOp.range(tokens//TM if live is None else (live.unbind_all()[0]+TM-1)//TM, 0, AxisType.GLOBAL)
  row_block, lane = UOp.range(out_features//TN, 1, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP)
  warp = UOp.range(row_warps*token_warps, 2, AxisType.LOCAL)
  tid, g, t = warp*32+lane, lane//4, lane%4
  row0, tok0 = row_block*TN, token_block*TM
  wrow, wtok = (warp//token_warps)*(16*row_tiles), (warp%token_warps)*(8*token_tiles)
  # 20-word rows: the fragment reads of 8 rows x 4 words fall in distinct banks, and 16-byte stores stay aligned
  # double buffered: iteration s computes from buffer s%2 while it stages s+1 into the other one; the barrier that ends each
  # iteration orders both the stores before the next iteration's reads and the reads before the stores that overwrite them
  w_s = UOp.alloc((2, TN, 20), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  x_s = UOp.alloc((2, TM, 20), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  # Q4_K: (scale*d, min*dmin) per row and (d, -(sum of quants)*d) per token; Q6_K: scale*d and d
  c_s = UOp.alloc((2, groups, TN, 2 if q4 else 1), dtypes.float32, addrspace=AddrSpace.LOCAL)
  t_s = UOp.alloc((2, groups, TM, 2 if q4 else 1), dtypes.float32, addrspace=AddrSpace.LOCAL)
  acc = UOp.alloc((row_tiles*token_tiles*4,), dtypes.float32, addrspace=AddrSpace.REG)
  acc = acc.after(acc.store(acc.const_like(0)))
  stages = in_features//64

  def stage_stores(stage:UOp, buf:UOp|int) -> list[UOp]:
    sb, part = stage//4, stage%4
    stores:list[UOp] = []
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
        return [w_s[buf, masked(r, v), h*8+quarter*4+k].store((qs[k] >> (h*4)) & 0x0f0f0f0f) for h in range(2) for k in range(4)]
      coop(TN*2, weights)
      def coefficients(i, v):
        sub, r = i//TN, i%TN
        head, s0, s1, s2 = _vload(raw, row_base(r), 4)
        a, b, c = ((x >> (((part*2+sub)%4)*8).cast(dtypes.uint32)) & 255 for x in (s0, s1, s2))
        low = part < 2
        scale, minimum = low.where(a & 63, (c & 15) | ((a >> 6) << 4)), low.where(b & 63, (c >> 4) | ((b >> 6) << 4))
        return [c_s[buf, sub, masked(r, v), 0].store(_half(head & 0xffff)*scale.float()),
                c_s[buf, sub, masked(r, v), 1].store(_half(head >> 16)*minimum.float())]
      coop(TN*groups, coefficients)
    else:
      def weights(i, v):
        r, w = i//16, i%16
        n, q, l = part//2, (part%2)*2+w//8, (w%8)*4
        ql, qh = _u32(raw, row_base(r)+(n*64+(q%2)*32+l)//2), _u32(raw, row_base(r)+64+(n*32+l)//2)
        word = ((ql >> ((q//2)*4).cast(dtypes.uint32)) & 0x0f0f0f0f) | (((qh >> (q*2).cast(dtypes.uint32)) & 0x03030303) << 4)
        # bytewise q-32 without borrows: clearing bit 5 subtracts 32 from q >= 32, q < 32 also gets the sign bits 0xc0
        return [w_s[buf, masked(r, v), w].store((word ^ 0x20202020) | (((word ^ 0xffffffff) & 0x20202020) * 6))]
      coop(TN*16, weights)
      def coefficients(i, v):
        sub, r = i//TN, i%TN
        s16 = part*4+sub
        scale = (raw[row_base(r)+96+s16//2].load() >> ((s16%2)*8).cast(dtypes.uint16)).cast(dtypes.uint8).bitcast(dtypes.int8).float()
        return [c_s[buf, sub, masked(r, v), 0].store(_half(raw[row_base(r)+104].load())*scale)]
      coop(TN*groups, coefficients)
    def activations(i, v):
      tk, quarter = i//4, i%4
      x = _vload(xq, (tok0+tk)*(in_features//4)+stage*16+quarter*4, 4)
      return [x_s[buf, masked(tk, v), quarter*4+k].store(x[k]) for k in range(4)]
    coop(TM*4, activations)
    def token_terms(i, v):
      sub, tk = i//TM, i%TM
      tok, grp = tok0+tk, stage*groups+sub
      if not q4: return [t_s[buf, sub, masked(tk, v), 0].store(xd[tok, grp//2].load())]
      d = xd[tok, grp].load()
      total = xs[tok, grp*2].load()+xs[tok, grp*2+1].load()
      return [t_s[buf, sub, masked(tk, v), 0].store(d), t_s[buf, sub, masked(tk, v), 1].store(-total*d)]
    coop(TM*groups, token_terms)
    return stores

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
    for r, a in enumerate(a_frags):
      for j, b in enumerate(b_frags):
        dot = UOp.wmma(a, b, UOp.stack(*(UOp.const(0, dtypes.int32) for _ in range(4))), (8,16,group_size), 32)
        for i in range(4):
          c, tt, v = coeffs[r*2+i//2], terms[j*2+i%2], (r*token_tiles+j)*4+i
          values[v] = values[v] + c[0]*dot[i].float()*tt[0] + (c[1]*tt[1] if q4 else 0)
  update = UOp.group(acc.store(UOp.stack(*values)), *stores).end(stage)
  outs = [out[tok0+wtok+j*8+t*2+i%2, row0+wrow+r*16+g+(i//2)*8].store(acc.after(update)[(r*token_tiles+j)*4+i].load())
          for r in range(row_tiles) for j in range(token_tiles) for i in range(4)]
  return UOp.group(*outs).end(token_block, row_block, lane, warp).sink(arg=KernelInfo(name="q4k_mmq" if q4 else "q6k_mmq", opts_to_apply=()))

def nv_mmq(weight:Tensor, x:Tensor, in_features:int, out_features:int, ggml_type:int, live:UOp|None=None) -> Tensor:
  # live: the symbolic token count of a chunk padded to its max size; the rows past it are left unwritten
  shape = x.shape[:-1]
  tokens = int(x.numel())//in_features
  padded_tokens = (tokens+7)//8*8
  x = x.reshape(tokens, in_features).pad(((0, padded_tokens-tokens), (0, 0)))
  out = Tensor.empty(padded_tokens, out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_mmq_kernel, tokens=padded_tokens, out_features=out_features, in_features=in_features, ggml_type=ggml_type,
                          live=live)
  xq, xd, xs = q8_quantize(x, padded_tokens, in_features)
  if live is not None: xq = Tensor(xq.uop.after(live))
  return Tensor.custom_kernel(out, weight, xq, xd, xs, fxn=fxn)[0][:tokens].reshape(*shape, out_features)
