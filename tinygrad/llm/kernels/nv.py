from __future__ import annotations
import functools, math
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
def _fma(a:UOp, b:UOp, c:UOp) -> UOp: return _custom("fmaf({0}, {1}, {2})", dtypes.float32, a, b, c)
def _half(bits:UOp) -> UOp: return bits.cast(dtypes.uint16).bitcast(dtypes.float16).float()
def _u32(raw:UOp, idx:UOp) -> UOp: return raw[idx].load().cast(dtypes.uint32) | (raw[idx+1].load().cast(dtypes.uint32) << 16)

def _tile(n:int, *sizes:int) -> int: return next(s for s in sizes if n % s == 0)

# ******** q8 activations ********

@functools.cache
def _q8_quantize_kernel(xq:UOp, xd:UOp, xs:UOp, x:UOp, *up:UOp, tokens:int, in_features:int) -> UOp:
  # each lane loads four consecutive values and owns one packed word: a warp covers four groups of 32, the group maximum
  # reduces over 8 lanes and the per-16 sums over 4. Several warps per block keep the occupancy up.
  chunks = in_features//128
  warps = _tile(tokens*chunks, 8, 4, 2, 1)
  block, lane, warp = UOp.range(tokens*chunks//warps, 0, AxisType.GLOBAL), UOp.range(32, -1, AxisType.WARP), \
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
  stores = (xq[token, chunk*32 + lane].store(word), xd[token, group.valid(w.eq(0))].store(d),
            xs[token, group*2 + (w//4).valid((w%4).eq(0))].store(warp_reduce(sum(quants[1:], quants[0]), width=4)))
  return UOp.group(*stores).end(block, lane, warp).sink(arg=KernelInfo(name="q8_quantize", opts_to_apply=()))

def q8_quantize(x:Tensor, tokens:int, in_features:int, cache:dict|None=None, up:Tensor|None=None) -> tuple[Tensor, Tensor, Tensor]:
  # cache: shared by projections of the same input (Linear.project), which then quantize it once
  # up: x is a SwiGLU gate, the quantized input is silu(x)*up (never written out)
  if cache is not None and up is None and (hit:=cache.get((tokens, in_features))) is not None: return hit
  xq = Tensor.empty(tokens, in_features//4, dtype=dtypes.uint32, device=x.device)
  xd = Tensor.empty(tokens, in_features//32, dtype=dtypes.float32, device=x.device)
  xs = Tensor.empty(tokens, in_features//16, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_q8_quantize_kernel, tokens=tokens, in_features=in_features)
  xq, xd, xs = Tensor.custom_kernel(xq, xd, xs, x.contiguous(), *(() if up is None else (up.contiguous(),)), fxn=fxn)[:3]
  if cache is not None and up is None: cache[(tokens, in_features)] = (xq, xd, xs)
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

def nv_q8_linear(weight:Tensor, ggml_type:int, x:Tensor, in_features:int, out_features:int, q8:dict|None=None,
                 up:Tensor|None=None) -> Tensor:
  tokens = int(x.numel()) // in_features
  out = Tensor.empty(tokens, out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_q4k_gemv_kernel if ggml_type == Q4_K else _q6k_gemv_kernel, tokens=tokens, out_features=out_features,
                          in_features=in_features)
  return Tensor.custom_kernel(out, weight, *q8_quantize(x, tokens, in_features, q8, up), fxn=fxn)[0].reshape(*x.shape[:-1], out_features)

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

@functools.cache
def _mmq_kernel(out:UOp, raw:UOp, xq:UOp, xd:UOp, xs:UOp, tokens:int, out_features:int, in_features:int, ggml_type:int,
                live:UOp|None=None) -> UOp:
  # Each warp owns row_tiles 16-row x token_tiles 8-token MMA tiles, so every weight fragment feeds token_tiles MMAs and every
  # activation fragment row_tiles MMAs. The block stages 64 weights per row at a time: it unpacks its rows to int8 words in
  # shared memory, decodes each row's group coefficients once, and stages the activations next to them.
  # Q4_K groups (32) match m16n8k32; Q6_K scales every 16 weights, so it uses m16n8k16.
  q4 = ggml_type == Q4_K
  group_size, words = (32, 36) if q4 else (16, 105)
  # weights per row per stage: 128-weight stages halve the barriers per K and their 90 KB need dynamic shared memory. Measured on an
  # RTX 4060 at 512 tokens, they speed up Q4_K with out_features <= in_features (attention 4096x4096 -7%, FFN down -6%) but slow
  # the wide gate/up shapes (+10%) and Q6_K (+9%)
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
      d = xd[tok, grp].load()
      total = xs[tok, grp*2].load()+xs[tok, grp*2+1].load()
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

def nv_mmq(weight:Tensor, x:Tensor, in_features:int, out_features:int, ggml_type:int, live:UOp|None=None, q8:dict|None=None,
           up:Tensor|None=None) -> Tensor:
  # live: the symbolic token count of a chunk padded to its max size; the rows past it are left unwritten
  shape = x.shape[:-1]
  tokens = int(x.numel())//in_features
  padded_tokens = (tokens+7)//8*8
  x = x.reshape(tokens, in_features).pad(((0, padded_tokens-tokens), (0, 0)))
  if up is not None: up = up.reshape(tokens, in_features).pad(((0, padded_tokens-tokens), (0, 0)))
  out = Tensor.empty(padded_tokens, out_features, dtype=dtypes.float32, device=x.device)
  fxn = functools.partial(_mmq_kernel, tokens=padded_tokens, out_features=out_features, in_features=in_features, ggml_type=ggml_type,
                          live=live)
  xq, xd, xs = q8_quantize(x, padded_tokens, in_features, q8, up)
  if live is not None: xq = Tensor(xq.uop.after(live))
  return Tensor.custom_kernel(out, weight, xq, xd, xs, fxn=fxn)[0][:tokens].reshape(*shape, out_features)

# ******** prefill: flash attention ********

FA_BLOCK_M, FA_BLOCK_N, FA_WARPS, LOG2E = 64, 64, 4, math.log2(math.e)

@functools.cache
def _flash_attention_kernel(o:UOp, q:UOp, cache:UOp, valid_kv_len:UOp|int, q_start:UOp|int) -> UOp:
  # FlashAttention-2 on m16n8k16 half MMAs: each warp owns 16 query rows of a 64-row block and keeps them in registers. Per
  # 64-key tile the block stages K and V^T in shared memory; the S accumulators are reused as the A fragments of P@V.
  # q and the cache arrive as uint32 views: two halves per word
  BH, M, words = q.shape
  _, B, H_KV, N, _ = cache.shape
  group, D = BH//(B*H_KV), words*2
  assert M % FA_BLOCK_M == 0 and D % 16 == 0 and N % FA_BLOCK_N == 0
  valid_kv_len = valid_kv_len.unbind_all()[0] if isinstance(valid_kv_len, UOp) else valid_kv_len
  q_start = q_start.unbind_all()[0] if isinstance(q_start, UOp) else q_start
  # a padded symbolic chunk launches only the query blocks holding its valid_kv_len - q_start live rows
  live = M if isinstance(valid_kv_len, int) and isinstance(q_start, int) else valid_kv_len - q_start
  block_bh, block_m = UOp.range(BH, 0, AxisType.GLOBAL), UOp.range((live + FA_BLOCK_M - 1)//FA_BLOCK_M, 1, AxisType.GLOBAL)
  lane, warp = UOp.range(32, -1, AxisType.WARP), UOp.range(FA_WARPS, 2, AxisType.LOCAL)
  tid, g, t = warp*32 + lane, lane//4, lane%4
  kv = block_bh//group  # (batch, kv head)
  qw, cw = q, cache.reshape(2, B*H_KV, N, words)
  # padded strides keep the fragment reads (8 rows x 4 words per instruction) in distinct banks
  K_s = UOp.alloc((FA_BLOCK_N, words + 4), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  V_s = UOp.alloc((D, FA_BLOCK_N//2 + 4), dtypes.uint32, addrspace=AddrSpace.LOCAL)
  rows = [block_m*FA_BLOCK_M + warp*16 + g + h*8 for h in range(2)]
  q_frags = [UOp.stack(*(qw[block_bh, rows[i%2], ks*8 + t + (i//2)*4].load() for i in range(4))).bitcast(dtypes.half) for ks in range(D//16)]
  n_tiles = D//8  # output column tiles of P@V
  acc = UOp.alloc((n_tiles*4,), dtypes.float32, addrspace=AddrSpace.REG)
  stats = UOp.alloc((4,), dtypes.float32, addrspace=AddrSpace.REG)  # (row max, partial row sum) for rows g and g+8
  acc = acc.after(acc.store(acc.const_like(0)))
  stats = stats.after(stats.store(UOp.stack(*(UOp.const(x, dtypes.float32) for x in (-1e30, -1e30, 0, 0)))))
  last_key = (q_start + (block_m+1)*FA_BLOCK_M).minimum(valid_kv_len)
  tile = UOp.range((last_key + FA_BLOCK_N - 1)//FA_BLOCK_N, 3, AxisType.LOOP)
  threads = 32*FA_WARPS
  stores = []
  for i in range(FA_BLOCK_N*words//4//threads):  # K rows: 16-byte chunks straight into shared memory
    idx = tid + i*threads
    key, chunk = idx//(words//4), idx%(words//4)
    kvals = _vload(cw, ((0*B*H_KV + kv)*N + tile*FA_BLOCK_N + key)*words + chunk*4, 4)
    stores += [K_s[key, chunk*4 + j].store(kvals[j]) for j in range(4)]
  for i in range(FA_BLOCK_N//2*words//4//threads):  # V^T: two keys' 16-byte chunks become eight (d, key pair) words
    idx = tid + i*threads
    pair, chunk = idx//(words//4), idx%(words//4)
    r0, r1 = (_vload(cw, ((1*B*H_KV + kv)*N + tile*FA_BLOCK_N + pair*2 + e)*words + chunk*4, 4) for e in range(2))
    for j in range(4):
      stores.append(V_s[chunk*8 + j*2, pair].store((r0[j] & 0xffff) | (r1[j] << 16)))
      stores.append(V_s[chunk*8 + j*2 + 1, pair].store((r0[j] >> 16) | (r1[j] & 0xffff0000)))
  K_r, V_r = K_s.after(*stores), V_s.after(*stores)
  zero = UOp.stack(*(UOp.const(0.0, dtypes.float32) for _ in range(4)))
  # S = Q K^T for 8 key tiles of 8
  s = []
  for j in range(FA_BLOCK_N//8):
    c = zero
    for ks in range(D//16):
      b = UOp.stack(*(K_r[j*8 + g, ks*8 + t + i*4].load() for i in range(2))).bitcast(dtypes.half)
      c = UOp.wmma(q_frags[ks], b, c, (8,16,16), 32)
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
    for n in range(n_tiles):
      b = UOp.stack(*(V_r[n*8 + g, kstep*8 + t + i*4].load() for i in range(2))).bitcast(dtypes.half)
      res = UOp.wmma(a, b, UOp.stack(*outs[n*4:n*4+4]), (8,16,16), 32)
      outs[n*4:n*4+4] = [res[i] for i in range(4)]
  update = UOp.group(acc.store(UOp.stack(*outs)), stats.store(UOp.stack(*new_max, *sums))).end(tile)
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
  qp = q.pad_to((B, H, T_pad, D)) if T_pad != q.max_shape[2] or not isinstance(T, int) else q
  out = Tensor.empty(B*H, T_pad, D, dtype=dtypes.float32, device=q.device)
  if isinstance(valid_end, UOp): assigned_kv = Tensor(assigned_kv.uop.after(valid_end))
  fxn = functools.partial(_flash_attention_kernel, valid_kv_len=valid_end, q_start=q_start)
  qw = qp.half().reshape(B*H, T_pad, D).contiguous().bitcast(dtypes.uint32)
  out = Tensor.custom_kernel(out, qw, assigned_kv.bitcast(dtypes.uint32), fxn=fxn)[0].reshape(B, H, T_pad, D)
  return out if qp is q else out[:, :, :T]
