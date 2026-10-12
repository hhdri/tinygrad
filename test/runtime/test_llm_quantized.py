import gc, itertools, unittest, weakref
import numpy as np
from tinygrad import Tensor, UOp, dtypes, function, Device
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported, QUANT_SIZES, HALFWORD_QUANTS, iq4_half_lut, _iq_grid, flash_attention
from tinygrad.llm.kernels.nv import nv_custom_kernels_supported, nv_mmq, nv_flash_attention, nv_mmq_supported, nv_qk_prep, nv_rmsnorm_q8, \
  nv_flash_decode_q8
from tinygrad.llm.gguf import ggml_data_to_tensor
from test.helpers import not_support_multi_device

class QuantLinearMixin:
  def _test_quant_linear(self, ggml_type, block_bytes, in_features=2048, out_features=64, token_counts=(1, 3, 32, 64, 128),
                         bias=False, custom=True, symbolic=False):
    if custom and not amd_custom_kernels_supported(Tensor.empty(1).device): self.skipTest("RDNA3 required")
    rng = np.random.default_rng(42)
    packed = rng.integers(0, 256, (out_features*in_features//256, block_bytes), dtype=np.uint8)
    if ggml_type in (11, 14): packed[:, -2:] = np.array([0.001], dtype=np.float16).view(np.uint8)
    elif ggml_type == 10:
      packed[:, 80:82] = np.array([0.001], dtype=np.float16).view(np.uint8)
      packed[:, 82:84] = np.array([0.0002], dtype=np.float16).view(np.uint8)
    elif ggml_type == 20: packed.reshape(-1, 18)[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    else: packed[:, :2] = np.array([0.001], dtype=np.float16).view(np.uint8)
    if ggml_type in (12, 13): packed[:, 2:4] = np.array([0.0002], dtype=np.float16).view(np.uint8)
    raw = Tensor(np.pad(packed.flatten(), (4, 0))).contiguous().realize()[4:]
    decoded = ggml_data_to_tensor(raw, out_features*in_features, ggml_type).reshape(out_features, in_features)
    weight = decoded.numpy()
    linear = Linear(in_features, out_features, bias=False)
    linear.weight = decoded
    bias_value = rng.normal(size=out_features).astype(np.float32) if bias else 0
    if bias: linear.bias = Tensor(bias_value)
    @function(allow_implicit=True)
    def run(x:Tensor): return linear(x)
    for tokens in token_counts:
      with self.subTest(tokens=tokens):
        x = rng.normal(size=(tokens, in_features)).astype(np.float32 if tokens == 3 else np.float16)
        reference_x = x.astype(np.float32)
        wmma = custom and (32 if symbolic else tokens) % 16 == 0 and out_features % 16 == 0
        if wmma: reference_x = x.astype(np.float16).astype(np.float32)
        if custom and not wmma:
          grouped = reference_x.reshape(tokens, -1, 32)
          scale = np.maximum(np.abs(grouped).max(-1, keepdims=True) / 127, 1e-8)
          reference_x = (np.clip(np.rint(grouped/scale), -127, 127)*scale).reshape(tokens, in_features)
        reference_w = weight.astype(np.float16).astype(np.float32) if wmma else weight
        inp = Tensor(x) if not symbolic else Tensor(np.pad(x, ((0, 32-tokens), (0, 0)))).contiguous()[:
          UOp.variable("wmma_tokens", 1, 32).bind(tokens)]
        actual = (run if tokens == 1 or symbolic else linear)(inp)[:tokens].numpy()
        self.assertEqual(linear.ggml_type, ggml_type if custom else None)
        np.testing.assert_allclose(actual, reference_x @ reference_w.T + bias_value, rtol=3e-3, atol=2e-2)
        if not symbolic and tokens == 3 and ggml_type not in (12, 13, 14, 23):
          sym = Tensor(np.pad(x, ((0, 1), (0, 0)))).contiguous()[:UOp.variable("tokens", 1, 4).bind(3)]
          np.testing.assert_allclose(linear(sym)[:3].numpy(), reference_x @ reference_w.T + bias_value, rtol=3e-3, atol=2e-2)
    self.assertEqual(linear.ggml_type, ggml_type if custom else None)

class TestQ8Quantize(QuantLinearMixin, unittest.TestCase):
  def test_quant_tables_not_retained(self):
    # one _iq_grid table and the iq4 lut cover both table creation paths
    for typ in (18, 23):
      table = (iq4_half_lut(Device.DEFAULT) if typ == 23 else _iq_grid(Device.DEFAULT, typ)).realize()
      ref = weakref.ref(table)
      del table
      gc.collect()
      self.assertIsNone(ref())

  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_quant_tables_multidevice(self):
    devices = (Device.DEFAULT, f"{Device.DEFAULT}:1")
    for typ in (18, 23):
      with self.subTest(ggml_type=typ):
        single = iq4_half_lut(Device.DEFAULT) if typ == 23 else _iq_grid(Device.DEFAULT, typ)
        table = (iq4_half_lut(devices) if typ == 23 else _iq_grid(devices, typ)).realize()
        self.assertEqual(table.device, devices)
        for i in range(len(devices)):
          np.testing.assert_array_equal(Tensor(table.uop.mselect(i)).numpy(), single.numpy())

  @unittest.skipIf(not_support_multi_device(), "no multi")
  def test_quant_linear_multidevice(self):
    if not amd_custom_kernels_supported(Device.DEFAULT): self.skipTest("RDNA3 required")
    devices = (Device.DEFAULT, f"{Device.DEFAULT}:1")
    rng = np.random.default_rng(42)
    # Exercise both the grid-backed decode path and the IQ4 WMMA lookup table without a TP model.
    for typ, tokens in ((18, 1), (23, 16)):
      with self.subTest(ggml_type=typ):
        raw = Tensor(rng.integers(0, 0x3c, 16*QUANT_SIZES[typ]+4, dtype=np.uint8))
        x = Tensor(rng.normal(size=(tokens, 256)).astype(np.float32))
        def run(device):
          linear = Linear(256, 16, bias=False)
          linear.weight = ggml_data_to_tensor(raw.to(device).realize()[4:], 16*256, typ).reshape(16, 256)
          out = linear(x.to(device)).realize()
          self.assertEqual(linear.ggml_type, typ)
          return out
        single, parallel = run(Device.DEFAULT).numpy(), run(devices)
        for i in range(len(devices)):
          np.testing.assert_allclose(Tensor(parallel.uop.mselect(i)).numpy(), single, atol=1e-5, rtol=1e-5)

  def test_quant_weights_share_storage(self):
    for (ggml_type, type_size), offset in itertools.product(QUANT_SIZES.items(), (0, 4)):
      with self.subTest(ggml_type=ggml_type, offset=offset):
        packed = np.arange(type_size + offset, dtype=np.uint8)
        raw = Tensor(packed).realize()[offset:]
        if raw.uop.contiguous_view() is None: self.skipTest("requires buffer views")
        decoded = ggml_data_to_tensor(raw, 256, ggml_type).reshape(1, 256)
        linear = Linear(256, 1, bias=False)
        linear.set_quantized(decoded)
        linear.weight.realize()
        self.assertEqual(linear.ggml_type, ggml_type)
        self.assertEqual(linear.weight.dtype, dtypes.uint16 if ggml_type in HALFWORD_QUANTS else dtypes.uint32)
        self.assertEqual(linear.weight.nbytes(), type_size)
        np.testing.assert_array_equal(linear.weight.bitcast(dtypes.uint8).numpy(), packed[offset:])
        raw.assign(raw.full_like(1)).realize()
        np.testing.assert_array_equal(linear.weight.bitcast(dtypes.uint8).numpy(), np.ones(type_size, dtype=np.uint8))

  @unittest.skipIf(Device.DEFAULT == "WEBGPU", "slow on WEBGPU")
  def test_quant_linear_fallback(self):
    if amd_custom_kernels_supported(dev:=Tensor.empty(1).device) or nv_custom_kernels_supported(dev):
      self.skipTest("run with DISABLE_AMD_KERNELS=1 DISABLE_NV_KERNELS=1")
    # per-type dequant math on the generic path is covered by test_gguf, spot check a representative set here
    for typ in (12, 14, 17, 23):
      with self.subTest(ggml_type=typ):
        self._test_quant_linear(typ, QUANT_SIZES[typ], in_features=256, out_features=16, token_counts=(1, 3), bias=True, custom=False)

class TestNVQuantLinear(unittest.TestCase):
  def setUp(self):
    if not nv_custom_kernels_supported(Device.DEFAULT): self.skipTest("NV or CUDA (sm_61+) required")

  def test_mmq(self):
    if int(Device[Device.DEFAULT].renderer.target.arch[3:]) < 80: self.skipTest("SM80 required")
    rng = np.random.default_rng(42)
    # shapes cover 1-4 row and token tiles per warp, 1-2 row and 1-4 token warps per block, and padded tokens
    for typ,(k,n,t) in itertools.product((12,14), ((256,16,8), (1024,32,16), (256,16,3), (512,128,128), (256,48,40))):
      for d,dmin,zero in ((0.001,0.0002,False), (-0.01,-0.002,False), (0.001,0.0002,True)):
        with self.subTest(typ=typ, k=k, n=n, tokens=t, d=d, zero=zero):
          packed = rng.integers(0,256,(n*k//256,QUANT_SIZES[typ]),dtype=np.uint8)
          if typ == 12: packed[:,:4] = np.array([d,dmin],np.float16).view(np.uint8)
          else: packed[:,-2:] = np.array([d],np.float16).view(np.uint8)
          raw = Tensor(packed.ravel()).realize()
          weight = ggml_data_to_tensor(raw,n*k,typ).reshape(n,k).numpy().astype(np.float64)
          x = rng.normal(size=(t,k)).astype(np.float32)
          if zero: x.fill(0)
          grouped = x.reshape(t,-1,32)
          scale = np.maximum(np.abs(grouped).max(-1,keepdims=True)/127,1e-30)
          reference = (np.rint(grouped/scale)*scale).reshape(t,k).astype(np.float64) @ weight.T
          actual = nv_mmq(raw.bitcast(dtypes.uint32 if typ == 12 else dtypes.uint16),Tensor(x),k,n,typ).numpy()
          np.testing.assert_allclose(actual,reference,atol=2e-4,rtol=2e-4)

  def test_mmq_symbolic_dispatch(self):
    if int(Device[Device.DEFAULT].renderer.target.arch[3:]) < 80: self.skipTest("SM80 required")
    rng = np.random.default_rng(11)
    for typ in (12,14):
      # 384 padded tokens span three 128-token blocks: 3 and 16 live tokens launch only the first, 130 and 200 two, 300 all. Live counts
      # ending in a 1-64 token tail (3, 16, 130, and 300 for Q4_K) run on the 64-token-block kernels, the others on the 128-token ones
      for tokens in (3,16,100,130,200,300):
        with self.subTest(typ=typ, tokens=tokens):
          packed = rng.integers(0,256,(128*1024//256,QUANT_SIZES[typ]),dtype=np.uint8)
          if typ == 12: packed[:,:4] = np.array([.001,.0002],np.float16).view(np.uint8)
          else: packed[:,-2:] = np.array([.001],np.float16).view(np.uint8)
          raw = Tensor(packed.ravel()).realize()
          layer = Linear(1024,128,bias=False)
          layer.weight = ggml_data_to_tensor(raw,128*1024,typ).reshape(128,1024)
          x = rng.normal(size=(tokens,1024)).astype(np.float32)
          inp = Tensor(np.pad(x,((0,384-tokens),(0,0)))).contiguous()[:UOp.variable("mmq_tokens",1,384).bind(tokens)]
          actual = function(lambda x: layer(x), allow_implicit=True)(inp)[:tokens].numpy()
          np.testing.assert_allclose(actual,layer(Tensor(x)).numpy(),atol=2e-4,rtol=2e-4)
          # a lazy input is written into an uninitialized max-size buffer (live rows only) instead of passed on as a buffer slice
          actual = function(lambda x: layer(x*2), allow_implicit=True)(inp)[:tokens].numpy()
          np.testing.assert_allclose(actual,layer(Tensor(x*2)).numpy(),atol=4e-4,rtol=2e-4)

  def test_qk_prep(self):
    # the decode attention prologue kernel vs the generic path: per-head RMS norm with weights, rotary embedding, KV cache write
    from tinygrad.llm.model import apply_rope, precompute_freqs_cis
    rng = np.random.default_rng(3)
    H, HKV, D, N, eps = 8, 2, 128, 64, 1e-6
    q, k, v = (Tensor(rng.normal(size=(1, 1, h*D)).astype(np.float32)).realize() for h in (H, HKV, HKV))
    qw, kw = (Tensor(rng.normal(size=(D,)).astype(np.float32)).realize() for _ in range(2))
    freqs = precompute_freqs_cis(D, N, 1e6).realize()
    for sp in (0, 5, N-1):
      with self.subTest(start_pos=sp):
        cache = Tensor(rng.normal(size=(2, 1, HKV, N, D)).astype(np.float16)).realize()
        before = cache.numpy()
        start = UOp.variable("start_pos", 0, N-1).bind(sp)
        qh, kv = nv_qk_prep(q, k, v, qw, kw, freqs, cache, start, H, HKV, D, eps)
        Tensor.realize(qh, kv)  # together: realizing one output of a multi-output custom kernel leaves the other to run it again
        def ref(x, h, w): return apply_rope((x.reshape(1, 1, h, D).float().rmsnorm(eps=eps)*w).transpose(1, 2), freqs[sp:sp+1])
        np.testing.assert_allclose(qh.float().numpy(), ref(q, H, qw).half().float().numpy(), atol=2e-3, rtol=2e-3)
        got, want = kv.numpy(), before.copy()
        want[0, 0, :, sp] = ref(k, HKV, kw).half().numpy()[0, :, 0]
        want[1, 0, :, sp] = v.reshape(HKV, D).half().numpy()
        np.testing.assert_allclose(got.astype(np.float32), want.astype(np.float32), atol=2e-3, rtol=2e-3)

  def test_rmsnorm_q8(self):
    # one decode token: the residual sum, RMS norm and q8 quantization kernel vs the generic norm (dequantized within a step)
    rng = np.random.default_rng(4)
    D, eps = 4096, 1e-6
    x, a = (Tensor(rng.normal(size=(1, 1, D)).astype(np.float32)*3).realize() for _ in range(2))
    w = Tensor(rng.normal(size=(D,)).astype(np.float32)).realize()
    for parts, write_sum, write_normed in (([x, a], True, False), ([x], False, True)):
      with self.subTest(parts=len(parts)):
        h, xn, q8 = nv_rmsnorm_q8(parts, w, eps, write_sum=write_sum, write_normed=write_normed)
        xq, xd, xs = q8[(1, D, True)]
        Tensor.realize(*(t for t in (h, xn, xq, xd, xs) if t is not None))
        s_ = (x + a if len(parts) == 2 else x).numpy().reshape(D)
        ref = s_ * (1/np.sqrt((s_.astype(np.float64)**2).mean() + eps)) * w.numpy()
        if write_sum: np.testing.assert_allclose(h.numpy().reshape(D), s_, atol=1e-6, rtol=1e-6)
        if write_normed: np.testing.assert_allclose(xn.numpy().reshape(D), ref, atol=1e-4, rtol=1e-4)
        q = xq.numpy().reshape(-1).view(np.int8).astype(np.float32)
        d = np.repeat(xd.numpy().reshape(-1), 32)
        np.testing.assert_allclose(q*d, ref, atol=float(d.max())*0.51)
        sums = q.reshape(-1, 16).sum(-1)
        np.testing.assert_allclose(xs.numpy().reshape(-1), sums)

  def test_flash_decode_q8(self):
    # the flash-decoding combine with q8 output vs the shared combine's float output (dequantized within a step)
    from tinygrad.llm.kernels.amd import flash_attention
    rng = np.random.default_rng(6)
    H, HKV, D, N = 32, 8, 128, 512
    q = Tensor(rng.normal(size=(1, H, 1, D)).astype(np.float16)).realize()
    cache = Tensor(rng.normal(size=(2, 1, HKV, N, D)).astype(np.float16)).realize()
    for end in (1, 77, N):
      with self.subTest(valid_end=end):
        v = UOp.variable("start_pos", 1, N).bind(end)
        attn, q8 = nv_flash_decode_q8(q, cache, v, N)
        xq, xd, xs = q8[(1, H*D, True)]
        Tensor.realize(xq, xd, xs)
        ref = flash_attention(q, cache, v).numpy().reshape(-1)
        qv = xq.numpy().reshape(-1).view(np.int8).astype(np.float32)
        d = np.repeat(xd.numpy().reshape(-1), 32)
        np.testing.assert_allclose(qv*d, ref, atol=float(d.max())*0.51+1e-6)
        np.testing.assert_allclose(xs.numpy().reshape(-1), qv.reshape(-1, 16).sum(-1))

  def test_swiglu(self):
    # silu(gate)*up quantized inside the q8 kernel vs the product as an ordinary input: decode, static and symbolic prefill (gate and up
    # are slices of padded buffers, passed on whole), and a dense layer that falls back to the product
    rng = np.random.default_rng(5)
    mmq = int(Device[Device.DEFAULT].renderer.target.arch[3:]) >= 80
    for typ in (12, 14):
      packed = rng.integers(0, 256, (32*1024//256, QUANT_SIZES[typ]), dtype=np.uint8)
      if typ == 12: packed[:, :4] = np.array([.001, .0002], np.float16).view(np.uint8)
      else: packed[:, -2:] = np.array([.001], np.float16).view(np.uint8)
      layer = Linear(1024, 32, bias=False)
      layer.weight = ggml_data_to_tensor(Tensor(packed.ravel()).realize(), 32*1024, typ).reshape(32, 1024)
      for tokens, symbolic in ((1, False), (40, False), (3, True), (130, True)):
        if tokens > 1 and not mmq: continue
        with self.subTest(typ=typ, tokens=tokens, symbolic=symbolic):
          g, u = (rng.normal(size=(1, tokens, 1024)).astype(np.float32)*3 for _ in range(2))
          expected = layer(Tensor(g).silu()*Tensor(u)).numpy()
          if symbolic:
            v = UOp.variable("swiglu_tokens", 1, 256).bind(tokens)
            gi, ui = (Tensor(np.pad(a, ((0, 0), (0, 256-tokens), (0, 0)))).contiguous()[:, :v] for a in (g, u))
            actual = function(lambda g, u: layer.swiglu(g, u), allow_implicit=True)(gi, ui)[:, :tokens].numpy()
          else: actual = layer.swiglu(Tensor(g), Tensor(u)).numpy()
          np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
          if not symbolic:  # a residual is added to the output (the decode GEMV adds it in its store)
            r = rng.normal(size=(1, tokens, 32)).astype(np.float32)
            np.testing.assert_allclose(layer.swiglu(Tensor(g), Tensor(u), Tensor(r)).numpy(), expected + r, atol=2e-4, rtol=2e-4)
    dense = Linear(256, 4, bias=False)
    dense.weight = Tensor(rng.normal(size=(4, 256)).astype(np.float16))
    g, u = (Tensor(rng.normal(size=(1, 1, 256)).astype(np.float32)) for _ in range(2))
    np.testing.assert_allclose(dense.swiglu(g, u).numpy(), dense(g.silu()*u).numpy(), atol=1e-5, rtol=1e-5)

  def test_dense_decode(self):
    rng = np.random.default_rng(3)
    for k, n in ((256, 4), (4096, 64), (4096, 1024), (768, 7), (384, 4)):
      for dtype in (np.float16, np.float32):
        with self.subTest(k=k, n=n, dtype=dtype):
          weight = rng.normal(size=(n, k)).astype(np.float16)
          linear = Linear(k, n, bias=True)
          linear.weight = Tensor(weight).cast(dtypes.float32)
          bias = rng.normal(size=n).astype(np.float32)
          linear.bias = Tensor(bias)
          for tokens in (1, 3):
            x = rng.normal(size=(1, tokens, k)).astype(dtype)
            # prefill is an F16 matmul with F32 accumulation; without tensor cores the products round to F16
            ref_x, tol = (x.astype(np.float16).astype(np.float32), 1e-3*np.sqrt(k)) if tokens > 1 else (x.astype(np.float32), 2e-4)
            np.testing.assert_allclose(linear(Tensor(x)).numpy(), ref_x @ weight.astype(np.float32).T + bias, rtol=tol, atol=tol)
          np.testing.assert_allclose(linear(Tensor.zeros(1, k)).numpy(), bias[None], atol=1e-6)

  def test_dense_extreme_values(self):
    weight = np.tile(np.array([65504, -65504, 6e-8, -6e-8], np.float16), (4, 64))
    linear = Linear(256, 4, bias=False)
    linear.weight = Tensor(weight)
    x = np.tile(np.array([1, -1, 1e-3, -1e-3], np.float32), (1, 64))
    np.testing.assert_allclose(linear(Tensor(x)).numpy(), x @ weight.astype(np.float32).T, rtol=1e-6)

  def _check(self, ggml_type:int, in_features:int, out_features:int, d:float, dmin:float, x:np.ndarray, offset:int=0):
    rng = np.random.default_rng(ggml_type)
    packed = rng.integers(0, 256, (out_features*in_features//256, QUANT_SIZES[ggml_type]), dtype=np.uint8)
    if ggml_type == 14: packed[:, -2:] = np.array([d], dtype=np.float16).view(np.uint8)
    else: packed[:, :4] = np.array([d, dmin], dtype=np.float16).view(np.uint8)
    # the offset checks views into the GGUF storage that are only halfword aligned
    raw = Tensor(np.pad(packed.flatten(), (offset, 0))).contiguous().realize()[offset:]
    decoded = ggml_data_to_tensor(raw, out_features*in_features, ggml_type).reshape(out_features, in_features)
    weight = decoded.numpy().astype(np.float64)
    linear = Linear(in_features, out_features, bias=False)
    linear.weight = decoded
    actual = linear(Tensor(x)).numpy()
    self.assertEqual(linear.ggml_type, ggml_type)
    # quantize in float32 like the kernel: float64 division rounds some ties the other way
    grouped = x.astype(np.float32).reshape(*x.shape[:-1], -1, 32)
    scale = np.maximum(np.abs(grouped).max(-1, keepdims=True) / np.float32(127), np.float32(1e-30))
    ref = (np.clip(np.rint(grouped/scale), -127, 127).astype(np.float64)*scale).reshape(x.shape) @ weight.T
    np.testing.assert_allclose(actual, ref, rtol=2e-4, atol=2e-4*max(1.0, float(np.abs(ref).max())))

  def test_decode_shapes(self):
    rng = np.random.default_rng(0)
    for ggml_type, (k, n) in itertools.product((12, 14), ((1024, 4), (4096, 64), (12288, 8))):
      with self.subTest(ggml_type=ggml_type, k=k, n=n):
        self._check(ggml_type, k, n, 0.001, 0.0002, rng.normal(size=(1, k)).astype(np.float16))

  def test_decode_edge_values(self):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1, 1, 2048)).astype(np.float32) * np.repeat(rng.choice([0, 1e-3, 1, 300], 64), 32)
    for ggml_type, (d, dmin) in itertools.product((12, 14), ((0.0, 0.0), (-0.75, 0.5), (60000.0, -60000.0), (6e-8, 3e-5))):
      with self.subTest(ggml_type=ggml_type, d=d, dmin=dmin):
        self._check(ggml_type, 2048, 16, d, dmin, x, offset=2 if ggml_type == 14 else 0)
    self._check(12, 2048, 16, 0.01, 0.01, np.zeros((1, 2048), np.float32))

  def test_unsupported_shapes_fall_back(self):
    for typ, (k, n) in ((12, (512, 4)), (14, (1024, 6))):
      raw = Tensor(np.random.default_rng(2).integers(0, 0x3c, n*k//256*QUANT_SIZES[typ], dtype=np.uint8)).realize()
      linear = Linear(k, n, bias=False)
      linear.weight = ggml_data_to_tensor(raw, n*k, typ).reshape(n, k)
      x = Tensor(np.ones((1, k), np.float32))
      np.testing.assert_allclose(linear(x).numpy(), (x @ linear.weight.T).numpy(), rtol=1e-5)
      self.assertIsNone(linear.ggml_type)

if __name__ == "__main__": unittest.main()

class TestNVFlashAttention(unittest.TestCase):
  def setUp(self):
    if not nv_mmq_supported(Device.DEFAULT): self.skipTest("NV or CUDA (sm_80+) required")

  def test_flash_attention_prefill(self):
    # causal attention over the half KV cache vs float64 NumPy: GQA, chunk offsets, padded symbolic chunks, a single token
    rng = np.random.default_rng(0)
    B, H, KVH, D, N = 1, 4, 2, 64, 256
    # T_max None: a static length that is not a whole 64-row block
    for start, T, T_max in ((0, 64, 64), (0, 37, 64), (64, 100, 128), (5, 128, 128), (0, 1, 64), (3, 45, None)):
      with self.subTest(start=start, T=T, T_max=T_max):
        cache = np.zeros((2, B, KVH, N, D), np.float16)
        cache[:, :, :, :start+T] = rng.standard_normal((2, B, KVH, start+T, D)).astype(np.float16)
        q = rng.standard_normal((B, H, T, D)).astype(np.float32)
        if T_max is None: actual = nv_flash_attention(Tensor(q).realize(), Tensor(cache).realize(), start+T).numpy()
        else:
          qt = Tensor(np.pad(q, ((0,0), (0,0), (0,T_max-T), (0,0)))).realize()
          v = UOp.variable("toks", 1, T_max).bind(T)
          qin, end = (qt, start+T) if T == T_max else (qt[:, :, :v], start+v)
          out = nv_flash_attention(qin, Tensor(cache).realize(), end).realize()
          actual = np.frombuffer(out.uop.base.buffer.as_memoryview(), np.float32).reshape(B, H, T_max, D)[:, :, :T]
        k, v = (np.repeat(cache[i, :, :, :start+T].astype(np.float64), H//KVH, axis=1) for i in range(2))
        scores = q.astype(np.float16).astype(np.float64) @ k.transpose(0, 1, 3, 2) / np.sqrt(D)
        scores[..., np.arange(start+T)[None, :] > (start + np.arange(T))[:, None]] = -np.inf
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        np.testing.assert_allclose(actual, (probs / probs.sum(-1, keepdims=True)) @ v, atol=5e-3, rtol=5e-3)

  def test_flash_attention_decode(self):
    # single-token attention over the half KV cache vs float64 NumPy: Qwen3's 4-query GQA group, chunk boundaries, a context longer
    # than one round of partial chunks, and a symbolic length
    rng = np.random.default_rng(1)
    B, H, KVH, D, N = 1, 8, 2, 128, 4096
    for L, symbolic in ((1, False), (31, False), (32, False), (33, False), (560, True), (2049, False), (4096, True)):
      with self.subTest(L=L, symbolic=symbolic):
        cache = np.zeros((2, B, KVH, N, D), np.float16)
        cache[:, :, :, :L] = rng.standard_normal((2, B, KVH, L, D)).astype(np.float16)
        q = rng.standard_normal((B, H, 1, D)).astype(np.float32)
        end = UOp.variable("start_pos", 0, N-1).bind(L-1) + 1 if symbolic else L
        actual = flash_attention(Tensor(q).realize(), Tensor(cache).realize(), end).numpy()
        k, v = (np.repeat(cache[i, :, :, :L].astype(np.float64), H//KVH, axis=1) for i in range(2))
        scores = q.astype(np.float16).astype(np.float64) @ k.transpose(0, 1, 3, 2) / np.sqrt(D)
        probs = np.exp(scores - scores.max(-1, keepdims=True))
        np.testing.assert_allclose(actual, (probs / probs.sum(-1, keepdims=True)) @ v, atol=5e-3, rtol=5e-3)
