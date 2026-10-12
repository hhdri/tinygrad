import gc, itertools, unittest, weakref
import numpy as np
from tinygrad import Tensor, UOp, dtypes, function, Device
from tinygrad.llm.kernels.amd import Linear, amd_custom_kernels_supported, QUANT_SIZES, HALFWORD_QUANTS, iq4_half_lut, _iq_grid
from tinygrad.llm.kernels.nv import nv_custom_kernels_supported
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
            np.testing.assert_allclose(linear(Tensor(x)).numpy(), x.astype(np.float32) @ weight.astype(np.float32).T + bias,
                                       rtol=2e-4, atol=2e-4)
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
