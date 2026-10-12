import unittest
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import CStyleLanguage, CUDARenderer
from tinygrad.renderer.wgsl import WGSLRenderer
from test.helpers import check_schedule
from tinygrad.uop.ops import UOp, Ops, KernelInfo, python_alu
from tinygrad.tensor import Tensor

@unittest.skipIf(not isinstance(Device[Device.DEFAULT].renderer, CStyleLanguage), "uops are for cstyle")
class TestCStyleFailures(unittest.TestCase):
  def _test_src_strip_paren(self, op: Ops, should_strip_paren:bool=True):
    dtype = "bool" if op in (Ops.OR, Ops.XOR, Ops.AND) else None
    ret = Tensor.empty(1, dtype=dtype)
    for _ in range(5): ret = python_alu[op](ret, Tensor.empty(1, dtype=dtype))
    linear, _ = check_schedule(ret, 1)
    src = to_program(linear.src[0].src[0], Device[Device.DEFAULT].renderer).src[2].arg
    self.assertEqual("("*5 not in src, should_strip_paren)

  def test_repeat_add(self): self._test_src_strip_paren(Ops.ADD)

  def test_repeat_mul(self): self._test_src_strip_paren(Ops.MUL)

  def test_repeat_xor(self): self._test_src_strip_paren(Ops.XOR)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, WGSLRenderer), "wgsl ends up with '(' * 5")
  def test_repeat_or(self): self._test_src_strip_paren(Ops.OR)

  @unittest.skipIf(isinstance(Device[Device.DEFAULT].renderer, WGSLRenderer), "wgsl ends up with '(' * 5")
  def test_repeat_and(self): self._test_src_strip_paren(Ops.AND)

  def test_repeat_sub(self): self._test_src_strip_paren(Ops.SUB, should_strip_paren=False)

class TestWGSLFailures(unittest.TestCase):
  def test_folded_packed_store(self):
    b = UOp.param(0, dtypes.char, 4)
    idx = b.index(UOp.const(0).cast(dtypes.int))
    store = UOp.store(idx, idx.cast(dtypes.uint32).load() & UOp.const(0xffffff00).cast(dtypes.uint32))
    src = WGSLRenderer(Target("WEBGPU")).render(UOp.sink(store, arg=KernelInfo()).toposort())
    self.assertIn("atomicAnd(&data0_4[0],4294967040u);", src)
    self.assertNotIn("atomicAdd", src)

class TestCUDAFailures(unittest.TestCase):
  def _render(self, kernel) -> str:
    from tinygrad.uop.ops import AxisType, AddrSpace
    out, inp = UOp.param(0, dtypes.uint32, 256), UOp.param(1, dtypes.uint32, 256)
    lane, buf = UOp.range(32, 0, AxisType.WARP), UOp.alloc((256,), dtypes.uint32, addrspace=AddrSpace.LOCAL)
    return to_program(kernel(out, inp, lane, buf), CUDARenderer(Target("CUDA", arch="sm_89"))).src[2].arg

  def test_min_blocks_launch_bounds(self):
    # KernelInfo.min_blocks becomes __launch_bounds__'s minimum blocks per SM, so the compiler caps registers for that occupancy
    def kernel(out, inp, lane, buf):
      return out[lane].store(inp[lane]).end(lane).sink(arg=KernelInfo(name="lb", opts_to_apply=(), min_blocks=2))
    self.assertIn("__launch_bounds__(32, 2)", self._render(kernel))

  def test_ldmatrix_load(self):
    # a LOCAL load flagged ldmatrix_x4 renders as the ldmatrix helper (with a memory clobber, so it stays between its barriers)
    def kernel(out, inp, lane, buf):
      buf = buf.after(UOp.group(*[buf[lane*8+i].store(inp[lane*8+i]) for i in range(8)]).barrier())
      frag = UOp(Ops.SHRINK, src=(buf, lane*8, UOp.const(4))).load(arg="ldmatrix_x4")
      return UOp.group(*[out[lane*4+i].store(frag[i]) for i in range(4)]).end(lane).sink(arg=KernelInfo(name="ldm", opts_to_apply=()))
    src = self._render(kernel)
    self.assertIn("tg_ldmatrix_x4(", src)
    self.assertIn('ldmatrix.sync.aligned.m8n8.x4.shared.b16', src)
    self.assertIn('"memory"', src)

  def test_ldmatrix_trans_load(self):
    # ldmatrix_x4_trans loads each 8x8 matrix transposed (an MMA B fragment from a row-major tile)
    def kernel(out, inp, lane, buf):
      buf = buf.after(UOp.group(*[buf[lane*8+i].store(inp[lane*8+i]) for i in range(8)]).barrier())
      frag = UOp(Ops.SHRINK, src=(buf, lane*8, UOp.const(4))).load(arg="ldmatrix_x4_trans")
      return UOp.group(*[out[lane*4+i].store(frag[i]) for i in range(4)]).end(lane).sink(arg=KernelInfo(name="ldm", opts_to_apply=()))
    src = self._render(kernel)
    self.assertIn("tg_ldmatrix_x4_trans(", src)
    self.assertIn('ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16', src)

if __name__ == '__main__':
  unittest.main()
