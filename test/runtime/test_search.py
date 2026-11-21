import sqlite3, unittest
from contextlib import closing
from unittest.mock import patch

from tinygrad import Tensor, UOp, helpers
from tinygrad.engine.realize import compile_linear
from tinygrad.helpers import Context
from tinygrad.uop.ops import Ops


class TestSearch(unittest.TestCase):
  def test_beam_cache_precision(self):
    out = Tensor.ones(32, 32, device="CPU").realize() @ Tensor.ones(32, 32, device="CPU").realize()
    linear, _ = out.linear_with_vars()
    with closing(sqlite3.connect(":memory:")) as conn:
      # Preserve compatibility with an existing BEAM cache table, including its columns.
      conn.execute(f"CREATE TABLE beam_search_{helpers.VERSION} (ast blob, amt integer, allow_test_size integer, device text, suffix text, "
                   "val blob, PRIMARY KEY (ast, amt, allow_test_size, device, suffix))")
      with patch("tinygrad.helpers.db_connection", return_value=conn), patch("tinygrad.helpers._db_tables", set()):
        for allow_tf32 in (0, 1):
          with Context(BEAM=1, ALLOW_TF32=allow_tf32, IGNORE_BEAM_CACHE=0, CACHELEVEL=1): compile_linear(linear, beam=1)
      # A precision change must search its own legal options instead of reusing the previous schedule.
      self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM beam_search_{helpers.VERSION}").fetchone()[0], 2)

  def test_beam_symbolic_kernel(self):
    size = UOp.variable("size", 1, 8)
    out = Tensor.empty(8, 8, device="CPU")[:size.bind(4)] + 1
    linear, _ = out.linear_with_vars()
    with Context(BEAM=1, IGNORE_BEAM_CACHE=1, CACHELEVEL=0): compiled = compile_linear(linear, beam=1)
    program = next(u for u in compiled.toposort(enter_calls=True) if u.op is Ops.PROGRAM)
    self.assertNotEqual(program.src[0].arg.applied_opts, ())


if __name__ == "__main__":
  unittest.main()
