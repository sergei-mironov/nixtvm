import tvm
import topi
import numpy as np
import time

from exercise.runners import run_tvm,with_tvm

def get_shape(t):
  return [tvm.ir_pass.Simplify(s).value for s in t.shape]

def test_reduce():
  n = 10
  m = 10
  A = tvm.placeholder((n, m), name='A')

  k = tvm.reduce_axis((0, m), "k")
  k2 = tvm.reduce_axis((0, m), "k")
  B1 = tvm.compute((n,), lambda i: tvm.sum(A[i,k], axis=[k,k2]), name="B")

  # === B1 = tensor_map (\i -> fold1 (+) 0 (map (\k -> A[i,k]) [k_begin..k_end])) A

  r=run_tvm(0,1, { A:2*np.ones(get_shape(A)).astype(np.float32)}, B1, debug=True)
  print(r.last_data)
