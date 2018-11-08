import tvm
import topi
import numpy as np
import time

from exercise.runners import run_tvm,with_tvm

def get_shape(t):
  return [tvm.ir_pass.Simplify(s).value for s in t.shape]

def test_compute():
  A = tvm.placeholder((10,), name='A')
  B = tvm.placeholder((30,), name='B')
  C = tvm.compute((30,), lambda i: A[i/3]+B[i], name="C")

  def scheduling(s):
    s[C].split(C.op.axis[0], nparts=5)

  r=run_tvm(0,1,
      { A:1*np.ones(get_shape(A)).astype(np.float32)
      , B:2*np.ones(get_shape(B)).astype(np.float32)
      },
      C,
      scheduling=scheduling,
      debug=True)

  print(r.last_data)
