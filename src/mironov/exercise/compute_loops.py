"""
Loops problem illustration for the
http://code.huawei.com/mrc-cbg-opensource/hitvm-internal/issues/4
issue.
"""
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

  r=run_tvm(0,1,
      { A:1*np.ones(get_shape(A)).astype(np.float32)
      , B:2*np.ones(get_shape(B)).astype(np.float32)
      },
      C,
      debug=True)

  print(r.last_data)
