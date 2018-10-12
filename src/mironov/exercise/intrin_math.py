from __future__ import absolute_import, print_function

import tvm
import numpy as np

def lesson1():
  """
  The most straight-forward way to call target specific function is via
  extern function call construct in tvm.
  In th following example, we use :any:`tvm.call_pure_extern` to call
  :code:`__expf` function, which is only available under CUDA.
  """

  n = tvm.var("n")
  A = tvm.placeholder((n,), name='A')
  B = tvm.compute(A.shape,
                  lambda i: tvm.call_pure_extern("float32", "__expf", A[i]),
                  name="B")
  s = tvm.create_schedule(B.op)
  num_thread = 64
  bx, tx = s[B].split(B.op.axis[0], factor=num_thread)
  s[B].bind(bx, tvm.thread_axis("blockIdx.x"))
  s[B].bind(tx, tvm.thread_axis("threadIdx.x"))
  f = tvm.build(s, [A, B], "cuda", name="myexp")
  print(f.imported_modules[0].get_source())




