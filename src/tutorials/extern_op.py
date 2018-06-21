"""
External Tensor Functions
=========================
**Author**: `Tianqi Chen <https://tqchen.github.io>`_

While TVM supports transparent code generation, sometimes
it is also helpful to incorporate manual written code into
the pipeline. For example, we might want to use cuDNN for
some of the convolution kernels and define the rest of the stages.

TVM supports these black box function calls natively.
Specfically, tvm support all the tensor functions that are DLPack compatible.
Which means we can call any function with POD types(pointer, int, float)
or pointer to DLTensor as argument.
"""
from __future__ import absolute_import, print_function

import tvm
import numpy as np
from tvm.contrib import cblas

def lesson1():
  ######################################################################
  # Use Extern Tensor Function
  # --------------------------
  # In the example below, we use :any:`tvm.extern` to add an extern
  # array function call. In the extern call, we declare the shape
  # of output tensors. In the second argument we provide the list of inputs.
  #
  # User will need to provide a function describing how to compute the result.
  # The compute function takes list of symbolic placeholder for the inputs,
  # list of symbolic placeholder for the outputs and returns the executing statement.
  #
  # In this case we simply call a registered tvm function, which invokes a CBLAS call.
  # TVM does not control internal of the extern array function and treats it as blackbox.
  # We can further mix schedulable TVM calls that add a bias term to the result.
  #
  n = 1024
  l = 128
  m = 235
  bias = tvm.var('bias', dtype=tvm.float32)
  A = tvm.placeholder((n, l), name='A')
  B = tvm.placeholder((l, m), name='B')
  C = tvm.extern((n, m), [A, B],
                 lambda ins, outs: tvm.call_packed(
                     "tvm.contrib.cblas.matmul",
                     ins[0], ins[1], outs[0], False, False), name="C")
  D = tvm.compute(C.shape, lambda i, j: C[i,j] + bias, name="D")
  s = tvm.create_schedule(D.op)
  ######################################################################
  # Verify the Result
  # -----------------
  # We can verify that the result matches what we expected.
  #
  ctx = tvm.cpu(0)
  f = tvm.build(s, [A, B, D, bias], "llvm")
  a = tvm.nd.array(np.random.uniform(size=(n, l)).astype(A.dtype), ctx)
  b = tvm.nd.array(np.random.uniform(size=(l, m)).astype(B.dtype), ctx)
  d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), ctx)
  bb = 10.0
  f(a, b, d, bb)
  np.testing.assert_allclose(
      d.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()) + 10, rtol=1e-5)

