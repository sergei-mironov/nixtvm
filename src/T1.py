from __future__ import absolute_import, print_function

import tvm
import numpy as np

# Global declarations of environment.

tgt_host="llvm"
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="cuda"

n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))


s = tvm.create_schedule(C.op)


bx, tx = s[C].split(C.op.axis[0], factor=64)


if tgt == "cuda":
  s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
  s[C].bind(tx, tvm.thread_axis("threadIdx.x"))


fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")



