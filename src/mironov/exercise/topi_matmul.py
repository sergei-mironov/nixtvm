import os
import numpy as np
import tvm
import topi
from topi.util import get_const_tuple


def with_tvm(lam, *args):
    ctx = tvm.cpu(0)
    pls = [] # placeholders
    vals_nd = [] # initial values
    for i in range(len(args)):
        pls.append(tvm.placeholder(args[i].shape, name='pl'+str(i)))
        vals_nd.append(tvm.nd.array(args[i], ctx))

    out = lam(*pls)
    print(out.shape)
    out_nd = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=out.dtype), ctx)
    s = tvm.create_schedule([out.op])
    m = tvm.build(s, pls + [out], "llvm")
    m(*vals_nd, out_nd)
    return out_nd.asnumpy()

def verify_matmul(sa, sb):
    a = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(sa)).astype(np.float32)
    b = np.random.uniform(low=-1.0, high=1.0, size=get_const_tuple(sb)).astype(np.float32)
    c1 = np.matmul(a, b)
    c2 = with_tvm(lambda A,B: topi.cpp.nn.matmul(A,B), a,b)
    np.testing.assert_allclose(c1, c2, rtol=1e-5)

def test_matmul():
    verify_matmul((1,1),(1,1))
    verify_matmul((2,2),(2,2))
    verify_matmul((2,3),(3,5))
    verify_matmul((5,3),(3,2))
