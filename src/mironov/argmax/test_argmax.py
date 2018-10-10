import math
import numpy as np
import tvm
from tvm.contrib import graph_runtime
import topi
import nnvm.symbol as sym
import nnvm.compiler
from nnvm.testing.config import ctx_list
# from ipdb import set_trace

def test_argmax():
    dshape = (4,4)
    oshape = (4,1)

    dtype = "float32"
    x = sym.Variable("x", shape=dshape, dtype=dtype)
    y = sym.argmax(x+1, axis=0, keepdims=True)

    for target, ctx in ctx_list():
        print('target',target)
        with tvm.build_config(dump_pass_ir=True):
            graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
        m = graph_runtime.create(graph, lib, ctx)
        data = np.random.uniform(size=dshape).astype(dtype)
        print('data', data)
        print('numpy', np.argmax(data+1, axis=0))
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(shape=oshape, dtype='int32'))
        print('out',out)
        return out

