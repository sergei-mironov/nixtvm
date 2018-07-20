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
    dshape = (2,)
    oshape = (1,)

    dtype = "float32"
    x = sym.Variable("x", shape=dshape, dtype=dtype)
    # y = sym.min(x)
    # y = sym.argmax(x, axis=[0], exclude=True)
    y = sym.argmax(x+1)

    # graph = nnvm.graph.create(y)
    # graph = graph.apply("InferShape")
    # print(graph.ir(join_node_attrs=['shape']))

    for target, ctx in ctx_list():
        print('target',target)
        with tvm.build_config(dump_pass_ir=True):
            graph, lib, _ = nnvm.compiler.build(y, target, {"x": dshape})
            # print(graph.ir(join_node_attrs=['shape']))
        m = graph_runtime.create(graph, lib, ctx)
        # set input
        # data = np.random.uniform(size=dshape).astype(dtype)
        # data = np.full(shape=dshape, fill_value=33, dtype=dtype)
        data = np.array([33,42], dtype=dtype)
        print('data', data)
        print('numpy', np.argmax(data+1))
        m.run(x=data)
        out = m.get_output(0, tvm.nd.empty(oshape))
        print('out',out)
        return out

