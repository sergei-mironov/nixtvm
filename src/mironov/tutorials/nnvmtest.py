import tvm
import nnvm
import numpy as np

from nnvm import symbol as sym
from tvm.contrib import graph_runtime

def test1():

    in_shape = [3,3,3]
    out_shape = [3,3,3,2]
    data = {
        "x" : np.arange(np.prod(in_shape), dtype=np.float32).reshape(in_shape),
        "y" : np.arange(np.prod(in_shape), dtype=np.float32).reshape(in_shape)
        }

    axis = -4
    x = sym.Variable("x")
    y = sym.Variable("y")

    x = sym.expand_dims(x, axis=axis, num_newaxis=1) # sym.elemwise_add(x,y)
    y = sym.expand_dims(y, axis=axis, num_newaxis=1) # sym.elemwise_add(x,y)
    z = sym.concatenate(x,y, axis=-4)

    nnvm_graph = nnvm.graph.create(z)
    print('Got NNVM graph')
    print(nnvm_graph.json())

    in_shapes_dict = {n:list(np.shape(v)) for n,v in data.items()}
    tvm_graph,lib,params = nnvm.compiler.build(nnvm_graph, 'llvm', in_shapes_dict)
    print('Got TVM graph')

    ctx = tvm.cpu(0)
    graph_module = graph_runtime.create(tvm_graph, lib, ctx)
    print('Got graph module')

    print(tvm_graph.__dir__())
    print(tvm_graph.json())

    for name,value in data.items():
        graph_module.set_input(name, value)

    graph_module.run()

    out_value = graph_module.get_output(0, tvm.nd.empty((out_shape), 'float32'))

    # print('Inputs:\nX:', data["x"], "\nY:", data["y"])
    print('Output value:', type(out_value), '\nShape:', out_value.shape, '\nO:', out_value)


