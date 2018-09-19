import tvm
import nnvm
import numpy as np

from nnvm import symbol as sym
from tvm.contrib import graph_runtime

class ToyRNN:
  def __init__(self,b,c,W,V,U,x,h):
    """
    Parameters:
    ----------
    b,c   : biases
    U,V,W : weights
    x,h   : input and state vectors
    """

    a=b+W*h+U*x
    h2=sym.tanh(a)
    o=c+V*h2
    y=sym.softmax(o)

    self.a=a
    self.h2=h2
    self.o=o
    self.y=y


def toylstm():
  """ FIXME: Just a copy from tensorflow.py. Can't really be interpreted """
  in_data = inputs[0]
  in_weight = inputs[3]
  in_bias = inputs[7]

  forget_bias = attr.pop('forget_bias')
  input_shape = attr['_input_shapes'][inputs[0]]
  weight_shape = attr['_input_shapes'][inputs[3]]

  batch_size, input_size = input_shape[0][0], input_shape[0][1]
  num_hidden_layers = weight_shape[0][1]
  num_hidden = num_hidden_layers // 4

  in_data = _sym.reshape(in_data, shape=(batch_size, input_size))
  ixh = _sym.concatenate(*[in_data, in_state_h], axis=1)

  in_weight = _sym.transpose(in_weight)

  gates = _sym.dense(ixh, in_weight, in_bias, use_bias=True,
                     units=num_hidden_layers, name="dense")

  gate_list = _sym.split(gates, indices_or_sections=4, axis=1)

  in_gate      = _sym.sigmoid(gate_list[0])
  in_transform = _sym.tanh(gate_list[1])
  forget_gate  = _sym.sigmoid(gate_list[2])
  forget_gate  = forget_gate + forget_bias

  out_gate = _sym.sigmoid(gate_list[3])

  next_c = _sym.broadcast_add(_sym.broadcast_mul(forget_gate, in_state_c),
                              _sym.broadcast_mul(in_gate, in_transform))

  next_h = out_gate * _sym.tanh(next_c)

  out_state = _sym.concatenate(*[next_c, next_h])
  out_state = _sym.reshape(out_state, shape=(2, batch_size, num_hidden))

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

    graph_module = graph_runtime.create(tvm_graph, lib, tvm.cpu(0))
    print('Got graph module')

    for name,value in data.items():
        graph_module.set_input(name, value)

    graph_module.run()

    out_value = graph_module.get_output(0, tvm.nd.empty((out_shape), 'float32'))

    # print('Inputs:\nX:', data["x"], "\nY:", data["y"])
    print('Output value:', type(out_value), '\nShape:', out_value.shape, '\nO:', out_value)


