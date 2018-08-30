import nnvm
import tvm
import numpy as np

from nnvm.compiler.graph_util import gradients
from nnvm.compiler.optimizer import SGD
from nnvm import symbol as sym
from tvm.contrib import graph_runtime
from matplotlib import pyplot as plt
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

def batches(batch_size, x=x_train, y=y_train, repeat=True):
  while True:
    for i in range(int(x.shape[0] / batch_size)):
      yield (x[i:i+batch_size, ...].astype('float32')/255.0, np.eye(10)[y[i:i+batch_size]].astype('float32'))
      if not repeat:
        return

BATCH_SIZE = 32

class Binary:
  def __init__(self):
    pass

def binary_build(b:Binary, graph, target='llvm'):
  b.cgraph, b.libmod, b.params = nnvm.compiler.build(graph, target)
  b.module = graph_runtime.create(b.cgraph, b.libmod, tvm.cpu(0))
  return b

def binary_init0(b:Binary):
  if b.params:
    b.module.set_input(**b.params)

class Model():
  def __init__(self):
    pass

def nn(m:Model):
  v_images=sym.Variable("images", shape=(BATCH_SIZE, 1, 28, 28), dtype=0)
  v_true_labels=sym.Variable("true_labels", shape=(BATCH_SIZE, 10), dtype=0)

  x=v_images
  x=sym.reshape(data=x,shape=(BATCH_SIZE, 28*28))
  x=sym.dense(data=x,units=10)
  logits=x

  x=-sym.elemwise_mul(v_true_labels,sym.log_softmax(x))
  loss=sym.sum(x)/BATCH_SIZE

  # This is not really accuracy, because we use softmax instead of hardmax
  accuracy=sym.sum(v_true_labels*sym.softmax(logits)) / BATCH_SIZE

  # We have to somehow list all weights (the corresponding variables are generated automatically)
  weight_vars=[v for v in loss.list_input_variables() if v.attr('name') not in ['images', 'true_labels']]

  optimizer=SGD(learning_rate=1e-4)
  update_step=optimizer.minimize(loss,var=weight_vars)

  tgraph=nnvm.graph.create(sym.Group([loss,update_step])).apply("InferShape").apply("InferType")
  fgraph=nnvm.graph.create(sym.Group([loss,accuracy])).apply("InferShape").apply("InferType")

  m.tgraph=tgraph
  m.fgraph=fgraph
  m.optimizer=optimizer
  m.loss=loss
  return m

def model_build(m:Model):
  m.tbin=Binary()
  binary_build(m.tbin, m.tgraph, 'llvm')
  m.fbin=Binary()
  binary_build(m.fbin, m.fgraph, 'llvm')
  return m

def model_init(m:Model):
  binary_init0(m.tbin)
  binary_init0(m.fbin)
  tshapes=m.tgraph.json_attr('shape')
  for v in m.loss.list_input_variables():
    shape = tshapes[m.tgraph.index.node_id(v.attr('name'))]
    print("Initializing " + str(v.attr('name')) + " " + str(shape))
    m.tbin.module.set_input(v.attr('name'), np.random.normal(scale=0.1, size=shape).astype('float32'))


def model_train(m:Model):
  seen = 0
  tr_loss = np.inf
  for i, b in enumerate(batches(BATCH_SIZE)):
    if i % 1000 == 0:
      print("step:", i, "seen:", seen, "tr_loss:", tr_loss)
      # if i % 10000 == 0:
      #   l,a=test()
      #   print("test_loss:", l, "test_accuracy:", a)

    # load data
    m.tbin.module.set_input('images', b[0][:, None, ...])
    m.tbin.module.set_input('true_labels', b[1])

    # run a training step
    m.tbin.module.run()
    seen += b[0].shape[0]
    tr_loss = m.tbin.module.get_output(0, tvm.nd.empty((1,))).asnumpy()

