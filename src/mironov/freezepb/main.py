
from os import environ
from os.path import isfile, join
from typing import List,Tuple
from time import strftime

from tensorflow.gfile import FastGFile
from tensorflow.summary import FileWriter
from tensorflow import Graph,GraphDef
from tensorflow.python.ops import variables

from tvm.contrib import graph_runtime
from nnvm.frontend import from_tensorflow
from nnvm.compiler import build

import tensorflow as tf
import numpy as np
import nnvm
import tvm

MODEL_PB=join(environ['CWD'], "data/freeze.pb")
MODEL_INPUTS='Rcnn_ctcV3/Inputs:0'
MODEL_OUTPUTS='Rcnn_ctcV3/conv2d_116/BiasAdd:0'

DEF_LOG_DIR='./_logs'

def get_log_dir(tag:str=""):
  return join(DEF_LOG_DIR,((str(tag)+'-') if len(tag)>0 else '')+strftime("%Y%m%d-%H:%M:%S"))

def fropen()->Tuple[Graph,GraphDef]:
  with tf.Session(graph=tf.Graph()) as sess:
    with FastGFile(MODEL_PB, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name="")
      graphdef=sess.graph.as_graph_def(add_shapes=True)
  return sess.graph, graphdef

def totb(g:Graph):
  """ Export to TensorBoard """
  writer=FileWriter(get_log_dir("freezepb"))
  writer.add_graph(g)

def dump(graphdef:GraphDef, suffix:str='restored')->None:
  """ Export to file """
  with open('graphdef%s' %('_'+suffix if len(suffix)>0 else '',)+'.txt', "w") as f:
    f.write(str(graphdef))

def run():
  assert isfile(MODEL_PB)
  g,gd=fropen()
  print(g)
  # totb(g)
  sym,params=nnvm.frontend.from_tensorflow(gd)
  print(sym)


def tf_run()->np.array:
  """ Run the model using tensorflow with zero inputs """
  with tf.Session(graph=tf.Graph()) as sess:
    with FastGFile(MODEL_PB, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name="")
      sess.run(variables.global_variables_initializer())
      g=tf.get_default_graph()
      i=g.get_tensor_by_name(MODEL_INPUTS)
      print("Input node:",type(i), i.name, i.dtype, i)
      o=g.get_tensor_by_name(MODEL_OUTPUTS)
      print("Output node:",type(o), o.name, o.dtype, o)
      i_dict={i: np.zeros(shape=i.shape, dtype=i.dtype.as_numpy_dtype()) }
      o_data=sess.run(o, i_dict)
      return o_data

def tvm_run()->np.array:
  """ Run the model using TVM with zero input """
  g,gd=fropen()
  sym,params=nnvm.frontend.from_tensorflow(gd)
  i=g.get_tensor_by_name(MODEL_INPUTS)
  o=g.get_tensor_by_name(MODEL_OUTPUTS)
  i_shape_dict={MODEL_INPUTS: i.shape.as_list()}
  i_dtype_dict={MODEL_INPUTS: i.dtype.as_numpy_dtype()}
  graph,lib,params=nnvm.compiler.build(graph=sym, target='llvm', shape=i_shape_dict, dtype=i_dtype_dict, params=params)
  m=graph_runtime.create(graph, lib, ctx=tvm.cpu(0))

  i_data=np.zeros(shape=i.shape.as_list(), dtype=i.dtype.as_numpy_dtype())
  m.set_input(MODEL_INPUTS, tvm.nd.array(i_data))
  m.set_input(**params)
  m.run()
  o_data = m.get_output(0, tvm.nd.empty(o.shape.as_list(), o.dtype.name))
  return o_data.asnumpy()




