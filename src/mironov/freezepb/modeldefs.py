import tensorflow as tf
import numpy as np
import nnvm
import tvm

from tensorflow import Tensor as TF_Tensor
from tensorflow.gfile import FastGFile
from tensorflow.summary import FileWriter
from tensorflow import Graph as TF_Graph, GraphDef as TF_GraphDef
from tensorflow.python.ops import variables

from os import environ
from os.path import isfile, join
from typing import Dict,Any,List,Tuple

from nnvm.frontend import from_tensorflow

# Type of nnvm params
Params = Dict[str,Any]

def fropen()->Tuple[TF_Graph,TF_GraphDef]:
  with tf.Session(graph=tf.Graph()) as sess:
    with FastGFile(MODEL_PB, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name="")
      graphdef=sess.graph.as_graph_def(add_shapes=True)
  return sess.graph, graphdef

def common_init(init_method, shape, dtype):
  if init_method=='zeros':
    return np.zeros(shape=shape, dtype=dtype)
  elif init_method=='std':
    return np.random.uniform(low=-50, high=51, size=shape).astype(dtype=dtype)
  else:
    raise ValueError("invalid 'init' argument")

MODEL_PB=join(environ['CWD'], "data/freeze.pb")
MODEL_INPUT='Rcnn_ctcV3/Inputs'

MODEL_OUTPUTS=[
   'Rcnn_ctcV3/expand_conv1/add_1/add'
  ,'Rcnn_ctcV3/expand_conv2/add_7/add'
  ,'Rcnn_ctcV3/expand_conv3/add_13/add'
  ,'Rcnn_ctcV3/expand_conv4/add_17/add'
  ,'Rcnn_ctcV3/conv2d_116/BiasAdd']
MODEL_OUTPUT=MODEL_OUTPUTS[-1]

DEF_LOG_DIR='./_logs'

MODEL_PARAMS=from_tensorflow(fropen()[1])[1]

