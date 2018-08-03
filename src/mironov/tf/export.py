"""
References:
 * https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators
 * https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0
"""

import nnvm as nnvm
import tvm as tvm
import tensorflow as tf

from typing import List,Tuple
from time import strftime
from os.path import split
from os import environ

from tensorflow import GraphDef
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from tensorflow.python.platform import gfile
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.monitored_session import Scaffold
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.basic_session_run_hooks import SummarySaverHook
from tensorflow.python.estimator.export.export_output import ExportOutput,PredictOutput
from tensorflow.python.saved_model.tag_constants import TRAINING,SERVING


DEF_LOG_DIR='./_logs'

def get_log_dir(tag:str=""):
  return DEF_LOG_DIR+"/"+((str(tag)+'-') if len(tag)>0 else '')+strftime("%Y%m%d-%H:%M:%S")

def dump(graphdef:GraphDef, suffix:str='restored')->None:
  with open('graphdef%s' %('_'+suffix if len(suffix)>0 else '',)+'.txt', "w") as f:
    f.write(str(graphdef))

def restore(path:str, output_node_names:List[str], frozen:bool=True)->GraphDef:
  """ Restore the model, optionally freeze variables """
  export_dir=path
  with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [SERVING], export_dir)
    graphdef=sess.graph.as_graph_def(add_shapes=True)
    dump(graphdef,'restored')
    if frozen:
      graphdef_f=tf.graph_util.convert_variables_to_constants(sess,graphdef,output_node_names)
      dump(graphdef_f,'frozen')
      return graphdef_f
    else:
      return graphdef

def view(export_dir:str, output_node_names:List[str], frozen:bool):
  """ Restore saved model and show it in the Tensorboard """
  writer=tf.summary.FileWriter(get_log_dir(split(export_dir)[-1]+('-f' if frozen else '')))
  g=restore(export_dir, output_node_names, frozen)
  writer.add_graph(g)

def export(path:str, output_node_names, **kwargs):
  graphdef=restore(path,output_node_names,frozen=True,**kwargs)
  dump(graphdef)
  sym,params=nnvm.frontend.from_tensorflow(graphdef)
  return sym,params
  # print(sym)
  # print(params)

PATH_FC=environ['CWD']+'/data/saved_fully_connected_network/1533038462'
def export_fully_connected():
  export(PATH_FC, output_node_names=['pred_classes'])
def view_fully_connected():
  view(PATH_FC, output_node_names=['pred_classes'], frozen=False)
  view(PATH_FC, output_node_names=['pred_classes'], frozen=True)

PATH_CONV=environ['CWD']+'/data/saved_convolutional_network/1533295789'
def export_convolutional():
  export(PATH_CONV, output_node_names=['pred_classes'])
def restore_convolutional():
  pass

PATH_AUTOENC=environ['CWD']+'/data/saved_autoencoder'
def export_autoencoder():
  export(PATH_AUTOENC, output_node_names=['decoder_op'])

PATH_RNN=environ['CWD']+'/data/saved_recurrent_network'
def export_recurrent_network():
  export(PATH_RNN, ['prediction'])
def view_recurrent_network():
  view(PATH_RNN, ['prediction'], frozen=False)
  view(PATH_RNN, ['prediction'], frozen=True)

PATH_BIRNN=environ['CWD']+'/data/saved_bidirectional_rnn'
def export_bidirectional_rnn():
  export(PATH_BIRNN, ['prediction'])
def view_bidirectional_rnn():
  view(PATH_BIRNN, ['prediction'], frozen=False)
  view(PATH_BIRNN, ['prediction'], frozen=True)

PATH_DYRNN=environ['CWD']+'/data/saved_dynamic_rnn'
def export_dynamic_rnn():
  export(PATH_DYRNN, ['prediction'])
def view_dynamic_rnn():
  view(PATH_DYRNN, ['prediction'], frozen=False)
  view(PATH_DYRNN, ['prediction'], frozen=True)

PATH_VARENC=environ['CWD']+'/data/saved_variational_autoencoder'
def export_variational_autoencoder():
  export(PATH_VARENC, ['decoder'])
def view_variational_autoencoder():
  view(PATH_VARENC, ['decoder'], frozen=False)
  view(PATH_VARENC, ['decoder'], frozen=True)

