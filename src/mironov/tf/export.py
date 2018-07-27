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
  print(sym)
  print(params)

def export_fully_connected():
  export('data/1531399643', output_node_names=['pred_classes'])

def export_convolutional():
  export('data/1531490880', output_node_names=['pred_classes'])

def export_autoencoder():
  export('data/autoencoder', output_node_names=['decoder_op'])

def export_recurrent_network():
  export('data/saved_recurrent_network', ['prediction'])
def view_recurrent_network():
  view('data/saved_recurrent_network', ['prediction'], frozen=False)
  view('data/saved_recurrent_network', ['prediction'], frozen=True)

def export_bidirectional_rnn():
  export('data/saved_bidirectional_rnn', ['prediction'])
def view_bidirectional_rnn():
  view('data/saved_bidirectional_rnn', ['prediction'], frozen=False)
  view('data/saved_bidirectional_rnn', ['prediction'], frozen=True)

def export_dynamic_rnn():
  export('data/saved_dynamic_rnn', ['prediction'])
def view_dynamic_rnn():
  view('data/saved_dynamic_rnn', ['prediction'], frozen=False)
  view('data/saved_dynamic_rnn', ['prediction'], frozen=True)

def export_variational_autoencoder():
  export('data/saved_variational_autoencoder', ['decoder'])
def view_variational_autoencoder():
  view('data/saved_variational_autoencoder', ['decoder'], frozen=False)
  view('data/saved_variational_autoencoder', ['decoder'], frozen=True)
