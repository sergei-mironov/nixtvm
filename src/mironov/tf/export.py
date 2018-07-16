"""
References:
 * https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators
 * https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0
"""

import nnvm as nnvm
import tvm as tvm
import tensorflow as tf

from typing import List,Tuple

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


def _get_log_dir(self, tag:str=""):
  return DEF_LOG_DIR+"/"+ (str(tag)+'-' if len(tag)>0 else '') +strftime("%c")


def dump(graphdef:GraphDef, suffix:str='restored')->None:
  with open('graphdef%s' %('_'+suffix if len(suffix)>0 else '',)+'.txt', "w") as f:
    f.write(str(graphdef))

def restore(path:str, output_node_names:List[str], frozen:bool=True)->GraphDef:
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

def log(graphdef:GraphDef)->None:
  """ FIXME: unhardcode `pred_classes` """
  with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graphdef,name='')
    writer=tf.summary.FileWriter(_get_log_dir('freezed'))
    writer.add_graph(sess.graph)


def view(path:str, output_node_names:List[str], freeze_constants:bool=True):
  """ Restore saved model and show it in the Tensorboard

  FIXME: output_node_names was ['import/pred_classes']
  """

  with tf.Session(graph=tf.Graph()) as sess:
    model_filename = path + '/saved_model.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
      data=compat.as_bytes(f.read())
      sm=saved_model_pb2.SavedModel()
      sm.ParseFromString(data)
      assert 1==len(sm.meta_graphs), 'More than one graph found. Not sure which to write'
      tf.import_graph_def(sm.meta_graphs[0].graph_def)

      if freeze_constants:
        graph_def=tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(add_shapes=True),
          output_node_names)
        tf.import_graph_def(graph_def)

    train_writer=tf.summary.FileWriter(_get_log_dir('view'))
    train_writer.add_graph(sess.graph)


def export(path:str, **kwargs):
  graphdef=restore(path,frozen=True,**kwargs)
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

def export_rnn():
  export('data/rnn', output_node_names=['prediction'])
