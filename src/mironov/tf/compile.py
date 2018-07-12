"""
References:
 * https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators
 * https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0
"""

import nnvm
import tvm as tvm
import tensorflow as tf

from tf.mnist import Model, load
from tensorflow.python.saved_model.tag_constants import TRAINING,SERVING


def compile(m:Model):
  """ Take trained model, return tvm scheme """
  with tf.Session() as sess:
      graph_def = tf.graph_util.convert_variables_to_constants(
          sess,
          sess.graph.as_graph_def(add_shapes=True),
          ['out'],
          )

def restore(path:str):
  export_dir = path
  with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [SERVING], export_dir)
    print(sess.graph_def)
