"""

References: https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators
"""

import nnvm
import tvm
import tensorflow as tf

from tf.mnist import Model, load


def compile(m:Model):
  """ Take trained model, return tvm scheme """
  with tf.Session() as sess:
      graph_def = tf.graph_util.convert_variables_to_constants(
          sess,
          sess.graph.as_graph_def(add_shapes=True),
          ['out'],
          )

