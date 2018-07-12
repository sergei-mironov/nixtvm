"""
This example is aimed at studying Model saving/loading issues.

References:

  * MNIST dataset
    http://yann.lecun.com/exdb/mnist/

  * Original source location
    https://github.com/aymericdamien/TensorFlow-Examples/

  * TensorBoard tutorial
    https://www.tensorflow.org/guide/summaries_and_tensorboard

"""

import tensorflow as tf

from time import strftime


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


# Parameters
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
  # TF Estimator input is a dict, in case of multiple inputs
  x = features['images']
  # Hidden fully connected layer with 256 neurons
  layer_1 = tf.layers.dense(x, n_hidden_1)
  # Hidden fully connected layer with 256 neurons
  layer_2 = tf.layers.dense(layer_1, n_hidden_2)
  # Output fully connected layer with a neuron for each class
  logits = tf.layers.dense(layer_2, num_classes)

  # Predictions
  pred_classes = tf.argmax(logits, axis=1, name='pred_classes')
  pred_probas = tf.nn.softmax(logits)

  # If prediction mode, early return
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        export_outputs={"out":PredictOutput(pred_classes)} )

  # Define loss and optimizer
  loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(loss_op,
                                global_step=tf.train.get_global_step())

  # Evaluate the accuracy of the model
  acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

  # TF Estimators requires to return a EstimatorSpec, that specify
  # the different ops for training, evaluating, ...
  return tf.estimator.EstimatorSpec(
      mode=mode
    , predictions=pred_classes
    , loss=loss_op
    , train_op=train_op
    , eval_metric_ops={'accuracy': acc_op}
    )

class Model:
  def __init__(self):
    self.log_dir='./_logs'
    pass
  def get_log_dir(self, tag:str=""):
    return self.log_dir+"/"+ (str(tag)+'-' if len(tag)>0 else '') +strftime("%c")

class GraphDefExport(SessionRunHook):
  def __init__(self,m:Model):
    self.m=m
  def end(self,sess):
    self.m.graph_def=tf.graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(add_shapes=True), ['pred_classes'])
    print("Freezing complete")

def load(m:Model=None)->Model:
  """ Load the model and set up graphdef export hook """
  if m is None:
    m=Model()
  # Build the Estimator
  m.mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
  def _hooked_model_fn(features, labels, mode):
    spec = model_fn(features, labels, mode)._replace(
        training_hooks=[
            GraphDefExport(m)
          , SummarySaverHook(50, output_dir=m.get_log_dir('mnist'), scaffold=Scaffold())
          ]
      )
    return spec
  m.estimator = tf.estimator.Estimator(_hooked_model_fn)
  return m

def train(m:Model):
  # Define the input function for training
  mnist=m.mnist
  input_fn=tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
  # Train the Model
  m.estimator.train(input_fn, steps=num_steps)
  # Export the GD
  assert m.graph_def is not None, "GraphDefExport hook not working"
  with open('graph_def.txt', "w") as f:
    f.write(str(m.graph_def))

def eval(m:Model):
  # Evaluate the Model
  # Define the input function for evaluating
  mnist=m.mnist
  input_fn=tf.estimator.inputs.numpy_input_fn(
      x={'images': mnist.test.images}, y=mnist.test.labels,
      batch_size=batch_size, shuffle=False)
  # Use the Estimator 'evaluate' method
  e = m.estimator.evaluate(input_fn)
  print("Testing Accuracy:", e['accuracy'])

def save(m:Model):
  fspec={}
  fspec['images'] = tf.placeholder(tf.float32, shape=[1,784], name='images')
  feat=tf.estimator.export.build_raw_serving_input_receiver_fn(fspec)
  m.estimator.export_savedmodel('data', feat, strip_default_attrs=True)

def restore(m:Model, path:str, freezed:bool=True, dump:bool=False)->GraphDef:
  export_dir=path
  with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [SERVING], export_dir)
    if dump:
      with open('graphdef_restored.txt', "w") as f:
        f.write(str(sess.graph_def))
    graphdef=sess.graph.as_graph_def(add_shapes=True)
    if freezed:
      return tf.graph_util.convert_variables_to_constants(sess,graphdef,['pred_classes'])
    else:
      return graphdef

def log(m:Model, graphdef:GraphDef)->None:
  """ FIXME: unhardcode `pred_classes` """
  with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graphdef,name='')
    writer=tf.summary.FileWriter(m.get_log_dir('freezed'))
    writer.add_graph(sess.graph)


def view(m, path:str, freeze_constants:bool=True):
  """ Restore saved model and show it in the Tensorboard """

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
          ['import/pred_classes'])
        tf.import_graph_def(graph_def)

    train_writer=tf.summary.FileWriter(m.get_log_dir('view'))
    train_writer.add_graph(sess.graph)


