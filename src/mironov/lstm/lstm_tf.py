"""
Poor man's LSTM cell applied to MNIST. Use `train` function to train the model.
"""

import tensorflow as tf
import numpy as np
import argparse

from typing import Any,List,Dict
from keras.datasets import mnist
from tensorflow import Tensor as TF_Tensor
from tensorflow.python.ops import variables

def lstm_gate(op, U,V,b, x,h):
  """
  op - nonlinearity operation
  x,h - input tensor of shape (1,a)
  W,U - matrices of shape (a,b)
  b - bias (1,b)

  return tensor of shape (1,b)
  """
  return op(tf.matmul(x,U) + tf.matmul(h,V) + b)

def lstm_cell(Ug,Vg,bg, Ui,Vi,bi, Uf,Vf,bf, Uo,Vo,bo):
  """ LSTM cell. Ideomatic TF code would define all the variable here """
  def call(xt,st,ht):
    g = lstm_gate(tf.tanh,    Ug,Vg,bg, xt,ht)
    i = lstm_gate(tf.sigmoid, Ui,Vi,bi, xt,ht)
    f = lstm_gate(tf.sigmoid, Uf,Vf,bf, xt,ht)
    o = lstm_gate(tf.sigmoid, Uo,Vo,bo, xt,ht)

    st2 = st*f + g*i
    ht2 = tf.tanh(st2)*o
    return (st2,ht2)
  return call

def lstm_layer(cell, xs:List[TF_Tensor], s0, h0)->List[TF_Tensor]:
  h = h0
  s = s0
  hs = []
  for i in range(len(xs)):
    s,h = cell(xs[i],s,h)
    hs.append(h)
  return hs

def model(batch_size:int, num_timesteps:int, num_inputs:int, num_units:int, init=tf.random_normal, bias_init=tf.zeros):
  """
  Create a single cell and replicate it `num_timesteps` times for training.
  Return X,[(batch_size,num_classes) x num_timesteps]
  """
  X = tf.placeholder(tf.float32, shape=(batch_size, num_timesteps, num_inputs))

  Ug = tf.Variable(init([num_inputs, num_units]))
  Vg = tf.Variable(init([num_units, num_units]))
  bg = tf.Variable(bias_init([1, num_units]))

  Ui = tf.Variable(init([num_inputs, num_units]))
  Vi = tf.Variable(init([num_units, num_units]))
  bi = tf.Variable(bias_init([1, num_units]))

  Uf = tf.Variable(init([num_inputs, num_units]))
  Vf = tf.Variable(init([num_units, num_units]))
  bf = tf.Variable(bias_init([1, num_units]) + tf.constant(1.0))

  Uo = tf.Variable(init([num_inputs, num_units]))
  Vo = tf.Variable(init([num_units, num_units]))
  bo = tf.Variable(bias_init([1, num_units]))

  cell = lstm_cell(Ug,Vg,bg, Ui,Vi,bi, Uf,Vf,bf, Uo,Vo,bo)

  # TODO: Not clear: should the initial state be a trainable parameter or a
  # constant?
  s0 = tf.constant(np.zeros(shape=[1, num_units], dtype=np.float32))
  # TODO: Not clear: shoule the initial h be a trainable parameter or a
  # constant?
  h0 = tf.constant(np.zeros(shape=[1, num_units], dtype=np.float32))
  xs = tf.unstack(X, num_timesteps, 1)

  outputs = lstm_layer(cell, xs, s0, h0)
  return X,outputs

def model2(batch_size:int, num_timesteps:int, num_inputs:int, num_classes:int, num_hidden:int, init=tf.random_normal):
  """
  Use `model` with 128 "classes", but translate them back to 10 classes via
  dense layer.
  """
  W=tf.Variable(init([num_hidden, num_classes]))
  b=tf.Variable(init([1, num_classes]))

  X,outputs=model(batch_size, num_timesteps, num_inputs, num_units=num_hidden, init=init)
  cls=tf.squeeze(tf.matmul(outputs[-1],W)+b)
  return X,cls

def mnist_load():
  """ Load MNIST and convert its ys to one-hot encoding """
  (Xl,yl),(Xt,yt)=mnist.load_data()
  def oh(y):
    yoh=np.zeros((y.shape[0],10),dtype=np.float32)
    yoh[np.arange(y.shape[0]),y]=1
    return yoh
  Xl = Xl.astype(np.float32) / 255.0
  Xt = Xt.astype(np.float32) / 255.0
  return (Xl,oh(yl)),(Xt,oh(yt))

(Xl,yl),(Xt,yt)=mnist_load()

def train():
  """ Main train """
  batch_size=128
  num_inputs=28
  num_channels=28
  num_hidden=128
  num_classes=10
  training_steps=10000
  learning_rate=0.001
  num_examples = Xl.shape[0]
  with tf.Session(graph=tf.Graph()) as sess:
    X,logits=model2(batch_size,num_inputs,num_channels,num_classes=num_classes,num_hidden=num_hidden,init=tf.random_normal)
    y = tf.placeholder(tf.float32, shape=(batch_size, num_classes))

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    prediction = tf.nn.softmax(logits, name='prediction')
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    sess.run(variables.global_variables_initializer())

    epoch = -1
    batch_start = 0
    batch_end = batch_size

    def next_batch():
      nonlocal epoch, batch_start, batch_end
      global Xl, yl
      if batch_end > num_examples or epoch == -1:
        epoch += 1
        batch_start = 0
        batch_end = batch_size
        perm0 = np.arange(num_examples)
        np.random.shuffle(perm0)
        Xl = Xl[perm0]
        yl = yl[perm0]
      Xi_ = Xl[batch_start:batch_end, :, :]
      yi_ = yl[batch_start:batch_end, :]
      batch_start = batch_end
      batch_end = batch_start + batch_size
      return {X: Xi_, y: yi_}

    for step in range(training_steps):
      batch = next_batch()
      sess.run(train_op, feed_dict=batch)

      if step % 100 == 0:
        loss_, acc_ = sess.run((loss_op, accuracy_op), feed_dict=batch)
        print("epoch", epoch, "step", step, "loss", "{:.4f}".format(loss_), "acc",
              "{:.2f}".format(acc_))

def go2():
  """ Test facility 2 """
  batch_size=3
  num_inputs=2
  num_channels=28
  num_hidden=128
  num_classes=10
  with tf.Session(graph=tf.Graph()) as sess:
    X,os=model2(batch_size,num_inputs,num_channels,num_classes,num_hidden,init=tf.ones)
    sess.run(variables.global_variables_initializer())

    Xz_=np.zeros(shape=(batch_size,num_inputs,num_channels))
    yz_=np.zeros(shape=(batch_size,num_classes))
    o_=sess.run(os, feed_dict={X:Xz_})
    return o_

def go1():
  """ Test facility 1 """
  batch_size=3
  num_inputs=2
  num_channels=28
  num_classes=10
  with tf.Session(graph=tf.Graph()) as sess:
    X,os=model(batch_size,num_inputs,num_channels,init=tf.ones)
    sess.run(variables.global_variables_initializer())

    Xz_=np.zeros(shape=(batch_size,num_inputs,num_channels))
    o_=sess.run(os, feed_dict={X:Xz_})
    return o_[-1]

if __name__ == '__main__':
  train()