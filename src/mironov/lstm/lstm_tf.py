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
  def call(xt,st,ht):
    g = lstm_gate(tf.tanh,    Ug,Vg,bg, xt,ht)
    i = lstm_gate(tf.sigmoid, Ui,Vi,bi, xt,ht)
    f = lstm_gate(tf.sigmoid, Uf,Vf,bf, xt,ht)
    o = lstm_gate(tf.sigmoid, Uo,Vo,bo, xt,ht)

    st2 = st*f + g*i
    ht2 = tf.tanh(st)*o
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

def model(batch_size, num_inputs, num_channels, num_classes:int=10, init=tf.random_normal):
  """
  Return X,[(batch_size,num_classes) x num_inputs]
  """
  X = tf.placeholder(tf.float32, shape=(batch_size, num_inputs, num_channels))

  Ug = tf.Variable(init([num_channels, num_classes]))
  Vg = tf.Variable(init([num_classes, num_classes]))
  bg = tf.Variable(init([1, num_classes]))

  Ui = tf.Variable(init([num_channels, num_classes]))
  Vi = tf.Variable(init([num_classes, num_classes]))
  bi = tf.Variable(init([1, num_classes]))

  Uf = tf.Variable(init([num_channels, num_classes]))
  Vf = tf.Variable(init([num_classes, num_classes]))
  bf = tf.Variable(init([1, num_classes]))

  Uo = tf.Variable(init([num_channels, num_classes]))
  Vo = tf.Variable(init([num_classes, num_classes]))
  bo = tf.Variable(init([1, num_classes]))

  cell = lstm_cell(Ug,Vg,bg, Ui,Vi,bi, Uf,Vf,bf, Uo,Vo,bo)

  s0 = tf.Variable(init([1, num_classes]))
  h0 = tf.constant(np.zeros(shape=[1, num_classes], dtype=np.float32))
  xs = tf.unstack(X, num_inputs, 1)

  outputs = lstm_layer(cell, xs, s0, h0)
  return X,outputs

def model2(batch_size, num_inputs, num_channels, num_classes:int=10, num_hidden:int=128, init=tf.random_normal):
  W=tf.Variable(init([num_hidden, num_classes]))
  b=tf.Variable(init([1, num_classes]))

  X,outputs=model(batch_size, num_inputs, num_channels, num_classes=num_hidden, init=init)
  cls=tf.squeeze(tf.matmul(outputs[-1],W)+b)
  return X,cls

def mnist_load():
  (Xl,yl),(Xt,yt)=mnist.load_data()
  def oh(y):
    yoh=np.zeros((y.shape[0],10),dtype=np.float32)
    yoh[np.arange(y.shape[0]),y]=1
    return yoh
  return (Xl,oh(yl)),(Xt,oh(yt))

(Xl,yl),(Xt,yt)=mnist_load()

def train():
  batch_size=50
  num_inputs=28
  num_channels=28
  num_hidden=32
  num_classes=10
  training_steps=int(60000/batch_size)
  num_epochs=1000
  learning_rate=0.001
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
    for e in range(num_epochs):
      for step in range(training_steps):
        Xi_=Xl[step*batch_size:(step+1)*batch_size,:,:]
        yi_=yl[step*batch_size:(step+1)*batch_size,:]
        loss_,acc_,_=sess.run((loss_op,accuracy_op,train_op), feed_dict={X:Xi_, y:yi_})

        if step%100==0:
          print("epoch",e,"step",e*training_steps + step,"loss","{:.4f}".format(loss_),"acc","{:.3f}".format(acc_))


def go2():
  batch_size=3
  num_inputs=2
  num_channels=28
  num_hidden=128
  num_classes=10
  with tf.Session(graph=tf.Graph()) as sess:
    X,os=model2(batch_size,num_inputs,num_channels,init=tf.ones)
    sess.run(variables.global_variables_initializer())

    Xz_=np.zeros(shape=(batch_size,num_inputs,num_channels))
    yz_=np.zeros(shape=(batch_size,num_classes))
    o_=sess.run(os, feed_dict={X:Xz_})
    return o_

def go1():
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

