"""
Poor man's LSTM cell applied to MNIST. Use `train` function to train the model.
"""

import tensorflow as tf
import numpy as np

from typing import Any, List, Dict
from keras.datasets import mnist
from tensorflow import Tensor as TF_Tensor
from tensorflow.python.ops import variables
from tensorflow.python.saved_model.simple_save import simple_save

def lstm_gate(op, U, b, x):
    """
    op - nonlinearity operation
    x - input tensor of shape (1,a)
    U - weight matrix of shape (a,b)
    b - bias (1,b)

    return tensor of shape (1,b)
    """
    return op(tf.matmul(x, U) + b)


def lstm_cell(Ug, bg, Ui, bi, Uf, bf, Uo, bo):
    """ LSTM cell. Ideomatic TF code would define all the variable here """

    def call(xt, st, ht):
        input = tf.concat([xt, ht], 1)
        g = lstm_gate(tf.tanh, Ug, bg, input)
        i = lstm_gate(tf.sigmoid, Ui, bi, input)
        f = lstm_gate(tf.sigmoid, Uf, bf, input)
        o = lstm_gate(tf.sigmoid, Uo, bo, input)

        st2 = st * f + g * i
        ht2 = tf.tanh(st2) * o
        return (st2, ht2)

    return call


def lstm_layer(cell, xs: List[TF_Tensor], s0, h0) -> List[TF_Tensor]:
    h = h0
    s = s0
    hs = []
    for i in range(len(xs)):
        s, h = cell(xs[i], s, h)
        hs.append(h)
    return hs


def model(num_timesteps: int, num_inputs: int, num_units: int, init=tf.initializers.glorot_uniform(),
          bias_init=tf.zeros):
    """
    Create a single cell and replicate it `num_timesteps` times for training.
    Return X,[(batch_size,num_classes) x num_timesteps]
    """
    X = tf.placeholder(tf.float32, shape=(None, num_timesteps, num_inputs))

    U_shape = [num_inputs + num_units, num_units]
    b_shape = [1, num_units]
    Ug = tf.Variable(init(U_shape))
    bg = tf.Variable(bias_init(b_shape))

    Ui = tf.Variable(init(U_shape))
    bi = tf.Variable(bias_init(b_shape))

    Uf = tf.Variable(init(U_shape))
    bf = tf.Variable(bias_init(b_shape) + tf.ones(b_shape))

    Uo = tf.Variable(init(U_shape))
    bo = tf.Variable(bias_init(b_shape))

    cell = lstm_cell(Ug, bg, Ui, bi, Uf, bf, Uo, bo)

    xs = tf.unstack(X, num_timesteps, 1)
    x = xs[0]
    s_shape = tf.stack([x.shape[0].value or tf.shape(x)[0], num_units], name="s_shape")
    s0 = tf.zeros(s_shape, dtype=np.float32)
    h0 = s0

    outputs = lstm_layer(cell, xs, s0, h0)
    return X, outputs


def model2(num_timesteps: int, num_inputs: int, num_classes: int, num_hidden: int,
           init=tf.random_normal):
    """
    Use `model` with 128 "classes", but translate them back to 10 classes via
    dense layer.
    """
    W = tf.Variable(init([num_hidden, num_classes]))
    b = tf.Variable(init([1, num_classes]))

    X, outputs = model(num_timesteps, num_inputs, num_units=num_hidden)
    cls = tf.matmul(outputs[-1], W) + b
    return X, cls


def mnist_load():
    """ Load MNIST and convert its ys to one-hot encoding """
    (Xl, yl), (Xt, yt) = mnist.load_data()

    def oh(y):
        yoh = np.zeros((y.shape[0], 10), dtype=np.float32)
        yoh[np.arange(y.shape[0]), y] = 1
        return yoh

    def as_float(X):
        return X.astype(np.float32) / 255.0

    Xl = as_float(Xl)
    Xt = as_float(Xt)
    return (Xl, oh(yl)), (Xt, oh(yt))


def train():
    (Xl, yl), (Xt, yt) = mnist_load()
    """ Main train """
    batch_size = 64
    num_timesteps = 28 # number of rows (each row in the image is considered as a timestep)
    num_inputs = 28 # length of each row
    num_hidden = 128
    num_classes = 10
    training_steps = 500 # TODO temporary to make faster, 10000 to actually train
    learning_rate = 0.001
    num_examples = Xl.shape[0]
    with tf.Session(graph=tf.Graph()) as sess:
        X, logits = model2(num_timesteps, num_inputs, num_classes=num_classes, num_hidden=num_hidden,
                           init=tf.random_normal)
        y = tf.placeholder(tf.float32, shape=(None, num_classes))

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))
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
            nonlocal epoch, batch_start, batch_end, Xl, yl
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

        for step in range(training_steps + 1):
            batch = next_batch()
            sess.run(train_op, feed_dict=batch)

            if step % 100 == 0:
                loss_, acc_ = sess.run((loss_op, accuracy_op), feed_dict=batch)
                print("epoch", epoch, "step", step, "loss", "{:.4f}".format(loss_), "acc",
                      "{:.2f}".format(acc_))

        print("Optimization Finished!")
        print("Testing Accuracy:",
            sess.run(accuracy_op, feed_dict={X: Xt, y: yt}))

        save_model = False
        if save_model:
            print("Saving the model")
            simple_save(sess, export_dir='./lstm', inputs={"images":X}, outputs={"out":prediction})


if __name__ == '__main__':
    train()
