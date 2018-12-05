"""
Poor man's LSTM cell applied to MNIST. Use `train` function to train the model.
"""

import tensorflow as tf
import numpy as np

from keras.datasets import mnist
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


def lstm_layer(num_timesteps: int, num_inputs: int, num_units: int,
               init=tf.initializers.glorot_uniform(), bias_init=tf.zeros):
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

    def cell(x_t, s_t, h_t):
        xh_t = tf.concat([x_t, h_t], 1)
        g = lstm_gate(tf.tanh, Ug, bg, xh_t)
        i = lstm_gate(tf.sigmoid, Ui, bi, xh_t)
        f = lstm_gate(tf.sigmoid, Uf, bf, xh_t)
        o = lstm_gate(tf.sigmoid, Uo, bo, xh_t)

        s_t1 = s_t * f + g * i
        h_t1 = tf.tanh(s_t1) * o
        return (s_t1, h_t1)

    xs = tf.unstack(X, num_timesteps, 1)
    batch_size = tf.shape(X)[0]
    s_shape = tf.stack([batch_size, num_units], name="s_shape")

    s = tf.zeros(s_shape, dtype=np.float32)
    h = s
    outputs = []
    for x in xs:
        s, h = cell(x, s, h)
        outputs.append(h)

    return X, outputs


def lstm_and_dense_layer(num_timesteps: int, num_inputs: int, num_classes: int, num_hidden: int,
                         dense_init=tf.random_normal):
    """
    Use `model` with `num_hidden` LSTM units, translate them `num_classes` classes using a dense layer.
    """
    W = tf.Variable(dense_init([num_hidden, num_classes]))
    b = tf.Variable(dense_init([1, num_classes]))

    X, outputs = lstm_layer(num_timesteps, num_inputs, num_units=num_hidden)
    cls = tf.matmul(outputs[-1], W) + b
    return X, cls


def mnist_load():
    """ Load MNIST data and convert its ys to one-hot encoding """
    (Xl, yl), (Xt, yt) = mnist.load_data()

    def oh(y):
        yoh = np.zeros((y.shape[0], 10), dtype=np.float32)
        yoh[np.arange(y.shape[0]), y] = 1
        return yoh

    def as_float(X):
        return X.astype(np.float32) / 255.0

    return (as_float(Xl), oh(yl)), (as_float(Xt), oh(yt))


def train(Xl, yl, Xt, yt):
    """ Main train
    :param Xl training examples tensor, shape [num_examples, num_inputs, num_timesteps]
    :param yl one-hot encoded training labels tensor, shape [num_examples, num_classes]
    :param Xt test examples tensor, shape [num_examples, num_inputs, num_timesteps]
    :param yt one-hot encoded test labels tensor, shape [num_examples, num_classes]
    """
    batch_size = 64
    num_timesteps = Xl.shape[2] # number of rows (each row in the image is considered as a timestep)
    num_inputs = Xl.shape[1] # length of each row
    num_hidden = 128
    num_classes = yl.shape[1]
    training_steps = 500 # TODO temporary to make faster, 10000 to actually train
    learning_rate = 0.001
    num_examples = Xl.shape[0]
    with tf.Session(graph=tf.Graph()) as sess:
        X, lstm_out = lstm_and_dense_layer(num_timesteps, num_inputs, num_classes=num_classes, num_hidden=num_hidden)
        y = tf.placeholder(tf.float32, shape=(None, num_classes))

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=lstm_out, labels=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

        prediction = tf.nn.softmax(lstm_out, name='prediction')
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
    (Xl, yl), (Xt, yt) = mnist_load()
    train(Xl, yl, Xt, yt)
