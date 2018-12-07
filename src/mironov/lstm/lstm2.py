# Original version from Dmitry Murygin's repo
#
import tvm
import topi
import numpy as np
import time

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#Training Parameters
lr = 0.001
num_steps = 10000
batch_size = 64
display_step = 100

# Network Parameters
# TODO: fix error with bigger num_timesteps
num_timesteps = 28
num_input = 28
num_hidden = 128
num_classes = 10

# Make parallel computing
def parallel_schedule(sched):
    for s in sched.stages:
        if isinstance(s.op, tvm.tensor.ComputeOp) and isinstance(s.op.body[0], tvm.expr.Reduce):
            ax = s.fuse(*s.op.axis)
            axo, axi = s.split(ax, nparts=20)
            s.parallel(axo)

# Weights sizes
sizes = [
    (num_input + num_hidden, num_hidden),
    (num_hidden,),
    (num_input + num_hidden, num_hidden),
    (num_hidden,),
    (num_input + num_hidden, num_hidden),
    (num_hidden,),
    (num_input + num_hidden, num_hidden),
    (num_hidden,),
    (num_hidden, num_classes),
    (num_classes,)
]
inits = [
    (np.zeros, 'shape'),
    (np.zeros, 'shape'),
    (np.zeros, 'shape'),
    (np.zeros, 'shape'),
    (np.zeros, 'shape'),
    (np.ones, 'shape'),
    (np.zeros, 'shape'),
    (np.zeros, 'shape'),
    (np.random.normal, 'size'),
    (np.random.normal, 'size')
]

# Graph input
x = tvm.placeholder((batch_size, num_timesteps * num_input), 'float32')
y = tvm.placeholder((batch_size, num_classes), 'float32')
s = tvm.placeholder((batch_size, num_hidden), 'float32')
h = tvm.placeholder((batch_size, num_hidden), 'float32')

# Tensors and vars for training graph
weights = [tvm.placeholder(x, 'float32') for x in sizes]

#Construct model
xs = topi.split(topi.reshape(x, (batch_size, num_timesteps, num_input)), num_timesteps, axis=1)
xs = [topi.reshape(x, (batch_size, num_input)) for x in xs]
new_s = s
new_h = h
for i in range(num_timesteps):
    inp = topi.concatenate([xs[i], new_h], 1)
    g = topi.tanh(topi.matmul(inp, weights[0]) + weights[1])
    j = topi.sigmoid(topi.matmul(inp, weights[2]) + weights[3])
    f = topi.sigmoid(topi.matmul(inp, weights[4]) + weights[5])
    o = topi.sigmoid(topi.matmul(inp, weights[6]) + weights[7])

    new_s = new_s * f + g * j
    new_h = topi.tanh(new_s) * o

logits = topi.matmul(new_h, weights[8]) + weights[9]

# compute accuracy
pred = topi.nn.softmax(logits)
correct_pred = topi.equal(topi.argmax(y, 1), topi.argmax(pred, 1))
accuracy = topi.sum(correct_pred.astype('float32')) / batch_size

# Define loss and optimizer
loss = topi.sum(-topi.sum(y * topi.nn.log_softmax(logits), axis=1)) / batch_size

head = topi.full((1,), 'float32', 1.0)
gradients = list(tvm.differentiate(topi.reshape(loss, (1,)), weights, head))
new_weights = [w - lr * g for (w, g) in zip(weights, gradients)]

# Define model
sched = tvm.create_schedule([loss.op, accuracy.op] + [x.op for x in new_weights])
parallel_schedule(sched)
train_model = tvm.build(sched, [x, y, s, h, loss, accuracy, *weights, *new_weights])

# Define variables for input to graph
train_weights = [tvm.ndarray.array(op[0](**{op[1] : x}).astype('float32')) for (op, x) in zip(inits, sizes)]
train_s = tvm.ndarray.array(np.zeros((batch_size, num_hidden)).astype('float32'))
train_h = tvm.ndarray.array(np.zeros((batch_size, num_hidden)).astype('float32'))
train_loss = tvm.ndarray.array(np.array(0).astype('float32'))
train_accuracy = tvm.ndarray.array(np.array(0).astype('float32'))

# Training loop
start_time = time.time()
for step in range(1, num_steps + 1):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = tvm.ndarray.array(batch_x.astype('float32'))
    batch_y = tvm.ndarray.array(batch_y.astype('float32'))
    train_model(batch_x, batch_y, train_s, train_h, train_loss, train_accuracy, *train_weights, *train_weights)
    if step % display_step == 0 or step == 1:
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(train_loss.asnumpy()) + ", Training Accuracy= " + \
              "{:.3f}".format(train_accuracy.asnumpy()))

print("Optimization Finished!")
print("Train_time :", time.time() - start_time)

