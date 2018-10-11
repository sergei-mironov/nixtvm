import tvm
import topi
import numpy as np
import time
import keras.datasets.mnist as mnist

from train.runners import run_tvm


def mnist_load():
  (Xtr,ytr),(Xte,yte) = mnist.load_data("/tmp/mnist.npz")
  return (Xtr,ytr),(Xte,yte)


def mnist_img(i):
  return mnist_load()[0][0][i]



def get_shape(tensor):
  return [tvm.ir_pass.Simplify(s).value for s in tensor.shape]


def conv_run(out, inp, args=[], in_range=(-10,10)):
  sout = tvm.create_schedule(out.op)
  mout = tvm.build(sout, [out] + inp + args)

  ones = topi.full_like(out, 1.0)

  t = time.time()
  jacs = list(tvm.ir_pass.JacobianRecursive(out, inp, ones))
  print("JAC TIME: ", time.time() - t)

  t = time.time()
  sjac = tvm.create_schedule([j.op for j in jacs])
  mjac = tvm.build(sjac, jacs + inp + args)
  print("BUILD TIME: ", time.time() - t)
  return mjac

def conv_build():
  num_classes = 10
  batch_size = 1
  img_h = 28
  img_w = 28
  img_c = 1

  f1_c = 1

  x = tvm.placeholder((batch_size, img_h, img_w, img_c),name='x')
  w1 = tvm.placeholder((3,3,img_c,f1_c),name='w1')
  # b1 = tvm.placeholder((f1_c,))
  t = topi.nn.conv2d(x, w1, 1, 0, layout='NHWC', out_dtype=tvm.float32)

  npt = run_tvm(
      0,1,
      { x:mnist_img(3).reshape(get_shape(x)).astype(np.float32)
      , w1:np.zeros(get_shape(w1)).astype(np.float32)
      },t)

  print(npt.last_data)
  return npt


  # w2 = tvm.placeholder((features, features, 3, 5))
  # b2 = tvm.placeholder((features,))
  # b3 = tvm.placeholder((dense_units,))
  # w4 = tvm.placeholder((num_classes, dense_units))
  # b4 = tvm.placeholder((num_classes,))
  # y = tvm.placeholder((batch_size, num_classes))

  # t = topi.nn.relu(topi.nn.conv2d(t, w1, 1, 0, layout='NHWC') + topi.reshape(b1, (1, features, 1, 1)))

  # t = topi.transpose(x, [0, 3, 1, 2])
  # t = topi.nn.relu(topi.nn.conv2d(t, w1, 1, 0) + topi.reshape(b1, (1, features, 1, 1)))
  # t = topi.nn.relu(topi.nn.conv2d(t, w2, 1, 0) + topi.reshape(b2, (1, features, 1, 1)))
  # t = topi.nn.pool(t, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')
  # t = topi.transpose(t, [0, 2, 3, 1])
  # t = topi.nn.flatten(t)
  # w3 = tvm.placeholder((dense_units, get_shape(t)[1]))
  # t = topi.nn.relu(topi.nn.dense(t, w3, b3))
  # t = topi.nn.dense(t, w4, b4)

  # t = - topi.sum(y * topi.nn.log_softmax(t)) / batch_size

  # weights = [w1, b1, w2, b2, w3, b3, w4, b4]

  # return conv_run(t, weights, [x, y], in_range=(-1.0, 1.0))

