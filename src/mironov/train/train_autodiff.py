import tvm
import topi
import numpy as np
from nnvm.testing.check_computation import check_numerical_grads
import time


def run():
  batch_size = 1
  num_classes = 10

  features = 4
  dense_units = 16

  x = tvm.placeholder((batch_size, 28, 14, 1))
  y = tvm.placeholder((batch_size, num_classes))

  w1 = tvm.placeholder((features, 1, 3, 5))
  b1 = tvm.placeholder((features,))
  w2 = tvm.placeholder((features, features, 3, 5))
  b2 = tvm.placeholder((features,))
  b3 = tvm.placeholder((dense_units,))
  w4 = tvm.placeholder((num_classes, dense_units))
  b4 = tvm.placeholder((num_classes,))

  t = topi.transpose(x, [0, 3, 1, 2])
  t = topi.nn.relu(topi.nn.conv2d(t, w1, 1, 0) + topi.reshape(b1, (1, features, 1, 1)))
  t = topi.nn.relu(topi.nn.conv2d(t, w2, 1, 0) + topi.reshape(b2, (1, features, 1, 1)))
  t = topi.nn.pool(t, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')
  t = topi.transpose(t, [0, 2, 3, 1])
  t = topi.nn.flatten(t)
  w3 = tvm.placeholder((dense_units, get_shape(t)[1]))
  t = topi.nn.relu(topi.nn.dense(t, w3, b3))
  t = topi.nn.dense(t, w4, b4)

  t = - topi.sum(y * topi.nn.log_softmax(t)) / batch_size

  weights = [w1, b1, w2, b2, w3, b3, w4, b4]



def test_grad(out, inp, args=[], in_range=(-10,10)):
  if not isinstance(inp, (list, tuple)):
    inp = [inp]

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

  def fun(*arguments):
    aaa = [tvm.nd.empty(get_shape(out), out.dtype)] + [tvm.nd.array(a) for a in arguments]
    mout(*aaa)
    return aaa[0].asnumpy().sum()

  arg_vals = [
    tvm.nd.array(np.random.uniform(in_range[0], in_range[1], size=get_shape(a)).astype(a.dtype))
      for a in inp + args
    ]

  j_arg_vals = [tvm.nd.empty(get_shape(i), j.dtype) for i, j in zip(inp, jacs)] + arg_vals
  t = time.time()
  mjac(*j_arg_vals)
  j_res = [j_arg_vals[j].asnumpy() for j, _ in enumerate(jacs)]
  print("JAC EXEC TIME: ", time.time() - t)

  t = time.time()
  check_numerical_grads(fun, [a.asnumpy() for a in arg_vals], j_res)
  print("NUMGRAD TIME: ", time.time() - t)

