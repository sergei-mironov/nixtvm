import tvm
import topi
import numpy as np
from nnvm.testing.check_computation import check_numerical_grads
import time

import keras.datasets.mnist as mnist

def minst_load():
  (X,y),(Xte,yte) = mnist.load_data("/tmp/mnist.npz")
  return (X,y),(Xte,yte)


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

  return conv_run(t, weights, [x, y], in_range=(-1.0, 1.0))


def ex1_build():
  """
  gdb --args `which python`  -c 'from train.train_autodiff import * ; ex1_build();'

  (gdb) bt
  #0  0x00007ffff7e16048 in default_function ()
  #1  0x00007fffdfbdd0c0 in tvm::runtime::WrapPackedFunc(int (*)(void*, int*, int), std::shared_ptr<tvm::runtime::ModuleNode> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) const [clone .isra.71] ()
     from /home/grwlf/proj/nixtvm/tvm/build-native/libtvm.so
  #2  0x00007fffdfbc48de in TVMFuncCall () from /home/grwlf/proj/nixtvm/tvm/build-native/libtvm.so
  #3  0x00007fffef013a3e in ffi_call_unix64 () from /nix/store/6z0spj65xwl2wf7dbn0fd8jxcka3yx3h-libffi-3.2.1/lib/libffi.so.6
  #4  0x00007fffef012a03 in ffi_call () from /nix/store/6z0spj65xwl2wf7dbn0fd8jxcka3yx3h-libffi-3.2.1/lib/libffi.so.6
  #5  0x00007fffef22850d in _ctypes_callproc ()
  """

  x = tvm.placeholder((1,))
  k = tvm.placeholder((1,))
  y = tvm.compute((1,), lambda i: x[i]*k[0], name='y')

  ones = topi.full_like(y, 1.0)
  jac = list(tvm.ir_pass.JacobianRecursive(y, [x], ones))
  print('jac',type(jac),jac[0])

  sjac = tvm.create_schedule([j.op for j in jac])
  mjac = tvm.build(sjac, jac + [y,x,k])  # TODO: Why should we specify y,x,k?
  # print(type(mjac), mjac.get_source())


  args = [tvm.nd.empty(get_shape(i), j.dtype) for i, j in zip([y], jac)] +\
         [tvm.nd.array(np.array([1.0]).astype(a.dtype)) for a in [x,k]]
  print('args',type(args),len(args),args)

  mjac(*args)

  print('Done??')
  print(jout.asnumpy())



