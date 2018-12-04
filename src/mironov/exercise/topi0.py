import tvm
import topi
import numpy as np
import time

from exercise.runners import run_tvm,with_tvm


def get_shape(t):
  return [tvm.ir_pass.Simplify(s).value for s in t.shape]


def demo_broadcast():
  """ Check that broad works as expected """

  num_classes = 10
  batch_size = 1
  img_h = 28
  img_w = 28
  img_c = 1

  f1_c = 1

  x = tvm.placeholder((batch_size, img_h, img_w, img_c),name='x')
  b = tvm.placeholder((img_c,), name='b')

  # Plus here will perform auto-broadcast
  y = x + topi.broadcast_to(b, (batch_size,1,1,img_c))

  npy = run_tvm(0,1,
          { x:np.ones(get_shape(x)).astype(np.float32)
          , b:np.ones(get_shape(b)).astype(np.float32)
          },y)

  print(npy.last_data[0,:,:,0])



def demo_softmax():
  x = np.array([[0,0],[0, 1],[1,0],[1,1]]).astype(np.float32)
  print( with_tvm(0,1,[x], lambda a: topi.nn.softmax(a)))
  print( with_tvm(0,1,[x], lambda a: topi.nn.log_softmax(a)))


def demo_argmax():
  x = np.array([[0,0,1,0,0],[3,1,2,0,0],[0,0,0,1,2]]).astype(np.float32)
  return with_tvm(0,1,[x],lambda a: topi.argmax(a,axis=1))

def demo_sigmoid():
  x = np.array([[0,0,1,0,0],[3,1,2,0,0],[0,0,0,1,2]]).astype(np.float32)
  r = with_tvm(0,1,[x],lambda a: topi.sigmoid(a))
  print(r)

