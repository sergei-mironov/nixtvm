import tvm
import topi
import numpy as np
import time
import keras.datasets.mnist as mnist

from train.runners import run_tvm


def mnist_load():
  (Xtr,ytr),(Xte,yte) = mnist.load_data("/tmp/mnist.npz")
  return (Xtr,ytr),(Xte,yte)

def mnist_img(ids):
  return np.expand_dims(mnist_load()[0][0][ids], axis=3).astype(np.float32)

def mnist_cls(ids):
  return mnist_load()[0][1][ids]

def mnist_cls_oh(ids):
  z=np.zeros((len(ids),10),dtype=np.float32)
  z[np.arange(len(ids)),mnist_cls(ids)]=1
  return z.astype(np.float32)

def get_shape(tensor):
  return [tvm.ir_pass.Simplify(s).value for s in tensor.shape]



def demo_conv2d():
  lrate = 0.1
  nbatches = 100 # batches to train

  num_classes = 10
  batch_size = 10
  img_h = 28
  img_w = 28
  img_c = 1

  f1_c = 4
  f2_c = 5
  f3_units = 16

  x = tvm.placeholder((batch_size, img_h, img_w, img_c),name='x')
  y = tvm.placeholder((batch_size, num_classes),name='y')

  print('Block1')
  w1 = tvm.placeholder((3,3,img_c,f1_c),name='w1')
  b1 = tvm.placeholder((f1_c,), name='b1')
  t = topi.nn.conv2d(x, w1, 1, 0, layout='NHWC', out_dtype=tvm.float32)
  t = t + topi.broadcast_to(b1, (batch_size,1,1,f1_c))
  print('Block1: after-biasing shape is', get_shape(t))
  t = topi.nn.pool(t, [2, 2], [2, 2], [0, 0, 0, 0], 'max', layout='NHWC')
  print('Block1: after-pooling shape is', get_shape(t))
  t = topi.nn.relu(t)
  print('Block1: after-relu shape is', get_shape(t))


  print('Block2')
  w2 = tvm.placeholder((3,3,f1_c,f2_c),name='w2')
  b2 = tvm.placeholder((f2_c,), name='b2')
  t = topi.nn.conv2d(t, w2, 1, 0, layout='NHWC', out_dtype=tvm.float32)
  t = t + topi.broadcast_to(b2, (batch_size,1,1,f2_c))
  print('Block2: after-biasing shape is', get_shape(t))
  t = topi.nn.pool(t, [2, 2], [2, 2], [0, 0, 0, 0], 'max', layout='NHWC')
  print('Block2: after-pooling shape is', get_shape(t))
  t = topi.nn.relu(t)
  print('Block2: after-relu shape is', get_shape(t))
  t = topi.nn.flatten(t)
  print('Block2: after-flattern shape is', get_shape(t))


  print('Block3')
  w3 = tvm.placeholder((f3_units, get_shape(t)[1]))
  b3 = tvm.placeholder((f3_units,))
  t = topi.nn.dense(t,w3,b3)
  print('Block3: after-dense shape is', get_shape(t))


  print('Block4')
  w4 = tvm.placeholder((num_classes, get_shape(t)[1]))
  b4 = tvm.placeholder((num_classes,))
  t = topi.nn.dense(t,w4,b4)
  print('Block4: after-dense shape is', get_shape(t))
  t = topi.nn.relu(t)

  p = topi.argmax(t,axis=1)
  # TODO: check the correctnesss of the log_softmax expression
  # TODO: figure out the difference between it and standard cross-entropy loss
  l = - topi.sum(y * topi.nn.log_softmax(t)) / batch_size

  print('Block4: loss shape is', get_shape(l))

  ones = topi.full_like(l, 1.0)
  #[dl_dw1,dl_db1,dl_dw2,dl_db2,dl_dw3,dl_db3,dl_dw4,dl_db4]
  params = [w1,b1,w2,b2,w3,b3,w4,b4]

  dl = list(tvm.ir_pass.JacobianRecursive(l, params, ones))
  assert len(params)==len(dl)
  print('dl_dw1 weight is', get_shape(params[0]))

  sdl = tvm.create_schedule([p.op for p in [x,y,l] + params + dl])
  mdl = tvm.build(sdl, [x,y,l] + params + dl)
  print('Train+Inference module', mdl)

  # sl = tvm.create_schedule([l.op])
  # ml = tvm.build(sdl, [x,y] + params + [l])
  # print('Inference module',ml)

  state={}
  for p in params:
    state.update({p:tvm.nd.array(np.random.uniform(-1.0, 1.0, size=get_shape(p)).astype(np.float32))})

  grads={}
  for p,g in zip(params,dl):
    grads.update({p:tvm.nd.empty(get_shape(g))})

  for ib in range(nbatches):
    b=range(ib*batch_size,(ib+1)*batch_size)
    tx=tvm.nd.array(mnist_img(b))
    ty=tvm.nd.array(mnist_cls_oh(b))
    tl=tvm.nd.empty(shape=(), dtype=tvm.float32)

    print('Entering')
    mdl(*([tx,ty,tl]+list(state.values())+list(grads.values())))
    print('Done','loss',tl.asnumpy())

    state2={}
    for p in params:
      state2.update({p:tvm.nd.array(state[p].asnumpy()-lrate*grads[p].asnumpy())})

    state = state2



