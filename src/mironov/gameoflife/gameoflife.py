import tvm
import topi
import numpy as np
import time

def get_shape(t):
  return [tvm.ir_pass.Simplify(s).value for s in t.shape]

init = \
   [[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   ,[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   ,[ 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
   ,[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   ,[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   ,[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   ,[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   ,[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   ,[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   ,[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   ]

def create():
  A = tvm.placeholder((10,10), name='A')
  A1 = topi.expand_dims(A, 0, 2)
  k = tvm.compute((1,1,3,3), lambda x,y,i,j: tvm.select(i==1, tvm.select(j==1, 0, 1), 1), name='k')
  A2 = topi.nn.pad(A1,pad_before=(0,0,1,1),pad_after=(0,0,1,1))
  N = topi.nn.conv2d(A2,k,strides=1,padding=0,dilation=1)
  A3 = tvm.compute((1,1,10,10), lambda b,c,h,w:
         tvm.select(A1[b,c,h,w] == 1.0,
           tvm.select(N[b,c,h,w]<2, 0.0,
             tvm.select(N[b,c,h,w]>3, 0.0, A1[b,c,h,w])),
           tvm.select(N[b,c,h,w]==3, 1.0, 0.0)))
  A3 = topi.reshape(A3, (10,10))

  dtype = tvm.float32
  sout = tvm.create_schedule(A3.op)
  mout = tvm.build(sout, [A,A3])
  ctx = tvm.cpu(0)

  def run(A):
    inp_nd = tvm.nd.array(np.array(A).astype(np.float32),ctx=ctx)
    out_nd = tvm.nd.array(np.zeros((10,10), dtype=dtype), ctx=ctx)
    mout(inp_nd, out_nd)
    return out_nd.asnumpy()

  return run


def run():
  gol = create()
  s = init
  for i in range(10):
    s = gol(s)
    print(s)

