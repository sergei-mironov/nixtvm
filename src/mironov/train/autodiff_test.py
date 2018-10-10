import tvm
import topi
import numpy as np

def example1_build():
  x = tvm.placeholder((1,))
  k = tvm.placeholder((1,))
  y = tvm.compute((1,), lambda i: x[i]*x[i]*k[0], name='y')

  [dy] = tvm.ir_pass.JacobianRecursive(y, [x], topi.full_like(y, 1.0))

  sdy = tvm.create_schedule(dy.op)
  mdy = tvm.build(sdy, [dy,x,k])

  dy_out = tvm.nd.empty(get_shape(y), tvm.float32)

  mdy(dy_out,tvm.nd.array(np.array([1.0]).astype(x.dtype))
            ,tvm.nd.array(np.array([4.0]).astype(k.dtype)))
  print(dy_out)


def example2_build():
  npoints = 20
  x = tvm.placeholder((1,))
  k = tvm.placeholder((2,))
  y = tvm.compute((1,), lambda i: x[i]*x[i]*k[0] + x[i]*k[1], name='y')

  ones = topi.full_like(y, 1.0)
  [dy] = list(tvm.ir_pass.JacobianRecursive(y, [x], ones))
  # print('jac',type(jac),jac[0])

  sdy = tvm.create_schedule(dy.op)
  sy = tvm.create_schedule(y.op)

  my = tvm.build(sy, [y,x,k])
  mdy = tvm.build(sdy, [dy,x,k])

  xs=np.linspace(-10.0,10.0,20);ys=[];dys=[]
  for xi in xs:
    k_in = tvm.nd.array(np.array([4.0, 2.0]).astype(k.dtype))

    y_out = tvm.nd.empty(get_shape(y), y.dtype)
    my(y_out, tvm.nd.array(np.array([xi]).astype(x.dtype)), k_in)
    ys.append(y_out.asnumpy())

    dy_out = tvm.nd.empty(get_shape(y), dy.dtype)
    mdy(dy_out, tvm.nd.array(np.array([xi]).astype(x.dtype)), k_in)
    dys.append(dy_out.asnumpy())

    # print(xi, y_out.asnumpy(), dy_out.asnumpy())

  print(xs,ys,dys)


