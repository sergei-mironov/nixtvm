"""
This example was provided by Canada TVM team. Should illustrate memory
consumption.


I tried to run your autodiff with a little bigger network (containing 3 layers
conv2d/relu/maxpool). And I think it has problem with overflow (the error
message was “Error:
Total size for allocation auto.extracted_reduction.v0 is constant but
exceeds 2^31 - 1. Aborted”) when the sizes are bigger. The attached
small program work for batch_size = 5 but not for batch_size = 6 (and
bigger). Increasing other sizes (such as num_channels, kernel_size,…)
also causes similar error.

"""
import tvm
import topi
import numpy as np
from nnvm.testing.check_computation import check_numerical_grads
import nnvm.symbol as sym
import nnvm.graph as graph
import nnvm.compiler.graph_util as graph_util
import nnvm.compiler


def train():

    h = 28
    w = 28
    batch_size = 10
    num_channels = 3
    kernel_size = 8

    X = tvm.placeholder((batch_size, 3, h, w), name='X')
    W = tvm.placeholder((num_channels, 3, kernel_size, kernel_size), name='W')

    R = topi.nn.conv2d(X, W, strides=1, padding=1) # [100, 64, 23, 23]
    R = topi.nn.relu(R)
    R = topi.nn.pool(R, kernel=(2,2), stride=(1,1), padding = (1,1,1,1), pool_type='max')

    sout = tvm.create_schedule(R.op)
    ir_sout  = tvm.lower(sout, [X, W, R], simple_mode=True)
    mout = tvm.build(sout, [R] + [X, W])

    ones = topi.full_like(R, 1.0)
    jacs = list(tvm.ir_pass.JacobianRecursive(R, [X, W], ones))

    sjac = tvm.create_schedule([j.op for j in jacs])
    ir_jac  = tvm.lower(sjac, [X, W, R], simple_mode=True)
    mjac = tvm.build(sjac, jacs + [X, W])

    print("IR of Output:..................................\n",ir_sout.body)
    print("IR of Jacobian:................................\n",ir_jac.body)

