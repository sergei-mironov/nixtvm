import tvm
import nnvm
import numpy as np
import tensorflow as tf

from nnvm import sym as _sym
from nnvm.frontend import from_tensorflow
from nnvm.top.nn import schedule_conv2d, compute_conv2d
from nnvm.top import registry as reg
from topi.generic.nn import _default_schedule

from tvm.tensor import Tensor as TVM_Tensor
from copy import copy
from tensorflow import Tensor as TF_Tensor
from tensorflow.gfile import FastGFile
from tensorflow.summary import FileWriter
from tensorflow import Graph as TF_Graph, GraphDef as TF_GraphDef
from tensorflow.python.ops import variables
from typing import Any,Dict,List

from freezepb.runners import *



@reg.register_schedule("conv2d",level=11)
def schedule_conv2d(attrs, outs, target):
  print('Setting custom scheduling','target', target)

  """ Default schedule for llvm. """
  target = tvm.target.current_target(allow_none=False)
  outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
  if target.target_name != "llvm":
    raise RuntimeError("schedule not registered for '%s'" % target)
  s = tvm.create_schedule([x.op for x in outs])

  for x in outs:
    print('x:', type(x))
    print('op:', type(x.op), x.op)
    print('op-dir:', x.op.__dir__())
    print('op-axis:', x.op.axis)
    print('op-body:', x.op.body)

  # print(tvm.lower(s, [x], simple_mode=True))

  # Auto inline branch of the `_default_schedule`
  # x = outs[0]
  # tvm.schedule.AutoInlineInjective(s)
  # s[x].fuse(s[x].op.axis)
  return s


def nnvm_conv_test(nblocks=200,ks=1,w=54,h=6,c=256,verbose:bool=False,opt_level:int=2):
  """ Test convolution performance for different shape """

  shape=(1,h,w,c)
  kshape=(ks,ks,c,c)
  x=_sym.Variable(init=np.zeros(shape=shape),name='x')
  k=_sym.Variable(init=np.zeros(shape=kshape),name='k')
  t=x

  def _print_shape(t):
    print(run_nnvm(0,1,
        {x:np.zeros(shape=shape),
         k:np.zeros(shape=kshape)},
        t).last_data.shape)

  t=_sym.conv2d(t,k,
      dilation=(1,1),
      layout="NHWC",
      strides=(1,1),
      padding=[0,0],
      kernel_size=(ks,ks),
      channels=c,
      kernel_layout="HWIO",
      name="conv1",
      use_bias=False)

  # print(t.__dir__())
  # print(t.list_attr())

  t=_sym.strided_slice(t,begin=[0,0,0,0],end=[1,1,1,c])
  t=_sym.expand_like(t,x,axis=[1,2])

  r=run_nnvm(1,15,
    {x:np.zeros(shape=shape)
    ,k:np.zeros(shape=kshape)}, t,
    verbose=verbose,
    opt_level=opt_level)

