import tvm
import nnvm
import numpy as np
import tensorflow as tf

from nnvm import sym as _sym
from copy import copy
from tensorflow import Tensor as TF_Tensor
from tensorflow.gfile import FastGFile
from tensorflow.summary import FileWriter
from tensorflow import Graph as TF_Graph, GraphDef as TF_GraphDef
from tensorflow.python.ops import variables
from nnvm.frontend import from_tensorflow
from typing import Any,Dict,List
from tvm.tensor import Tensor as TVM_Tensor

from freezepb.runners import *


def tvm_shape(t):
  return [tvm.ir_pass.Simplify(s).value for s in t.shape]

def nnvm_shape(sym):
  """ Doesn't work for some reason. In model0v2 this worked fine """
  g = nnvm.graph.create(sym)
  g._set_json_attr("shape_attr_key", "shape")
  g = g.apply("InferShape")
  sdict = {}
  vshape = g.json_attr("shape")
  entry_ptr = g.index.entry_ptr
  for i, n in enumerate(g.index.nodes):
    begin, end = entry_ptr[i], entry_ptr[i + 1]
    sdict[n["name"]] = vshape[begin:end]
  return sdict


def conv_test(nblocks=200,ks=1,w=54,h=6,c=256):
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

  for i in range(nblocks):
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
    t=_sym.strided_slice(t,begin=[0,0,0,0],end=[1,1,1,c])
    t=_sym.expand_like(t,x,axis=[1,2])

  r=run_nnvm(1,15,
    {x:np.zeros(shape=shape),
     k:np.zeros(shape=kshape)},
    t,verbose=True)
  return r


def experiment1():
  nblocks=200
  r=conv_test(nblocks,h=54,w=6,c=256,ks=1)
  print(r)
  r=conv_test(nblocks,h=54,w=6,c=256,ks=3)
  print(r)
  r=conv_test(nblocks,h=108,w=21,c=64,ks=1)
  print(r)
  r=conv_test(nblocks,h=108,w=21,c=64,ks=3)
  print(r)


