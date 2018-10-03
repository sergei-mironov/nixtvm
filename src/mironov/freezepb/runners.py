import tvm
import topi
import nnvm
import numpy as np
import tensorflow as tf

from tensorflow.python.ops import variables
from tvm.contrib import graph_runtime
from topi.util import get_const_tuple
from nnvm import sym
from nnvm.testing.check_computation import infer_shapes_dtypes

def with_nnvm(args,lam, params={})->np.array:
  """ Take numpy arrays as args, convert them to TVM tensors and call `lam`.
  Result of lambda is converted back to numpy array and returned.
  """
  tgt='llvm'
  ctx=tvm.cpu(0)
  inps=[];ishapes={};itypes={};idata={}
  for i,arg in enumerate(args):
    nm='pl'+str(i)
    inps.append(sym.Variable(name=nm))
    ishapes.update({nm:arg.shape})
    idata.update({nm:arg})
    itypes.update({nm:"float32"})

  out=lam(*inps)
  graph,lib,_ = nnvm.compiler.build(out,tgt,ishapes)

  forward_graph,_,_,out_shapes,out_types = \
      infer_shapes_dtypes(nnvm.graph.create(out), shape=ishapes, dtype=itypes, fallback_dtype='float32')

  out_nd=tvm.nd.array(np.zeros(out_shapes[0], dtype=out_types[0]), ctx)
  m=graph_runtime.create(graph,lib,ctx)
  m.set_input(**idata)
  m.set_input(**params)
  m.run()
  out_np=m.get_output(0, tvm.nd.empty(shape=out_shapes[0],dtype=out_types[0],ctx=ctx)).asnumpy()
  return out_np

def with_tvm(args,lam)->np.array:
  """ Take numpy arrays as args, convert them to TVM tensors and call `lam`.
  Result of lambda is converted back to numpy array and returned.
  """
  ctx = tvm.cpu(0)
  pls = []     # placeholders
  vals_nd = [] # initial values
  for i,arg in enumerate(args):
    pls.append(tvm.placeholder(arg.shape, name='pl'+str(i)))
    vals_nd.append(tvm.nd.array(arg, ctx))
  out = lam(*pls)

  with nnvm.compiler.build_config(opt_level=opt_level):
    graph,lib,params=nnvm.compiler.build(graph=sym, target='llvm', shape=i_shape_dict, dtype=i_dtype_dict, params=params)

  out_nd = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=out.dtype), ctx)
  s = tvm.create_schedule([out.op])
  m = tvm.build(s, pls + [out], "llvm")
  m(*(vals_nd+[out_nd]))
  return out_nd.asnumpy()


def with_tf(args,lam)->np.array:

  with tf.Session(graph=tf.Graph()) as sess:
    inits={}; pls=[]
    for i,arg in enumerate(args):
      pls.append(tf.placeholder(tf.float32, shape=arg.shape, name='pl'+str(i)))
      inits.update({pls[-1]:arg})

    o_t=lam(*pls)
    sess.run(variables.global_variables_initializer())
    o_np=sess.run(o_t, inits)
    return o_np

