import tvm
import topi
import nnvm
import numpy as np
import tensorflow as tf

from time import strftime, perf_counter
from typing import Dict,Any,List,Tuple
from os import environ

from tensorflow.python.ops import variables
from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime
from topi.util import get_const_tuple
from nnvm import sym
from nnvm.testing.check_computation import infer_shapes_dtypes
from tensorflow import Tensor as TF_Tensor
from numpy import array as NP_Array
from tvm.tensor import Tensor as TVM_Tensor

Time=float

class Result:

  @classmethod
  def fromSinglePass(cls, out_data:np.array, time:Time, debug_datum=None)->"Result":
    r=Result()
    r.set_perfs([time])
    r.last_data=out_data
    r.last_debug_datum=debug_datum
    return r

  @classmethod
  def fromPasses(cls, out_data:np.array, perfs:List[Time], debug_datum=None)->"Result":
    r=Result()
    r.set_perfs(perfs)
    r.last_data=out_data
    r.last_debug_datum=debug_datum
    return r

  def __init__(s):
    s.perfs:List[float]=None
    s.perf_mean:float=None
    s.perf_std:float=None
    s.last_data:np.array=None
    s.last_debug_datum=None
    s.err=None
    pass

  def set_perfs(s,perfs):
    s.perfs=perfs
    s.perf_mean=np.mean(perfs)
    s.perf_std=np.std(perfs)

  def __repr__(s)->str:
    return "Result(%s)" % (
        (str(s.perf_mean)+'+-'+str(s.perf_std)) if len(s.perfs)>1 else str(s.perf_mean) ,
        )

def run_nnvm(nwarmup:int,nloops:int,
            args:Dict[TVM_Tensor,np.array],
            out:TVM_Tensor,
            verbose:bool=False,
            debug:bool=False,
            opt_level:int=2)->Result:

  if debug:
    runtime=debug_runtime
    create_args={'dump_root':environ['CWD']+'/_debug'}
  else:
    runtime=graph_runtime
    create_args={}

  tgt='llvm'
  ctx=tvm.cpu(0)
  inps=[];ishapes={};itypes={};idata={}
  for i,(arg,val) in enumerate(args.items()):
    nm=arg.list_output_names()[0]
    ishapes.update({nm:val.shape})
    idata.update({nm:val})
    itypes.update({nm:"float32"})

  with nnvm.compiler.build_config(opt_level=opt_level) as cfg:
    print('config', cfg) if verbose else None
    graph,lib,_ = nnvm.compiler.build(out,tgt,ishapes)
    print(graph.ir()) if verbose else None
    #print(graph.json()) if verbose else None

  forward_graph,_,_,out_shapes,out_types = \
      infer_shapes_dtypes(nnvm.graph.create(out), shape=ishapes, dtype=itypes, fallback_dtype='float32')

  out_nd=tvm.nd.array(np.zeros(out_shapes[0], dtype=out_types[0]), ctx)
  m=runtime.create(graph,lib,ctx,**create_args)
  m.set_input(**idata)

  debug_datum=None
  perfs:List[float]=[]
  for i in range(nwarmup+nloops):
    tb=perf_counter()
    m.run()
    te=perf_counter()
    if i>=nwarmup:
      perfs.append(te-tb)
    if verbose:
      print("NNVM",te-tb)
    if debug:
      debug_datum=m.debug_datum
  out_nd=m.get_output(0, tvm.nd.empty(shape=out_shapes[0],dtype=out_types[0],ctx=ctx))
  return Result.fromPasses(out_nd.asnumpy(),perfs,debug_datum=debug_datum)


def with_nnvm(nwarmup:int,nloops:int,args:list,lam, params={},verbose:bool=False,opt_level:int=2)->Result:
  """ Take numpy arrays as args, convert them to TVM tensors and call `lam`.
  Result of lambda is converted back to numpy array and returned.

  TODO: re-implement using run_nnvm
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
  with nnvm.compiler.build_config(opt_level=opt_level):
    graph,lib,_ = nnvm.compiler.build(out,tgt,ishapes)

  forward_graph,_,_,out_shapes,out_types = \
      infer_shapes_dtypes(nnvm.graph.create(out), shape=ishapes, dtype=itypes, fallback_dtype='float32')

  out_nd=tvm.nd.array(np.zeros(out_shapes[0], dtype=out_types[0]), ctx)
  m=graph_runtime.create(graph,lib,ctx)
  m.set_input(**idata)
  m.set_input(**params)

  perfs:List[float]=[]
  for i in range(nwarmup+nloops):
    tb=perf_counter()
    m.run()
    te=perf_counter()
    if i>=nwarmup:
      perfs.append(te-tb)
    if verbose:
      print("NNVM",te-tb)
  out_nd=m.get_output(0, tvm.nd.empty(shape=out_shapes[0],dtype=out_types[0],ctx=ctx))
  return Result.fromPasses(out_nd.asnumpy(),perfs)

def with_tvm(nwarmup:int,nloops:int,args,lam,verbose:bool=False)->Result:
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

  with nnvm.compiler.build_config(opt_level=3):
    graph,lib,params=nnvm.compiler.build(graph=sym, target='llvm', shape=i_shape_dict, dtype=i_dtype_dict, params=params)

  out_nd = tvm.nd.array(np.zeros(get_const_tuple(out.shape), dtype=out.dtype), ctx)
  s = tvm.create_schedule([out.op])
  m = tvm.build(s, pls + [out], "llvm")

  perfs:List[float]=[]
  for i in range(nwarmup+nloops):
    tb=perf_counter()
    m(*(vals_nd+[out_nd]))
    te=perf_counter()
    if i>=nwarmup:
      perfs.append(te-tb)
    if verbose:
      print("TVM",te-tb)
  return Result.fromPasses(out_nd.asnumpy(),perfs)


def run_tf(sess,nwarmup:int,nloops:int,args:Dict[TF_Tensor,NP_Array],out:TF_Tensor,verbose:bool=False)->Result:
  inits={}; pls=[]
  for pl,val in args.items():
    pls.append(pl)
    inits.update({pl:val})

  perfs:List[float]=[]
  for i in range(nwarmup+nloops):
    sess.run(variables.global_variables_initializer())
    tb=perf_counter()
    o_np=sess.run(out, inits)
    te=perf_counter()
    if i>=nwarmup:
      perfs.append(te-tb)
    if verbose:
      print("TF",te-tb)

  return Result.fromPasses(o_np,perfs)


def with_tf(nwarmup:int,nloops:int,args,lam,verbose:bool=False)->Result:
  """ Deprecated: 1) implement via run_tf 2) implement using functional approach
  """
  with tf.Session(graph=tf.Graph()) as sess:
    inits={}; pls=[]
    for i,arg in enumerate(args):
      pls.append(tf.placeholder(tf.float32, shape=arg.shape, name='pl'+str(i)))
      inits.update({pls[-1]:arg})
    o_t=lam(*pls)
    o_np:np.array=None
    perfs:List[float]=[]
    for i in range(nwarmup+nloops):
      sess.run(variables.global_variables_initializer())
      tb=perf_counter()
      o_np=sess.run(o_t, inits)
      te=perf_counter()
      if i>=nwarmup:
        perfs.append(te-tb)
      if verbose:
        print("TF",te-tb)

    return Result.fromPasses(o_np,perfs)


