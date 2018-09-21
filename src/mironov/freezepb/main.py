
from os import environ
from os.path import isfile, join
from typing import List,Tuple,Dict,Any
from time import strftime, perf_counter
from copy import copy

from tensorflow import Tensor as TF_Tensor
from tensorflow.gfile import FastGFile
from tensorflow.summary import FileWriter
from tensorflow import Graph as TF_Graph, GraphDef as TF_GraphDef
from tensorflow.python.ops import variables

from tvm.contrib import graph_runtime
from tvm._ffi.function import get_global_func
from nnvm.frontend import from_tensorflow
from nnvm.compiler import build
from nnvm.symbol import Symbol
from nnvm.graph import Graph as TVM_Graph

import tensorflow as tf
import numpy as np
import nnvm
import tvm
import json

# Type of nnvm params
Params = Dict[str,Any]

MODEL_PB=join(environ['CWD'], "data/freeze.pb")
MODEL_INPUT='Rcnn_ctcV3/Inputs'

MODEL_OUTPUTS=[
   'Rcnn_ctcV3/expand_conv1/add_1/add'
  ,'Rcnn_ctcV3/expand_conv2/add_7/add'
  ,'Rcnn_ctcV3/expand_conv3/add_13/add'
  ,'Rcnn_ctcV3/expand_conv4/add_17/add'
  ,'Rcnn_ctcV3/conv2d_116/BiasAdd']
MODEL_OUTPUT=MODEL_OUTPUTS[-1]

DEF_LOG_DIR='./_logs'

def get_log_dir(tag:str=""):
  return join(DEF_LOG_DIR,((str(tag)+'-') if len(tag)>0 else '')+strftime("%Y%m%d-%H:%M:%S"))

def fropen()->Tuple[TF_Graph,TF_GraphDef]:
  with tf.Session(graph=tf.Graph()) as sess:
    with FastGFile(MODEL_PB, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name="")
      graphdef=sess.graph.as_graph_def(add_shapes=True)
  return sess.graph, graphdef

def totb(g:TF_Graph):
  """ Export to TensorBoard """
  writer=FileWriter(get_log_dir("freezepb"))
  writer.add_graph(g)

def dump(graphdef:TF_GraphDef, suffix:str='restored')->None:
  """ Export to file """
  with open('graphdef%s' %('_'+suffix if len(suffix)>0 else '',)+'.txt', "w") as f:
    f.write(str(graphdef))

def run():
  assert isfile(MODEL_PB)
  g,gd=fropen()
  print(g)
  # totb(g)
  sym,params=nnvm.frontend.from_tensorflow(gd)
  print(sym)


class Result:
  def __init__(s):
    s.perfs:List[float]=None
    s.last_data:np.array=None
    s.desc:str=''
    s.err=None
    pass

def result_print(r:Result)->None:
  if r.perfs:
    return r.desc+':'+str(np.mean(r.perfs))+'+-'+str(np.std(r.perfs))
  else:
    return r.desc+': no results'

def common_init(init_method, shape, dtype):
  if init_method=='zeros':
    return np.zeros(shape=shape, dtype=dtype)
  elif init_method=='std':
    return np.random.uniform(low=-50, high=51, size=shape).astype(dtype=dtype)
  else:
    raise ValueError("invalid 'init' argument")


def tf_run(iname:str=MODEL_INPUT, oname:str=MODEL_OUTPUT, init_method='std', nwarmup:int=10, nloops:int=100)->Result:
  """ Run the model on tensorflow with zero inputs """
  r=Result()
  r.desc='tf running time'
  try:
    with tf.Session(graph=tf.Graph()) as sess:
      with FastGFile(MODEL_PB, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name="")
      sess.run(variables.global_variables_initializer())
      g=tf.get_default_graph()
      i=g.get_tensor_by_name(iname+':0')
      print("Input node:",type(i), i.name, i.dtype, i)
      o=g.get_tensor_by_name(oname+':0')
      print("Output node:",type(o), o.name, o.dtype, o)

      perfs:List[float]=[]
      for it in range(nwarmup+nloops):
        i_dict={i: common_init(init_method, i.shape, i.dtype.as_numpy_dtype())}

        b=perf_counter()
        o_data=sess.run(o, i_dict)
        e=perf_counter()
        print('tf', e-b)

        if it>=nwarmup:
          perfs.append(e-b)

      r.perfs=perfs
      r.last_data=o_data
  except KeyboardInterrupt:
    raise
  except Exception as e:
    r.err=e

  return r

def tvm_import(opt_level:int=2, iname:str=MODEL_INPUT, oname:str=MODEL_OUTPUT) -> Tuple[TVM_Graph,str,Params,TF_Tensor,TF_Tensor]:
  g,gd=fropen()
  sym,params=nnvm.frontend.from_tensorflow(gd)

  i=g.get_tensor_by_name(iname+':0')
  o=g.get_tensor_by_name(oname+':0')
  i_shape_dict={iname+':0': i.shape.as_list()}
  i_dtype_dict={iname+':0': i.dtype.as_numpy_dtype()}

  with nnvm.compiler.build_config(opt_level=opt_level):
    graph,lib,params=nnvm.compiler.build(graph=sym, target='llvm', shape=i_shape_dict, dtype=i_dtype_dict, params=params)
    # print(graph.ir())
  return graph,lib,params,i,o

def tvm_run(opt_level:int=2, nthreads:int=40, iname=MODEL_INPUT, oname=MODEL_OUTPUT, init_method='std', nwarmup:int=10, nloops:int=100)->Result:
  r=Result()
  r.desc='tvm running time'
  try:
    graph,lib,params,i,o=tvm_import(opt_level, iname, oname)

    get_global_func('runtime.config_threadpool')(1,nthreads)
    m=graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
    print('compiled')

    perfs:List[float]=[]
    for it in range(nwarmup+nloops):
      i_data=common_init(init_method, shape=i.shape.as_list(), dtype=i.dtype.as_numpy_dtype())
      m.set_input(iname, tvm.nd.array(i_data))
      m.set_input(**params)

      b=perf_counter()
      m.run()
      e=perf_counter()
      o_data=m.get_output(0, tvm.nd.empty(o.shape.as_list(), o.dtype.name)).asnumpy()
      print('tvm', e-b)

      if it>=nwarmup:
        perfs.append(e-b)

    r.perfs=perfs
    r.last_data=o_data
  except KeyboardInterrupt:
    raise
  except Exception as e:
    r.err=e

  return r

RUN_ARGS={'init_method':'zeros', 'nwarmup':3, 'nloops':10}


def meanerr():
  print('Running TF')
  rtf=tf_run(**RUN_ARGS)
  print(result_print(rtf))
  print('Running TVM')
  rtvm=tvm_run(**RUN_ARGS)
  print(result_print(rtvm))
  np.testing.assert_allclose(rtvm.last_data, rtf.last_data, rtol=5e-1)
  return rtf,rtvm

# FIXME: tvm doesn't work with intermediate outputs

def dumbsearch():
  res={}
  with open("dumbsearch.json","w") as f:
    for output in [MODEL_OUTPUTS[-1]]:
      args=copy(RUN_ARGS)
      args.update({'oname':output})
      print('TF',args)
      res_tf=tf_run(**args)
      for opt_level in [3,1,2,0]:
        for nthreads in [1,5,20,30,40]:
          args.update({'nthreads':nthreads})
          args.update({'opt_level':opt_level})
          print('TVM',args)
          res_tvm=tvm_run(**args)

          np.testing.assert_allclose(res_tvm.last_data, res_tf.last_data, rtol=1e-1)

          # print(res_tvm.last_data - res_tf.last_data)
          # return -1
          # res.update({str(args):(res_tf.__dict__,result_print(res_tf),res_tvm.__dict__,result_print(res_tvm))})
          # print(res)
          # f.write(json.dumps(res))
  return res



