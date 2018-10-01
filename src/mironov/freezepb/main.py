import tensorflow as tf
import numpy as np
import nnvm
import tvm
import json

from os import environ
from os.path import isfile, join
from typing import List,Tuple,Dict,Any
from time import strftime, perf_counter
from copy import copy
from warnings import warn

from tensorflow import Tensor as TF_Tensor
from tensorflow.gfile import FastGFile
from tensorflow.summary import FileWriter
from tensorflow import Graph as TF_Graph, GraphDef as TF_GraphDef
from tensorflow.python.ops import variables

from tvm.contrib import graph_runtime
from tvm._ffi.function import get_global_func
from nnvm.staging import stage_tensorflow
from nnvm.frontend import from_tensorflow
from nnvm.compiler import build
from nnvm.symbol import Symbol
from nnvm.graph import Graph as TVM_Graph

from freezepb.model import staged_model

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


def common_init(init_method, shape, dtype):
  if init_method=='zeros':
    return np.zeros(shape=shape, dtype=dtype)
  elif init_method=='std':
    return np.random.uniform(low=-50, high=51, size=shape).astype(dtype=dtype)
  else:
    raise ValueError("invalid 'init' argument")

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
  sym,params=from_tensorflow(gd)
  print(sym)


class Result:
  def __init__(s):
    s.perfs:List[float]=None
    s.perf_mean:float=0.0
    s.perf_std:float=0.0
    s.last_data:np.array=None
    s.desc:str=''
    s.err=None
    s.mismatch=None
    pass
  def set_perfs(s,perfs):
    s.perfs=perfs
    s.perf_mean=np.mean(perfs)
    s.perf_std=np.std(perfs)


def result_print(r:Result)->None:
  if r.perfs:
    return r.desc+':'+str(r.perf_mean)+'+-'+str(r.perf_std)
  else:
    return r.desc+': no results'


def tf_run(iname:str=MODEL_INPUT, oname:str=MODEL_OUTPUT, init_method='std', nwarmup:int=10, nloops:int=100, **kwargs)->Result:
  """ Run the model on tensorflow with zero inputs """
  print("Warning: unused args:", kwargs) if kwargs != {} else None
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

      r.set_perfs(perfs)
      r.last_data=o_data
  except KeyboardInterrupt:
    raise
  except Exception as e:
    r.err=e
  return r

def tvm_stage_test():
  g,gd=fropen()
  print(g)
  dump(gd,'staged')
  sym,params=stage_tensorflow(gd,"out.py")
  return sym,params

def tvm_import(opt_level:int=2, iname:str=MODEL_INPUT, oname:str=MODEL_OUTPUT) -> Tuple[TVM_Graph,str,Params,TF_Tensor,TF_Tensor]:
  g,gd=fropen()
  sym,params=from_tensorflow(gd)

  i=g.get_tensor_by_name(iname+':0')
  o=g.get_tensor_by_name(oname+':0')
  i_shape_dict={iname+':0': i.shape.as_list()}
  i_dtype_dict={iname+':0': i.dtype.as_numpy_dtype()}

  with nnvm.compiler.build_config(opt_level=opt_level):
    graph,lib,params=nnvm.compiler.build(graph=sym, target='llvm', shape=i_shape_dict, dtype=i_dtype_dict, params=params)
    # print(graph.ir())
  return graph,lib,params,i,o



def tvm_run(opt_level:int=2, nthreads:int=None, iname=MODEL_INPUT, oname=MODEL_OUTPUT, init_method='std', nwarmup:int=10, nloops:int=100)->Result:
  """ Compile Model using TVM and run it multiple times """
  r=Result()
  r.desc='tvm running time'
  try:
    graph,lib,params,i,o=tvm_import(opt_level, iname, oname)

    if nthreads is None:
      nthread=20

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

    r.set_perfs(perfs)
    r.last_data=o_data
  except KeyboardInterrupt:
    raise
  except Exception as e:
    r.err=e

  return r


def tvmS_run(nthreads:int=None, iname:str=MODEL_INPUT, oname:str=MODEL_OUTPUT, init_method='std', nwarmup:int=0, nloops:int=1, **kwargs)->Result:
  """ Staged TVM model runner """
  r=Result()
  print("Warning: unused args:", kwargs) if kwargs != {} else None
  try:
    g,gd=fropen()
    _,params=from_tensorflow(gd) # We still need from_tensorflow to get the parameters
    mo,savepoints=staged_model()
    nnvm_graph=nnvm.graph.create(savepoints[oname])
    print('synthesized')

    i=g.get_tensor_by_name(iname+':0')
    o=g.get_tensor_by_name(oname+':0')
    i_shape_dict={iname+':0': i.shape.as_list()}
    i_dtype_dict={iname+':0': i.dtype.as_numpy_dtype()}

    with nnvm.compiler.build_config(opt_level=2):
      graph,lib,params=nnvm.compiler.build(graph=nnvm_graph, target='llvm', shape=i_shape_dict, dtype=i_dtype_dict, params=params)
      # print(graph.ir())

    if nthreads is None:
      nthreads=20

    get_global_func('runtime.config_threadpool')(1,nthreads)
    m=graph_runtime.create(graph,lib,ctx=tvm.cpu(0))
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
      print('tvms', e-b)

      if it>=nwarmup:
        perfs.append(e-b)

    r.set_perfs(perfs)
    r.last_data=o_data
  except KeyboardInterrupt:
    raise
  except Exception as e:
    warn('exception: '+str(e))
    r.err=e
  return r


def correctness()->Result:
  run_args={'init_method':'zeros', 'nwarmup':0, 'nloops':1}
  print('Running TF')
  rtf=tf_run(**run_args)
  print('Running Staged TVM')
  rtvms=tvmS_run(**run_args)
  np.testing.assert_allclose(rtvms.last_data, rtvms.last_data, rtol=5e-1)
  print('Running TVM')
  rtvm=tvm_run(**run_args)
  np.testing.assert_allclose(rtvm.last_data, rtf.last_data, rtol=5e-1)
  return rtvms




def meanerr()->Tuple[Result,Result]:
  run_args={'init_method':'zeros', 'nwarmup':3, 'nloops':10}
  print('Running TF')
  rtf=tf_run(**run_args)
  print(result_print(rtf))
  print('Running TVM')
  rtvm=tvm_run(**run_args)
  print(result_print(rtvm))
  np.testing.assert_allclose(rtvm.last_data, rtf.last_data, rtol=1e-1)
  return rtf,rtvm

# FIXME: tvm doesn't work with intermediate outputs

def dumbsearch()->dict:
  run_args={'init_method':'zeros', 'nwarmup':3, 'nloops':10}
  res={}
  with open("dumbsearch.json","w") as f:
    for output in [MODEL_OUTPUTS[-1]]:
      args=copy(run_args)
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



def partsearch()->dict:
  run_args={'init_method':'zeros', 'nwarmup':3, 'nloops':30}
  res={}
  i=0

  def cleanup(typ,x,args):
    x2=copy(x)
    x2.typ=typ
    x2.last_data=None
    x2.perfs=None
    x2.args=copy(args)
    return x2.__dict__

  for output in MODEL_OUTPUTS:

    args=copy(run_args)
    args.update({'oname':output})
    print('TF',args)
    res_tf=tf_run(**args)
    res[i]=cleanup('TF',res_tf,args)
    i+=1

    for nthreads in [3,None,10,20,40]:

      args.update({'nthreads':nthreads})

      for opt_level in [3,2,1]:

        args.update({'opt_level':opt_level})
        print('TVMS',args)
        res_tvms=tvmS_run(**args)

        if not np.isclose(res_tvms.last_data, res_tf.last_data, rtol=1e-1, atol=1e-5).any():
          res_tf.mismatch=True
          res_tvms.mismatch=True

        res[i]=cleanup('TVMS',res_tvms,args)
        i+=1

        with open("partsearch.json","w") as f:
          json.dump(res,f,indent=4)

  return res

def partsearch_restore_plot(fname):
  import matplotlib.pyplot as plt

  with open(fname,"r") as f:
    r=json.load(f)

  def extract(pats):
    ks=[k for k in r.keys() if all([pat in k for pat in pats])]
    results=[]
    for k in ks:
      nt='N'+str(r[k]['args']['nthreads'])
      results.append((r[k]['perf_mean'],r[k]['perf_std'],nt))
    return sorted(results, key=lambda x:x[0])[0]

  keys=r.keys()
  idx=np.ndarray(shape=(len(MODEL_OUTPUTS),))
  tf_m=np.ndarray(shape=(len(MODEL_OUTPUTS),))
  tf_s=np.ndarray(shape=(len(MODEL_OUTPUTS),))
  tvms_m=np.ndarray(shape=(len(MODEL_OUTPUTS),))
  tvms_s=np.ndarray(shape=(len(MODEL_OUTPUTS),))
  tf_notes=[]; tvms_notes=[]
  for i,output in enumerate(MODEL_OUTPUTS):
    idx[i]=i
    m,s,n=extract(['TF',output])
    tf_m[i]=m
    tf_s[i]=s
    tf_notes.append(n)
    m,s,n=extract(['TVMS',output])
    tvms_m[i]=m
    tvms_s[i]=s
    tvms_notes.append(n)

  fig,ax=plt.subplots()
  def plot_ci(x,mean,std,shade,notes,**kwargs):
    plt.fill_between(x,mean+std,mean-std,color=shade,alpha=0.3)
    plt.plot(x,mean,**kwargs)
    for i,txt in enumerate(notes):
      ax.annotate(txt,(x[i],meanp[i]))

  plot_ci(idx,tf_m,tf_s,tf_notes,shade='orange',color='orange',label='TF')
  plot_ci(idx,tvms_m,tvms_s,tvms_notes,shade='skyblue',color='skyblue',label='TVM')
  plt.legend()
  plt.show()
  plt.savefig('partsearch.png')
  # tf_oindex.append(i)
  # tf_times.append(min([r]))
  return

