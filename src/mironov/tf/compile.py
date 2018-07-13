"""
References:
 * https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators
 * https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0
"""

import nnvm as nnvm
import tvm as tvm
import tensorflow as tf

from tf.mnist import Model,load,restore,log,dump
from tensorflow.python.saved_model.tag_constants import TRAINING,SERVING

DEF_PATH:str='data/1531399643'

def export(path:str=DEF_PATH):
  graphdef=restore(path,frozen=True)
  dump(graphdef)
  sym,params=nnvm.frontend.from_tensorflow(graphdef)
  print(sym)
  print(params)

