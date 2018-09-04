
from os import environ
from os.path import isfile, join
from typing import List,Tuple
from time import strftime

from tensorflow.gfile import FastGFile
from tensorflow.summary import FileWriter
from tensorflow import Graph,GraphDef

import tensorflow as tf
import nnvm

FREEZE_PB=join(environ['CWD'], "data/freeze.pb")
DEF_LOG_DIR='./_logs'

def get_log_dir(tag:str=""):
  return join(DEF_LOG_DIR,((str(tag)+'-') if len(tag)>0 else '')+strftime("%Y%m%d-%H:%M:%S"))

def fropen()->Tuple[Graph,GraphDef]:
  with tf.Session(graph=tf.Graph()) as sess:
    with FastGFile(FREEZE_PB, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name="")
      graphdef=sess.graph.as_graph_def(add_shapes=True)
  return sess.graph, graphdef

def totb(g:Graph):
  """ To TensorBoard """
  writer=FileWriter(get_log_dir("freezepb"))
  writer.add_graph(g)

def dump(graphdef:GraphDef, suffix:str='restored')->None:
  with open('graphdef%s' %('_'+suffix if len(suffix)>0 else '',)+'.txt', "w") as f:
    f.write(str(graphdef))

def run():
  assert isfile(FREEZE_PB)
  g,gd=fropen()
  print(g)
  # totb(g)
  sym,params=nnvm.frontend.from_tensorflow(gd)
  print(sym)

