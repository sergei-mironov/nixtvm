import tvm
import topi
import numpy as np
import time
from nnvm import sym
from exercise.runners import run_tvm,with_tvm,with_nnvm

def demo_softmax_debug():
  x = np.array([[0,0],[0, 1],[1,0],[1,1]]).astype(np.float32)
  print( with_nnvm(0,1,[x], lambda a: sym.relu(sym.softmax(a)), debug=True))

