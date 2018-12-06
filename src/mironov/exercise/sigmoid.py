import tvm
import topi
import numpy as np
import time

from exercise.runners import run_tvm,with_tvm

def demo_sigmoid():
  x = np.array([[0,0,1,0,0],[3,1,2,0,0],[0,0,0,1,2]]).astype(np.float32)
  r = with_tvm(0,1,[x],lambda a: topi.sigmoid(a))
  print(r)
