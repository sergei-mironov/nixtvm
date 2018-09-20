"""
Ref.: http://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/
"""
import tvm
import topi
import numpy as np
from nnvm.testing.check_computation import check_numerical_grads

def matmul(a,b):
  # from tvm.contrib import cblas
  # return cblas.matmul(a, b)
  return topi.cpp.nn.matmul(a,b)

def sigmoid(x):
  # return topi.sigmoid(x)
  return x

def tanh(x):
  # return topi.tanh(x)
  return x

def lstm_cell(bg,Ug,Vg, bi,Ui,Vi, bf,Uf,Vf, bo,Uo,Vo, x,s,h):
  """
  Parameters:
  ----------
  b_,U_,V_ : Biases and weights (gate-,input-,forget- and output-)
  x        : Data input
  s        : Internal state
  h        : Prev. iteration output

  Return:
  ------
  st2 : New state
  h2  : New LSTM output
  """

  g=tanh(bg+matmul(x,Ug)+matmul(h,Vg))
  i=sigmoid(bi+matmul(x,Ui)+matmul(h,Vi))
  f=sigmoid(bf+matmul(x,Uf)+matmul(h,Vf))
  s2=s*f+g*i
  o=sigmoid(bo+matmul(x,Uo)+matmul(h,Vo))
  h2=tanh(s)*o

  return (s2,h2)


def lstm_driver():
  bg=tvm.placeholder((10,1),   name='bg')
  Ug=tvm.placeholder((10,10), name='Ug')
  Vg=tvm.placeholder((10,10), name='Vg')

  bi=tvm.placeholder((10,1),   name='bi')
  Ui=tvm.placeholder((10,10), name='Ui')
  Vi=tvm.placeholder((10,10), name='Vi')

  bf=tvm.placeholder((10,1),   name='bf')
  Uf=tvm.placeholder((10,10), name='Uf')
  Vf=tvm.placeholder((10,10), name='Vf')

  bo=tvm.placeholder((10,1),   name='bo')
  Uo=tvm.placeholder((10,10), name='Uo')
  Vo=tvm.placeholder((10,10), name='Vo')

  x=tvm.placeholder((10,1),   name='x')
  s=tvm.placeholder((10,1),   name='s')
  h=tvm.placeholder((10,1),   name='h')

  s2,h2=lstm_cell(bg,Ug,Vg, bi,Ui,Vi, bf,Uf,Vf, bo,Uo,Vo, x,s,h)
  print(s2,h2)

  sout=tvm.create_schedule([s2.op,h2.op])
  mout=tvm.build(sout,[s2,h2,bg,Ug,Vg, bi,Ui,Vi, bf,Uf,Vf, bo,Uo,Vo, x,s,h],'llvm')
  print(mout)

  ones = topi.full_like(s2, 1.0)
  jac = tvm.ir_pass.JacobianRecursive(s2, x, ones)
  print(jac)

