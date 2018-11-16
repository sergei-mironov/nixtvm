import tvm
import topi
import numpy as np
import time

from exercise.runners import run_tvm,with_tvm

def get_shape(t):
  return [tvm.ir_pass.Simplify(s).value for s in t.shape]

n = 10
A = tvm.placeholder((n,n), name='A')
B = tvm.placeholder((n,n), name='B')

ra = tvm.reduce_axis((0, n), name='k')
C = tvm.compute((n,), lambda i: tvm.sum(A[i,ra]*B[ra,i], axis=ra), name="C")

def run(sched_lambda):
  return run_tvm(0,1,
      { A:1*np.ones(get_shape(A)).astype(np.float32)
      , B:2*np.ones(get_shape(B)).astype(np.float32)
      }, C,
      scheduling=sched_lambda,
      debug=True)

def test0():
  def scheduling(s):
    pass
  r=run(scheduling)
  print(r.last_data)

def test_split():
  def scheduling(s):
    s[C].split(C.op.axis[0], nparts=5)
  r=run(scheduling)
  print(r.last_data)

def test_reorder():
  def scheduling(s):
    s[C].reorder(ra,C.op.axis[0])
  r=run(scheduling)
  print(r.last_data)

def test_cache_write():
  """ Results in `allocate` instruction on caches """
  def scheduling(s):
    CC=s.cache_write(C, 'global')
    ai,=s[C].op.axis
    s[CC].compute_at(s[C],ai)

  r=run(scheduling)
  print(r.last_data)

def test_cache_read():
  """ Cache A when reading C """
  def scheduling(s):
    CR=s.cache_read(A, 'local', C)
    s[CR].compute_at(s[C], s[C].op.axis[0])

  r=run(scheduling)
  print(r.last_data)

def test_vectorize():
  def scheduling(s):
    s[C].vectorize(C.op.axis[0])

  r=run(scheduling)
  print(r.last_data)

