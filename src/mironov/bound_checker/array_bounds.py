"""
Array bounds issue illustration
"""
import tvm
import topi
import numpy as np
import time

from exercise.runners import run_tvm,with_tvm


def get_shape(t):
  return [tvm.ir_pass.Simplify(s).value for s in t.shape]


def demo_array1():
  with tvm.build_config(instrument_bound_checkers=True):
    a = tvm.placeholder((10,), name='a')
    b = tvm.placeholder((10,), name='b')
    c = tvm.compute((10,), lambda i: a[i+100050000])

    npy = run_tvm(0,1,
            { a:np.ones(get_shape(a)).astype(np.float32)
            , b:np.ones(get_shape(b)).astype(np.float32)
            },c,debug=True)

    print(npy.last_data)


def demo_array2():
  with tvm.build_config(instrument_bound_checkers=True):
    a = tvm.placeholder((10,), name='a')
    b = tvm.placeholder((10,), name='b')
    c = tvm.compute((10,), lambda i: a[i]+b[i], name='c')
    d = tvm.compute((10,), lambda i: c[i+100500], name='d')

    npy = run_tvm(0,1,
            { a:np.ones(get_shape(a)).astype(np.float32)
            , b:np.ones(get_shape(b)).astype(np.float32)
            },d, debug=True)

    print(npy.last_data)


def demo_array3():
  with tvm.build_config(instrument_bound_checkers=True):
    a = tvm.placeholder((10,), name='a')
    b = tvm.placeholder((10,), name='b')
    c = tvm.compute((10,), lambda i: a[i]+b[i], name='c')
    d = tvm.compute((10,), lambda i: c[i]*c[i+100500], name='d')

    npy = run_tvm(0,1,
            { a:np.ones(get_shape(a)).astype(np.float32)
            , b:np.ones(get_shape(b)).astype(np.float32)
            },d, debug=True)

    print(npy.last_data)

