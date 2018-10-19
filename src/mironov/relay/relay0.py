import ctypes
from ctypes import CDLL, RTLD_GLOBAL

lib = CDLL("src/mironov/relay/_build/relay0.so", RTLD_GLOBAL)
lib.test_dispatch()

from tvm import relay
from tvm._ffi.function import list_global_func_names, get_global_func
from tvm.relay.expr import debug_print

def test1():
  x=relay.Var('x')
  y=relay.exp(x)

  print('y', type(y), y)
  print('y.__dir__()', y.__dir__())
  print('y.op.__dir__()', y.op.__dir__())
  lib.test_call_node(y.handle)
  print('done')


def test2():
  f=get_global_func("test2")
  print(f)
  x=relay.Var('x')
  y=relay.exp(x)
  f(y)


def test_where():
  """ from test_op_level4.py, added debug printing """
  ib = relay.ir_builder.IRBuilder()
  cond = ib.param("cond", relay.TensorType((3, 4), "float32"))
  x = ib.param("x", relay.TensorType((3, 4), "float32"))
  y = ib.param("y", relay.TensorType((3, 4), "float32"))
  with ib.function(cond, x, y) as func:
      ib.ret(relay.where(cond, x, y))
  ib.ret(func)
  func = relay.ir_pass.infer_type(ib.env, func.to_func())

  print(debug_print(ib.env, func))
  ftype = func.checked_type
  print(ftype)
  assert ftype.ret_type == relay.TensorType((3, 4), "float32")
