import ctypes
from ctypes import CDLL, RTLD_GLOBAL

lib = CDLL("src/mironov/relay/_build/relay0.so", RTLD_GLOBAL)
lib.test_dispatch()

from tvm import relay
from tvm._ffi.function import list_global_func_names, get_global_func

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
  cond = relay.var("cond", relay.TensorType((3, 4), "float32"))
  x = relay.var("x", relay.TensorType((3, 4), "float32"))
  y = relay.var("y", relay.TensorType((3, 4), "float32"))
  z = relay.where(cond, x, y)
  print(z.astext())



def test_let_if_scope():
  x = relay.var("x", "float32")
  y = relay.var("y", "float32")
  cond = relay.var("cond", "bool")
  sb = relay.ScopeBuilder()
  with sb.if_scope(cond):
    v1 = sb.let("v", relay.const(1, "float32"))
    v2 = sb.let("v", x)
    sb.ret(relay.subtract(v1, v2))
  with sb.else_scope():
    v3 = relay.var("v")
    let2 = relay.Let(v3, y, v3)
    sb.ret(relay.add(let2, let2))
  result = sb.get()
  f = relay.Function([x, y, cond], result)
  print(f.astext())
