import ctypes
from ctypes import CDLL, RTLD_GLOBAL

# libc = CDLL("libc.so.6")
lib = CDLL("relay0.so", RTLD_GLOBAL)
# print('relay0.py')
# print(libc)
lib.test_dispatch()



from tvm import relay

def test1():
  x=relay.Var('x')
  y=relay.exp(x)

  print(type(y), y)
  print(y.__dir__())
  print(y.op.__dir__())

  lib.test_call_node(y.handle)

  print('done')


