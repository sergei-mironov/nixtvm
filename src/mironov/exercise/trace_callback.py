import tvm
import numpy as np


@tvm.register_func
def my_debug(x):
    print("array=", x.asnumpy())
    return 0


x = tvm.placeholder((4,), name="x", dtype="int32")
xbuffer = tvm.decl_buffer(x.shape, dtype=x.dtype)

y = tvm.compute(x.shape, lambda i: tvm.call_packed("my_debug", xbuffer))
s = tvm.create_schedule(y.op)

print(tvm.lower(s, [x, y], binds={x:xbuffer}, simple_mode=True))

f = tvm.build(s, [xbuffer, y], binds={x:xbuffer})
xnd = tvm.nd.array(np.ones((4,), dtype=x.dtype))
ynd = tvm.nd.array(np.zeros((4,), dtype=y.dtype))

f(xnd, ynd)
