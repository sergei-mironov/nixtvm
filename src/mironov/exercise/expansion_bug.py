"""
Ref https://discuss.tvm.ai/t/strange-expansion-of-tensors-bounds/1077
"""
import tvm

def show_lowered(outputs, inputs):
    sout = tvm.create_schedule([o.op for o in outputs])
    mout = tvm.lower(sout, outputs + inputs, simple_mode=True)
    print(mout)

A = tvm.compute((10, 2), lambda i, j: i + j)
show_lowered([A], [])
B = tvm.compute((10,), lambda i: A[i, i + (0 - i)])
show_lowered([B], [])
