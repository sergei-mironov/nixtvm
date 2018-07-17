This document lists known limitations of TVM/NNVM


* https://discuss.tvm.ai/t/use-tvm-for-shufflenet-onnx-model/427
  For now conv2d in TVM does not support multiple groups

* `Shape` is implemented in a weird way, need to check the consequences
    def _shape():
        def _impl(inputs, attr, params):
            # Result of this operator is prominently used by reshape operator.
            # Just pass the input as it is so that reshape_like can be used there.
            return inputs[0]
        return _impl
