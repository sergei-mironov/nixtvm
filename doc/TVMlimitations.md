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

 * Target architecture: host and host\_device labels are not first-class

 * Duplicated implementation in Python and C++

 * The following minimal program segfaults:

    #include <tvm/tvm.h>
    #include <tvm/operation.h>
    #include <tvm/tensor.h>

    using namespace std;
    using namespace tvm;

    int main()
    {
      Var row("row"), col("col");
      Tensor A = placeholder({row, col}, Float(32), "A");
      Tensor B = placeholder({row, col}, Float(32), "B");

      IterVar ra = reduce_axis(Range{0, col}, "ra");

      auto C = compute({row}, [&](Var i, Var j) {  // <--- Var j is redundant here

          return sum(max(A[i][ra]+0, B[i][ra]+0), {ra});

        }, "C");

      return 0;
    }
