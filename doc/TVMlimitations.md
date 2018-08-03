This document describes TVM/NNVM issues, that were not reported to upstream yet.


 * https://discuss.tvm.ai/t/use-tvm-for-shufflenet-onnx-model/427
   For now conv2d in TVM does not support multiple groups

 * `Shape` is implemented in a weird way, need to check the consequences
     def _shape():
         def _impl(inputs, attr, params):
             # Result of this operator is prominently used by reshape operator.
             # Just pass the input as it is so that reshape_like can be used there.
             return inputs[0]
         return _impl

 * NNVM uses Shapes `[1]` to represent scalar values, while it should mean
   vectors of size one. For some reason, they dont use empty shapes for that

 * Target architecture: host and host\_device labels are not first-class

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

 * Axis parameter convention differs from numpy in corner cases.

 * In NNVM, axis=None is not supported by the parsers (reduce operations)

 * Corner-cases of operations are not covered by tests. For example, one
   should test vector operations on scalars, min/max operations on equal
   values, etc.

 * Duplicated logic in Python and C++ in various places:

    1. Lower function, containing list of passes

       - `tvm/python/tvm/build_module.py` vs
       - `tvm/src/codegen/build_module.cc`

    2. TOPI operation porimitives

       - `tvm/topi/python/topi/reduction.py` vs
       - `tvm/topi/include/topi/reduction.h`

    3. Strided slice primitive
       - `nnvm/python/nnvm/frontend.md` vs
       - NNVM: strided_slice vs
       - TOPI

 * Exporting models from TF/ONNX/Others should reaaly be done on the TVM level,
   but not on the NNVM level

 * Variable-length shape support is missing from TVM, at least when exporting
   Tensorflow models. E.g. `ReshapeInferShape` doesn't support input shapes of
   (-1,x) form

 * `TShape::Size()` returns `size_t`, while its fields are all `int64_t`.



