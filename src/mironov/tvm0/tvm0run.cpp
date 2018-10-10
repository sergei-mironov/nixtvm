#include <iostream>
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

using namespace std;
using namespace tvm;

int main(void) {

  tvm::runtime::Module mod =
    tvm::runtime::Module::LoadFromFile(TVM_SO);

  tvm::runtime::PackedFunc f = mod.GetFunction("vecadd");
  CHECK(f != nullptr);

  DLTensor* a;
  DLTensor* b;
  DLTensor* c;
  int ndim = 1;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[1] = {10};

  /* Preparing the input data */
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
    device_type, device_id, &a);
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
    device_type, device_id, &b);

  /* Preparing the placeholder for output data */
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
    device_type, device_id, &c);

  for (int i = 0; i < shape[0]; ++i) {
    static_cast<float*>(a->data)[i] = i;
    static_cast<float*>(b->data)[i] = i*10;
  }

  /* Calling the function */
  f(a,b,c);

  /* Printing the result */
  for (int i = 0; i < shape[0]; ++i) {
    cout << static_cast<float*>(c->data)[i] << " ";
  }

  cout << endl;
  return 0;
}
