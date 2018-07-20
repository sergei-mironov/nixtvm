#include <iostream>
#include <cstdio>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

using namespace std;
using namespace tvm;

int main(void) {

  DLTensor* a;
  int ndim = 1;
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int64_t shape[1] = {10};
  TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &a);

  for (int i = 0; i < shape[0]; ++i) {
    static_cast<float*>(a->data)[i] = i;
  }

  DLTensor* c;
  int odtype_code = kDLInt;
  int64_t oshape[1] = {1};
  TVMArrayAlloc(oshape, ndim, odtype_code, dtype_bits, dtype_lanes,
                device_type, device_id, &c);

  tvm::runtime::Module mod = tvm::runtime::Module::LoadFromFile("argmax0.so");
  tvm::runtime::PackedFunc f = mod.GetFunction("vecargmax");
  CHECK(f != nullptr);

  f(a,c);

  for (int i = 0; i < oshape[0]; ++i) {
    cout << static_cast<int32_t*>(c->data)[i] << " ";
  }
  cout << endl;
  return 0;
}
