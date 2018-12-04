Basic workflow of TVM
=====================

This document describes the basic stages of TVM workflow. In contrast to
[Deploy](https://docs.tvm.ai/deploy/cpp_deploy.html) example, all the
components are implemented in C++ language.

TVM is a domain specific language for building computation graphs.
TVM uses Halide's workflow as a basis, but extends it with support for Tensors,
Tensor-based optimisations and multi-platform code-generation.

The basic workflow of including the TVM model into a project may be split in the
following stages:

 1. Defining a model in C++ or Python
 2. Compiling the model into Dynamic Library
 3. Integrating the library with the main project
 4. Using the library

### Defining a model

TVM functionality in model defining may be demonstrated by following C++ program, which
builds a computation for adding 2 vectors:

    // File: gen.cc
    #include <random>
    #include <iomanip>
    #include <array>
    #include <exception>

    #include <tvm/tvm.h>
    #include <tvm/operation.h>
    #include <tvm/tensor.h>
    #include <tvm/build_module.h>
    #include <topi/broadcast.h>

    using namespace std;

    int main()
    {
      /* Stage 1 */
      auto n = tvm::var("n");
      tvm::Array<tvm::Expr> shape = {n};
      tvm::Tensor A = tvm::placeholder(shape, tvm::Float(32), "A");
      tvm::Tensor B = tvm::placeholder(shape, tvm::Float(32), "B");
      tvm::Tensor C = tvm::compute(shape, tvm::FCompute([=](auto i){ return A(i) + B(i); } )) ;

      /* Stage 2 */
      tvm::Schedule s = tvm::create_schedule({C->op});

      /* Stage 3 */
      tvm::BuildConfig config = tvm::build_config();
      std::unordered_map<tvm::Tensor, tvm::Buffer> binds;
      auto args = tvm::Array<tvm::Tensor>({A, B, C});
      auto lowered = tvm::lower(s, args, "vecadd", binds, config);
      cerr << lowered << endl;

      /* Stage 4 */
      auto target = tvm::Target::create("llvm");
      auto target_host = tvm::Target::create("llvm");
      tvm::runtime::Module mod = tvm::build(lowered, target, target_host, config);
      cout << mod->GetSource() << endl;

      return 0;
    }

Below the basic stages are explained in more details.

##### Stage 1. Declaring the placeholders for input.

`tvm::Tensor` objects are created, every object is assigned to a symbolic name.
The `C` Tensor is defined to be a result of addition of `A` and `B` tensors.

##### Stage 2. Compiling the model

`Schedule` object is created which may be a subject to optimisations. This
example just uses the default.

##### Stage 3. Obtaining the lowered code

At this stage we obtain the TVM IR code which is expected to be passed to
target platform code generator

##### Stage 4. Obtaining the LLVM source

Here we generate LLVM IR code in its text form. The code is printed to the
standard output.

### Compiling the model

In order to compile to obtain the dynamic library, we should first obtain the
generator. In order to do this, we compile the `gen.cc` file above using `gcc`.

    $ g++ -std=c++11  gen.cc -ltvm -o gen

Next, we run the generator to obtain the LLVM IR code on its standard output

    $ ./gen > vecadd.ll

LLVM allows us to compile the IR into native hardware code, in this case, the
X86 assembly.

    $ llc vecadd.ll -o vecadd.s

Next we compile assembly code of our `vecadd` function into shared library,
using common GCC toolset

    $ gcc -c -o vecadd.o vecadd.s
    $ gcc -shared -fPIC -o vecadd.so vecadd.o

The resulting `vecadd.so` library defines `vecadd` function taking three
arguments: two input operands and one output. Next we proceed to inserting this
library into our test project.

### Using the library in third-party project


In order to use our `vecadd` function we need to load the library and prepare
input data. This operations is demonstrated by the following C++ program:


    // main.cc
    #include <iostream>
    #include <cstdio>
    #include <dlpack/dlpack.h>
    #include <tvm/runtime/module.h>
    #include <tvm/runtime/registry.h>
    #include <tvm/runtime/packed_func.h>

    using namespace std;
    using namespace tvm;

    int main(void) {

      /* Stage 1 */
      tvm::runtime::Module mod =
          tvm::runtime::Module::LoadFromFile("vecadd.so");

      tvm::runtime::PackedFunc f = mod.GetFunction("vecadd");
      CHECK(f != nullptr);

      /* Stage 2 */
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
      TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &a);
      TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &b);
      TVMArrayAlloc(shape, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &c);
      for (int i = 0; i < shape[0]; ++i) {
        static_cast<float*>(a->data)[i] = i;
        static_cast<float*>(b->data)[i] = i*10;
      }

      /* Stage 3 */
      f(a,b,c);

      for (int i = 0; i < shape[0]; ++i) {
        cout << static_cast<float*>(c->data)[i] << " ";
      }
      cout << endl;
      return 0;
    }


We link this program using GCC compiler with a following command:

    $ g++ -std=c+=11 main.cc -ltvm_runtime -o main

##### Stage 1. Loading the library

The `vecadd.so` library is loaded from the current directory. After that, the
`vecadd` function is accessed.

##### Stage 2. Preparing the data

Preparation of the input data includes allocation of two input arrays and one
output array. Input arrays are Initialized.

##### Stage 3. Executing the function

Here we pass pointers to input and output arrays to `vecadd` function. After
the computation is performed, the output array contain the result of addition
of the input arrays.


### Using the library

To demonstrate how to use the library we run the computation requested:

    $ ./main
    0 11 22 33 44 55 66 77 88 99

The result of addition is directed to the standard output.



