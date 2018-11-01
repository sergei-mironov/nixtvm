This project demonstrates different aspects of training using TVM

LOG
===

#### 01.11.2018
* Created issue http://code.huawei.com/mrc-cbg-opensource/hitvm-internal/issues/7
* Segfault couldn't be reproduced with d579b52
* Added simple LSTM test

#### 12.10.2018
* Implemented conv2d model and training loop, see `autodiff_conv.py`.
* Setting `batch_size` to 10 leads to Segfault
    ```
    Thread 1 "python3" received signal SIGSEGV, Segmentation fault.
    0x00007f6e8b43a624 in ?? ()
    (gdb) bt
    #0 0x00007f6e8b43a624 in ?? ()
    #1 0x00007f6e8b433250 in default_function ()
    #2 0x00007f6e24546471 in tvm::runtime::WrapPackedFunc(int (*)(void*, int*,
        int), std::shared_ptr<tvm::runtime::ModuleNode>
        const&)::{lambda(tvm::runtime::TVMArgs,
        tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs,
        tvm::runtime::TVMRetValue*) const [clone .isra.56] () from
        /workspace/src/mironov/tvm/build-docker/libtvm.so
    #3 0x00007f6e24535cce in TVMFuncCall () from
        /workspace/src/mironov/tvm/build-docker/libtvm.so
    #4 0x00007f6e8b4d2e40 in ffi_call_unix64 () from
        /usr/lib/x86_64-linux-gnu/libffi.so.6
    #5 0x00007f6e8b4d28ab in ffi_call () from /usr/lib/x86_64-linux-gnu/libffi.so.6
    #6 0x00007f6e8b6e619d in _ctypes_callproc () from
        /usr/lib/python3.6/lib-dynload/_ctypes.cpython-36m-x86_64-linux-gnu.so
    #7 0x00007f6e8b6e4df4 in ?? () from
        /usr/lib/python3.6/lib-dynload/_ctypes.cpython-36m-x86_64-linux-gnu.so
    #8 0x00000000005bb24c in _PyObject_FastCallDict ()
    ```

* The Loss doesn't decrease, probably there is a bug

#### 11.10.2018
* Studied convolution basics, compiled a simple program in TVM.
* Implemented a handy wrappers `run_tvm`, `with_tvm`
