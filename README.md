TVM/NNVM environment setup
==========================


Overall procedure
-----------------

1. Login to ws machine

  $ ssh 10.122.85.37

2. Clone this repository recursively

  $ git clone --recursive https://http://code.huawei.com/mrc-cbg-opensource/nixtvm
  $ cd nixtvm

3. Enter the development environment

  $ nix-shell tvm-llvm.nix -A shell

  The build hook (see tvm.nix) will set up environment and change directory to
  `nixtvm/tvm`.


4. Build either using upstream docker, or manually on the current system


Build TVM using docker (preferred)
----------------------------------

Original TVM sources needed proxy patch which is already applied in the current
TVM submodule.

Execute `build` shell function defined by environment hook script (see tvm.nix
for details)

    $ build

Note, that in this case the tvm binaries would be linked with LIBC form docker
image so all subsequent runs should be performed using docker.

Currently, the only running command is implemented

    $ test

Additional test commands may be extracted using `cat Jenkinsfile | grep docker`


Build manually
--------------

Manual build may be performed directly in TVM source tree.

    $ cd tvm
    $ mkdir build
    $ cd build
    $ cp ../cmake/config.cmale .
    $ vi config.cmake
    $ cmake ..
    $ make -j5

Running tests in this mode is possible but requires some hacks due to PYTHONPATH collisions.









