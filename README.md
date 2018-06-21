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

     The build hook (see tvm.nix) will set up environment and change directory to
     `nixtvm/tvm`.

        $ nix-shell tvm-llvm.nix -A shell

  4. Build either using upstream docker, or manually on the current system


Build TVM using docker (preferred)
----------------------------------

Original TVM sources needed proxy patch which is already applied in the current
TVM submodule.

Execute `build` shell function defined by environment hook script (see tvm.nix
for details)

    $ type dmake  # Review the build algorithm
    $ dmake

Note, that in this case the tvm binaries would be linked with LIBC form docker
image so all subsequent runs should be performed using docker.

Currently, the only working command implemented so far is

    $ dtest

More test commands may be extracted using `cat Jenkinsfile | grep docker`


Build natively
--------------

Manual build may be performed directly in TVM source tree. The build procedure
is included in `nmake` shell function (native make)

    $ type nmake  # Review the build algorithm
    $ nmake

