TVM/NNVM environment setup
==========================

This document describes the working process regarding TVM/NNVM project arranged
in Moscow Research center.

Overall procedure
-----------------

  1. Login to `ws` machine (contact Sergey Mironov mWX579795 in order to get user
     account)

        $ ssh 10.122.85.37

  2. Clone the current repository recursively

        $ git clone --recursive https://http://code.huawei.com/mrc-cbg-opensource/hitvm
        $ cd hitvm

  3. Enter the development environment

     The build hook (see `tvm.nix`) will set up the environment.

        $ nix-shell

     Important tasks such as `make`, `clean` and `test` are wrapped with shell
     functions. See `shell` expression defined in `default.nix` for details.

  4. Build the project using either docker, or directly using `cmake` build
     system, as described in [Official manual](https://docs.tvm.ai/install/index.html).

  5. Use TVM as a library to build and test solutions


Building the TVM/NNVM
=====================

Obtaining TVM repository
------------------------

**Solutions to typical problems may be found
[here](http://code.huawei.com/mrc-cbg-opensource/hitvm-internal/tree/master/mironov/md/README.md)**

Download upstream TVM repository and save to to some `/path/to/tvm` folder.

Running Docker container
------------------------

This environment provides [rundocler.sh](../rundocler.sh) script which can be
used to run interactive docker session for local development.

First, we have to setup `./src/$USER/tvm` link to point to valid TVM repo. This
link will be used to execute Docker rules, defined in TVM project

    $ ln -s ./src/$USER/tvm /path/to/tvm

Next, execute the following script to build and run the docker:

    $ ./rundocker.sh --map-sockets

The `Dockerfile.dev` will be used for building the docker image. One may
adjust/modify/duplicate it as needed.

Finally, source dockerenv.sh to access useful helpers `dmake`, `dclean`, `dtest`
and others:

    (docker) $ . dockerenv.sh
    (docker) $ dmake


Running TVM tasks
=================

Running Python programs using TVM/NNVM
--------------------------------------
Same as native mode, but one should use `ipython3` instead of `ipython`.

    (docker) $ ipython3
    >> from reduction import *
    >> lesson1()


Running C++ programs using TVM/NNVM
-----------------------------------

Should be the same as for native mode.


Running Tensorboard within docker
---------------------------------

Tensorboard is a web-application allowing users to examine the structure and
performance metrics of Tensorflow models.

`rundocker.sh` script maps port equal to `6000 + USER_ID -
1000` to port 6006 of the Host (if the `--map-sockets` option is passed).
The exact values should be printed during container startup:

    *****************************
    Your Jupyter port: XXXX
    Your Tensorboard port: YYYY
    *****************************

Tensorboard may then be run with a helper (one may invoke `type dtensorboard`
to review its code):

    (docker) $ dtensorboard &

Tensorboard will run in a background. After that, browser may be used to
connect to Tensorboard webserver:

    $ chromium http://10.122.85.190:YYYY

The connection should be made to port YYYY of the Host machine, the traffic
should be redirected to port 8008 of the Docker container

`dtensorboard` creates and selects `./_logs` as a directory to search for logs.
Tensorflow models should be set up accordingly to save the logs into this
directory.


Running jupyter-notebook within docker
--------------------------------------

Jupyter notebook may be started by typing `djupyter` command, defined by
`dockerenv.sh` script.

    (docker) $ djupyter

After that, a browser may be started from the remote workstation.

    $ chromium http://10.122.85.190:XXXX

Mind the exact value of XXXX from the output of `./rundocker.sh` script. The
connection should be made to port XXXX of the Host machine, which redirects the
traffic to port 8888 of the Docker


Obtaining core dumps in Docker
------------------------------

Docker containers use host's core patterns file `/proc/sys/kernel/core_pattern`
but don't have `apport` installed, so the default setup doesn't do anything
useful on segmentation fault. If you have a segmentation fault which doesn't
produce a core file: 

1. Run `cat /proc/sys/kernel/core_pattern` (on the host or in the container,
it doesn't matter. 
2. If it contains `apport`, do (on the host!)

        $ echo '/tmp/core.%t.%e.%p' | sudo tee /proc/sys/kernel/core_pattern
        
Reference: https://le.qun.ch/en/blog/core-dump-file-in-docker/
