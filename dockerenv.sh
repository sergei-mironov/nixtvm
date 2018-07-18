#!/bin/sh
# This file is intended to be sourced from Docker container's interactive shell

CWD=`pwd`
mkdir $HOME/.ipython-profile 2>/dev/null || true
cat >$HOME/.ipython-profile/ipython_config.py <<EOF
print("Enabling autoreload")
c = get_config()
c.InteractiveShellApp.exec_lines = []
c.InteractiveShellApp.exec_lines.append('%load_ext autoreload')
c.InteractiveShellApp.exec_lines.append('%autoreload 2')
EOF

if test -n "$DISPLAY"; then
  alias ipython3='ipython3 --profile-dir=$HOME/.ipython-profile'
fi

# User Aliasing
case $USER in
  grwlf) USER=mironov ;;
  *) ;;
esac

export TVM=$CWD/tvm
export PYTHONPATH="$CWD/src/$USER:$TVM/python:$TVM/topi/python:$TVM/nnvm/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$TVM/build-docker:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$TVM/include:$TVM/dmlc-core/include:$TVM/HalideIR/src:$TVM/dlpack/include:$TVM/topi/include:$TVM/nnvm/include"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
export LIBRARY_PATH=$TVM/build-docker

cdtvm() { cd $TVM ; }
cdex() { cd $TVM/nnvm/examples; }

dclean() {(
  cdtvm
  cd build-docker
  make clean
  rm CMakeCache.txt
  rm -rf CMakeFiles
)}

dmake() {(
  cdtvm
  mkdir build-docker 2>/dev/null
  if ! test -f $TVM/build-docker/config.cmake ; then
    cat >$TVM/build-docker/config.cmake <<EOF
      set(USE_CUDA OFF)
      set(USE_ROCM OFF)
      set(USE_OPENCL OFF)
      set(USE_METAL OFF)
      set(USE_VULKAN OFF)
      set(USE_OPENGL OFF)
      set(USE_RPC ON)
      set(USE_GRAPH_RUNTIME ON)
      set(USE_GRAPH_RUNTIME_DEBUG OFF)
      set(USE_LLVM ON)
      set(USE_BLAS openblas)
      set(USE_RANDOM OFF)
      set(USE_NNPACK OFF)
      set(USE_CUDNN OFF)
      set(USE_CUBLAS OFF)
      set(USE_MIOPEN OFF)
      set(USE_MPS OFF)
      set(USE_ROCBLAS OFF)
      set(USE_SORT ON)
EOF
    echo "Generating 'build-docker/config.cmake'"
  else
    echo "Re-using config 'build-docker/config.cmake'"
  fi
  ./tests/scripts/task_build.sh build-docker -j6 "$@"
  ln -f -s build-docker build # FIXME: Python uses 'build' name
)}

alias build="dmake"

dtest() {(
  cdtvm
  ./tests/scripts/task_python_nnvm.sh
)}

djupyter() {(
  jupyter-notebook --ip 0.0.0.0 --port 8888 --NotebookApp.token='' --NotebookApp.password='' "$@" --no-browser
)}

dtensorboard() {(
  mkdir $CWD/_logs 2>/dev/null
  tensorboard --logdir=$CWD/_logs "$@"
)}

ipython() {(
  # FIXME: One should figure out how to setup non-file backend on Ubuntu
  ipython3 -i -c "import matplotlib; matplotlib.use('agg'); import matplotlib.pyplot; matplotlib.pyplot.ioff()" "$@"
)}

cdc() {(
  cd $CWD
)}


