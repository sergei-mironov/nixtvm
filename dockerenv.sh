#!/bin/sh
# This file is intended to be sourced from Docker container's interactive shell

export CWD=`pwd`
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
export PYTHONPATH="$CWD/src/$USER:$TVM/python:$TVM/topi/python:$TVM/nnvm/python:$TVM/nnvm/tests/python:$PYTHONPATH"
export LD_LIBRARY_PATH="$TVM/build-docker:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$TVM/include:$TVM/dmlc-core/include:$TVM/HalideIR/src:$TVM/dlpack/include:$TVM/topi/include:$TVM/nnvm/include"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
export LIBRARY_PATH=$TVM/build-docker

if test -n "$https_proxy" ; then
export PROXY_HOST=`echo $https_proxy | sed 's@.*//\(.*\):.*@\1@'`
export PROXY_PORT=`echo $https_proxy | sed 's@.*//.*:\(.*\)@\1@'`
cat >$HOME/.gradle/gradle.properties <<EOF
systemProp.http.proxyHost=$PROXY_HOST
systemProp.http.proxyPort=$PROXY_PORT
systemProp.http.nonProxyHosts=localhost|127.0.0.1|10.10.1.*
systemProp.https.proxyHost=$PROXY_HOST
systemProp.https.proxyPort=$PROXY_PORT
systemProp.https.nonProxyHosts=localhost|127.0.0.1|10.10.1.*

systemProp.javax.net.ssl.trustStore=/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/security/cacerts
systemProp.javax.net.ssl.trustStorePassword=changeit
EOF

cat >$HOME/.android/androidtool.cfg <<EOF
http.proxyHost=$PROXY_HOST
http.proxyPort=$PROXY_PORT
https.proxyHost=$PROXY_HOST
https.proxyPort=$PROXY_PORT
EOF

export GRADLE_OPTS="-Dorg.gradle.daemon=false -Dandroid.builder.sdkDownload=true -Dorg.gradle.jvmargs=-Xmx2048M -Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT -Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT"
export HTTPS_PROXY=$https_proxy
export HTTP_PROXY=$http_proxy
export _JAVA_OPTIONS="-Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT -Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT"
fi

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
  cp $TVM/cmake/config.cmake $TVM/build-docker/config.cmake
  sed -i 's/USE_LLVM OFF/USE_LLVM ON/g' $TVM/build-docker/config.cmake
  ./tests/scripts/task_build.sh build-docker "$@" -j6
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


