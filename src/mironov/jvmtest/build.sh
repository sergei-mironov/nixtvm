#!/bin/sh

set -e -x

JAR=$TVM/jvm/assembly/linux-x86_64/target/tvm4j-full-linux-x86_64-0.0.1-SNAPSHOT.jar
SO=$TVM/jvm/native/linux-x86_64/target/libtvm4j-linux-x86_64.so

export LD_LIBRARY_PATH=`dirname $SO`:$LD_LIBRARY_PATH
export CLASSPATH=$JAR:.

case "$1" in
  clean)
    rm Vecadd.class
    rm -rf build
    ;;

  *)
    javac Vecadd.java
    python Vecadd.py build
    java Vecadd build
    ;;
esac
