#!/bin/sh

set -e
set -x

NM=argmax0

g++ -std=c++11  $NM.cpp -ltvm -o gen

./gen >$NM.s 2>/dev/null

g++ -c -o $NM.o $NM.s

g++ -shared -fPIC -o $NM.so $NM.o

