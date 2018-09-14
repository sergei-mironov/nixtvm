#!/bin/sh

set -e
set -x

NM=vis

g++ -std=c++14  $NM.cpp -ltvm -o gen

./gen >$NM.s 2>/dev/null

g++ -c -o $NM.o $NM.s

g++ -shared -fPIC -o $NM.so $NM.o

# g++ -std=c++11 ${NM}run.cpp -ltvm -o run

# ./run

