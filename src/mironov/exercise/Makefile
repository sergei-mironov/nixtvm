#!/bin/sh

SO = $(B)/$(NM).so

$(B):
	touch $(B)

N1 = tvm1

_build/$(N1).gen: $(N1).cpp
	mkdir -p _build || true
	g++ -std=c++14  $(N1).cpp -ltvm -o _build/$(N1).gen
	_build/$(N1).gen > _build/$(N1).s


all: $(N1)
