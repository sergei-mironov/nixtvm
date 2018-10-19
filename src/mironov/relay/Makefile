#!/bin/sh

NM = relay0

run: $(NM).so
	$(CWD)/ipython.sh -c 'from relay.relay0 import *'

$(NM).so: $(NM).cpp
	mkdir _build 2>/dev/null || true
	g++ -std=c++11 -fPIC $(NM).cpp -ltvm -shared -o _build/$(NM).so

