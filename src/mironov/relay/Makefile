#!/bin/sh

NM = relay0
SO = _build/$(NM).so

run: $(SO)
	$(CWD)/ipython.sh -c 'from relay.relay0 import *'

$(SO): $(NM).cpp
	mkdir _build 2>/dev/null || true
	g++ -std=c++11 -fPIC $(NM).cpp -ltvm -shared -o $(SO)

