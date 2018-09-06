#!/bin/sh

NM = relay0

run: $(NM).so
	ipython -i -c "from relay import *"

$(NM).so: $(NM).cpp
	g++ -std=c++11 -fPIC $(NM).cpp -ltvm -shared -o $(NM).so

