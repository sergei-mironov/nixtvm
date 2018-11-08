B = _build
N1 = tvm1
S1 = scheduling0

.PHONY: all
all: $(B)/$(N1).gen $(B)/$(S1).gen

$(B)/$(N1).gen: $(N1).cpp
	mkdir -p $(B) || true
	g++ -std=c++14  $(N1).cpp -ltvm -o $(B)/$(N1).gen
	-$(B)/$(N1).gen > $(B)/$(N1).s


$(B)/$(S1).gen: $(S1).cpp
	mkdir -p $(B) || true
	g++ -std=c++14  $(S1).cpp -ltvm -o $(B)/$(S1).gen
	-$(B)/$(S1).gen > $(B)/$(S1).s

