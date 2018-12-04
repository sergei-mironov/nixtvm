B = _build
N1 = tvm1
S1 = scheduling0
SM = sigmoid

.PHONY: all clean
all: $(B)/$(N1).gen $(B)/$(S1).gen $(B)/$(SM).gen

$(B)/$(N1).gen: $(N1).cpp
	mkdir -p $(B) || true
	g++ -std=c++14  $(N1).cpp -ltvm -o $(B)/$(N1).gen
	-$(B)/$(N1).gen > $(B)/$(N1).s

$(B)/$(S1).gen: $(S1).cpp
	mkdir -p $(B) || true
	g++ -std=c++14  $(S1).cpp -ltvm -o $(B)/$(S1).gen
	-$(B)/$(S1).gen > $(B)/$(S1).s

$(B)/$(SM).gen: $(SM).cpp
	mkdir -p $(B) || true
	g++ -std=c++14  $(SM).cpp -ltvm -o $(B)/$(SM).gen
	$(B)/$(SM).gen > $(B)/$(SM).s

clean:
	rm -rf $(B)
