export CC = gcc
export CXX = g++
export CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result
#export CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -g # debug mode
LDFLAGS = -lgsl -lm -lgslcblas

BIN = emb_model.o sup_model.o

all: main

main: main.cpp $(BIN)
	$(CXX) $(CFLAGS) -o $@ main.cpp $(BIN) -I. $(LDFLAGS)

$(BIN):
	$(CXX) $(CFLAGS) -c -o $@ $(subst .o,.cpp,$@) -I.
	$(CXX) -MM $(CFLAGS) -c $(subst .o,.cpp,$@) -I. > $(subst .o,.d,$@)

-include $(BIN:.o=.d)

.PHONY: clean

clean:
	rm -f main $(BIN) $(BIN:.o=.d) *~
