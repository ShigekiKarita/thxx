.PHONY: all

all: test.out
	./test.out

test.out: test_main.o test_net.o test_meta.o
	$(CXX) $(CXX_FLAGS) $^ -o $@  -Wl,-rpath,$(LIBPATH) -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS) -L$(CONDA_PREFIX)/lib -L$(LIBPATH)

test_main.o: test_main.cpp
	$(CXX) $(CXX_FLAGS) -c $^ -o $@ -I../include

test_%.o: test_%.cpp # catch.hpp
	$(CXX) $(CXX_FLAGS) -c $< $(INCPATH) -I../include
# -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS)

# catch.hpp:
# 	wget https://raw.githubusercontent.com/catchorg/Catch2/master/single_include/catch2/catch.hpp

