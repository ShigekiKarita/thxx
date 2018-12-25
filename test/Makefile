.PHONY: all

all: test.out test_parser.out
	./test.out
	./test_parser.out --batch_size 4 --use_cuda false --units 1 2 3 4

test_parser.out: test_parser.cpp
	$(CXX) $(CXX_FLAGS) $^ -o $@  -Wl,-rpath,$(LIBPATH) -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS) -L$(CONDA_PREFIX)/lib -L$(LIBPATH) $(INCPATH) $(THXX_LOCAL_INCPATH)

test.out: test_main.o test_net.o test_meta.o
	$(CXX) $(CXX_FLAGS) $^ -o $@  -Wl,-rpath,$(LIBPATH) -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS) -L$(CONDA_PREFIX)/lib -L$(LIBPATH)

test_main.o: test_main.cpp
	$(CXX) $(CXX_FLAGS) -c $^ -o $@ $(INCPATH) $(THXX_LOCAL_INCPATH)

test_%.o: test_%.cpp # catch.hpp
	$(CXX) $(CXX_FLAGS) -c $< $(INCPATH) $(THXX_LOCAL_INCPATH)



