.PHONY: all

all: test_main.out test_parser_main.out
	./test_main.out
	./test_parser_main.out --batch_size 3 --use_cuda false --expdir "/home" --units 100 200
	./test_parser_main.out --json '{"--batch_size": 3, "--use_cuda": false, "--expdir": "/home", "--units": [100, 200]}'

test_parser_main.out: test_parser_main.cpp
	$(CXX) -std=c++11 $^ -o $@  $(INCPATH) $(THXX_LOCAL_INCPATH)

test_main.out: test_main.o test_net.o test_meta.o test_parser.o
	$(CXX) $(CXX_FLAGS) $^ -o $@  -Wl,-rpath,$(LIBPATH) -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS) -L$(CONDA_PREFIX)/lib -L$(LIBPATH)

test_main.o: test_main.cpp
	$(CXX) $(CXX_FLAGS) -c $^ -o $@ $(INCPATH) $(THXX_LOCAL_INCPATH)

test_%.o: test_%.cpp # catch.hpp
	$(CXX) $(CXX_FLAGS) -c $< $(INCPATH) $(THXX_LOCAL_INCPATH)



