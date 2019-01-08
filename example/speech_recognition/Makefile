THXX_ROOT := ../../
include ../../conf.mk

ESPNET_ROOT := $(abspath espnet)
KALDI_ROOT := $(ESPNET_ROOT)/tools/kaldi
KALDI_FLAGS :=  -isystem$(KALDI_ROOT)/src -isystem$(KALDI_ROOT)/src/util -isystem$(KALDI_ROOT)/src/matrix -isystem$(KALDI_ROOT)/tools/openfst/include -L$(KALDI_ROOT)/src/lib -Wl,-rpath,$(KALDI_ROOT)/src/lib -L$(KALDI_ROOT)/tools/openfst/lib -Wl,-rpath,L$(KALDI_ROOT)/tools/openfst/lib -pthread  -lkaldi-base -lkaldi-util -lkaldi-matrix

.PHONY: main test

main: main.out
	./main.out

main.out: main.cpp
	$(CXX) $(CXX_FLAGS) $^ -o $@  -Wl,-rpath,$(LIBPATH) -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS) -L$(CONDA_PREFIX)/lib -L$(LIBPATH) $(INCPATH) -I../../include $(THXX_LOCAL_INCPATH)

decode.out: decode.cpp
	$(CXX) $(CXX_FLAGS) $^ -o $@  -Wl,-rpath,$(LIBPATH) -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS) -L$(CONDA_PREFIX)/lib -L$(LIBPATH) $(INCPATH) -I../../include $(THXX_LOCAL_INCPATH)


test.out: test_main.o test_dataset.o
	$(CXX) $(CXX_FLAGS) $^ -o $@  -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS)

test_%.o: test_%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -I. $(INCPATH) -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS)

