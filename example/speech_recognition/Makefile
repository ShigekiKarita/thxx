# INSTALL: conda install pytorch-cpu=1.0.0 -c pytorch

# CXX_FLAGS := -std=c++17 -g3 -O0 -pthread -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Wno-unused-function -D_LIBCPP_DEBUG # -D_GLIBCXX_DEBUG
# -ftrapv

# INCPATH := $(shell python -c "import torch.utils.cpp_extension as C; print('-isystem' + str.join(' -I', C.include_paths()))")

# LIBPATH := $(shell python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../')")

ESPNET_ROOT := espnet
KALDI_ROOT := $(ESPNET_ROOT)/tools/kaldi
KALDI_FLAGS :=  -isystem$(KALDI_ROOT)/src -isystem$(KALDI_ROOT)/src/util -isystem$(KALDI_ROOT)/src/matrix -isystem$(KALDI_ROOT)/tools/openfst/include -L$(KALDI_ROOT)/src/lib -Wl,-rpath,$(KALDI_ROOT)/src/lib -L$(KALDI_ROOT)/tools/openfst/lib -Wl,-rpath,L$(KALDI_ROOT)/tools/openfst/lib -pthread  -lkaldi-base -lkaldi-util -lkaldi-matrix

.PHONY: main test

main: main.out
	./main.out

main.out: main.cpp
	$(CXX) $(CXX_FLAGS) $^ -o $@  -Wl,-rpath,$(LIBPATH) -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS) -L$(CONDA_PREFIX)/lib -L$(LIBPATH) $(INCPATH) -I../../include $(THXX_LOCAL_INCPATH)

test.out: test_main.o test_dataset.o
	$(CXX) $(CXX_FLAGS) $^ -o $@  -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS)

test_%.o: test_%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -I. $(INCPATH) -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS)

