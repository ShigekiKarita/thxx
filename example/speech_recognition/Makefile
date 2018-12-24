# INSTALL: conda install pytorch-cpu=1.0.0 -c pytorch

CXX_FLAGS := -std=c++17 -g3 -O0 -pthread -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Wno-unused-function -D_LIBCPP_DEBUG # -D_GLIBCXX_DEBUG
# -ftrapv

INCPATH := $(shell python -c "import torch.utils.cpp_extension as C; print('-isystem' + str.join(' -I', C.include_paths()))")

LIBPATH := $(shell python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../')")

ESPNET_ROOT := espnet
KALDI_ROOT := $(ESPNET_ROOT)/tools/kaldi
KALDI_FLAGS :=  -isystem$(KALDI_ROOT)/src -isystem$(KALDI_ROOT)/src/util -isystem$(KALDI_ROOT)/src/matrix -isystem$(KALDI_ROOT)/tools/openfst/include -L$(KALDI_ROOT)/src/lib -Wl,-rpath,$(KALDI_ROOT)/src/lib -L$(KALDI_ROOT)/tools/openfst/lib -Wl,-rpath,L$(KALDI_ROOT)/tools/openfst/lib -pthread  -lkaldi-base -lkaldi-util -lkaldi-matrix

USE_CUDA := $(shell python -c "import torch; print(torch.cuda.is_available())")

ifeq ($(USE_CUDA),True)
	TORCH_LIBS=-ltorch -lcaffe2 -lcaffe2_gpu -lc10 -lc10_cuda -lcuda -lnvrtc -lnvToolsExt # -lnccl -lmkldnn -lmkl_rt 
else
	TORCH_LIBS=-ltorch -lcaffe2 -lc10
endif

INCPATH += -I../../include

.PHONY: main test

main: main.out
	./main.out

main.out: main.cpp rapidjson
	$(CXX) $(CXX_FLAGS) $< -o $@ $(INCPATH) -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS)

test: test.out
	./test.out

test.out: test_main.o test_dataset.o
	$(CXX) $(CXX_FLAGS) $^ -o $@  -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS)

test_%.o: test_%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -I. $(INCPATH) -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS)

rapidjson:
	wget -O- https://github.com/Tencent/rapidjson/archive/v1.1.0.tar.gz | tar zxvf -
	ln -s `pwd`/rapidjson-1.1.0/include/rapidjson rapidjson

