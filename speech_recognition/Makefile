# INSTALL: conda install pytorch-cpu=1.0.0 -c pytorch

CXX_FLAGS := -std=c++11
# -g3 -O0 -ftrapv -Wall -Wextra -Wno-unused-function -D_LIBCPP_DEBUG -D_GLIBCXX_DEBUG  -pthread -pipe -fpic

INCPATH := $(shell python -c "import torch.utils.cpp_extension as C; print('-isystem' + str.join(' -I', C.include_paths()))")

LIBPATH := $(shell python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../')")

ESPNET_ROOT := espnet
KALDI_ROOT := $(ESPNET_ROOT)/tools/kaldi
KALDI_FLAGS :=  -isystem$(KALDI_ROOT)/src -isystem$(KALDI_ROOT)/src/util -isystem$(KALDI_ROOT)/src/matrix -isystem$(KALDI_ROOT)/tools/openfst/include -L$(KALDI_ROOT)/src/lib -L$(KALDI_ROOT)/tools/openfst/lib -pthread  -lkaldi-base -lkaldi-util -lkaldi-matrix

train: train.out
	LD_LIBRARY_PATH=$(LIBPATH):$(KALDI_ROOT)/src/lib:$(LD_LIBRARY_PATH) ./train.out

train.out: train.cpp rapidjson
	$(CXX) $(CXX_FLAGS)  $< $(INCPATH) -L$(LIBPATH) $(KALDI_FLAGS) -ltorch -lcaffe2 -lc10 -pthread -o $@

# -D_GLIBCXX_USE_CXX11_ABI=0 

rapidjson:
	wget -O- https://github.com/Tencent/rapidjson/archive/v1.1.0.tar.gz | tar zxvf -
	ln -s `pwd`/rapidjson-1.1.0 rapidjson
