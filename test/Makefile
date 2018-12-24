# INSTALL: conda install pytorch-cpu=1.0.0 -c pytorch

CXX_FLAGS := $(CXXFLAGS) -std=c++17 -g3 -O0 -pthread -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Wno-unused-function -D_LIBCPP_DEBUG # -D_GLIBCXX_DEBUG
# -ftrapv

INCPATH := $(shell python -c "import torch.utils.cpp_extension as C; print('-isystem' + str.join(' -isystem', C.include_paths()))")

LIBPATH := $(shell python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../')")

USE_CUDA := $(shell python -c "import torch; print(torch.cuda.is_available())")

ifeq ($(USE_CUDA),True)
	TORCH_LIBS=-ltorch -lcaffe2 -lcaffe2_gpu -lc10 -lc10_cuda -lcuda -lnvrtc -lnvToolsExt # -lnccl -lmkldnn -lmkl_rt 
else
	TORCH_LIBS=-ltorch -lcaffe2 -lc10
endif

# add thxx options
INCPATH += -I../include

.PHONY: all

all: test.out
	./test.out

test.out: test_main.o test_net.o test_meta.o
	$(CXX) $(CXX_FLAGS) $^ -o $@  -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS)

test_main.o: test_main.cpp
	$(CXX) $(CXX_FLAGS) -c $^ -o $@ -I../include

test_%.o: test_%.cpp # catch.hpp
	$(CXX) $(CXX_FLAGS) -c $< -I. $(INCPATH) -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS)

# catch.hpp:
# 	wget https://raw.githubusercontent.com/catchorg/Catch2/master/single_include/catch2/catch.hpp

