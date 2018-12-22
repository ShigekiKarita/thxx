# INSTALL: conda install pytorch-cpu=1.0.0 -c pytorch

INCPATH := $(shell python -c "import torch.utils.cpp_extension as C; print('-I' + str.join(' -I', C.include_paths()))")

LIBPATH := $(shell python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../')")

USE_CUDA := $(shell python -c "import torch; print(torch.cuda.is_available())")

ifeq ($(USE_CUDA),True)
	TORCH_LIBS=-ltorch -lcaffe2 -lcaffe2_gpu -lc10 -lc10_cuda -lcuda -lnvrtc -lnvToolsExt # -lnccl -lmkldnn -lmkl_rt 
else
	TORCH_LIBS=-ltorch -lcaffe2 -lc10
endif

test: mnist data/train-images-idx3-ubyte
	./mnist

mnist: mnist.cpp
	$(CXX) mnist.cpp -o mnist $(INCPATH) -L$(LIBPATH) -Wl,-rpath,$(LIBPATH) -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) -pthread -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0

#  -lcudart -lnvrtc -lnvrtc-builtins -lnvToolsExt
#  -Wl,-rpath,$(LIBPATH)
#  -lcaffe2_gpu -lc10
data/train-images-idx3-ubyte:
	python download_mnist.py
	mkdir ./data
	cp *ubyte ./data
