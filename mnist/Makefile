# INSTALL: conda install pytorch-cpu=1.0.0 -c pytorch

INCPATH := $(shell python -c "import torch.utils.cpp_extension as C; print('-I' + str.join(' -I', C.include_paths()))")

LIBPATH := $(shell python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../')")

test: mnist data/train-images-idx3-ubyte
	LD_LIBRARY_PATH=$(LIBPATH):$(LD_LIBRARY_PATH) ./mnist

mnist: mnist.cpp
	$(CXX) mnist.cpp $(INCPATH) -L$(LIBPATH) -ltorch -lcaffe2 -lc10 -pthread -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -o mnist

data/train-images-idx3-ubyte:
	python download_mnist.py
	mkdir ./data
	cp *ubyte ./data
