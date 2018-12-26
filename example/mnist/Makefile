manist: mnist.out data/train-images-idx3-ubyte
	./mnist.out

mnist.out: mnist.cpp
	$(CXX) $(CXX_FLAGS) $^ -o $@  -Wl,-rpath,$(LIBPATH) -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS) -L$(CONDA_PREFIX)/lib -L$(LIBPATH) $(INCPATH) $(THXX_LOCAL_INCPATH)

data/train-images-idx3-ubyte:
	python download_mnist.py
	mkdir ./data
	cp *ubyte ./data
