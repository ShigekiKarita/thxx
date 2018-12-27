.PHONY: run

run: main.out
	./main.out

main.out: main.cpp
	$(CXX) $(CXX_FLAGS) $^ -o $@  -Wl,-rpath,$(LIBPATH) -Wl,-rpath,$(CONDA_PREFIX)/lib $(TORCH_LIBS) $(KALDI_FLAGS) -L$(CONDA_PREFIX)/lib -L$(LIBPATH) $(INCPATH) -I../../include $(THXX_LOCAL_INCPATH)
