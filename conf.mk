RELEASE := false
ifeq ($(RELEASE),true)
export CXX_FLAGS := $(CXXFLAGS) -std=c++17 -O2 -pthread -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Wno-unused-function -march=native
else
export CXX_FLAGS := $(CXXFLAGS) -std=c++17 -g3 -O0 -pthread -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Wextra -Wno-unused-function -D_LIBCPP_DEBUG
endif

# -D_GLIBCXX_DEBUG
# -ftrapv

export INCPATH := $(shell python -c "import torch.utils.cpp_extension as C; print('-isystem' + str.join(' -isystem', C.include_paths()))")

ifeq ($(INCPATH),)
export INCPATH := -isystem $(THXX_ROOT)/third_party/libtorch/include -isystem $(THXX_ROOT)/third_party/libtorch/include/torch/csrc/api/include
endif

export LIBPATH := $(shell python -c "import torch.utils.cpp_extension as C; print(C.include_paths()[0] + '/../')")

ifeq ($(LIBPATH),)
export LIBPATH := $(THXX_ROOT)/third_party/libtorch/lib
endif

export USE_CUDA := $(shell python -c "import torch; print(torch.cuda.is_available())")

ifeq ($(USE_CUDA),True)
export TORCH_LIBS=-ltorch -lcaffe2 -lcaffe2_gpu -lc10 -lc10_cuda -lcuda -lnvrtc -lnvToolsExt # -lnccl -lmkldnn -lmkl_rt
else
export TORCH_LIBS=-ltorch -lcaffe2 -lc10
endif

export THXX_LOCAL_INCPATH := -I$(THXX_ROOT)/include -isystem$(THXX_ROOT)/third_party -isystem$(THXX_ROOT)/third_party/cxx11-typed-argparser/include
