# thxx - libtorch C++ API extentions

[![Build Status](https://travis-ci.org/ShigekiKarita/thxx.svg?branch=master)](https://travis-ci.org/ShigekiKarita/thxx)

## CI status

| compiler | conda package                                                                                     | latest zip                                                                                         |
| -------- | ------------------------------------------------------------------------------------------------- | ----------                                                                                         |
| gcc-7    | ![gcc-7](https://travis-matrix-badges.herokuapp.com/repos/ShigekiKarita/thxx/branches/master/1)   | ![gcc-7](https://travis-matrix-badges.herokuapp.com/repos/ShigekiKarita/thxx/branches/master/2)    |
| gcc-8    | ![gcc-8](https://travis-matrix-badges.herokuapp.com/repos/ShigekiKarita/thxx/branches/master/3)   | ![gcc-7](https://travis-matrix-badges.herokuapp.com/repos/ShigekiKarita/thxx/branches/master/4)    |
| clang-5  | ![clang-5](https://travis-matrix-badges.herokuapp.com/repos/ShigekiKarita/thxx/branches/master/5) | ![clang-5](https://travis-matrix-badges.herokuapp.com/repos/ShigekiKarita/thxx/branches/master/6)  |
| clang-6  | ![clang-6](https://travis-matrix-badges.herokuapp.com/repos/ShigekiKarita/thxx/branches/master/7) | ![clang-6](https://travis-matrix-badges.herokuapp.com/repos/ShigekiKarita/thxx/branches/master/8)  |
| clang-7  | ![clang-7](https://travis-matrix-badges.herokuapp.com/repos/ShigekiKarita/thxx/branches/master/9) | ![clang-6](https://travis-matrix-badges.herokuapp.com/repos/ShigekiKarita/thxx/branches/master/10) |

## requirements

- gcc 7, 8 or clang 5, 6, 7
- compiler option `-std=c++17`
- libtorch 1.0.0 (recommend `conda install pytorch-cpu=1.0.0 -c pytorch`)

## how to run

- using conda (recommend)
``` console
$ conda install pytorch-cpu=1.0.0 -c pytorch
$ make test
$ make example-mnist
```

- using libtorch latest zip
```console
$ cd 
$ make --directory ./third_party libtorch-shared-with-deps-latest.zip
$ make test
$ make example-mnist
```

for more details, see [.travis.yml](.travis.yml).

## features

### functional meta modules

- thxx::meta::Lambda<...> : torch::nn::ModuleHolder<...>

``` c++
auto x = torch::rand({2, 3, 4});

// single input/output
Lambda f1 = lambda([](auto&& x) { return x * 2; });
auto x1 = f1->forward(x);
CHECK_THAT( x1, testing::TensorEq(x * 2) );

// multi input/output
Lambda f2 = lambda([](auto&& x1, auto&& x2) { return std::make_tuple(x1.relu(), x2 * 2); });
{
    auto [a, b] = f2->forward(x, x);
    CHECK_THAT( a, testing::TensorEq(torch::relu(x)) );
    CHECK_THAT( b, testing::TensorEq(x * 2) );
}
{
    // automatic tuple arg unpack
    auto [a, b] = f1->forward(std::make_tuple(x, x));
    CHECK_THAT( a, testing::TensorEq(torch::relu(x)) );
    CHECK_THAT( b, testing::TensorEq(x * 2) );
}
```

- thxx::meta::Seq<...> : torch::nn::ModuleHolder<...>
``` c++
auto x1 = torch::rand({2, 3, 4});

// single input/output with Lambda
auto f1 = torch::nn::Linear(4, 5);
auto f2 = sequential(f1, lambda(torch::relu));
auto x2 = f2->forward(x1);
CHECK_THAT( x2, testing::TensorEq(torch::relu(f1->forward(x1)) );

// share submodules/parameters
CHECK_THAT( f2->parameters[0], testing::TensorEq(f1->weight) );
CHECK_THAT( f2->parameters[1], testing::TensorEq(f1->bias) );

// also support multi input/output with Lambda
auto f3 = lambda([](auto&& x1, auto&& x2) { return std::make_tuple(x1, x2 * 2); });
auto f4 = sequential(f3, f3);;
auto [x3, x4] = f4->forward(x1);
CHECK_THAT( x3, testing::TensorEq(x1) );
CHECK_THAT( x4, testing::TensorEq(x1 * 4) );
```

see [test/test_meta.cpp](https://github.com/ShigekiKarita/thxx/blob/master/test/test_meta.cpp)

### networks/modules

- Transformer
  - MultiHeadedAttention
  - PositionalEncoding
  - PositionwiseFeedforward
  - pad/masking functions
- Normalization
  - LayerNorm
- Math (wip)
  - copy C++ batched/complex linalg funcitons from https://github.com/ShigekiKarita/thxx-py
- Loss
  - label smoothing KLDivLoss

see [test/test_net.cpp](https://github.com/ShigekiKarita/thxx/blob/master/test/test_net.cpp)

### serialization

- HDF5 (wip)
- numpy (wip)


## acknowledgement

mnist example is forked from https://github.com/goldsborough/examples/tree/cpp/cpp/mnist

## license

BSL-1.0 (except for example/mnist and catch.hpp)

