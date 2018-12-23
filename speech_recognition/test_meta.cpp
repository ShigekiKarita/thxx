#include "testing.hpp"
#include "meta.hpp"

using namespace meta;


struct Twice {
    auto operator()(torch::Tensor x) {
        return x * 2;
    }

    auto operator()(torch::Tensor x1, torch::Tensor x2) {
        return std::make_tuple(x1, x2 * 2);
    }
};

TEST_CASE( "Lambda with single in/out", "[meta]" ) {
    auto x = torch::rand({2, 5, 6});

    auto f1 = lambda([](auto&& x) { return x * 2; });
    auto x1 = f1->forward(x);
    CHECK_THAT( x1, testing::TensorEq(x * 2) );

    Lambda<Twice> f2;
    auto x2 = f2->forward(x);
    CHECK_THAT( x2, testing::TensorEq(x * 2) );

    auto f3 = sequential(f1, f2);
    auto x3 = f3->forward(x);
    CHECK_THAT( x3, testing::TensorEq(f2->forward(f1->forward(x))) );

    auto relu = lambda(torch::relu);
    auto x_relu = relu->forward(x);
    CHECK_THAT( x_relu, testing::TensorEq(torch::relu(x)));

    auto f4 = sequential(f3, torch::relu);
    // auto x4 = f4->forward(x);
    // CHECK_THAT( x4, testing::TensorEq(torch::relu(f3->forward(x))) );
}

TEST_CASE( "Lambda with mult in/out", "[meta]" ) {
    auto x = torch::rand({2, 5, 6});

    auto f1 = lambda([](auto&& x1, auto&& x2) { return std::make_tuple(x1, x2 * 2); });
    {
        auto [a, b] = f1->forward(x, x);
        CHECK_THAT( a, testing::TensorEq(x) );
        CHECK_THAT( b, testing::TensorEq(x * 2) );
    }
    Lambda<Twice> f2;
    {
        auto [a, b] = f2->forward(x, x);
        CHECK_THAT( a, testing::TensorEq(x) );
        CHECK_THAT( b, testing::TensorEq(x * 2) );
    }

}
