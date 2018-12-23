#include "meta.hpp"
#include "testing.hpp"

using namespace meta;


struct TwiceRight {
    auto operator()(torch::Tensor x1, torch::Tensor x2) {
        return std::make_tuple(x1, x2 * 2);
    }
};

TEST_CASE( "Lambda/Seq with mult in/out", "[meta]" ) {
    auto x = torch::rand({2, 5, 6});

    Lambda f1 = lambda(
        [](auto&& x1, auto&& x2) { return std::make_tuple(torch::relu(x1), x2 * 2); });
    {
        auto [a, b] = f1->forward(x, x);
        CHECK_THAT( a, testing::TensorEq(torch::relu(x)) );
        CHECK_THAT( b, testing::TensorEq(x * 2) );
    }
    {
        auto [a, b] = f1->forward(std::make_tuple(x, x));
        CHECK_THAT( a, testing::TensorEq(torch::relu(x)) );
        CHECK_THAT( b, testing::TensorEq(x * 2) );
    }

    Lambda<TwiceRight> f2;
    {
        auto [a, b] = f2->forward(x, x);
        CHECK_THAT( a, testing::TensorEq(x) );
        CHECK_THAT( b, testing::TensorEq(x * 2) );
    }

    Seq<decltype(f1), Lambda<TwiceRight>> f3 = sequential(f1, f2);
    {
        auto [a, b] = f3->forward(x, x);
        CHECK_THAT( a, testing::TensorEq(torch::relu(x)) );
        CHECK_THAT( b, testing::TensorEq(x * 4) );
    }
}

struct Twice {
    auto operator()(torch::Tensor x) {
        return x * 2;
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

    static_assert(std::is_function<decltype(torch::relu)>::value);
    auto f4 = sequential(f3, relu);
    auto x4 = f4->forward(x);
    CHECK_THAT( x4, testing::TensorEq(torch::relu(f3->forward(x))) );
}

TEST_CASE( "sequential and its submodules", "[meta]" ) {
    auto l1 = torch::nn::Linear(2, 3);
    auto l2 = torch::nn::Linear(3, 4);
    // Actualy, you can use `auto` but this is useful for member type or doc
    Seq<torch::nn::Linear, torch::nn::Linear> seq = sequential(l1, l2);

    // modified after submodules registered
    l1->weight.set_requires_grad(false);
    l1->weight.zero_();
    l2->bias.set_requires_grad(false);
    l2->bias.zero_();

    CHECK_THAT( seq->named_children()["0"]->parameters()[0], testing::TensorEq(l1->weight) );
    CHECK_THAT( seq->named_children()["0"]->parameters()[1], testing::TensorEq(l1->bias) );
    CHECK_THAT( seq->named_children()["1"]->parameters()[0], testing::TensorEq(l2->weight) );
    CHECK_THAT( seq->named_children()["1"]->parameters()[1], testing::TensorEq(l2->bias) );
}
