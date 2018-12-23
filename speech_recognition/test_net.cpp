#include "testing.hpp"
#include "net.hpp"

using namespace net;


TEST_CASE( "make_pad_mask", "[net]" ) {
    std::vector<std::int64_t> ls = {1, 2, 3, 4};

    auto m = make_pad_mask(ls);
    auto minv = make_pad_mask(ls) != 1;

    for (std::int64_t i = 0; i < m.size(0); ++i) {
        for (std::int64_t j = 0; j < m.size(1); ++j) {
            std::uint8_t b = j < ls[i] ? 1 : 0;
            CHECK(m[i][j].template item<std::uint8_t>() == b);
            CHECK(minv[i][j].template item<std::uint8_t>() != b);
        }
    }
}

TEST_CASE( "LayerNorm", "[net]" ) {
    LayerNorm norm(3);
    auto x = torch::rand({1, 2, 3});

    auto x_ = (x - x.mean(-1, true)) / x.std(-1, true).unsqueeze(-1);
    auto y = norm->forward(x);
    CHECK_THAT( y, testing::TensorEq(x_) );

    // test grad
    CHECK_FALSE( norm->scale.grad().defined() );
    CHECK_FALSE( norm->bias.grad().defined() );
    y.sum().backward();
    CHECK( norm->scale.grad().defined() );
    CHECK( norm->bias.grad().defined() );

    auto params = norm->parameters();
    CHECK_THAT( params[0], testing::TensorEq(norm->scale) );
    CHECK_THAT( params[1], testing::TensorEq(norm->bias) );
}

TEST_CASE( "MultiHeadedAttention", "[net]" ) {
    MultiHeadedAttention att(2, 6, 0.1);
    auto x = torch::rand({2, 5, 6});
    x.set_requires_grad(true);
    auto m = (make_pad_mask({3, 5})).unsqueeze(-2);
    auto ret = att->forward(x, x, x, m);

    for (auto& p : att->parameters()) {
        CHECK_FALSE( p.grad().defined() );
    }
    ret.sum().backward();
    for (auto& p : att->parameters()) {
        CHECK( p.grad().defined() );
    }
    // std::cout << att->attn[0][0] << std::endl;
}


TEST_CASE( "Transformer", "[net]" ) {
    Transformer model;
}
