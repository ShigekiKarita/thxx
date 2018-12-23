#include "testing.hpp"
#include "net.hpp"

using namespace net;


TEST_CASE( "pad_mask", "[net]" ) {
    std::vector<std::int64_t> ls = {1, 2, 3, 4};

    auto m = pad_mask(ls);
    auto minv = pad_mask(ls) != 1;

    for (std::int64_t i = 0; i < m.size(0); ++i) {
        for (std::int64_t j = 0; j < m.size(1); ++j) {
            std::uint8_t b = j < ls[i] ? 1 : 0;
            CHECK(m[i][j].template item<std::uint8_t>() == b);
            CHECK(minv[i][j].template item<std::uint8_t>() != b);
        }
    }
}

TEST_CASE( "subsequent_mask", "[net]" ) {
    auto m = subsequent_mask(3);
    auto p = pad_mask({1, 2, 3});
    CHECK_THAT( m, testing::TensorEq(p) );
}

TEST_CASE( "LayerNorm", "[net]" ) {
    LayerNorm norm(3);
    auto x = torch::rand({1, 2, 3});

    auto x_ = (x - x.mean(-1, true)) / x.std(-1, true).unsqueeze(-1);
    auto y = norm->forward(x);
    CHECK_THAT( y, testing::TensorEq(x_) );

    // test grad
    y.sum().backward();
    CHECK_THAT( *norm, testing::HasGrad(true) );

    auto params = norm->parameters();
    CHECK_THAT( params[0], testing::TensorEq(norm->scale) );
    CHECK_THAT( params[1], testing::TensorEq(norm->bias) );
}

TEST_CASE( "MultiHeadedAttention", "[net]" ) {
    MultiHeadedAttention att(2, 6, 0.1);
    auto x = torch::rand({2, 5, 6});
    x.set_requires_grad(true);
    auto m = pad_mask({3, 5}).unsqueeze(-2);
    auto ret = att->forward(x, x, x, m);
    ret.sum().backward();
    CHECK_THAT( *att, testing::HasGrad(true) );
    // std::cout << att->attn[0][0] << std::endl;
}


TEST_CASE( "transformer", "[net]" ) {
    namespace T = transformer;
    std::int64_t n_input = 6;
    auto x = torch::rand({2, 14, n_input});
    auto m = pad_mask({6, 14}).unsqueeze(-2);
    auto y = torch::rand({2, 5, n_input});
    auto ym = pad_mask({4, 5}).unsqueeze(-2).__and__(subsequent_mask(5).unsqueeze(0));

    T::Config conf;
    conf.d_model = n_input;
    conf.d_ff = 3;
    conf.heads = 3;
    {
        T::PositionwiseFeedforward ff = T::positionwise_feedforward(n_input, 3, 0.1);
        auto h = ff->forward(x);
    }
    {
        auto f = T::PositionalEncoding(n_input, 0.1);
        auto h = f->forward(x);
        h.sum().backward();
        CHECK_THAT( *f, testing::HasGrad(true) );
    }
    {
        auto f = T::EncoderLayer(n_input, 3, 4, 0.1);
        auto [h, hm] = f->forward(x, m);
        h.sum().backward();
        CHECK_THAT( *f, testing::HasGrad(true) );
    }
    {
        auto f = T::DecoderLayer(n_input, 3, 4, 0.1);
        auto [p, pm] = f->forward(y, ym, x, m);
        p.sum().backward();
        CHECK_THAT( *f, testing::HasGrad(true) );
    }
    {
        auto f = T::Conv2dSubsampling(n_input, 10, 0.1);
        auto [h, hm] = f->forward(x, m);
        h.sum().backward();
        CHECK_THAT( *f, testing::HasGrad(true) );
    }
    {
        auto f = T::Encoder(n_input, conf);
        auto [h, hm] = f->forward(x, m);
        h.sum().backward();
        CHECK_THAT( *f, testing::HasGrad(true) );
    }
    Transformer model;
}
