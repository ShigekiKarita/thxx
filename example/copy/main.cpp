/**
   toy task introduced in https://github.com/harvardnlp/annotated-transformer
   TODO: cuda backend, greedy/beam decoding
*/
#include <torch/torch.h>
#include <thxx/net.hpp>

constexpr std::int64_t idim = 11;
constexpr std::int64_t odim = 11;

auto gen_data(std::int64_t batch_size = 128) {
    auto x = torch::randint(1, idim - 1, {batch_size, 10}).to(at::kLong);
    std::vector<std::int64_t> xlen(batch_size, 10);
    return std::make_tuple(x, xlen);
}

int main() {
    thxx::net::transformer::Config config;
    config.heads = 2;
    config.d_model = 16;
    config.d_ff = 16;
    config.elayers = 1;
    config.dlayers = 1;

    using InputLayer = thxx::net::transformer::PositonalEmbedding;
    thxx::net::Transformer<InputLayer> model(idim, odim, config);
    torch::optim::Adam optimizer(model.parameters(), 0.01);

    for (int i = 1; i <= 1000; ++i) {
        auto [x, xlen] = gen_data();
        optimizer.zero_grad();
        auto [loss, acc] = model.forward(x, xlen, x, xlen);
        loss.backward();
        optimizer.step();
        if (i % 100 == 0) {
            std::cout << "step: " << i
                      << ", loss: " << loss.template item<double>()
                      << ", acc: " << acc << std::endl;
        }
    }
}
