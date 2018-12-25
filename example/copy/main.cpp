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

#include <unordered_map>

class ArgParser {
public:
    std::unordered_map<std::string, std::string> parsed_map;

    ArgParser(int argc, char *argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::cout << "[" << argv[i] << "]" << std::endl;
        }
    }

    bool is_set(const std::string& key) {
        return this->parsed_map.find(key) != this->parsed_map.end();
    }

    void required(const std::string& key, std::string& value) {
        if (this->is_set(key)) {
            value = this->parsed_map[key];
        }
    }

    void required(const std::string& key, std::int64_t* value) {
        if (this->is_set(key)) {
            *value = std::stoi(this->parsed_map[key]);
        }
    }
};

struct Opt {
    std::int64_t batch_size = 32;
    bool use_cuda = true;

    Opt(ArgParser parser) {
        parser.required("--batch_size", &batch_size);
        // optional("--use_cuda", use_cuda);
    }
};

int main(int argc, char *argv[]) {
    // Opt opt(ArgParser(argc, argv));
    // return;
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
