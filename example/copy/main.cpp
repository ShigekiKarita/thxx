/**
   toy task introduced in https://github.com/harvardnlp/annotated-transformer
   TODO: cuda backend, greedy/beam decoding
*/
#include <torch/torch.h>
#include <thxx/net.hpp>
#include <typed_argparser.hpp>


struct Config : thxx::net::transformer::Config {
    // new config
    std::int64_t dim = 10;
    std::int64_t len = 10;
    bool use_cuda = false;

    Config() {
        // update defaults
        heads = 2;
        d_model = 16;
        d_ff = 16;
        elayers = 1;
        dlayers = 1;
    }

    // parse cmd args
    void parse(int argc, const char* const argv[]) {
        typed_argparser::ArgParser parser(argc, argv);

        std::string json;
        parser.add("--json", json);
        if (!json.empty()) {
            parser.from_json(json);
        }

        // data setting
        parser.add("--data_dim", dim, "data dim");
        parser.add("--data_len", len, "data length");

        // model setting
        parser.add("--d_model", d_model, "the number of the entire model dim.");
        parser.add("--d_ff", d_ff, "the number of the feed-forward layer dim.");
        parser.add("--heads", heads, "the number of heads in the attention layer.");
        parser.add("--elayers", elayers, "the number of encoder layers.");
        parser.add("--dlayers", dlayers, "the number of decoder layers.");
        parser.add("--dropout_rate", dropout_rate, "dropout rate.");
        parser.add("--label_smoothing", label_smoothing, "label smoothing penalty.");

        // training setting
        parser.add("--use_cuda", use_cuda, "use CUDA for training.");
        parser.add("--lr", lr, "learning rate.");
        parser.add("--warmup_steps", warmup_steps, "warmup steps for lr scheduler.");
        parser.add("--batch_size", batch_size, "minibatch size.");
        parser.add("--max_len_in", max_len_in, "max length for input sequence.");
        parser.add("--max_len_out", max_len_out, "max length for output sequence.");

        if (parser.help_wanted) {
            std::cout << parser.help_message() << std::endl;
            std::exit(0);
        }

        parser.check();

        std::ofstream ofs("config.json");
        ofs << parser.to_json();
    }

    auto gen_data() {
        auto x = torch::randint(1, dim - 1, {batch_size, len}).to(at::kLong);
        std::vector<std::int64_t> xlen(batch_size, 10);
        return std::make_tuple(x, xlen);
    }
};

int main(int argc, char *argv[]) {
    Config config;
    config.parse(argc, argv);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) // && config.use_cuda)
    {
        std::cout << "CUDA available! Training on GPU" << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU" << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    using InputLayer = thxx::net::transformer::PositonalEmbedding;
    thxx::net::Transformer<InputLayer> model(config.dim, config.dim, config);
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), 0.01);

    for (int i = 1; i <= 1000; ++i) {
        auto [_x, xlen] = config.gen_data();
        optimizer.zero_grad();
        auto x = _x.to(device);
        auto [loss, acc] = model->forward(x, xlen, x, xlen);
        loss.backward();
        optimizer.step();
        if (i % 100 == 0) {
            std::cout << "step: " << i
                      << ", loss: " << loss.template item<double>()
                      << ", acc: " << acc << std::endl;
        }
    }
}
