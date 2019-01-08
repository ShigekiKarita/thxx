#include <torch/torch.h>

#include <algorithm>
#include <random>
#include <iostream>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <kaldi-io.h>

#include <thxx/net.hpp>
#include <thxx/chrono.hpp>
#include <thxx/optim.hpp>
#include <thxx/dataset.hpp>
#include <typed_argparser.hpp>

struct Config : thxx::net::transformer::Config
{
    // new config
    std::mt19937::result_type seed = 0;
    bool use_cuda = false;

    std::string model = "model.pt";
    std::string char_list = "espnet/egs/an4/asr1/data/lang_1char/train_nodev_units.txt";
    std::string decode_json = "espnet/egs/an4/asr1/dump/train_dev/deltafalse/data.json";
    std::string decode_scp = "espnet/egs/an4/asr1/dump/train_dev/deltafalse/feats.scp";

    std::string json;
    typed_argparser::ArgParser parser;

    // parse cmd args
    void parse(int argc, const char *const argv[])
    {
        parser = typed_argparser::ArgParser(argc, argv);

        // update default values
        lr = 10;
        warmup_steps = 25000;

        parser.add("--json", json);
        if (!json.empty())
        {
            parser.from_json(json);
        }

        // data setting
        parser.add("--char_list", char_list, "character (token) list for model output.");
        parser.add("--decode_json", decode_json, "decode datafor meta data.");
        parser.add("--decode_scp", decode_scp, "decode dataset scp for speech data.");

        // model setting
        parser.add("--model", model, "trained model path");
        parser.add("--seed", seed, "random generator seed.");
        parser.add("--d_model", d_model, "the number of the entire model dim.");
        parser.add("--d_ff", d_ff, "the number of the feed-forward layer dim.");
        parser.add("--heads", heads, "the number of heads in the attention layer.");
        parser.add("--elayers", elayers, "the number of encoder layers.");
        parser.add("--dlayers", dlayers, "the number of decoder layers.");
        parser.add("--dropout_rate", dropout_rate, "dropout rate.");
        parser.add("--label_smoothing", label_smoothing, "label smoothing penalty.");

        // decode setting
        parser.add("--use_cuda", use_cuda, "use cuda for training.");
        parser.add("--beam_size", batch_size, "beam size.");
        parser.add("--batch_size", batch_size, "minibatch size.");
        parser.add("--max_len_ratio", max_len_ratio, "max length ratio for output/input sequence.");
        parser.add("--max_len_ratio", min_len_ratio, "min length ratio for output/input sequence.");

        if (parser.help_wanted)
        {
            std::cout << parser.help_message() << std::endl;
            std::exit(0);
        }

        // parser.check();

        json = parser.to_json();
        std::ofstream ofs("decode.json");
        ofs << json;
    }

    thxx::optim::NoamOptions noam_options() const {
        return {d_model, lr, warmup_steps};
    }
};

int main(int argc, const char *argv[])
{
    Config config;
    config.parse(argc, argv);
    std::cout << "[config] " << config.json << std::endl;

    torch::manual_seed(config.seed);
    std::mt19937 engine(config.seed);

    torch::DeviceType device_type;
    if (torch::cuda::is_available() && config.use_cuda)
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

    auto char_list = thxx::dataset::read_char_list(std::ifstream(config.char_list));
    auto decode_json = thxx::dataset::read_json(config.decode_json);
    auto decode_scp = kaldi::SequentialBaseFloatMatrixReader("scp:" + config.decode_scp);

    auto idim = decode_scp.Value().NumCols();
    auto odim = char_list.size();
    std::cout << "idim: " << idim << ", odim: " << odim << std::endl;
    using InputLayer = thxx::net::transformer::Conv2dSubsampling;
    thxx::net::Transformer<InputLayer> model(idim, odim, config);
    model->to(device);
    for (; !decode_scp.Done(); decode_scp.Next()) {
        auto key = decode_scp.Key();
        std::cout << key << std::endl;
        auto ptr = std::make_shared<kaldi::Matrix<float>>(decode_scp.Value());
        torch::NoGradGuard no_grad;
        auto n_best = model->recognize(torch::autograd::make_variable(thxx::memory::make_tensor(ptr)));
        std::cout << "gold: " << (*decode_json)["utts"][key.c_str()]["output"][0]["text"].GetString() << std::endl;
        std::cout << "pred: " << n_best.front().to_string(char_list) << std::endl;
    }
}
