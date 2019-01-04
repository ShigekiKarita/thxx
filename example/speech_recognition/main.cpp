#include <torch/torch.h>

#include <iostream>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include <kaldi-io.h>

#include <thxx/net.hpp>
#include <thxx/dataset.hpp>
#include <typed_argparser.hpp>

struct Config : thxx::net::transformer::Config
{
    // new config
    std::int64_t dim = 10;
    std::int64_t len = 10;

    Config()
    {
        // update defaults
        heads = 2;
        d_model = 256;
        d_ff = 512;
        elayers = 3;
        dlayers = 3;
    }

    // parse cmd args
    void parse(int argc, const char *const argv[])
    {
        typed_argparser::ArgParser parser(argc, argv);

        std::string json;
        parser.add("--json", json);
        if (!json.empty())
        {
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
        parser.add("--lr", lr, "learning rate.");
        parser.add("--warmup_steps", warmup_steps, "warmup steps for lr scheduler.");
        parser.add("--batch_size", batch_size, "minibatch size.");
        parser.add("--max_len_in", max_len_in, "max length for input sequence.");
        parser.add("--max_len_out", max_len_out, "max length for output sequence.");

        if (parser.help_wanted)
        {
            std::cout << parser.help_message() << std::endl;
            std::exit(0);
        }

        parser.check();

        std::ofstream ofs("config.json");
        ofs << parser.to_json();
    }
};

int main(int argc, const char *argv[])
{
    Config config;
    config.parse(argc, argv);

    torch::manual_seed(0);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) // && !config.no_cuda)
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

    auto train_json = thxx::dataset::read_json("espnet/egs/an4/asr1/dump/train_nodev/deltafalse/data.json");
    auto dev_json = thxx::dataset::read_json("espnet/egs/an4/asr1/dump/train_dev/deltafalse/data.json");
    auto train_scp = thxx::dataset::open_scp("espnet/egs/an4/asr1/dump/train_nodev/deltafalse/feats.scp");
    auto dev_scp = thxx::dataset::open_scp("espnet/egs/an4/asr1/dump/train_dev/deltafalse/feats.scp");

    auto train_batch = thxx::dataset::make_batchset(train_json, train_scp);
    auto dev_batch = thxx::dataset::make_batchset(dev_json, dev_scp);

    auto idim = train_batch[0][0].idim;
    auto odim = train_batch[0][0].odim;
    std::cout << "idim: " << idim << ", odim: " << odim << std::endl;
    using InputLayer = thxx::net::transformer::Conv2dSubsampling;
    thxx::net::Transformer<InputLayer> model(idim, odim, config);
    torch::optim::Adam optimizer(model->parameters(), 0.01);

    using torch::autograd::make_variable;

    double best_acc = 0;
    for (size_t epoch = 0; epoch < 200; ++epoch)
    {
        std::cout << "==== epoch " << epoch << " ====" << std::endl;
        model->train();
        double sum_train_acc = 0;
        size_t sum_train_sample = 0;
        for (auto batch : train_batch)
        {
            thxx::dataset::MiniBatch mb(batch);
            optimizer.zero_grad();
            auto [loss, acc] = model->forward(
                make_variable(*mb.inputs),
                mb.input_lengths,
                make_variable(*mb.targets),
                mb.target_lengths);
            loss.backward();
            optimizer.step();
            std::cout << "[train] loss: " << loss.item<double>() << ", acc: " << acc << std::endl;
            auto samples = mb.input_lengths[0];
            sum_train_acc += acc * samples;
            sum_train_sample += samples;
        }
        std::cout << "[train] average acc: " << sum_train_acc / sum_train_sample << std::endl;

        model->eval();
        torch::NoGradGuard no_grad;
        double sum_dev_acc = 0;
        size_t sum_dev_sample = 0;
        for (auto batch : dev_batch)
        {
            thxx::dataset::MiniBatch mb(batch);

            auto [loss, acc] = model->forward(
                make_variable(*mb.inputs),
                mb.input_lengths,
                make_variable(*mb.targets),
                mb.target_lengths);
            auto samples = mb.input_lengths[0];
            sum_dev_acc += acc * samples;
            sum_dev_sample += samples;
        }
        double dev_acc = sum_dev_acc / sum_dev_sample;
        std::cout << "[dev] average acc: " << dev_acc << std::endl;

        if (dev_acc > best_acc)
        {
            best_acc = dev_acc;
            torch::save(model, "model.pt");
            torch::save(optimizer, "optimizer.pt");
            std::cout << "the best model is saved" << std::endl;
        }
        else
        {
            torch::load(model, "model.pt");
            torch::load(optimizer, "optimizer.pt");
            optimizer.options.learning_rate_ *= 0.5;
            std::cout << "the best model is loaded. lr decayed to "  << optimizer.options.learning_rate_ << std::endl;
        }
    }
}
