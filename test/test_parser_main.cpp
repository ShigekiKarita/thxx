#include <thxx/parser.hpp>
#include <iostream>

using thxx::parser::ArgParser;

struct Opt {
    std::int64_t batch_size = 32;
    std::vector<std::int64_t> units = {1, 2, 3};
    bool use_cuda = true;
    std::string expdir = "";

    ArgParser parser;

    Opt(ArgParser p) : parser(p) {
        parser.required("--batch_size", batch_size, "batch size for training");
        parser.add("--use_cuda", use_cuda, "flag to enable cuda");
        parser.add("--units", units, "number of units");
        parser.add("--expdir", expdir, "experiment directory");
        if (parser.help_wanted) {
            std::exit(0);
        }
        parser.check();
    }
};

int main(int argc, char *argv[]) {
    Opt opt(ArgParser(argc, argv));
    // AT_ASSERT(opt.batch_size == 4);
    std::cout << opt.parser.to_json() << std::endl;
}
