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
        parser.required("--batch_size", batch_size);
        parser.add("--use_cuda", use_cuda);
        parser.add("--units", units);
        parser.add("--expdir", expdir);
        parser.check();
    }
};

int main(int argc, char *argv[]) {
    Opt opt(ArgParser(argc, argv));
    // AT_ASSERT(opt.batch_size == 4);
    std::cout << opt.parser.to_json() << std::endl;
}
