#include <iostream>
#include <catch.hpp>
#include <thxx/parser.hpp>


using thxx::parser::ArgParser;

struct Opt : thxx::parser::ArgParser {
    std::int64_t batch_size = 32;
    std::vector<std::int64_t> units = {1, 2, 3};
    bool use_cuda = true;
    std::string expdir = "";

    Opt(int argc, char* argv[]) : ArgParser(argc, argv) {
        auto& parser = *this;
        parser.required("--batch_size", batch_size);
        parser.add("--use_cuda", use_cuda);
        parser.add("--units", units);
        parser.add("--expdir", expdir);
        parser.check();
    }
};

TEST_CASE( "to_json", "[parser]" ) {
    char* argv[] = {"prog.exe", "--batch_size", "4", "--units", "100", "200", "300"};
    Opt opt(4, argv);
    std::cout << opt.to_json() << std::endl;
}

// int main(int argc, char *argv[]) {
//     Opt opt(ArgParser(argc, argv));
//     // AT_ASSERT(opt.batch_size == 4);
//     std::cout << opt.parser.to_json() << std::endl;
// }
