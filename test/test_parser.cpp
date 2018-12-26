#include <iostream>
#include <catch.hpp>
#include <thxx/parser.hpp>


using thxx::parser::ArgParser;

struct Opt : thxx::parser::ArgParser {
    std::int64_t batch_size = 32;
    std::vector<std::int64_t> units = {1, 2, 3};
    bool use_cuda = true;
    std::string expdir = "";
    std::string json = "";

    Opt() {
        parse();
    }

    Opt(int argc, const char* const argv[]) : ArgParser(argc, argv) {
        add("--json", json);
        if (!json.empty()) {
            from_json(json);
        }
        else {
            parse();
        }
    }

    void parse() {
        required("--batch_size", batch_size);
        add("--use_cuda", use_cuda);
        add("--units", units);
        add("--expdir", expdir);
        check();
    }
};

template <typename T, size_t N>
constexpr size_t asizeof(T(&)[N]) { return N; }

TEST_CASE( "to_json", "[parser]" ) {
    const char* argv[] = {"prog.exe", "--batch_size", "4", "--units", "100", "200", "300"};
    static_assert(asizeof(argv) == 7);
    Opt opt(asizeof(argv), (char**) argv);

    CHECK( opt.batch_size == 4 );
    CHECK( opt.units == decltype(opt.units){100, 200, 300} );
    CHECK( opt.use_cuda );
    CHECK( opt.to_json(false) == R"({"--use_cuda":true,"--json":"","--expdir":"","--batch_size":4,"--units":[100,200,300]})" );
    std::cout << opt.to_json(false) << std::endl;
}

TEST_CASE( "from_json", "[parser]" ) {
    Opt opt;
    opt.from_json( R"({"--use_cuda":true,"--expdir":"","--batch_size":4,"--units":[100,200,300]})" );
    std::cout << opt.to_json() << std::endl;
}


// int main(int argc, char *argv[]) {
//     Opt opt(ArgParser(argc, argv));
//     // AT_ASSERT(opt.batch_size == 4);
//     std::cout << opt.to_json() << std::endl;
// }
