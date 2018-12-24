#include <thxx/testing.hpp>
#include <thxx/dataset.hpp>

using namespace thxx;
using namespace thxx::dataset;

/// sample from catch2
unsigned int Factorial( unsigned int number ) {
    return number <= 1 ? number : Factorial(number-1)*number;
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
    REQUIRE( Factorial(1) == 1 );
    REQUIRE( Factorial(2) == 2 );
    REQUIRE( Factorial(3) == 6 );
    REQUIRE( Factorial(10) == 3628800 );
}

TEST_CASE( "Read json and scp", "[dataset]" ) {
    // NOTE: these are copied from espnet/egs/an4/asr1/dump/train_dev/deltafalse/
    auto json = read_json("test_data/data.1.json");
    auto scp = open_scp("test_data/feats.1.scp"); // renamed dir
    auto batchset = make_batchset(json, scp, 5);

    std::unique_ptr<at::Tensor> prev_targets;
    for (auto& bs : batchset) {
        MiniBatch b(bs);
        if (prev_targets) {
            // test sorted by output length
            REQUIRE(prev_targets->size(1) <= b.targets->size(1));
        }
        prev_targets = std::move(b.targets);
    }
}
