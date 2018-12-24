#include <torch/torch.h>

constexpr std::int64_t idim = 11;
constexpr std::int64_t odim = 11;

auto gen_data(std::int64_t batch_size) {
    auto x = torch::randint(0, idim - 1, {batch_size, 10});
    auto xlen = torch::empty({batch_size});
    return std::make_tuple(x, xlen);
}

int main() {
    for (int i = 0; i < 100; ++i) {
        auto [x, xlen] = gen_data();
    }
}
