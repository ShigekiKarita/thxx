#include <torch/torch.h>

int main() {
    auto model = torch::nn::Linear(4096, 4096);
    auto optimizer = torch::optim::Adam(model->parameters(), 0.01);

    auto loss = model->forward(torch::zeros({1, 4096})).sum();
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();
    // torch::save(model, "/tmp/model.pt");
    torch::save(optimizer, "/tmp/optimizer.pt");
    while (true) {
        // torch::load(model, "/tmp/model.pt");
        torch::load(optimizer, "/tmp/optimizer.pt");
    }
}
