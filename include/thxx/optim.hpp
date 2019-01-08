#pragma once
#include <torch/torch.h>

#include <cmath>
#include <utility>

namespace thxx::optim {
    using torch::serialize::InputArchive;
    using torch::serialize::OutputArchive;

    struct NoamOptions {
        std::int64_t model_size;
        double learning_rate = 2;
        std::int64_t warmup_steps = 4000;

        double lr(double n_step) {
            return learning_rate
                * std::min(1.0 / std::sqrt(n_step), n_step * std::pow(warmup_steps, -1.5))
                / std::sqrt(model_size);
        }
    };

    class Noam { // : public torch::optim::Adam {
    public:
        torch::optim::Adam super;
        NoamOptions options;

        template <typename ParameterContainer>
        Noam(ParameterContainer&& parameters, const NoamOptions& options,
             const torch::optim::AdamOptions& super_opt = torch::optim::AdamOptions(0.0)
             .beta1(0.9)
             .beta2(0.98)
             .eps(1e-9)
             .weight_decay(0)
             .amsgrad(false))
            : super(std::forward<ParameterContainer>(parameters), super_opt), options(options) {}

        void step() {
            // TODO assert all step buffers are same
            auto n_step = super.step_buffers.size() == 0 ? 1 : super.step_buffers[0];
            super.options.learning_rate(options.lr(n_step));
            super.step();
        }

        void zero_grad() {
            super.zero_grad();
        }

    };

    OutputArchive& operator<<(OutputArchive& archive, const Noam& optimizer) {
        return archive << optimizer.super;
    }

    InputArchive& operator>>(InputArchive& archive, Noam& optimizer) {
        return archive >> optimizer.super;
    }
} // namespace detail

