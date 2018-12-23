#pragma once
#include "catch.hpp"
#include <torch/torch.h>


namespace testing {

    class TensorEq : public Catch::MatcherBase<at::Tensor> {
    public:
        at::Tensor a;
        TensorEq(at::Tensor a) : a(a) {}

        virtual bool match(const at::Tensor& b) const override {
            return (a == b).all().template item<std::uint8_t>() == 1;
        }

        virtual std::string describe() const {
            std::ostringstream ss;
            ss << "\nis not equal to\n" << a;
            return ss.str();
        }
    };

}
