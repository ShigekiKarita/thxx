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

    class HasGrad : public Catch::MatcherBase<torch::nn::Module> {
    public:
        bool has;
        std::string s;
        HasGrad(bool has) : has(has), s(has ? " " : " not ") {}

        virtual bool match(const torch::nn::Module& m) const override {
            bool ret = true;
            for (const auto& p : m.named_parameters()) {
                if (p->grad().defined() != this->has) {
                    WARN("param: [" + p.key() + "] should" + this->s + "have a grad");
                    ret = false;
                }
            }
            return ret;
        }

        virtual std::string describe() const {
            std::ostringstream ss;
            ss << "should" << this->s << "have a grad";
            return ss.str();
        }
    };

}
