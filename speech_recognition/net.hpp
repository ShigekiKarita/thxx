/**

   NOTE:
   - torch::nn::Linear is a good example
   https://github.com/pytorch/pytorch/blob/v1.0.0/torch/csrc/api/include/torch/nn/modules/linear.h
   - use TORCH_MODULE for ModuleHolder?
   https://github.com/pytorch/pytorch/blob/eb5d28ecefb9d78d4fff5fac099e70e5eb3fbe2e/torch/csrc/api/include/torch/nn/pimpl.h#L195
   - torch::Tensor API
   https://pytorch.org/cppdocs/api/classat_1_1_tensor.html
*/
#pragma once

#include <torch/torch.h>
#include <cmath>
#include <limits>

#include "meta.hpp"

namespace net {
    /// convert lengths {1, 2} to mask {{1, 0}, {1, 1}}
    at::Tensor pad_mask(at::IntList lengths) {
        auto maxlen = *std::max_element(lengths.begin(), lengths.end());
        auto bs = static_cast<std::int64_t>(lengths.size());
        auto ret = at::zeros({bs, maxlen}, at::kByte);
        for (size_t i = 0; i < lengths.size(); ++i) {
            ret[i].slice(0, 0, lengths[i]) = 1;
        }
        return ret;
    }

    class LayerNormImpl : public torch::nn::Cloneable<LayerNormImpl> {
    public:
        std::int64_t features;
        float eps;
        torch::Tensor scale;
        torch::Tensor bias;

        LayerNormImpl(std::int64_t features, float eps=1e-6)
            : features(features), eps(eps) {
            this->reset();
        }

        void reset() override {
            this->scale = register_parameter("scale", torch::ones(features));
            this->bias = register_parameter("bias", torch::zeros(features));
        }

        torch::Tensor forward(torch::Tensor x) {
            auto mean = x.mean(-1, true);
            auto std = x.std(-1, true).unsqueeze(-1);
            return this->scale * (x - mean) / std + this->bias; // .expand_as(x); // this->scale * (x - mean) / (std + eps) + this->bias;
        }
    };
    TORCH_MODULE(LayerNorm);


    class MultiHeadedAttentionImpl : public torch::nn::Cloneable<MultiHeadedAttentionImpl> {
    public:
        // configurations
        std::int64_t heads;
        std::int64_t d_model;
        std::int64_t d_k;
        float dropout_rate;

        // submodules
        torch::nn::Linear linear_q = nullptr;
        torch::nn::Linear linear_k = nullptr;
        torch::nn::Linear linear_v = nullptr;
        torch::nn::Linear linear_out = nullptr;
        torch::nn::Dropout dropout = nullptr;

        torch::Tensor attn;
        static constexpr float min_value = std::numeric_limits<float>::min();

        MultiHeadedAttentionImpl(std::int64_t heads, std::int64_t d_model, float dropout_rate)
            : heads(heads), d_model(d_model), d_k(d_model / heads), dropout_rate(dropout_rate) {
            AT_ASSERT(d_model % heads == 0);
            this->reset();
        }

        void reset() override {
            this->linear_q = register_module("linear_q", torch::nn::Linear(this->d_model, this->d_model));
            this->linear_k = register_module("linear_k", torch::nn::Linear(this->d_model, this->d_model));
            this->linear_v = register_module("linear_v", torch::nn::Linear(this->d_model, this->d_model));
            this->linear_out = register_module("linear_out", torch::nn::Linear(this->d_model, this->d_model));
            this->dropout = register_module("dropout", torch::nn::Dropout(this->dropout_rate));
        }

        torch::Tensor forward(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor mask) {
            // check minibatch size
            auto n_batch = query.size(0);
            AT_ASSERT(key.size(0) == n_batch);
            AT_ASSERT(value.size(0) == n_batch);
            AT_ASSERT(mask.size(0) == n_batch);
            // check time length
            auto q_len = query.size(1);
            // AT_ASSERT(mask.size(1) == q_len);
            auto kv_len = key.size(1);
            AT_ASSERT(value.size(1) == kv_len);
            // check feature size
            AT_ASSERT(query.size(2) == this->d_model);
            AT_ASSERT(key.size(2) == this->d_model);
            AT_ASSERT(value.size(2) == this->d_model);
            // type check
            AT_ASSERT(query.scalar_type() == at::kFloat);
            AT_ASSERT(key.scalar_type() == at::kFloat);
            AT_ASSERT(value.scalar_type() == at::kFloat);
            AT_ASSERT(mask.scalar_type() == at::kByte);

            auto q = this->linear_q->forward(query).view({n_batch, q_len, this->heads, this->d_k}).transpose(1, 2);
            auto k = this->linear_k->forward(key).view({n_batch, kv_len, this->heads, this->d_k}).transpose(1, 2);
            auto v = this->linear_v->forward(value).view({n_batch, kv_len, this->heads, this->d_k}).transpose(1, 2);
            auto scores = q.matmul(k.transpose(-2, -1)) / std::sqrt(this->d_k);
            // TODO: create non destructive masked_fill?
            // auto m0 = torch::autograd::make_variable((mask.unsqueeze(1) == 0).to(at::kFloat));
            // auto m1 = torch::autograd::make_variable(mask.unsqueeze(1).to(at::kFloat));
            // auto masked_scores = scores * m1 + min_value * m0;
            auto m = torch::autograd::make_variable(mask.unsqueeze(1) == 0);
            auto masked_scores = scores.masked_fill_(m, min_value);
            this->attn = masked_scores.softmax(-1);
            auto p_attn = this->dropout->forward(this->attn);
            auto weighted = p_attn.matmul(v);
            auto y = weighted.transpose(1, 2).contiguous().view({n_batch, q_len, this->d_model});
            return this->linear_out->forward(y);
        }
    };
    TORCH_MODULE(MultiHeadedAttention);

    namespace transformer {
        auto positionwise_feedforward(std::int64_t d_model, std::int64_t d_ff, float dropout_rate) {
            return meta::sequential(
                torch::nn::Linear(d_model, d_ff),
                torch::nn::Dropout(dropout_rate),
                meta::lambda(torch::relu),
                torch::nn::Linear(d_ff, d_model)
                );
        }

        /// for convinience
        using PositionwiseFeedforward = decltype(positionwise_feedforward(0,0,0.0));

        class PositionalEncodingImpl : public torch::nn::Cloneable<PositionalEncodingImpl> {
        public:
            std::int64_t d_model;
            float dropout_rate;
            std::int64_t max_len;
            float scale;

            // submodules
            torch::nn::Dropout dropout;
            torch::Tensor pe;

            PositionalEncodingImpl(std::int64_t d_model, float dropout_rate, std::int64_t max_len = 5000)
                : d_model(d_model), dropout_rate(dropout_rate), max_len(max_len), scale(std::sqrt(d_model)) {
                this->reset();
            }

            void reset() override {
                torch::NoGradGuard no_grad;
                this->dropout = this->register_module("dropout", torch::nn::Dropout(this->dropout_rate));
                auto pe = torch::zeros({1, this->max_len, this->d_model});
                auto position = torch::arange(0, max_len).unsqueeze(1);
                auto div_term = torch::exp(torch::arange(0, d_model, 2) * -std::log(10000.0) / d_model);
                pe.slice(2, 0, pe.size(2), 2) = torch::sin(position * div_term);
                pe.slice(2, 1, pe.size(2), 2) = torch::cos(position * div_term);
                this->pe = this->register_buffer("pe", pe);
            }

            auto forward(torch::Tensor x) {
                torch::NoGradGuard no_grad;
                auto y = this->scale * x + this->pe.slice(1, 0, x.size(1));
                return this->dropout->forward(y);
            }
        };
        TORCH_MODULE(PositionalEncoding);

    }

    class Transformer : torch::nn::Module {
    public:
    };
}
