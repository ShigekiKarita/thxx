/**

   NOTE:
   - torch::nn::Linear is a good example
   https://github.com/pytorch/pytorch/blob/v1.0.0/torch/csrc/api/include/torch/nn/modules/linear.h
   - use TORCH_MODULE for ModuleHolder?
   https://github.com/pytorch/pytorch/blob/eb5d28ecefb9d78d4fff5fac099e70e5eb3fbe2e/torch/csrc/api/include/torch/nn/pimpl.h#L195
   
*/
#include <torch/torch.h>


namespace net {
    /// convert lengths {1, 2} to mask {{1, 0}, {1, 1}}
    at::Tensor make_pad_mask(at::IntList lengths) {
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
            std::cout << x.sizes() << mean.sizes() << std.sizes() << std::endl;
            return (x - mean) / std + this->bias; // .expand_as(x); // this->scale * (x - mean) / (std + eps) + this->bias;
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
    };
    TORCH_MODULE(MultiHeadedAttention);

    namespace transformer {
    }

    class Transformer : torch::nn::Module {
    public:
    };
}
