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

namespace thxx {

    namespace net {

        double accuracy(torch::Tensor output, torch::Tensor target, std::int64_t ignore_label) {
            AT_ASSERT(target.dim() + 1 == output.dim());
            for (std::int64_t i = 0; i < target.dim(); ++i) {
                AT_ASSERT(target.size(i) == output.size(i));
            }
            auto pred = output.argmax(-1);
            auto mask = target != ignore_label;
            auto num = torch::sum(pred.masked_select(mask) == target.masked_select(mask));;
            auto den = torch::sum(mask);
            return num.template item<double>() / den.template item<double>();
        }

        /// convert lengths {1, 2} to mask {{1, 0}, {1, 1}}
        static at::Tensor pad_mask(at::IntList lengths) {
            auto maxlen = *std::max_element(lengths.begin(), lengths.end());
            auto bs = static_cast<std::int64_t>(lengths.size());
            auto ret = at::zeros({bs, maxlen}, at::kByte);
            for (size_t i = 0; i < lengths.size(); ++i) {
                ret[i].slice(0, 0, lengths[i]) = 1;
            }
            return ret;
        }

        static at::Tensor subsequent_mask(std::int64_t size, at::DeviceType device = at::kCPU) {
            return at::ones({size, size}, at::kByte).to(device).tril_();
        }

        static auto label_smoothing_kl_div(torch::Tensor pred, torch::Tensor target, float smoothing=0, std::int64_t padding_idx=-1) {
            AT_ASSERT(0.0 <= smoothing);
            AT_ASSERT(smoothing <= 1.0);
            torch::Tensor true_dist, ignore_mask, n_valid;
            {
                torch::NoGradGuard no_grad;
                true_dist = at::empty_like(pred);
                auto n_false_class = pred.size(1) - 1; // only one for true
                true_dist.fill_(smoothing / n_false_class);
                ignore_mask = target == padding_idx;
                auto t = target.detach().clone();
                n_valid = (t != padding_idx).sum();
                // NOTE this is only worked for ctc loss using 0 index for eps
                // if (pa) t = t.masked_fill_(ignore_mask, 0); // avoid -1 index
                true_dist.scatter_(1, t.unsqueeze(1), 1.0 - smoothing);
            }
            auto kl = torch::kl_div(pred.log_softmax(1), true_dist, Reduction::None);
            return kl.masked_fill_(ignore_mask.unsqueeze(1), 0).sum() / n_valid;
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
                return this->scale * (x - mean) / std + this->bias;
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

            struct Config {
                // model
                std::int64_t d_model = 256;
                std::int64_t d_ff = 1024;
                std::int64_t heads = 4;
                std::int64_t elayers = 6;
                std::int64_t dlayers = 6;
                float dropout_rate = 0.1;
                float label_smoothing = 0.1;

                // training
                float lr = 10.0;
                std::int64_t warmup_steps = 25000;
                std::int64_t batch_size = 64;
                std::int64_t max_len_in = 512;
                std::int64_t max_len_out = 150;
            };


            static auto positionwise_feedforward(std::int64_t d_model, std::int64_t d_ff, float dropout_rate) {
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
                    AT_ASSERT(x.size(1) <= this->max_len);
                    auto y = this->scale * x + this->pe.slice(1, 0, x.size(1));
                    return this->dropout->forward(y);
                }
            };
            TORCH_MODULE(PositionalEncoding);


            class EncoderLayerImpl : public torch::nn::Cloneable<EncoderLayerImpl> {
            public:
                // configurations
                std::int64_t d_model;
                std::int64_t heads;
                std::int64_t d_ff;
                float dropout_rate;

                // submodules
                MultiHeadedAttention self_attn = nullptr;
                PositionwiseFeedforward pff = nullptr;
                torch::nn::Dropout dropout = nullptr;
                LayerNorm norm1 = nullptr;
                LayerNorm norm2 = nullptr;

                EncoderLayerImpl(Config c) : EncoderLayerImpl(c.d_model, c.heads, c.d_ff, c.dropout_rate) {}

                EncoderLayerImpl(std::int64_t d_model, std::int64_t heads, std::int64_t d_ff, float dropout_rate)
                    : d_model(d_model), heads(heads), d_ff(d_ff), dropout_rate(dropout_rate) {
                    this->reset();
                }

                void reset() override {
                    this->self_attn = register_module("self_attn", MultiHeadedAttention(heads, d_model, dropout_rate));
                    this->pff = register_module("pff", positionwise_feedforward(d_model, d_ff, dropout_rate));
                    this->dropout = this->register_module("dropout", torch::nn::Dropout(this->dropout_rate));
                    this->norm1 = register_module("norm1", LayerNorm(d_model));
                    this->norm2 = register_module("norm2", LayerNorm(d_model));
                }

                auto forward(torch::Tensor x, torch::Tensor mask) {
                    auto nx = this->norm1->forward(x);
                    x = x + this->dropout->forward(this->self_attn->forward(nx, nx, nx, mask));
                    nx = this->norm2->forward(x);
                    return std::make_tuple(x + this->dropout->forward(this->pff->forward(nx)), mask);
                }
            };
            TORCH_MODULE(EncoderLayer);


            class DecoderLayerImpl : public torch::nn::Cloneable<DecoderLayerImpl> {
            public:
                // configurations
                std::int64_t d_model;
                std::int64_t heads;
                std::int64_t d_ff;
                float dropout_rate;

                // submodules
                MultiHeadedAttention self_attn = nullptr;
                MultiHeadedAttention src_attn = nullptr;
                PositionwiseFeedforward pff = nullptr;
                torch::nn::Dropout dropout = nullptr;
                LayerNorm norm1 = nullptr;
                LayerNorm norm2 = nullptr;
                LayerNorm norm3 = nullptr;

                DecoderLayerImpl(Config c) : DecoderLayerImpl(c.d_model, c.heads, c.d_ff, c.dropout_rate) {}

                DecoderLayerImpl(std::int64_t d_model, std::int64_t heads, std::int64_t d_ff, float dropout_rate)
                    : d_model(d_model), heads(heads), d_ff(d_ff), dropout_rate(dropout_rate) {
                    this->reset();
                }

                void reset() override {
                    this->self_attn = register_module("self_attn", MultiHeadedAttention(heads, d_model, dropout_rate));
                    this->src_attn = register_module("src_attn", MultiHeadedAttention(heads, d_model, dropout_rate));
                    this->pff = register_module("pff", positionwise_feedforward(d_model, d_ff, dropout_rate));
                    this->dropout = this->register_module("dropout", torch::nn::Dropout(this->dropout_rate));
                    this->norm1 = register_module("norm1", LayerNorm(d_model));
                    this->norm2 = register_module("norm2", LayerNorm(d_model));
                    this->norm3 = register_module("norm3", LayerNorm(d_model));
                }

                auto forward(torch::Tensor tgt, torch::Tensor tgt_mask,
                             torch::Tensor memory, torch::Tensor memory_mask) {
                    auto nx = this->norm1->forward(tgt);
                    auto x = tgt + this->dropout->forward(this->self_attn->forward(nx, nx, nx, tgt_mask));
                    nx = this->norm2->forward(x);
                    x = x + this->dropout->forward(this->src_attn->forward(nx, memory, memory, memory_mask));
                    nx = this->norm3->forward(x);
                    x = x + this->dropout->forward(this->pff->forward(nx));
                    return std::make_tuple(x, tgt_mask);
                }
            };
            TORCH_MODULE(DecoderLayer);


            class Conv2dSubsamplingImpl : public torch::nn::Cloneable<Conv2dSubsamplingImpl> {
            public:
                // configurations
                std::int64_t n_freq;
                std::int64_t n_feat;
                float dropout_rate;

                // submodules
                torch::nn::Conv2d conv1 = nullptr;
                torch::nn::Conv2d conv2 = nullptr;
                torch::nn::Linear out = nullptr;
                PositionalEncoding pe = nullptr;

                Conv2dSubsamplingImpl(std::int64_t n_freq, std::int64_t n_feat, float dropout_rate)
                    : n_freq(n_freq), n_feat(n_feat), dropout_rate(dropout_rate) {
                    this->reset();
                }

                void reset() override {
                    auto c1 = torch::nn::Conv2dOptions(1, n_feat, 3).stride(2);
                    auto c2 = torch::nn::Conv2dOptions(n_feat, n_feat, 3).stride(2);
                    this->conv1 = register_module("conv1", torch::nn::Conv2d(c1));
                    this->conv2 = register_module("conv2", torch::nn::Conv2d(c2));
                    this->out = register_module("out", torch::nn::Linear(n_feat * (n_freq / 4), n_feat));
                    this->pe = register_module("pe", PositionalEncoding(n_feat, dropout_rate));
                }

                auto subsample_mask(torch::Tensor mask) {
                    for (std::int64_t i = 0; i < 2; ++i) {
                        mask = mask.slice(2, 0, mask.size(2) - 2, 2);
                    }
                    return mask;
                }

                auto forward(torch::Tensor x, torch::Tensor mask) {
                    AT_ASSERT(x.dim() == 3); // (b, t, f)
                    AT_ASSERT(x.size(2) == n_freq);
                    auto c1 = this->conv1->forward(x.unsqueeze(1)).relu();
                    auto c2 = this->conv2->forward(c1).relu();

                    auto n_batch = c2.size(0);
                    auto n_time = c2.size(2);
                    auto h = c2.transpose(1, 2).contiguous().view({n_batch, n_time, -1});
                    auto y = this->pe->forward(this->out->forward(h));
                    return std::make_tuple(y, this->subsample_mask(mask));
                }
            };
            TORCH_MODULE(Conv2dSubsampling);

            template <typename InputLayer>
            class EncoderImpl : public torch::nn::Cloneable<EncoderImpl<InputLayer>> {
            public:
                using torch::nn::Cloneable<EncoderImpl<InputLayer>>::register_module;

                // configurations
                std::int64_t idim;
                Config config;

                // submodules
                InputLayer input_layer = nullptr;
                std::vector<EncoderLayer> layers;
                LayerNorm norm = nullptr;

                EncoderImpl(std::int64_t idim, Config config)
                    : idim(idim), config(config) {
                    this->reset();
                }

                void reset() override {
                    this->input_layer = register_module("input_layer", InputLayer(this->idim, this->config.d_model, this->config.dropout_rate));
                    this->layers.reserve(this->config.elayers);
                    for (std::int64_t i = 0; i < this->config.elayers; ++i) {
                        this->layers.push_back(register_module("e" + std::to_string(i), EncoderLayer(this->config)));
                    }
                    this->norm = register_module("norm", LayerNorm(this->config.d_model));
                }

                auto forward(torch::Tensor x, torch::Tensor mask) {
                    std::tie(x, mask) = this->input_layer->forward(x, mask);
                    for (auto& l : this->layers) {
                        std::tie(x, mask) = l->forward(x, mask);
                    }
                    return std::make_tuple(this->norm->forward(x), mask);
                }
            };

            template <typename InputLayer>
            class Encoder : public torch::nn::ModuleHolder<EncoderImpl<InputLayer>> {
            public:
                using torch::nn::ModuleHolder<EncoderImpl<InputLayer>>::ModuleHolder;
            };


            // TODO implement ignore_idx with weight option
            class EmbedIDImpl : public torch::nn::Cloneable<EmbedIDImpl> {
            public:
                std::int64_t vocab;
                std::int64_t feat;
                torch::nn::Embedding embed = nullptr;

                EmbedIDImpl(std::int64_t vocab, std::int64_t feat)
                    : EmbedIDImpl(vocab, feat, 0) {}

                EmbedIDImpl(std::int64_t vocab, std::int64_t feat, float)
                    : vocab(vocab), feat(feat) {
                    this->reset();
                }

                void reset() override {
                    this->embed = register_module("embed", torch::nn::Embedding(vocab, feat));
                }

                auto forward(torch::Tensor x, torch::Tensor mask) {
                    return std::make_tuple(this->embed->forward(x), mask);
                }
            };
            TORCH_MODULE(EmbedID);

            class DecoderImpl : public torch::nn::Cloneable<DecoderImpl> {
            public:
                // configurations
                std::int64_t odim;
                Config config;

                // submodules
                EmbedID embed = nullptr;
                PositionalEncoding pe = nullptr;
                std::vector<DecoderLayer> layers;
                LayerNorm output_norm = nullptr;
                torch::nn::Linear output_layer = nullptr;

                DecoderImpl(std::int64_t odim, Config config)
                    : odim(odim), config(config) {
                    this->reset();
                }

                void reset() override {
                    this->embed = register_module("embed", EmbedID(this->odim, this->config.d_model));
                    this->pe = register_module("pe", PositionalEncoding(this->config.d_model, this->config.dropout_rate));
                    this->output_norm = register_module("output_norm", LayerNorm(this->config.d_model));
                    this->output_layer = register_module("output_layer", torch::nn::Linear(this->config.d_model, this->odim));
                    this->layers.reserve(this->config.elayers);
                    for (std::int64_t i = 0; i < this->config.elayers; ++i) {
                        this->layers.push_back(register_module("d" + std::to_string(i), DecoderLayer(this->config)));
                    }
                }

                auto forward(torch::Tensor tgt, torch::Tensor tgt_mask,
                             torch::Tensor memory, torch::Tensor memory_mask) {
                    auto [e, mask] = this->embed->forward(tgt, tgt_mask);
                    auto x = this->pe->forward(e);
                    for (auto& l : this->layers) {
                        std::tie(x, mask) = l->forward(x, mask, memory, memory_mask);
                    }
                    x = this->output_layer->forward(this->output_norm->forward(x));
                    return std::make_tuple(x, mask);
                }
            };
            TORCH_MODULE(Decoder);

        } // namespace transformer

        template <typename InputLayer>
        class Transformer : public torch::nn::Module {
        public:
            // configurations
            std::int64_t idim;
            std::int64_t odim;
            transformer::Config config;
            std::int64_t sos;
            std::int64_t eos;
            std::int64_t ignore_index;

            // submodules
            transformer::Encoder<InputLayer> encoder = nullptr;
            transformer::Decoder decoder = nullptr;

            Transformer(std::int64_t idim, std::int64_t odim, transformer::Config config)
                : idim(idim), odim(odim), config(config), sos(odim-1), eos(odim-1), ignore_index(odim) {
                this->encoder = register_module("encoder", transformer::Encoder<InputLayer>(idim, config));
                this->decoder = register_module("decoder", transformer::Decoder(odim + 1, config));
            }

            auto forward(torch::Tensor src, at::IntList src_length,
                         torch::Tensor tgt, at::IntList tgt_length) {
                auto src_mask = pad_mask(src_length).unsqueeze(-2);
                auto [mem, mem_mask] = this->encoder->forward(src, src_mask);

                auto tgt_mask = pad_mask(tgt_length).unsqueeze(-2);
                tgt_mask = tgt_mask.__and__(subsequent_mask(tgt_mask.size(-1)).unsqueeze(0));
                auto tgt_in = tgt.clone().fill_(this->ignore_index);
                auto tgt_out = tgt.clone().fill_(this->ignore_index); // NOTE fill eos?
                for (size_t i = 0; i < tgt_length.size(); ++i) {
                    auto n = tgt_length[i];
                    tgt_in[i][0] = this->sos;
                    tgt_in[i].slice(0, 1, n) = tgt[i].slice(0, 0, n - 1);

                    tgt_out[i].slice(0, 0, n - 1) = tgt[i].slice(0, 1, tgt_length[i]);
                    tgt_out[i][n - 1] = this->eos;
                }

                auto [pred, pred_mask] = this->decoder->forward(tgt_in, tgt_mask, mem, mem_mask);
                // TODO calc accuracy
                auto target = tgt_out.view({-1});
                auto loss = label_smoothing_kl_div(pred.view({target.size(0), -1}), target,
                                                   this->config.label_smoothing, this->ignore_index);
                return loss;
            }
        };
    } // namespace net

} //namespace thxx
