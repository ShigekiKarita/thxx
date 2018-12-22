#pragma once

#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <algorithm>
#include <utility>

/// for kaldi
#include <kaldi-io.h>
#include <kaldi-table.h>
#include <kaldi-matrix.h>
#include <table-types.h>

/// for json
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/document.h>
#include <kaldi-matrix.h>


namespace traits {
    template <typename T>
    struct ScalarTypeof;

    using at::Half;

#define SCALAR_TYPE_OF(_1,n,_2)                                    \
    template <> struct ScalarTypeof<_1> { constexpr static at::ScalarType value = at::k##n ; };
    AT_FORALL_SCALAR_TYPES(SCALAR_TYPE_OF)
#undef SCALAR_TYPE_OF

    // template <typename T>
    // constexpr at::ScalarType scalar_typeof = ScalarTypeof<T>::value;
}


namespace memory {
    // note: this implementation does not disable this overload for array types
    template<typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
    }

    template <typename T, typename Elem>
    at::Tensor make_tensor(std::shared_ptr<T> ptr, Elem* begin, at::IntList dims, at::IntList strides={}) {
        constexpr auto s = traits::ScalarTypeof<Elem>::value;
        auto deleter = [ptr](void*) mutable { ptr.reset(); };
        if (strides.size() > 0) {
            return at::CPU(s).tensorFromBlob(begin, dims, strides, deleter);
        } else {
            return at::CPU(s).tensorFromBlob(begin, dims, deleter);
        }
    }

    template <typename Elem>
    at::Tensor make_tensor(std::shared_ptr<kaldi::Vector<Elem>> ptr) {
        return make_tensor(ptr, ptr->Data(), ptr->Dim());
    }

    template <typename Elem>
    at::Tensor make_tensor(std::shared_ptr<kaldi::Matrix<Elem>> ptr) {
        return make_tensor(ptr, ptr->Data(), {ptr->NumRows(), ptr->NumCols()}, {ptr->Stride(), 1});
    }
}


namespace dataset {

    /// short cut for easy access
    using InputReader = kaldi::RandomAccessBaseFloatMatrixReader;
    using InputReaderPtr = std::shared_ptr<InputReader>;
    using DocPtr = std::shared_ptr<rapidjson::Document>;
    using DocIter = typename rapidjson::Document::ConstMemberIterator;


    /// read 1d token-id target from json
    at::Tensor read_target(DocIter iter) {
        const auto& target = iter->value["output"][0];
        auto olen = target["shape"][0].GetInt();

        auto t = at::zeros({olen}, at::kLong);
        std::string id;
        long i = 0;
        std::istringstream iss(target["tokenid"].GetString());
        while (std::getline(iss, id, ' ')) {
            t[i] = std::stoi(id);
            ++i;
        }
        AT_ASSERT(i == olen);
        return t;
    }

    /// read 2d time-freq input (e.g., FBANK feature) using kaldi
    at::Tensor read_input(InputReaderPtr reader, DocIter iter) {
        // TODO assert reader has utt-id
        auto m = std::make_shared<kaldi::Matrix<float>>(reader->Value(iter->name.GetString()));
        return memory::make_tensor(m);
    }

    /// Handle sample information in json to read input and target tensors
    struct Sample {
        DocPtr doc; // keep this for iter
        InputReaderPtr reader;
        DocIter iter;
        std::int64_t ilen, idim, olen, odim;

        const rapidjson::Value& get(const char* query) const {
            AT_ASSERT(iter->value.HasMember(query));
            return iter->value[query];
        }

        std::string key() const {
            AT_ASSERT(iter->name.IsString());
            return iter->name.GetString();
        }

        at::Tensor input() const {
            AT_ASSERT(doc);
            AT_ASSERT(reader);
            return read_input(reader, iter);
        }

        at::Tensor target() const {
            AT_ASSERT(doc);
            return read_target(iter);
        }
    };

    /// Gather input and target in sorted order by the length, and combine them into minibatch
    std::vector<std::vector<Sample>>
    make_batchset(DocPtr doc, InputReaderPtr reader, size_t batch_size=32,
                  size_t max_length_in=800, size_t max_length_out=150,
                  size_t max_num_batches=std::numeric_limits<size_t>::max()) {
        // read json
        std::vector<Sample> keys;
        auto& data = doc->FindMember("utts")->value;
        for (auto d = data.MemberBegin(); d != data.MemberEnd(); ++d) {
            std::cout << d->name.GetString() << std::endl;
            Sample s = {
                doc, reader, d,
                d->value["input"][0]["shape"][0].GetInt(),
                d->value["input"][0]["shape"][1].GetInt(),
                d->value["output"][0]["shape"][0].GetInt(),
                d->value["output"][0]["shape"][1].GetInt()
            };
            auto t = s.input();
            std::cout << t.sizes() << std::endl;
            auto y = s.target();
            std::cout << y.sizes() << std::endl;
            keys.push_back(s);
        }

        // shorter first
        std::sort(keys.begin(), keys.end(),
                  [](const Sample& a, const Sample& b) {
                      return a.olen < b.olen;
                  });

        // merge samples into minibatches
        std::vector<std::vector<Sample>> batchset;
        size_t start_id = 0;
        while (true) {
            const auto& start = keys[start_id];
            auto factor = std::max<size_t>(start.ilen / max_length_in, start.olen / max_length_out);
            auto b = std::max<size_t>(1, batch_size / (1 + factor));
            auto end_id = std::min<size_t>(keys.size(), start_id + b);
            std::vector<Sample> mb(keys.begin() + start_id, keys.begin() + end_id);
            batchset.push_back(mb);
            if (end_id == keys.size() || batchset.size() > max_num_batches) break;
            start_id = end_id;
        }
        return batchset;
    }

    /// Keep tensor unique_ptr for input/target as two padded tensors in a minibatch
    struct MiniBatch {
        MiniBatch(const std::vector<Sample>& minibatch) {
            this->input_lengths.reserve(minibatch.size());
            this->target_lengths.reserve(minibatch.size());
            std::int64_t max_ilen = 0, max_olen = 0;
            for (const auto& sample : minibatch) {
                this->input_lengths.push_back(sample.ilen);
                this->target_lengths.push_back(sample.olen);
                if (sample.ilen > max_ilen) max_ilen = sample.ilen;
                if (sample.olen > max_olen) max_olen = sample.olen;
            }
            auto mb_size = static_cast<std::int64_t>(minibatch.size());
            // maybe the bug of at::Tensor, memory leaks unless this unique_ptr
            // TODO make Variable().data unique_ptr
            this->inputs = memory::make_unique<at::Tensor>(at::zeros({mb_size, max_ilen, minibatch.front().idim}));
            this->targets = memory::make_unique<at::Tensor>(at::zeros({mb_size, max_olen}, at::kLong));
            for (size_t batch_idx = 0; batch_idx < minibatch.size(); ++batch_idx) {
                auto x = minibatch[batch_idx].input();
                auto t = minibatch[batch_idx].target();
                (*inputs)[batch_idx].slice(0, 0, x.size(0)) = x;
                (*targets)[batch_idx].slice(0, 0, t.size(0)) = t;
            }
        }

        std::unique_ptr<at::Tensor> inputs, targets;
        std::vector<std::int64_t> input_lengths, target_lengths;
    };

    /// read a json from a filename
    std::shared_ptr<rapidjson::Document> read_json(const std::string& filename) {
        auto doc = std::make_shared<rapidjson::Document>();
        std::ifstream ifs(filename);
        rapidjson::IStreamWrapper isw(ifs);
        doc->ParseStream(isw);
        return doc;
    }

    std::shared_ptr<kaldi::RandomAccessBaseFloatMatrixReader>
    open_scp(const std::string& filename) {
        return std::make_shared<kaldi::RandomAccessBaseFloatMatrixReader>("scp:" + filename);
    }
}
