#include <torch/torch.h>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>


namespace meta {
    namespace detail {
        template<size_t N>
        struct TupleApply {
            template<typename F, typename T, typename... A>
            static inline auto apply(F && f, T && t, A &&... a) {
                return TupleApply<N-1>::apply(
                    std::forward<F>(f), std::forward<T>(t),
                    std::get<N-1>(std::forward<T>(t)), std::forward<A>(a)...
                    );
            }
        };

        template<>
        struct TupleApply<0> {
            template<typename F, typename T, typename... A>
            static inline auto apply(F && f, T &&, A &&... a) {
                return std::forward<F>(f)(std::forward<A>(a)...);
            }
        };

        template<typename F, typename T>
        inline auto tuple_apply(F && f, T && t) {
            constexpr auto N = std::tuple_size< std::decay_t<T> > ::value;
            return TupleApply<N>::apply(std::forward<F>(f), std::forward<T>(t));
        }

        template <typename Func>
        class LambdaImpl : public torch::nn::Module {
        public:
            Func func;

            LambdaImpl() {}
            LambdaImpl(Func f) : func(f) {}

            template <typename ... Args>
            auto forward(Args&&... args) {
                return func(std::forward<Args>(args)...);
            }

            template <typename ... Args>
            auto forward(std::tuple<Args...> args) {
                // TODO perfect forward with SFINAE enable_if<is_tuple<...>>
                return tuple_apply(func, args);
            }

            template <size_t>
            void register_modules() {}

            template <size_t i, typename A, typename ... Args>
            void register_modules(A&& a, Args&& ... args) {
                this->register_module(std::to_string(i), std::forward<A>(a));
                this->template register_modules<i + 1>(std::forward<Args>(args)...);
            }
        };

        auto sequential_impl() {
            return [](auto&& x) { return x; };
        }

        template <typename A, typename ... Args>
        auto sequential_impl(A&& a, Args&& ... args) {
            return
                [=](auto&& ... x) mutable {
                    return sequential_impl(std::forward<Args>(args)...)(
                        a->forward(std::forward<decltype(x)>(x)...));
                };
        }
    }

    template <typename Func>
    class Lambda : public torch::nn::ModuleHolder<detail::LambdaImpl<Func>> {
    public:
        using torch::nn::ModuleHolder<detail::LambdaImpl<Func>>::ModuleHolder;
    };

    template <typename Func>
    Lambda<Func> lambda(Func&& f) {
        return Lambda<Func>(std::forward<Func>(f));
    }

    template <typename ... Args>
    auto sequential(Args&& ... args) {
        auto ret = lambda(detail::sequential_impl(args...));
        ret->template register_modules<0>(std::forward<Args>(args)...);
        return ret;
    }

    template <typename ... Modules>
    using Seq = decltype(sequential(std::declval<Modules>()...));
}
