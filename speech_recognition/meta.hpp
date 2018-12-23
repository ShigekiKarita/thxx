#include <torch/torch.h>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>


namespace meta {
    namespace detail {
        template<size_t N>
        struct Apply {
            template<typename F, typename T, typename... A>
            static inline auto apply(F && f, T && t, A &&... a) {
                return Apply<N-1>::apply(::std::forward<F>(f), ::std::forward<T>(t),
                                         ::std::get<N-1>(::std::forward<T>(t)), ::std::forward<A>(a)...
                    );
            }
        };

        template<>
        struct Apply<0> {
            template<typename F, typename T, typename... A>
            static inline auto apply(F && f, T &&, A &&... a) {
                return ::std::forward<F>(f)(::std::forward<A>(a)...);
            }
        };

        template<typename F, typename T>
        inline auto apply(F && f, T && t) {
            return Apply< ::std::tuple_size< ::std::decay_t<T>
                                             >::value>::apply(::std::forward<F>(f), ::std::forward<T>(t));
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
            auto forward(const std::tuple<Args...>& args) {
                return apply(func, args);
            }
        };

        auto sequential_impl() {
            return [](auto&& x) { return x; };
        }

        template <typename A, typename ... Args>
        auto sequential_impl(A&& a, Args&& ... args) {
            return
                // NOTE: cannot forward these ? http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0780r0.html
                // [=, a=std::forward<A>(a), args=std::forward<Args>(args)...]
                [=]
                (auto&& ... x) mutable {
                    auto f = sequential_impl(std::forward<Args>(args)...);
                    return f(a->forward(x...));
                };
        }
    }

    template <typename Func>
    class Lambda : public torch::nn::ModuleHolder<detail::LambdaImpl<Func>> {
    public:
        using torch::nn::ModuleHolder<detail::LambdaImpl<Func>>::ModuleHolder;
    };

    template <typename Func>
    Lambda<Func> lambda(Func f) {
        return Lambda<Func>(f);
    }

    template <typename ... Args>
    auto sequential(Args&& ... args) {
        return lambda(detail::sequential_impl(std::forward<Args>(args)...));
    }
}
