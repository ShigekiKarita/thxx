#pragma once
#include <ATen/ATen.h>

namespace thxx {
    namespace traits {
        template <typename T>
        struct ScalarTypeof;

        using at::Half;

#define SCALAR_TYPE_OF(_1,n,_2)                                         \
        template <> struct ScalarTypeof<_1> { constexpr static at::ScalarType value = at::k##n ; };
        AT_FORALL_SCALAR_TYPES(SCALAR_TYPE_OF)
#undef SCALAR_TYPE_OF

        // template <typename T>
        // constexpr at::ScalarType scalar_typeof = ScalarTypeof<T>::value;
    }
}
