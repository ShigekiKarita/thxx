#pragma once
#include <chrono>

namespace thxx::chrono {
    namespace C = std::chrono;

    struct StopWatch {
        const C::high_resolution_clock::time_point start;

        using C::high_resolution_clock::now;

        StopWatch() : start(now()) {}

        double elapsed() const {
            return 1e-9 * C::duration_cast<C::nanoseconds>(now() - this->start).count();
        }
    };
}
