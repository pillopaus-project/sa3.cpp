// rng.h — deterministic Gaussian noise, reproducible within sa3.cpp.
// std::mt19937 is standardized; Box-Muller is hand-rolled so output is identical
// across compilers/platforms for a given seed. (Not bit-matched to torch.randn —
// cross-framework seed parity is intentionally out of scope; see notes.)
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace sa3 {

struct Rng {
    std::mt19937 g;
    bool has_spare = false;
    float spare = 0.0f;

    explicit Rng(uint64_t seed) : g((uint32_t)seed) {}

    float uniform() { return ((g() >> 8) + 0.5f) * (1.0f / 16777216.0f); } // (0,1)

    float normal() {
        if (has_spare) { has_spare = false; return spare; }
        float u1 = uniform(), u2 = uniform();
        float r = sqrtf(-2.0f * logf(u1)), th = 6.283185307179586f * u2;
        spare = r * sinf(th); has_spare = true;
        return r * cosf(th);
    }

    void fill_normal(float* x, size_t n) { for (size_t i = 0; i < n; i++) x[i] = normal(); }
};

} // namespace sa3
