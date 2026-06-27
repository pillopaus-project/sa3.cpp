// nn.h — stateless GGML neural-net building blocks shared across the SA3 stack.
// Free functions over a ggml_context; no weights or state of their own.
#pragma once

#include "ggml.h"

#include <cstdlib>
#include <cstring>

namespace sa3::nn {

inline bool flash_attn_enabled() {
    const char* p = getenv("SA3_FLASH_ATTN");
    return p && strcmp(p, "0") != 0;
}

// DynamicTanh: y = tanh(alpha * x) * gamma + beta.
// alpha is [1] (broadcast); gamma/beta are [ne0] (broadcast over the rest).
inline ggml_tensor* dyt(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* alpha, ggml_tensor* gamma, ggml_tensor* beta) {
    ggml_tensor* y = ggml_tanh(ctx, ggml_mul(ctx, x, alpha));
    y = ggml_mul(ctx, y, gamma);
    y = ggml_add(ctx, y, beta);
    return y;
}

// RMSNorm with a learned gamma (no bias). Used by the DiT (Phase 4).
inline ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), gamma);
}

// Linear: w is [in,out] (ggml ne), x is [in,N] -> [out,N]; optional bias [out].
inline ggml_tensor* linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* b = nullptr) {
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// Partial NeoX rotary embedding over the first n_dims of each head.
// a: [head_dim, n_head, seq]; pos: I32 [seq].
inline ggml_tensor* rope_neox(ggml_context* ctx, ggml_tensor* a, ggml_tensor* pos,
                              int n_dims, float base) {
    return ggml_rope_ext(ctx, a, pos, nullptr, n_dims, GGML_ROPE_TYPE_NEOX,
                         /*n_ctx_orig=*/0, base, /*freq_scale=*/1.0f,
                         /*ext_factor=*/0.0f, /*attn_factor=*/1.0f,
                         /*beta_fast=*/0.0f, /*beta_slow=*/0.0f);
}

// Block-diagonal self-attention: q,k,v [d, N, H] with N = cc*nb; each block of cc tokens
// attends only within itself. Mathematically identical to dense sdpa with a block-diagonal
// mask, but O(N*cc) instead of O(N^2) — the SAME-S (chunked) decoder/encoder path.
inline ggml_tensor* attn_blockdiag(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k,
                                   ggml_tensor* v, int cc, float scale) {
    const int64_t d = q->ne[0], N = q->ne[1], H = q->ne[2], nb = N / cc;
    auto blk = [&](ggml_tensor* t){ return ggml_reshape_4d(ctx, t, d, cc, nb, H); };   // [d, cc, nb, H]
    ggml_tensor* kq = ggml_mul_mat(ctx, blk(k), blk(q));               // [cc(k), cc(q), nb, H]
    kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);            // softmax over keys, no mask
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, blk(v), 1, 0, 2, 3)); // [cc, d, nb, H]
    ggml_tensor* o = ggml_mul_mat(ctx, vt, kq);                       // [d, cc, nb, H]
    return ggml_reshape_3d(ctx, o, d, N, H);
}

// Scaled-dot-product attention with an additive mask.
// q,k,v: [d, N, H]; mask: [Nk, Nq] (0 / -inf). Returns [d, Nq, H].
inline ggml_tensor* sdpa(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                         ggml_tensor* mask, float scale) {
    if (flash_attn_enabled() && (!mask || mask->type == GGML_TYPE_F16)) {
        ggml_tensor* out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3)); // [d, Nq, H, 1]
        return ggml_reshape_3d(ctx, out, q->ne[0], q->ne[1], q->ne[2]);
    }
    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);                 // [Nk, Nq, H]
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);        // softmax over Nk
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [Nk, d, H]
    return ggml_mul_mat(ctx, vt, kq);                          // [d, Nq, H]
}

} // namespace sa3::nn
