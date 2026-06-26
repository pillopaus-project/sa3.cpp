// t5gemma.h — the T5Gemma (google/t5gemma-b-b-ul2) ENCODER in GGML.
// Gemma2-style bidirectional encoder: sandwiched RMSNorms, full MHA with RoPE +
// attention-logit soft-capping, GeGLU MLP. Produces last_hidden_state for the
// DiT cross-attention conditioning. Norm weights carry Gemma's (1+w) baked in.
#pragma once

#include "ggml.h"
#include "gguf_model.h"
#include "nn.h"

#include <cmath>
#include <string>

namespace sa3 {

struct T5GemmaConfig {
    int dim, layers, heads, head_dim, intermediate, vocab;
    float eps, rope_theta, attn_softcap, query_scalar, normalizer;

    static T5GemmaConfig from(const GgufModel& m) {
        T5GemmaConfig c;
        c.dim          = m.u32("t5g.dim");
        c.layers       = m.u32("t5g.layers");
        c.heads        = m.u32("t5g.heads");
        c.head_dim     = m.u32("t5g.head_dim");
        c.intermediate = m.u32("t5g.intermediate");
        c.vocab        = m.u32("t5g.vocab");
        c.eps          = m.f32("t5g.eps");
        c.rope_theta   = m.f32("t5g.rope_theta");
        c.attn_softcap = m.f32("t5g.attn_softcap");
        c.query_scalar = m.f32("t5g.query_scalar");
        c.normalizer   = m.f32("t5g.normalizer");
        return c;
    }
};

// Encode token ids -> last_hidden_state [dim, seq].
// ids: I32 [seq]; pos: I32 [seq]; mask: F32 [seq, seq] (0 / -inf, key padding, bidirectional).
inline ggml_tensor* t5gemma_encode(ggml_context* ctx, const GgufModel& W, ggml_tensor* ids,
                                   ggml_tensor* pos, ggml_tensor* mask, const T5GemmaConfig& c) {
    const int dim = c.dim, hd = c.head_dim, nh = c.heads;
    const int64_t seq = ids->ne[0];
    const float scaling = 1.0f / sqrtf(c.query_scalar);
    const float softcap = c.attn_softcap;

    // token embedding * sqrt(dim)
    ggml_tensor* x = ggml_get_rows(ctx, W.get("te.embed.weight"), ids);   // [dim, seq]
    x = ggml_scale(ctx, x, c.normalizer);

    for (int l = 0; l < c.layers; l++) {
        const std::string p = "te." + std::to_string(l) + ".";

        // --- self-attention (sandwiched RMSNorm) ---
        ggml_tensor* res = x;
        ggml_tensor* h = nn::rms_norm(ctx, x, W.get(p+"attn_norm.weight"), c.eps);
        ggml_tensor* q = nn::linear(ctx, W.get(p+"q.weight"), h);
        ggml_tensor* k = nn::linear(ctx, W.get(p+"k.weight"), h);
        ggml_tensor* v = nn::linear(ctx, W.get(p+"v.weight"), h);
        q = ggml_reshape_3d(ctx, q, hd, nh, seq);
        k = ggml_reshape_3d(ctx, k, hd, nh, seq);
        v = ggml_reshape_3d(ctx, v, hd, nh, seq);
        q = nn::rope_neox(ctx, q, pos, hd, c.rope_theta);
        k = nn::rope_neox(ctx, k, pos, hd, c.rope_theta);
        auto toAttn = [&](ggml_tensor* a){ return ggml_cont(ctx, ggml_permute(ctx, a, 0, 2, 1, 3)); }; // [hd,seq,nh]
        q = toAttn(q); k = toAttn(k); v = toAttn(v);

        ggml_tensor* kq = ggml_mul_mat(ctx, k, q);                 // [seq_k, seq_q, nh]
        kq = ggml_scale(ctx, kq, scaling);
        // attention-logit soft-capping: softcap * tanh(kq / softcap)
        kq = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, kq, 1.0f/softcap)), softcap);
        kq = ggml_soft_max_ext(ctx, kq, mask, 1.0f, 0.0f);
        ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [seq_k, hd, nh]
        ggml_tensor* o = ggml_mul_mat(ctx, vt, kq);                // [hd, seq_q, nh]
        o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));      // [hd, nh, seq]
        o = ggml_reshape_2d(ctx, o, dim, seq);
        o = nn::linear(ctx, W.get(p+"o.weight"), o);
        o = nn::rms_norm(ctx, o, W.get(p+"attn_post_norm.weight"), c.eps);
        x = ggml_add(ctx, res, o);

        // --- GeGLU MLP (sandwiched RMSNorm) ---
        res = x;
        ggml_tensor* f = nn::rms_norm(ctx, x, W.get(p+"ffn_norm.weight"), c.eps);
        ggml_tensor* gate = ggml_gelu(ctx, nn::linear(ctx, W.get(p+"gate.weight"), f)); // gelu-tanh
        ggml_tensor* up   = nn::linear(ctx, W.get(p+"up.weight"), f);
        f = ggml_mul(ctx, gate, up);
        f = nn::linear(ctx, W.get(p+"down.weight"), f);
        f = nn::rms_norm(ctx, f, W.get(p+"ffn_post_norm.weight"), c.eps);
        x = ggml_add(ctx, res, f);
    }

    return nn::rms_norm(ctx, x, W.get("te.norm.weight"), c.eps);   // [dim, seq]
}

} // namespace sa3
