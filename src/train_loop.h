// train_loop.h - single-step training loop helpers.
#pragma once

#include "sa3_pipeline.h"
#include "train_conditioning.h"
#include "train_diffusion.h"
#include "train_dit.h"
#include "train_lora.h"
#include "train_optimizer.h"

#include <string>
#include <vector>

namespace sa3 {

struct TrainLoopState {
    int step = 0;
    std::vector<TrainAdamWTensorState> A_state;
    std::vector<TrainAdamWTensorState> B_state;
    std::vector<TrainAdamWTensorState> mxs_state;
    std::vector<TrainAdamWTensorState> mag_state;
    std::vector<TrainAdamWTensorState> mag_r_state;
    std::vector<TrainAdamWTensorState> mag_c_state;
};

struct TrainLoraGradAccum {
    int count = 0;
    std::vector<std::vector<float>> A;
    std::vector<std::vector<float>> B;
    std::vector<std::vector<float>> mxs;
    std::vector<std::vector<float>> mag;
    std::vector<std::vector<float>> mag_r;
    std::vector<std::vector<float>> mag_c;
};

inline void upload_train_lora_state(const TrainLoraState& state, const TrainDitGraph& graph) {
    for (size_t i = 0; i < state.params.size(); ++i) {
        const TrainLoraParam& hp = state.params[i];
        const TrainDitParamTensors& tp = graph.params[i];
        if (tp.lora_A) ggml_backend_tensor_set(tp.lora_A, hp.lora_A.data(), 0, hp.lora_A.size() * sizeof(float));
        if (tp.lora_B) ggml_backend_tensor_set(tp.lora_B, hp.lora_B.data(), 0, hp.lora_B.size() * sizeof(float));
        if (tp.U) ggml_backend_tensor_set(tp.U, hp.U.data(), 0, hp.U.size() * sizeof(float));
        if (tp.V) ggml_backend_tensor_set(tp.V, hp.V.data(), 0, hp.V.size() * sizeof(float));
        if (tp.M_xs) ggml_backend_tensor_set(tp.M_xs, hp.M_xs.data(), 0, hp.M_xs.size() * sizeof(float));
        if (tp.magnitude) ggml_backend_tensor_set(tp.magnitude, hp.magnitude.data(), 0, hp.magnitude.size() * sizeof(float));
        if (tp.magnitude_r) ggml_backend_tensor_set(tp.magnitude_r, hp.magnitude_r.data(), 0, hp.magnitude_r.size() * sizeof(float));
        if (tp.magnitude_c) ggml_backend_tensor_set(tp.magnitude_c, hp.magnitude_c.data(), 0, hp.magnitude_c.size() * sizeof(float));
    }
}

inline bool train_read_grad(ggml_cgraph* graph, ggml_tensor* param, std::vector<float>& grad,
                            bool& found, std::string& err) {
    ggml_tensor* g = ggml_graph_get_grad(graph, param);
    if (!g) g = ggml_graph_get_grad_acc(graph, param);
    if (!g) {
        found = false;
        grad.assign((size_t)ggml_nelements(param), 0.0f);
        return true;
    }
    found = true;
    grad.resize((size_t)ggml_nelements(param));
    ggml_backend_tensor_get(g, grad.data(), 0, grad.size() * sizeof(float));
    return true;
}

inline void train_accum_add(std::vector<float>& dst, const std::vector<float>& grad) {
    if (dst.empty()) dst.assign(grad.size(), 0.0f);
    for (size_t i = 0; i < grad.size(); ++i) dst[i] += grad[i];
}

inline void train_clear_accum(TrainLoraGradAccum& accum) {
    accum = TrainLoraGradAccum{};
}

inline bool train_accumulate_adamw_gradients(TrainDitGraph& graph, const TrainLoraState& state,
                                             TrainLoraGradAccum& accum, std::string& err) {
    accum.A.resize(state.params.size());
    accum.B.resize(state.params.size());
    accum.mxs.resize(state.params.size());
    accum.mag.resize(state.params.size());
    accum.mag_r.resize(state.params.size());
    accum.mag_c.resize(state.params.size());
    std::vector<float> grad;
    bool any_grad = false;
    bool found = false;
    for (size_t i = 0; i < state.params.size(); ++i) {
        const TrainDitParamTensors& tp = graph.params[i];
        if (tp.lora_A) {
            if (!train_read_grad(graph.graph, tp.lora_A, grad, found, err)) return false;
            any_grad = any_grad || found;
            train_accum_add(accum.A[i], grad);
        }
        if (tp.lora_B) {
            if (!train_read_grad(graph.graph, tp.lora_B, grad, found, err)) return false;
            any_grad = any_grad || found;
            train_accum_add(accum.B[i], grad);
        }
        if (tp.M_xs) {
            if (!train_read_grad(graph.graph, tp.M_xs, grad, found, err)) return false;
            any_grad = any_grad || found;
            train_accum_add(accum.mxs[i], grad);
        }
        if (tp.magnitude) {
            if (!train_read_grad(graph.graph, tp.magnitude, grad, found, err)) return false;
            any_grad = any_grad || found;
            train_accum_add(accum.mag[i], grad);
        }
        if (tp.magnitude_r) {
            if (!train_read_grad(graph.graph, tp.magnitude_r, grad, found, err)) return false;
            any_grad = any_grad || found;
            train_accum_add(accum.mag_r[i], grad);
        }
        if (tp.magnitude_c) {
            if (!train_read_grad(graph.graph, tp.magnitude_c, grad, found, err)) return false;
            any_grad = any_grad || found;
            train_accum_add(accum.mag_c[i], grad);
        }
    }
    if (!any_grad) {
        err = "training graph produced no adapter gradients";
        return false;
    }
    accum.count += 1;
    return true;
}

inline void train_average_grad(std::vector<float>& grad, int count) {
    if (count <= 1) return;
    const float inv = 1.0f / (float)count;
    for (float& v : grad) v *= inv;
}

// Average every accumulated gradient by count, then (if grad_clip > 0) scale them all so the
// GLOBAL L2 norm across every trainable vector is <= grad_clip. Matches torch clip_grad_norm_
// over all LoRA params, applied to the mean gradient of the accumulation window.
inline void train_average_and_clip(TrainLoraGradAccum& accum, const TrainAdamWParams& hp) {
    const int count = accum.count;
    auto avg = [&](std::vector<float>& g) { if (count > 1) { const float inv = 1.0f/(float)count; for (float& v : g) v *= inv; } };
    for (size_t i = 0; i < accum.A.size(); ++i) {
        avg(accum.A[i]); avg(accum.B[i]); avg(accum.mxs[i]);
        avg(accum.mag[i]); avg(accum.mag_r[i]); avg(accum.mag_c[i]);
    }
    if (hp.grad_clip <= 0.0f) return;
    double sumsq = 0.0;
    auto acc = [&](const std::vector<float>& g) { for (float v : g) sumsq += (double)v * v; };
    for (size_t i = 0; i < accum.A.size(); ++i) {
        acc(accum.A[i]); acc(accum.B[i]); acc(accum.mxs[i]);
        acc(accum.mag[i]); acc(accum.mag_r[i]); acc(accum.mag_c[i]);
    }
    const double norm = std::sqrt(sumsq);
    if (norm <= (double)hp.grad_clip) return;
    const float scale = (float)((double)hp.grad_clip / (norm + 1e-6));
    auto scl = [&](std::vector<float>& g) { for (float& v : g) v *= scale; };
    for (size_t i = 0; i < accum.A.size(); ++i) {
        scl(accum.A[i]); scl(accum.B[i]); scl(accum.mxs[i]);
        scl(accum.mag[i]); scl(accum.mag_r[i]); scl(accum.mag_c[i]);
    }
}

inline bool train_apply_accumulated_adamw(TrainLoraState& state, TrainLoraGradAccum& accum,
                                          TrainLoopState& loop, const TrainAdamWParams& hp,
                                          std::string& err) {
    if (accum.count <= 0) {
        err = "cannot apply empty gradient batch";
        return false;
    }
    loop.step += 1;
    loop.A_state.resize(state.params.size());
    loop.B_state.resize(state.params.size());
    loop.mxs_state.resize(state.params.size());
    loop.mag_state.resize(state.params.size());
    loop.mag_r_state.resize(state.params.size());
    loop.mag_c_state.resize(state.params.size());
    train_average_and_clip(accum, hp);   // mean over the accumulation window + global grad-norm clip
    for (size_t i = 0; i < state.params.size(); ++i) {
        TrainLoraParam& hpv = state.params[i];
        if (!hpv.lora_A.empty()) {
            if (!adamw_update_vector(hpv.lora_A, accum.A[i], loop.A_state[i], hp, loop.step, err)) return false;
        }
        if (!hpv.lora_B.empty()) {
            if (!adamw_update_vector(hpv.lora_B, accum.B[i], loop.B_state[i], hp, loop.step, err)) return false;
        }
        if (!hpv.M_xs.empty()) {
            if (!adamw_update_vector(hpv.M_xs, accum.mxs[i], loop.mxs_state[i], hp, loop.step, err)) return false;
        }
        if (!hpv.magnitude.empty()) {
            if (!adamw_update_vector(hpv.magnitude, accum.mag[i], loop.mag_state[i], hp, loop.step, err)) return false;
        }
        if (!hpv.magnitude_r.empty()) {
            if (!adamw_update_vector(hpv.magnitude_r, accum.mag_r[i], loop.mag_r_state[i], hp, loop.step, err)) return false;
        }
        if (!hpv.magnitude_c.empty()) {
            if (!adamw_update_vector(hpv.magnitude_c, accum.mag_c[i], loop.mag_c_state[i], hp, loop.step, err)) return false;
        }
    }
    train_clear_accum(accum);
    return true;
}

inline bool run_train_dit_accumulate(ggml_backend_t backend, TrainDitGraph& graph, const TrainLoraState& lora,
                                     TrainLoraGradAccum& accum,
                                     const TrainDiffusionSample& sample, const TrainConditioning& cond,
                                     const DitConfig& dc, float& loss_out, std::string& err) {
    if (sample.x_t.empty() || sample.velocity_target.empty() || cond.cross.empty() || cond.global.empty()) {
        err = "training step inputs are empty";
        return false;
    }
    upload_train_lora_state(lora, graph);
    std::vector<float> tf;
    expo_features(sample.t, tf, dc.time_dim, dc.time_min_freq, dc.time_max_freq);
    std::vector<int32_t> pos((size_t)(dc.mem_tokens + graph.target->ne[1]));
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = (int32_t)i;
    const float one = 1.0f;
    ggml_backend_tensor_set(graph.x, sample.x_t.data(), 0, sample.x_t.size() * sizeof(float));
    ggml_backend_tensor_set(graph.target, sample.velocity_target.data(), 0, sample.velocity_target.size() * sizeof(float));
    ggml_backend_tensor_set(graph.tfeat, tf.data(), 0, tf.size() * sizeof(float));
    ggml_backend_tensor_set(graph.cross, cond.cross.data(), 0, cond.cross.size() * sizeof(float));
    ggml_backend_tensor_set(graph.global, cond.global.data(), 0, cond.global.size() * sizeof(float));
    ggml_backend_tensor_set(graph.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(graph.ones, &one, 0, sizeof(float));
    ggml_graph_reset(graph.graph);
    ggml_backend_graph_compute(backend, graph.graph);
    ggml_backend_tensor_get(graph.loss, &loss_out, 0, sizeof(float));
    return train_accumulate_adamw_gradients(graph, lora, accum, err);
}

inline bool run_train_dit_step(ggml_backend_t backend, TrainDitGraph& graph, TrainLoraState& lora,
                               TrainLoopState& loop, const TrainAdamWParams& opt,
                               const TrainDiffusionSample& sample, const TrainConditioning& cond,
                               const DitConfig& dc, float& loss_out, std::string& err) {
    TrainLoraGradAccum accum;
    if (!run_train_dit_accumulate(backend, graph, lora, accum, sample, cond, dc, loss_out, err)) return false;
    return train_apply_accumulated_adamw(lora, accum, loop, opt, err);
}

inline bool run_eval_dit_loss(ggml_backend_t backend, TrainDitGraph& graph, const TrainLoraState& lora,
                              const TrainDiffusionSample& sample, const TrainConditioning& cond,
                              const DitConfig& dc, float& loss_out, std::string& err) {
    if (sample.x_t.empty() || sample.velocity_target.empty() || cond.cross.empty() || cond.global.empty()) {
        err = "evaluation inputs are empty";
        return false;
    }
    upload_train_lora_state(lora, graph);
    std::vector<float> tf;
    expo_features(sample.t, tf, dc.time_dim, dc.time_min_freq, dc.time_max_freq);
    std::vector<int32_t> pos((size_t)(dc.mem_tokens + graph.target->ne[1]));
    for (size_t i = 0; i < pos.size(); ++i) pos[i] = (int32_t)i;
    const float one = 1.0f;
    ggml_backend_tensor_set(graph.x, sample.x_t.data(), 0, sample.x_t.size() * sizeof(float));
    ggml_backend_tensor_set(graph.target, sample.velocity_target.data(), 0, sample.velocity_target.size() * sizeof(float));
    ggml_backend_tensor_set(graph.tfeat, tf.data(), 0, tf.size() * sizeof(float));
    ggml_backend_tensor_set(graph.cross, cond.cross.data(), 0, cond.cross.size() * sizeof(float));
    ggml_backend_tensor_set(graph.global, cond.global.data(), 0, cond.global.size() * sizeof(float));
    ggml_backend_tensor_set(graph.pos, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(graph.ones, &one, 0, sizeof(float));
    ggml_graph_reset(graph.graph);
    ggml_backend_graph_compute(backend, graph.graph);
    ggml_backend_tensor_get(graph.loss, &loss_out, 0, sizeof(float));
    return true;
}

} // namespace sa3
