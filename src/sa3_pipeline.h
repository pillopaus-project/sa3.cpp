// sa3_pipeline.h — the reusable SA3 generation pipeline.
//
// Load the four nets + conditioner ONCE, then call generate() per request. This is the single
// code path behind all three surfaces, so they share behavior + the config/resolution vocabulary:
//   - sa3-generate (CLI)   : parse args -> Pipeline::load -> generate -> write_wav
//   - sa3-server (HTTP)    : load once -> generate() per POST /generate
//   - libsa3 (C ABI)       : sa3_init -> sa3_generate -> sa3_free, wrapping a Pipeline
//
// Residency is a per-request policy (GenParams::keep_models), NOT a surface-specific thing — the
// server and the C API both expose it:
//   - keep_models=true  (resident): nothing freed between requests -> lowest latency, but peak VRAM
//                                    caps gen length on small cards. Best for back-to-back ~12s clips.
//   - keep_models=false (frugal):   free T5 before sampling + DiT before decode (the CLI's trick), then
//                                    reload them at the next generate() (~0.5-1.5s). Costs a reload per
//                                    request but fits long-form on 8 GB (e.g. gary4local). generate()
//                                    lazily (re)loads whatever a prior frugal call freed, so the flag
//                                    can vary per call. Reloading the DiT also re-bases it for that
//                                    request's LoRAs.
//
// Threading: generate() runs one ggml graph at a time and mutates the DiT's LoRA overrides, so a
// single Pipeline is NOT safe to call concurrently — the server serializes requests (a queue).
//
// Style: header-only (inline) to match the rest of src/ — the CLI, server, and C API just #include
// this. (Open question for the docs: revisit a sa3_pipeline.cpp if incremental build time bites.)
#pragma once

#include "gguf_model.h"
#include "tokenizer.h"
#include "t5gemma.h"
#include "dit.h"
#include "same_ae.h"
#include "lora.h"

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sa3 {

// Find the one file in `dir` whose name starts with `prefix` and ends with `suffix`. "" if none;
// returns "" + sets ambiguous=true if >1 match. Resolves --model / --lora by the naming convention.
inline std::string resolve_one(const std::string& dir, const std::string& prefix,
                               const std::string& suffix, bool* ambiguous = nullptr) {
    namespace fs = std::filesystem;
    std::string found; std::error_code ec;
    if (ambiguous) *ambiguous = false;
    if (!fs::is_directory(dir, ec)) return found;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        const std::string n = e.path().filename().string();
        if (n.size() >= prefix.size() + suffix.size()
            && n.compare(0, prefix.size(), prefix) == 0
            && n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0) {
            if (!found.empty()) { if (ambiguous) *ambiguous = true; return found; }
            found = e.path().string();
        }
    }
    return found;
}

// The five gguf paths a generation needs. Resolve by the naming convention (--model / download_models.py)
// or set explicitly (CLI --tok/--dit/... overrides). resolve() globs <models_dir> by prefix+suffix so the
// size label (1.5B/0.5B) and exact encoder name aren't hardcoded.
struct ModelPaths {
    std::string tok;     // t5gemma-b-b-ul2-v1.0-vocab.gguf
    std::string t5;      // t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf  (shared)
    std::string cond;    // stable-audio-3-<variant>-conditioner-v1.0-F32.gguf  (empty => read from t5)
    std::string dit;     // stable-audio-3-<variant>-dit-*-<ENC>.gguf
    std::string same;    // stable-audio-3-<variant>-same-*-<ENC>.gguf

    // variant in {medium, small-music, small-sfx}; encoding in {f16, f32}. Returns false + a message
    // in `err` if a required file is missing (so the caller can surface a friendly hint).
    static bool resolve(const std::string& models_dir, const std::string& variant,
                        const std::string& encoding, ModelPaths& out, std::string& err);
};

inline bool ModelPaths::resolve(const std::string& md, const std::string& variant,
                                const std::string& encoding, ModelPaths& out, std::string& err) {
    const std::string ENC = (encoding == "f32" || encoding == "F32") ? "F32" : "F16";
    auto one = [&](const std::string& prefix, const std::string& suffix, const char* what) {
        std::string p = resolve_one(md, prefix, suffix);
        if (p.empty())
            err += (err.empty() ? "" : "; ") + ("no " + std::string(what) + " (" + prefix + "*" + suffix + ")");
        return p;
    };
    out.tok  = one("t5gemma-b-b-ul2-v1.0-vocab",             ".gguf",            "tokenizer");
    out.t5   = one("t5gemma-b-b-ul2-encoder-",               ".gguf",            "encoder");
    out.cond = one("stable-audio-3-" + variant + "-conditioner-", ".gguf",       "conditioner");
    out.dit  = one("stable-audio-3-" + variant + "-dit-",  "-" + ENC + ".gguf",  "DiT");
    out.same = one("stable-audio-3-" + variant + "-same-", "-" + ENC + ".gguf",  "SAME");
    if (!err.empty()) err += " in " + md + "/ (run: python tools/download_models.py --variant " + variant + ")";
    return err.empty();
}

// One generation request. Defaults reproduce the CLI's text2music defaults.
struct GenParams {
    std::string prompt;
    int      frames = 128;             // output length in latent frames (EVEN for SAME-S); 128 ~= 12 s
    int      steps  = 8;
    uint64_t seed   = 0;

    // audio2audio / inpaint (optional). init_audio is planar [init_n_samp, init_n_ch] @ 44.1 kHz;
    // empty => text2music. Length-derivation (T from the source, padding) happens inside generate().
    std::vector<float> init_audio;
    int   init_n_samp = 0, init_n_ch = 0;
    float init_noise_level = 0.85f;    // sigma_max for a2a (1.0 == text2music)
    float inpaint_start = -1.0f;       // inpaint region in seconds; needs init_audio + a local-cond DiT
    float inpaint_end   = -1.0f;       // also the TOTAL output duration (a short clip can extend)

    // Adapters applied (in order) for THIS request, then reset. Paths are full gguf paths
    // (the CLI/server resolve names -> paths before building GenParams).
    std::vector<std::pair<std::string, float>> loras;   // (gguf path, strength)

    // Long-audio decode tiling; 0 = monolithic (the sliding-window decoder is already linear).
    int decode_chunk_size = 0;
    int decode_overlap    = 32;

    // Residency for THIS request (see the header banner). Default true = resident (keep models loaded
    // for the next request — the right default when a server/lib is reused). Set false for frugal
    // early-free: fits long-form on small VRAM at the cost of a reload next request. The one-shot CLI
    // sets this false by default (it frees before exit anyway).
    bool keep_models = true;
};

struct GenResult {
    std::vector<float> samples;        // planar [n_samp, n_ch]
    int n_samp = 0;
    int n_ch = 2;
    int sample_rate = 44100;
};

// Holds the loaded models for the life of the process. Move-only (owns the backend + buffers).
class Pipeline {
public:
    Pipeline() = default;
    ~Pipeline();
    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    // Load all nets onto one shared backend (the GPU if available; SA3_DEVICE=cpu forces CPU).
    // Throws std::runtime_error on a missing/!@#$ file. Idempotent guard: load() once per Pipeline.
    void load(const ModelPaths& paths);
    bool loaded() const { return loaded_; }

    // Run one generation. At entry it (re)loads any net a prior frugal (keep_models=false) call freed,
    // so keep_models can vary per call. Applies params.loras over the base DiT for this call and resets
    // after, so consecutive calls may use different adapters. Serialize calls per Pipeline.
    GenResult generate(const GenParams& params);

    // Cheap accessors a server/health-check wants.
    const SameConfig& same_config() const { return sc_; }
    const DitConfig&  dit_config()  const { return dc_; }

private:
    bool loaded_ = false;
    ModelPaths paths_;                 // kept so frugal mode can reload T5/DiT freed mid-request
    ggml_backend_t backend_ = nullptr;
    Tokenizer tok_;
    GgufModel TE_, DIT_, AE_;
    std::unique_ptr<GgufModel> CD_;    // per-variant conditioner sidecar, or null => use TE_
    T5GemmaConfig tc_{};
    DitConfig     dc_{};
    SameConfig    sc_{};
};

} // namespace sa3
