// sa3-server: a small HTTP server over the shared sa3 Pipeline.
//   POST /generate {prompt, frames|seconds, steps, seed, loras[], keep_models, ...} -> audio/wav bytes
//   POST /unload   -> free the model (full VRAM release; orchestrator owns the unload policy)
//   GET  /health   -> {status, model, loaded, device}
//
// Default is FRUGAL (keep_models=false): the model frees after each generation and reloads on the next
// request. That keeps it VST/DAW-memory-safe and makes per-request LoRA strength changes correct for free.
// A request may set "keep_models": true to keep it resident between calls (lower latency, more VRAM).
#include "sa3_pipeline.h"
#include "wav.h"

#include "httplib.h"
#include "yyjson.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

namespace {

std::mutex g_mtx;                         // serialize: one generation (one GPU graph) at a time
std::unique_ptr<sa3::Pipeline> g_pipe;    // loaded lazily on first /generate; freed on /unload
std::string g_variant   = "medium";
std::string g_encoding  = "f16";
std::string g_models_dir;
std::string g_adapters_dir;

std::string json_err(const std::string& msg) { return "{\"error\":\"" + msg + "\"}"; }

// (re)load the pipeline under the caller's lock. Returns false + message on failure.
bool ensure_loaded(std::string& err) {
    if (g_pipe && g_pipe->loaded()) return true;
    sa3::ModelPaths mp;
    if (!sa3::ModelPaths::resolve(g_models_dir, g_variant, g_encoding, mp, err)) return false;
    try {
        g_pipe = std::make_unique<sa3::Pipeline>();
        g_pipe->load(mp);
    } catch (const std::exception& e) { g_pipe.reset(); err = e.what(); return false; }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 8086;
    if (const char* e = getenv("SA3_MODELS_DIR"))   g_models_dir   = e;
    if (const char* e = getenv("SA3_ADAPTERS_DIR")) g_adapters_dir = e;
    if (g_models_dir.empty()) g_models_dir = "models";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* d){ return i + 1 < argc ? argv[++i] : d; };
        if      (a == "--host")         host = next("127.0.0.1");
        else if (a == "--port")         port = atoi(next("8086"));
        else if (a == "--model")        g_variant = next("medium");
        else if (a == "--encoding")     g_encoding = next("f16");
        else if (a == "--models-dir")   g_models_dir = next("models");
        else if (a == "--adapters-dir") g_adapters_dir = next("");
    }
    const std::string adir = g_adapters_dir.empty() ? g_models_dir : g_adapters_dir;

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        const bool loaded = g_pipe && g_pipe->loaded();
        std::string body = "{\"status\":\"ok\",\"model\":\"" + g_variant + "\",\"encoding\":\"" +
                           g_encoding + "\",\"loaded\":" + (loaded ? "true" : "false") + "}";
        res.set_content(body, "application/json");
    });

    svr.Post("/unload", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_pipe.reset();   // Pipeline dtor frees nets + backend (full VRAM release)
        res.set_content("{\"status\":\"unloaded\"}", "application/json");
    });

    svr.Post("/generate", [&adir](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(g_mtx);

        yyjson_doc* doc = yyjson_read(req.body.c_str(), req.body.size(), 0);
        if (!doc) { res.status = 400; res.set_content(json_err("invalid json"), "application/json"); return; }
        yyjson_val* root = yyjson_doc_get_root(doc);
        auto S = [&](const char* k, const char* d) { yyjson_val* v = yyjson_obj_get(root, k); return std::string(v && yyjson_is_str(v) ? yyjson_get_str(v) : d); };
        auto I = [&](const char* k, int d) { yyjson_val* v = yyjson_obj_get(root, k); return v && yyjson_is_int(v) ? (int)yyjson_get_int(v) : d; };
        auto D = [&](const char* k, double d) { yyjson_val* v = yyjson_obj_get(root, k); return v && yyjson_is_num(v) ? yyjson_get_num(v) : d; };
        auto B = [&](const char* k, bool d) { yyjson_val* v = yyjson_obj_get(root, k); return v && yyjson_is_bool(v) ? yyjson_get_bool(v) : d; };

        sa3::GenParams params;
        params.prompt           = S("prompt", "");
        params.frames           = I("frames", 128);
        if (yyjson_obj_get(root, "seconds"))                 // ~10.77 latent frames/sec (44100/4096)
            params.frames = (int)(D("seconds", 12.0) * 44100.0 / 4096.0 + 0.5);
        params.steps            = I("steps", 8);
        const uint64_t seed_resolved = sa3::pick_seed(I("seed", 0));   // seed -1 => random
        params.seed             = seed_resolved;
        params.keep_models      = B("keep_models", false);   // FRUGAL default
        params.init_noise_level = (float)D("init_noise_level", 0.85);
        params.inpaint_start    = (float)D("inpaint_start", -1.0);   // inpaint/continuation region (sec)
        params.inpaint_end      = (float)D("inpaint_end", -1.0);
        params.duration_padding_sec = (float)D("duration_padding_sec", 6.0);   // text2music schedule headroom (0 = let it end)
        // classifier-free guidance (all inert unless cfg_scale != 1.0)
        params.negative_prompt   = S("negative_prompt", "");
        params.cfg_scale         = (float)D("cfg_scale", 1.0);
        params.cfg_rescale       = (float)D("cfg_rescale", 0.0);
        params.apg_scale         = (float)D("apg_scale", 1.0);
        params.cfg_norm_threshold= (float)D("cfg_norm_threshold", 0.0);
        params.cfg_interval_min  = (float)D("cfg_interval_min", 0.0);
        params.cfg_interval_max  = (float)D("cfg_interval_max", 1.0);
        // schedule warp: "dist_shift" type + its defaults, optionally overridden by a 4-number "dist_shift_params".
        params.dist_shift       = S("dist_shift", "LogSNR");
        sa3::dist_shift_defaults(params.dist_shift, params.ds_p1, params.ds_p2, params.ds_p3, params.ds_p4);
        if (yyjson_val* dsp = yyjson_obj_get(root, "dist_shift_params"); dsp && yyjson_is_arr(dsp)) {
            float* slots[4] = { &params.ds_p1, &params.ds_p2, &params.ds_p3, &params.ds_p4 };
            yyjson_val* v; yyjson_arr_iter di; yyjson_arr_iter_init(dsp, &di);
            for (int k = 0; k < 4 && (v = yyjson_arr_iter_next(&di)); k++)
                if (yyjson_is_num(v)) *slots[k] = (float)yyjson_get_num(v);
        }
        std::string init_path   = S("init_path", "");                // local WAV for audio2audio / inpaint

        std::string perr;
        yyjson_val* loras = yyjson_obj_get(root, "loras");
        if (loras && yyjson_is_arr(loras)) {
            yyjson_val* it; yyjson_arr_iter ai; yyjson_arr_iter_init(loras, &ai);
            while ((it = yyjson_arr_iter_next(&ai))) {
                yyjson_val* nv = yyjson_obj_get(it, "name");
                if (!nv) nv = yyjson_obj_get(it, "path");
                std::string name = nv && yyjson_is_str(nv) ? yyjson_get_str(nv) : "";
                yyjson_val* sv = yyjson_obj_get(it, "strength");
                float strength = sv && yyjson_is_num(sv) ? (float)yyjson_get_num(sv) : 1.0f;
                if (name.empty()) continue;
                std::string p = std::filesystem::exists(name) ? name
                                : sa3::resolve_one(adir, "lora-" + name + "-", ".gguf");
                if (p.empty()) { perr = "unknown lora '" + name + "'"; break; }
                params.loras.push_back({p, strength});
            }
        }
        yyjson_doc_free(doc);

        if (!perr.empty())            { res.status = 400; res.set_content(json_err(perr), "application/json"); return; }
        if (params.prompt.empty())    { res.status = 400; res.set_content(json_err("prompt required"), "application/json"); return; }

        if (!init_path.empty()) {     // audio2audio / inpaint source (local path — the server is localhost)
            if (!std::filesystem::exists(init_path)) {
                res.status = 400; res.set_content(json_err("init_path not found: " + init_path), "application/json"); return;
            }
            int ns = 0, nc = 0, sr = 0;
            params.init_audio = sa3::read_wav_planar(init_path, ns, nc, sr);
            params.init_n_samp = ns; params.init_n_ch = nc;
        }

        std::string err;
        if (!ensure_loaded(err)) { res.status = 500; res.set_content(json_err(err), "application/json"); return; }
        try {
            sa3::GenResult r = g_pipe->generate(params);
            res.set_header("X-Seed", std::to_string(seed_resolved));   // the seed used (resolved if -1)
            res.set_content(sa3::wav_planar_bytes(r.samples.data(), r.n_samp, r.n_ch, r.sample_rate), "audio/wav");
        } catch (const std::exception& e) {
            res.status = 500; res.set_content(json_err(e.what()), "application/json");
        }
    });

    fprintf(stderr, "[sa3-server] http://%s:%d  model=%s/%s  models=%s  adapters=%s  (frugal default)\n",
            host.c_str(), port, g_variant.c_str(), g_encoding.c_str(), g_models_dir.c_str(), adir.c_str());
    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "[sa3-server] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
