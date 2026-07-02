/* sa3-libcancel - C ABI cooperative-cancel smoke test for embedded hosts.
 *
 * This uses the same request knobs the IPlug2 demo relies on for long text2music
 * renders: frugal/early-free mode plus SAME-L chunked decode. By default the
 * cancel callback fires immediately, so the test validates callback plumbing and
 * cleanup without performing a full generation.
 *
 * Usage:
 *   sa3-libcancel [models_dir] [cancel_after_polls]
 */
#include "libsa3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct cancel_state {
    int polls;
    int cancel_after_polls;
    int progress_calls;
} cancel_state;

static int should_cancel(void* user) {
    cancel_state* state = (cancel_state*)user;
    if (!state) return 1;
    state->polls += 1;
    return state->polls > state->cancel_after_polls;
}

static void on_progress(void* user, const char* stage, int step, int total, float fraction) {
    cancel_state* state = (cancel_state*)user;
    if (state) state->progress_calls += 1;
    printf("  [%3.0f%%] %s %d/%d\n", fraction * 100.0f, stage ? stage : "render", step, total);
    fflush(stdout);
}

static int contains_cancelled(const char* text) {
    return text && strstr(text, "cancelled") != NULL;
}

int main(int argc, char** argv) {
    const char* models_dir = argc > 1 ? argv[1] : NULL;
    cancel_state state;
    memset(&state, 0, sizeof state);
    state.cancel_after_polls = argc > 2 ? atoi(argv[2]) : 0;
    if (state.cancel_after_polls < 0) state.cancel_after_polls = 0;

    char err[512] = {0};
    printf("%s\n", sa3_version());
    printf("cancel_after_polls=%d\n", state.cancel_after_polls);

    sa3_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.models_dir = models_dir;
    cfg.variant = "medium";
    cfg.encoding = "f16";

    sa3_context* ctx = sa3_init(&cfg, err, (int)sizeof err);
    if (!ctx) {
        fprintf(stderr, "sa3_init failed: %s\n", err);
        return 1;
    }

    sa3_request_ex req;
    memset(&req, 0, sizeof req);
    req.request.prompt = "warm analog cancellation smoke test";
    req.request.frames = 128;
    req.request.steps = 8;
    req.request.seed = -1;
    req.request.cfg_scale = 1.0f;
    req.request.duration_padding_sec = 6.0f;
    req.request.keep_models = 0;
    req.request.user = &state;
    req.request.on_progress = on_progress;
    req.decode_chunk_size = 128;
    req.decode_overlap = 32;
    req.should_cancel = should_cancel;
    req.cancel_user = &state;

    sa3_audio audio;
    memset(&audio, 0, sizeof audio);
    const int rc = sa3_generate_ex(ctx, &req, &audio, err, (int)sizeof err);
    if (rc == 0) {
        fprintf(stderr, "expected cancellation, but sa3_generate_ex succeeded\n");
        sa3_free_audio(&audio);
        sa3_free(ctx);
        return 2;
    }

    if (!contains_cancelled(err)) {
        fprintf(stderr, "expected cancellation error, got rc=%d err=%s\n", rc, err);
        sa3_free_audio(&audio);
        sa3_free(ctx);
        return 3;
    }

    if (audio.samples != NULL || audio.n_samp != 0 || audio.n_ch != 0) {
        fprintf(stderr, "cancelled request left output audio populated\n");
        sa3_free_audio(&audio);
        sa3_free(ctx);
        return 4;
    }

    printf("cancelled successfully after %d poll(s), progress callbacks=%d\n",
           state.polls, state.progress_calls);
    sa3_free(ctx);
    return 0;
}
