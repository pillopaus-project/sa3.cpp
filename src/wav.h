// wav.h — minimal 16-bit PCM WAV writer.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace sa3 {

// Write interleaved 16-bit WAV from planar f32 channels (ch0[0..n-1], ch1[0..n-1], ...).
inline void write_wav_planar(const std::string& path, const float* data, int n_samples,
                             int n_ch, int sample_rate) {
    auto clip = [](float v){ v = v < -1.f ? -1.f : (v > 1.f ? 1.f : v); return (int16_t)(v * 32767.0f); };
    std::vector<int16_t> inter((size_t)n_samples * n_ch);
    for (int s = 0; s < n_samples; s++)
        for (int c = 0; c < n_ch; c++)
            inter[(size_t)s*n_ch + c] = clip(data[(size_t)c*n_samples + s]);

    const uint32_t data_bytes = (uint32_t)inter.size() * sizeof(int16_t);
    const uint32_t byte_rate  = sample_rate * n_ch * 2;
    FILE* f = fopen(path.c_str(), "wb");
    auto u32 = [&](uint32_t v){ fwrite(&v, 4, 1, f); };
    auto u16 = [&](uint16_t v){ fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); u32(36 + data_bytes); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); u32(16); u16(1); u16((uint16_t)n_ch);
    u32(sample_rate); u32(byte_rate); u16((uint16_t)(n_ch*2)); u16(16);
    fwrite("data", 1, 4, f); u32(data_bytes);
    fwrite(inter.data(), 1, data_bytes, f);
    fclose(f);
}

// Read a 16-bit PCM WAV into planar f32 channels (ch0[0..n-1], ch1[0..n-1], ...),
// the same layout write_wav_planar consumes and same_encode expects ([L, ch], L fastest).
inline std::vector<float> read_wav_planar(const std::string& path, int& n_samples,
                                          int& n_ch, int& sample_rate) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    char riff[12];
    if (fread(riff, 1, 12, f) != 12 || memcmp(riff, "RIFF", 4) || memcmp(riff+8, "WAVE", 4)) {
        fprintf(stderr, "%s is not a RIFF/WAVE file\n", path.c_str()); exit(1);
    }
    uint16_t fmt = 0, ch = 0, bits = 0; uint32_t rate = 0; std::vector<int16_t> pcm;
    char id[4]; uint32_t sz;
    while (fread(id, 1, 4, f) == 4 && fread(&sz, 4, 1, f) == 1) {
        if (!memcmp(id, "fmt ", 4)) {
            uint16_t hdr[8]; fread(hdr, 1, 16, f);
            fmt = hdr[0]; ch = hdr[1]; rate = *(uint32_t*)&hdr[2]; bits = hdr[7];
            if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
        } else if (!memcmp(id, "data", 4)) {
            pcm.resize(sz / sizeof(int16_t));
            fread(pcm.data(), 1, sz, f);
        } else {
            fseek(f, sz + (sz & 1), SEEK_CUR);   // skip unknown chunk (chunks are word-aligned)
        }
    }
    fclose(f);
    if (fmt != 1 || bits != 16 || ch == 0) {
        fprintf(stderr, "%s: only 16-bit PCM supported (fmt=%u bits=%u ch=%u)\n", path.c_str(), fmt, bits, ch);
        exit(1);
    }
    n_ch = ch; sample_rate = (int)rate; n_samples = (int)(pcm.size() / ch);
    std::vector<float> planar((size_t)n_samples * n_ch);
    for (int s = 0; s < n_samples; s++)
        for (int c = 0; c < n_ch; c++)
            planar[(size_t)c*n_samples + s] = pcm[(size_t)s*n_ch + c] / 32768.0f;
    return planar;
}

} // namespace sa3
