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

} // namespace sa3
