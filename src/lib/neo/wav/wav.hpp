#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace neo {

enum struct wav_format_tag : std::uint16_t
{
    pcm        = 0x1,
    adpcm      = 0x2,
    ieee_float = 0x3,
    alaw       = 0x6,
    mulaw      = 0x7,
    dvi_adpcm  = 0x11,
    extensible = 0xFFFE,
};

struct wav_header
{
    std::array<std::byte, 4> magic_header;  // Should contain the letters "RIFF"
    std::uint32_t file_size;                // File size, minus chunk_id and chunk_size.
    std::array<std::byte, 4> format;        // Should contain the letters "WAVE"
    std::array<std::byte, 4> subchunk_id;   // Should contain the letters "fmt "
    std::uint32_t subchunk_size;            // 16 for PCM.
    wav_format_tag format_tag;              //
    std::uint16_t channels;                 // mono = 1, stereo = 2, etc.
    std::uint32_t sample_rate;              // self-explanatory
    std::uint32_t byte_rate;                // sample_rate * channels * bits per sample / 8
    std::uint16_t block_align;              // Bytes for one sample including all channels.
    std::uint16_t bits_per_sample;          // self-explanatory. BITS, not BYTES.
    std::array<std::byte, 4> data_id;       // Should contain the letters "data"
    std::uint32_t data_size;                // samples * channels * bits per sample / 8
};

struct wav_file
{
    wav_header header;
    std::span<std::byte const> data;
};

[[nodiscard]] auto parse_wav_header(std::span<std::byte const> data) -> wav_file;

struct wav_decoder
{
    explicit wav_decoder(wav_file const& file);

    [[nodiscard]] auto operator()(std::span<float> out) -> std::span<float>;

private:
    using decoder_func_t = float (*)(std::span<std::byte const>);

    wav_file _file;
    std::uint32_t _frameIndex{0};
    decoder_func_t _decoder{nullptr};
};

}  // namespace neo
