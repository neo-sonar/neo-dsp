#include "neo/wav.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

inline auto peak_normalization_factor(std::span<float const> buf) -> float
{
    if (buf.empty()) { return 1.0F; }
    auto abs_less = [](auto l, auto r) { return std::abs(l) < std::abs(r); };
    return 1.0F / (*std::max_element(buf.begin(), buf.end(), abs_less));
}

inline auto rms_normalization_factor(std::span<float const> buf) -> float
{
    auto const squared_sum = [](auto sum, auto val) { return sum + (val * val); };
    auto const sum         = std::accumulate(buf.begin(), buf.end(), 0.0F, squared_sum);
    auto const mean_square = sum / static_cast<float>(buf.size());

    auto factor = 1.0F;
    if (mean_square > 0.0F) { factor = 1.0F / std::sqrt(mean_square); }
    return factor;
}

inline auto juce_normalization_factor(std::span<float const> buf) -> float
{
    auto const squared_sum = [](auto sum, auto val) { return sum + (val * val); };
    auto const sum         = std::accumulate(buf.begin(), buf.end(), 0.0F, squared_sum);
    if (sum < 1e-8F) { return 1.0F; }
    return 0.125F / std::sqrt(sum);
}

// normalized_sample = sample / max(abs(buffer))
[[maybe_unused]] static auto peak_normalization(std::span<float> buffer) -> void
{
    auto const factor   = peak_normalization_factor(buffer);
    auto const multiply = [factor](auto sample) { return sample * factor; };
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), multiply);
}

// normalized_sample = sample / sqrt(mean(buffer^2))
[[maybe_unused]] static auto rms_normalization(std::span<float> buffer) -> void
{
    if (buffer.empty()) return;
    auto const factor = rms_normalization_factor(buffer);
    auto const mul    = [factor](auto v) { return v * factor; };
    std::transform(buffer.begin(), buffer.end(), buffer.begin(), mul);
}

[[nodiscard]] static auto load_file(char const* path)
{
    auto const file = std::fopen(path, "rb");
    std::fseek(file, 0, SEEK_END);
    auto const fileSize = static_cast<size_t>(std::ftell(file));
    std::fseek(file, 0, SEEK_SET);

    auto buffer = std::vector<std::byte>(fileSize);
    if (std::fread(buffer.data(), 1, buffer.size(), file) != fileSize) {
        throw std::runtime_error{"failed to read file"};
    }
    std::fclose(file);
    return buffer;
}

auto main(int argc, char** argv) -> int
{
    if (argc != 2) {
        std::printf("Usage: ./a.out path/to/file.wav\n");
        return EXIT_FAILURE;
    }

    auto const* path = argv[1];

    auto const file_data  = load_file(path);
    auto const wav        = neo::parse_wav_header(file_data);
    auto const num_frames = wav.header.data_size / wav.header.block_align;

    std::printf("path: %s, size: %zu KB\n", path, file_data.size() / 1024);
    std::printf(
        "format: %u, sr: %u, channels: %u, bitrate: %u, frames: %u, sec: %.2f\n",
        static_cast<std::uint16_t>(wav.header.format_tag),
        wav.header.sample_rate,
        wav.header.channels,
        wav.header.bits_per_sample,
        num_frames,
        static_cast<double>(num_frames) / static_cast<double>(wav.header.sample_rate)
    );

    try {
        auto decoder = neo::wav_decoder{wav};
        auto buffer  = std::vector<float>(size_t(num_frames * wav.header.channels), 0.0F);
        auto decoded = decoder(buffer);

        std::printf("buffer: %zu - decoded: %zu\n", buffer.size(), decoded.size());

        std::printf("raw[%f", decoded[0]);
        for (auto i{500U}; i < 512U; ++i) { std::printf(", %.8f", decoded[i]); }
        std::printf("]\n");

        std::printf("peak_factor: %.2f\n", peak_normalization_factor(decoded));
        std::printf("rms_factor: %.2f\n", rms_normalization_factor(decoded));
        std::printf("juce_factor: %.2f\n", juce_normalization_factor(decoded));

        peak_normalization(decoded);
        std::printf("normalized[%f", decoded[0]);
        for (auto i{500U}; i < 512U; ++i) { std::printf(", %.8f", decoded[i]); }
        std::printf("]\n");

    } catch (std::exception const& e) {
        std::printf("error: %s\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
