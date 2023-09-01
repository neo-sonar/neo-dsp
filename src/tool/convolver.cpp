#include "wav.hpp"

#include <neo/algorithm.hpp>
#include <neo/container.hpp>
#include <neo/convolution.hpp>
#include <neo/math.hpp>

#include <cstdio>
#include <cstdlib>

[[nodiscard]] static auto
convolve(neo::audio_buffer<float> const& signal, neo::audio_buffer<float> const& impulse, int block_size = 512)
    -> neo::audio_buffer<float>
{
    auto impulse_copy = impulse;
    neo::normalize_impulse(impulse_copy.to_mdspan());
    auto const partitions = neo::fft::uniform_partition(impulse_copy.to_mdspan(), static_cast<size_t>(block_size));

    auto output       = neo::audio_buffer<float>{signal.extent(0), signal.extent(1)};
    auto block_buffer = stdex::mdarray<float, stdex::dextents<size_t, 1>>(size_t(block_size));

    for (auto channel{0}; std::cmp_less(channel, signal.extent(0)); ++channel) {
        auto convolver  = neo::fft::upols_convolver<std::complex<float>>{};
        auto const full = stdex::full_extent;
        convolver.filter(stdex::submdspan(partitions.to_mdspan(), channel, full, full));

        for (auto i{0}; std::cmp_less(i, output.extent(1)); i += block_size) {
            neo::fill(block_buffer.to_mdspan(), 0.0F);

            auto const num_samples   = std::min(static_cast<int>(output.extent(1)) - i, block_size);
            auto const input_block   = stdex::submdspan(signal.to_mdspan(), channel, std::tuple{i, i + num_samples});
            auto const process_block = stdex::submdspan(block_buffer.to_mdspan(), std::tuple{0, num_samples});
            neo::copy(input_block, process_block);

            convolver(block_buffer.to_mdspan());

            auto const output_block = stdex::submdspan(output.to_mdspan(), channel, std::tuple{i, i + num_samples});
            neo::copy(process_block, output_block);
        }
    }

    return output;
}

auto main(int argc, char** argv) -> int
{
    if (argc != 4) {
        std::printf("Usage: ./neo_dsp_convolver path/to/signal.wav path/to/filter.wav path/to/output.wav\n");
        return EXIT_FAILURE;
    }

    auto const [signal, signal_sr]   = neo::load_wav_file<float>(argv[1]);
    auto const [filter, filter_sr]   = neo::load_wav_file<float>(argv[2]);
    auto const output_length         = signal.extent(1);
    auto const output_length_seconds = static_cast<double>(output_length) / signal_sr;

    if (signal.extent(0) != filter.extent(0)) {
        std::printf("Channel mismatch: signal = %d filter = %d\n", int(signal.extent(0)), int(filter.extent(0)));
        return EXIT_FAILURE;
    }
    if (not neo::float_equality::exact(signal_sr, filter_sr)) {
        std::printf("Sample-rate mismatch: signal = %d filter = %d\n", int(signal_sr), int(filter_sr));
        return EXIT_FAILURE;
    }

    std::printf(
        "Filter: %d channels %d frames (%.2f sec) at %d kHz\n",
        static_cast<int>(filter.extent(0)),
        static_cast<int>(filter.extent(1)),
        static_cast<double>(filter.extent(1)) / filter_sr,
        static_cast<int>(filter_sr)
    );

    std::printf(
        "Signal: %d channels %d frames (%.2f sec) at %d kHz\n",
        static_cast<int>(signal.extent(0)),
        static_cast<int>(signal.extent(1)),
        static_cast<double>(signal.extent(1)) / signal_sr,
        static_cast<int>(signal_sr)
    );

    auto const start   = std::chrono::system_clock::now();
    auto output        = convolve(signal, filter);
    auto const stop    = std::chrono::system_clock::now();
    auto const runtime = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);

    neo::normalize_peak(output.to_mdspan());
    neo::write_wav_file(output, signal_sr, argv[3]);
    std::printf("Runtime: %.2f sec / %.1f x real-time\n", runtime.count(), output_length_seconds / runtime.count());

    return EXIT_SUCCESS;
}
