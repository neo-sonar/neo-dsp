#pragma once

#include <neo/fft/config.hpp>

#include <neo/fft/algorithm/copy.hpp>
#include <neo/fft/container/mdspan.hpp>
#include <neo/fft/math/ilog2.hpp>
#include <neo/fft/math/power.hpp>
#include <neo/fft/transform/conjugate_view.hpp>
#include <neo/fft/transform/direction.hpp>
#include <neo/fft/transform/reorder.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

namespace neo::fft {

template<inout_vector OutVec>
auto fill_radix2_twiddles(OutVec table, direction dir = direction::forward) -> void
{
    using Complex = typename OutVec::value_type;
    using Float   = typename Complex::value_type;

    auto const tableSize = table.size();
    auto const fftSize   = tableSize * 2ULL;
    auto const sign      = dir == direction::forward ? Float(-1) : Float(1);
    auto const twoPi     = static_cast<Float>(std::numbers::pi * 2.0);

    for (std::size_t i = 0; i < tableSize; ++i) {
        auto const angle = sign * twoPi * Float(i) / Float(fftSize);
        table[i]         = std::polar(Float(1), angle);
    }
}

template<typename Complex>
auto make_radix2_twiddles(std::size_t size, direction dir = direction::forward) -> std::vector<Complex>
{
    auto table = std::vector<Complex>(size / 2U);
    fill_radix2_twiddles(Kokkos::mdspan{table.data(), Kokkos::extents{table.size()}}, dir);
    return table;
}

template<typename Complex, std::size_t Size>
auto make_radix2_twiddles(direction dir = direction::forward) -> std::array<Complex, Size / 2>
{
    auto table = std::array<Complex, Size / 2>{};
    fill_radix2_twiddles(Kokkos::mdspan{table.data(), Kokkos::extents{table.size()}}, dir);
    return table;
}

template<inout_vector InOutVec, typename TwiddleTable>
auto c2c_radix2(InOutVec x, TwiddleTable const& twiddles) -> void
{
    // bit-reverse ordering
    auto const len   = x.size();
    auto const order = static_cast<std::int32_t>(ilog2(len));
    bit_reverse_permutation(x);

    // butterfly computation
    auto stage_length = 1;
    auto stride       = 2;
    for (auto stage = 0; stage < order; ++stage) {

        // auto const stage_length = power<2>(stage);
        // auto const stride    = power<2>(stage + 1);
        auto const tw_stride = power<2>(order - stage - 1);

        for (auto k = 0; std::cmp_less(k, len); k += stride) {
            for (auto pair = 0; pair < stage_length; ++pair) {
                auto const tw = twiddles[static_cast<std::size_t>(pair * tw_stride)];

                auto const i1 = static_cast<std::size_t>(k + pair);
                auto const i2 = static_cast<std::size_t>(k + pair + stage_length);

                auto const temp = x[i1] + tw * x[i2];
                x[i2]           = x[i1] - tw * x[i2];
                x[i1]           = temp;
            }
        }

        stage_length *= 2;
        stride *= 2;
    }
}

template<inout_vector InOutVec, typename TwiddleTable>
auto c2c_radix2_alt(InOutVec x, TwiddleTable const& twiddles) -> void
{
    auto const len = x.size();

    // Rearrange the input in bit-reversed order
    bit_reverse_permutation(x);

    auto stage_size = 2U;
    while (stage_size <= len) {
        auto const halfStage = stage_size / 2;
        auto const k_step    = len / stage_size;

        for (auto i{0U}; i < len; i += stage_size) {
            for (auto k{i}; k < i + halfStage; ++k) {
                auto const index = k - i;
                auto const tw    = twiddles[index * k_step];

                auto const idx1 = k;
                auto const idx2 = k + halfStage;

                auto const even = x[idx1];
                auto const odd  = x[idx2];

                auto const tmp = odd * tw;
                x[idx1]        = even + tmp;
                x[idx2]        = even - tmp;
            }
        }

        stage_size *= 2;
    }
}

template<typename Complex>
struct fft_radix2_plan
{
    using complex_type = Complex;

    explicit fft_radix2_plan(std::size_t size) : _size{size} {}

    [[nodiscard]] auto size() const noexcept -> std::size_t { return _size; }

    [[nodiscard]] auto order() const noexcept -> std::size_t { return _order; }

    template<inout_vector InOutVec>
        requires(std::same_as<typename InOutVec::value_type, Complex>)
    auto operator()(InOutVec vec, direction dir) -> void
    {
        NEO_FFT_PRECONDITION(std::cmp_equal(vec.size(), _size));

        bit_reverse_permutation(vec, _index_table);

        if (dir == direction::forward) {
            run(vec, _twiddles);
        } else {
            run(vec, conjugate_view<Complex>{_twiddles});
        }
    }

    template<in_vector InVec, out_vector OutVec>
        requires(std::same_as<typename InVec::value_type, Complex> and std::same_as<typename OutVec::value_type, Complex>)
    auto operator()(InVec in, OutVec out, direction dir) -> void
    {
        NEO_FFT_PRECONDITION(std::cmp_equal(in.size(), _size));
        NEO_FFT_PRECONDITION(std::cmp_equal(out.size(), _size));

        copy(in, out);
        (*this)(out, dir);
    }

private:
    auto run(inout_vector auto vec, auto const& twiddles) const
    {
        for (auto stage{0ULL}; stage < _order; ++stage) {

            auto const stage_length = power<2ULL>(stage);
            auto const stride       = power<2ULL>(stage + 1);
            auto const tw_stride    = power<2ULL>(_order - stage - 1ULL);

            for (auto k{0ULL}; k < _size; k += stride) {
                for (auto pair{0ULL}; pair < stage_length; ++pair) {
                    auto const tw = twiddles[pair * tw_stride];

                    auto const i1 = k + pair;
                    auto const i2 = k + pair + stage_length;

                    auto const temp = vec[i1] + tw * vec[i2];
                    vec[i2]         = vec[i1] - tw * vec[i2];
                    vec[i1]         = temp;
                }
            }
        }
    }

    std::size_t _size;
    std::size_t _order{ilog2(_size)};
    std::vector<std::size_t> _index_table{make_bit_reversed_index_table(_size)};
    std::vector<Complex> _twiddles{make_radix2_twiddles<Complex>(_size)};
};

}  // namespace neo::fft
