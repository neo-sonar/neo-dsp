#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/transform/conjugate_view.hpp>
#include <neo/fft/transform/direction.hpp>
#include <neo/fft/transform/reorder.hpp>
#include <neo/math/complex.hpp>
#include <neo/math/ilog2.hpp>
#include <neo/math/power.hpp>

#include <algorithm>
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

struct radix2_kernel_v1
{
    radix2_kernel_v1() = default;

    auto operator()(inout_vector auto x, auto const& twiddles) const -> void
    {
        auto const size  = x.size();
        auto const order = static_cast<std::int32_t>(ilog2(size));

        auto stage_length = 1;
        auto stride       = 2;

        for (auto stage = 0; stage < order; ++stage) {
            auto const tw_stride = power<2>(order - stage - 1);

            for (auto k = 0; std::cmp_less(k, size); k += stride) {
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
};

struct radix2_kernel_v2
{
    radix2_kernel_v2() = default;

    auto operator()(inout_vector auto x, auto const& twiddles) const -> void
    {
        auto const size = x.size();

        auto stage_size = 2U;
        while (stage_size <= size) {
            auto const halfStage = stage_size / 2;
            auto const k_step    = size / stage_size;

            for (auto i{0U}; i < size; i += stage_size) {
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
};

struct radix2_kernel_v3
{
    radix2_kernel_v3() = default;

    auto operator()(inout_vector auto x, auto const& twiddles) const -> void
    {
        auto const size  = x.size();
        auto const order = ilog2(size);

        for (auto stage{0ULL}; stage < order; ++stage) {

            auto const stage_length = power<2ULL>(stage);
            auto const stride       = power<2ULL>(stage + 1);
            auto const tw_stride    = power<2ULL>(order - stage - 1ULL);

            for (auto k{0ULL}; k < size; k += stride) {
                for (auto pair{0ULL}; pair < stage_length; ++pair) {
                    auto const tw = twiddles[pair * tw_stride];

                    auto const i1 = k + pair;
                    auto const i2 = k + pair + stage_length;

                    auto const temp = x[i1] + tw * x[i2];
                    x[i2]           = x[i1] - tw * x[i2];
                    x[i1]           = temp;
                }
            }
        }
    }
};

struct radix2_kernel_v4
{
    radix2_kernel_v4() = default;

    auto operator()(inout_vector auto x, auto const& twiddles) const -> void
    {
        auto const size  = x.size();
        auto const order = ilog2(size);

        {
            // stage 0
            static constexpr auto const stage_length = 1ULL;  // power<2ULL>(0)
            static constexpr auto const stride       = 2ULL;  // power<2ULL>(0 + 1)

            auto const tw_stride = power<2ULL>(order - 1ULL);

            for (auto k{0ULL}; k < size; k += stride) {
                for (auto pair{0ULL}; pair < stage_length; ++pair) {
                    auto const tw = twiddles[pair * tw_stride];

                    auto const i1 = k + pair;
                    auto const i2 = k + pair + stage_length;

                    auto const temp = x[i1] + tw * x[i2];
                    x[i2]           = x[i1] - tw * x[i2];
                    x[i1]           = temp;
                }
            }
        }

        for (auto stage{1ULL}; stage < order; ++stage) {

            auto const stage_length = power<2ULL>(stage);
            auto const stride       = power<2ULL>(stage + 1);
            auto const tw_stride    = power<2ULL>(order - stage - 1ULL);

            for (auto k{0ULL}; k < size; k += stride) {
                for (auto pair{0ULL}; pair < stage_length; pair += 2ULL) {
                    {
                        auto const p0 = pair;
                        auto const tw = twiddles[p0 * tw_stride];

                        auto const i1 = k + p0;
                        auto const i2 = k + p0 + stage_length;

                        auto const temp = x[i1] + tw * x[i2];
                        x[i2]           = x[i1] - tw * x[i2];
                        x[i1]           = temp;
                    }

                    {
                        auto const p1 = pair + 1ULL;
                        auto const tw = twiddles[p1 * tw_stride];

                        auto const i1 = k + p1;
                        auto const i2 = k + p1 + stage_length;

                        auto const temp = x[i1] + tw * x[i2];
                        x[i2]           = x[i1] - tw * x[i2];
                        x[i1]           = temp;
                    }
                }
            }
        }
    }
};

inline constexpr auto fft_radix2 = [](auto const& kernel, inout_vector auto x, auto const& twiddles) -> void {
    bit_reverse_permutation(x);
    kernel(x, twiddles);
};

template<typename Complex, typename Kernel = radix2_kernel_v3>
struct fft_radix2_plan
{
    using complex_type = Complex;
    using size_type    = std::size_t;

    explicit fft_radix2_plan(size_type order) : _order{order} {}

    [[nodiscard]] auto size() const noexcept -> size_type { return _size; }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    template<inout_vector InOutVec>
        requires(std::same_as<typename InOutVec::value_type, Complex>)
    auto operator()(InOutVec vec, direction dir) -> void
    {
        NEO_FFT_PRECONDITION(std::cmp_equal(vec.size(), _size));

        auto const twiddles = Kokkos::mdspan{_twiddles.data(), Kokkos::extents{_twiddles.size()}};
        bit_reverse_permutation(vec, _index_table);

        if (dir == direction::forward) {
            Kernel{}(vec, twiddles);
        } else {
            Kernel{}(vec, conjugate_view{twiddles});
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
    size_type _order;
    size_type _size{1ULL << _order};
    std::vector<size_type> _index_table{make_bit_reversed_index_table(_size)};
    std::vector<Complex> _twiddles{make_radix2_twiddles<Complex>(_size)};
};

}  // namespace neo::fft
