#pragma once

#include "neo/fft/transform/conjugate_view.hpp"
#include "neo/fft/transform/direction.hpp"
#include "neo/fft/transform/math.hpp"
#include "neo/fft/transform/reorder.hpp"

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <vector>

namespace neo::fft {

template<typename Complex>
auto twiddle_table_radix2(std::size_t size, bool inverse = false) -> std::vector<Complex>
{
    using Float = typename Complex::value_type;

    auto const pi   = static_cast<Float>(std::numbers::pi);
    auto const sign = inverse ? Float(1) : Float(-1);
    auto table      = std::vector<Complex>(size / 2U, Float(0));
    for (std::size_t i = 0; i < size / 2U; ++i) {
        auto const angle = sign * Float(2) * pi * Float(i) / Float(size);
        table[i]         = std::polar(Float(1), angle);
    }
    return table;
}

template<typename Complex, std::size_t Size>
auto twiddle_table_radix2(bool inverse = false) -> std::array<Complex, Size / 2>
{
    using Float = typename Complex::value_type;

    auto const pi   = static_cast<Float>(std::numbers::pi);
    auto const sign = inverse ? Float(1) : Float(-1);
    auto table      = std::array<Complex, Size / 2>{};
    for (std::size_t i = 0; i < Size / 2; ++i) {
        auto const angle = sign * Float(2) * pi * Float(i) / Float(Size);
        table[i]         = std::polar(Float(1), angle);
    }
    return table;
}

template<typename Complex, std::size_t Extent, typename Range>
auto c2c_radix2(std::span<Complex, Extent> x, Range const& twiddles) -> void
{
    // bit-reverse ordering
    auto const len   = x.size();
    auto const order = ilog2(len);
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
                auto const tw = twiddles[pair * tw_stride];

                auto const i1 = k + pair;
                auto const i2 = k + pair + stage_length;

                auto const temp = x[i1] + tw * x[i2];
                x[i2]           = x[i1] - tw * x[i2];
                x[i1]           = temp;
            }
        }

        stage_length *= 2;
        stride *= 2;
    }
}

template<typename Complex, std::size_t Extent, typename Range>
auto c2c_radix2_alt(std::span<Complex, Extent> x, Range const& twiddles) -> void
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
struct c2c_radix2_plan
{
    explicit c2c_radix2_plan(std::size_t len) : _len{len} {}

    [[nodiscard]] auto size() const noexcept -> std::size_t { return _len; }

    auto operator()(std::span<Complex> x, direction dir) -> void
    {
        assert(std::cmp_equal(x.size(), _len));

        auto run = [this](std::span<Complex> buffer, auto const& twiddles) {
            for (auto stage{0ULL}; stage < _order; ++stage) {

                auto const stage_length = power<2ULL>(stage);
                auto const stride       = power<2ULL>(stage + 1);
                auto const tw_stride    = power<2ULL>(_order - stage - 1ULL);

                for (auto k{0ULL}; k < _len; k += stride) {
                    for (auto pair{0ULL}; pair < stage_length; ++pair) {
                        auto const tw = twiddles[pair * tw_stride];

                        auto const i1 = k + pair;
                        auto const i2 = k + pair + stage_length;

                        auto const temp = buffer[i1] + tw * buffer[i2];
                        buffer[i2]      = buffer[i1] - tw * buffer[i2];
                        buffer[i1]      = temp;
                    }
                }
            }
        };

        bit_reverse_permutation(x, _indexTable);
        if (dir == direction::forward) {
            run(x, _twiddleTable);
        } else {
            run(x, conjugate_view<Complex>{_twiddleTable});
        }
    }

    auto operator()(std::span<Complex const> in, std::span<Complex> out, direction dir) -> void
    {
        std::copy(in.begin(), in.end(), out.begin());
        (*this)(out, dir);
    }

private:
    std::size_t _len;
    std::size_t _order{ilog2(_len)};
    std::vector<std::size_t> _indexTable{make_bit_reversed_index_table(size_t(_len))};
    std::vector<Complex> _twiddleTable{twiddle_table_radix2<Complex>(_len)};
};

template<typename Float>
struct rfft_radix2_plan
{
    explicit rfft_radix2_plan(std::size_t order)
        : _order{order}
        , _size{1ULL << order}
        , _cfft{_size}
        , _tmp(_size, std::complex<Float>(0))
    {}

    [[nodiscard]] auto size() const noexcept -> std::size_t { return _size; }

    auto operator()(std::span<Float const> in, std::span<std::complex<Float>> out) -> void
    {
        std::copy(in.begin(), in.end(), _tmp.begin());
        _cfft(_tmp, direction::forward);

        auto const coeffs = std::span{_tmp}.subspan(0, _size / 2 + 1);
        std::copy(coeffs.begin(), coeffs.end(), out.begin());
    }

    auto operator()(std::span<std::complex<Float> const> in, std::span<Float> out) -> void
    {
        std::copy(in.begin(), in.end(), _tmp.begin());

        // Fill upper half with conjugate
        for (auto i = _size / 2 + 1; i < _size; ++i) { _tmp[i] = std::conj(_tmp[_size - i]); }

        _cfft(_tmp, direction::backward);
        std::transform(_tmp.begin(), _tmp.end(), out.begin(), [](auto cx) { return cx.real(); });
    }

private:
    std::size_t _order;
    std::size_t _size;
    c2c_radix2_plan<std::complex<Float>> _cfft;
    std::vector<std::complex<Float>> _tmp;
};

}  // namespace neo::fft
