#pragma once

#include <neo/fft/transform/radix2.hpp>

#include <complex>
#include <span>

namespace neo::fft {

template<typename Float>
struct rfft_plan
{
    explicit rfft_plan(std::size_t order) : _order{order}, _size{1ULL << order}, _cfft{_size}, _buffer{_size} {}

    [[nodiscard]] auto size() const noexcept -> std::size_t { return _size; }

    template<in_vector InVec, out_vector OutVec>
        requires(std::same_as<typename InVec::value_type, Float> and std::same_as<typename OutVec::value_type, std::complex<Float>>)
    auto operator()(InVec in, OutVec out) -> void
    {
        auto const buf    = _buffer.to_mdspan();
        auto const coeffs = _size / 2 + 1;

        copy(in, buf);
        _cfft(buf, direction::forward);
        copy(KokkosEx::submdspan(buf, std::tuple{0ULL, coeffs}), KokkosEx::submdspan(out, std::tuple{0ULL, coeffs}));
    }

    template<in_vector InVec, out_vector OutVec>
        requires(std::same_as<typename InVec::value_type, std::complex<Float>> and std::same_as<typename OutVec::value_type, Float>)
    auto operator()(InVec in, OutVec out) -> void
    {
        auto const buf    = _buffer.to_mdspan();
        auto const coeffs = _size / 2 + 1;

        copy(in, KokkosEx::submdspan(buf, std::tuple{0, in.extent(0)}));

        // Fill upper half with conjugate
        for (auto i{coeffs}; i < _size; ++i) {
            buf[i] = std::conj(buf[_size - i]);
        }

        _cfft(buf, direction::backward);
        for (auto i{0UL}; i < _size; ++i) {
            out[i] = buf[i].real();
        }
    }

private:
    std::size_t _order;
    std::size_t _size;
    fft_plan<std::complex<Float>> _cfft;
    KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<std::size_t, 1>> _buffer;
};

template<typename Float>
auto extract_two_real_dfts(
    std::span<std::complex<Float> const> dft,
    std::span<std::complex<Float>> a,
    std::span<std::complex<Float>> b
) -> void
{
    auto const n = dft.size();
    auto const i = std::complex{Float(0), Float(-1)};

    a[0] = dft[0].real();
    b[0] = dft[0].imag();

    for (auto k{1U}; k < n / 2 + 1; ++k) {
        a[k] = (dft[k] + std::conj(dft[n - k])) * Float(0.5);
        b[k] = (i * (dft[k] - std::conj(dft[n - k]))) * Float(0.5);
    }
}

}  // namespace neo::fft
