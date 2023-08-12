#pragma once

#include <neo/fft/transform/radix2.hpp>

#include <complex>
#include <span>

namespace neo::fft {

template<typename Float>
struct rfft_radix2_plan
{
    using real_type    = Float;
    using complex_type = std::complex<Float>;
    using size_type    = std::size_t;

    explicit rfft_radix2_plan(size_type order) : _order{order} {}

    [[nodiscard]] auto size() const noexcept -> size_type { return _size; }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

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
    size_type _order;
    size_type _size{1ULL << _order};
    fft_radix2_plan<std::complex<Float>> _cfft{_order};
    KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_type, 1>> _buffer{_size};
};

template<in_vector InVec, out_vector OutVecA, out_vector OutVecB>
auto extract_two_real_dfts(InVec dft, OutVecA a, OutVecB b) -> void
{
    using std::conj;
    using Complex = typename InVec::value_type;
    using Float   = typename Complex::value_type;

    auto const n = dft.size();
    auto const i = Complex{Float(0), Float(-1)};

    a[0] = dft[0].real();
    b[0] = dft[0].imag();

    for (auto k{1U}; k < n / 2 + 1; ++k) {
        a[k] = (dft[k] + conj(dft[n - k])) * Float(0.5);
        b[k] = (i * (dft[k] - conj(dft[n - k]))) * Float(0.5);
    }
}

}  // namespace neo::fft
