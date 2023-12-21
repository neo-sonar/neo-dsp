#pragma once

#include <neo/complex.hpp>
#include <neo/fft/fft.hpp>

namespace neo::fft {

template<typename Plan, in_vector InVec, out_vector OutVec>
    requires(std::floating_point<typename InVec::value_type> and complex<typename OutVec::value_type>)
constexpr auto rfft(Plan& plan, InVec input, OutVec output)
{
    return plan(input, output);
}

template<typename Plan, in_vector InVec, out_vector OutVec>
    requires(complex<typename InVec::value_type> and std::floating_point<typename OutVec::value_type>)
constexpr auto irfft(Plan& plan, InVec input, OutVec output)
{
    return plan(input, output);
}

template<typename Float, typename Complex = std::complex<Float>>
struct fallback_rfft_plan
{
    using real_type    = Float;
    using complex_type = Complex;
    using size_type    = std::size_t;

    explicit fallback_rfft_plan(size_type order) : _order{order} {}

    [[nodiscard]] auto size() const noexcept -> size_type { return _size; }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    template<in_vector InVec, out_vector OutVec>
        requires(std::same_as<typename InVec::value_type, Float> and std::same_as<typename OutVec::value_type, Complex>)
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        auto const buf    = _buffer.to_mdspan();
        auto const coeffs = _size / 2 + 1;

        copy(in, buf);
        _fft(buf, direction::forward);
        copy(stdex::submdspan(buf, std::tuple{0ULL, coeffs}), stdex::submdspan(out, std::tuple{0ULL, coeffs}));
    }

    template<in_vector InVec, out_vector OutVec>
        requires(std::same_as<typename InVec::value_type, Complex> and std::same_as<typename OutVec::value_type, Float>)
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        auto const buf    = _buffer.to_mdspan();
        auto const coeffs = _size / 2 + 1;

        copy(in, stdex::submdspan(buf, std::tuple{0, in.extent(0)}));

        // Fill upper half with conjugate
        for (auto i{coeffs}; i < _size; ++i) {
            using std::conj;
            buf[i] = conj(buf[_size - i]);
        }

        _fft(buf, direction::backward);
        for (auto i{0UL}; i < _size; ++i) {
            out[i] = buf[i].real();
        }
    }

private:
    size_type _order;
    size_type _size{1ULL << _order};
    fft_plan<Complex> _fft{_order};
    stdex::mdarray<Complex, stdex::dextents<size_type, 1>> _buffer{_size};
};

template<in_vector InVec, out_vector OutVecA, out_vector OutVecB>
auto rfft_deinterleave(InVec dft, OutVecA a, OutVecB b) -> void
{
    using Complex = typename InVec::value_type;
    using Float   = typename Complex::value_type;

    auto const n = dft.size();
    auto const i = Complex{Float(0), Float(-1)};

    a[0] = dft[0].real();
    b[0] = dft[0].imag();

    for (auto k{1U}; k < n / 2 + 1; ++k) {
        using std::conj;
        a[k] = (dft[k] + conj(dft[n - k])) * Float(0.5);
        b[k] = (i * (dft[k] - conj(dft[n - k]))) * Float(0.5);
    }
}

#if defined(NEO_HAS_INTEL_IPP)
template<std::floating_point Float, typename Complex = std::complex<Float>>
using rfft_plan = intel_ipp_rfft_plan<Float, Complex>;
#else
template<std::floating_point Float, typename Complex = std::complex<Float>>
using rfft_plan = fallback_rfft_plan<Float, Complex>;
#endif

}  // namespace neo::fft
