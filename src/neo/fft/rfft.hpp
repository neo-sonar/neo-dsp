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

template<typename Float, typename ComplexPlan = fft_plan<std::complex<Float>>>
struct rfft_plan
{
    using complex_plan_type = ComplexPlan;
    using complex_type      = typename ComplexPlan::value_type;
    using size_type         = typename ComplexPlan::size_type;
    using real_type         = Float;

    explicit rfft_plan(size_type order) : _order{order} {}

    [[nodiscard]] auto size() const noexcept -> size_type { return _size; }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    template<in_vector InVec, out_vector OutVec>
        requires(std::same_as<typename InVec::value_type, Float> and std::same_as<typename OutVec::value_type, complex_type>)
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        auto const buf    = _buffer.to_mdspan();
        auto const coeffs = _size / 2 + 1;

        copy(in, buf);
        _fft(buf, direction::forward);
        copy(stdex::submdspan(buf, std::tuple{0ULL, coeffs}), stdex::submdspan(out, std::tuple{0ULL, coeffs}));
    }

    template<in_vector InVec, out_vector OutVec>
        requires(std::same_as<typename InVec::value_type, complex_type> and std::same_as<typename OutVec::value_type, Float>)
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
    ComplexPlan _fft{_order};
    stdex::mdarray<complex_type, stdex::dextents<size_type, 1>> _buffer{_size};
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

}  // namespace neo::fft
