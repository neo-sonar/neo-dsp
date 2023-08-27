#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/transform/direction.hpp>

#include <Accelerate/Accelerate.h>

#include <cassert>
#include <utility>

namespace neo::fft {

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float>)
struct fft_apple_vdsp_plan
{
    using complex_type = Complex;
    using real_type    = typename Complex::value_type;
    using size_type    = std::size_t;

    explicit fft_apple_vdsp_plan(size_type order, direction default_direction = direction::forward);
    ~fft_apple_vdsp_plan();

    [[nodiscard]] auto order() const noexcept -> size_type;
    [[nodiscard]] auto size() const noexcept -> size_type;

    template<inout_vector InOutVec>
        requires std::same_as<typename InOutVec::value_type, Complex>
    auto operator()(InOutVec x, direction dir) noexcept -> void;

private:
    size_type _order;
    size_type _size{1ULL << _order};
    FFTSetup _plan{vDSP_create_fftsetup(_order, 2)};
    stdex::mdarray<real_type, stdex::dextents<size_t, 2>> _input{2, _size};
    stdex::mdarray<real_type, stdex::dextents<size_t, 2>> _output{2, _size};
};

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float>)
fft_apple_vdsp_plan<Complex>::fft_apple_vdsp_plan(size_type order, direction /*default_direction*/) : _order{order}
{
    assert(_plan != nullptr);
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float>)
fft_apple_vdsp_plan<Complex>::~fft_apple_vdsp_plan()
{
    if (_plan != nullptr) {
        vDSP_destroy_fftsetup(_plan);
    }
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float>)
auto fft_apple_vdsp_plan<Complex>::size() const noexcept -> size_type
{
    return _size;
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float>)
auto fft_apple_vdsp_plan<Complex>::order() const noexcept -> size_type
{
    return _order;
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float>)
template<inout_vector InOutVec>
    requires std::same_as<typename InOutVec::value_type, Complex>
auto fft_apple_vdsp_plan<Complex>::operator()(InOutVec x, direction dir) noexcept -> void
{
    assert(std::cmp_equal(x.extent(0), _size));

    auto const in   = DSPSplitComplex{.realp = std::addressof(_input(0, 0)), .imagp = std::addressof(_input(1, 0))};
    auto const out  = DSPSplitComplex{.realp = std::addressof(_output(0, 0)), .imagp = std::addressof(_output(1, 0))};
    auto const sign = dir == direction::forward ? kFFTDirection_Forward : kFFTDirection_Inverse;

    for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
        in.realp[i] = x[i].real();
        in.imagp[i] = x[i].imag();
    }

    vDSP_fft_zop(_plan, &in, 1, &out, 1, _order, sign);

    for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
        x[i] = Complex{out.realp[i], out.imagp[i]};
    }
}

}  // namespace neo::fft
