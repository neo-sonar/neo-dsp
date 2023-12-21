#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>

#include <Accelerate/Accelerate.h>

#include <cassert>
#include <utility>

namespace neo::fft {

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
struct apple_vdsp_fft_plan
{
    using value_type         = Complex;
    using real_type          = typename Complex::value_type;
    using size_type          = std::size_t;
    using native_handle_type = std::conditional_t<std::same_as<real_type, float>, FFTSetup, FFTSetupD>;

    explicit apple_vdsp_fft_plan(size_type order, direction default_direction = direction::forward);
    ~apple_vdsp_fft_plan();

    [[nodiscard]] auto order() const noexcept -> size_type;
    [[nodiscard]] auto size() const noexcept -> size_type;

    template<inout_vector InOutVec>
        requires std::same_as<typename InOutVec::value_type, Complex>
    auto operator()(InOutVec x, direction dir) noexcept -> void;

private:
    size_type _order;
    size_type _size{1ULL << _order};
    native_handle_type _plan;
    stdex::mdarray<real_type, stdex::dextents<size_t, 2>> _input{2, _size};
    stdex::mdarray<real_type, stdex::dextents<size_t, 2>> _output{2, _size};
};

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
apple_vdsp_fft_plan<Complex>::apple_vdsp_fft_plan(size_type order, direction /*default_direction*/)
    : _order{order}
    , _plan{[order] {
    if constexpr (std::same_as<real_type, float>) {
        return vDSP_create_fftsetup(order, 2);
    } else {
        return vDSP_create_fftsetupD(order, 2);
    }
}()}
{
    assert(_plan != nullptr);
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
apple_vdsp_fft_plan<Complex>::~apple_vdsp_fft_plan()
{
    if (_plan != nullptr) {
        if constexpr (std::same_as<real_type, float>) {
            vDSP_destroy_fftsetup(_plan);
        } else {
            vDSP_destroy_fftsetupD(_plan);
        }
    }
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
auto apple_vdsp_fft_plan<Complex>::size() const noexcept -> size_type
{
    return _size;
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
auto apple_vdsp_fft_plan<Complex>::order() const noexcept -> size_type
{
    return _order;
}

template<typename Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
template<inout_vector InOutVec>
    requires std::same_as<typename InOutVec::value_type, Complex>
auto apple_vdsp_fft_plan<Complex>::operator()(InOutVec x, direction dir) noexcept -> void
{
    assert(std::cmp_equal(x.extent(0), _size));

    using split_complex = std::conditional_t<std::same_as<real_type, float>, DSPSplitComplex, DSPDoubleSplitComplex>;

    auto const in   = split_complex{.realp = std::addressof(_input(0, 0)), .imagp = std::addressof(_input(1, 0))};
    auto const out  = split_complex{.realp = std::addressof(_output(0, 0)), .imagp = std::addressof(_output(1, 0))};
    auto const sign = dir == direction::forward ? kFFTDirection_Forward : kFFTDirection_Inverse;

    for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
        in.realp[i] = x[i].real();
        in.imagp[i] = x[i].imag();
    }

    if constexpr (std::same_as<real_type, float>) {
        vDSP_fft_zop(_plan, &in, 1, &out, 1, _order, sign);
    } else {
        vDSP_fft_zopD(_plan, &in, 1, &out, 1, _order, sign);
    }

    for (auto i{0}; std::cmp_less(i, x.extent(0)); ++i) {
        x[i] = Complex{out.realp[i], out.imagp[i]};
    }
}

}  // namespace neo::fft
