#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>

#include <ipp.h>

#include <memory>

namespace neo::fft {

namespace detail {
struct ipp_free
{
    auto operator()(auto* ptr) const noexcept -> void { ::ippsFree(ptr); }
};

using ipp_buffer = std::unique_ptr<Ipp8u[], ipp_free>;

}  // namespace detail

template<complex Complex>
    requires(std::same_as<typename Complex::value_type, float> or std::same_as<typename Complex::value_type, double>)
struct intel_ipp_fft_plan
{
    using value_type = Complex;
    using real_type  = typename Complex::value_type;
    using size_type  = std::size_t;

    explicit intel_ipp_fft_plan(size_type order, direction /*default_direction*/ = direction::forward) : _order{order}
    {
        static constexpr auto flag = IPP_FFT_NODIV_BY_ANY;
        static constexpr auto hint = ippAlgHintNone;

        int spec_size = 0;
        int init_size = 0;
        int work_size = 0;
        if (traits::get_size(static_cast<int>(order), flag, hint, &spec_size, &init_size, &work_size) != ippStsNoErr) {
            assert(false);
        }

        auto* handle        = static_cast<typename traits::handle_type*>(nullptr);
        auto spec_buf       = detail::ipp_buffer{::ippsMalloc_8u(spec_size)};
        auto const init_buf = detail::ipp_buffer{::ippsMalloc_8u(init_size)};

        if (traits::init(&handle, static_cast<int>(order), flag, hint, spec_buf.get(), init_buf.get()) != ippStsNoErr) {
            assert(false);
        }

        _handle   = handle;
        _spec_buf = std::move(spec_buf);
        _work_buf = detail::ipp_buffer{::ippsMalloc_8u(work_size)};
    }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return size_type(1) << order(); }

    template<inout_vector InOutVec>
        requires std::same_as<typename InOutVec::value_type, Complex>
    auto operator()(InOutVec x, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(x.extent(0), size()));

        auto transform = dir == direction::forward ? traits::forward : traits::backward;

        if constexpr (has_layout_left_or_right<InOutVec> and has_default_accessor<InOutVec>) {
            auto buffer = reinterpret_cast<typename traits::complex_type*>(x.data_handle());
            transform(buffer, _handle, _work_buf.get());
        } else {
            copy(x, _buffer.to_mdspan());
            auto buffer = reinterpret_cast<typename traits::complex_type*>(_buffer.data());
            transform(buffer, _handle, _work_buf.get());
            copy(_buffer.to_mdspan(), x);
        }
    }

private:
    struct traits_f32
    {
        using complex_type             = ::Ipp32fc;
        using handle_type              = ::IppsFFTSpec_C_32fc;
        static constexpr auto get_size = ::ippsFFTGetSize_C_32fc;
        static constexpr auto init     = ::ippsFFTInit_C_32fc;
        static constexpr auto forward  = ::ippsFFTFwd_CToC_32fc_I;
        static constexpr auto backward = ::ippsFFTInv_CToC_32fc_I;
    };

    struct traits_f64
    {
        using complex_type             = ::Ipp64fc;
        using handle_type              = ::IppsFFTSpec_C_64fc;
        static constexpr auto get_size = ::ippsFFTGetSize_C_64fc;
        static constexpr auto init     = ::ippsFFTInit_C_64fc;
        static constexpr auto forward  = ::ippsFFTFwd_CToC_64fc_I;
        static constexpr auto backward = ::ippsFFTInv_CToC_64fc_I;
    };

    using traits = std::conditional_t<std::same_as<real_type, float>, traits_f32, traits_f64>;

    size_type _order;
    stdex::mdarray<Complex, stdex::dextents<size_t, 1>> _buffer{size()};
    typename traits::handle_type* _handle;
    detail::ipp_buffer _spec_buf;
    detail::ipp_buffer _work_buf;
};

template<std::floating_point Float, complex Complex = std::complex<Float>>
    requires((std::same_as<Float, float> or std::same_as<Float, double>) and std::same_as<typename Complex::value_type, Float>)
struct intel_ipp_rfft_plan
{
    using value_type   = Complex;
    using complex_type = Complex;
    using real_type    = Float;
    using size_type    = std::size_t;

    explicit intel_ipp_rfft_plan(size_type order) : _order{order}
    {
        static constexpr auto flag = IPP_FFT_NODIV_BY_ANY;
        static constexpr auto hint = ippAlgHintNone;

        int spec_size = 0;
        int init_size = 0;
        int work_size = 0;
        if (traits::get_size(static_cast<int>(order), flag, hint, &spec_size, &init_size, &work_size) != ippStsNoErr) {
            assert(false);
        }

        auto* handle        = static_cast<typename traits::handle_type*>(nullptr);
        auto spec_buf       = detail::ipp_buffer{::ippsMalloc_8u(spec_size)};
        auto const init_buf = detail::ipp_buffer{::ippsMalloc_8u(init_size)};

        if (traits::init(&handle, static_cast<int>(order), flag, hint, spec_buf.get(), init_buf.get()) != ippStsNoErr) {
            assert(false);
        }

        _handle   = handle;
        _spec_buf = std::move(spec_buf);
        _work_buf = detail::ipp_buffer{::ippsMalloc_8u(work_size)};
    }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return size_type(1) << order(); }

    template<in_vector InVec, out_vector OutVec>
        requires(std::same_as<typename InVec::value_type, Float> and std::same_as<typename OutVec::value_type, complex_type>)
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        assert(std::cmp_equal(in.extent(0), size()));

        auto buf = _buffer.to_mdspan();
        copy(in, stdex::submdspan(buf, std::tuple{0, size()}));

        traits::forward(_buffer.data(), _handle, _work_buf.get());

        auto const coeffs = size() / 2 + 1;
        for (auto i{0U}; i < coeffs; ++i) {
            out[i] = complex_type{buf[i * 2], buf[i * 2 + 1]};
        }
    }

    template<in_vector InVec, out_vector OutVec>
        requires(std::same_as<typename InVec::value_type, complex_type> and std::same_as<typename OutVec::value_type, Float>)
    auto operator()(InVec in, OutVec out) noexcept -> void
    {
        assert(std::cmp_equal(in.extent(0), size() / 2 + 1));
        assert(std::cmp_equal(out.extent(0), size()));

        auto buf = _buffer.to_mdspan();
        for (auto i{0U}; i < in.size(); ++i) {
            buf[i * 2]     = real(in[i]);
            buf[i * 2 + 1] = imag(in[i]);
        }

        traits::backward(_buffer.data(), _handle, _work_buf.get());
        copy(stdex::submdspan(buf, std::tuple{0, size()}), out);
    }

private:
    struct traits_f32
    {
        using float_type               = ::Ipp32f;
        using handle_type              = ::IppsFFTSpec_R_32f;
        static constexpr auto get_size = ::ippsFFTGetSize_R_32f;
        static constexpr auto init     = ::ippsFFTInit_R_32f;
        static constexpr auto forward  = ::ippsFFTFwd_RToCCS_32f_I;
        static constexpr auto backward = ::ippsFFTInv_CCSToR_32f_I;
    };

    struct traits_f64
    {
        using float_type               = ::Ipp64f;
        using handle_type              = ::IppsFFTSpec_R_64f;
        static constexpr auto get_size = ::ippsFFTGetSize_R_64f;
        static constexpr auto init     = ::ippsFFTInit_R_64f;
        static constexpr auto forward  = ::ippsFFTFwd_RToCCS_64f_I;
        static constexpr auto backward = ::ippsFFTInv_CCSToR_64f_I;
    };

    using traits = std::conditional_t<std::same_as<real_type, float>, traits_f32, traits_f64>;

    size_type _order;
    stdex::mdarray<typename traits::float_type, stdex::dextents<size_t, 1>> _buffer{size() * 2};
    typename traits::handle_type* _handle;
    detail::ipp_buffer _spec_buf;
    detail::ipp_buffer _work_buf;
};

}  // namespace neo::fft
