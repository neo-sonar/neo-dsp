#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>
#include <neo/fft/direction.hpp>

#include <ipp.h>

#include <memory>

namespace neo::fft {

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

        auto* handle        = static_cast<traits::handle_type*>(nullptr);
        auto spec_buf       = ipp_buffer{::ippsMalloc_8u(spec_size)};
        auto const init_buf = ipp_buffer{::ippsMalloc_8u(init_size)};

        if (traits::init(&handle, static_cast<int>(order), flag, hint, spec_buf.get(), init_buf.get()) != ippStsNoErr) {
            assert(false);
        }

        _handle   = handle;
        _spec_buf = std::move(spec_buf);
        _work_buf = ipp_buffer{::ippsMalloc_8u(work_size)};
    }

    [[nodiscard]] auto order() const noexcept -> size_type { return _order; }

    [[nodiscard]] auto size() const noexcept -> size_type { return size_type(1) << order(); }

    template<inout_vector InOutVec>
        requires std::same_as<typename InOutVec::value_type, Complex>
    auto operator()(InOutVec x, direction dir) noexcept -> void
    {
        assert(std::cmp_equal(x.extent(0), size()));

        auto transform = dir == direction::forward ? traits::forward : traits::backward;
        auto buffer    = reinterpret_cast<traits::complex_type*>(_buffer.data());

        copy(x, _buffer.to_mdspan());
        transform(buffer, _handle, _work_buf.get());
        copy(_buffer.to_mdspan(), x);
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

    struct ipp_free
    {
        auto operator()(auto* ptr) const noexcept -> void { ::ippsFree(ptr); }
    };

    using traits     = std::conditional_t<std::same_as<real_type, float>, traits_f32, traits_f64>;
    using ipp_buffer = std::unique_ptr<Ipp8u[], ipp_free>;

    size_type _order;
    stdex::mdarray<Complex, stdex::dextents<size_t, 1>> _buffer{size()};
    typename traits::handle_type* _handle;
    ipp_buffer _spec_buf;
    ipp_buffer _work_buf;
};

}  // namespace neo::fft
