#pragma once

#include <neo/config.hpp>

#include <neo/algorithm/detail/linalg_unary_op.hpp>

#if defined(NEO_HAS_INTEL_MKL)
    #include <neo/algorithm/detail/blas_mkl.hpp>
#endif

namespace neo {

template<typename Scalar, inout_object InOutObj>
constexpr auto scale(Scalar alpha, InOutObj obj) -> void
{
#if defined(NEO_HAS_INTEL_MKL)
    if constexpr (InOutObj::rank() == 1) {
        constexpr auto is_blas_type = detail::is_blas_type<Scalar>;
        constexpr auto same_type    = std::same_as<Scalar, typename InOutObj::value_type>;
        if constexpr (is_blas_type and same_type and has_default_accessor<InOutObj>) {
            detail::cblas_traits<Scalar>::scale(
                static_cast<MKL_INT>(obj.extent(0)),
                alpha,
                obj.data_handle(),
                static_cast<MKL_INT>(obj.stride(0))
            );
            return;
        }
    }
#endif
    detail::linalg_unary_op(obj, [alpha](auto const& val) { return val * alpha; });
}

}  // namespace neo
