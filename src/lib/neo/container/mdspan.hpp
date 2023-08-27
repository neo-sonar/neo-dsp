#pragma once

#if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wextra-semi"
#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wextra-semi"
#endif

#include <mdspan/mdarray.hpp>
#include <mdspan/mdspan.hpp>

#if defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

#include <concepts>
#include <type_traits>

namespace stdex {
using Kokkos::default_accessor;
using Kokkos::dextents;
using Kokkos::extents;
using Kokkos::full_extent;
using Kokkos::layout_left;
using Kokkos::layout_right;
using Kokkos::layout_stride;
using Kokkos::mdspan;
using Kokkos::Experimental::mdarray;
using Kokkos::Experimental::submdspan;
}  // namespace stdex

namespace neo {

template<typename T>
inline constexpr auto const is_mdspan = false;

template<typename T, typename Extents, typename Layout, typename Accessor>
inline constexpr auto const is_mdspan<stdex::mdspan<T, Extents, Layout, Accessor>> = true;

template<typename T>
concept in_vector = is_mdspan<T> && T::rank() == 1;

template<typename T>
concept out_vector =                                                                          //
    is_mdspan<T>                                                                              //
    && T::rank() == 1                                                                         //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

template<typename T>
concept inout_vector =                                                                        //
    is_mdspan<T>                                                                              //
    && T::rank() == 1                                                                         //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

template<typename T>
concept in_matrix = is_mdspan<T> && T::rank() == 2;

template<typename T>
concept out_matrix =                                                                          //
    is_mdspan<T>                                                                              //
    && T::rank() == 2                                                                         //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

template<typename T>
concept inout_matrix =                                                                        //
    is_mdspan<T>                                                                              //
    && T::rank() == 2                                                                         //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

template<typename T>
concept in_object = is_mdspan<T> && (T::rank() == 1 || T::rank() == 2);

template<typename T>
concept out_object =                                                                          //
    is_mdspan<T>                                                                              //
    && (T::rank() == 1 || T::rank() == 2)                                                     //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

template<typename T>
concept inout_object =                                                                        //
    is_mdspan<T>                                                                              //
    && (T::rank() == 1 || T::rank() == 2)                                                     //
    && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type>  //
    && T::is_always_unique();                                                                 //

namespace detail {

[[nodiscard]] constexpr auto all_extents_match(in_object auto ref, in_object auto... others) noexcept -> bool
{
    return ((ref.extents() == others.extents()) and ...);
}

}  // namespace detail

}  // namespace neo
