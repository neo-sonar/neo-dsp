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
#include <span>
#include <type_traits>

namespace KokkosEx = Kokkos::Experimental;

namespace neo::fft {

template<typename T>
struct is_mdspan : std::false_type
{};

template<typename T, typename Extents, typename Layout, typename Accessor>
struct is_mdspan<Kokkos::mdspan<T, Extents, Layout, Accessor>> : std::true_type
{};

template<typename T>
concept in_vector = is_mdspan<T>::value && T::rank() == 1;

template<typename T>
concept out_vector
    = is_mdspan<T>::value && T::rank() == 1
   && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type> && T::is_always_unique();

template<typename T>
concept inout_vector
    = is_mdspan<T>::value && T::rank() == 1
   && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type> && T::is_always_unique();

template<typename T>
concept in_matrix = is_mdspan<T>::value && T::rank() == 2;

template<typename T>
concept out_matrix
    = is_mdspan<T>::value && T::rank() == 2
   && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type> && T::is_always_unique();

template<typename T>
concept inout_matrix
    = is_mdspan<T>::value && T::rank() == 2
   && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type> && T::is_always_unique();

template<typename T>
concept in_object = is_mdspan<T>::value && (T::rank() == 1 || T::rank() == 2);

template<typename T>
concept out_object
    = is_mdspan<T>::value && (T::rank() == 1 || T::rank() == 2)
   && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type> && T::is_always_unique();

template<typename T>
concept inout_object
    = is_mdspan<T>::value && (T::rank() == 1 || T::rank() == 2)
   && std::same_as<std::remove_const_t<typename T::element_type>, typename T::element_type> && T::is_always_unique();

}  // namespace neo::fft
