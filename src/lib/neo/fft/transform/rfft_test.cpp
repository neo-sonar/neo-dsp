#include "rfft.hpp"

#include <neo/fft/algorithm/allclose.hpp>
#include <neo/fft/math/complex.hpp>
#include <neo/fft/testing/testing.hpp>
#include <neo/fft/transform/radix2.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <random>

namespace {

template<typename Real, typename Kernel>
struct rfft_radix2_plan_builder
{
    using complex_plan_type = neo::fft::fft_radix2_plan<std::complex<Real>, Kernel>;
    using plan_type         = neo::fft::rfft_radix2_plan<Real, complex_plan_type>;
};

}  // namespace

using namespace neo::fft;

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft/transform:",
    "",
    (rfft_radix2_plan_builder),

    ((float, radix2_kernel_v1),
     (float, radix2_kernel_v2),
     (float, radix2_kernel_v3),
     (float, radix2_kernel_v4),

     (double, radix2_kernel_v1),
     (double, radix2_kernel_v2),
     (double, radix2_kernel_v3),
     (double, radix2_kernel_v4))
)
{
    using Plan  = typename TestType::plan_type;
    using Float = typename Plan::real_type;

    auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);

    auto rfft = Plan{order};
    REQUIRE(rfft.order() == order);
    REQUIRE(rfft.size() == 1UL << order);

    auto signal         = neo::fft::generate_noise_signal<Float>(rfft.size(), Catch::getSeed());
    auto spectrum       = std::vector<std::complex<Float>>(rfft.size() / 2UL + 1UL, Float(0));
    auto const original = signal;

    auto const real    = Kokkos::mdspan{signal.data(), Kokkos::extents{signal.size()}};
    auto const complex = Kokkos::mdspan{spectrum.data(), Kokkos::extents{spectrum.size()}};
    rfft(real, complex);
    rfft(complex, real);

    auto const scale = Float(1) / static_cast<Float>(rfft.size());
    std::transform(signal.begin(), signal.end(), signal.begin(), [scale](auto c) { return c * scale; });

    REQUIRE(neo::fft::allclose(Kokkos::mdspan{original.data(), Kokkos::extents{original.size()}}, real));
}

TEMPLATE_TEST_CASE("neo/fft/transform: extract_two_real_dfts", "", float, double)
{
    using Float = TestType;

    auto order           = GENERATE(as<std::size_t>{}, 4, 5, 6, 7, 8);
    auto const size      = std::size_t(1) << order;
    auto const numCoeffs = size / 2 + 1;
    CAPTURE(order);
    CAPTURE(size);
    CAPTURE(numCoeffs);

    auto fft = neo::fft::fft_radix2_plan<std::complex<Float>>{order};
    REQUIRE(fft.size() == size);
    REQUIRE(fft.order() == order);

    auto rfft = neo::fft::rfft_radix2_plan<Float>{order};
    REQUIRE(rfft.size() == size);
    REQUIRE(rfft.order() == order);

    auto rng    = std::mt19937{Catch::getSeed()};
    auto dist   = std::uniform_real_distribution<Float>{Float(-1), Float(1)};
    auto random = [&dist, &rng] { return dist(rng); };

    auto a = KokkosEx::mdarray<Float, Kokkos::dextents<size_t, 1>>{size};
    auto b = KokkosEx::mdarray<Float, Kokkos::dextents<size_t, 1>>{size};
    std::generate(a.data(), a.data() + a.size(), random);
    std::generate(b.data(), b.data() + b.size(), random);

    auto a_rev = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{numCoeffs};
    auto b_rev = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{numCoeffs};
    rfft(a.to_mdspan(), a_rev.to_mdspan());
    rfft(b.to_mdspan(), b_rev.to_mdspan());

    auto inout = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{size};
    auto merge = [](Float ra, Float rb) { return std::complex{ra, rb}; };
    std::transform(a.data(), a.data() + a.size(), b.data(), inout.data(), merge);

    fft(inout.to_mdspan(), neo::fft::direction::forward);

    auto ca = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{numCoeffs};
    auto cb = KokkosEx::mdarray<std::complex<Float>, Kokkos::dextents<size_t, 1>>{numCoeffs};
    neo::fft::extract_two_real_dfts(inout.to_mdspan(), ca.to_mdspan(), cb.to_mdspan());

    REQUIRE(neo::fft::allclose(a_rev.to_mdspan(), ca.to_mdspan()));
    REQUIRE(neo::fft::allclose(b_rev.to_mdspan(), cb.to_mdspan()));
}
