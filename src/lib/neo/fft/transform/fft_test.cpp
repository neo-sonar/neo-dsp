#include "fft.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <random>
#include <vector>

namespace {
template<typename Real, typename Kernel>
struct fft_radix2_plan_builder
{
    using plan_type = neo::fft::fft_radix2_plan<std::complex<Real>, Kernel>;
};
}  // namespace

namespace fft = neo::fft;

using namespace neo::fft;

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft/transform:",
    "",
    (fft_radix2_plan_builder),

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
    using Plan    = typename TestType::plan_type;
    using Complex = typename Plan::complex_type;
    using Float   = typename Complex::value_type;

    auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);

    auto plan = Plan{order};
    REQUIRE(plan.order() == order);

    auto const original = neo::generate_noise_signal<std::complex<Float>>(plan.size(), Catch::getSeed());
    auto const noise    = Kokkos::mdspan{original.data(), Kokkos::extents{original.size()}};

    auto copy = original;
    auto io   = Kokkos::mdspan{copy.data(), Kokkos::extents{copy.size()}};

    plan(io, fft::direction::forward);
    plan(io, fft::direction::backward);

    neo::scale(Float(1) / static_cast<Float>(plan.size()), io);
    REQUIRE(neo::allclose(noise, io));
}
