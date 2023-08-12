#include "radix2.hpp"

#include <neo/fft/algorithm/allclose.hpp>
#include <neo/fft/algorithm/scale.hpp>
#include <neo/fft/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <random>
#include <vector>

namespace fft = neo::fft;

TEMPLATE_TEST_CASE("neo/fft/transform: make_radix2_twiddles", "", float, double)
{
    using Float   = TestType;
    using Complex = std::complex<Float>;

    auto const array  = fft::make_radix2_twiddles<Complex, 64>();
    auto const vector = fft::make_radix2_twiddles<Complex>(64);

    for (auto i{0UL}; i < array.size(); ++i) {
        CAPTURE(i);
        REQUIRE(array[i].real() == Catch::Approx(vector[i].real()));
        REQUIRE(array[i].imag() == Catch::Approx(vector[i].imag()));
    }
}

namespace {
template<typename Real, typename Kernel>
struct fft_radix2_plan_builder
{
    using plan_type = neo::fft::fft_radix2_plan<std::complex<Real>, Kernel>;
};

}  // namespace

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

    auto const original = fft::generate_noise_signal<std::complex<Float>>(plan.size(), Catch::getSeed());
    auto const noise    = Kokkos::mdspan{original.data(), Kokkos::extents{original.size()}};

    auto copy = original;
    auto io   = Kokkos::mdspan{copy.data(), Kokkos::extents{copy.size()}};

    plan(io, fft::direction::forward);
    plan(io, fft::direction::backward);

    fft::scale(Float(1) / static_cast<Float>(plan.size()), io);
    REQUIRE(fft::allclose(noise, io));
}
