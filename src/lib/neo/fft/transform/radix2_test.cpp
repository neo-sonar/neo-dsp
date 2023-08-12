#include "dft.hpp"
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

TEMPLATE_TEST_CASE("neo/fft/transform/radix2: make_radix2_twiddles", "", float, double)
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

TEMPLATE_TEST_CASE("neo/fft/transform/radix2: test_path(c2c)", "", double)
{
    using Float   = TestType;
    using Complex = std::complex<Float>;

    auto const paths = GENERATE(
        fft::test_path{"./test_data/c2c_8_input.csv", "./test_data/c2c_8_output.csv"},
        fft::test_path{"./test_data/c2c_16_input.csv", "./test_data/c2c_16_output.csv"},
        fft::test_path{"./test_data/c2c_32_input.csv", "./test_data/c2c_32_output.csv"},
        fft::test_path{"./test_data/c2c_16_input.csv", "./test_data/c2c_16_output.csv"},
        fft::test_path{"./test_data/c2c_32_input.csv", "./test_data/c2c_32_output.csv"},
        fft::test_path{"./test_data/c2c_64_input.csv", "./test_data/c2c_64_output.csv"},
        fft::test_path{"./test_data/c2c_128_input.csv", "./test_data/c2c_128_output.csv"},
        fft::test_path{"./test_data/c2c_512_input.csv", "./test_data/c2c_512_output.csv"}
    );

    auto const testCase = fft::load_test_data<Float>(paths);
    auto const expected = Kokkos::mdspan{testCase.expected.data(), Kokkos::extents{testCase.expected.size()}};

    SECTION("dft")
    {
        auto in  = testCase.input;
        auto out = std::vector<std::complex<Float>>(in.size());

        auto inVec  = Kokkos::mdspan{in.data(), Kokkos::extents{in.size()}};
        auto outVec = Kokkos::mdspan{out.data(), Kokkos::extents{out.size()}};
        fft::dft(inVec, outVec);

        REQUIRE(fft::allclose(expected, outVec));
    }

    SECTION("kernel")
    {
        auto test = [=](auto kernel) {
            auto inout = testCase.input;
            auto io    = Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}};

            auto tw = fft::make_radix2_twiddles<Complex>(inout.size());
            fft::fft_radix2(kernel, io, tw);

            return fft::allclose(expected, io);
        };

        REQUIRE(test(fft::radix2_kernel_v1{}));
        REQUIRE(test(fft::radix2_kernel_v2{}));
        REQUIRE(test(fft::radix2_kernel_v3{}));
        REQUIRE(test(fft::radix2_kernel_v4{}));
    }

    SECTION("plan")
    {
        auto test = [=](auto kernel) {
            auto inout = testCase.input;
            auto io    = Kokkos::mdspan{inout.data(), Kokkos::extents{inout.size()}};

            auto order = fft::ilog2(inout.size());
            auto plan  = fft::fft_radix2_plan<Complex, decltype(kernel)>{order};

            plan(io, fft::direction::forward);
            return fft::allclose(expected, io);
        };

        REQUIRE(test(fft::radix2_kernel_v1{}));
        REQUIRE(test(fft::radix2_kernel_v2{}));
        REQUIRE(test(fft::radix2_kernel_v3{}));
        REQUIRE(test(fft::radix2_kernel_v4{}));
    }
}

namespace {
template<typename Real, typename Kernel>
struct fft_radix2_plan_builder
{
    using plan_type = neo::fft::fft_radix2_plan<std::complex<Real>, Kernel>;
};

}  // namespace

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft/transform/radix2: fft_radix2_plan",
    "",
    (fft_radix2_plan_builder),

    ((float, neo::fft::radix2_kernel_v1),
     (float, neo::fft::radix2_kernel_v2),
     (float, neo::fft::radix2_kernel_v3),
     (float, neo::fft::radix2_kernel_v4),

     (double, neo::fft::radix2_kernel_v1),
     (double, neo::fft::radix2_kernel_v2),
     (double, neo::fft::radix2_kernel_v3),
     (double, neo::fft::radix2_kernel_v4))
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
