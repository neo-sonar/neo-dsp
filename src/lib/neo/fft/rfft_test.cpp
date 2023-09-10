#include "rfft.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/complex.hpp>
#include <neo/fft/experimental/rfft.hpp>
#include <neo/fft/fft.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <random>

namespace {

template<typename Float, typename Kernel, typename Complex>
struct tester
{
    static_assert(std::same_as<Float, typename Complex::value_type>);
    using complex_plan_type = neo::fft::fft_plan<Complex, Kernel>;
    using plan_type         = neo::fft::rfft_plan<Float, complex_plan_type>;
};

template<typename Float, typename Kernel>
using std_complex = tester<Float, Kernel, std::complex<Float>>;

template<typename Float, typename Kernel>
using neo_complex = tester<Float, Kernel, neo::scalar_complex<Float>>;

}  // namespace

using namespace neo::fft;

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft: rfft_plan",
    "",
    (std_complex, neo_complex),

    ((float, kernel::c2c_dit2_v1),
     (float, kernel::c2c_dit2_v2),
     (float, kernel::c2c_dit2_v3),

     (double, kernel::c2c_dit2_v1),
     (double, kernel::c2c_dit2_v2),
     (double, kernel::c2c_dit2_v3))
)
{
    using Plan    = typename TestType::plan_type;
    using Float   = typename Plan::real_type;
    using Complex = typename Plan::complex_type;

    auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);

    auto rfft = Plan{order};
    REQUIRE(rfft.order() == order);
    REQUIRE(rfft.size() == std::size_t(1) << order);

    auto signal         = neo::generate_noise_signal<Float>(rfft.size(), Catch::getSeed());
    auto spectrum       = std::vector<Complex>(rfft.size() / 2UL + 1UL, Float(0));
    auto const original = signal;

    auto const real    = signal.to_mdspan();
    auto const complex = stdex::mdspan{spectrum.data(), stdex::extents{spectrum.size()}};
    rfft(real, complex);
    rfft(complex, real);

    neo::scale(Float(1) / static_cast<Float>(rfft.size()), real);
    REQUIRE(neo::allclose(stdex::mdspan{original.data(), stdex::extents{original.size()}}, real));
}

TEMPLATE_PRODUCT_TEST_CASE("neo/fft: extract_two_real_dfts", "", (std::complex, neo::scalar_complex), (float, double))
{
    using Complex = TestType;
    using Float   = typename Complex::value_type;

    auto order           = GENERATE(as<std::size_t>{}, 4, 5, 6, 7, 8);
    auto const size      = std::size_t(1) << order;
    auto const numCoeffs = size / 2 + 1;
    CAPTURE(order);
    CAPTURE(size);
    CAPTURE(numCoeffs);

    auto fft = neo::fft::fft_plan<Complex>{order};
    REQUIRE(fft.size() == size);
    REQUIRE(fft.order() == order);

    auto rfft = neo::fft::rfft_plan<Float, decltype(fft)>{order};
    REQUIRE(rfft.size() == size);
    REQUIRE(rfft.order() == order);

    auto rng    = std::mt19937{Catch::getSeed()};
    auto dist   = std::uniform_real_distribution<Float>{Float(-1), Float(1)};
    auto random = [&dist, &rng] { return dist(rng); };

    auto a = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{size};
    auto b = stdex::mdarray<Float, stdex::dextents<size_t, 1>>{size};
    std::generate(a.data(), a.data() + a.size(), random);
    std::generate(b.data(), b.data() + b.size(), random);

    auto a_rev = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{numCoeffs};
    auto b_rev = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{numCoeffs};
    rfft(a.to_mdspan(), a_rev.to_mdspan());
    rfft(b.to_mdspan(), b_rev.to_mdspan());

    auto inout = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{size};
    auto merge = [](Float ra, Float rb) { return Complex{ra, rb}; };
    std::transform(a.data(), a.data() + a.size(), b.data(), inout.data(), merge);

    fft(inout.to_mdspan(), neo::fft::direction::forward);

    auto ca = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{numCoeffs};
    auto cb = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{numCoeffs};
    neo::fft::extract_two_real_dfts(inout.to_mdspan(), ca.to_mdspan(), cb.to_mdspan());

    REQUIRE(neo::allclose(a_rev.to_mdspan(), ca.to_mdspan()));
    REQUIRE(neo::allclose(b_rev.to_mdspan(), cb.to_mdspan()));
}

TEMPLATE_TEST_CASE("neo/fft: experimental::rfft_plan", "", float, double)
{
    using Float = TestType;

    auto const order = GENERATE(as<std::size_t>{}, 1, 2, 3, 4, 5, 6, 7, 8);
    auto const size  = std::size_t(1) << order;
    CAPTURE(order);
    CAPTURE(size);

    auto plan = neo::fft::experimental::rfft_plan<Float>{order};
    REQUIRE(plan.order() == order);
    REQUIRE(plan.size() == size);

    SECTION("identity")
    {
        auto buffer_storage = stdex::mdarray<Float, stdex::dextents<size_t, 1>>(size);

        auto buffer = buffer_storage.to_mdspan();
        buffer[0]   = Float(1);

        plan(buffer, neo::fft::direction::forward);
        REQUIRE(buffer[0] == Catch::Approx(1.0));
        REQUIRE(buffer[1] == Catch::Approx(1.0));
        for (auto i{2U}; i < buffer.extent(0); i += 2) {
            CAPTURE(i);
            REQUIRE(buffer[i] == Catch::Approx(1.0));
            REQUIRE(buffer[i + 1] == Catch::Approx(0.0));
        }

        plan(buffer, neo::fft::direction::backward);
        REQUIRE(buffer[0] == Catch::Approx(1.0 * double(size / 2UL)));
    }

    SECTION("random")
    {
        auto const signal = neo::generate_noise_signal<Float>(size, Catch::getSeed());
        auto copy         = signal;

        plan(copy.to_mdspan(), neo::fft::direction::forward);
        plan(copy.to_mdspan(), neo::fft::direction::backward);

        for (auto i{0U}; i < signal.extent(0); ++i) {
            CAPTURE(i);
            REQUIRE(copy(i) == Catch::Approx(signal(i) * double(size / 2UL)).scale(double(order)));
        }
    }
}