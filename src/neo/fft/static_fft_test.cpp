#include "static_fft.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/fft/fft.hpp>
#include <neo/testing/testing.hpp>

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

namespace {

template<typename Complex, typename Order>
struct static_plan
{
    inline static constexpr auto order = Order::value;
    inline static constexpr auto size  = size_t(1U) << Order::value;

    using value_type = Complex;
    using plan_type  = neo::fft::static_fft_plan<Complex, order>;
};

template<typename Order>
using static_plan_cplx64 = static_plan<std::complex<float>, Order>;

template<typename Order>
using static_plan_cplx128 = static_plan<std::complex<double>, Order>;

template<std::size_t Order>
using order = std::integral_constant<std::size_t, Order>;

}  // namespace

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft: static_fft_plan",
    "",
    (static_plan_cplx64, static_plan_cplx128),
    (order<1>,
     order<2>,
     order<3>,
     order<4>,
     order<5>,
     order<6>,
     order<7>,
     order<8>,
     order<9>,
     order<10>,
     order<11>,
     order<12>,
     order<13>,
     order<14>,
     order<15>,
     order<16>)
)
{
    using Tester  = TestType;
    using Complex = typename Tester::value_type;
    using Float   = typename Complex::value_type;
    using Plan    = typename Tester::plan_type;

    STATIC_REQUIRE(std::same_as<typename Plan::value_type, Complex>);
    STATIC_REQUIRE(std::same_as<typename Plan::size_type, std::size_t>);

    auto plan = Plan{};
    REQUIRE(plan.order() == Tester::order);
    REQUIRE(plan.size() == Tester::size);

    auto const noise = neo::generate_noise_signal<Complex>(plan.size(), Catch::getSeed());

    SECTION("inplace")
    {
        auto copy = noise;
        auto io   = copy.to_mdspan();

        neo::fft::fft(plan, io);
        neo::fft::ifft(plan, io);

        neo::scale(Float(1) / static_cast<Float>(plan.size()), io);
        REQUIRE(neo::allclose(noise.to_mdspan(), io));
    }

    SECTION("copy")
    {
        auto tmpBuf = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{noise.extents()};
        auto outBuf = stdex::mdarray<Complex, stdex::dextents<size_t, 1>>{noise.extents()};

        auto tmp = tmpBuf.to_mdspan();
        auto out = outBuf.to_mdspan();

        neo::fft::fft(plan, noise.to_mdspan(), tmp);
        neo::fft::ifft(plan, tmp, out);

        neo::scale(Float(1) / static_cast<Float>(plan.size()), out);
        REQUIRE(neo::allclose(noise.to_mdspan(), out));
    }
}
