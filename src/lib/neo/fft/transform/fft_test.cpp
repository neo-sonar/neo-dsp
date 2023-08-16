#include "fft.hpp"

#include <neo/algorithm/allclose.hpp>
#include <neo/algorithm/scale.hpp>
#include <neo/simd.hpp>
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
struct plan_tester
{
    using plan_type = neo::fft::fft_radix2_plan<std::complex<Real>, Kernel>;
};
}  // namespace

namespace fft = neo::fft;

using namespace neo::fft;

TEMPLATE_PRODUCT_TEST_CASE(
    "neo/fft/transform: fft_radix2_plan",
    "",
    (plan_tester),

    ((float, radix2_kernel_v1),
     (float, radix2_kernel_v2),
     (float, radix2_kernel_v3),
     (float, radix2_kernel_v4),

     (double, radix2_kernel_v1),
     (double, radix2_kernel_v2),
     (double, radix2_kernel_v3),
     (double, radix2_kernel_v4),

     (long double, radix2_kernel_v1),
     (long double, radix2_kernel_v2),
     (long double, radix2_kernel_v3),
     (long double, radix2_kernel_v4))
)
{
    using Plan    = typename TestType::plan_type;
    using Complex = typename Plan::complex_type;
    using Float   = typename Complex::value_type;

    auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);

    auto plan = Plan{order};
    REQUIRE(plan.order() == order);
    REQUIRE(plan.size() == 1UL << order);

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

template<typename ComplexBatch, typename Kernel>
static auto test_complex_batch_roundtrip_fft()
{
    using ScalarBatch   = typename ComplexBatch::batch_type;
    using ScalarFloat   = typename ComplexBatch::real_scalar_type;
    using ScalarComplex = std::complex<ScalarFloat>;

    auto make_noise_signal = [](auto size) {
        auto noise = neo::generate_noise_signal<ScalarComplex>(size, Catch::getSeed());
        auto buf   = stdex::mdarray<ComplexBatch, stdex::dextents<size_t, 1>>{size};
        for (auto i{0UL}; i < size; ++i) {
            buf(i) = ComplexBatch{
                ScalarBatch::broadcast(noise(i).real()),
                ScalarBatch::broadcast(noise(i).imag()),
            };
        }
        return buf;
    };

    auto make_twiddles = [](auto size, neo::fft::direction dir) {
        auto tw  = neo::fft::make_radix2_twiddles<ScalarComplex>(size, dir);
        auto buf = stdex::mdarray<ComplexBatch, stdex::dextents<size_t, 1>>{tw.extents()};
        for (auto i{0UL}; i < buf.extent(0); ++i) {
            buf(i) = ComplexBatch{
                ScalarBatch::broadcast(tw(i).real()),
                ScalarBatch::broadcast(tw(i).imag()),
            };
        }
        return buf;
    };

    auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    auto const size  = 1UL << order;

    auto inout = make_noise_signal(size);

    auto const copy              = inout;
    auto const forward_twiddles  = make_twiddles(size, neo::fft::direction::forward);
    auto const backward_twiddles = make_twiddles(size, neo::fft::direction::backward);

    neo::fft::execute_radix2_kernel(Kernel{}, inout.to_mdspan(), forward_twiddles.to_mdspan());
    neo::fft::execute_radix2_kernel(Kernel{}, inout.to_mdspan(), backward_twiddles.to_mdspan());

    for (auto i{0U}; i < inout.extent(0); ++i) {
        auto const real_batch = inout(i).real();
        auto const imag_batch = inout(i).imag();

        auto reals = std::array<ScalarFloat, ScalarBatch::size>{};
        auto imags = std::array<ScalarFloat, ScalarBatch::size>{};
        real_batch.store_unaligned(reals.data());
        imag_batch.store_unaligned(imags.data());

        auto const expected_real_batch = copy(i).real();
        auto const expected_imag_batch = copy(i).imag();

        auto expected_reals = std::array<ScalarFloat, ScalarBatch::size>{};
        auto expected_imags = std::array<ScalarFloat, ScalarBatch::size>{};
        expected_real_batch.store_unaligned(expected_reals.data());
        expected_imag_batch.store_unaligned(expected_imags.data());

        auto const scalar = ScalarFloat(1) / static_cast<ScalarFloat>(size);
        neo::scale(scalar, stdex::mdspan{reals.data(), stdex::extents{reals.size()}});
        neo::scale(scalar, stdex::mdspan{imags.data(), stdex::extents{imags.size()}});

        REQUIRE(neo::allclose(
            stdex::mdspan{reals.data(), stdex::extents{reals.size()}},
            stdex::mdspan{expected_reals.data(), stdex::extents{expected_reals.size()}}
        ));

        REQUIRE(neo::allclose(
            stdex::mdspan{imags.data(), stdex::extents{imags.size()}},
            stdex::mdspan{expected_imags.data(), stdex::extents{expected_imags.size()}}
        ));
    }
}

namespace {

template<typename Real, typename Kernel>
struct simd_tester
{
    using real_type   = Real;
    using kernel_type = Kernel;
};

}  // namespace

#if defined(NEO_HAS_SIMD_SSE2)
TEMPLATE_TEST_CASE("neo/fft/transform: radix2_kernel(simd_batch)", "", neo::pcomplex64x4, neo::pcomplex128x2)
{
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v1>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v2>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v3>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v4>();
}
#endif

#if defined(NEO_HAS_SIMD_AVX)
TEMPLATE_TEST_CASE("neo/fft/transform: radix2_kernel(simd_batch)", "", neo::pcomplex64x8, neo::pcomplex128x4)
{
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v1>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v2>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v3>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v4>();
}
#endif

#if defined(NEO_HAS_SIMD_AVX512F)
TEMPLATE_TEST_CASE("neo/fft/transform: radix2_kernel(simd_batch)", "", neo::pcomplex64x16, neo::pcomplex128x8)
{
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v1>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v2>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v3>();
    test_complex_batch_roundtrip_fft<TestType, neo::fft::radix2_kernel_v4>();
}
#endif

#if defined(NEO_HAS_BASIC_FLOAT16)
TEMPLATE_TEST_CASE("neo/fft/transform: radix2_kernel(simd_batch)", "", neo::pcomplex32x8, neo::pcomplex32x16)
{
    using ComplexBatch  = TestType;
    using ScalarBatch   = typename ComplexBatch::batch_type;
    using ScalarFloat   = typename ComplexBatch::real_scalar_type;
    using ScalarComplex = std::complex<float>;

    auto test = [](auto kernel) {
        auto make_noise_signal = [](auto size) {
            auto noise = neo::generate_noise_signal<ScalarComplex>(size, Catch::getSeed());
            auto buf   = stdex::mdarray<ComplexBatch, stdex::dextents<size_t, 1>>{size};
            for (auto i{0UL}; i < size; ++i) {
                buf(i) = ComplexBatch{
                    ScalarBatch::broadcast(static_cast<ScalarFloat>(noise(i).real())),
                    ScalarBatch::broadcast(static_cast<ScalarFloat>(noise(i).imag())),
                };
            }
            return buf;
        };

        auto make_twiddles = [](auto size, neo::fft::direction dir) {
            auto tw  = neo::fft::make_radix2_twiddles<ScalarComplex>(size, dir);
            auto buf = stdex::mdarray<ComplexBatch, stdex::dextents<size_t, 1>>{tw.extents()};
            for (auto i{0UL}; i < buf.extent(0); ++i) {
                buf(i) = ComplexBatch{
                    ScalarBatch::broadcast(static_cast<ScalarFloat>(tw(i).real())),
                    ScalarBatch::broadcast(static_cast<ScalarFloat>(tw(i).imag())),
                };
            }
            return buf;
        };

        auto const order = GENERATE(as<std::size_t>{}, 2, 3, 4, 5, 6);
        auto const size  = 1UL << order;

        auto inout = make_noise_signal(size);

        auto const copy              = inout;
        auto const forward_twiddles  = make_twiddles(size, neo::fft::direction::forward);
        auto const backward_twiddles = make_twiddles(size, neo::fft::direction::backward);

        neo::fft::execute_radix2_kernel(kernel, inout.to_mdspan(), forward_twiddles.to_mdspan());
        neo::fft::execute_radix2_kernel(kernel, inout.to_mdspan(), backward_twiddles.to_mdspan());

        for (auto i{0U}; i < inout.extent(0); ++i) {
            auto const real_batch = inout(i).real();
            auto const imag_batch = inout(i).imag();

            auto reals = std::array<ScalarFloat, ScalarBatch::size>{};
            auto imags = std::array<ScalarFloat, ScalarBatch::size>{};
            real_batch.store_unaligned(reals.data());
            imag_batch.store_unaligned(imags.data());

            auto const expected_real_batch = copy(i).real();
            auto const expected_imag_batch = copy(i).imag();

            auto expected_reals = std::array<ScalarFloat, ScalarBatch::size>{};
            auto expected_imags = std::array<ScalarFloat, ScalarBatch::size>{};
            expected_real_batch.store_unaligned(expected_reals.data());
            expected_imag_batch.store_unaligned(expected_imags.data());

            auto reals_float          = std::array<float, ScalarBatch::size>{};
            auto imags_float          = std::array<float, ScalarBatch::size>{};
            auto expected_reals_float = std::array<float, ScalarBatch::size>{};
            auto expected_imags_float = std::array<float, ScalarBatch::size>{};
            std::copy(reals.begin(), reals.end(), reals_float.begin());
            std::copy(imags.begin(), imags.end(), imags_float.begin());
            std::copy(expected_reals.begin(), expected_reals.end(), expected_reals_float.begin());
            std::copy(expected_imags.begin(), expected_imags.end(), expected_imags_float.begin());

            auto const scalar = ScalarFloat(1) / static_cast<ScalarFloat>(size);
            neo::scale(scalar, stdex::mdspan{reals_float.data(), stdex::extents{reals_float.size()}});
            neo::scale(scalar, stdex::mdspan{imags_float.data(), stdex::extents{imags_float.size()}});

            REQUIRE(neo::allclose(
                stdex::mdspan{reals_float.data(), stdex::extents{reals_float.size()}},
                stdex::mdspan{expected_reals_float.data(), stdex::extents{expected_reals_float.size()}},
                0.1F
            ));

            REQUIRE(neo::allclose(
                stdex::mdspan{imags_float.data(), stdex::extents{imags_float.size()}},
                stdex::mdspan{expected_imags_float.data(), stdex::extents{expected_imags_float.size()}},
                0.1F
            ));
        }
    };

    test(neo::fft::radix2_kernel_v1{});
    test(neo::fft::radix2_kernel_v2{});
    test(neo::fft::radix2_kernel_v3{});
    test(neo::fft::radix2_kernel_v4{});
}
#endif
