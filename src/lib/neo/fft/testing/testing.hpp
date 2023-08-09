#pragma once

#include <algorithm>
#include <complex>
#include <concepts>
#include <filesystem>
#include <functional>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

namespace neo::fft {

[[nodiscard]] auto allclose(std::span<float const> lhs, std::span<float const> rhs, float tolerance = 1e-5F) -> bool;

[[nodiscard]] auto allclose(std::span<double const> lhs, std::span<double const> rhs, double tolerance = 1e-9) -> bool;

[[nodiscard]] auto
allclose(std::span<std::complex<float> const> lhs, std::span<std::complex<float> const> rhs, float tolerance = 1e-5F)
    -> bool;

[[nodiscard]] auto
allclose(std::span<std::complex<double> const> lhs, std::span<std::complex<double> const> rhs, double tolerance = 1e-9)
    -> bool;

template<typename T>
auto almost_equal(T x, T y, T tolerance = T(1e-5)) noexcept -> bool
{
    return std::abs(x - y) < tolerance;
}

template<typename T>
auto almost_equal(std::complex<T> x, std::complex<T> y, T tolerance = T(1e-5)) noexcept -> bool
{
    return std::abs(x - y) < tolerance;
}

namespace detail {

template<typename T, typename U = T>
auto allclose_impl(std::span<T const> lhs, std::span<T const> rhs, U tolerance) -> bool
{
    if (lhs.size() != rhs.size()) { return false; }
    return std::transform_reduce(
        lhs.begin(),
        lhs.end(),
        rhs.begin(),
        true,
        std::logical_and{},
        [tolerance](auto l, auto r) { return std::abs(l - r) < tolerance; }
    );
}

}  // namespace detail

inline auto allclose(std::span<float const> lhs, std::span<float const> rhs, float tolerance) -> bool
{
    return detail::allclose_impl<float>(lhs, rhs, tolerance);
}

inline auto allclose(std::span<double const> lhs, std::span<double const> rhs, double tolerance) -> bool
{
    return detail::allclose_impl<double>(lhs, rhs, tolerance);
}

inline auto
allclose(std::span<std::complex<float> const> lhs, std::span<std::complex<float> const> rhs, float tolerance) -> bool
{
    return detail::allclose_impl<std::complex<float>, float>(lhs, rhs, tolerance);
}

inline auto
allclose(std::span<std::complex<double> const> lhs, std::span<std::complex<double> const> rhs, double tolerance) -> bool
{
    return detail::allclose_impl<std::complex<double>, double>(lhs, rhs, tolerance);
}

template<typename Float>
auto rms_error(std::span<Float const> original, std::span<Float const> reconstructed) noexcept -> std::optional<Float>
{
    if (original.empty()) { return std::nullopt; }
    if (original.size() != reconstructed.size()) { return std::nullopt; }

    auto diffSquared = [](Float x, Float y) {
        auto const diff = x - y;
        return diff * diff;
    };

    auto sum = std::transform_reduce(
        original.begin(),
        original.end(),
        reconstructed.begin(),
        Float(0),
        std::plus{},
        diffSquared
    );

    return std::sqrt(sum / static_cast<Float>(original.size()));
}

template<typename Float>
auto rms_error(
    std::span<std::complex<Float const>> original,
    std::span<std::complex<Float const>> reconstructed
) noexcept -> std::optional<Float>
{
    if (original.empty()) { return std::nullopt; }
    if (original.size() != reconstructed.size()) { return std::nullopt; }

    auto diffSquared = [](std::complex<Float> x, std::complex<Float> y) {
        auto const re = x.real() - y.real();
        auto const im = x.imag() - y.imag();
        return (re * re) + (im * im);
    };

    auto sum = std::transform_reduce(
        original.begin(),
        original.end(),
        reconstructed.begin(),
        Float(0),
        std::plus{},
        diffSquared
    );

    return std::sqrt(sum / static_cast<Float>(original.size()));
}

}  // namespace neo::fft

struct test_path
{
    std::filesystem::path input;
    std::filesystem::path expected;
};

template<std::floating_point Float>
struct test_data
{
    std::vector<std::complex<Float>> input;
    std::vector<std::complex<Float>> expected;
};

template<std::floating_point Float>
[[nodiscard]] auto load_test_data_file(std::filesystem::path const& path)
    -> std::optional<std::vector<std::complex<Float>>>
{
    if (not std::filesystem::exists(path)) { return {}; }
    if (not std::filesystem::is_regular_file(path)) { return {}; }

    auto* file = std::fopen(path.string().c_str(), "r");
    if (file == nullptr) { return {}; }

    auto result = std::vector<std::complex<Float>>{};
    char line[512]{};
    while (std::fgets(line, sizeof(line), file)) {
        auto re = 0.0;
        auto im = 0.0;
        std::sscanf(line, "%lf,%lf\n", &re, &im);
        result.emplace_back(static_cast<Float>(re), static_cast<Float>(im));
    }

    std::fclose(file);

    return result;
}

template<std::floating_point Float>
[[nodiscard]] auto load_test_data(test_path const& paths) -> std::optional<test_data<Float>>
{
    auto input = load_test_data_file<Float>(paths.input);
    if (not input) { return {}; }

    auto expected = load_test_data_file<Float>(paths.expected);
    if (not expected) { return {}; }

    return test_data<Float>{
        .input    = std::move(*input),
        .expected = std::move(*expected),
    };
}
