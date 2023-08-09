#pragma once

#include <complex>
#include <concepts>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <vector>

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
