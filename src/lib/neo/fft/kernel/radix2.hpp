#pragma once

#include <neo/config.hpp>

#include <neo/container/mdspan.hpp>
#include <neo/math/ilog2.hpp>
#include <neo/math/ipow.hpp>

#include <utility>

namespace neo::fft {

struct radix2_kernel_v1
{
    radix2_kernel_v1() = default;

    auto operator()(inout_vector auto x, auto const& twiddles) const noexcept -> void
    {
        auto const size  = x.size();
        auto const order = static_cast<std::int32_t>(ilog2(size));

        {
            // stage 0
            static constexpr auto const stage_length = 1;  // ipow<2>(0)
            static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

            for (auto k{0}; k < static_cast<int>(size); k += stride) {
                auto const i1 = k;
                auto const i2 = k + stage_length;

                auto const temp = x[i1] + x[i2];
                x[i2]           = x[i1] - x[i2];
                x[i1]           = temp;
            }
        }

        auto stage_length = 2;
        auto stride       = 4;

        for (auto stage = 1; stage < order; ++stage) {
            auto const tw_stride = ipow<2>(order - stage - 1);

            for (auto k = 0; std::cmp_less(k, size); k += stride) {
                for (auto pair = 0; pair < stage_length; ++pair) {
                    auto const tw = twiddles[static_cast<std::size_t>(pair * tw_stride)];

                    auto const i1 = static_cast<std::size_t>(k + pair);
                    auto const i2 = static_cast<std::size_t>(k + pair + stage_length);

                    auto const temp = x[i1] + tw * x[i2];
                    x[i2]           = x[i1] - tw * x[i2];
                    x[i1]           = temp;
                }
            }

            stage_length *= 2;
            stride *= 2;
        }
    }
};

struct radix2_kernel_v2
{
    radix2_kernel_v2() = default;

    auto operator()(inout_vector auto x, auto const& twiddles) const noexcept -> void
    {
        auto const size = x.size();

        {
            // stage 0
            static constexpr auto const stage_length = 1;  // ipow<2>(0)
            static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

            for (auto k{0}; k < static_cast<int>(size); k += stride) {
                auto const i1 = k;
                auto const i2 = k + stage_length;

                auto const temp = x[i1] + x[i2];
                x[i2]           = x[i1] - x[i2];
                x[i1]           = temp;
            }
        }

        auto stage_size = 4U;
        while (stage_size <= size) {
            auto const halfStage = stage_size / 2;
            auto const k_step    = size / stage_size;

            for (auto i{0U}; i < size; i += stage_size) {
                for (auto k{i}; k < i + halfStage; ++k) {
                    auto const index = k - i;
                    auto const tw    = twiddles[index * k_step];

                    auto const idx1 = k;
                    auto const idx2 = k + halfStage;

                    auto const even = x[idx1];
                    auto const odd  = x[idx2];

                    auto const tmp = odd * tw;
                    x[idx1]        = even + tmp;
                    x[idx2]        = even - tmp;
                }
            }

            stage_size *= 2;
        }
    }
};

struct radix2_kernel_v3
{
    radix2_kernel_v3() = default;

    auto operator()(inout_vector auto x, auto const& twiddles) const noexcept -> void
    {
        auto const size  = x.size();
        auto const order = ilog2(size);

        {
            // stage 0
            static constexpr auto const stage_length = 1;  // ipow<2>(0)
            static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

            for (auto k{0}; k < static_cast<int>(size); k += stride) {
                auto const i1 = k;
                auto const i2 = k + stage_length;

                auto const temp = x[i1] + x[i2];
                x[i2]           = x[i1] - x[i2];
                x[i1]           = temp;
            }
        }

        for (auto stage{1ULL}; stage < order; ++stage) {

            auto const stage_length = ipow<2ULL>(stage);
            auto const stride       = ipow<2ULL>(stage + 1);
            auto const tw_stride    = ipow<2ULL>(order - stage - 1ULL);

            for (auto k{0ULL}; k < size; k += stride) {
                for (auto pair{0ULL}; pair < stage_length; ++pair) {
                    auto const tw = twiddles[pair * tw_stride];

                    auto const i1 = k + pair;
                    auto const i2 = k + pair + stage_length;

                    auto const temp = x[i1] + tw * x[i2];
                    x[i2]           = x[i1] - tw * x[i2];
                    x[i1]           = temp;
                }
            }
        }
    }
};

struct radix2_kernel_v4
{
    radix2_kernel_v4() = default;

    auto operator()(inout_vector auto x, auto const& twiddles) const noexcept -> void
    {
        auto const size  = static_cast<int>(x.size());
        auto const order = static_cast<int>(ilog2(size));

        {
            // stage 0
            static constexpr auto const stage_length = 1;  // ipow<2>(0)
            static constexpr auto const stride       = 2;  // ipow<2>(0 + 1)

            for (auto k{0}; k < size; k += stride) {
                auto const i1 = k;
                auto const i2 = k + stage_length;

                auto const temp = x[i1] + x[i2];
                x[i2]           = x[i1] - x[i2];
                x[i1]           = temp;
            }
        }

        for (auto stage{1}; stage < order; ++stage) {

            auto const stage_length = ipow<2>(stage);
            auto const stride       = ipow<2>(stage + 1);
            auto const tw_stride    = ipow<2>(order - stage - 1);

            for (auto k{0}; k < size; k += stride) {
                for (auto pair{0}; pair < stage_length; pair += 2) {
                    {
                        auto const p0 = pair;
                        auto const tw = twiddles[p0 * tw_stride];

                        auto const i1 = k + p0;
                        auto const i2 = k + p0 + stage_length;

                        auto const temp = x[i1] + tw * x[i2];
                        x[i2]           = x[i1] - tw * x[i2];
                        x[i1]           = temp;
                    }

                    {
                        auto const p1 = pair + 1;
                        auto const tw = twiddles[p1 * tw_stride];

                        auto const i1 = k + p1;
                        auto const i2 = k + p1 + stage_length;

                        auto const temp = x[i1] + tw * x[i2];
                        x[i2]           = x[i1] - tw * x[i2];
                        x[i1]           = temp;
                    }
                }
            }
        }
    }
};

}  // namespace neo::fft
