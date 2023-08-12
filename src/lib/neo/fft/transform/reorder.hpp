#pragma once

#include <neo/container/mdspan.hpp>
#include <neo/math/ilog2.hpp>

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace neo::fft {

[[nodiscard]] inline auto make_bit_reversed_index_table(std::size_t size) -> std::vector<std::size_t>
{
    auto const order = ilog2(size);
    auto table       = std::vector<std::size_t>(size, 0);
    for (auto i{0U}; i < size; ++i) {
        for (auto j{0U}; j < order; ++j) {
            table[i] |= ((i >> j) & 1) << (order - 1 - j);
        }
    }
    return table;
}

template<inout_vector InOutVec>
constexpr auto bit_reverse_permutation(InOutVec x) -> void
{
    // Rearrange the input in bit-reversed order
    std::size_t j = 0;
    for (std::size_t i = 0; i < x.size() - 1U; ++i) {
        if (i < j) {
            std::swap(x[i], x[j]);
        }
        std::size_t k = x.size() / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
}

template<inout_vector InOutVec, typename IndexTable>
auto bit_reverse_permutation(InOutVec x, IndexTable const& index) -> void
{
    for (auto i{0U}; i < x.extent(0); ++i) {
        if (i < index[i]) {
            std::swap(x[i], x[index[i]]);
        }
    }
}

}  // namespace neo::fft
