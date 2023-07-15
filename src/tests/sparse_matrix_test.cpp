#include "neo/convolution/container/sparse_matrix.hpp"

#undef NDEBUG
#include <cassert>
#include <cstdio>

[[maybe_unused]] static auto testSparseMatrix() -> bool
{
    auto lhs = KokkosEx::mdarray<float, Kokkos::dextents<std::size_t, 2>>{16, 32};
    std::fill(lhs.data(), std::next(lhs.data(), std::ssize(lhs)), 1.0F);

    auto rhs = neo::sparse_matrix<float>{16, 32};
    assert(rhs.rows() == 16);
    assert(rhs.columns() == 32);

    auto accumulator = std::vector<float>(lhs.extent(1));
    neo::multiply_elementwise_accumulate_columnwise<float>(lhs.to_mdspan(), rhs, std::span<float>{accumulator});
    assert(std::all_of(accumulator.begin(), accumulator.end(), [](auto x) { return x == 0.0F; }));

    rhs.insert(0, 0, 2.0F);
    neo::multiply_elementwise_accumulate_columnwise<float>(lhs.to_mdspan(), rhs, std::span<float>{accumulator});
    assert(accumulator[0] == 2.0F);
    assert(std::all_of(std::next(accumulator.begin()), accumulator.end(), [](auto x) { return x == 0.0F; }));
    // std::fill(accumulator.begin(), accumulator.end(), 0.0F);

    auto other = neo::sparse_matrix<float>{lhs.to_mdspan(), [](auto v) { return v >= 1.0F; }};
    assert(other.rows() == lhs.extent(0));
    assert(other.columns() == lhs.extent(1));
    assert(other.value_container().size() == lhs.size());

    other = neo::sparse_matrix<float>{lhs.to_mdspan(), [](auto v) { return v >= 2.0F; }};
    assert(other.rows() == lhs.extent(0));
    assert(other.columns() == lhs.extent(1));
    assert(other.value_container().size() == 0);

    auto row = std::vector<float>(lhs.extent(1));
    std::fill(row.begin(), row.end(), 2.0F);
    other.insert_row(0, std::span{row}, [](auto) { return true; });
    assert(other.value_container().size() == 32);
    assert(other.column_container().size() == 32);
    return true;
}

auto main() -> int
{
    assert(testSparseMatrix());
    return EXIT_SUCCESS;
}
