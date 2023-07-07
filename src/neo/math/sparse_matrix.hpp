#pragma once

#include "neo/mdspan.hpp"

#include <cstddef>
#include <iterator>
#include <vector>

namespace neo
{

template<typename T, typename IndexType = std::size_t, typename ValueContainer = std::vector<T>,
         typename IndexContainer = std::vector<IndexType>>
struct sparse_matrix
{
    using value_type           = T;
    using size_type            = std::size_t;
    using index_type           = IndexType;
    using value_container_type = ValueContainer;
    using index_container_type = IndexContainer;

    sparse_matrix() = default;
    sparse_matrix(size_type rows, size_type cols) : _rows{rows}, _columns{cols}, _rowIndices(_rows + 1UL, 0) {}

    template<typename U, typename Extents, typename Layout, typename Accessor, typename Filter>
    sparse_matrix(Kokkos::mdspan<U, Extents, Layout, Accessor> other, Filter filter);

    [[nodiscard]] auto rows() const noexcept -> size_type { return _rows; }
    [[nodiscard]] auto columns() const noexcept -> size_type { return _columns; }
    [[nodiscard]] auto size() const noexcept -> size_type { return columns() * rows(); }

    [[nodiscard]] auto operator()(index_type row, index_type col) const -> T;

    auto insert(index_type row, index_type col, T value) -> void;

    auto value_container() const noexcept -> value_container_type const& { return _values; }
    auto column_container() const noexcept -> index_container_type const& { return _columIndices; }
    auto row_container() const noexcept -> index_container_type const& { return _rowIndices; }

private:
    size_type _rows{0};
    size_type _columns{0};
    ValueContainer _values;
    IndexContainer _columIndices;
    IndexContainer _rowIndices;
};

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
template<typename U, typename Extents, typename Layout, typename Accessor, typename Filter>
sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::sparse_matrix(
    Kokkos::mdspan<U, Extents, Layout, Accessor> other, Filter filter)
    : sparse_matrix{other.extent(0), other.extent(1)}
{
    static_assert(std::is_convertible_v<U, T>);
    static_assert(decltype(other)::rank() == 2);

    auto count = 0UL;
    for (auto row{0UL}; row < other.extent(0); ++row)
    {
        for (auto col{0UL}; col < other.extent(1); ++col)
        {
            if (filter(other(row, col))) { ++count; }
        }
    }

    _values.resize(count);
    _columIndices.resize(count);

    auto idx = 0UL;
    for (auto row{0UL}; row < other.extent(0); ++row)
    {
        _rowIndices[row] = idx;

        for (auto col{0UL}; col < other.extent(1); ++col)
        {
            if (auto const& val = other(row, col); filter(val))
            {
                _values[idx]       = val;
                _columIndices[idx] = col;
                ++idx;
            }
        }
    }

    _rowIndices.back() = idx;
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::insert(index_type row, index_type col, T value)
    -> void
{
    auto idx = _rowIndices[row];
    while (idx < _rowIndices[row + 1] && _columIndices[idx] < col) { idx++; }

    auto const pidx = static_cast<std::ptrdiff_t>(idx);
    _values.insert(std::next(_values.begin(), pidx), value);
    _columIndices.insert(std::next(_columIndices.begin(), pidx), col);

    for (auto i{row + 1}; i <= rows(); ++i) { ++_rowIndices[i]; }
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::operator()(index_type row, index_type col) const -> T
{
    for (auto i = _rowIndices[row]; i < _rowIndices[row + 1]; i++)
    {
        if (_columIndices[i] == col) { return _values[i]; }
    }
    return T{};
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto schur_product_accumulate_columns(Kokkos::mdspan<T, Kokkos::dextents<std::size_t, 2>> lhs,
                                      sparse_matrix<T, IndexType, ValueContainer, IndexContainer> const& rhs,
                                      std::span<T> accumulator) -> void
{
    jassert(lhs.extent(0) == rhs.rows());
    jassert(lhs.extent(1) == rhs.columns());

    auto const& rrows = rhs.row_container();
    auto const& rcols = rhs.column_container();
    auto const& rvals = rhs.value_container();

    for (auto row(std::size_t{0}); row < rhs.rows(); ++row)
    {
        for (auto i = rrows[row]; i < rrows[row + 1]; i++)
        {
            auto col = rcols[i];
            accumulator[col] += rvals[i] * lhs(col, row);
        }
    }
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto schur_product_accumulate_columns(sparse_matrix<T, IndexType, ValueContainer, IndexContainer> const& lhs,
                                      sparse_matrix<T, IndexType, ValueContainer, IndexContainer> const& rhs,
                                      std::span<T> accumulator) -> void
{
    jassert(lhs.rows() == rhs.rows());
    jassert(lhs.columns() == rhs.columns());

    auto const& lrows = lhs.row_container();
    auto const& lcols = lhs.column_container();
    auto const& lvals = lhs.value_container();

    auto const& rrows = rhs.row_container();
    auto const& rcols = rhs.column_container();
    auto const& rvals = rhs.value_container();

    for (auto row(std::size_t{0}); row < lhs.rows(); ++row)
    {
        auto const rowStart = lrows[row];
        auto const rowEnd   = lrows[row + 1];

        auto const otherRowStart = rrows[row];
        auto const otherRowEnd   = rrows[row + 1];

        auto i = rowStart;
        auto j = otherRowStart;

        while (i < rowEnd && j < otherRowEnd)
        {
            auto colIndex      = lcols[i];
            auto otherColIndex = rcols[j];

            if (colIndex < otherColIndex) { i++; }
            else if (colIndex > otherColIndex) { j++; }
            else
            {
                auto const newValue = lvals[i] * rvals[j];
                accumulator[colIndex] += newValue;
                i++;
                j++;
            }
        }
    }
}

}  // namespace neo
