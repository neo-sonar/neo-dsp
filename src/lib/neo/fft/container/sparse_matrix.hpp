#pragma once

#include <neo/fft/container/mdspan.hpp>

#include <cassert>
#include <cstddef>
#include <iterator>
#include <span>
#include <vector>

namespace neo {

template<
    typename T,
    typename IndexType      = std::size_t,
    typename ValueContainer = std::vector<T>,
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

    template<typename U, std::size_t Extent, typename Filter>
    auto insert_row(index_type row, std::span<U, Extent> values, Filter filter) -> void;

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
    Kokkos::mdspan<U, Extents, Layout, Accessor> other,
    Filter filter
)
    : sparse_matrix{other.extent(0), other.extent(1)}
{
    static_assert(std::is_convertible_v<U, T>);
    static_assert(decltype(other)::rank() == 2);

    auto count = 0UL;
    for (auto rowIdx{0UL}; rowIdx < other.extent(0); ++rowIdx) {
        auto const row = KokkosEx::submdspan(other, rowIdx, Kokkos::full_extent);
        for (auto col{0UL}; col < other.extent(1); ++col) {
            if (filter(rowIdx, col, row(col))) {
                ++count;
            }
        }
    }

    _values.resize(count);
    _columIndices.resize(count);

    auto idx = 0UL;
    for (auto rowIdx{0UL}; rowIdx < other.extent(0); ++rowIdx) {
        auto const row      = KokkosEx::submdspan(other, rowIdx, Kokkos::full_extent);
        _rowIndices[rowIdx] = idx;

        for (auto col{0UL}; col < other.extent(1); ++col) {
            if (auto const& val = row(col); filter(rowIdx, col, val)) {
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
    while (idx < _rowIndices[row + 1] && _columIndices[idx] < col) {
        idx++;
    }

    auto const pidx = static_cast<std::ptrdiff_t>(idx);
    _values.insert(std::next(_values.begin(), pidx), value);
    _columIndices.insert(std::next(_columIndices.begin(), pidx), col);

    for (auto i{row + 1}; i <= rows(); ++i) {
        ++_rowIndices[i];
    }
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
template<typename U, std::size_t Extent, typename Filter>
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::insert_row(
    index_type row,
    std::span<U, Extent> values,
    Filter filter
) -> void
{
    auto const rowStart    = _rowIndices[row];
    auto const rowEnd      = _rowIndices[row + 1];
    auto const currentSize = static_cast<std::ptrdiff_t>(rowEnd - rowStart);
    auto const newSize     = [=] {
        auto count = 0L;
        for (auto i{0UL}; i < values.size(); ++i) {
            count += static_cast<long>(filter(row, i, values[i]));
        }
        return count;
    }();

    if (newSize < currentSize) {
        auto const delta = currentSize - newSize;
        std::shift_left(std::next(_values.begin(), static_cast<ptrdiff_t>(rowEnd)), _values.end(), delta);
        std::shift_left(std::next(_columIndices.begin(), static_cast<ptrdiff_t>(rowEnd)), _columIndices.end(), delta);

        auto nextRow = std::next(_rowIndices.begin(), static_cast<ptrdiff_t>(row + 1));
        std::transform(nextRow, _rowIndices.end(), nextRow, [delta](auto idx) { return idx - size_t(delta); });
    } else if (newSize > currentSize) {
        auto const delta = newSize - currentSize;

        _values.resize(_values.size() + size_t(delta));
        _columIndices.resize(_columIndices.size() + size_t(delta));

        std::shift_right(std::next(_values.begin(), static_cast<ptrdiff_t>(rowEnd)), _values.end(), delta);
        std::shift_right(std::next(_columIndices.begin(), static_cast<ptrdiff_t>(rowEnd)), _columIndices.end(), delta);

        auto nextRow = std::next(_rowIndices.begin(), static_cast<ptrdiff_t>(row + 1));
        std::transform(nextRow, _rowIndices.end(), nextRow, [delta](auto idx) { return idx + size_t(delta); });
    }

    auto idx = 0UL;
    for (auto i{0UL}; i < values.size(); ++i) {
        auto const val = values[i];
        if (filter(row, i, val)) {
            _values[rowStart + idx]       = val;
            _columIndices[rowStart + idx] = i;
            ++idx;
        }
    }
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::operator()(index_type row, index_type col) const -> T
{
    for (auto i = _rowIndices[row]; i < _rowIndices[row + 1]; i++) {
        if (_columIndices[i] == col) {
            return _values[i];
        }
    }
    return T{};
}

}  // namespace neo
