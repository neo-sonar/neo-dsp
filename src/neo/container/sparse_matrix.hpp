#pragma once

#include <neo/container/mdspan.hpp>

#include <cstddef>
#include <iterator>
#include <span>
#include <type_traits>
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
    sparse_matrix(size_type rows, size_type cols);

    template<in_matrix InMat, std::predicate<IndexType, IndexType, T> Filter>
        requires std::is_convertible_v<typename InMat::value_type, T>
    sparse_matrix(InMat matrix, Filter filter);

    [[nodiscard]] auto extent(size_type e) const noexcept -> size_type;
    [[nodiscard]] auto extents() const noexcept -> stdex::dextents<index_type, 2>;

    [[nodiscard]] auto rows() const noexcept -> size_type;
    [[nodiscard]] auto columns() const noexcept -> size_type;
    [[nodiscard]] auto size() const noexcept -> size_type;

    [[nodiscard]] auto operator()(index_type row, index_type col) const -> T;

    auto insert(index_type row, index_type col, T value) -> void;

    [[nodiscard]] auto value_container() const noexcept -> value_container_type const&;
    [[nodiscard]] auto column_container() const noexcept -> index_container_type const&;
    [[nodiscard]] auto row_container() const noexcept -> index_container_type const&;

private:
    stdex::dextents<index_type, 2> _extents;
    ValueContainer _values;
    IndexContainer _columIndices;
    IndexContainer _rowIndices;
};

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::sparse_matrix(size_type rows, size_type cols)
    : _extents{rows, cols}
    , _rowIndices(rows + 1UL, 0)
{}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
template<in_matrix InMat, std::predicate<IndexType, IndexType, T> Filter>
    requires std::is_convertible_v<typename InMat::value_type, T>
sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::sparse_matrix(InMat matrix, Filter filter)
    : sparse_matrix{matrix.extent(0), matrix.extent(1)}
{
    auto count = 0UL;
    for (auto row_idx{0UL}; row_idx < matrix.extent(0); ++row_idx) {
        auto const row = stdex::submdspan(matrix, row_idx, stdex::full_extent);
        for (auto col{0UL}; col < matrix.extent(1); ++col) {
            if (filter(row_idx, col, row(col))) {
                ++count;
            }
        }
    }

    _values.resize(count);
    _columIndices.resize(count);

    auto idx = 0UL;
    for (auto row_idx{0UL}; row_idx < matrix.extent(0); ++row_idx) {
        auto const row       = stdex::submdspan(matrix, row_idx, stdex::full_extent);
        _rowIndices[row_idx] = idx;

        for (auto col{0UL}; col < matrix.extent(1); ++col) {
            if (auto const& val = row(col); filter(row_idx, col, val)) {
                _values[idx]       = val;
                _columIndices[idx] = col;
                ++idx;
            }
        }
    }

    _rowIndices.back() = idx;
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::extent(size_type e) const noexcept -> size_type
{
    return _extents.extent(e);
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::extents() const noexcept
    -> stdex::dextents<index_type, 2>
{
    return _extents;
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::rows() const noexcept -> size_type
{
    return extent(0);
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::columns() const noexcept -> size_type
{
    return extent(1);
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::size() const noexcept -> size_type
{
    return columns() * rows();
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
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::value_container() const noexcept
    -> value_container_type const&
{
    return _values;
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::column_container() const noexcept
    -> index_container_type const&
{
    return _columIndices;
}

template<typename T, typename IndexType, typename ValueContainer, typename IndexContainer>
auto sparse_matrix<T, IndexType, ValueContainer, IndexContainer>::row_container() const noexcept
    -> index_container_type const&
{
    return _rowIndices;
}

}  // namespace neo
