#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>

namespace neo {

template<typename IndexType = std::size_t>
struct fdl_index
{
    using value_type = IndexType;

    fdl_index() noexcept = default;

    explicit fdl_index(IndexType num_subfilter) : _num_subfilter{num_subfilter} {}

    auto reset() -> void { _write_pos = 0; }

    template<std::invocable<IndexType> CopyCallback, std::invocable<IndexType, IndexType> MultiplyCallback>
    auto operator()(CopyCallback copy_callback, MultiplyCallback callback) -> void
    {
        copy_callback(_write_pos);

        for (IndexType i{0}; i < _num_subfilter; ++i) {
            auto const filter_index = static_cast<IndexType>((_write_pos + _num_subfilter - i) % _num_subfilter);
            callback(i, filter_index);
        }

        if (++_write_pos; _write_pos >= _num_subfilter) {
            reset();
        }
    }

private:
    IndexType _num_subfilter{0};
    IndexType _write_pos{0};
};

}  // namespace neo
