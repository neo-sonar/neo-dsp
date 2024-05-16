// SPDX-License-Identifier: MIT

#pragma once

#include <neo/algorithm/copy.hpp>
#include <neo/complex.hpp>
#include <neo/container/mdspan.hpp>

namespace neo::convolution {

/// \ingroup neo-convolution
template<typename IndexType = std::size_t>
struct fdl_index
{
    using value_type = IndexType;

    fdl_index() noexcept = default;

    explicit fdl_index(IndexType num_segments) : _num_segments{num_segments} {}

    auto reset() -> void { _write_pos = 0; }

    template<std::invocable<IndexType> CopyCallback, std::invocable<IndexType, IndexType> MultiplyCallback>
    auto operator()(CopyCallback copy_callback, MultiplyCallback callback) -> void
    {
        copy_callback(_write_pos);

        for (IndexType segment{0}; segment < _num_segments; ++segment) {
            auto const filter_index = static_cast<IndexType>((_write_pos + _num_segments - segment) % _num_segments);
            callback(segment, filter_index);
        }

        if (++_write_pos; _write_pos >= _num_segments) {
            reset();
        }
    }

private:
    IndexType _num_segments{0};
    IndexType _write_pos{0};
};

}  // namespace neo::convolution
