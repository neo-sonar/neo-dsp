#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace neo {

namespace detail {

template<typename StorageType>
constexpr auto saturate(std::int32_t x) -> StorageType
{
    auto const min_v = static_cast<std::int32_t>(std::numeric_limits<StorageType>::min());
    auto const max_v = static_cast<std::int32_t>(std::numeric_limits<StorageType>::max());
    return static_cast<StorageType>(std::clamp(x, min_v, max_v));
}

}  // namespace detail

struct underlying_value_t
{
    explicit underlying_value_t() = default;
};

inline constexpr auto underlying_value = underlying_value_t{};

template<int IntegerBits, int FractionalBits, typename StorageType>
struct fixed_point
{
    using storage_type = StorageType;

    static constexpr auto const integer_bits    = IntegerBits;
    static constexpr auto const fractional_bits = FractionalBits;
    static constexpr auto const scale           = static_cast<float>(1 << FractionalBits);
    static constexpr auto const inv_scale       = 1.0F / scale;

    constexpr fixed_point() = default;

    template<std::floating_point Float>
    explicit constexpr fixed_point(Float val) noexcept
        : _value{detail::saturate<storage_type>(static_cast<std::int32_t>(static_cast<float>(val) * scale))}
    {}

    constexpr fixed_point([[maybe_unused]] underlying_value_t tag, storage_type val) noexcept : _value{val} {}

    template<std::floating_point Float>
    [[nodiscard]] constexpr operator Float() const noexcept
    {
        return static_cast<float>(_value) * inv_scale;
    }

    [[nodiscard]] constexpr auto value() const noexcept -> storage_type { return _value; }

    [[nodiscard]] constexpr auto operator+() const -> fixed_point { return *this; }

    [[nodiscard]] constexpr auto operator-() const -> fixed_point
    {
        auto const min_v = std::numeric_limits<StorageType>::min();
        auto const max_v = std::numeric_limits<StorageType>::max();

        return fixed_point{
            underlying_value,
            value() == min_v ? max_v : static_cast<StorageType>(-value()),
        };
    }

    friend constexpr auto operator+(fixed_point lhs, fixed_point rhs) -> fixed_point
    {
        return {
            underlying_value,
            detail::saturate<StorageType>(lhs.value() + rhs.value()),
        };
    }

    friend constexpr auto operator-(fixed_point lhs, fixed_point rhs) -> fixed_point
    {
        return {
            underlying_value,
            detail::saturate<StorageType>(lhs.value() - rhs.value()),
        };
    }

    friend constexpr auto operator*(fixed_point lhs, fixed_point rhs) -> fixed_point
    {
        return {
            underlying_value,
            detail::saturate<StorageType>((lhs.value() * rhs.value()) >> fractional_bits),
        };
    }

    friend constexpr auto operator==(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() == rhs.value(); }

    friend constexpr auto operator!=(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() != rhs.value(); }

    friend constexpr auto operator<(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() < rhs.value(); }

    friend constexpr auto operator<=(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() <= rhs.value(); }

    friend constexpr auto operator>(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() > rhs.value(); }

    friend constexpr auto operator>=(fixed_point lhs, fixed_point rhs) -> bool { return lhs.value() >= rhs.value(); }

private:
    StorageType _value;
};

template<int IntegerBits, int FractionalBits, typename StorageType>
[[nodiscard]] constexpr auto to_float(fixed_point<IntegerBits, FractionalBits, StorageType> val) noexcept -> float
{
    return static_cast<float>(val);
}

template<int IntegerBits, int FractionalBits, typename StorageType>
[[nodiscard]] constexpr auto to_double(fixed_point<IntegerBits, FractionalBits, StorageType> val) noexcept -> double
{
    return static_cast<double>(val);
}

using q7  = fixed_point<0, 7, std::int8_t>;
using q15 = fixed_point<0, 15, std::int16_t>;

}  // namespace neo
