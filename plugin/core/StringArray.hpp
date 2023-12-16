#pragma once

#include <juce_core/juce_core.h>

namespace neo {

[[nodiscard]] auto toStringArray(auto const& values) -> juce::StringArray
{
    auto names = juce::StringArray{};
    for (auto const& value : values) {
        names.add(value.toString());
    }
    return names;
}

}  // namespace neo
