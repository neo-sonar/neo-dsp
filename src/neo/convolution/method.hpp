// SPDX-License-Identifier: MIT

#pragma once

namespace neo::convolution {

enum struct method
{
    automatic,
    direct,
    fft,
    ola,
    ols,
    upola,
    upols,
};

}  // namespace neo::convolution
