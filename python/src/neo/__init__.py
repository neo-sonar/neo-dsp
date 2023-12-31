# SPDX-License-Identifier: MIT

from __future__ import annotations

from . import fft
from . import _neo
from ._neo import (
    __doc__,
    __version__,
    a_weighting,
    fast_log2,
    fast_log10,
)

import numpy as np

__all__ = [
    "__doc__",
    "__version__",
    "a_weighting",
    "fast_log2",
    "fast_log10",
    "amplitude_to_db",
    "convolve",
    "fft",
]

CONVOLUTION_MODE = {
    "full": _neo.convolution_mode.full,
    "valid": _neo.convolution_mode.valid,
    "same": _neo.convolution_mode.same,
}


def amplitude_to_db(x, precision="accurate") -> np.ndarray:
    """Convert an amplitude to dB-Scale.
    """
    if precision == "estimate":
        return _neo.amplitude_to_db_estimate(x)
    return _neo.amplitude_to_db(x)


def convolve(in1: np.ndarray, in2: np.ndarray, mode: str = "full", method: str = "auto") -> np.ndarray:
    """Convolve two 1-dimensional arrays.
    """
    if method == "fft":
        return _neo.fft_convolve(in1, in2, CONVOLUTION_MODE[mode])
    return _neo.direct_convolve(in1, in2, CONVOLUTION_MODE[mode])
