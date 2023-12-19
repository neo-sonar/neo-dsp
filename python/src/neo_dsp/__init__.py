from __future__ import annotations

from . import fft
from . import _neo_dsp
from ._neo_dsp import (
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
    "full": _neo_dsp.convolution_mode.full,
    "valid": _neo_dsp.convolution_mode.valid,
    "same": _neo_dsp.convolution_mode.same,
}


def amplitude_to_db(x, precision="accurate") -> np.ndarray:
    """Convert an amplitude to dB-Scale.
    """
    if precision == "estimate":
        return _neo_dsp.amplitude_to_db_estimate(x)
    return _neo_dsp.amplitude_to_db(x)


def convolve(in1: np.ndarray, in2: np.ndarray, mode: str = "full", method: str = "auto") -> np.ndarray:
    """Convolve two 1-dimensional arrays.
    """
    if method == "fft":
        return _neo_dsp.fft_convolve(in1, in2, CONVOLUTION_MODE[mode])
    return _neo_dsp.direct_convolve(in1, in2, CONVOLUTION_MODE[mode])
