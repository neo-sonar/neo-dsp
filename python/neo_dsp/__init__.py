from __future__ import annotations

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
]

NORM_MODE = {
    "backward": _neo_dsp.norm.backward,
    "ortho": _neo_dsp.norm.ortho,
    "forward": _neo_dsp.norm.forward,
}


def fftfreq(n, d=1.0) -> np.ndarray:
    """Return the Discrete Fourier Transform sample frequencies.
    """
    return _neo_dsp.fftfreq(n, d)


def fft(x, n=None, norm="backward") -> np.ndarray:
    """Compute the 1-D discrete Fourier Transform.
    """
    return _neo_dsp.fft(x, n, NORM_MODE[norm])


def ifft(x, n=None, norm="backward") -> np.ndarray:
    """Compute the 1-D inverse discrete Fourier Transform.
    """
    return _neo_dsp.ifft(x, n, NORM_MODE[norm])


def amplitude_to_db(x, precision="accurate") -> np.ndarray:
    """Convert an amplitude to dB-Scale.
    """
    if precision == "estimate":
        return _neo_dsp.amplitude_to_db_estimate(x)
    return _neo_dsp.amplitude_to_db(x)
