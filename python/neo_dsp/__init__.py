from __future__ import annotations

from . import _core
from ._core import (
    __doc__,
    __version__,
    a_weighting,
    fast_log2,
    fast_log10
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
    "backward": _core.norm.backward,
    "ortho": _core.norm.ortho,
    "forward": _core.norm.forward,
}


def fft(x, n=None, norm="backward") -> np.ndarray:
    """Compute the 1-D discrete Fourier Transform.
    """
    return _core.fft(x, n, NORM_MODE[norm])


def ifft(x, n=None, norm="backward") -> np.ndarray:
    """Compute the 1-D inverse discrete Fourier Transform.
    """
    return _core.ifft(x, n, NORM_MODE[norm])
