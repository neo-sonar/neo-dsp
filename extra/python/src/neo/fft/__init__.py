# SPDX-License-Identifier: MIT

from __future__ import annotations

from .. import _neo

import numpy as np

__all__ = [
    "fft",
    "ifft",
    "rfftfreq",
]

NORM_MODE = {
    "backward": _neo.norm.backward,
    "ortho": _neo.norm.ortho,
    "forward": _neo.norm.forward,
}


def fft(x: np.ndarray, n=None, norm="backward") -> np.ndarray:
    """Compute the 1-D discrete Fourier Transform.
    """
    return _neo.fft(x, n, NORM_MODE[norm])


def ifft(x: np.ndarray, n=None, norm="backward") -> np.ndarray:
    """Compute the 1-D inverse discrete Fourier Transform.
    """
    return _neo.ifft(x, n, NORM_MODE[norm])


def rfftfreq(n: int, d=1.0) -> np.ndarray:
    """Return the Discrete Fourier Transform sample frequencies.
    """
    return _neo.rfftfreq(n, d)
