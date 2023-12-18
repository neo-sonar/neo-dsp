from __future__ import annotations

from .. import _neo_dsp

import numpy as np

__all__ = [
    "fft",
    "ifft",
    "rfftfreq",
]

NORM_MODE = {
    "backward": _neo_dsp.norm.backward,
    "ortho": _neo_dsp.norm.ortho,
    "forward": _neo_dsp.norm.forward,
}


def fft(x, n=None, norm="backward") -> np.ndarray:
    """Compute the 1-D discrete Fourier Transform.
    """
    return _neo_dsp.fft(x, n, NORM_MODE[norm])


def ifft(x, n=None, norm="backward") -> np.ndarray:
    """Compute the 1-D inverse discrete Fourier Transform.
    """
    return _neo_dsp.ifft(x, n, NORM_MODE[norm])


def rfftfreq(n, d=1.0) -> np.ndarray:
    """Return the Discrete Fourier Transform sample frequencies.
    """
    return _neo_dsp.rfftfreq(n, d)
