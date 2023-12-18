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
    "fft",
]


def amplitude_to_db(x, precision="accurate") -> np.ndarray:
    """Convert an amplitude to dB-Scale.
    """
    if precision == "estimate":
        return _neo_dsp.amplitude_to_db_estimate(x)
    return _neo_dsp.amplitude_to_db(x)
