import math

import neo_dsp as neo
import numpy as np

from pytest import approx


def test_fft():
    assert neo.fft(np.ones(shape=32, dtype=np.complex64)).shape[0] == 32
    assert neo.fft(np.ones(shape=64, dtype=np.complex64)).shape[0] == 64
    assert neo.fft(np.ones(shape=128, dtype=np.complex64)).shape[0] == 128
    assert neo.fft(np.ones(shape=512, dtype=np.complex64)).shape[0] == 512

    assert neo.fft(np.ones(shape=32, dtype=np.complex128)).shape[0] == 32
    assert neo.fft(np.ones(shape=64, dtype=np.complex128)).shape[0] == 64
    assert neo.fft(np.ones(shape=128, dtype=np.complex128)).shape[0] == 128
    assert neo.fft(np.ones(shape=512, dtype=np.complex128)).shape[0] == 512

    impulse = np.zeros(shape=8, dtype=np.complex128)
    impulse[0] = 1.0
    out: np.ndarray = neo.ifft(neo.fft(impulse, n=16))
    print(np.real(out))


def test_fast_log2():
    assert neo.fast_log2([64.0]) == approx(math.log2(64))
    assert neo.fast_log2([256.0]) == approx(math.log2(256))
    assert neo.fast_log2([1024.0]) == approx(math.log2(1024))


def test_fast_log10():
    assert neo.fast_log10([64.0]) == approx(math.log10(64))
    assert neo.fast_log10([256.0]) == approx(math.log10(256))
    assert neo.fast_log10([1024.0]) == approx(math.log10(1024))


def test_a_weighting():
    assert neo.a_weighting(24.5) == approx(-45.30166390)
    assert neo.a_weighting(49.0) == approx(-30.64262470)
    assert neo.a_weighting(98.0) == approx(-19.42442872)
