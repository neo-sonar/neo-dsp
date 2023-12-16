import math

import neo_dsp as neo
import numpy as np

import pytest
from pytest import approx


@pytest.mark.parametrize("n", [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("complex", [np.complex64, np.complex128])
def test_fft(n, complex):
    assert len(neo.fft(np.zeros(shape=n, dtype=complex)).shape) == 1
    assert neo.fft(np.zeros(shape=n, dtype=complex)).shape[0] == n

    impulse = np.zeros(shape=n, dtype=complex)
    impulse[0] = 1.0
    assert np.allclose(neo.ifft(neo.fft(impulse.copy())), impulse)


@pytest.mark.parametrize("float", [np.float32, np.float64])
def test_fast_log2(float):
    values = np.array([64.0, 128.0, 512.0, 2048.0], dtype=float)
    assert neo.fast_log2(values) == approx(np.log2(values))


@pytest.mark.parametrize("float", [np.float32, np.float64])
def test_fast_log10(float):
    values = np.array([64.0, 128.0, 512.0, 2048.0], dtype=float)
    assert neo.fast_log10(values) == approx(np.log10(values))


@pytest.mark.parametrize("float", [np.float32, np.float64])
@pytest.mark.parametrize("val, expected", [(1.0, 0.0), (0.5, -6.02059991328)])
def test_amplitude_to_db(float, val, expected):
    val = np.array([val], dtype=float)
    expected = np.array([expected], dtype=float)
    assert neo.amplitude_to_db(val) == approx(expected)


@pytest.mark.parametrize("float", [np.float32, np.float64])
@pytest.mark.parametrize("val, expected", [(0.5, -6.02059991328)])
def test_amplitude_to_db_estimate(float, val, expected):
    val = np.array([val], dtype=float)
    expected = np.array([expected], dtype=float)
    assert neo.amplitude_to_db(
        val, precision="estimate") == approx(expected, rel=1e-4)


def test_a_weighting():
    assert neo.a_weighting([24.5]) == approx([-45.30166390])
    assert neo.a_weighting([49.0]) == approx([-30.64262470])
    assert neo.a_weighting([98.0]) == approx([-19.42442872])
