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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fast_log2(dtype):
    values = np.array([64.0, 128.0, 512.0, 2048.0], dtype=dtype)
    assert neo.fast_log2(values) == approx(np.log2(values))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fast_log10(dtype):
    values = np.array([64.0, 128.0, 512.0, 2048.0], dtype=dtype)
    assert neo.fast_log10(values) == approx(np.log10(values))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("val, expected", [(1.0, 0.0), (0.5, -6.02059991328)])
def test_amplitude_to_db(dtype, val, expected):
    val = np.array([val], dtype=dtype)
    expected = np.array([expected], dtype=dtype)
    assert neo.amplitude_to_db(val) == approx(expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("val, expected", [(0.5, -6.02059991328)])
def test_amplitude_to_db_estimate(dtype, val, expected):
    val = np.array([val], dtype=dtype)
    expected = np.array([expected], dtype=dtype)
    assert neo.amplitude_to_db(
        val, precision="estimate") == approx(expected, rel=1e-4)


def test_fftfreq():
    assert neo.fftfreq(2) == approx([0.0, 0.5])
    assert neo.fftfreq(2, 1.0 / 20.0) == approx([0.0, 10.0])
    assert neo.fftfreq(2, 1.0 / 44100.0) == approx([0.0, 22050.0])


def test_a_weighting():
    assert neo.a_weighting([24.5]) == approx([-45.30166390])
    assert neo.a_weighting([49.0]) == approx([-30.64262470])
    assert neo.a_weighting([98.0]) == approx([-19.42442872])
