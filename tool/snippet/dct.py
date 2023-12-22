import numpy as np
import scipy.fft as fft


def dct2_scipy(x):
    return fft.dct(x, type=2)


def dct2_2nfft_1(x):
    N = len(x)
    k = np.arange(N)
    y = np.empty(2*N)
    y[:N] = x
    y[N:] = x[::-1]

    Y = fft.fft(y)[:N]

    Y *= np.exp(-1j*np.pi*k/(2*N))
    return Y.real


def dct2_2nfft_2(x):
    N = len(x)
    k = np.arange(N)
    y = np.zeros(2*N)
    y[:N] = x

    Y = fft.fft(y)[:N]

    Y *= 2 * np.exp(-1j*np.pi*k/(2*N))
    return Y.real


def dct2_nfft(x):
    N = len(x)
    k = np.arange(N)
    v = np.empty_like(x)
    v[:(N-1)//2+1] = x[::2]

    if N % 2:  # odd length
        v[(N-1)//2+1:] = x[-2::-2]
    else:  # even length
        v[(N-1)//2+1:] = x[::-2]

    V = fft.fft(v)

    V *= 2 * np.exp(-1j*np.pi*k/(2*N))
    return V.real


np.set_printoptions(linewidth=200)
print(dct2_scipy([1, 2, 3, 4, 5, 6, 7, 8]))
print(dct2_2nfft_1([1, 2, 3, 4, 5, 6, 7, 8]))
print(dct2_2nfft_2([1, 2, 3, 4, 5, 6, 7, 8]))
print(dct2_nfft([1, 2, 3, 4, 5, 6, 7, 8]))

x = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
v = np.empty_like(x)
N = len(x)
print(x[-2])
print(x)

v[:(N-1)//2+1] = x[::2]
print(v)

if N % 2:
    v[(N-1)//2+1:] = x[-2::-2]
else:
    v[(N-1)//2+1:] = x[::-2]
print(v)
