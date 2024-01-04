# import numpy as  np

# # Define the complex number class
# class Complex:
#     def __init__(self, real, imag):
#         self.real = real
#         self.imag = imag


# # Function to precompute twiddle factors
# def precompute_twiddle_factors(n, inverse=False):
#     twiddles = [Complex(0, 0)] * n
#     sign = -1 if inverse else 1  # Use -1 for the inverse FFT
#     for k in range(n):
#         angle = sign * 2 * math.pi * k / n
#         twiddles[k] = Complex(math.cos(angle), math.sin(angle))
#     return twiddles

# def fft_dit4(data, twiddles):
#     n = len(data)

#     # Check if the input size is a power of 4
#     if n == 1:
#         return

#     # Perform iterative FFT using a loop
#     stages = int(math.log2(n) / 2)  # Number of stages (log2(n) / 2)
#     step = 1  # Step size for each stage
#     for stage in range(stages):
#         for group in range(0, n, step * 4):
#             for k in range(step):
#                 twiddle1 = twiddles[k]
#                 twiddle2 = twiddles[k + step]
#                 twiddle3 = twiddles[k + 2 * step]

#                 # Perform butterflies
#                 index1 = group + k
#                 index2 = group + step + k
#                 index3 = group + 2 * step + k
#                 index4 = group + 3 * step + k

#                 t1 = Complex(twiddle1.real * data[index2].real - twiddle1.imag * data[index2].imag,
#                              twiddle1.real * data[index2].imag + twiddle1.imag * data[index2].real)
#                 t2 = Complex(twiddle2.real * data[index3].real - twiddle2.imag * data[index3].imag,
#                              twiddle2.real * data[index3].imag + twiddle2.imag * data[index3].real)
#                 t3 = Complex(twiddle3.real * data[index4].real - twiddle3.imag * data[index4].imag,
#                              twiddle3.real * data[index4].imag + twiddle3.imag * data[index4].real)

#                 # Update data in-place
#                 data[index1] = Complex(data[index1].real + t1.real + t2.real + t3.real,
#                                        data[index1].imag + t1.imag + t2.imag + t3.imag)
#                 data[index2] = Complex(data[index1].real - t1.real + t2.imag - t3.imag,
#                                        data[index1].imag - t1.imag - t2.real + t3.real)
#                 data[index3] = Complex(data[index1].real - t1.real - t2.real + t3.real,
#                                        data[index1].imag - t1.imag + t2.imag - t3.imag)
#                 data[index4] = Complex(data[index1].real + t1.real - t2.imag - t3.imag,
#                                        data[index1].imag + t1.imag + t2.real - t3.real)

#             group += step * 4  # Move to the next group
#         step *= 4  # Double the step size for the next stage


# # Usage
# if __name__ == "__main__":
#     import math

#     # Example input data (should be a power of 4)
#     data = [Complex(0, 0)]*4**2
#     data[0] = Complex(1, 0)

#     n = len(data)

#     fft_dit4(data, precompute_twiddle_factors(n, inverse=False))
#     for val in data:
#         print(f"({val.real}, {val.imag})")
#     print("")

#     fft_dit4(data, precompute_twiddle_factors(n, inverse=True))
#     for val in data:
#         print(f"({val.real}, {val.imag})")

import cmath
import math
import numpy as np

def radix4_fft(x):
    N = len(x)
    if N <= 1:
        return x

    # Bit-reversal permutation
    bit_reversed = [0] * N
    for i in range(N):
        j = int(format(i, '0{:d}b'.format(int(math.log2(N)))), 2)
        bit_reversed[i] = x[j]

    # Cooley-Tukey Radix-4 FFT
    stage = 1
    while stage < N:
        w_m = cmath.exp(-2j * cmath.pi / stage)
        for k in range(0, N, stage * 4):
            w = 1
            for j in range(stage):
                a = bit_reversed[k + j]
                b = bit_reversed[k + j + stage]
                c = bit_reversed[k + j + 2 * stage]
                d = bit_reversed[k + j + 3 * stage]

                u = a + w * b
                v = a - w * b
                x = c + w * d
                y = (c - w * d) * cmath.exp(-1j * cmath.pi * j / (2 * stage))

                bit_reversed[k + j] = u + x
                bit_reversed[k + j + stage] = v + 1j * y
                bit_reversed[k + j + 2 * stage] = u - x
                bit_reversed[k + j + 3 * stage] = v - 1j * y

                w *= w_m
        stage *= 4

    return bit_reversed

def inverse_radix4_fft(X):
    N = len(X)
    if N <= 1:
        return X

    # Conjugate the input for the inverse FFT
    X_conjugate = [np.conj(x) for x in X]

    # Compute the forward FFT of the conjugated input
    Y_conjugate = radix4_fft(X_conjugate)

    # Conjugate the result of the forward FFT and scale
    y = [np.conj(y) / N for y in Y_conjugate]

    return y

order = 2
x = [0]* 4**order
x[0]=1
X = radix4_fft(x)
print("Forward Radix-4 FFT:", X)

x_reconstructed = inverse_radix4_fft(X)
print("Inverse Radix-4 FFT:", x_reconstructed)


# def precompute_twiddle_factors(N):
#     twiddle_factors = [cmath.exp(-2j * cmath.pi * k / N) for k in range(N // 4)]
#     return twiddle_factors

# def radix4_fft(x, twiddle_factors):
#     N = len(x)
#     if N <= 1:
#         return x

#     # Bit-reversal permutation
#     bit_reversed = [0] * N
#     for i in range(N):
#         j = int(format(i, '0{:d}b'.format(int(math.log2(N)))), 2)
#         bit_reversed[i] = x[j]

#     # Cooley-Tukey Radix-4 FFT
#     stage = 1
#     while stage < N:
#         twiddle_index = 0
#         for k in range(0, N, stage * 4):
#             for j in range(stage):
#                 a = bit_reversed[k + j]
#                 b = bit_reversed[k + j + stage]
#                 c = bit_reversed[k + j + 2 * stage]
#                 d = bit_reversed[k + j + 3 * stage]

#                 twiddle = twiddle_factors[twiddle_index]
#                 u = a + twiddle * b
#                 v = a - twiddle * b
#                 x = c + twiddle * d
#                 y = (c - twiddle * d) * cmath.exp(-1j * cmath.pi * j / (2 * stage))

#                 bit_reversed[k + j] = u + x
#                 bit_reversed[k + j + stage] = v + 1j * y
#                 bit_reversed[k + j + 2 * stage] = u - x
#                 bit_reversed[k + j + 3 * stage] = v - 1j * y

#                 twiddle_index += 1
#         stage *= 4

#     return bit_reversed

# def inverse_radix4_fft(X, twiddle_factors):
#     N = len(X)
#     if N <= 1:
#         return X

#     # Conjugate the input for the inverse FFT
#     X_conjugate = [np.conj(x) for x in X]

#     # Compute the forward FFT of the conjugated input using the precomputed twiddle factors
#     Y_conjugate = radix4_fft(X_conjugate, twiddle_factors)

#     # Conjugate the result of the forward FFT and scale
#     y = [np.conj(y) / N for y in Y_conjugate]

#     return y

# # Example usage:
# order = 2
# x = [0]* 4**order
# x[0] = 1
# N = len(x)
# twiddle_factors = precompute_twiddle_factors(N)
# X = radix4_fft(x, twiddle_factors)
# print("Forward Radix-4 FFT:", X)

# x_reconstructed = inverse_radix4_fft(X, twiddle_factors)
# print("Inverse Radix-4 FFT:", x_reconstructed)
