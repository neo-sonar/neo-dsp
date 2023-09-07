# Define the complex number class
class Complex:
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

def conj(c: Complex):
    return Complex(c.real, -c.imag)

# Function to precompute twiddle factors
def precompute_twiddle_factors(n, inverse=False):
    twiddles = [Complex(0, 0)] * n
    sign = -1 if inverse else 1  # Use -1 for the inverse FFT
    for k in range(n):
        angle = sign * 2 * math.pi * k / n
        twiddles[k] = Complex(math.cos(angle), math.sin(angle))
    return twiddles

# Function to perform radix-4 DIT FFT in-place with precomputed twiddle factors
def radix4_fft(data, twiddles, inverse=False):
    n = len(data)
    
    # Check if the input size is a power of 4
    if n == 1:
        return

    # Perform iterative FFT using a loop
    stages = int(math.log2(n) / 2)  # Number of stages (log2(n) / 2)
    step = 1  # Step size for each stage
    for stage in range(stages):
        for group in range(0, n, step * 4):
            for k in range(step):
                twiddle1 = conj(twiddles[k]) if inverse else twiddles[k]
                twiddle2 = conj(twiddles[k + step]) if inverse else twiddles[k + step]
                twiddle3 = conj(twiddles[k + 2 * step]) if inverse else twiddles[k + 2 * step]

                # Perform butterflies
                index1 = group + k
                index2 = group + step + k
                index3 = group + 2 * step + k
                index4 = group + 3 * step + k

                t1 = Complex(twiddle1.real * data[index2].real - twiddle1.imag * data[index2].imag,
                             twiddle1.real * data[index2].imag + twiddle1.imag * data[index2].real)
                t2 = Complex(twiddle2.real * data[index3].real - twiddle2.imag * data[index3].imag,
                             twiddle2.real * data[index3].imag + twiddle2.imag * data[index3].real)
                t3 = Complex(twiddle3.real * data[index4].real - twiddle3.imag * data[index4].imag,
                             twiddle3.real * data[index4].imag + twiddle3.imag * data[index4].real)

                # Update data in-place
                data[index1] = Complex(data[index1].real + t1.real + t2.real + t3.real,
                                      data[index1].imag + t1.imag + t2.imag + t3.imag)
                data[index2] = Complex(data[index1].real - t1.real + t2.imag - t3.imag,
                                      data[index1].imag - t1.imag - t2.real + t3.real)
                data[index3] = Complex(data[index1].real - t1.real - t2.real + t3.real,
                                      data[index1].imag - t1.imag + t2.imag - t3.imag)
                data[index4] = Complex(data[index1].real + t1.real - t2.imag - t3.imag,
                                      data[index1].imag + t1.imag + t2.real - t3.real)

            group += step * 4  # Move to the next group
        step *= 4  # Double the step size for the next stage

# Usage
if __name__ == "__main__":
    import math

    # Example input data (should be a power of 4)
    data = [Complex(0, 0)]*4**1
    data[0] = Complex(1, 0)

    n = len(data)
    tw = precompute_twiddle_factors(n, inverse=False)

    # Perform radix-4 DIT FFT in-place with precomputed twiddle factors
    radix4_fft(data, tw,inverse=False)
    radix4_fft(data, tw,inverse=True)

    # Print the FFT result
    for val in data:
        print(f"({val.real/n}, {val.imag/n})")