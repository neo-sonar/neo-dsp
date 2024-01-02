import numpy as np

signal = np.array([0.0, 0.125, 0.25, 0.125, 0.0, -0.125, -0.075, -0.025]) * 3.0
filter = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
signal_spectrum = np.fft.fft(signal, norm="backward")
filter_spectrum = np.fft.fft(filter, norm="backward")
scaled = len(signal/2)**2*((signal_spectrum/len(signal/2))
                           * (filter_spectrum/len(signal/2)))
print("filter_spectrum", filter_spectrum)
print("signal_spectrum", signal_spectrum)
print("scaled", scaled)
print("Normal", np.fft.ifft(signal_spectrum * filter_spectrum))
print("Scaled", np.fft.ifft(scaled))


assert np.allclose(signal_spectrum, signal_spectrum * filter_spectrum)
assert np.allclose(signal_spectrum, scaled)
