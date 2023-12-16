import neo_dsp as neo
import numpy as np

neo.convolve(np.ones((8))[::1])
neo.convolve(np.ones((8))[::2])
neo.convolve(np.ones((4, 3), order='C', dtype=np.complex128))
neo.convolve(np.ones((4, 3), order='F', dtype=np.float32))
neo.convolve(np.ones((4, 3), order='F')[::2, :])
neo.convolve(np.ones((4, 3, 2), order='F'))
neo.convolve(np.ones((4, 3, 16), order='F')[::2, :, ::4])
neo.convolve(np.ones((16, 16, 2, 1, 2, 3, 4, 16), order='C'))

print(neo.a_weighting([440.0, 1024.0, 10000.0]))
print(neo.fast_log2([1024.0]))
print(np.log2([1024.0]))
print(neo.fast_log10([1024.0]))
print(np.log10([1024.0]))
