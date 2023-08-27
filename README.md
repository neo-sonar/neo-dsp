# Perceptual Convolution

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build](https://github.com/neo-sonar/plugin-perceptual-convolution/actions/workflows/build.yml/badge.svg)](https://github.com/neo-sonar/plugin-perceptual-convolution/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/neo-sonar/plugin-perceptual-convolution/branch/main/graph/badge.svg?token=PLQUR85CI6)](https://codecov.io/gh/neo-sonar/plugin-perceptual-convolution)

## Convolvers

### upols_convolver

Uniformly partitioned overlap-save convolver.

### sparse_upols_convolver

Uniformly partitioned overlap-save convolver with a sparse frequency delay line (FDL) on the filter.

## Frequency Delay Line

- dense `(mdarray)`
- sparse `(sparse_matrix)`
- sparse static mixed bit-depth `(Nx sparse_matrix, fixed non-overlapping slices)`
- sparse dynamic mixed bit-depth `(Nx sparse_matrix, overlapping slices)`
- sparse adaptive dynamic-depth `(individual scalar for each FDL row)`

## Math

### Compressed Float

- `static_cast<IntegerType>(val * scale)`
- `static_cast<float>(compressed * inv_scale)`

### Fixed-Point

- Modelled after arms `q7_t` & `q15_t`

### Float16

### Vector Hardware Support

|  Feature   | SSE2 |  SSE42  | AVX |  AVX2   | AVX512F | AVX512BF16 | AVX512FP | Apple Silicon | Raspberry Pi4 |
| :--------: | :--: | :-----: | :-: | :-----: | :-----: | :--------: | :------: | :-----------: | :-----------: |
|   `q7_t`   |      | **Yes** |     | **Yes** | **Yes** |            |          |    **Yes**    |  _Probably_   |
|  `q15_t`   |      | **Yes** |     | **Yes** | **Yes** |            |          |    **Yes**    |  _Probably_   |
|   `BF16`   |      |         |     |         |         |  **Yes**   |          |               |               |
| `_Float16` |      |         |     |         |         |            | **Yes**  |               |               |

## Resources

### FFT

- [COMPUTING THE FAST FOURIER TRANSFORMON SIMD MICROPROCESSOR](https://www.cs.waikato.ac.nz/~ihw/PhD_theses/Anthony_Blake.pdf)
- [Notes on FFTs: for implementers](https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/)
- [OTFFT Library](http://wwwa.pikara.ne.jp/okojisan/otfft-en/stockham1.html)
- [Discrete fourier transform and convolution -- scaling factor?](https://mathematica.stackexchange.com/questions/206788/discrete-fourier-transform-and-convolution-scaling-factor)
- [FXT Library](https://www.jjj.de/fxt/demo/arith/index.html)
- [FXT Library: Algorithms for programmers ideas and source code](http://dsp-book.narod.ru/fxtbook.pdf)

### DSP

- [musicinformationretrieval.com](https://musicinformationretrieval.com/index.html)
- [Audio Signal Processing for Machine Learning](https://youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0&si=51JQNk_IuZSZITxX)
- [AudioLabsErlangen](https://www.youtube.com/@AudioLabsErlangen/videos)
