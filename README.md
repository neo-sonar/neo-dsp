# Perceptual Convolution

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

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
