# SIMD

## Fixed-Point

- Modelled after arms `q7_t` & `q15_t`

## Vector Hardware Support

|  Feature   | SSE2 |  SSE42  | AVX |  AVX2   | AVX512F | AVX512BF16 | AVX512FP | Apple Silicon | Raspberry Pi4 |
| :--------: | :--: | :-----: | :-: | :-----: | :-----: | :--------: | :------: | :-----------: | :-----------: |
|   `q7_t`   |      | **Yes** |     | **Yes** | **Yes** |            |          |    **Yes**    |  _Probably_   |
|  `q15_t`   |      | **Yes** |     | **Yes** | **Yes** |            |          |    **Yes**    |  _Probably_   |
|   `BF16`   |      |         |     |         |         |  **Yes**   |          |               |               |
| `_Float16` |      |         |     |         |         |            | **Yes**  |               |               |
