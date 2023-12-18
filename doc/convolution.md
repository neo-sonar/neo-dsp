## Convolution

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
