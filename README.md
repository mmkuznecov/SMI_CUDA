# SMI_CUDA

Fast Sliced Mutual Information Estimator in CUDA

`smi_torch` provides GPU implementations of the Kraskov-Stogbauer-Grassberger (KSG)
mutual information estimator and of k-Sliced Mutual Information (SMI). The results
are exact with respect to the CPU reference implementation in `original_mutinfo`
(identical neighbour counts, MI equal up to floating point round-off).

## Installation

The CUDA toolkit used for the build must have the same major version as the CUDA
version PyTorch was built with (`python -c "import torch; print(torch.version.cuda)"`).
Point `CUDA_HOME` to it if it is not the default toolkit.

```bash
cd smi_torch/csrc/
pip install . --no-build-isolation
```

For development, the extension can also be built in place (`python setup.py build_ext --inplace`);
the tests pick up an in-place build automatically.

To check:

```bash
python -c "import mi_cuda; print('Success!')"
```

## Usage

```python
import torch
from smi_torch import KSG, SMI

x = torch.randn(10_000, 50, device="cuda")
y = x + torch.randn(10_000, 50, device="cuda")

mi, mi_std = KSG(k_neighbors=5)(x, y, std=True)
smi = SMI(KSG(k_neighbors=5), projection_dim=1, n_projection_samples=128)(x, y)
```

* Inputs must be CUDA tensors. float64 inputs are processed in float64, all other
  dtypes in float32. Samples must be finite.
* `KSG.estimate(x, y)` evaluates a batch of independent sample sets of shape
  `(batch, n_samples, dim)` and returns per-batch tensors without leaving the GPU.
* The raw extension exposes `mi_cuda.ksg_mi(x, y, k, algorithm="auto")` and
  `mi_cuda.ksg_statistics(x, y, k, algorithm="auto")` (k-th neighbour radii and
  marginal neighbour counts) for inputs of shape `(N, D)` or `(B, N, D)`.

## Algorithm

For every sample the KSG estimator needs the distance to its k-th nearest neighbour
in the joint space (Chebyshev norm) and the number of samples strictly inside that
distance in each marginal space.

* No `N x N` distance matrix is built. Every GPU thread handles one sample and keeps
  the k smallest distances in a max-heap; distance evaluation stops as soon as a
  candidate cannot enter the heap. Memory is `O(N)`.
* `sweep`: samples are sorted along their widest coordinate and every query scans
  outwards in sorted order until the gap along that coordinate reaches the current
  k-th distance. Since the Chebyshev distance is never smaller than a single coordinate
  gap this is exact, but only a thin slab of candidates is visited in low dimensions.
  Neighbour counts in one-dimensional marginal spaces (e.g. SMI with
  `projection_dim=1`) are two binary searches per sample.
* `brute`: all pairs are compared, which is faster in high dimensions where sorting
  along one coordinate prunes poorly.
* `auto` (default) chooses per stage: `sweep` for the joint search if
  `dim_x + dim_y <= 12`, and for marginal counts if the marginal dimension is `<= 6`.
  The choice only affects speed.
* SMI draws all projection matrices up front and evaluates the projections in large
  batches with a single kernel pass per batch instead of one Python-level call and
  host synchronisation per projection.

## Tests and benchmarks

```bash
python -m pytest tests          # requires a CUDA device
python benchmark.py             # CPU reference vs CUDA
python advanced_bencmark.py     # parameter sweep, writes reports/
```

The tests compare the kernels against an exact brute-force oracle (NumPy for small
inputs, chunked PyTorch for large ones), against the reference `mutinfo` estimators,
and check both algorithms, float32/float64, batching, ties and duplicate samples,
large heaps, split kernel launches and argument validation.
