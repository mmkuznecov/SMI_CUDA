"""Independent reference implementations used by the tests."""

import math

import numpy as np
import torch
from scipy.special import digamma


def _chebyshev_matrix(data: np.ndarray) -> np.ndarray:
    # Same floating point expression as the kernels: max_c |a_c - b_c| in the input dtype.
    return np.abs(data[:, None, :] - data[None, :, :]).max(-1)


def ksg_statistics_numpy(x: np.ndarray, y: np.ndarray, k: int):
    """Exact KSG radii and strict marginal counts by brute force (small N only)."""
    n = x.shape[0]
    dx = _chebyshev_matrix(x)
    dy = _chebyshev_matrix(y)
    dxy = np.maximum(dx, dy)
    np.fill_diagonal(dxy, np.inf)
    eps = np.partition(dxy, k - 1, axis=1)[:, k - 1]
    off_diagonal = ~np.eye(n, dtype=bool)
    counts_x = ((dx < eps[:, None]) & off_diagonal).sum(1)
    counts_y = ((dy < eps[:, None]) & off_diagonal).sum(1)
    return eps, counts_x, counts_y


def ksg_mi_numpy(x: np.ndarray, y: np.ndarray, k: int):
    n = x.shape[0]
    _, counts_x, counts_y = ksg_statistics_numpy(x, y, k)
    array = digamma(counts_x + 1.0) + digamma(counts_y + 1.0)
    mi = max(0.0, digamma(k) + digamma(n) - array.mean())
    return mi, array.std() / math.sqrt(n)


def ksg_statistics_torch(x: torch.Tensor, y: torch.Tensor, k: int, chunk: int = 64):
    """Exact KSG radii and strict marginal counts computed in chunks on the GPU."""
    n = x.shape[0]
    eps = torch.empty(n, dtype=x.dtype, device=x.device)
    counts_x = torch.empty(n, dtype=torch.int64, device=x.device)
    counts_y = torch.empty(n, dtype=torch.int64, device=x.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows = torch.arange(start, stop, device=x.device)
        dx = (x[start:stop, None, :] - x[None, :, :]).abs().amax(-1)
        dy = (y[start:stop, None, :] - y[None, :, :]).abs().amax(-1)
        dxy = torch.maximum(dx, dy)
        dxy[rows - start, rows] = float("inf")
        e = dxy.kthvalue(k, dim=1).values
        eps[start:stop] = e
        dx[rows - start, rows] = float("inf")
        dy[rows - start, rows] = float("inf")
        counts_x[start:stop] = (dx < e[:, None]).sum(1)
        counts_y[start:stop] = (dy < e[:, None]).sum(1)
    return eps, counts_x, counts_y
