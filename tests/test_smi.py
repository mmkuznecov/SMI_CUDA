import math

import numpy as np
import pytest
import torch

from original_mutinfo.mutinfo.knn import KSG as KSG_CPU
from original_mutinfo.mutinfo.smi import SMI as SMI_CPU
from smi_torch import KSG, SMI, MutualInformationEstimator


class WrappedKSG(MutualInformationEstimator):
    """A base estimator that is not a KSG instance, forcing the per-projection path."""

    def __init__(self, k_neighbors):
        self.ksg = KSG(k_neighbors)
        self.calls = 0

    def __call__(self, x, y, std=False):
        self.calls += 1
        return self.ksg(x, y, std=std)


def sample(n, dim_x, dim_y, noise, dtype=torch.float64, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, dim_x, device="cuda", dtype=dtype, generator=generator)
    mixing = torch.randn(dim_x, dim_y, device="cuda", dtype=dtype, generator=generator)
    y = x @ mixing / math.sqrt(dim_x) + noise * torch.randn(n, dim_y, device="cuda", dtype=dtype, generator=generator)
    return x, y


def test_projection_matrices_are_orthonormal():
    smi = SMI(KSG(1), projection_dim=3, n_projection_samples=16)
    Q = smi.generate_random_projection_matrices(16, 20, "cuda", torch.float64)
    assert Q.shape == (16, 20, 3)
    eye = torch.eye(3, device="cuda", dtype=torch.float64).expand(16, 3, 3)
    assert torch.allclose(Q.transpose(1, 2) @ Q, eye, atol=1e-12)
    assert smi.generate_random_projection_matrix(20, "cuda").shape == (20, 3)


def test_batched_projection_matches_matmul():
    x, _ = sample(300, 12, 1, 1.0)
    Q = SMI(KSG(1), projection_dim=2).generate_random_projection_matrices(9, 12, "cuda", torch.float64)
    projected = SMI._project(x, Q)
    assert projected.shape == (9, 300, 2) and projected.is_contiguous()
    for i in range(9):
        assert torch.allclose(projected[i], x @ Q[i], rtol=0, atol=1e-12)


@pytest.mark.parametrize("projection_dim", [1, 2])
def test_projection_estimates_match_reference_ksg(projection_dim):
    x, y = sample(700, 10, 6, 0.5, seed=1)
    n_projections, k = 12, 4
    smi = SMI(KSG(k), projection_dim=projection_dim, n_projection_samples=n_projections, max_batch_size=5)
    Q_x = smi.generate_random_projection_matrices(n_projections, 10, "cuda", torch.float64)
    Q_y = smi.generate_random_projection_matrices(n_projections, 6, "cuda", torch.float64)

    results = smi.estimate_projections(x, y, Q_x, Q_y)
    assert results.shape == (n_projections,) and results.dtype == torch.float64

    x_proj, y_proj = SMI._project(x, Q_x), SMI._project(y, Q_y)
    for i in range(n_projections):
        # Identical projected samples: GPU KSG and the CPU reference implementation.
        assert results[i].item() == pytest.approx(KSG(k)(x_proj[i], y_proj[i]), abs=1e-12)
        expected = KSG_CPU(k_neighbors=k)(x_proj[i].cpu().numpy(), y_proj[i].cpu().numpy())
        assert results[i].item() == pytest.approx(expected, abs=1e-10)


def test_result_does_not_depend_on_batch_size():
    x, y = sample(1000, 8, 8, 1.0, seed=2)
    values = []
    for max_batch_size in [None, 1, 7, 1000]:
        torch.manual_seed(123)
        values.append(SMI(KSG(3), n_projection_samples=40, max_batch_size=max_batch_size)(x, y, std=True))
    # Equal up to round-off: BLAS and reduction kernels depend on the tensor shapes.
    for mi, mi_std in values:
        assert mi == pytest.approx(values[0][0], abs=1e-12)
        assert mi_std == pytest.approx(values[0][1], abs=1e-12)


def test_generic_estimator_path_matches_batched_path():
    x, y = sample(400, 5, 4, 0.5, seed=3)
    torch.manual_seed(321)
    batched = SMI(KSG(2), projection_dim=2, n_projection_samples=10)(x, y, std=True)
    wrapped = WrappedKSG(2)
    torch.manual_seed(321)
    generic = SMI(wrapped, projection_dim=2, n_projection_samples=10)(x, y, std=True)
    assert wrapped.calls == 10
    assert generic[0] == pytest.approx(batched[0], abs=1e-12)
    assert generic[1] == pytest.approx(batched[1], abs=1e-12)


def test_input_handling():
    x, y = sample(300, 4, 3, 0.5, seed=4)
    smi = SMI(KSG(2), n_projection_samples=8)
    torch.manual_seed(0)
    value = smi(x, y)
    torch.manual_seed(0)
    assert smi(x.reshape(300, 2, 2), y) == value
    assert isinstance(value, float)
    assert len(smi(x, y, std=True)) == 2

    torch.manual_seed(0)
    assert 0.0 <= SMI(KSG(2), n_projection_samples=8)(x.float(), y.float()) < 10

    with pytest.raises(ValueError):
        smi(x, y[:10])
    with pytest.raises(ValueError):
        smi(x.cpu(), y.cpu())
    with pytest.raises(ValueError):
        SMI(KSG(1), projection_dim=0)
    with pytest.raises(ValueError):
        SMI(KSG(1), n_projection_samples=0)
    with pytest.raises(ValueError):
        SMI(KSG(1), max_batch_size=0)


def test_independent_variables_have_small_smi():
    generator = torch.Generator(device="cuda").manual_seed(5)
    x = torch.randn(4000, 20, device="cuda", generator=generator)
    y = torch.randn(4000, 20, device="cuda", generator=generator)
    torch.manual_seed(5)
    assert SMI(KSG(5), n_projection_samples=64)(x, y) < 0.02


def test_statistically_consistent_with_reference_smi():
    x, y = sample(1500, 6, 6, 0.5, seed=6)
    x_np, y_np = x.cpu().numpy(), y.cpu().numpy()

    np.random.seed(0)
    expected = SMI_CPU(KSG_CPU(k_neighbors=5), n_projection_samples=256)(x_np, y_np)

    torch.manual_seed(0)
    mi, mi_std = SMI(KSG(5), n_projection_samples=1024)(x, y, std=True)

    assert mi_std > 0
    # Different random projections: allow for the Monte Carlo error of both estimates.
    assert abs(mi - expected) < 5 * math.sqrt(mi_std ** 2 + 4 * mi_std ** 2)
