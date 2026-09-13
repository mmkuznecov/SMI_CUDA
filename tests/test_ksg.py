import math

import numpy as np
import pytest
import torch

import mi_cuda
from original_mutinfo.mutinfo.knn import KSG as KSG_CPU
from oracles import ksg_mi_numpy, ksg_statistics_numpy, ksg_statistics_torch
from smi_torch import KSG

ALGORITHMS = ["brute", "sweep", "auto"]
DTYPES = [(torch.float32, np.float32), (torch.float64, np.float64)]


def correlated_sample(rng, n, dim_x, dim_y, noise=0.5, dtype=np.float64):
    x = rng.standard_normal((n, dim_x))
    mixing = rng.standard_normal((dim_x, dim_y))
    y = x @ mixing / math.sqrt(dim_x) + noise * rng.standard_normal((n, dim_y))
    return x.astype(dtype), y.astype(dtype)


def to_cuda(*arrays):
    return [torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in arrays]


def assert_statistics_equal(stats, expected):
    eps, counts_x, counts_y = (s.cpu().numpy() for s in stats)
    np.testing.assert_array_equal(eps, expected[0])
    np.testing.assert_array_equal(counts_x, expected[1])
    np.testing.assert_array_equal(counts_y, expected[2])


# --------------------------------------------------------------------------------------
# Exactness of the CUDA kernels
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("torch_dtype,np_dtype", DTYPES)
@pytest.mark.parametrize(
    "n,dim_x,dim_y,k",
    [
        (2, 1, 1, 1),
        (3, 1, 1, 2),
        (10, 1, 1, 9),
        (257, 1, 1, 1),
        (257, 1, 3, 4),
        (500, 2, 2, 5),
        (400, 4, 1, 32),   # largest heap on the GPU thread stack
        (400, 3, 3, 33),   # smallest heap in the scratch buffer
        (300, 2, 5, 150),
        (400, 1, 40, 3),   # "auto": sweep counts for x, brute k-NN and counts for y
        (300, 7, 5, 6),    # "auto": sweep k-NN, brute counts for x, sweep counts for y
        (300, 8, 6, 2),    # "auto": brute k-NN and counts for x, sweep counts for y
        (600, 16, 16, 3),
        (300, 200, 50, 5),
    ],
)
def test_statistics_match_bruteforce_oracle(algorithm, torch_dtype, np_dtype, n, dim_x, dim_y, k):
    rng = np.random.default_rng(n * 1000 + dim_x * 10 + dim_y + k)
    x, y = correlated_sample(rng, n, dim_x, dim_y, dtype=np_dtype)
    stats = mi_cuda.ksg_statistics(*to_cuda(x, y), k, algorithm)
    assert stats[0].dtype == torch_dtype and stats[1].dtype == torch.int32
    assert_statistics_equal(stats, ksg_statistics_numpy(x, y, k))


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("dim_x,dim_y", [(1, 1), (2, 1), (3, 2)])
@pytest.mark.parametrize("k", [1, 3, 40])
def test_statistics_with_ties_and_duplicates(algorithm, dim_x, dim_y, k):
    rng = np.random.default_rng(dim_x * 100 + dim_y * 10 + k)
    n = 400
    # Few distinct values: many equal distances, duplicated points and zero radii.
    x = rng.integers(0, 4, size=(n, dim_x)).astype(np.float64)
    y = (x[:, :1] + rng.integers(0, 2, size=(n, dim_y))).astype(np.float64)
    x[:50] = x[50:100]
    y[:50] = y[50:100]
    expected = ksg_statistics_numpy(x, y, k)
    assert (expected[0] == 0).any() or k > 1
    assert_statistics_equal(mi_cuda.ksg_statistics(*to_cuda(x, y), k, algorithm), expected)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_batched_statistics_match_individual(algorithm):
    rng = np.random.default_rng(7)
    batch, n, k = 6, 350, 4
    xs, ys = zip(*(correlated_sample(rng, n, 2, 3, noise=0.1 + i) for i in range(batch)))
    x, y = np.stack(xs), np.stack(ys)
    eps, counts_x, counts_y = mi_cuda.ksg_statistics(*to_cuda(x, y), k, algorithm)
    assert eps.shape == counts_x.shape == counts_y.shape == (batch, n)
    for b in range(batch):
        assert_statistics_equal((eps[b], counts_x[b], counts_y[b]), ksg_statistics_numpy(x[b], y[b], k))


@pytest.mark.parametrize("dim_x,dim_y,dtype", [(1, 1, torch.float64), (3, 2, torch.float32)])
def test_large_sample_matches_gpu_oracle(dim_x, dim_y, dtype):
    torch.manual_seed(0)
    n, k = 40_000, 5
    x = torch.randn(n, dim_x, device="cuda", dtype=dtype)
    y = x[:, :1] * 0.8 + 0.6 * torch.randn(n, dim_y, device="cuda", dtype=dtype)
    expected = ksg_statistics_torch(x, y, k, chunk=512)
    for algorithm in ["brute", "sweep"]:
        eps, counts_x, counts_y = mi_cuda.ksg_statistics(x, y, k, algorithm)
        assert torch.equal(eps, expected[0])
        assert torch.equal(counts_x.long(), expected[1])
        assert torch.equal(counts_y.long(), expected[2])


def test_many_kernel_launches_match_gpu_oracle():
    # N * D is large enough that every kernel is split into several launches.
    torch.manual_seed(1)
    n, dim, k = 6_000, 300, 3
    x = torch.randn(n, dim, device="cuda")
    y = x + 0.3 * torch.randn(n, dim, device="cuda")
    expected = ksg_statistics_torch(x, y, k, chunk=16)
    for algorithm in ["brute", "sweep"]:
        eps, counts_x, counts_y = mi_cuda.ksg_statistics(x, y, k, algorithm)
        assert torch.equal(eps, expected[0])
        assert torch.equal(counts_x.long(), expected[1])
        assert torch.equal(counts_y.long(), expected[2])


@pytest.mark.parametrize("algorithm", ["brute", "sweep"])
def test_statistics_follow_sample_permutation(algorithm):
    torch.manual_seed(2)
    n, k = 3000, 4
    x = torch.randn(n, 2, device="cuda", dtype=torch.float64)
    y = x.sum(1, keepdim=True) + torch.randn(n, 1, device="cuda", dtype=torch.float64)
    perm = torch.randperm(n, device="cuda")
    stats = mi_cuda.ksg_statistics(x, y, k, algorithm)
    stats_perm = mi_cuda.ksg_statistics(x[perm], y[perm], k, algorithm)
    for s, sp in zip(stats, stats_perm):
        assert torch.equal(s[perm], sp)


def test_non_default_stream_and_non_contiguous_input():
    torch.manual_seed(3)
    n, k = 2000, 3
    x = torch.randn(4, n, device="cuda").t()  # non-contiguous (n, 4)
    y = torch.randn(n, 2, device="cuda") + x[:, :2]
    expected = mi_cuda.ksg_statistics(x.contiguous(), y, k)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        stats = mi_cuda.ksg_statistics(x, y, k)
    stream.synchronize()
    for s, e in zip(stats, expected):
        assert torch.equal(s, e)


# --------------------------------------------------------------------------------------
# MI values
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(
    "n,dim_x,dim_y,k,noise",
    [
        (100, 1, 1, 1, 0.5),
        (1000, 1, 1, 5, 0.1),
        (1000, 5, 5, 5, 0.1),
        (2000, 10, 3, 3, 1.0),
        (800, 3, 3, 50, 0.3),
        (500, 50, 50, 5, 2.0),
    ],
)
def test_mi_matches_reference_mutinfo(algorithm, n, dim_x, dim_y, k, noise):
    rng = np.random.default_rng(n + dim_x + dim_y + k)
    x, y = correlated_sample(rng, n, dim_x, dim_y, noise=noise)
    expected, expected_std = KSG_CPU(k_neighbors=k)(x, y, std=True)
    mi, mi_std = KSG(k_neighbors=k, algorithm=algorithm)(*to_cuda(x, y), std=True)
    assert mi == pytest.approx(expected, abs=1e-10)
    assert mi_std == pytest.approx(expected_std, abs=1e-12)


def test_mi_matches_oracle_with_clamping_and_ties():
    rng = np.random.default_rng(11)
    n = 300
    x = rng.standard_normal((n, 2))
    y = rng.standard_normal((n, 2))  # independent: raw estimate may be negative
    for k in [1, 4, 20]:
        expected, expected_std = ksg_mi_numpy(x, y, k)
        mi, mi_std = KSG(k)(*to_cuda(x, y), std=True)
        assert mi >= 0.0
        assert mi == pytest.approx(expected, abs=1e-10)
        assert mi_std == pytest.approx(expected_std, abs=1e-12)

    x = rng.integers(0, 5, size=(n, 1)).astype(np.float64)
    y = x + rng.integers(0, 2, size=(n, 1))
    for k in [1, 4]:
        assert KSG(k)(*to_cuda(x, y)) == pytest.approx(ksg_mi_numpy(x, y, k)[0], abs=1e-10)


@pytest.mark.parametrize("rho", [0.0, 0.6, 0.9])
def test_mi_of_gaussian_matches_analytic_value(rho):
    torch.manual_seed(4)
    n = 20_000
    x = torch.randn(n, 1, device="cuda", dtype=torch.float64)
    y = rho * x + math.sqrt(1 - rho ** 2) * torch.randn(n, 1, device="cuda", dtype=torch.float64)
    expected = -0.5 * math.log(1 - rho ** 2)
    mi, mi_std = KSG(k_neighbors=5)(x, y, std=True)
    assert abs(mi - expected) < 0.03
    assert 0 < mi_std < 0.05


def test_float32_input_close_to_float64():
    torch.manual_seed(5)
    n = 5000
    x = torch.randn(n, 3, device="cuda", dtype=torch.float64)
    y = x + 0.5 * torch.randn(n, 3, device="cuda", dtype=torch.float64)
    mi64 = KSG(5)(x, y)
    mi32 = KSG(5)(x.float(), y.float())
    assert mi32 == pytest.approx(mi64, abs=5e-3)


def test_batched_estimate_matches_single_calls():
    torch.manual_seed(6)
    batch, n = 5, 700
    x = torch.randn(batch, n, 2, device="cuda")
    y = x * torch.linspace(0.1, 2.0, batch, device="cuda")[:, None, None] + torch.randn(batch, n, 2, device="cuda")
    ksg = KSG(4)
    mi, mi_std = ksg.estimate(x, y)
    assert mi.shape == mi_std.shape == (batch,) and mi.dtype == torch.float64
    for b in range(batch):
        single_mi, single_std = ksg(x[b], y[b], std=True)
        assert mi[b].item() == pytest.approx(single_mi, abs=1e-12)
        assert mi_std[b].item() == pytest.approx(single_std, abs=1e-12)


def test_raw_extension_scalar_outputs():
    torch.manual_seed(7)
    x = torch.randn(500, 2, device="cuda")
    y = x + torch.randn(500, 2, device="cuda")
    mi, mi_std = mi_cuda.ksg_mi(x, y, 3)
    assert mi.dim() == 0 and mi_std.dim() == 0 and mi.dtype == torch.float64
    assert mi.item() == pytest.approx(KSG(3)(x, y), abs=1e-12)


# --------------------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------------------

def test_input_shapes_and_dtypes_are_normalised():
    torch.manual_seed(8)
    n = 600
    x = torch.randn(n, device="cuda", dtype=torch.float64)
    y = x + torch.randn(n, device="cuda", dtype=torch.float64)
    expected = KSG(3)(x[:, None], y[:, None])
    assert KSG(3)(x, y) == expected                        # 1-D samples
    assert KSG(3)(x[:, None, None], y[:, None]) == expected  # extra axes are flattened
    assert KSG(3)(x, y.float()) == KSG(3)(x, y.float().double())  # promoted to float64

    xi = torch.randint(0, 50, (n, 2), device="cuda")
    yi = xi + torch.randint(0, 3, (n, 2), device="cuda")
    assert KSG(2)(xi, yi) == KSG(2)(xi.float(), yi.float())  # integers -> float32


def test_k_neighbors_is_clamped_like_reference():
    rng = np.random.default_rng(9)
    x, y = correlated_sample(rng, 6, 1, 1)
    expected = KSG_CPU(k_neighbors=10)(x, y)
    assert KSG(10)(*to_cuda(x, y)) == pytest.approx(expected, abs=1e-12)


def test_invalid_arguments_raise():
    x = torch.randn(100, 2, device="cuda")
    y = torch.randn(100, 2, device="cuda")
    with pytest.raises(ValueError):
        KSG(0)
    with pytest.raises(ValueError):
        KSG(1, algorithm="kd_tree")
    with pytest.raises(TypeError):
        KSG(1)(x.cpu().numpy(), y)
    with pytest.raises(ValueError):
        KSG(1)(x.cpu(), y.cpu())
    with pytest.raises(ValueError):
        KSG(1)(x, y[:50])
    with pytest.raises(ValueError):
        KSG(1)(x[:1], y[:1])
    bad = x.clone()
    bad[3, 1] = float("nan")
    with pytest.raises(ValueError):
        KSG(1)(bad, y)
    bad[3, 1] = float("inf")
    with pytest.raises(ValueError):
        KSG(1)(x, bad)

    with pytest.raises(RuntimeError):
        mi_cuda.ksg_mi(x.half(), y.half(), 1)
    with pytest.raises(RuntimeError):
        mi_cuda.ksg_mi(x, y.double(), 1)
    with pytest.raises(RuntimeError):
        mi_cuda.ksg_mi(x.cpu(), y.cpu(), 1)
    with pytest.raises(RuntimeError):
        mi_cuda.ksg_mi(x, y, 0)
    with pytest.raises(RuntimeError):
        mi_cuda.ksg_mi(x, y, 100)
    with pytest.raises(RuntimeError):
        mi_cuda.ksg_mi(x, y, 1, "fast")
    with pytest.raises(RuntimeError):
        mi_cuda.ksg_mi(x[None], y, 1)
    with pytest.raises(RuntimeError):
        mi_cuda.ksg_mi(x[:, :0], y, 1)
