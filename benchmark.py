import numpy as np
import torch
import time

# === CPU baseline estimators ===
from original_mutinfo.mutinfo.knn import KSG as KSG_CPU

# === CUDA estimators ===
from smi_torch.knn import KSG as KSG_CUDA

def generate_correlated_gaussians(n_samples=1000, dim=5, noise=0.1):
    x = np.random.randn(n_samples, dim)
    y = x + noise * np.random.randn(n_samples, dim)
    return x, y

def test_estimator(cpu_estimator, cuda_estimator, x_np, y_np, name):
    print(f"\n=== Testing {name} ===")
    
    # --- CPU ---
    t0 = time.time()
    mi_cpu = cpu_estimator(x_np, y_np)
    t1 = time.time()
    print(f"[CPU] MI: {mi_cpu:.4f} (time: {t1 - t0:.3f}s)")

    # --- CUDA ---
    x_torch = torch.tensor(x_np, device='cuda')
    y_torch = torch.tensor(y_np, device='cuda')
    t0 = time.time()
    mi_cuda = cuda_estimator(x_torch, y_torch)
    t1 = time.time()
    print(f"[CUDA] MI: {mi_cuda:.4f} (time: {t1 - t0:.3f}s)")

    # --- Difference ---
    abs_diff = abs(mi_cpu - mi_cuda)
    rel_error = abs_diff / (mi_cpu + 1e-8)
    print(f"[DIFF] Abs: {abs_diff:.4f}, Rel: {rel_error:.2%}")

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    x_np, y_np = generate_correlated_gaussians(n_samples=2000, dim=1000, noise=0.0001)

    # KSG
    ksg_cpu = KSG_CPU(k_neighbors=5)
    ksg_cuda = KSG_CUDA(k_neighbors=5)
    test_estimator(ksg_cpu, ksg_cuda, x_np, y_np, "KSG Estimator")

if __name__ == "__main__":
    main()
