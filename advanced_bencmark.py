import numpy as np
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# === CPU baseline estimators ===
from original_mutinfo.mutinfo.knn import KSG as KSG_CPU

# === CUDA estimators ===
from smi_torch.knn import KSG as KSG_CUDA

def generate_correlated_gaussians(n_samples=1000, dim=5, noise=0.1):
    """Generate correlated Gaussian data for mutual information estimation."""
    x = np.random.randn(n_samples, dim)
    y = x + noise * np.random.randn(n_samples, dim)
    return x, y

def run_benchmark(n_samples_list, dim_list, noise_list, k_list, num_runs=3):
    """Run benchmark comparing CPU and CUDA implementations across parameter combinations."""
    results = []
    
    # Set up a progress bar for the total number of combinations
    total_combinations = len(n_samples_list) * len(dim_list) * len(noise_list) * len(k_list) * num_runs
    pbar = tqdm(total=total_combinations)
    
    # Run benchmarks for each parameter combination
    for n_samples in n_samples_list:
        for dim in dim_list:
            for noise in noise_list:
                for k in k_list:
                    for run in range(num_runs):
                        # Generate data
                        x_np, y_np = generate_correlated_gaussians(n_samples=n_samples, dim=dim, noise=noise)
                        
                        # CPU implementation
                        ksg_cpu = KSG_CPU(k_neighbors=k)
                        t0 = time.time()
                        mi_cpu = ksg_cpu(x_np, y_np)
                        cpu_time = time.time() - t0
                        
                        # CUDA implementation
                        x_torch = torch.tensor(x_np, dtype=torch.float32, device='cuda')
                        y_torch = torch.tensor(y_np, dtype=torch.float32, device='cuda')
                        ksg_cuda = KSG_CUDA(k_neighbors=k)
                        
                        # Warm-up run for GPU
                        _ = ksg_cuda(x_torch, y_torch)
                        torch.cuda.synchronize()
                        
                        # Timed run
                        t0 = time.time()
                        mi_cuda = ksg_cuda(x_torch, y_torch)
                        torch.cuda.synchronize()
                        cuda_time = time.time() - t0
                        
                        # Calculate speedup
                        speedup = cpu_time / cuda_time
                        
                        # Calculate absolute and relative differences
                        abs_diff = abs(float(mi_cpu) - float(mi_cuda))
                        rel_error = abs_diff / (float(mi_cpu) + 1e-8)
                        
                        # Store results
                        results.append({
                            'n_samples': n_samples,
                            'dim': dim,
                            'noise': noise,
                            'k_neighbors': k,
                            'run': run + 1,
                            'mi_cpu': float(mi_cpu),
                            'mi_cuda': float(mi_cuda),
                            'cpu_time': cpu_time,
                            'cuda_time': cuda_time,
                            'speedup': speedup,
                            'abs_diff': abs_diff,
                            'rel_error': rel_error
                        })
                        
                        pbar.update(1)
    
    pbar.close()
    return pd.DataFrame(results)

def plot_results(df):
    """Create various plots analyzing benchmark results."""
    # Create a directory for saving plots
    os.makedirs('reports', exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Speedup vs Sample Size (grouped by dimension)
    plt.figure(figsize=(12, 8))
    for dim in sorted(df['dim'].unique()):
        subset = df[df['dim'] == dim].groupby('n_samples')['speedup'].mean().reset_index()
        plt.plot(subset['n_samples'], subset['speedup'], marker='o', label=f'dim={dim}')
    
    plt.xscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Speedup (CPU time / CUDA time)')
    plt.title('Speedup vs Sample Size (by dimension)')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/speedup_vs_samples.png', dpi=300, bbox_inches='tight')
    
    # 2. Speedup vs Dimension (grouped by sample size)
    plt.figure(figsize=(12, 8))
    for n_samples in sorted(df['n_samples'].unique()):
        subset = df[df['n_samples'] == n_samples].groupby('dim')['speedup'].mean().reset_index()
        plt.plot(subset['dim'], subset['speedup'], marker='o', label=f'n_samples={n_samples}')
    
    plt.xscale('log')
    plt.xlabel('Dimension')
    plt.ylabel('Speedup (CPU time / CUDA time)')
    plt.title('Speedup vs Dimension (by sample size)')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/speedup_vs_dim.png', dpi=300, bbox_inches='tight')
    
    # 3. CPU and CUDA time vs Sample Size (for a fixed dimension)
    mid_dim = sorted(df['dim'].unique())[len(df['dim'].unique())//2]
    plt.figure(figsize=(12, 8))
    subset = df[df['dim'] == mid_dim].groupby('n_samples')[['cpu_time', 'cuda_time']].mean().reset_index()
    
    plt.plot(subset['n_samples'], subset['cpu_time'], marker='o', label='CPU Time')
    plt.plot(subset['n_samples'], subset['cuda_time'], marker='s', label='CUDA Time')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Time (seconds)')
    plt.title(f'Computation Time vs Sample Size (dim={mid_dim})')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/time_vs_samples.png', dpi=300, bbox_inches='tight')
    
    # 4. CPU and CUDA time vs Dimension (for a fixed sample size)
    mid_samples = sorted(df['n_samples'].unique())[len(df['n_samples'].unique())//2]
    plt.figure(figsize=(12, 8))
    subset = df[df['n_samples'] == mid_samples].groupby('dim')[['cpu_time', 'cuda_time']].mean().reset_index()
    
    plt.plot(subset['dim'], subset['cpu_time'], marker='o', label='CPU Time')
    plt.plot(subset['dim'], subset['cuda_time'], marker='s', label='CUDA Time')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Dimension')
    plt.ylabel('Time (seconds)')
    plt.title(f'Computation Time vs Dimension (n_samples={mid_samples})')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/time_vs_dim.png', dpi=300, bbox_inches='tight')
    
    # 5. Relative Error vs Dimension
    plt.figure(figsize=(12, 8))
    for n_samples in sorted(df['n_samples'].unique()):
        subset = df[df['n_samples'] == n_samples].groupby('dim')['rel_error'].mean().reset_index()
        plt.plot(subset['dim'], subset['rel_error'], marker='o', label=f'n_samples={n_samples}')
    
    plt.xscale('log')
    plt.xlabel('Dimension')
    plt.ylabel('Relative Error')
    plt.title('Relative Error vs Dimension (by sample size)')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/error_vs_dim.png', dpi=300, bbox_inches='tight')
    
    # 6. Boxplot of Speedup by K neighbors
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='k_neighbors', y='speedup', data=df)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Speedup (CPU time / CUDA time)')
    plt.title('Speedup Distribution by Number of Neighbors')
    plt.grid(True)
    plt.savefig('reports/speedup_by_k.png', dpi=300, bbox_inches='tight')

def create_heatmaps(df):
    """Create heatmaps to visualize speedup across parameter combinations."""
    # Create heatmap of speedup by sample size and dimension
    plt.figure(figsize=(12, 10))
    
    # Pivot data for heatmap - Fixed the pivot call
    heatmap_data = df.groupby(['n_samples', 'dim'])['speedup'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='n_samples', columns='dim', values='speedup')
    
    # Create heatmap
    sns.heatmap(heatmap_pivot, annot=True, fmt=".1f", cmap="viridis", 
                cbar_kws={'label': 'Speedup (CPU time / CUDA time)'})
    
    plt.title('Speedup Heatmap: Sample Size vs. Dimension')
    plt.savefig('reports/speedup_heatmap_samples_dim.png', dpi=300, bbox_inches='tight')
    
    # Create heatmap for k vs. noise level
    plt.figure(figsize=(10, 8))
    
    heatmap_data = df.groupby(['k_neighbors', 'noise'])['speedup'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='k_neighbors', columns='noise', values='speedup')
    
    sns.heatmap(heatmap_pivot, annot=True, fmt=".1f", cmap="viridis",
                cbar_kws={'label': 'Speedup (CPU time / CUDA time)'})
    
    plt.title('Speedup Heatmap: k-neighbors vs. Noise Level')
    plt.savefig('reports/speedup_heatmap_k_noise.png', dpi=300, bbox_inches='tight')

def generate_summary_tables(df):
    """Generate summary tables aggregating results by different parameters."""
    # Ensure reports directory exists
    os.makedirs('reports', exist_ok=True)
    
    # 1. Summary by sample size
    sample_summary = df.groupby('n_samples').agg({
        'cpu_time': ['mean', 'std'],
        'cuda_time': ['mean', 'std'],
        'speedup': ['mean', 'std', 'min', 'max'],
        'rel_error': ['mean', 'std', 'max']
    }).reset_index()
    
    # 2. Summary by dimension
    dim_summary = df.groupby('dim').agg({
        'cpu_time': ['mean', 'std'],
        'cuda_time': ['mean', 'std'],
        'speedup': ['mean', 'std', 'min', 'max'],
        'rel_error': ['mean', 'std', 'max']
    }).reset_index()
    
    # 3. Summary by k neighbors
    k_summary = df.groupby('k_neighbors').agg({
        'cpu_time': ['mean', 'std'],
        'cuda_time': ['mean', 'std'],
        'speedup': ['mean', 'std', 'min', 'max'],
        'rel_error': ['mean', 'std', 'max']
    }).reset_index()
    
    # 4. Summary by noise level
    noise_summary = df.groupby('noise').agg({
        'cpu_time': ['mean', 'std'],
        'cuda_time': ['mean', 'std'],
        'speedup': ['mean', 'std', 'min', 'max'],
        'rel_error': ['mean', 'std', 'max']
    }).reset_index()

    # 5. Summary by sample size and dimension
    combined_summary = df.groupby(['n_samples', 'dim']).agg({
        'speedup': ['mean', 'min', 'max'],
        'rel_error': ['mean', 'max']
    }).reset_index()
    
    return {
        'by_samples': sample_summary,
        'by_dim': dim_summary,
        'by_k': k_summary,
        'by_noise': noise_summary,
        'by_samples_dim': combined_summary
    }

def print_summary_tables(summary_tables):
    """Print formatted summary tables to console."""
    print("\n=== SUMMARY TABLES ===\n")
    
    # Print sample size summary
    print("Table 1: Performance by Sample Size")
    print("-" * 100)
    samples_table = summary_tables['by_samples'].copy()
    samples_table.columns = [f"{a}_{b}" if b else a for a, b in samples_table.columns]
    print(samples_table.to_string(index=False, float_format="%.3f"))
    
    print("\nTable 2: Performance by Dimension")
    print("-" * 100)
    dim_table = summary_tables['by_dim'].copy()
    dim_table.columns = [f"{a}_{b}" if b else a for a, b in dim_table.columns]
    print(dim_table.to_string(index=False, float_format="%.3f"))
    
    print("\nTable 3: Performance by k-neighbors")
    print("-" * 100)
    k_table = summary_tables['by_k'].copy()
    k_table.columns = [f"{a}_{b}" if b else a for a, b in k_table.columns]
    print(k_table.to_string(index=False, float_format="%.3f"))
    
    print("\nTable 4: Performance by Noise Level")
    print("-" * 100)
    noise_table = summary_tables['by_noise'].copy()
    noise_table.columns = [f"{a}_{b}" if b else a for a, b in noise_table.columns]
    print(noise_table.to_string(index=False, float_format="%.3f"))
    
    print("\nTable 5: Performance by Sample Size and Dimension")
    print("-" * 100)
    combined_table = summary_tables['by_samples_dim'].copy()
    combined_table.columns = [f"{a}_{b}" if b else a for a, b in combined_table.columns]
    print(combined_table.to_string(index=False, float_format="%.3f"))

def main():
    """Main function to run the benchmark suite."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Define parameter ranges
    n_samples_list = [500, 1000, 2000]
    dim_list = [10, 50, 100, 500]
    noise_list = [0.001, 0.01, 0.1]
    k_list = [3, 5]
    
    # Run benchmarks
    print("Running benchmarks...")
    results_df = run_benchmark(n_samples_list, dim_list, noise_list, k_list, num_runs=3)
    
    # Save raw results
    results_df.to_csv('reports/ksg_benchmark_results.csv', index=False)
    print(f"Raw results saved to reports/ksg_benchmark_results.csv")
    
    # Generate summary tables
    print("Generating summary tables...")
    summary_tables = generate_summary_tables(results_df)
    
    for name, table in summary_tables.items():
        table.to_csv(f'reports/summary_{name}.csv', index=False)
        print(f"Summary table saved to reports/summary_{name}.csv")
    
    # Print summary tables
    print_summary_tables(summary_tables)
    
    # Create plots
    print("Creating plots...")
    plot_results(results_df)
    create_heatmaps(results_df)
    print("Plots saved to 'reports' directory")
    
    # Print average and maximum speedups
    avg_speedup = results_df['speedup'].mean()
    max_speedup = results_df['speedup'].max()
    print(f"\nOverall Performance Summary:")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Maximum speedup: {max_speedup:.2f}x")
    
    # Find parameter configuration with maximum speedup
    max_speedup_row = results_df.loc[results_df['speedup'].idxmax()]
    print(f"\nBest performance configuration:")
    print(f"  Samples: {max_speedup_row['n_samples']}")
    print(f"  Dimensions: {max_speedup_row['dim']}")
    print(f"  Noise: {max_speedup_row['noise']}")
    print(f"  k neighbors: {max_speedup_row['k_neighbors']}")
    print(f"  Speedup: {max_speedup_row['speedup']:.2f}x")

if __name__ == "__main__":
    main()