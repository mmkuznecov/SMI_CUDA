#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA kernel for computing Chebyshev distances between all pairs of points
__global__ void chebyshev_distance_kernel(
    const float* x_y,
    const int n_samples,
    const int dim_xy,
    float* distances) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < n_samples && j < n_samples && i != j) {
        float max_dist = 0.0f;
        for (int d = 0; d < dim_xy; d++) {
            float diff = fabsf(x_y[i * dim_xy + d] - x_y[j * dim_xy + d]);
            if (diff > max_dist) {
                max_dist = diff;
            }
        }
        distances[i * n_samples + j] = max_dist;
    } else if (i < n_samples && j < n_samples && i == j) {
        distances[i * n_samples + j] = INFINITY; // Set self-distance to infinity
    }
}

// CUDA kernel for finding k-th nearest neighbor distances
__global__ void find_knn_distances_kernel(
    const float* distances,
    const int n_samples,
    const int k,
    float* knn_distances) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_samples) {
        // Create temporary array for sorting
        float temp_distances[2048]; // Assuming max 2048 samples
        int count = 0;
        
        // Copy distances to temp array, excluding self-distance
        for (int j = 0; j < n_samples; j++) {
            if (i != j) {
                temp_distances[count++] = distances[i * n_samples + j];
            }
        }
        
        // Sort the distances (simple insertion sort)
        for (int j = 1; j < count; j++) {
            float key = temp_distances[j];
            int l = j - 1;
            while (l >= 0 && temp_distances[l] > key) {
                temp_distances[l + 1] = temp_distances[l];
                l = l - 1;
            }
            temp_distances[l + 1] = key;
        }
        
        // Store the k-th nearest neighbor distance
        knn_distances[i] = temp_distances[k - 1];
    }
}

// CUDA kernel for counting points within radius in x and y spaces
__global__ void count_within_radius_kernel(
    const float* data,
    const int n_samples,
    const int dim,
    const float* radii,
    int* counts) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_samples) {
        float radius = radii[i];
        int count = 0;
        
        for (int j = 0; j < n_samples; j++) {
            if (i != j) {
                float max_dist = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float diff = fabsf(data[i * dim + d] - data[j * dim + d]);
                    if (diff > max_dist) {
                        max_dist = diff;
                    }
                }
                
                if (max_dist <= radius) {
                    count++;
                }
            }
        }
        
        counts[i] = count;
    }
}

// CUDA kernel for computing digamma values and MI estimates
__global__ void compute_mi_kernel(
    const int* counts_x,
    const int* counts_y,
    const int n_samples,
    const int k,
    float* mi_samples) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_samples) {
        // Precomputed digamma values (for common small integers)
        float digamma_values[16] = {
            -0.5772156649f,  // digamma(1)
            0.4227843351f,   // digamma(2)
            0.9227843351f,   // digamma(3)
            1.2561176684f,   // digamma(4)
            1.5061176684f,   // digamma(5)
            1.7061176684f,   // digamma(6)
            1.8727843351f,   // digamma(7)
            2.0156414779f,   // digamma(8)
            2.1406414779f,   // digamma(9)
            2.2517525890f,   // digamma(10)
            2.3517525890f,   // digamma(11)
            2.4426616799f,   // digamma(12)
            2.5259950132f,   // digamma(13)
            2.6026565464f,   // digamma(14)
            2.6733176097f,   // digamma(15)
            2.7385383806f    // digamma(16)
        };
        
        // Approximate digamma for values not in the table
        auto digamma_approx = [&digamma_values](int x) -> float {
            if (x <= 0) {
                return 0.0f;  // Error case, shouldn't happen
            } else if (x <= 16) {
                return digamma_values[x - 1];
            } else {
                // Asymptotic approximation for large x
                return logf(x) - 0.5f/x;
            }
        };
        
        float digamma_cx = digamma_approx(counts_x[i]);
        float digamma_cy = digamma_approx(counts_y[i]);
        float digamma_k_val = digamma_approx(k);
        float digamma_n_val = digamma_approx(n_samples);
        
        mi_samples[i] = digamma_k_val + digamma_n_val - digamma_cx - digamma_cy;
    }
}

// Main function to estimate MI using KSG method
std::vector<at::Tensor> ksg_mi_cuda(
    at::Tensor x,
    at::Tensor y,
    int k_neighbors) {
    
    // Get dimensions
    const auto n_samples = x.size(0);
    const auto dim_x = x.size(1);
    const auto dim_y = y.size(1);
    const auto dim_xy = dim_x + dim_y;
    
    // Ensure inputs are contiguous
    x = x.contiguous();
    y = y.contiguous();
    
    // Concatenate x and y for joint space
    auto x_y = at::cat({x, y}, 1);
    
    // Set current device
    at::cuda::CUDAGuard device_guard(x.device());
    
    // Create tensor options
    auto options = at::TensorOptions().device(x.device()).dtype(at::kFloat);
    auto int_options = at::TensorOptions().device(x.device()).dtype(at::kInt);
    
    // Allocate memory for distances
    auto distances = at::empty({n_samples, n_samples}, options);
    
    // Compute Chebyshev distances
    const dim3 block_size(16, 16);
    const dim3 grid_size((n_samples + block_size.x - 1) / block_size.x,
                        (n_samples + block_size.y - 1) / block_size.y);
    
    chebyshev_distance_kernel<<<grid_size, block_size>>>(
        x_y.data_ptr<float>(),
        n_samples,
        dim_xy,
        distances.data_ptr<float>());
    
    // Find k-th nearest neighbor distances
    auto knn_distances = at::empty({n_samples}, options);
    
    const int threads_per_block = 256;
    const int num_blocks = (n_samples + threads_per_block - 1) / threads_per_block;
    
    find_knn_distances_kernel<<<num_blocks, threads_per_block>>>(
        distances.data_ptr<float>(),
        n_samples,
        k_neighbors,
        knn_distances.data_ptr<float>());
    
    // Count points within radius in x space
    auto counts_x = at::empty({n_samples}, int_options);
    
    count_within_radius_kernel<<<num_blocks, threads_per_block>>>(
        x.data_ptr<float>(),
        n_samples,
        dim_x,
        knn_distances.data_ptr<float>(),
        counts_x.data_ptr<int>());
    
    // Count points within radius in y space
    auto counts_y = at::empty({n_samples}, int_options);
    
    count_within_radius_kernel<<<num_blocks, threads_per_block>>>(
        y.data_ptr<float>(),
        n_samples,
        dim_y,
        knn_distances.data_ptr<float>(),
        counts_y.data_ptr<int>());
    
    // Compute MI samples
    auto mi_samples = at::empty({n_samples}, options);
    
    compute_mi_kernel<<<num_blocks, threads_per_block>>>(
        counts_x.data_ptr<int>(),
        counts_y.data_ptr<int>(),
        n_samples,
        k_neighbors,
        mi_samples.data_ptr<float>());
    
    // Compute mean and standard deviation
    auto mi = at::clamp(at::mean(mi_samples), 0.0);
    auto mi_std = at::std(mi_samples) / sqrt(static_cast<float>(n_samples));
    
    return {mi, mi_std};
}
