#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

// Exact GPU implementation of the Kraskov-Stogbauer-Grassberger (KSG, algorithm 1)
// mutual information estimator with the Chebyshev (max) norm.
//
// For every sample i the estimator needs
//   eps_i     : distance to the k-th nearest neighbour in the joint space (x, y),
//   n_x(i)    : number of samples j != i with ||x_i - x_j||_inf < eps_i,
//   n_y(i)    : number of samples j != i with ||y_i - y_j||_inf < eps_i,
// and then  I = psi(k) + psi(N) - < psi(n_x + 1) + psi(n_y + 1) >.
//
// No N x N distance matrix is ever materialised: every GPU thread handles one
// sample and keeps the k smallest distances in a small max-heap, so memory is O(N).
// All inputs may carry a leading batch dimension (B, N, D), which lets sliced MI
// evaluate all random projections in a single pass.
//
// Two exact algorithms are provided:
//   "brute" : compares every sample with every other sample, O(B N^2 D) work.
//   "sweep" : samples are sorted along their widest coordinate and every query
//             scans outwards in sorted order, stopping as soon as the gap along
//             that coordinate reaches the current k-th distance. Since the Chebyshev
//             distance is never smaller than any single coordinate gap, the result
//             is identical to "brute", but only a thin slab of candidates is visited
//             in low dimensions. For one-dimensional marginal spaces the neighbour
//             counts reduce to two binary searches.
//
// Both algorithms use identical floating point expressions for the distances, so
// they return bit-identical radii and counts. "auto" chooses per stage (joint k-NN
// search, x counts, y counts) based on the dimension of that stage.

namespace {

constexpr int64_t kThreadsPerBlock = 256;
// k-NN heaps up to this size live on the GPU thread stack, larger ones use a scratch buffer.
constexpr int64_t kStackHeapCapacity = 32;
// Limits for a single kernel launch; large jobs are split into several launches.
constexpr int64_t kWorkPerLaunch = int64_t(1) << 32;
constexpr int64_t kMaxRowsPerLaunch = int64_t(1) << 22;
constexpr int64_t kScratchBytesPerLaunch = int64_t(1) << 28;
// Largest dimensions for which "auto" uses the sweep algorithm. Sorting along a single
// coordinate prunes poorly in higher dimensions, where "brute" is faster.
constexpr int64_t kSweepMaxJointDim = 12;
constexpr int64_t kSweepMaxMarginalDim = 6;

template <typename T>
__host__ __device__ inline T abs_diff(T a, T b) {
    // Bit-identical to fabs(a - b), because IEEE rounding is symmetric in sign.
    return a > b ? a - b : b - a;
}

constexpr int64_t kChebyshevBlock = 8;

// Chebyshev distance between `a` and `b`. The loop stops once the distance reaches
// `bound`, in which case the returned value is some number >= bound. Coordinates are
// processed in small branch-free blocks, which is much faster on GPUs than testing
// the bound after every coordinate.
template <typename T>
__device__ inline T chebyshev_bounded(const T* a, const T* b, int64_t dim, T bound) {
    T dist = 0;
    int64_t c = 0;
    for (; c + kChebyshevBlock <= dim;) {
        for (const int64_t stop = c + kChebyshevBlock; c < stop; ++c) {
            const T diff = abs_diff(a[c], b[c]);
            dist = diff > dist ? diff : dist;
        }
        if (dist >= bound) {
            return dist;
        }
    }
    for (; c < dim; ++c) {
        const T diff = abs_diff(a[c], b[c]);
        dist = diff > dist ? diff : dist;
    }
    return dist;
}

// Max-heap holding the k smallest values inserted so far.
template <typename T>
struct KSmallest {
    T* heap;
    int64_t k;
    int64_t size;

    __device__ bool full() const { return size == k; }

    // Candidates with a value >= bound() cannot change the k-th smallest value.
    __device__ T bound() const {
        return full() ? heap[0] : std::numeric_limits<T>::infinity();
    }

    // Precondition: !full() || value < heap[0].
    __device__ void insert(T value) {
        if (size < k) {
            int64_t child = size++;
            while (child > 0) {
                const int64_t parent = (child - 1) / 2;
                if (heap[parent] >= value) {
                    break;
                }
                heap[child] = heap[parent];
                child = parent;
            }
            heap[child] = value;
        } else {
            int64_t parent = 0;
            while (true) {
                const int64_t left = 2 * parent + 1;
                if (left >= k) {
                    break;
                }
                const int64_t right = left + 1;
                const int64_t largest = (right < k && heap[right] > heap[left]) ? right : left;
                if (heap[largest] <= value) {
                    break;
                }
                heap[parent] = heap[largest];
                parent = largest;
            }
            heap[parent] = value;
        }
    }

    // k-th smallest value; valid once full().
    __device__ T kth() const { return heap[0]; }
};

__device__ inline int64_t thread_index() {
    return static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
           static_cast<int64_t>(threadIdx.x);
}

// eps[b, i] = k-th nearest neighbour distance of sample i in the joint space (x, y).
template <typename T>
__global__ void knn_radius_brute_kernel(
    const T* x, const T* y,
    const int64_t n, const int64_t dim_x, const int64_t dim_y, const int64_t k,
    const int64_t offset, const int64_t count,
    T* scratch, T* eps) {

    const int64_t t = thread_index();
    if (t >= count) {
        return;
    }
    const int64_t row = offset + t;
    const int64_t b = row / n;
    const int64_t i = row % n;
    const T* xb = x + b * n * dim_x;
    const T* yb = y + b * n * dim_y;
    const T* xi = xb + i * dim_x;
    const T* yi = yb + i * dim_y;

    T local[kStackHeapCapacity];
    KSmallest<T> nearest{scratch != nullptr ? scratch + t * k : local, k, 0};

    for (int64_t j = 0; j < n; ++j) {
        if (j == i) {
            continue;
        }
        const T bound = nearest.bound();
        const T dx = chebyshev_bounded(xi, xb + j * dim_x, dim_x, bound);
        if (nearest.full() && dx >= bound) {
            continue;
        }
        const T dy = chebyshev_bounded(yi, yb + j * dim_y, dim_y, bound);
        if (nearest.full() && dy >= bound) {
            continue;
        }
        nearest.insert(dx > dy ? dx : dy);
    }
    eps[row] = nearest.kth();
}

// counts[b, i] = #{ j != i : ||data[b, i] - data[b, j]||_inf < eps[b, i] }.
template <typename T>
__global__ void count_brute_kernel(
    const T* data, const T* eps,
    const int64_t n, const int64_t dim,
    const int64_t offset, const int64_t count,
    int32_t* counts) {

    const int64_t t = thread_index();
    if (t >= count) {
        return;
    }
    const int64_t row = offset + t;
    const int64_t b = row / n;
    const int64_t i = row % n;
    const T* db = data + b * n * dim;
    const T* di = db + i * dim;
    const T radius = eps[row];

    int64_t c = 0;
    for (int64_t j = 0; j < n; ++j) {
        if (j != i && chebyshev_bounded(di, db + j * dim, dim, radius) < radius) {
            ++c;
        }
    }
    counts[row] = static_cast<int32_t>(c);
}

// Same as knn_radius_brute_kernel, but `z` is the joint sample (B, N, D) whose rows
// are sorted by the first coordinate within every batch entry.
template <typename T>
__global__ void knn_radius_sweep_kernel(
    const T* z,
    const int64_t n, const int64_t dim, const int64_t k,
    const int64_t offset, const int64_t count,
    T* scratch, T* eps) {

    const int64_t t = thread_index();
    if (t >= count) {
        return;
    }
    const int64_t row = offset + t;
    const int64_t b = row / n;
    const int64_t p = row % n;
    const T* zb = z + b * n * dim;
    const T* zp = zb + p * dim;
    const T key = zp[0];

    T local[kStackHeapCapacity];
    KSmallest<T> nearest{scratch != nullptr ? scratch + t * k : local, k, 0};

    // Visit candidates in order of increasing gap along the sorted coordinate.
    int64_t lo = p - 1;
    int64_t hi = p + 1;
    while (lo >= 0 || hi < n) {
        int64_t q;
        T gap;
        if (hi >= n) {
            q = lo--;
            gap = abs_diff(key, zb[q * dim]);
        } else if (lo < 0) {
            q = hi++;
            gap = abs_diff(key, zb[q * dim]);
        } else {
            const T gap_lo = abs_diff(key, zb[lo * dim]);
            const T gap_hi = abs_diff(key, zb[hi * dim]);
            if (gap_lo <= gap_hi) {
                q = lo--;
                gap = gap_lo;
            } else {
                q = hi++;
                gap = gap_hi;
            }
        }
        // Gaps never decrease along either direction, and the other direction's next
        // gap is at least `gap`: no remaining candidate can be strictly closer.
        if (nearest.full() && gap >= nearest.kth()) {
            break;
        }
        const T bound = nearest.bound();
        const T d = chebyshev_bounded(zp, zb + q * dim, dim, bound);
        if (nearest.full() && d >= bound) {
            continue;
        }
        nearest.insert(d);
    }
    eps[row] = nearest.kth();
}

// Same as count_brute_kernel, but rows of `data` (and `eps`) are sorted by the first
// coordinate within every batch entry.
template <typename T>
__global__ void count_sweep_kernel(
    const T* data, const T* eps,
    const int64_t n, const int64_t dim,
    const int64_t offset, const int64_t count,
    int32_t* counts) {

    const int64_t t = thread_index();
    if (t >= count) {
        return;
    }
    const int64_t row = offset + t;
    const int64_t b = row / n;
    const int64_t p = row % n;
    const T* db = data + b * n * dim;
    const T* dp = db + p * dim;
    const T key = dp[0];
    const T radius = eps[row];

    int64_t c = 0;
    if (dim == 1) {
        // abs_diff(key, db[q]) < radius is monotone on either side of p, because
        // rounded subtraction is monotone: count both runs with binary searches.
        int64_t lo = 0;
        int64_t hi = p;
        while (lo < hi) {  // first q in [0, p) inside the radius
            const int64_t mid = lo + (hi - lo) / 2;
            if (abs_diff(key, db[mid]) < radius) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        c += p - lo;
        lo = p + 1;
        hi = n;
        while (lo < hi) {  // first q in (p, n) outside the radius
            const int64_t mid = lo + (hi - lo) / 2;
            if (abs_diff(key, db[mid]) < radius) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        c += lo - (p + 1);
    } else {
        for (int64_t q = p - 1; q >= 0; --q) {
            const T* dq = db + q * dim;
            if (abs_diff(key, dq[0]) >= radius) {
                break;
            }
            if (chebyshev_bounded(dp, dq, dim, radius) < radius) {
                ++c;
            }
        }
        for (int64_t q = p + 1; q < n; ++q) {
            const T* dq = db + q * dim;
            if (abs_diff(key, dq[0]) >= radius) {
                break;
            }
            if (chebyshev_bounded(dp, dq, dim, radius) < radius) {
                ++c;
            }
        }
    }
    counts[row] = static_cast<int32_t>(c);
}

// Splits `total_rows` kernel threads into launches of bounded size and calls
// launch(num_blocks, offset, count) for each of them.
template <typename Launch>
void launch_rows(int64_t total_rows, int64_t rows_per_launch, Launch&& launch) {
    for (int64_t offset = 0; offset < total_rows; offset += rows_per_launch) {
        const int64_t count = std::min(rows_per_launch, total_rows - offset);
        const int64_t num_blocks = (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
        launch(num_blocks, offset, count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

int64_t rows_per_launch(int64_t total_rows, int64_t work_per_row) {
    const int64_t rows = kWorkPerLaunch / std::max<int64_t>(1, work_per_row);
    return std::max<int64_t>(1, std::min({rows, kMaxRowsPerLaunch, total_rows}));
}

// Scratch space for k-NN heaps that do not fit on the GPU thread stack.
struct HeapScratch {
    at::Tensor buffer;
    int64_t rows_per_launch;
};

HeapScratch make_heap_scratch(const at::Tensor& like, int64_t k, int64_t rows) {
    if (k <= kStackHeapCapacity) {
        return {at::Tensor(), rows};
    }
    const int64_t max_rows = std::max<int64_t>(1, kScratchBytesPerLaunch / (k * like.element_size()));
    rows = std::min(rows, max_rows);
    return {at::empty({rows * k}, like.options()), rows};
}

template <typename T>
T* scratch_ptr(const HeapScratch& scratch) {
    return scratch.buffer.defined() ? scratch.buffer.data_ptr<T>() : nullptr;
}

template <typename T>
at::Tensor knn_radius_brute(const at::Tensor& x, const at::Tensor& y, int64_t k) {
    const int64_t batch = x.size(0), n = x.size(1), dim_x = x.size(2), dim_y = y.size(2);
    auto eps = at::empty({batch, n}, x.options());
    const int64_t total = batch * n;
    const auto scratch = make_heap_scratch(x, k, rows_per_launch(total, n * (dim_x + dim_y)));
    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_rows(total, scratch.rows_per_launch, [&](int64_t blocks, int64_t offset, int64_t count) {
        knn_radius_brute_kernel<T> <<<blocks, kThreadsPerBlock, 0, stream.stream()>>>(
            x.data_ptr<T>(), y.data_ptr<T>(), n, dim_x, dim_y, k, offset, count,
            scratch_ptr<T>(scratch), eps.data_ptr<T>());
    });
    return eps;
}

template <typename T>
at::Tensor count_brute(const at::Tensor& data, const at::Tensor& eps) {
    const int64_t batch = data.size(0), n = data.size(1), dim = data.size(2);
    auto counts = at::empty({batch, n}, data.options().dtype(at::kInt));
    const int64_t total = batch * n;
    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_rows(total, rows_per_launch(total, n * dim), [&](int64_t blocks, int64_t offset, int64_t count) {
        count_brute_kernel<T> <<<blocks, kThreadsPerBlock, 0, stream.stream()>>>(
            data.data_ptr<T>(), eps.data_ptr<T>(), n, dim, offset, count, counts.data_ptr<int32_t>());
    });
    return counts;
}

// Moves the coordinate with the largest spread to the front (per batch entry) and
// sorts the rows by it. Returns the sorted data and the sorting permutation.
std::pair<at::Tensor, at::Tensor> sort_by_widest_coordinate(const at::Tensor& data) {
    const int64_t batch = data.size(0), n = data.size(1), dim = data.size(2);
    at::Tensor reordered = data;
    if (dim > 1) {
        const auto key = data.std(1).argmax(1);  // (B,)
        auto columns = at::arange(dim, data.options().dtype(at::kLong)).repeat({batch, 1});
        columns.scatter_(1, key.unsqueeze(1), 0);  // swap the key column with column 0
        columns.select(1, 0).copy_(key);
        reordered = data.gather(2, columns.unsqueeze(1).expand({batch, n, dim}));
    }
    const auto order = std::get<1>(reordered.select(2, 0).sort(/*dim=*/1));  // (B, N)
    auto sorted = reordered.gather(1, order.unsqueeze(2).expand({batch, n, dim})).contiguous();
    return {sorted, order};
}

template <typename T>
at::Tensor knn_radius_sweep(const at::Tensor& x, const at::Tensor& y, int64_t k) {
    const auto sorted = sort_by_widest_coordinate(at::cat({x, y}, 2));
    const auto& z = sorted.first;
    const int64_t batch = z.size(0), n = z.size(1), dim = z.size(2);
    auto eps_sorted = at::empty({batch, n}, z.options());
    const int64_t total = batch * n;
    const auto scratch = make_heap_scratch(z, k, rows_per_launch(total, n * dim));
    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_rows(total, scratch.rows_per_launch, [&](int64_t blocks, int64_t offset, int64_t count) {
        knn_radius_sweep_kernel<T> <<<blocks, kThreadsPerBlock, 0, stream.stream()>>>(
            z.data_ptr<T>(), n, dim, k, offset, count,
            scratch_ptr<T>(scratch), eps_sorted.data_ptr<T>());
    });
    return at::empty_like(eps_sorted).scatter_(1, sorted.second, eps_sorted);
}

template <typename T>
at::Tensor count_sweep(const at::Tensor& data, const at::Tensor& eps) {
    const auto sorted = sort_by_widest_coordinate(data);
    const auto& d = sorted.first;
    const auto eps_sorted = eps.gather(1, sorted.second).contiguous();
    const int64_t batch = d.size(0), n = d.size(1), dim = d.size(2);
    auto counts_sorted = at::empty({batch, n}, d.options().dtype(at::kInt));
    const int64_t total = batch * n;
    const auto stream = at::cuda::getCurrentCUDAStream();
    launch_rows(total, rows_per_launch(total, n * dim), [&](int64_t blocks, int64_t offset, int64_t count) {
        count_sweep_kernel<T> <<<blocks, kThreadsPerBlock, 0, stream.stream()>>>(
            d.data_ptr<T>(), eps_sorted.data_ptr<T>(), n, dim, offset, count,
            counts_sorted.data_ptr<int32_t>());
    });
    return at::empty_like(counts_sorted).scatter_(1, sorted.second, counts_sorted);
}

template <typename T>
std::vector<at::Tensor> ksg_statistics_impl(
    const at::Tensor& x, const at::Tensor& y, int64_t k, const std::string& algorithm) {
    // "auto" picks the faster algorithm for every stage based on its dimension; the
    // thresholds were measured on an H200 and only affect speed, never the result.
    const auto use_sweep = [&](int64_t dim, int64_t max_sweep_dim) {
        return algorithm == "sweep" || (algorithm == "auto" && dim <= max_sweep_dim);
    };
    const int64_t dim_x = x.size(2), dim_y = y.size(2);
    auto eps = use_sweep(dim_x + dim_y, kSweepMaxJointDim) ? knn_radius_sweep<T>(x, y, k)
                                                           : knn_radius_brute<T>(x, y, k);
    auto counts_x = use_sweep(dim_x, kSweepMaxMarginalDim) ? count_sweep<T>(x, eps) : count_brute<T>(x, eps);
    auto counts_y = use_sweep(dim_y, kSweepMaxMarginalDim) ? count_sweep<T>(y, eps) : count_brute<T>(y, eps);
    return {eps, counts_x, counts_y};
}

}  // namespace

// Returns {eps, counts_x, counts_y}, each of shape (N) for 2-D inputs (N, D) or
// (B, N) for batched 3-D inputs (B, N, D).
std::vector<at::Tensor> ksg_statistics_cuda(
    at::Tensor x,
    at::Tensor y,
    int64_t k_neighbors,
    const std::string& algorithm) {

    TORCH_CHECK(x.is_cuda() && y.is_cuda(), "ksg: `x` and `y` must be CUDA tensors");
    TORCH_CHECK(x.device() == y.device(), "ksg: `x` and `y` must be on the same device");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "ksg: `x` and `y` must have the same dtype");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
                "ksg: only float32 and float64 tensors are supported");
    TORCH_CHECK(x.dim() == y.dim() && (x.dim() == 2 || x.dim() == 3),
                "ksg: `x` and `y` must both have shape (N, D) or (B, N, D)");
    TORCH_CHECK(algorithm == "auto" || algorithm == "sweep" || algorithm == "brute",
                "ksg: `algorithm` must be one of 'auto', 'sweep', 'brute'");

    const bool batched = x.dim() == 3;
    if (!batched) {
        x = x.unsqueeze(0);
        y = y.unsqueeze(0);
    }
    TORCH_CHECK(x.size(0) == y.size(0), "ksg: batch sizes of `x` and `y` must be equal");
    TORCH_CHECK(x.size(1) == y.size(1), "ksg: the number of samples in `x` and `y` must be equal");
    const int64_t n = x.size(1);
    TORCH_CHECK(n >= 2, "ksg: at least two samples are required");
    TORCH_CHECK(n <= std::numeric_limits<int32_t>::max(), "ksg: too many samples");
    TORCH_CHECK(k_neighbors >= 1 && k_neighbors < n,
                "ksg: `k_neighbors` must satisfy 1 <= k_neighbors < number of samples");
    TORCH_CHECK(x.size(2) >= 1 && y.size(2) >= 1, "ksg: `x` and `y` must have at least one feature");

    const c10::cuda::CUDAGuard device_guard(x.device());
    x = x.contiguous();
    y = y.contiguous();

    auto result = x.scalar_type() == at::kDouble
        ? ksg_statistics_impl<double>(x, y, k_neighbors, algorithm)
        : ksg_statistics_impl<float>(x, y, k_neighbors, algorithm);
    if (!batched) {
        for (auto& tensor : result) {
            tensor = tensor.squeeze(0);
        }
    }
    return result;
}

// Returns {mi, mi_std} as float64 tensors: scalars for 2-D inputs, shape (B) for
// batched 3-D inputs. Matches the reference `mutinfo.knn.KSG` estimator.
std::vector<at::Tensor> ksg_mi_cuda(
    at::Tensor x,
    at::Tensor y,
    int64_t k_neighbors,
    const std::string& algorithm) {

    const auto stats = ksg_statistics_cuda(x, y, k_neighbors, algorithm);
    const int64_t n = x.size(x.dim() - 2);
    const auto as_double = stats[1].options().dtype(at::kDouble);

    // psi(n_x + 1) + psi(n_y + 1): the +1 accounts for the sample itself.
    const auto psi = at::digamma(stats[1].to(as_double).add_(1)) + at::digamma(stats[2].to(as_double).add_(1));
    const auto psi_mean = psi.mean(-1);
    const auto constant = at::digamma(
        at::tensor(std::vector<double>{static_cast<double>(k_neighbors), static_cast<double>(n)},
                   at::TensorOptions().dtype(at::kDouble)));
    const double offset = constant[0].item<double>() + constant[1].item<double>();

    const auto mi = at::clamp_min(psi_mean.neg().add_(offset), 0.0);
    const auto mi_std = (psi - psi_mean.unsqueeze(-1)).square_().mean(-1).sqrt_().div_(std::sqrt(static_cast<double>(n)));
    return {mi, mi_std};
}
