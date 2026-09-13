import torch
import math
from typing import Optional, Union, Tuple

from .base import MutualInformationEstimator
from .knn import KSG

# Rough upper bound on the GPU memory used by one batch of projections.
_MAX_BATCH_BYTES = 2 * 1024 ** 3


class SMI(MutualInformationEstimator):
    """
    k-Sliced mutual information estimator implemented with PyTorch and CUDA.

    With a `KSG` base estimator all projections are evaluated by the CUDA kernels
    in large batches, without a host synchronisation per projection. Any other
    base estimator is called once per projection.

    References
    ----------
    .. [1] Z. Goldfeld, K. Greenewald and T. Nuradha, "k-Sliced
           Mutual Information: A Quantitative Study of Scalability
           with Dimension". NeurIPS, 2022.
    """

    def __init__(self, estimator: MutualInformationEstimator,
                 projection_dim: int=1,
                 n_projection_samples: int=128,
                 max_batch_size: Optional[int]=None) -> None:
        """
        Create a k-Sliced Mutual Information estimator

        Parameters
        ----------
        estimator : MutualInformationEstimator
            Base estimator used to estimate MI between projections.
        projection_dim : int, optional
            Dimensionality of the projection subspace.
        n_projection_samples : int, optional
            Number of Monte Carlo samples to estimate SMI.
        max_batch_size : int, optional
            Maximum number of projections evaluated by one batched KSG call.
            Chosen from the available GPU memory by default. Does not affect
            the result beyond floating point round-off.

        References
        ----------
        .. [1] Z. Goldfeld, K. Greenewald and T. Nuradha, "k-Sliced
               Mutual Information: A Quantitative Study of Scalability
               with Dimension". NeurIPS, 2022.
        """

        if projection_dim < 1:
            raise ValueError("The projection dimension must be at least 1")

        if n_projection_samples < 1:
            raise ValueError("The number of projection samples must be at least 1")

        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("The maximum batch size must be at least 1")

        self.estimator = estimator
        self.projection_dim = projection_dim
        self.n_projection_samples = n_projection_samples
        self.max_batch_size = max_batch_size

    def generate_random_projection_matrix(self, dim: int, device, dtype=torch.float32) -> torch.Tensor:
        """
        Sample a random projection matrix from the uniform distribution
        of orthogonal linear projectors from `dim` to `self.projection_dim`

        Parameters
        ----------
        dim : int
            Dimension of the data to be projected
        device : torch.device
            Device to create the matrix on
        dtype : torch.dtype, optional
            Floating point type of the matrix

        Returns
        -------
        Q : torch.Tensor
            Orthogonal projection matrix of shape (dim, projection_dim)
        """
        return self.generate_random_projection_matrices(1, dim, device, dtype)[0]

    def generate_random_projection_matrices(self, n_matrices: int, dim: int, device,
                                            dtype=torch.float32) -> torch.Tensor:
        """
        Sample `n_matrices` independent random projection matrices, see
        `generate_random_projection_matrix`.

        Returns
        -------
        Q : torch.Tensor
            Orthogonal projection matrices of shape (n_matrices, dim, projection_dim)
        """
        random_matrix = torch.randn(n_matrices, dim, self.projection_dim, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(random_matrix)
        return Q

    @staticmethod
    def _project(data: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Project `data` of shape (n_samples, dim) with every matrix of `Q` of shape
        (batch, dim, projection_dim). Returns a tensor of shape
        (batch, n_samples, projection_dim).
        """
        batch, dim, projection_dim = Q.shape
        # A single matrix product; avoids materialising `data` once per projection.
        Q_flat = Q.permute(1, 0, 2).reshape(dim, batch * projection_dim)
        projected = data @ Q_flat
        return projected.reshape(data.shape[0], batch, projection_dim).transpose(0, 1).contiguous()

    def _batch_size(self, x: torch.Tensor, projection_dim_x: int, projection_dim_y: int) -> int:
        if self.max_batch_size is not None:
            return self.max_batch_size

        n_samples = x.shape[0]
        # Projections, joint samples, sorted copies, permutations, radii and counts.
        bytes_per_projection = n_samples * (
            8 * (projection_dim_x + projection_dim_y) * x.element_size() + 64
        )
        free_bytes, _ = torch.cuda.mem_get_info(x.device)
        budget = min(_MAX_BATCH_BYTES, free_bytes // 4)
        return max(1, budget // bytes_per_projection)

    def estimate_projections(self, x: torch.Tensor, y: torch.Tensor,
                             Q_x: torch.Tensor, Q_y: torch.Tensor) -> torch.Tensor:
        """
        Estimate MI between the projections `x @ Q_x[i]` and `y @ Q_y[i]` for
        every projection pair `i`.

        Parameters
        ----------
        x, y : torch.Tensor
            Samples of shape (n_samples, dim_x) and (n_samples, dim_y).
        Q_x, Q_y : torch.Tensor
            Projection matrices of shape (n_projections, dim_x, proj_dim_x) and
            (n_projections, dim_y, proj_dim_y).

        Returns
        -------
        mutual_information : torch.Tensor
            float64 tensor of shape (n_projections,).
        """

        n_projections = Q_x.shape[0]

        if not isinstance(self.estimator, KSG):
            values = [float(self.estimator(x @ Q_x[i], y @ Q_y[i])) for i in range(n_projections)]
            return torch.tensor(values, dtype=torch.float64, device=x.device)

        results = torch.empty(n_projections, dtype=torch.float64, device=x.device)
        batch_size = self._batch_size(x, Q_x.shape[2], Q_y.shape[2])
        for start in range(0, n_projections, batch_size):
            stop = min(start + batch_size, n_projections)
            mi, _ = self.estimator.estimate(
                self._project(x, Q_x[start:stop]), self._project(y, Q_y[start:stop])
            )
            results[start:stop] = mi
        return results

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, std: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """
        Estimate the value of k-sliced mutual information between two random vectors
        using samples `x` and `y`.

        Parameters
        ----------
        x, y : torch.Tensor
            Samples from corresponding random vectors.
        std : bool
            Calculate standard deviation.

        Returns
        -------
        mutual_information : float
            Estimated value of mutual information.
        mutual_information_std : float or None
            Standard deviation of the estimate, or None if `std=False`
        """

        self._check_arguments(x, y)

        n_samples = x.shape[0]
        x = x.reshape(n_samples, -1)
        y = y.reshape(n_samples, -1)

        if isinstance(self.estimator, KSG):
            x, y = KSG._prepare(x, y)
        else:
            y = y.to(x.device)
            if not x.is_floating_point():
                x = x.to(torch.float32)
            y = y.to(x.dtype)

        # All matrices are drawn up front, so the result for a given seed does not
        # depend on how the projections are batched.
        Q_x = self.generate_random_projection_matrices(
            self.n_projection_samples, x.shape[1], x.device, x.dtype)
        Q_y = self.generate_random_projection_matrices(
            self.n_projection_samples, y.shape[1], y.device, y.dtype)

        results = self.estimate_projections(x, y, Q_x, Q_y)
        mi = results.mean()

        if std:
            mi_std = results.std() / math.sqrt(self.n_projection_samples)
            return mi.item(), mi_std.item()
        else:
            return mi.item()
