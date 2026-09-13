import torch
from typing import Union, Tuple

from .base import MutualInformationEstimator

# Import the CUDA extension
try:
    import mi_cuda
except ImportError:
    raise ImportError(
        "CUDA extension mi_cuda not found. "
        "Please build the extension by running: "
        "cd smi_torch/csrc && pip install . --no-build-isolation"
    )

_ALGORITHMS = ("auto", "sweep", "brute")


class KSG(MutualInformationEstimator):
    """
    Kraskov-Stogbauer-Grassberger k-NN based mutual information estimator
    implemented with PyTorch and CUDA.

    The estimate is exact with respect to the reference `mutinfo.knn.KSG`
    implementation: neighbours are counted strictly inside the k-th neighbour
    distance using the Chebyshev norm, without building an N x N distance matrix.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """

    def __init__(self, k_neighbors: int = 1, algorithm: str = "auto") -> None:
        """
        Create a Kraskov-Stogbauer-Grassberger k-NN based
        mutual information estimator.

        Parameters
        ----------
        k_neighbors : int, optional
            Number of nearest neighbors to use for estimation.
        algorithm : {'auto', 'sweep', 'brute'}, optional
            Neighbour search strategy of the CUDA kernels. 'sweep' sorts the samples
            along one coordinate and prunes candidates, 'brute' compares all pairs.
            Both produce identical results; 'auto' selects the faster one for every
            stage based on its dimension ('sweep' in low dimensions).

        References
        ----------
        .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
               information". Phys. Rev. E 69, 2004.
        """

        if k_neighbors < 1:
            raise ValueError("The number of neighbors must be at least 1")

        if algorithm not in _ALGORITHMS:
            raise ValueError(f"The `algorithm` must be one of {_ALGORITHMS}")

        self.k_neighbors = k_neighbors
        self.algorithm = algorithm

    @staticmethod
    def _prepare(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Move `y` to the device of `x`, cast both to a common float32/float64 dtype
        and check that the samples are finite.
        """

        if x.device.type != "cuda":
            raise ValueError("Inputs must be CUDA tensors")

        y = y.to(x.device)

        # float64 inputs keep full precision, everything else is computed in float32.
        dtype = torch.float64 if torch.float64 in (x.dtype, y.dtype) else torch.float32
        x = x.to(dtype).contiguous()
        y = y.to(dtype).contiguous()

        if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
            raise ValueError("Inputs must not contain NaN or infinite values")

        return x, y

    def estimate(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate mutual information for a batch of independent sample sets
        without leaving the GPU.

        Parameters
        ----------
        x : torch.Tensor
            CUDA tensor of shape (batch, n_samples, dim_x).
        y : torch.Tensor
            CUDA tensor of shape (batch, n_samples, dim_y).

        Returns
        -------
        mutual_information : torch.Tensor
            float64 tensor of shape (batch,).
        mutual_information_std : torch.Tensor
            float64 tensor of shape (batch,).
        """

        if x.dim() != 3 or y.dim() != 3 or x.shape[:2] != y.shape[:2]:
            raise ValueError("`x` and `y` must have shapes (batch, n_samples, dim)")

        n_samples = x.shape[1]
        if n_samples < 2:
            raise ValueError("At least two samples are required")

        x, y = self._prepare(x, y)
        k_neighbors = min(self.k_neighbors, n_samples - 1)
        mi, mi_std = mi_cuda.ksg_mi(x, y, k_neighbors, self.algorithm)
        return mi, mi_std

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, std: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """
        Estimate the value of mutual information between two random vectors
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
        x = x.reshape(1, n_samples, -1)
        y = y.reshape(1, n_samples, -1)

        mi, mi_std = self.estimate(x, y)

        if std:
            return mi.item(), mi_std.item()
        else:
            return mi.item()
