import torch
import math
from typing import Union, Tuple

from .base import MutualInformationEstimator

# Import the CUDA extension
try:
    import mi_cuda
except ImportError:
    raise ImportError(
        "CUDA extension mi_cuda not found. "
        "Please build the extension by running: "
        "cd csrc && python setup.py install"
    )

class KSG(MutualInformationEstimator):
    """
    Kraskov-Stogbauer-Grassberger k-NN based mutual information estimator
    implemented with PyTorch and CUDA.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """

    def __init__(self, k_neighbors: int = 1) -> None:
        """
        Create a Kraskov-Stogbauer-Grassberger k-NN based
        mutual information estimator.

        Parameters
        ----------
        k_neighbors : int, optional
            Number of nearest neighbors to use for estimation.

        References
        ----------
        .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
               information". Phys. Rev. E 69, 2004.
        """

        if k_neighbors < 1:
            raise ValueError("The number of neighbors must be at least 1")

        self.k_neighbors = k_neighbors

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

        # Ensure inputs are float32 (for CUDA compatibility)
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        # Reshape if necessary
        n_samples = x.shape[0]
        k_neighbors = min(self.k_neighbors, n_samples - 1)
        
        x = x.reshape(n_samples, -1)
        y = y.reshape(n_samples, -1)

        # Ensure inputs are on the same device
        device = x.device
        if y.device != device:
            y = y.to(device)

        # Ensure inputs are contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        if not y.is_contiguous():
            y = y.contiguous()

        # Call the CUDA kernel function for KSG computation
        mi, mi_std = mi_cuda.ksg_mi(x, y, k_neighbors)

        if std:
            return mi.item(), mi_std.item()
        else:
            return mi.item()