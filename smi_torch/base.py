import torch

class MutualInformationEstimator:
    """
    Base class for mutual information estimators.
    """

    def _check_arguments(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Check samples `x` and `y`.

        Parameters
        ----------
        x : torch.Tensor
            Samples from the first random vector.
        y : torch.Tensor
            Samples from the second random vector.
        """

        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("Inputs must be PyTorch tensors")

        if y.shape[0] != x.shape[0]:
            raise ValueError("The number of samples in `x` and `y` must be equal")

    def __call__(self, x: torch.Tensor, y: torch.Tensor, std: bool = False):
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
        mutual_information : float or tuple
            Estimated value of mutual information, or a tuple of (estimate, std_dev)
            if std=True.
        """

        raise NotImplementedError