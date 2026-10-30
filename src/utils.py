import numpy as np
import torch


def is_symmetric(
    tensor: np.ndarray | torch.Tensor, dims: tuple[int, int] = (-2, -1)
) -> bool:
    """
    Check if a tensor is symmetric along the specified dimensions.

    Args:
        tensor (np.ndarray | torch.Tensor): The tensor to check.
        dims (tuple[int, int]): The dimensions to check for symmetry.

    Returns:
        bool: True if the tensor is symmetric along the specified dimensions, False otherwise.
    """
    if isinstance(tensor, np.ndarray):
        return np.all(tensor == tensor.transpose(dims[0], dims[1])).item()
    elif isinstance(tensor, torch.Tensor):
        return torch.all(tensor == tensor.transpose(dims[0], dims[1])).item()
    else:
        raise TypeError("Input must be a numpy array or a PyTorch tensor.")
