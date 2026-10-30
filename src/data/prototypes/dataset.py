import torch
from torch import Tensor
from torch.utils.data import TensorDataset as _TensorDataset_


class TensorDataset(_TensorDataset_):
    def __init__(self, *tensors: Tensor, dim: int = 0) -> None:
        assert all(
            tensors[0].size(dim) == tensor.size(dim) for tensor in tensors
        ), "Size mismatch between tensors"
        self.dim = dim
        self.tensors = tensors

    def __getitem__(self, index):
        if len(self.tensors) == 1:
            return torch.select(self.tensors[0], dim=self.dim, index=index)
        else:
            return tuple(
                torch.select(tensor, dim=self.dim, index=index)
                for tensor in self.tensors
            )

    def __len__(self):
        return self.tensors[0].size(self.dim)
