import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_block_fast_1d(
        x: torch.Tensor,
        drop_prob: float = 0.5,
        block_size: int = 4,
        gamma_scale: float = 1.0,
        normalize: bool = True
):
    B, C, N = x.shape

    clipped_block_size = min(block_size, N)
    gamma = gamma_scale * drop_prob * N / clipped_block_size / (N - block_size + 1)

    block_mask = F.pad(torch.empty_like(x),
                       (clipped_block_size // 2, clipped_block_size - clipped_block_size // 2 - 1),
                       "constant",
                       0).bernoulli_(gamma)
    
    block_mask = F.max_pool1d(block_mask.to(x.dtype),
                              kernel_size=clipped_block_size , stride=1)

    if normalize:
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-6)).to(dtype=x.dtype)
        x.mul_(block_mask).mul_(normalize_scale)
    else:
        x.mul_(block_mask)
    return x

def drop_path(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True
):

    
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

def drop_channel(x: torch.Tensor,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True
):
    
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[:2],) + (1,) * (x.ndim - 2)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class DropChannel(nn.Module):
    
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropChannel, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_channel(x, self.drop_prob, self.training, self.scale_by_keep)