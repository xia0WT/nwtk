import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import Norm1d

class DownSample1d(nn.Module):
    def __init__(
            self, 
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            channel_norm = Norm1d,
    ):
        super().__init__()
        self.norm = Norm1d(in_channels)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.stride = stride
        self.kernel_size = kernel_size
        self.channel_norm = channel_norm
        
    def forward(self, x):
        B, C, N = x.shape
        if self.channel_norm: x = self.norm(x)
        p = (N // self.stride - 1) * self.stride + self.kernel_size - N
        pad_left = p // 2
        pad_right = p - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        x = self.conv(x)
        return x