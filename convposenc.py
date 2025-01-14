import torch
import torch.nn as nn

# in channels equal to out channels
class ConvPosEnc1d(nn.Module):
    def __init__(self, dim: int, k: int = 3, act: bool = False):
        super().__init__()

        self.proj = nn.Conv1d(
            dim,
            dim,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=dim,
        )
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        feat = self.proj(x)
        x = x + self.act(feat)
        return x