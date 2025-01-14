from __future__ import annotations

from torch import nn

from enum import Enum

class SoftPlus2(nn.Module):
    """SoftPlus2 activation function:
    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow.
    """

    def __init__(self) -> None:
        """Initializes the SoftPlus2 class."""
        super().__init__()
        self.ssp = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate activation function given the input tensor x.

        Args:
            x (torch.tensor): Input tensor

        Returns:
            out (torch.tensor): Output tensor
        """
        return self.ssp(x) - math.log(2.0)
        
class ActivationFunc(Enum):
    
    relu = nn.ReLU
    selu = nn.SELU
    tanh = nn.Tanh
    silu = nn.SiLU
    softplus = nn.Softplus
    softplus2 = SoftPlus2
    sigmoid = nn.Sigmoid