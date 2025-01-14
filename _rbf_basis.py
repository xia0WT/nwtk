import torch
from torch import nn
import math
from .config import th_float
class GaussianExpansion(nn.Module):
    def __init__(
        self,
        floor = 0.0,
        ceiling = 4.0,
        num_rbfs = 20,
        delta = 0.5,
        interval = None,
        req_grad = False
    ):
        super().__init__()
        
        delta = torch.tensor(delta, dtype=th_float)
        self.delta = nn.Parameter(delta, requires_grad=req_grad)
        
        if interval:
            self.centers = nn.Parameter(interval, requires_grad=req_grad)
        else: self.centers = nn.Parameter(torch.linspace(floor, ceiling, num_rbfs), requires_grad=req_grad)
        self.shape_par = -self.delta * torch.exp( -torch.diff(self.centers).min() / torch.sqrt(torch.tensor(num_rbfs, dtype=th_float)))

    def reset_parameters(self):
        self.centers = nn.Parameter(self.centers, requires_grad=req_grad)
        self.delta = nn.Parameter(delta, requires_grad=req_grad)

    def forward(self, x):
        diff = x[:, None] - self.centers[None, :]
        return torch.exp(self.shape_par * torch.pow(diff, 2))

class CosineCutoff(nn.Module):
    def __init__(
        self,
        floor = 0.0,
        ceiling = 4.0,
        num_rbfs = 20
    ):
        super().__init__()
        self.floor = floor
        self.ceiling = ceiling
    def forward(self, x):
        x = torch.where(x <= self.ceiling , 0.5 * (torch.cos(math.pi * (x - self.floor) / (self.ceiling - self.floor)) +1), 0)
        return x
        
