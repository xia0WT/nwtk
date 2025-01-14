from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from .config import atom_list
from .config import th_int

from ._activation import ActivationFunc
if TYPE_CHECKING :
    from collections.abc import Sequence

class CleverConv1d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            out_dims,
            kernel_size,
            stride,
            groups=1
    ):
        super().__init__()
        self.out_dims = out_dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups)
        
    def forward(self, x):
        in_dims = x.size()[-1]
        p = (self.out_dims - 1) * self.stride + self.kernel_size - in_dims
        pad_left = p // 2
        pad_right = p - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        x = self.conv(x)
        return x

class CleverMaxPool1d(nn.Module):
    def __init__(self, kernel_size, out_dims):
        super().__init__()
        self.out_dims = out_dims
        self.kernel_size = kernel_size
        self.maxpool = torch.nn.MaxPool1d(kernel_size=kernel_size)

    def forward(self, x):
        in_dims = x.size()[-1]
        p = self.out_dims * self.kernel_size - in_dims
        pad_left = p // 2
        pad_right = p - pad_left
        x = F.pad(x, (pad_left, pad_right), "constant", 0)
        x = self.maxpool(x)
        return x

class MLP(nn.Module):
    def __init__(
        self,
        layer_sequence: Sqeuence[int],
        acf: Callable[[str], None] = 'silu', #activation function
        add_acf: bool = True,
        add_bias: bool = True
    ) -> None:
        
        super().__init__()

        activationfunc = ActivationFunc[acf].value
        self.layers = nn.ModuleList()
        for i ,(in_dims, out_dims) in enumerate(zip(layer_sequence[:-1], layer_sequence[1:])):
            self.layers.append(nn.Linear(in_dims, out_dims, add_bias))
            if add_acf:
                self.layers.append(activationfunc())

    def forward(self, 
                inputs,
               ):

        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x

class Resblock1d(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        out_dims_sequence, 
        kernel_size, 
        stride,
        #layer_sequence: Sqeuence[int],
        acf: Callable[[str], None] = 'relu', #activation function
        add_acf: bool = True,
        add_bn:bool = True,
        add_dropout:bool = True,
        drop_ratio:float = 0.5,
        add_bias: bool = True
    ) -> None:
        
        super().__init__()

        activationfunc = ActivationFunc[acf].value
        self.layers = nn.ModuleList()
        self.in_channels = in_channels
        self.out_channels = out_channels
        for out_dims in out_dims_sequence:
            if not self.layers:
                self.layers.append(CleverConv1d(in_channels, out_channels, out_dims, kernel_size, stride,))
            else: self.layers.append(CleverConv1d(out_channels, out_channels, out_dims, kernel_size, stride,))
            if add_bn:
                self.layers.append(nn.BatchNorm1d(out_channels))
            if add_dropout:
                self.layers.append(nn.Dropout(p = drop_ratio))
            if add_acf:
                self.layers.append(activationfunc())

        self.res = nn.ModuleList()
        for out_dims in out_dims_sequence:
            self.res.append(CleverMaxPool1d(stride, out_dims))

    def forward(self, 
                inputs,
               ):

        x = inputs
        res_ = inputs
        
        for layer in self.layers:
            x = layer(x)

        for res_layer in self.res:
            res_ = res_layer(res_)

        left_pad = (self.out_channels - self.in_channels) // 2
        right_pad = self.out_channels - self.in_channels - left_pad
        
        x = x + F.pad(res_, (0, 0, left_pad, right_pad), "constant", 0)
        return x

class Convblock1d(nn.Module):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        out_dims_sequence, 
        kernel_size, 
        stride,
        #layer_sequence: Sqeuence[int],
        acf: Callable[[str], None] = 'relu', #activation function
        add_acf: bool = True,
        add_bn:bool = True,
        add_dropout:bool = True,
        drop_ratio: float = 0.5,
        add_bias: bool = True
    ) -> None:
        
        super().__init__()

        activationfunc = ActivationFunc[acf].value
        self.layers = nn.ModuleList()
        for out_dims in out_dims_sequence:
            if not self.layers:
                self.layers.append(CleverConv1d(in_channels, out_channels, out_dims, kernel_size, stride,))
            else: self.layers.append(CleverConv1d(out_channels, out_channels, out_dims, kernel_size, stride,))
            if add_bn:
                self.layers.append(nn.BatchNorm1d(out_channels))
            if add_dropout:
                self.layers.append(nn.Dropout(p = drop_ratio))
            if add_acf:
                self.layers.append(activationfunc())

    def forward(self, 
                inputs,
               ):

        x = inputs
        
        for layer in self.layers:
            x = layer(x)
        return x
        