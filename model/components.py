# Based on:
# https://github.com/rosinality/swapping-autoencoder-pytorch

# Definitions for basic components:
#
#   - EqualConvTranspose2d: Convolution with Equalized Learning Rate
#   - ConvLayer: Composite Convolution Layer Block
#   - ResBlock: Basic ResBlock
#   - StyledResBlock: ResBlock with Style Condition

import math

import torch
from torch import nn
from torch.nn import functional as F

from .stylegan2.model import StyledConv, Blur, EqualLinear, EqualConv2d, ScaledLeakyReLU
from .stylegan2.op import *

class EqualConvTranspose2d(nn.Module):
    # Basic wrapper for 2d convolution (constant size)
    # that enforces equalized learning rate (see self.scale)
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True
                ):
        super().__init__()

        # basic settings        
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        
        # learnable parameters
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )
    

class ConvLayer(nn.Sequential):
    # ConvLayer combines potential blurring and convolution (upsample / downsample / constant)
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=(1, 3, 3, 1),
        bias=True,
        activate=True,
        padding="zero",
    ):
        layers = []

        self.padding = 0
        stride = 1
        
        # blur before downsampling
        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
        
        # blur after upsampling
        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
                
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            # add padding if not downsampling (constant size)
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channels))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)

class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample,
        padding="zero",
        blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        self.conv1 = ConvLayer(in_channels, out_channels, 3, padding=padding)

        self.conv2 = ConvLayer(
            out_channels,
            out_channels,
            3,
            downsample=downsample,
            padding=padding,
            blur_kernel=blur_kernel,
        )

        if downsample or in_channels != out_channels:
            self.skip = ConvLayer(
                in_channels,
                out_channels,
                1,
                downsample=downsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False,
            )

        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input
            
        return (out + skip) / math.sqrt(2)
    
class StyledResBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, style_dim, upsample, blur_kernel=(1, 3, 3, 1)
    ):
        super().__init__()

        self.conv1 = StyledConv(
            in_channels,
            out_channels,
            3,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
        )

        self.conv2 = StyledConv(out_channels, out_channels, 3, style_dim)

        if upsample or in_channels != out_channels:
        # Skip Connection
            self.skip = ConvLayer(
                in_channels,
                out_channels,
                1,
                upsample=upsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False,
            )

        else:
            self.skip = None

    def forward(self, input, style, noise=None):
        out = self.conv1(input, style, noise)
        out = self.conv2(out, style, noise)

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        return (out + skip) / math.sqrt(2)