"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.weight = Parameter(
            init.kaiming_uniform(
                in_channels * kernel_size * kernel_size,
                out_channels * kernel_size * kernel_size,
                shape=(kernel_size, kernel_size, in_channels, out_channels),
                device=device,
                dtype=dtype,
            ),
            requires_grad=True
        )
        if bias:
            self.bias = Parameter(
                init.rand(
                    out_channels,
                    low=-1.0 / np.sqrt(in_channels * kernel_size * kernel_size),
                    high=1.0 / np.sqrt(in_channels * kernel_size * kernel_size),
                    device=device,
                    dtype=dtype,
                ),
                requires_grad=True
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # convert NCHW to NHWC
        x = x.transpose((1, 2))  # NCHW -> NHCW
        x = x.transpose((2, 3))  # NHCW -> NHWC

        K = self.kernel_size
        pad = (K - 1) // 2
        out = ops.conv(
            x,
            self.weight,
            stride=self.stride,
            padding=pad,
        )
        if self.bias is not None:
            out = out + self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(
                (out.shape[0], out.shape[1], out.shape[2], self.out_channels))
            
        # convert back to NCHW
        out = out.transpose((2, 3))  # NHWC -> NHCW
        out = out.transpose((1, 2))  # NHCW -> NCHW
        return out