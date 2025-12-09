"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose()
            )
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        N = X.shape[0]
        if self.bias is None:
            return X @ self.weight
        else:
            return X @ self.weight + self.bias.broadcast_to((N, self.out_features))


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        # X has shape (N, d1, d2, ..., dk)
        N = X.shape[0]
        D = 1
        for s in X.shape[1:]:
            D *= s
        return X.reshape((N, D))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        y_one_hot = init.one_hot(logits.shape[-1], y, dtype=logits.dtype, device=logits.device)
        # normalize negative axis to positive index
        axis = len(logits.shape) - 1 
        loss_per_ex = ops.logsumexp(logits, axes=(axis,)) - (logits * y_one_hot).sum(axes=(axis,))
        return ops.summation(loss_per_ex) / logits.shape[0]



class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        N, D = x.shape
        w = self.weight.reshape((1, D)).broadcast_to((N, D))
        b = self.bias.reshape((1, D)).broadcast_to((N, D))
        if self.training:
            batch_mean = ops.summation(x, axes=(0,)) / N
            batch_mean_broadcasted = batch_mean.reshape((1, D)).broadcast_to((N, D))
            x_centered = x - batch_mean_broadcasted
            batch_var = ops.summation(x_centered ** 2, axes=(0,)) / N
            batch_var_broadcasted = batch_var.reshape((1, D)).broadcast_to((N, D))
            x_hat = x_centered / (batch_var_broadcasted + self.eps) ** 0.5

            # update running mean and var
            self.running_mean = ((1 - self.momentum) * self.running_mean + 
                                 self.momentum * batch_mean)
            self.running_var = ((1 - self.momentum) * self.running_var + 
                                self.momentum * batch_var)
            
            return w * x_hat + b 
        else:
            running_mean = self.running_mean.reshape((1, D)).broadcast_to((N, D))
            running_var = self.running_var.reshape((1, D)).broadcast_to((N, D))
            x_hat = (x - running_mean) / (running_var + self.eps) ** 0.5
            return w * x_hat + b


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        N, D = x.shape
        mean = ops.summation(x, axes=(1,)) / D
        mean = mean.reshape((N, 1)).broadcast_to((N, D))
        x_centered = x - mean
        var = ops.summation(x_centered ** 2, axes=(1,)) / D
        var = var.reshape((N, 1)).broadcast_to((N, D))
        x_hat = x_centered / (var + self.eps) ** 0.5
        w = self.weight.reshape((1, D)).broadcast_to((N, D))
        b = self.bias.reshape((1, D)).broadcast_to((N, D))
        return w * x_hat + b


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype=x.dtype)
            return (x * mask) / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
