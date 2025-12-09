"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        input_tensor = node.inputs[0]
        return (
            out_grad * self.scalar * power_scalar(input_tensor, self.scalar - 1)
        )


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # reverses the order of two axes (axis1, axis2), 
        # defaults to the last two axes
        axes_to_swap = self.axes
        if axes_to_swap is None:
            axes_to_swap = (len(a.shape) - 1, len(a.shape) - 2)

        new_axes = list(range(len(a.shape)))
        axis1, axis2 = axes_to_swap
        new_axes[axis1], new_axes[axis2] = new_axes[axis2], new_axes[axis1]
        return a.permute(tuple(new_axes))

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a.compact(), self.shape)

    def gradient(self, out_grad, node):
        input_tensor = node.inputs[0]
        return reshape(out_grad, input_tensor.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_tensor = node.inputs[0]
        in_shape = input_tensor.shape
        out_shape = out_grad.shape
        # Pad in_shape to match out_shape length
        num_missing = len(out_shape) - len(in_shape)
        padded_in_shape = (1,) * num_missing + in_shape
        # Find axes that were broadcast
        axes = tuple(
            i for i, (in_dim, out_dim) in enumerate(
                zip(padded_in_shape, out_shape))
            if in_dim == 1 and out_dim > 1
        )
        grad = summation(out_grad, axes)
        return reshape(grad, input_tensor.shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        axes = self.axes
        if isinstance(axes, int):
            return array_api.sum(a, axis=axes)
        elif axes is None:
            axes = tuple(range(len(a.shape)))
        ndim = len(a.shape)
        norm_axes = sorted({ax % ndim for ax in axes}, reverse=True)
        for ax in norm_axes:
            a = array_api.sum(a, axis=ax)
        return a

    def gradient(self, out_grad, node):
        input_tensor = node.inputs[0]
        in_shape = input_tensor.shape
        axes = self.axes
        if axes is None:
            axes = tuple(range(len(in_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        else:
            # normalize negatives and deduplicate
            axes = tuple(sorted({ax % len(in_shape) for ax in axes}))
        # Create shape with 1s in summed axes, original dims elsewhere
        shape = list(in_shape)
        for ax in axes:
            shape[ax] = 1
        # Reshape out_grad to this shape so we can broadcast
        grad = reshape(out_grad, tuple(shape))
        return broadcast_to(grad, in_shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = out_grad @ transpose(b)
        grad_b = transpose(a) @ out_grad
        
        def reduce_grad(grad, in_shape):
            # Pad in_shape to match grad length
            num_missing = len(grad.shape) - len(in_shape)
            padded_in_shape = (1,) * num_missing + in_shape
            # Find axes that were broadcast
            axes = tuple(
                i for i, (in_dim, grad_dim) in enumerate(
                    zip(padded_in_shape, grad.shape))
                if in_dim == 1 and grad_dim > 1
            )
            grad = summation(grad, axes)
            return reshape(grad, in_shape)

        grad_a = reduce_grad(grad_a, a.shape)
        grad_b = reduce_grad(grad_b, b.shape)
        return grad_a, grad_b


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        input_tensor = node.inputs[0]
        return out_grad / input_tensor


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        input_tensor = node.inputs[0]
        return out_grad * exp(input_tensor)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        input_tensor = node.inputs[0]
        return out_grad * (input_tensor.realize_cached_data() > 0)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        return array_api.tanh(a)

    def gradient(self, out_grad, node):
        input_tensor = node.inputs[0]
        return out_grad * (-(tanh(input_tensor) ** 2) + 1)


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        arrays = list(args)
        base_shape = arrays[0].shape
        for arr in arrays:
            if arr.shape != base_shape:
                raise ValueError("All input arrays must have the same shape")
            
        axis = self.axis
        if axis < 0:
            axis += len(base_shape) + 1

        out_shape = list(base_shape)
        out_shape.insert(axis, len(arrays))

        out = array_api.empty(
            tuple(out_shape), dtype=arrays[0].dtype, device=arrays[0].device)
        for i, arr in enumerate(arrays):
            # build slicing object
            slicing = [slice(None)] * len(out_shape)
            slicing[axis] = slice(i, i + 1)
            out[tuple(slicing)] = arr
        return out

    def gradient(self, out_grad, node):
        return split(out_grad, self.axis)


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        axis = self.axis
        if axis < 0:
            axis += len(A.shape)
        n = A.shape[axis]
        out = []
        for i in range(n):
            slicing = [slice(None)] * len(A.shape)
            slicing[axis] = slice(i, i + 1)
            part = A[tuple(slicing)]
            new_shape = tuple(list(A.shape[:axis]) + list(A.shape[axis + 1:]))
            out.append(part.compact().reshape(new_shape))
        return tuple(out)

    def gradient(self, out_grad, node):
        return stack(list(out_grad), self.axis)


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.flip(a, self.axes)

    def gradient(self, out_grad, node):
        return flip(out_grad, self.axes)


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        out_shape = list(a.shape)
        for axis in self.axes:
            if axis < 0 or axis >= len(a.shape):
                continue
            out_shape[axis] = a.shape[axis] * (self.dilation + 1)
        out = array_api.full(
            tuple(out_shape), 
            fill_value=0, 
            dtype=a.dtype, 
            device=a.device
        )
        # build slicing object
        slicing = [slice(None)] * len(out_shape)
        for axis in self.axes:
            if axis < 0 or axis >= len(out_shape):
                continue
            slicing[axis] = slice(0, out_shape[axis], self.dilation + 1)
        out[tuple(slicing)] = a
        return out

    def gradient(self, out_grad, node):
        return undilate(out_grad, self.axes, self.dilation)


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # build slicing object
        slicing = [slice(None)] * len(a.shape)
        for axis in self.axes:
            if axis < 0 or axis >= len(a.shape):
                continue
            slicing[axis] = slice(0, a.shape[axis], self.dilation + 1)
        out = a[tuple(slicing)]
        return out

    def gradient(self, out_grad, node):
        return dilate(out_grad, self.axes, self.dilation)


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        out_H = (H + 2 * self.padding - K) // self.stride + 1
        out_W = (W + 2 * self.padding - K) // self.stride + 1

        if self.padding > 0:
            A = A.pad(((0,0), 
                       (self.padding, self.padding), 
                       (self.padding, self.padding), 
                       (0,0)))
        
        Ns, Hs, Ws, Cs = A.strides

        transformed = array_api.NDArray.make(
            shape=(N, out_H, out_W, K, K, C_in),
            strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
            device=A.device,
            handle=A._handle,
            offset=A._offset
        ).compact().reshape((N * out_H * out_W, K * K * C_in))

        out = transformed @ B.compact().reshape((K * K * C_in, C_out))
        return out.reshape((N, out_H, out_W, C_out))

    def gradient(self, out_grad, node):
        A, B = node.inputs
        K, _, _, _ = B.shape

        dilated_out_grad = dilate(
            out_grad, axes=(1, 2), dilation=self.stride - 1)

        A_grad = conv(
            dilated_out_grad,
            transpose(flip(B, axes=(0, 1)), axes=(2, 3)),
            padding=K - 1 - self.padding
        )
        
        B_grad = conv(
            transpose(A, axes=(0, 3)),
            transpose(transpose(dilated_out_grad, axes=(0, 1)), axes=(1, 2)),
            padding=self.padding
        )

        return A_grad, transpose(transpose(B_grad, axes=(0, 1)), axes=(1, 2))


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


