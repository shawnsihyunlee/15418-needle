from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        max_Z_keep = Z.max(axis=self.axes, keepdims=True)
        # when reducing over all axes, reshape to match Z.ndim for broadcasting
        if self.axes is None:
            max_Z_keep = max_Z_keep.reshape((1,) * Z.ndim)
        max_Z_no_keep = Z.max(axis=self.axes, keepdims=False)
        Z = Z - max_Z_keep.broadcast_to(Z.shape)
        return (
            array_api.log(array_api.sum(array_api.exp(Z), axis=self.axes)) +
            max_Z_no_keep
        )
    
    def gradient(self, out_grad: Tensor, node: Tensor):
        Z = node.inputs[0]

        # normalize axes
        if self.axes is None:
            axes = tuple(range(len(Z.shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes % len(Z.shape),)
        else:
            axes = tuple(ax % len(Z.shape) for ax in self.axes)
        
        # Optional: for numerical stability
        # max_Z = Z.realize_cached_data().max(axis=axes, keepdims=True)  
        # shifted_Z = Z - broadcast_to(max_Z, Z.shape)                            
        exp_Z = exp(Z)               
        sum_exp = summation(exp_Z, axes=axes)       
        
        expand_shape = tuple(
            1 if i in axes else Z.shape[i] for i in range(len(Z.shape))
        )
        sum_exp_reshaped = sum_exp.reshape(expand_shape).broadcast_to(Z.shape)
        softmax_Z = exp_Z / sum_exp_reshaped
        out_grad_reshaped = out_grad.reshape(expand_shape).broadcast_to(Z.shape)
        return out_grad_reshaped * softmax_Z

def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)