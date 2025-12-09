"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            # detach gradient to avoid creating a computation graph
            p_grad = p.grad.detach()

            # apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                p_grad = p_grad + self.weight_decay * p.data

            # momentum update
            if self.momentum != 0:
                if p not in self.u: 
                    # initialize velocity with zeros on same device/dtype
                    self.u[p] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype)
                self.u[p] = self.momentum * self.u[p] + (1 - self.momentum) * p_grad
                p_grad = self.u[p]

            p.data = p.data - self.lr * p_grad


    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}  # first moment vector
        self.v = {}  # second moment vector

    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            
            # detach gradient to avoid creating a computation graph
            p_grad = p.grad.detach()

            # apply weight decay (L2 regularization)
            if self.weight_decay != 0:
                p_grad = p_grad + self.weight_decay * p.data

            if p not in self.m:
                self.m[p] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype).data
                self.v[p] = ndl.init.zeros(*p.shape, device=p.device, dtype=p.dtype).data

            # update biased first moment estimate and second moment estimate
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * p_grad
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (p_grad * p_grad)

            # bias correction
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)

            p.data = p.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
