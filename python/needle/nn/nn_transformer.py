from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        a_shape = (*a.shape[:-1], 1, *a.shape[-1:])
        a = a.reshape(a_shape)

        b_transpose_shape = (*b_transpose.shape[:-2], 1, *b_transpose.shape[-2:])
        b_transpose = b_transpose.reshape(b_transpose_shape)

        broadcast_shape = list(a_shape)
        broadcast_shape[-2] = b_transpose_shape[-2]
        a = a.broadcast_to(broadcast_shape)

        broadcast_shape = list(b_transpose_shape)
        broadcast_shape[-3] = a_shape[-3]
        b_transpose = b_transpose.broadcast_to(broadcast_shape)

        return (a * b_transpose).sum(len(a.shape) - 1)

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        # Compute attention scores
        attn_scores = self.matmul(q, k) 
        attn_scores = attn_scores / (q_dim ** 0.5)

        if self.causal:
            mask_nd = self.create_causal_mask(queries_len, keys_values_len, q.device)
            mask = Tensor(mask_nd, device=q.device, dtype=q.dtype, requires_grad=False)
            mask = mask.broadcast_to((batch_size, num_head, queries_len, keys_values_len))
            attn_scores = attn_scores + mask

        # Compute attention probabilities
        probs = self.softmax(attn_scores)
        probs = self.dropout(probs)

        # Compute attention output
        result = self.matmul(probs, v.transpose((2, 3)))

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        # Reshape for prenorm
        q = q.reshape((batch_size * queries_len, q_dim))
        k = k.reshape((batch_size * keys_values_len, k_dim))
        v = v.reshape((batch_size * keys_values_len, v_dim))

        # Prenorm
        q = self.prenorm_q(q)  # batch_size * queries_len, q_dim
        k = self.prenorm_k(k)  # batch_size * keys_values_len, k_dim
        v = self.prenorm_v(v)  # batch_size * keys_values_len, v_dim

        q = self.q_projection(q)  # batch_size * queries_len, num_head * dim_head
        q = q.reshape((batch_size, queries_len, self.num_head, self.dim_head))  # batch_size, queries_len, num_head, dim_head
        q = q.transpose((1, 2))  # -> batch_size, num_head, queries_len, dim_head

        k = self.k_projection(k)  # batch_size * keys_values_len, num_head * dim_head
        k = k.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))  # batch_size, keys_values_len, num_head, dim_head
        k = k.transpose((1, 2))  # -> batch_size, num_head, keys_values_len, dim_head

        v = self.v_projection(v)  # batch_size * keys_values_len, num_head * dim_head
        v = v.reshape((batch_size, keys_values_len, self.num_head, self.dim_head))  # batch_size, keys_values_len, num_head, dim_head
        v = v.transpose((1, 2))  # -> batch_size, num_head, keys_values_len, dim_head

        result, _ = self.attn(q, k, v)  # batch_size, num_head, queries_len, dim_head
        result = result.transpose((2, 1))  # -> batch_size, queries_len, num_head, dim_head
        result = result.reshape((batch_size * queries_len, self.num_head * self.dim_head))  # batch_size * queries_len, num_head * dim_head

        result = self.out_projection(result)
        result = result.reshape((batch_size, queries_len, self.out_features))  # batch_size, queries_len, out_features

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.self_attn = AttentionLayer(
            q_features=q_features,
            num_head=num_head,
            dim_head=dim_head,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype
        )

        self.linear1 = Linear(
            q_features, hidden_size,
            device=device, dtype=dtype
        )
        self.linear2 = Linear(
            hidden_size, q_features,
            device=device, dtype=dtype
        )
        self.layernorm = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.dropout = Dropout(dropout)
        self.relu = ReLU()

    
    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        x = x + self.dropout(self.self_attn(x))

        x_ = x.reshape((batch_size * seq_len, x_dim))
        x_ = self.dropout(self.linear2(self.dropout(self.relu(self.linear1(self.layernorm(x_))))))
        x_ = x_.reshape((batch_size, seq_len, x_dim))
        x = x + x_

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        self.positional_embedding = Embedding(
            num_embeddings=sequence_len,
            embedding_dim=embedding_size,
            device=device,
            dtype=dtype
        )

        self.layers = Sequential(*[TransformerLayer(
            q_features=embedding_size,
            num_head=num_head,
            dim_head=dim_head,
            hidden_size=hidden_size,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype
        ) for _ in range(num_layers)])


    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        batch_size, seq_len, _ = x.shape
        positions = Tensor(
            np.arange(seq_len, dtype=np.int32), 
            device=x.device, 
            requires_grad=False
        )
        positions = positions.reshape((seq_len, 1)).broadcast_to((seq_len, batch_size))  # seq_len, batch_size
        pos_embeddings = self.positional_embedding(positions)  # seq_len, batch_size, embedding_size
        pos_embeddings = ops.transpose(pos_embeddings, axes=(0, 1))  # -> batch_size, seq_len, embedding_size

        x = x + pos_embeddings
        x = self.layers(x)

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)