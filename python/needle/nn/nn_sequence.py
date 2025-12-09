"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.power_scalar(1 + ops.exp(-x), -1)

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        self.hidden_size = hidden_size

        bound = np.sqrt(1/hidden_size)

        self.W_ih = Parameter(
            init.rand(input_size, hidden_size, low=-bound, high=bound), 
            requires_grad=True, device=device, dtype=dtype
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, hidden_size, low=-bound, high=bound), 
            requires_grad=True, device=device, dtype=dtype
        )

        if bias:
            self.bias_ih = Parameter(
                init.rand(hidden_size, low=-bound, high=bound),
                requires_grad=True, device=device, dtype=dtype
            )
            self.bias_hh = Parameter(
                init.rand(hidden_size, low=-bound, high=bound),
                requires_grad=True, device=device, dtype=dtype
            )
        else:
            self.bias_ih = None
            self.bias_hh = None

        if nonlinearity == 'tanh':
            self.nonlinearity = ops.tanh
        elif nonlinearity == 'relu':
            self.nonlinearity = ops.relu
        else:
            raise ValueError("Invalid nonlinearity. Expected 'tanh' or 'relu'.")

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        if h is None:
            h = init.zeros(X.shape[0], self.W_hh.shape[0], 
                           device=X.device, dtype=X.dtype)

        z = X @ self.W_ih + h @ self.W_hh
        if self.bias_ih is not None:
            z = z + self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(z.shape)
        if self.bias_hh is not None:
            z = z + self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(z.shape)

        return self.nonlinearity(z)


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_cells = []
        for layer in range(num_layers):
            curr_input_size = input_size if layer == 0 else hidden_size
            rnn_cell = RNNCell(
                curr_input_size,
                hidden_size,
                bias=bias,
                nonlinearity=nonlinearity,
                device=device,
                dtype=dtype
            )
            self.rnn_cells.append(rnn_cell)
        

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, bs, _ = X.shape

        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, 
                            device=X.device, dtype=X.dtype)

        h0_by_layer = ops.split(h0, axis=0)
        X_by_time = ops.split(X, axis=0)

        h_n = []
        layer_input = X_by_time
        for layer in range(self.num_layers):
            h_t = h0_by_layer[layer]
            outputs = []
            for t in range(seq_len):
                h_t = self.rnn_cells[layer](layer_input[t], h_t)
                outputs.append(h_t)
            layer_input = outputs
            h_n.append(h_t)
        output = ops.stack(layer_input, axis=0)
        h_n = ops.stack(h_n, axis=0)
        return output, h_n


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()

        self.hidden_size = hidden_size

        bound = np.sqrt(1/hidden_size)

        self.W_ih = Parameter(
            init.rand(input_size, 4*hidden_size, low=-bound, high=bound), 
            requires_grad=True, device=device, dtype=dtype
        )
        self.W_hh = Parameter(
            init.rand(hidden_size, 4*hidden_size, low=-bound, high=bound), 
            requires_grad=True, device=device, dtype=dtype
        ) 

        if bias:
            self.bias_ih = Parameter(
                init.rand(4*hidden_size, low=-bound, high=bound),
                requires_grad=True, device=device, dtype=dtype
            )
            self.bias_hh = Parameter(
                init.rand(4*hidden_size, low=-bound, high=bound),
                requires_grad=True, device=device, dtype=dtype
            )
        else:
            self.bias_ih = None
            self.bias_hh = None

        self.sigmoid = Sigmoid()

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        batch_size = X.shape[0]

        if h is None:
            h0 = init.zeros(batch_size, self.hidden_size, dtype=X.dtype, device=X.device)
            c0 = init.zeros(batch_size, self.hidden_size, dtype=X.dtype, device=X.device)
        else:
            h0, c0 = h

        gates = X @ self.W_ih + h0 @ self.W_hh  # (bs, 4*hidden_size)
        if self.bias_ih is not None: 
            gates = gates + self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(gates.shape)
        if self.bias_hh is not None: 
            gates = gates + self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(gates.shape)
        
        # prepare gates for split
        gates = gates.reshape((batch_size, 4, self.hidden_size))

        i, f, g, o = ops.split(gates, axis=1)
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = ops.tanh(g)
        o = self.sigmoid(o)

        c = f * c0 + i * g
        h = o * ops.tanh(c)

        return h, c


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_cells = []
        for layer in range(num_layers):
            curr_input_size = input_size if layer == 0 else hidden_size
            lstm_cell = LSTMCell(
                curr_input_size,
                hidden_size,
                bias=bias,
                device=device,
                dtype=dtype
            )
            self.lstm_cells.append(lstm_cell)
        
    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        if h is None:
            h = (init.zeros(self.num_layers, X.shape[1], self.hidden_size, 
                            device=X.device, dtype=X.dtype),
                 init.zeros(self.num_layers, X.shape[1], self.hidden_size, 
                            device=X.device, dtype=X.dtype))

        X_by_time = ops.split(X, axis=0)
        h0_by_layer = ops.split(h[0], axis=0)
        c0_by_layer = ops.split(h[1], axis=0)
        h_n = []
        c_n = []
        layer_input = X_by_time
        for layer in range(self.num_layers):
            h_t = h0_by_layer[layer]
            c_t = c0_by_layer[layer]
            outputs = []
            for t in range(X.shape[0]):
                h_t, c_t = self.lstm_cells[layer](layer_input[t], (h_t, c_t))
                outputs.append(h_t)
            layer_input = outputs
            h_n.append(h_t)
            c_n.append(c_t)
        output = ops.stack(layer_input, axis=0)
        h = ops.stack(h_n, axis=0)
        c = ops.stack(c_n, axis=0)

        return output, (h, c)

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        self.weight = Parameter(
            init.randn(num_embeddings, embedding_dim, mean=0.0, std=1.0,
                       device=device, dtype=dtype),
            requires_grad=True, device=device, dtype=dtype
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        num_embeddings, embedding_dim = self.weight.shape
        seq_len, bs = x.shape
        one_hots = init.one_hot(
            num_embeddings, x, device=x.device, dtype=self.weight.dtype
        )  # (seq_len, bs, num_embeddings)
        # reshape one_hots to (seq_len*bs, num_embeddings) for matmul
        one_hots = one_hots.reshape((seq_len * bs, num_embeddings))
        embeddings = one_hots @ self.weight  # (seq_len*bs, embedding_dim)
        embeddings = embeddings.reshape((seq_len, bs, embedding_dim))  # (seq_len, bs, embedding_dim)
        return embeddings