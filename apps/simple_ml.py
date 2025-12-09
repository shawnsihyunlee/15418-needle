"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np
from tqdm import tqdm

import sys

sys.path.append("python/")
import needle as ndl

import needle.nn as nn
from apps.models import *
import time

def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION

### CIFAR-10 training ###
def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None, device=ndl.cpu_numpy()):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)

    if opt is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for X, y in tqdm(dataloader, desc=f"Training...", total=len(dataloader)):
        if device is not None:
            X = ndl.Tensor(X.cached_data, requires_grad=False, device=device)
            y = ndl.Tensor(y.cached_data, requires_grad=False, device=device)

        # X: (N, C, H, W), y: (N,)
        N = X.shape[0]

        if opt is not None:
            opt.reset_grad()

        logits = model(X)              # (N, num_classes)
        loss = loss_fn(logits, y)      # scalar

        # accuracy via NumPy
        logits_np = logits.cached_data.numpy()
        y_np = y.cached_data.numpy()
        preds_np = logits_np.argmax(axis=1)
        correct = (preds_np == y_np).sum()
        loss_val = float(loss.cached_data.numpy())

        total_loss += loss_val * N
        total_correct += float(correct)
        total_samples += N

        if opt is not None:
            loss.backward()
            opt.step()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_acc, avg_loss


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss, device=ndl.cpu_numpy()):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = loss_fn()

    last_acc, last_loss = 0.0, 0.0
    for i in range(n_epochs):
        print(f"Epoch {i}:")
        last_acc, last_loss = epoch_general_cifar10(
            dataloader,
            model,
            loss_fn=criterion,
            opt=opt,
            device=device
        )
        print(f"Accuracy: {last_acc}, Loss: {last_loss}")

    return last_acc, last_loss


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss, device=ndl.cpu_numpy()):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    criterion = loss_fn()
    avg_acc, avg_loss = epoch_general_cifar10(
        dataloader,
        model,
        loss_fn=criterion,
        opt=None,
        device=device
    )
    return avg_acc, avg_loss


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()

    nbatch = data.shape[0]
    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0

    h = None  # recurrent state (RNN or LSTM tuple)

    i = 0
    pbar = tqdm(total=nbatch - 1, desc="PTB Training" if opt else "PTB Eval")
    while i < nbatch - 1:
        X, y = ndl.data.datasets.get_batch(data, i, seq_len, device=device, dtype=dtype)
        T, B = X.shape
        N = y.shape[0]  # T * B

        if device is not None:
          X = ndl.Tensor(X.cached_data, device=device, dtype=dtype)
          y = ndl.Tensor(y.cached_data, device=device, dtype=dtype)


        if opt is not None:
            opt.reset_grad()

        # detach hidden state for truncated BPTT
        if h is not None:
            if isinstance(h, tuple):
                h = tuple(
                    ndl.Tensor(hj.cached_data, device=device).detach()
                    for hj in h
                )
            else:
                h = ndl.Tensor(h.cached_data, device=device).detach()
        else:
            # first iteration: initialize h on the device
            pass

        # forward
        logits, h = model(X, h)   # logits: (N, vocab_size)

        loss = loss_fn(logits, y)

        # ----- accuracy via NumPy -----
        logits_np = logits.cached_data.numpy()   # (N, V)
        y_np = y.cached_data.numpy()             # (N,)
        pred_np = logits_np.argmax(axis=1)       # (N,)
        correct_val = (pred_np == y_np).sum()

        loss_val = float(loss.cached_data.numpy())

        total_correct += float(correct_val)
        total_loss += loss_val * N   # loss is mean over N
        total_tokens += N

        if opt is not None:
            loss.backward()

            # optional gradient clipping by global norm
            if clip is not None:
                total_norm_sq = 0.0
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    g_np = p.grad.cached_data.numpy()
                    total_norm_sq += float((g_np ** 2).sum())
                total_norm = total_norm_sq ** 0.5
                if total_norm > clip:
                    scale = clip / (total_norm + 1e-6)
                    for p in model.parameters():
                        if p.grad is None:
                            continue
                        p.grad.cached_data *= scale

            opt.step()

        i += T  # move forward by actual sequence length
        pbar.update(T)
    pbar.close()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = loss_fn()

    last_acc, last_loss = 0.0, 0.0
    for _ in range(n_epochs):
        last_acc, last_loss = epoch_general_ptb(
            data,
            model,
            seq_len=seq_len,
            loss_fn=criterion,
            opt=opt,
            clip=clip,
            device=device,
            dtype=dtype,
        )

    return last_acc, last_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    criterion = loss_fn()
    avg_acc, avg_loss = epoch_general_ptb(
        data,
        model,
        seq_len=seq_len,
        loss_fn=criterion,
        opt=None,      # no optimizer: eval mode
        clip=None,
        device=device,
        dtype=dtype,
    )
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
