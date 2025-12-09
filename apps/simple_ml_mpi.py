"""MPI-based distributed training for CIFAR-10 and PTB.

This file contains MPI versions of the training functions from simple_ml.py.
Each MPI process runs on a separate GPU and processes different data in parallel.

Launch with:
    mpirun -np 4 python apps/simple_ml_mpi.py

Requirements:
    pip install mpi4py
"""

import numpy as np
from tqdm import tqdm
import sys
sys.path.append("python/")
import needle as ndl
import needle.nn as nn
from needle import mpi
from needle.data import DistributedDataLoader
from models import *



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

### CIFAR-10 MPI Training ###

def epoch_general_cifar10_mpi(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None, device=None):
    """
    MPI version of epoch_general_cifar10.

    Each process:
    - Iterates over its subset of data (via DistributedDataLoader)
    - Computes forward/backward pass locally
    - Synchronizes gradients across all processes via MPI
    - Aggregates metrics (loss, accuracy) across all processes

    Args:
        dataloader: DistributedDataLoader instance (each process gets different data)
        model: nn.Module instance (replicated on each GPU)
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        device: Device for this process

    Returns:
        avg_acc: average accuracy over entire dataset (all processes)
        avg_loss: average loss over entire dataset (all processes)
    """
    np.random.seed(4)

    rank = mpi.get_rank()
    world_size = mpi.get_world_size()

    if opt is None:
        model.eval()
    else:
        model.train()

    # Local metrics (for this process's subset of data)
    local_total_loss = 0.0
    local_total_correct = 0.0
    local_total_samples = 0

    # Use tqdm
    if rank == 0:
        iterator = tqdm(
            dataloader,
            desc=f"Training rank {rank}",
            ncols=80,
            leave=True,
            dynamic_ncols=True,
            total=len(dataloader),
            file=sys.stdout,
            disable=False,
        )
    else:
        iterator = dataloader

    for X, y in iterator:
        if device is not None:
            X = ndl.Tensor(X.cached_data, requires_grad=False, device=device)
            y = ndl.Tensor(y.cached_data, requires_grad=False, device=device)

        # X: (N, C, H, W), y: (N,)
        N = X.shape[0]

        if opt is not None:
            opt.reset_grad()

        # Forward pass (on this process's batch)
        logits = model(X)
        loss = loss_fn(logits, y)

        # Compute accuracy (local)
        logits_np = logits.cached_data.numpy()
        y_np = y.cached_data.numpy()
        preds_np = logits_np.argmax(axis=1)
        correct = (preds_np == y_np).sum()
        loss_val = float(loss.cached_data.numpy())

        local_total_loss += loss_val * N
        local_total_correct += float(correct)
        local_total_samples += N

        if opt is not None:
            # Backward pass (compute gradients locally)
            loss.backward()

            # *** MPI: Synchronize gradients across all processes ***
            mpi.sync_gradients(model)

            # Optimizer step (each process updates its local copy)
            opt.step()

        if rank == 0:
            sys.stdout.flush()

    # *** MPI: Aggregate metrics across all processes ***
    # Sum losses and samples across all processes
    global_total_loss = mpi.all_reduce_scalar(local_total_loss, op='sum')
    global_total_correct = mpi.all_reduce_scalar(local_total_correct, op='sum')
    global_total_samples = mpi.all_reduce_scalar(local_total_samples, op='sum')

    # Compute global averages
    avg_loss = global_total_loss / global_total_samples if global_total_samples > 0 else 0.0
    avg_acc = global_total_correct / global_total_samples if global_total_samples > 0 else 0.0

    return avg_acc, avg_loss


def train_cifar10_mpi(model, dataset, n_epochs=1, batch_size=128, optimizer=ndl.optim.Adam,
                       lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss, device=None):
    """
    MPI version of train_cifar10.

    Performs {n_epochs} epochs of distributed training across multiple GPUs.

    Args:
        model: nn.Module instance (will be replicated on each GPU)
        dataset: Dataset instance (will be split across processes)
        n_epochs: number of epochs (int)
        batch_size: batch size per process (int) - NOT global batch size!
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        device: Device for this process (default: auto-detect from rank)

    Returns:
        avg_acc: average accuracy over dataset from last epoch
        avg_loss: average loss over dataset from last epoch

    Example:
        >>> # Launch with: mpirun -np 4 python script.py
        >>> from needle import mpi
        >>> from needle.data.datasets import CIFAR10Dataset
        >>>
        >>> # Setup device for this rank
        >>> device = mpi.setup_device_for_rank()
        >>>
        >>> # Load dataset
        >>> dataset = CIFAR10Dataset("data/cifar-10-batches-py", train=True)
        >>>
        >>> # Create model
        >>> model = ResNet9(device=device)
        >>>
        >>> # CRITICAL: Broadcast initial parameters
        >>> mpi.broadcast_parameters(model, root=0)
        >>>
        >>> # Train
        >>> acc, loss = train_cifar10_mpi(
        >>>     model, dataset, n_epochs=10, batch_size=128, device=device
        >>> )
    """
    np.random.seed(4)

    rank = mpi.get_rank()
    world_size = mpi.get_world_size()

    if device is None:
        device = mpi.setup_device_for_rank()

    # Create distributed dataloader (each process gets different data)
    dataloader = DistributedDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        rank=rank,
        world_size=world_size
    )

    # Print info from master only
    mpi.print_once(f"=" * 60)
    mpi.print_once(f"MPI Distributed Training")
    mpi.print_once(f"=" * 60)
    mpi.print_once(f"World size: {world_size} processes")
    mpi.print_once(f"Batch size per GPU: {batch_size}")
    mpi.print_once(f"Effective global batch size: {batch_size * world_size}")
    mpi.print_once(f"Number of epochs: {n_epochs}")
    mpi.print_once(f"Device: {device}")
    mpi.print_once()

    # Create optimizer and loss function
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = loss_fn()

    last_acc, last_loss = 0.0, 0.0
    for epoch in range(n_epochs):
        mpi.print_once(f"Epoch {epoch+1}/{n_epochs}:")

        last_acc, last_loss = epoch_general_cifar10_mpi(
            dataloader,
            model,
            loss_fn=criterion,
            opt=opt,
            device=device
        )

        mpi.print_once(f"  Accuracy: {last_acc:.4f}, Loss: {last_loss:.4f}")

    mpi.print_once()
    mpi.print_once(f"Training complete!")
    mpi.print_once(f"Final - Accuracy: {last_acc:.4f}, Loss: {last_loss:.4f}")

    return last_acc, last_loss


def evaluate_cifar10_mpi(model, dataset, batch_size=128, loss_fn=nn.SoftmaxLoss, device=None):
    """
    MPI version of evaluate_cifar10.

    Computes the test accuracy and loss across all processes.

    Args:
        model: nn.Module instance
        dataset: Dataset instance
        batch_size: batch size per process (int)
        loss_fn: nn.Module class
        device: Device for this process

    Returns:
        avg_acc: average accuracy over entire dataset
        avg_loss: average loss over entire dataset
    """
    np.random.seed(4)

    rank = mpi.get_rank()
    world_size = mpi.get_world_size()

    if device is None:
        device = mpi.setup_device_for_rank()

    # Create distributed dataloader
    dataloader = DistributedDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for evaluation
        rank=rank,
        world_size=world_size
    )

    criterion = loss_fn()
    avg_acc, avg_loss = epoch_general_cifar10_mpi(
        dataloader,
        model,
        loss_fn=criterion,
        opt=None,  # No optimizer for evaluation
        device=device
    )

    return avg_acc, avg_loss


### Example usage ###

if __name__ == "__main__":
    """
    Train ResNet9 on CIFAR-10 with MPI distributed training.

    Launch with:
        mpirun -np 4 python apps/simple_ml_mpi.py

    For single GPU (no MPI):
        python apps/simple_ml_mpi.py
    """
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int, help="number of epochs")
    parser.add_argument("batchsize", type=int, help="per process batch size")
    parser.add_argument("device", type=str, choices=["cpu", "gpu", "numpy", "msl", "mps"],
                        help="device: cpu, gpu, or numpy")
    args = parser.parse_args()

    # unpack
    epochs = args.epochs
    device_name = args.device
    batch_size = args.batchsize

    # map device strings to actual needle devices
    if device_name == "cpu":
        device = ndl.cpu()
    elif device_name == "numpy":
        device = ndl.cpu_numpy()
    elif device_name == "gpu":
        # If you have Needle's GPU backend, otherwise throw error
        try:
            device = ndl.cuda()
        except:
            raise RuntimeError("GPU backend not available")
    elif device_name == "msl":
        # If you have MSL
        try:
            device = ndl.msl()
        except:
            raise RuntimeError("MSL backend not available")
    elif device_name == "mps":
        # If you have MPS
        try:
            device = ndl.mps()
        except:
            raise RuntimeError("MPS backend not available")

    # Check MPI availability
    if not mpi.is_available():
        print("WARNING: MPI not available. Running on single device.")
        print("For multi-GPU training, install mpi4py: pip install mpi4py")
        print()

    # Get MPI info
    rank = mpi.get_rank()
    world_size = mpi.get_world_size()

    # Setup device for this rank
    # device = mpi.setup_device_for_rank()

    mpi.print_once("=" * 60)
    mpi.print_once("CIFAR-10 Training with MPI")
    mpi.print_once("=" * 60)
    mpi.print_once(f"Number of processes: {world_size}")
    mpi.print_once(f"Device: {device}")
    mpi.print_once()

    # Load CIFAR-10 dataset
    mpi.print_once("Loading CIFAR-10 dataset...")
    try:
        train_dataset = ndl.data.CIFAR10Dataset(
            "data/cifar-10-batches-py",
            train=True
        )
        test_dataset = ndl.data.CIFAR10Dataset(
            "data/cifar-10-batches-py",
            train=False
        )
        mpi.print_once(f"  Training samples: {len(train_dataset)}")
        mpi.print_once(f"  Test samples: {len(test_dataset)}")
    except Exception as e:
        mpi.print_once(f"ERROR: Could not load CIFAR-10 dataset: {e}")
        mpi.print_once("Make sure data is in: data/cifar-10-batches-py/")
        sys.exit(1)

    mpi.print_once()

    # Create model
    mpi.print_once("Creating ResNet9 model...")
    model = ResNet9(device=device, dtype="float32")
    mpi.print_once(f"  Parameters: {sum(p.size for p in model.parameters() if hasattr(p, 'size'))}")

    # *** CRITICAL: Synchronize initial parameters across all processes ***
    if world_size > 1:
        mpi.print_once("Broadcasting initial parameters from rank 0 to all processes...")
        mpi.broadcast_parameters(model, root=0)
        mpi.print_once("✓ All processes now have identical parameters")

    mpi.print_once()

    # Training
    mpi.print_once("Starting training...")
    mpi.print_once(f"Num epochs: {epochs}")
    mpi.print_once("-" * 60)

    train_acc, train_loss = train_cifar10_mpi(
        model,
        train_dataset,
        n_epochs=epochs,
        batch_size=batch_size,
        optimizer=ndl.optim.Adam,
        lr=0.001,
        weight_decay=0.001,
        device=device
    )

    mpi.print_once("-" * 60)
    mpi.print_once()

    # Evaluation on test set
    
    mpi.print_once("Evaluating on test set...")
    if rank == 0:
        test_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=False)
        mpi.print_once("Evaluating on test set new...")
        test_dataloader = ndl.data.DataLoader(\
         dataset=test_dataset,
         batch_size=128,
         shuffle=True,)
        test_acc, test_loss = evaluate_cifar10(
            model,
            test_dataloader,
            device=device
        )

    mpi.print_once()
    mpi.print_once("=" * 60)
    mpi.print_once("FINAL RESULTS")
    mpi.print_once("=" * 60)
    mpi.print_once(f"Train - Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")
    if rank == 0:
        print(f"Test  - Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")
    mpi.print_once("=" * 60)


