"""MPI utilities for distributed multi-GPU training.

This module provides MPI-based data parallelism for Needle. Each MPI process
runs on a separate GPU and processes communicate via MPI collectives.

Requirements:
    - mpi4py: pip install mpi4py
    - MPI implementation (OpenMPI, MPICH, etc.)

Usage:
    mpirun -np 4 python train_script.py
"""

import numpy as np
from typing import Optional

# Try to import MPI
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None


def is_available() -> bool:
    """Check if MPI is available."""
    return HAS_MPI


def get_world_size() -> int:
    """Get total number of MPI processes."""
    if not HAS_MPI:
        return 1
    return MPI.COMM_WORLD.Get_size()


def get_rank() -> int:
    """Get rank of current MPI process (0 to world_size-1)."""
    if not HAS_MPI:
        return 0
    return MPI.COMM_WORLD.Get_rank()


def is_master() -> bool:
    """Check if this is the master process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if HAS_MPI:
        MPI.COMM_WORLD.Barrier()


def broadcast_parameters(model, root: int = 0, comm=None):
    """Broadcast model parameters from root process to all others.

    This ensures all processes start with identical parameters, solving
    the random initialization problem.

    Args:
        model: Needle model (nn.Module)
        root: Source rank for broadcast (default: 0)
        comm: MPI communicator (default: MPI.COMM_WORLD)

    Example:
        >>> # On all processes
        >>> model = ResNet9(device=ndl.cuda(rank))
        >>> # Model initialized randomly (different on each process!)
        >>>
        >>> # Synchronize so all processes have same initial parameters
        >>> mpi.broadcast_parameters(model, root=0)
        >>> # Now all processes have identical parameters from rank 0
    """
    if not HAS_MPI:
        return

    if comm is None:
        comm = MPI.COMM_WORLD

    rank = comm.Get_rank()

    # Import here to avoid circular dependency
    from .autograd import Tensor

    for param in model.parameters():
        # Get parameter data as numpy array
        if rank == root:
            param_data = param.data.numpy()
        else:
            # Allocate buffer for receiving
            param_data = np.empty(param.shape, dtype=np.float32)

        # Broadcast from root to all processes
        comm.Bcast(param_data, root=root)

        # Update parameter on non-root processes
        if rank != root:
            param.data = Tensor(param_data, device=param.device)


def sync_gradients(model, comm=None):
    """Synchronize (average) gradients across all MPI processes.

    This performs an all-reduce operation on gradients, averaging them
    across all processes. Call this after loss.backward() and before
    optimizer.step().

    Args:
        model: Needle model with computed gradients
        comm: MPI communicator (default: MPI.COMM_WORLD)

    Example:
        >>> loss = loss_fn(model(X), y)
        >>> loss.backward()
        >>>
        >>> # Average gradients across all GPUs
        >>> mpi.sync_gradients(model)
        >>>
        >>> optimizer.step()
    """
    if not HAS_MPI:
        return

    if comm is None:
        comm = MPI.COMM_WORLD

    world_size = comm.Get_size()

    if world_size == 1:
        return  # No sync needed for single process

    # Import here to avoid circular dependency
    from .autograd import Tensor

    for param in model.parameters():
        if param.grad is None:
            continue

        # Get gradient as numpy array (on this GPU)
        local_grad = np.ascontiguousarray(param.grad.numpy())

        # Allocate buffer for averaged gradient
        avg_grad = np.empty_like(local_grad)

        # All-reduce: sum gradients across all processes
        comm.Allreduce(local_grad, avg_grad, op=MPI.SUM)

        # Average by dividing by world size
        avg_grad /= world_size

        # Update gradient with averaged value
        param.grad = Tensor(avg_grad, device=param.device, requires_grad=False)


def all_reduce_scalar(value: float, op: str = 'sum', comm=None) -> float:
    """All-reduce a scalar value across all processes.

    Useful for aggregating metrics (loss, accuracy) across processes.

    Args:
        value: Scalar value to reduce
        op: Reduction operation - 'sum', 'mean', 'max', 'min'
        comm: MPI communicator (default: MPI.COMM_WORLD)

    Returns:
        Reduced value

    Example:
        >>> local_loss = compute_loss()  # Different on each GPU
        >>> avg_loss = mpi.all_reduce_scalar(local_loss, op='mean')
        >>> if mpi.is_master():
        >>>     print(f"Average loss: {avg_loss}")
    """
    if not HAS_MPI:
        return value

    if comm is None:
        comm = MPI.COMM_WORLD

    # Map operation name to MPI operation
    mpi_ops = {
        'sum': MPI.SUM,
        'max': MPI.MAX,
        'min': MPI.MIN,
    }

    if op == 'mean':
        # For mean, do sum then divide
        result = np.array(value, dtype=np.float32)
        comm.Allreduce(MPI.IN_PLACE, result, op=MPI.SUM)
        return float(result) / comm.Get_size()
    elif op in mpi_ops:
        result = np.array(value, dtype=np.float32)
        comm.Allreduce(MPI.IN_PLACE, result, op=mpi_ops[op])
        return float(result)
    else:
        raise ValueError(f"Unknown operation: {op}")


def print_once(*args, **kwargs):
    """Print from master process only.

    Useful for logging during distributed training.

    Example:
        >>> mpi.print_once(f"Epoch {epoch}, Loss: {loss}")
        >>> # Only rank 0 will print
    """
    if is_master():
        print(*args, **kwargs)


def setup_device_for_rank(rank: Optional[int] = None):
    """Get the appropriate CUDA device for this MPI rank.

    Args:
        rank: MPI rank (default: auto-detect from MPI.COMM_WORLD)

    Returns:
        BackendDevice for the appropriate GPU

    Example:
        >>> import needle as ndl
        >>> from needle import mpi
        >>>
        >>> device = mpi.setup_device_for_rank()
        >>> model = ResNet9(device=device)
    """
    if rank is None:
        rank = get_rank()

    from . import backend_ndarray as nd

    # Check if CUDA is available
    num_gpus = nd.cuda_device_count()

    if num_gpus == 0:
        # No GPUs available, use CPU
        print_once("Warning: No CUDA devices found, using CPU")
        return nd.cpu()

    # Assign GPU based on rank (round-robin if more ranks than GPUs)
    gpu_id = rank % num_gpus

    return nd.cuda(gpu_id)


class MPIContext:
    """Context manager for MPI distributed training.

    Provides convenient access to MPI information and ensures proper
    initialization.

    Example:
        >>> with MPIContext() as ctx:
        >>>     print(f"Rank {ctx.rank} of {ctx.world_size}")
        >>>     model = ResNet9(device=ctx.device)
        >>>     ctx.broadcast_parameters(model)
    """

    def __init__(self):
        if not HAS_MPI:
            raise RuntimeError("MPI not available. Install with: pip install mpi4py")

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
        self.device = setup_device_for_rank(self.rank)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up if needed
        pass

    def broadcast_parameters(self, model, root: int = 0):
        """Broadcast parameters from root."""
        broadcast_parameters(model, root=root, comm=self.comm)

    def sync_gradients(self, model):
        """Synchronize gradients."""
        sync_gradients(model, comm=self.comm)

    def all_reduce(self, value: float, op: str = 'sum') -> float:
        """All-reduce a scalar."""
        return all_reduce_scalar(value, op=op, comm=self.comm)

    def barrier(self):
        """Synchronize all processes."""
        self.comm.Barrier()

    def is_master(self) -> bool:
        """Check if this is the master process."""
        return self.rank == 0

    def print(self, *args, **kwargs):
        """Print from master only."""
        if self.is_master():
            print(*args, **kwargs)
