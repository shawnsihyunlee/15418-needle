# Apple Silicon acceleration and data parallelism for Needle

This project for CMU's 15-418: Parallel Computer Architecture and Programming implements an Apple Silicon accelerated backend and adds multi-node data parallelism for Needle, a home-built deep learning framework originally developed for CMU’s Deep Learning Systems course. Our project consists of three major deliverables:

- A Metal Shader Language (MSL) backend that provides parallel GPU kernels for all NDArray primitive operations, including a fully optimized tiled matrix multiplication.
- A Metal Performance Shaders (MPS) backend that leverages Apple’s vendor-optimized BLAS and tensor operations for benchmarking and comparison.
- A data-parallel distributed training system using MPI, enabling synchronous gradient averaging across multiple processes.

Our implementation runs on Apple Silicon (M3 Pro) and achieves substantial speedups: up to 10x faster matrix multiplication, 20x faster MLP train-step, and 9x faster ResNet-9 inference compared to our CPU backend. For our data-parallel setup, with MPI across two and four ranks, we achieve good scaling when training ResNet9 on the CIFAR-10 dataset, and expect similar speedups in setups with more compute nodes.

# Quickstart

Run `make` to build on any platform. The CMake file will automatically detect CUDA / Apple Silicon support and compile the appropriate binaries.

All benchmarks are done in the following two files:
- `benchmarking.ipynb`
- `mpi_benchmarking.ipynb`