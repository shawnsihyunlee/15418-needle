#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4

#define L 64
#define S 8
#define V 4

typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call.
   * Basically says: I want 256 threads per block, and enough blocks to cover all elements.
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem calls
////////////////////////////////////////////////////////////////////////////////

static __device__ size_t GetNonCompactIndex(const CudaVec& strides, const CudaVec& shape, 
                                            size_t offset, size_t compactIdx) {
  /**
   * Given a compact index into an array, return the corresponding non-compact index.
   * 
   * Args:
   *   strides: strides of the non-compact array
   *   shape: shape of the non-compact array
   *   offset: offset of the non-compact array
   *   compactIdx: index into the compact array
   * 
   * Returns:
   *   nonCompactIdx: corresponding index into the non-compact array
   */
  size_t nonCompactIdx = offset;
  size_t tmp = compactIdx;
  for (size_t i = shape.size - 1; i < shape.size; i--) {
    // (tmp % shape[i]) is the index along dimension i
    // then multiply by stride (which tell us how far apart consecutive elements 
    // along that dimension are in memory)
    nonCompactIdx += strides.data[i] * (tmp % shape.data[i]);
    // move onto next dimension
    tmp /= shape.data[i];
  }
  return nonCompactIdx;
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of a array
   *   offset: offset of a array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index and index into out
  if (gid >= size) return;

  // convert gid into index into a
  size_t a_idx = GetNonCompactIndex(strides, shape, offset, gid);
  out[gid] = a[a_idx];
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the elementwise setitem operation. 
   *
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of a array
   *   shape: vector of shapes of a and out arrays
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index and index into out
  if (gid >= size) return;

  // convert gid into index into out
  size_t out_idx = GetNonCompactIndex(strides, shape, offset, gid);
  out[out_idx] = a[gid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  You will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
}


__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                                    CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the scalar setitem operation. 
   *
   * Args:
   *   val: scalar value to write to
   *   out: CUDA point to out array
   *   size: number of elements to write in out array
   *   shape: vector of shapes of out array
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index and index into out
  if (gid >= size) return;

  // convert gid into index into out
  size_t out_idx = GetNonCompactIndex(strides, shape, offset, gid);
  out[out_idx] = val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                               VecToCuda(strides), offset);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

namespace {  // internal utilities for this file only

template <typename Op>  // typename or class is equivalent here
__global__ void EwiseApplyKernel(const scalar_t* __restrict__ a,
                                 const scalar_t* __restrict__ b,
                                 scalar_t* __restrict__ out,
                                 size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid], b[gid]);
}

template <typename Op>
__global__ void ScalarApplyKernel(const scalar_t* __restrict__ a,
                                  scalar_t val,
                                  scalar_t* __restrict__ out,
                                  size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid], val);
}

template <typename Op>
__global__ void UnaryApplyKernel(const scalar_t* __restrict__ a,
                                 scalar_t* __restrict__ out,
                                 size_t size, Op op) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = op(a[gid]);
}

// Using functor instead of lambda to avoid --extended-lambda nvcc compiler flag
struct AddOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x + y; } };
struct MulOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x * y; } };
struct DivOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x / y; } };
struct PowOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return powf(x, y); } };
struct MaxOp { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x > y ? x : y; } };
struct EqOp  { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x == y ? 1.0f : 0.0f; } };
struct GeOp  { __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x >= y ? 1.0f : 0.0f; } };

struct LogOp  { __device__ scalar_t operator()(scalar_t x) const { return logf(x); } };
struct ExpOp  { __device__ scalar_t operator()(scalar_t x) const { return expf(x); } };
struct TanhOp { __device__ scalar_t operator()(scalar_t x) const { return tanhf(x); } };

} // unnamed namespace

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be added
   *   b: Input array 'b' to be added
   *   out: Output array to store the result of 'a + b'
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseApplyKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, AddOp{});
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add a scalar value to every element of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to be added
   *   out: Output array to store the result of 'a + val'
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarApplyKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, AddOp{});
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Element-wise multiplication of two CUDA arrays.
   * Args:
   *   a: Input array 'a' to be multiplied
   *   b: Input array 'b' to be multiplied
   *   out: Output array to store the result of 'a * b'
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseApplyKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, MulOp{});
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Multiply every element of a CUDA array by a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value to multiply
   *   out: Output array to store the result of 'a * val'
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarApplyKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, MulOp{});
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Element-wise division of two CUDA arrays.
   * Args:
   *   a: Input array 'a' (numerator)
   *   b: Input array 'b' (denominator)
   *   out: Output array to store the result of 'a / b'
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseApplyKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, DivOp{});
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Divide every element of a CUDA array by a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value (denominator)
   *   out: Output array to store the result of 'a / val'
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarApplyKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, DivOp{});
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Raise every element of a CUDA array to the power of a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar exponent
   *   out: Output array to store the result of 'a ** val'
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarApplyKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, PowOp{});
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Element-wise maximum of two CUDA arrays.
   * Args:
   *   a: Input array 'a'
   *   b: Input array 'b'
   *   out: Output array to store the result of 'max(a, b)'
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseApplyKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, MaxOp{});
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Take the maximum of every element of a CUDA array and a scalar value.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value
   *   out: Output array to store the result of 'max(a, val)'
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarApplyKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, MaxOp{});
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Element-wise equality comparison of two CUDA arrays.
   * Args:
   *   a: Input array 'a'
   *   b: Input array 'b'
   *   out: Output array to store the result of 'a == b' (1.0 for true, 0.0 for false)
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseApplyKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, EqOp{});
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Compare every element of a CUDA array to a scalar value for equality.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value
   *   out: Output array to store the result of 'a == val' (1.0 for true, 0.0 for false)
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarApplyKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, EqOp{});
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Element-wise greater-than-or-equal-to comparison of two CUDA arrays.
   * Args:
   *   a: Input array 'a'
   *   b: Input array 'b'
   *   out: Output array to store the result of 'a >= b' (1.0 for true, 0.0 for false)
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseApplyKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, GeOp{});
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Compare every element of a CUDA array to a scalar value for greater-than-or-equal-to.
   * Args:
   *   a: Input array 'a'
   *   val: Scalar value
   *   out: Output array to store the result of 'a >= val' (1.0 for true, 0.0 for false)
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarApplyKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, GeOp{});
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  /**
   * Element-wise natural logarithm of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   out: Output array to store the result of 'log(a)'
   */
  CudaDims dim = CudaOneDim(out->size);
  UnaryApplyKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, LogOp{});
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  /**
   * Element-wise exponential of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   out: Output array to store the result of 'exp(a)'
   */
  CudaDims dim = CudaOneDim(out->size);
  UnaryApplyKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, ExpOp{});
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  /**
   * Element-wise hyperbolic tangent of a CUDA array.
   * Args:
   *   a: Input array 'a'
   *   out: Output array to store the result of 'tanh(a)'
   */
  CudaDims dim = CudaOneDim(out->size);
  UnaryApplyKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, TanhOp{});
}

////////////////////////////////////////////////////////////////////////////////
// Matmul
////////////////////////////////////////////////////////////////////////////////

__global__ void MatmulKernel(const scalar_t* __restrict__ a,
                             const scalar_t* __restrict__ b,
                             scalar_t* __restrict__ out,
                             uint32_t M, uint32_t N, uint32_t P) {
  __shared__ scalar_t sA[L][S];   
  __shared__ scalar_t sB[S][L];  

  scalar_t a_reg[V], b_reg[V];
  scalar_t c[V][V] = {0};

  const uint32_t blockRow = blockIdx.y * L;
  const uint32_t blockCol = blockIdx.x * L;
  const uint32_t laneRow0 = blockRow + threadIdx.y * V;
  const uint32_t laneCol0 = blockCol + threadIdx.x * V;

  // iterate over K in tiles of S
  for (uint32_t ko = 0; ko < N; ko += S) {
    // --- cooperative loads (A: coalesced along k) ---
    for (int t = threadIdx.y; t < L; t += blockDim.y) {
      uint32_t r = blockRow + t;
      #pragma unroll
      for (int u = threadIdx.x; u < S; u += blockDim.x) {
        uint32_t k = ko + u;
        sA[t][u] = (r < M && k < N) ? a[r * N + k] : 0.f;   // contiguous in k
      }
    }
    // B load (already coalesced along c)
    for (int t = threadIdx.y; t < S; t += blockDim.y) {
      uint32_t k = ko + t;
      #pragma unroll
      for (int u = threadIdx.x; u < L; u += blockDim.x) {
        uint32_t ccol = blockCol + u;
        sB[t][u] = (k < N && ccol < P) ? b[k * P + ccol] : 0.f;
      }
    }

    __syncthreads();  // loads into shared memory complete

    // --- compute: register-blocked outer products ---
    #pragma unroll
    for (int ki = 0; ki < (int)S; ++ki) {
      #pragma unroll
      for (int v = 0; v < V; ++v) {
        int l = threadIdx.y * V + v;          // 0..L-1
        a_reg[v] = (laneRow0 + v < M) ? sA[l][ki] : 0.f;  // note sA[L][S]
      }
      #pragma unroll
      for (int v = 0; v < V; ++v) {
        int l = threadIdx.x * V + v;
        b_reg[v] = (laneCol0 + v < P) ? sB[ki][l] : 0.f;
      }
      #pragma unroll
      for (int y = 0; y < V; ++y)
        #pragma unroll
        for (int x = 0; x < V; ++x)
          c[y][x] += a_reg[y] * b_reg[x];
    }
    __syncthreads(); // reads from shared memory complete
  }

  // --- single store after accumulating all K-tiles ---
  #pragma unroll
  for (int y = 0; y < V; ++y) {
    uint32_t r = laneRow0 + y;
    if (r >= M) break;
    #pragma unroll
    for (int x = 0; x < V; ++x) {
      uint32_t col = laneCol0 + x;
      if (col < P) out[r * P + col] = c[y][x];
    }
  }
}
  

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
  dim3 block(L / V, L / V, 1);                 // (16,16,1)
  dim3 grid((P + L - 1) / L, (M + L - 1) / L, 1);
  MatmulKernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  /**
   * The CUDA kernel for the max reduction operation.
   * 
   * Args:
   *   a: input array to reduce over
   *   out: output array to write reduced results into
   *   reduce_size: size of the dimension to reduce over
   *   out_size: size of the output array
   */
  // One thread per output element
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;
  size_t base = gid * reduce_size;
  scalar_t accum = a[base];
  for (size_t i = 1; i < reduce_size; i++) {
    if (a[base + i] > accum) {
      accum = a[base + i];
    }
  }
  out[gid] = accum;
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  assert(a.size == out->size * reduce_size);
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}


__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  /**
   * The CUDA kernel for the sum reduction operation.
   * 
   * Args:
   *   a: input array to reduce over
   *   out: output array to write reduced results into
   *   reduce_size: size of the dimension to reduce over
   *   out_size: size of the output array
   */
  // One thread per output element
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_size) return;
  size_t base = gid * reduce_size;
  scalar_t accum = 0;
  for (size_t i = 0; i < reduce_size; i++) {
    accum += a[base + i];
  }
  out[gid] = accum;
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
  assert(a.size == out->size * reduce_size);
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  // Multi-GPU device management
  m.def("set_device", [](int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("get_device", []() {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return device;
  });

  m.def("get_device_count", []() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    return count;
  });
}
