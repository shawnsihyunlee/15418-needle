#if !defined(__APPLE__) || !__has_include(<Metal/Metal.h>)
#error "Metal backend requires macOS with Metal headers"
#endif

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace needle {
namespace metal {

#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
constexpr uint32_t MAX_VEC_SIZE = 8;

struct MetalVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

////////////////////////////////////////////////////////////////////////////////
// Metal shader source defining all kernels
////////////////////////////////////////////////////////////////////////////////

static const char* kMetalShaderSrc = R"METAL(
using namespace metal;

constant uint MAX_VEC_SIZE = 8;

struct IntVec {
  uint size;
  int data[MAX_VEC_SIZE];
};

inline ulong gid_to_strided_index(ulong gid, constant IntVec& shape, constant IntVec& strides, ulong offset) {
  ulong idx = gid;
  ulong out = offset;
  for (int d = int(shape.size) - 1; d >= 0; --d) {
    uint dim = shape.data[d] == 0 ? 1 : uint(shape.data[d]);
    ulong coord = dim == 0 ? 0 : idx % dim;
    idx /= dim;
    out += coord * ulong(strides.data[d]);
  }
  return out;
}

kernel void fill_kernel(device float* out [[buffer(0)]],
                        constant float& val [[buffer(1)]],
                        constant ulong& size [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid < size) out[gid] = val;
}

kernel void compact_kernel(const device float* a [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant ulong& size [[buffer(2)]],
                           constant IntVec& shape [[buffer(3)]],
                           constant IntVec& strides [[buffer(4)]],
                           constant ulong& offset [[buffer(5)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  ulong in_index = gid_to_strided_index(gid, shape, strides, offset);
  out[gid] = a[in_index];
}

kernel void ewise_setitem_kernel(const device float* a [[buffer(0)]],
                                 device float* out [[buffer(1)]],
                                 constant ulong& size [[buffer(2)]],
                                 constant IntVec& shape [[buffer(3)]],
                                 constant IntVec& strides [[buffer(4)]],
                                 constant ulong& offset [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  ulong out_index = gid_to_strided_index(gid, shape, strides, offset);
  out[out_index] = a[gid];
}

kernel void scalar_setitem_kernel(const constant float& val [[buffer(0)]],
                                  device float* out [[buffer(1)]],
                                  constant ulong& size [[buffer(2)]],
                                  constant IntVec& shape [[buffer(3)]],
                                  constant IntVec& strides [[buffer(4)]],
                                  constant ulong& offset [[buffer(5)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  ulong out_index = gid_to_strided_index(gid, shape, strides, offset);
  out[out_index] = val;
}

// op codes for binary ops
// 0: add, 1: mul, 2: div, 3: eq, 4: ge, 5: max
kernel void ewise_binary_kernel(const device float* a [[buffer(0)]],
                                const device float* b [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant ulong& size [[buffer(3)]],
                                constant uint& op [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  float av = a[gid];
  float bv = b[gid];
  float res = 0.0f;
  switch (op) {
    case 0: res = av + bv; break;
    case 1: res = av * bv; break;
    case 2: res = av / bv; break;
    case 3: res = (av == bv) ? 1.0f : 0.0f; break;
    case 4: res = (av >= bv) ? 1.0f : 0.0f; break;
    case 5: res = fmax(av, bv); break;
    default: res = 0.0f; break;
  }
  out[gid] = res;
}

kernel void scalar_binary_kernel(const device float* a [[buffer(0)]],
                                 device float* out [[buffer(1)]],
                                 constant ulong& size [[buffer(2)]],
                                 constant float& val [[buffer(3)]],
                                 constant uint& op [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  float av = a[gid];
  float res = 0.0f;
  switch (op) {
    case 0: res = av + val; break;
    case 1: res = av * val; break;
    case 2: res = av / val; break;
    case 3: res = (av == val) ? 1.0f : 0.0f; break;
    case 4: res = (av >= val) ? 1.0f : 0.0f; break;
    case 5: res = fmax(av, val); break;
    default: res = 0.0f; break;
  }
  out[gid] = res;
}

// unary ops: 0 log, 1 exp, 2 tanh
kernel void unary_kernel(const device float* a [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant ulong& size [[buffer(2)]],
                         constant uint& op [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  float av = a[gid];
  float res = 0.0f;
  switch (op) {
    case 0: res = log(av); break;
    case 1: res = exp(av); break;
    case 2: res = tanh(av); break;
    default: res = 0.0f; break;
  }
  out[gid] = res;
}

kernel void scalar_power_kernel(const device float* a [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                constant ulong& size [[buffer(2)]],
                                constant float& val [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  out[gid] = pow(a[gid], val);
}

constant uint TILE = 8;

kernel void matmul_kernel(const device float* A [[buffer(0)]],
                          const device float* B [[buffer(1)]],
                          device float* C [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& N [[buffer(4)]],
                          constant uint& P [[buffer(5)]],
                          ushort2 lid [[thread_position_in_threadgroup]],
                          ushort2 bid [[threadgroup_position_in_grid]]) {
  threadgroup float As[TILE][TILE];
  threadgroup float Bs[TILE][TILE];

  uint global_row = bid.y * TILE + lid.y;
  uint global_col = bid.x * TILE + lid.x;
  float acc = 0.0f;

  uint num_tiles = (N + TILE - 1) / TILE;
  for (uint t = 0; t < num_tiles; ++t) {
    uint a_col = t * TILE + lid.x;
    uint b_row = t * TILE + lid.y;

    As[lid.y][lid.x] = (global_row < M && a_col < N) ? A[global_row * N + a_col] : 0.0f;
    Bs[lid.y][lid.x] = (b_row < N && global_col < P) ? B[b_row * P + global_col] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < TILE; ++k) {
      acc += As[lid.y][k] * Bs[k][lid.x];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (global_row < M && global_col < P) {
    C[global_row * P + global_col] = acc;
  }
}

kernel void reduce_sum_kernel(const device float* a [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant ulong& out_size [[buffer(2)]],
                              constant ulong& reduce_size [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= out_size) return;
  ulong base = gid * reduce_size;
  float total = 0.0f;
  for (ulong i = 0; i < reduce_size; ++i) {
    total += a[base + i];
  }
  out[gid] = total;
}

kernel void reduce_max_kernel(const device float* a [[buffer(0)]],
                              device float* out [[buffer(1)]],
                              constant ulong& out_size [[buffer(2)]],
                              constant ulong& reduce_size [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  if (gid >= out_size) return;
  ulong base = gid * reduce_size;
  float max_val = a[base];
  for (ulong i = 1; i < reduce_size; ++i) {
    float v = a[base + i];
    max_val = v > max_val ? v : max_val;
  }
  out[gid] = max_val;
}

)METAL";

////////////////////////////////////////////////////////////////////////////////
// Metal runtime context/pipeline cache
////////////////////////////////////////////////////////////////////////////////

struct MetalContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

  MetalContext() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) throw std::runtime_error("Metal device not available");
    queue = [device newCommandQueue];
    if (!queue) throw std::runtime_error("Failed to create Metal command queue");

    NSError* err = nil;
    NSString* source = [[NSString alloc] initWithUTF8String:kMetalShaderSrc];
    library = [device newLibraryWithSource:source options:nil error:&err];
#if !__has_feature(objc_arc)
    [source release];
#endif
    if (!library || err) {
      std::string msg = "Metal library compile failed";
      if (err) {
        msg += ": ";
        msg += [[err localizedDescription] UTF8String];
      }
      throw std::runtime_error(msg);
    }
  }

  id<MTLComputePipelineState> Pipeline(const std::string& name) {
    auto it = pipelines.find(name);
    if (it != pipelines.end()) return it->second;
    NSString* fn_name = [[NSString alloc] initWithUTF8String:name.c_str()];
    id<MTLFunction> fn = [library newFunctionWithName:fn_name];
#if !__has_feature(objc_arc)
    [fn_name release];
#endif
    if (!fn) throw std::runtime_error("Metal function not found: " + name);
    NSError* err = nil;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
#if !__has_feature(objc_arc)
    [fn release];
#endif
    if (!pso || err) {
      std::string msg = "Failed to create pipeline for " + name;
      if (err) {
        msg += ": ";
        msg += [[err localizedDescription] UTF8String];
      }
      throw std::runtime_error(msg);
    }
    pipelines[name] = pso;
    return pso;
  }
};

static MetalContext& GetContext() {
  static MetalContext ctx;
  return ctx;
}

struct MetalArray {
  MetalArray(const size_t size) {
    MetalContext& ctx = GetContext();
    MTLResourceOptions opts = MTLResourceStorageModeShared;
    buffer = [ctx.device newBufferWithLength:size * ELEM_SIZE options:opts];
    if (!buffer) throw std::runtime_error("Failed to allocate Metal buffer");
    this->size = size;
  }
  ~MetalArray() {
#if !__has_feature(objc_arc)
    [buffer release];
#endif
  }
  size_t ptr_as_int() { return reinterpret_cast<size_t>([buffer contents]); }

  id<MTLBuffer> buffer;
  size_t size;
};

static inline scalar_t* buffer_ptr(const MetalArray& arr) {
  return reinterpret_cast<scalar_t*>(([arr.buffer contents]));
}

////////////////////////////////////////////////////////////////////////////////
// Helpers for shape conversion and launches
////////////////////////////////////////////////////////////////////////////////

static void MetalSync() {
  MetalContext& ctx = GetContext();
  id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
  [cb commit];
  [cb waitUntilCompleted];
}

static MetalVec VecToMetal(const std::vector<int32_t>& x) {
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded Metal supported max dimensions");
  MetalVec v{};
  v.size = static_cast<uint32_t>(x.size());
  for (size_t i = 0; i < x.size(); i++) v.data[i] = x[i];
  return v;
}

// Launch helpers
template <typename F>
static void Launch1D(const std::string& kernel, size_t size, F bind_buffers) {
  MetalContext& ctx = GetContext();
  id<MTLComputePipelineState> pso = ctx.Pipeline(kernel);
  id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  bind_buffers(enc);
  MTLSize grid = MTLSizeMake(size, 1, 1);
  NSUInteger tg = std::min<NSUInteger>(pso.maxTotalThreadsPerThreadgroup, 256);
  MTLSize tgs = MTLSizeMake(tg, 1, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
  [enc endEncoding];
  [cb commit];
  // [cb waitUntilCompleted];
}

template <typename F>
static void Launch2D(const std::string& kernel, uint32_t width, uint32_t height, F bind_buffers) {
  MetalContext& ctx = GetContext();
  id<MTLComputePipelineState> pso = ctx.Pipeline(kernel);
  id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  bind_buffers(enc);
  MTLSize grid = MTLSizeMake(width, height, 1);
  NSUInteger tx = 16;
  NSUInteger ty = std::max<NSUInteger>(1, pso.maxTotalThreadsPerThreadgroup / tx);
  ty = std::min<NSUInteger>(ty, 16);
  MTLSize tgs = MTLSizeMake(tx, ty, 1);
  [enc dispatchThreads:grid threadsPerThreadgroup:tgs];
  [enc endEncoding];
  [cb commit];
  // [cb waitUntilCompleted];
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

void Fill(MetalArray* out, scalar_t val) {
  uint64_t size = out->size;
  Launch1D("fill_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:out->buffer offset:0 atIndex:0];
    [enc setBytes:&val length:sizeof(val) atIndex:1];
    [enc setBytes:&size length:sizeof(uint64_t) atIndex:2];
  });
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

void Compact(const MetalArray& a, MetalArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  MetalVec mshape = VecToMetal(shape);
  MetalVec mstrides = VecToMetal(strides);
  uint64_t size = out->size;
  uint64_t off = offset;
  Launch1D("compact_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&size length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&mshape length:sizeof(MetalVec) atIndex:3];
    [enc setBytes:&mstrides length:sizeof(MetalVec) atIndex:4];
    [enc setBytes:&off length:sizeof(uint64_t) atIndex:5];
  });
}

void EwiseSetitem(const MetalArray& a, MetalArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  MetalVec mshape = VecToMetal(shape);
  MetalVec mstrides = VecToMetal(strides);
  uint64_t size = a.size;
  uint64_t off = offset;
  Launch1D("ewise_setitem_kernel", a.size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&size length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&mshape length:sizeof(MetalVec) atIndex:3];
    [enc setBytes:&mstrides length:sizeof(MetalVec) atIndex:4];
    [enc setBytes:&off length:sizeof(uint64_t) atIndex:5];
  });
}

void ScalarSetitem(size_t size, scalar_t val, MetalArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  MetalVec mshape = VecToMetal(shape);
  MetalVec mstrides = VecToMetal(strides);
  uint64_t sz = size;
  uint64_t off = offset;
  Launch1D("scalar_setitem_kernel", size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBytes:&val length:sizeof(val) atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&sz length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&mshape length:sizeof(MetalVec) atIndex:3];
    [enc setBytes:&mstrides length:sizeof(MetalVec) atIndex:4];
    [enc setBytes:&off length:sizeof(uint64_t) atIndex:5];
  });
}

enum BinaryOp : uint32_t { kAdd = 0, kMul = 1, kDiv = 2, kEq = 3, kGe = 4, kMax = 5 };
enum UnaryOp : uint32_t { kLog = 0, kExp = 1, kTanh = 2 };

static void LaunchBinary(const MetalArray& a, const MetalArray& b, MetalArray* out, BinaryOp op) {
  uint64_t size = out->size;
  uint32_t op_code = op;
  Launch1D("ewise_binary_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:b.buffer offset:0 atIndex:1];
    [enc setBuffer:out->buffer offset:0 atIndex:2];
    [enc setBytes:&size length:sizeof(uint64_t) atIndex:3];
    [enc setBytes:&op_code length:sizeof(uint32_t) atIndex:4];
  });
}

static void LaunchScalarBinary(const MetalArray& a, scalar_t val, MetalArray* out, BinaryOp op) {
  uint64_t size = out->size;
  uint32_t op_code = op;
  Launch1D("scalar_binary_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&size length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&val length:sizeof(val) atIndex:3];
    [enc setBytes:&op_code length:sizeof(uint32_t) atIndex:4];
  });
}

void EwiseAdd(const MetalArray& a, const MetalArray& b, MetalArray* out) { LaunchBinary(a, b, out, kAdd); }
void ScalarAdd(const MetalArray& a, scalar_t val, MetalArray* out) { LaunchScalarBinary(a, val, out, kAdd); }
void EwiseMul(const MetalArray& a, const MetalArray& b, MetalArray* out) { LaunchBinary(a, b, out, kMul); }
void ScalarMul(const MetalArray& a, scalar_t val, MetalArray* out) { LaunchScalarBinary(a, val, out, kMul); }
void EwiseDiv(const MetalArray& a, const MetalArray& b, MetalArray* out) { LaunchBinary(a, b, out, kDiv); }
void ScalarDiv(const MetalArray& a, scalar_t val, MetalArray* out) { LaunchScalarBinary(a, val, out, kDiv); }
void EwiseEq(const MetalArray& a, const MetalArray& b, MetalArray* out) { LaunchBinary(a, b, out, kEq); }
void ScalarEq(const MetalArray& a, scalar_t val, MetalArray* out) { LaunchScalarBinary(a, val, out, kEq); }
void EwiseGe(const MetalArray& a, const MetalArray& b, MetalArray* out) { LaunchBinary(a, b, out, kGe); }
void ScalarGe(const MetalArray& a, scalar_t val, MetalArray* out) { LaunchScalarBinary(a, val, out, kGe); }
void EwiseMaximum(const MetalArray& a, const MetalArray& b, MetalArray* out) { LaunchBinary(a, b, out, kMax); }
void ScalarMaximum(const MetalArray& a, scalar_t val, MetalArray* out) { LaunchScalarBinary(a, val, out, kMax); }

static void LaunchUnary(const MetalArray& a, MetalArray* out, UnaryOp op) {
  uint64_t size = out->size;
  uint32_t op_code = op;
  Launch1D("unary_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&size length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&op_code length:sizeof(uint32_t) atIndex:3];
  });
}

void EwiseLog(const MetalArray& a, MetalArray* out) { LaunchUnary(a, out, kLog); }
void EwiseExp(const MetalArray& a, MetalArray* out) { LaunchUnary(a, out, kExp); }
void EwiseTanh(const MetalArray& a, MetalArray* out) { LaunchUnary(a, out, kTanh); }

void ScalarPower(const MetalArray& a, scalar_t val, MetalArray* out) {
  uint64_t size = out->size;
  Launch1D("scalar_power_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&size length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&val length:sizeof(val) atIndex:3];
  });
}

////////////////////////////////////////////////////////////////////////////////
// Matmul and reductions
////////////////////////////////////////////////////////////////////////////////

void Matmul(const MetalArray& a, const MetalArray& b, MetalArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  MetalContext& ctx = GetContext();
  id<MTLComputePipelineState> pso = ctx.Pipeline("matmul_kernel");
  id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  uint32_t m = M, n = N, p = P;
  [enc setBuffer:a.buffer offset:0 atIndex:0];
  [enc setBuffer:b.buffer offset:0 atIndex:1];
  [enc setBuffer:out->buffer offset:0 atIndex:2];
  [enc setBytes:&m length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&n length:sizeof(uint32_t) atIndex:4];
  [enc setBytes:&p length:sizeof(uint32_t) atIndex:5];

  MTLSize tgs = MTLSizeMake(TILE, TILE, 1);
  uint32_t tg_x = (P + TILE - 1) / TILE;
  uint32_t tg_y = (M + TILE - 1) / TILE;
  MTLSize tgc = MTLSizeMake(tg_x, tg_y, 1);
  [enc dispatchThreadgroups:tgc threadsPerThreadgroup:tgs];
  [enc endEncoding];
  [cb commit];
  // [cb waitUntilCompleted];
}

void ReduceSum(const MetalArray& a, MetalArray* out, size_t reduce_size) {
  uint64_t out_size = out->size;
  uint64_t r = reduce_size;
  Launch1D("reduce_sum_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&out_size length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&r length:sizeof(uint64_t) atIndex:3];
  });
}

void ReduceMax(const MetalArray& a, MetalArray* out, size_t reduce_size) {
  uint64_t out_size = out->size;
  uint64_t r = reduce_size;
  Launch1D("reduce_max_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&out_size length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&r length:sizeof(uint64_t) atIndex:3];
  });
}

}  // namespace metal
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_msl, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace metal;

  m.attr("__device_name__") = "msl";
  m.attr("__tile_size__") = TILE;

  py::class_<MetalArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &MetalArray::size)
      .def("ptr", &MetalArray::ptr_as_int);

  m.def("to_numpy", [](const MetalArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    MetalSync();
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (!host_ptr) throw std::bad_alloc();
    std::memcpy(host_ptr, buffer_ptr(a), a.size * ELEM_SIZE);

    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  m.def("from_numpy", [](py::array_t<scalar_t> a, MetalArray* out) {
    MetalSync();
    std::memcpy(buffer_ptr(*out), a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("synchronize", MetalSync);

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
}
