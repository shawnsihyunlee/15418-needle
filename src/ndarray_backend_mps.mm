#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace needle {
namespace mps {

#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);
constexpr uint32_t MAX_VEC_SIZE = 8;

struct MetalVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

// =================================================================================
// METAL SHADERS (Identical to MSL backend for Ewise ops)
// =================================================================================
static const char* kMetalShaderSrc = R"METAL(
using namespace metal;
constant uint MAX_VEC_SIZE = 8;
struct IntVec { uint size; int data[MAX_VEC_SIZE]; };

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

kernel void fill_kernel(device float* out [[buffer(0)]], constant float& val [[buffer(1)]], constant ulong& size [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < size) out[gid] = val;
}

kernel void compact_kernel(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant ulong& size [[buffer(2)]], constant IntVec& shape [[buffer(3)]], constant IntVec& strides [[buffer(4)]], constant ulong& offset [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  ulong in_index = gid_to_strided_index(gid, shape, strides, offset);
  out[gid] = a[in_index];
}

kernel void ewise_setitem_kernel(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant ulong& size [[buffer(2)]], constant IntVec& shape [[buffer(3)]], constant IntVec& strides [[buffer(4)]], constant ulong& offset [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  ulong out_index = gid_to_strided_index(gid, shape, strides, offset);
  out[out_index] = a[gid];
}

kernel void scalar_setitem_kernel(const constant float& val [[buffer(0)]], device float* out [[buffer(1)]], constant ulong& size [[buffer(2)]], constant IntVec& shape [[buffer(3)]], constant IntVec& strides [[buffer(4)]], constant ulong& offset [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  ulong out_index = gid_to_strided_index(gid, shape, strides, offset);
  out[out_index] = val;
}

kernel void ewise_binary_kernel(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]], device float* out [[buffer(2)]], constant ulong& size [[buffer(3)]], constant uint& op [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  float av = a[gid]; float bv = b[gid]; float res = 0.0f;
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

kernel void scalar_binary_kernel(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant ulong& size [[buffer(2)]], constant float& val [[buffer(3)]], constant uint& op [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  float av = a[gid]; float res = 0.0f;
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

kernel void unary_kernel(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant ulong& size [[buffer(2)]], constant uint& op [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  float av = a[gid]; float res = 0.0f;
  switch (op) {
    case 0: res = log(av); break;
    case 1: res = exp(av); break;
    case 2: res = tanh(av); break;
    default: res = 0.0f; break;
  }
  out[gid] = res;
}

kernel void scalar_power_kernel(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant ulong& size [[buffer(2)]], constant float& val [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= size) return;
  out[gid] = pow(a[gid], val);
}

kernel void reduce_sum_kernel(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant ulong& out_size [[buffer(2)]], constant ulong& reduce_size [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= out_size) return;
  ulong base = gid * reduce_size;
  float total = 0.0f;
  for (ulong i = 0; i < reduce_size; ++i) total += a[base + i];
  out[gid] = total;
}

kernel void reduce_max_kernel(const device float* a [[buffer(0)]], device float* out [[buffer(1)]], constant ulong& out_size [[buffer(2)]], constant ulong& reduce_size [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= out_size) return;
  ulong base = gid * reduce_size;
  float max_val = a[base];
  for (ulong i = 1; i < reduce_size; ++i) { float v = a[base + i]; max_val = v > max_val ? v : max_val; }
  out[gid] = max_val;
}
)METAL";

// =================================================================================
// MPS CONTEXT & HELPERS
// =================================================================================

struct MPSContext {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id<MTLLibrary> library;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

  MPSContext() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) throw std::runtime_error("Metal device not available");
    queue = [device newCommandQueue];
    if (!queue) throw std::runtime_error("Failed to create Metal command queue");

    NSError* err = nil;
    NSString* source = [[NSString alloc] initWithUTF8String:kMetalShaderSrc];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    library = [device newLibraryWithSource:source options:opts error:&err];
#if !__has_feature(objc_arc)
    [source release];
    [opts release];
#endif
    if (!library || err) {
      std::string msg = "Metal library compile failed: ";
      if (err) msg += [[err localizedDescription] UTF8String];
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
    if (!pso || err) throw std::runtime_error("Failed to create pipeline for " + name);
    pipelines[name] = pso;
    return pso;
  }
};

static MPSContext& GetContext() {
  static MPSContext ctx;
  return ctx;
}

struct MPSArray {
  MPSArray(const size_t size) {
    MPSContext& ctx = GetContext();
    MTLResourceOptions opts = MTLResourceStorageModeShared;
    buffer = [ctx.device newBufferWithLength:size * ELEM_SIZE options:opts];
    if (!buffer) throw std::runtime_error("Failed to allocate MPS buffer");
    this->size = size;
  }
  ~MPSArray() {
#if !__has_feature(objc_arc)
    [buffer release];
#endif
  }
  size_t ptr_as_int() { return reinterpret_cast<size_t>([buffer contents]); }
  id<MTLBuffer> buffer;
  size_t size;
};

static inline scalar_t* buffer_ptr(const MPSArray& arr) {
  return reinterpret_cast<scalar_t*>(([arr.buffer contents]));
}

static MetalVec VecToMetal(const std::vector<int32_t>& x) {
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded Metal supported max dimensions");
  MetalVec v{};
  v.size = static_cast<uint32_t>(x.size());
  for (size_t i = 0; i < x.size(); i++) v.data[i] = x[i];
  return v;
}

// ---------------------------------------------------------------------------------
// ASYNC EXECUTION HELPERS
// ---------------------------------------------------------------------------------

// Explicit Synchronization used by Python for timing/debugging
static void MPSSync() {
  MPSContext& ctx = GetContext();
  id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
  [cb commit];
  [cb waitUntilCompleted];
}

template <typename F>
static void Launch1D(const std::string& kernel, size_t size, F bind_buffers) {
  MPSContext& ctx = GetContext();
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
  // ASYNC: Removed [cb waitUntilCompleted];
}

// =================================================================================
// OPERATIONS
// =================================================================================

void Fill(MPSArray* out, scalar_t val) {
  uint64_t size = out->size;
  Launch1D("fill_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:out->buffer offset:0 atIndex:0];
    [enc setBytes:&val length:sizeof(val) atIndex:1];
    [enc setBytes:&size length:sizeof(uint64_t) atIndex:2];
  });
}

void Compact(const MPSArray& a, MPSArray* out, std::vector<int32_t> shape, std::vector<int32_t> strides, size_t offset) {
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

void EwiseSetitem(const MPSArray& a, MPSArray* out, std::vector<int32_t> shape, std::vector<int32_t> strides, size_t offset) {
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

void ScalarSetitem(size_t size, scalar_t val, MPSArray* out, std::vector<int32_t> shape, std::vector<int32_t> strides, size_t offset) {
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

static void LaunchBinary(const MPSArray& a, const MPSArray& b, MPSArray* out, BinaryOp op) {
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

static void LaunchScalarBinary(const MPSArray& a, scalar_t val, MPSArray* out, BinaryOp op) {
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

void EwiseAdd(const MPSArray& a, const MPSArray& b, MPSArray* out) { LaunchBinary(a, b, out, kAdd); }
void ScalarAdd(const MPSArray& a, scalar_t val, MPSArray* out) { LaunchScalarBinary(a, val, out, kAdd); }
void EwiseMul(const MPSArray& a, const MPSArray& b, MPSArray* out) { LaunchBinary(a, b, out, kMul); }
void ScalarMul(const MPSArray& a, scalar_t val, MPSArray* out) { LaunchScalarBinary(a, val, out, kMul); }
void EwiseDiv(const MPSArray& a, const MPSArray& b, MPSArray* out) { LaunchBinary(a, b, out, kDiv); }
void ScalarDiv(const MPSArray& a, scalar_t val, MPSArray* out) { LaunchScalarBinary(a, val, out, kDiv); }
void EwiseEq(const MPSArray& a, const MPSArray& b, MPSArray* out) { LaunchBinary(a, b, out, kEq); }
void ScalarEq(const MPSArray& a, scalar_t val, MPSArray* out) { LaunchScalarBinary(a, val, out, kEq); }
void EwiseGe(const MPSArray& a, const MPSArray& b, MPSArray* out) { LaunchBinary(a, b, out, kGe); }
void ScalarGe(const MPSArray& a, scalar_t val, MPSArray* out) { LaunchScalarBinary(a, val, out, kGe); }
void EwiseMaximum(const MPSArray& a, const MPSArray& b, MPSArray* out) { LaunchBinary(a, b, out, kMax); }
void ScalarMaximum(const MPSArray& a, scalar_t val, MPSArray* out) { LaunchScalarBinary(a, val, out, kMax); }

static void LaunchUnary(const MPSArray& a, MPSArray* out, UnaryOp op) {
  uint64_t size = out->size;
  uint32_t op_code = op;
  Launch1D("unary_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&size length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&op_code length:sizeof(uint32_t) atIndex:3];
  });
}

void EwiseLog(const MPSArray& a, MPSArray* out) { LaunchUnary(a, out, kLog); }
void EwiseExp(const MPSArray& a, MPSArray* out) { LaunchUnary(a, out, kExp); }
void EwiseTanh(const MPSArray& a, MPSArray* out) { LaunchUnary(a, out, kTanh); }

void ScalarPower(const MPSArray& a, scalar_t val, MPSArray* out) {
  uint64_t size = out->size;
  Launch1D("scalar_power_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&size length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&val length:sizeof(val) atIndex:3];
  });
}

void ReduceSum(const MPSArray& a, MPSArray* out, size_t reduce_size) {
  uint64_t out_size = out->size;
  uint64_t r = reduce_size;
  Launch1D("reduce_sum_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&out_size length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&r length:sizeof(uint64_t) atIndex:3];
  });
}

void ReduceMax(const MPSArray& a, MPSArray* out, size_t reduce_size) {
  uint64_t out_size = out->size;
  uint64_t r = reduce_size;
  Launch1D("reduce_max_kernel", out->size, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setBuffer:a.buffer offset:0 atIndex:0];
    [enc setBuffer:out->buffer offset:0 atIndex:1];
    [enc setBytes:&out_size length:sizeof(uint64_t) atIndex:2];
    [enc setBytes:&r length:sizeof(uint64_t) atIndex:3];
  });
}

void Matmul(const MPSArray& a, const MPSArray& b, MPSArray* out, uint32_t M, uint32_t N, uint32_t P) {
  MPSContext& ctx = GetContext();
  id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];

  MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:P rowBytes:P * sizeof(float) dataType:MPSDataTypeFloat32];
  MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:P rowBytes:P * sizeof(float) dataType:MPSDataTypeFloat32];

  MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:a.buffer descriptor:descA];
  MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:b.buffer descriptor:descB];
  MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:out->buffer descriptor:descC];

  MPSMatrixMultiplication* kernel = [[MPSMatrixMultiplication alloc] initWithDevice:ctx.device transposeLeft:false transposeRight:false resultRows:M resultColumns:P interiorColumns:N alpha:1.0f beta:0.0f];

  [kernel encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matB resultMatrix:matC];

  [cb commit];
  // ASYNC: Removed [cb waitUntilCompleted];

#if !__has_feature(objc_arc)
  [matA release];
  [matB release];
  [matC release];
  [kernel release];
#endif
}

}  // namespace mps
}  // namespace needle

// =================================================================================
// PYTHON BINDINGS
// =================================================================================

PYBIND11_MODULE(ndarray_backend_mps, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace mps;

  m.attr("__device_name__") = "mps";
  m.attr("__tile_size__") = TILE;

  py::class_<MPSArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &MPSArray::size)
      .def("ptr", &MPSArray::ptr_as_int);

  m.def("to_numpy", [](const MPSArray& a, std::vector<size_t> shape, std::vector<size_t> strides, size_t offset) {
    MPSSync(); // Sync before reading
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(), [](size_t& c) { return c * ELEM_SIZE; });
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (!host_ptr) throw std::bad_alloc();
    std::memcpy(host_ptr, buffer_ptr(a), a.size * ELEM_SIZE);
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  m.def("from_numpy", [](py::array_t<scalar_t> a, MPSArray* out) {
    MPSSync(); // Sync to ensure buffer is ready? Mostly for safety.
    std::memcpy(buffer_ptr(*out), a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("synchronize", MPSSync); // EXPOSE TO PYTHON

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