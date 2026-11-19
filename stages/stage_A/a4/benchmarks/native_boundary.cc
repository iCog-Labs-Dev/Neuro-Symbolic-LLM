// Stream-ordered GPU retrieval boundary with reusable page-locked host buffers.
#include <cuda_runtime_api.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using Clock = std::chrono::steady_clock;
using CpuQuery = int (*)(float*, float*, double*);
static std::mutex mutex;
static float* host_query = nullptr;
static float* host_result = nullptr;
static std::size_t query_bytes = 0;
static std::size_t result_bytes = 0;
static CpuQuery cpu_query = nullptr;
static double last_stats[4] = {};

static double seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

extern "C" int Configure(std::size_t input_bytes, std::size_t output_bytes,
                         CpuQuery callback) {
  std::lock_guard<std::mutex> lock(mutex);
  if (host_query || host_result || !callback || !input_bytes || !output_bytes) return -1;
  cudaError_t error = cudaMallocHost(reinterpret_cast<void**>(&host_query), input_bytes);
  if (error != cudaSuccess) return static_cast<int>(error);
  error = cudaMallocHost(reinterpret_cast<void**>(&host_result), output_bytes);
  if (error != cudaSuccess) {
    cudaFreeHost(host_query); host_query = nullptr;
    return static_cast<int>(error);
  }
  query_bytes = input_bytes; result_bytes = output_bytes; cpu_query = callback;
  return 0;
}

extern "C" void LastStats(double* output) {
  std::lock_guard<std::mutex> lock(mutex);
  std::memcpy(output, last_stats, sizeof(last_stats));
}

extern "C" int ReleaseBuffers() {
  std::lock_guard<std::mutex> lock(mutex);
  cudaError_t first = host_query ? cudaFreeHost(host_query) : cudaSuccess;
  cudaError_t second = host_result ? cudaFreeHost(host_result) : cudaSuccess;
  host_query = nullptr; host_result = nullptr; cpu_query = nullptr;
  query_bytes = 0; result_bytes = 0;
  return static_cast<int>(first != cudaSuccess ? first : second);
}

ffi::Error RetrieveImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> query,
                       ffi::ResultBuffer<ffi::F32> output) {
  std::lock_guard<std::mutex> lock(mutex);
  if (!cpu_query || query.element_count() * sizeof(float) != query_bytes ||
      output->element_count() * sizeof(float) != result_bytes) {
    return ffi::Error::InvalidArgument("Pinned retrieval buffer shape mismatch");
  }
  auto start = Clock::now();
  cudaError_t error = cudaMemcpyAsync(host_query, query.typed_data(), query_bytes,
                                     cudaMemcpyDeviceToHost, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
  auto copied = Clock::now();
  double index_seconds = 0;
  if (cpu_query(host_query, host_result, &index_seconds) != 0) {
    return ffi::Error::Internal("CPU retrieval callback failed");
  }
  auto searched = Clock::now();
  error = cudaMemcpyAsync(output->typed_data(), host_result, result_bytes,
                          cudaMemcpyHostToDevice, stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
  if (error != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(error));
  auto done = Clock::now();
  last_stats[0] = seconds(start, copied);
  last_stats[1] = seconds(copied, searched);
  last_stats[2] = seconds(searched, done);
  last_stats[3] = index_seconds;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NativeRetrieve, RetrieveImpl,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());
