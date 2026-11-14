// CPU-owned FAISS snapshots and stream-ordered CUDA query/result transfers.
#include <cuda_runtime_api.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>
#include <faiss/utils/utils.h>
#include <omp.h>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
using Clock = std::chrono::steady_clock;
static_assert(sizeof(float) == sizeof(int32_t));

struct Bank {
  std::mutex mutex;
  std::atomic<bool> valid{true};
  std::unique_ptr<faiss::Index> index;
  std::vector<float> keys, values;
  std::vector<int32_t> registry;
  size_t rows = 0, dim = 0;
  int threads = 2;
  bool capture = false, capture_overflow = false;
  size_t capture_limit = 0;
  std::vector<std::vector<float>> captured;
};
struct Context {
  std::mutex mutex;
  std::shared_ptr<Bank> bank;
  float* query = nullptr;
  float* result = nullptr;
  std::vector<float> normalized;
  std::vector<faiss::idx_t> labels;
  size_t count = 0, matches = 0, query_bytes = 0, result_bytes = 0;
  int device = 0;
  bool closed = false, unusable = false;
  cudaEvent_t return_done = nullptr;
  cudaStream_t unfenced_stream = nullptr;
  bool return_pending = false, unfenced = false;
  std::array<cudaEvent_t, 4> transfer_events{};
  bool transfer_timing = false, transfer_pending = false;
  // Successful profiled calls, D2H event seconds, H2D event seconds.
  double transfer_stats[3] = {};
  std::string error;
  // H2D is enqueue-only in event mode; callers must time completed GPU work.
  // calls, D2H, native CPU routine, H2D, CPU normalization/search/KV gather.
  double stats[5] = {};
  ~Context();
};

static std::mutex registry_mutex;
static std::unordered_map<int64_t, std::shared_ptr<Bank>> banks;
static std::unordered_map<int64_t, std::shared_ptr<Context>> contexts;
static int64_t next_id = 1;
static thread_local std::string last_error;

static size_t Product(std::initializer_list<size_t> dimensions) {
  size_t result = 1;
  for (size_t value : dimensions) {
    if (!value || result > std::numeric_limits<size_t>::max() / value)
      throw std::invalid_argument("Invalid or overflowing native buffer dimensions");
    result *= value;
  }
  return result;
}
template <typename T>
static std::shared_ptr<T> Find(
    const std::unordered_map<int64_t, std::shared_ptr<T>>& registry, int64_t id) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto found = registry.find(id);
  return found == registry.end() ? nullptr : found->second;
}
static std::shared_ptr<Bank> RequireBank(int64_t id) {
  auto bank = Find(banks, id);
  if (!bank || !bank->valid.load())
    throw std::invalid_argument("Native bank generation changed, closed or unknown");
  return bank;
}
static int64_t NextId() {
  // Caller holds registry_mutex; handles are never reused.
  if (next_id == std::numeric_limits<int64_t>::max())
    throw std::overflow_error("Native handle registry exhausted");
  return next_id++;
}
static void CudaCheck(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

class DeviceScope {
  int previous;
  bool changed;
 public:
  explicit DeviceScope(int device) {
    CudaCheck(cudaGetDevice(&previous));
    changed = previous != device;
    if (changed) CudaCheck(cudaSetDevice(device));
  }
  ~DeviceScope() { if (changed) cudaSetDevice(previous); }
};

// Caller owns the context mutex and has selected its CUDA device.
static void Drain(Context& context) {
  if (context.unfenced) {
    CudaCheck(cudaStreamSynchronize(context.unfenced_stream));
    context.unfenced = false;
  }
  if (context.return_pending) {
    CudaCheck(cudaEventSynchronize(context.return_done));
    context.return_pending = false;
  }
}

static void CollectTransfers(Context& context) {
  if (!context.transfer_pending) return;
  CudaCheck(cudaEventSynchronize(context.transfer_events[3]));
  float d2h = 0, h2d = 0;
  CudaCheck(cudaEventElapsedTime(&d2h, context.transfer_events[0], context.transfer_events[1]));
  CudaCheck(cudaEventElapsedTime(&h2d, context.transfer_events[2], context.transfer_events[3]));
  if (!std::isfinite(d2h) || !std::isfinite(h2d) || d2h < 0 || h2d < 0)
    throw std::runtime_error("Invalid CUDA transfer event timing");
  context.transfer_stats[0] += 1;
  context.transfer_stats[1] += static_cast<double>(d2h) / 1000;
  context.transfer_stats[2] += static_cast<double>(h2d) / 1000;
  context.transfer_pending = false;
}

Context::~Context() {
  bool allocated = query || result || return_done;
  for (auto event : transfer_events) allocated = allocated || event;
  if (!allocated) return;
  try {
    DeviceScope device_scope(device);
    Drain(*this);
    if (query) cudaFreeHost(query);
    if (result) cudaFreeHost(result);
    if (return_done) cudaEventDestroy(return_done);
    for (auto event : transfer_events) if (event) cudaEventDestroy(event);
  } catch (...) {
    // Retain allocations rather than free a possibly in-flight DMA buffer.
  }
}

// Caller holds bank.mutex. All buffers and the index are CPU-owned.
static double Search(Bank& bank, size_t count, size_t matches, const float* query,
                     float* result, float* normalized, faiss::idx_t* labels) {
  if (!bank.valid.load()) throw std::invalid_argument("Native bank generation changed or closed");
  // Match the previous timer boundary: thread setup is dispatch, not CPU index work.
  omp_set_num_threads(bank.threads);
  auto started = Clock::now();
  for (size_t row = 0; row < count; ++row) {
    const float* input = query + row * bank.dim;
    float* output = normalized + row * bank.dim;
    float sum = 0;
    for (size_t d = 0; d < bank.dim; ++d) {
      if (!std::isfinite(input[d]))
        throw std::invalid_argument("Query vectors must contain only finite values");
      sum += input[d] * input[d];
    }
    float norm = std::sqrt(sum);
    for (size_t d = 0; d < bank.dim; ++d)
      output[d] = norm > 0 ? input[d] / (norm + 1e-8f) : input[d];
  }
  size_t vectors = Product({count, matches, bank.dim});
  size_t metadata = Product({count, matches});
  float* scores = result + 2 * vectors;
  bank.index->search(static_cast<faiss::idx_t>(count), normalized,
                     static_cast<faiss::idx_t>(matches), scores, labels);
  for (size_t i = 0; i < metadata; ++i) {
    faiss::idx_t label = labels[i];
    if (label < 0 || static_cast<size_t>(label) >= bank.rows)
      throw std::runtime_error("FAISS returned an invalid template label");
    std::memcpy(result + i * bank.dim, bank.keys.data() + label * bank.dim, bank.dim * sizeof(float));
    std::memcpy(result + vectors + i * bank.dim, bank.values.data() + label * bank.dim, bank.dim * sizeof(float));
  }
  double seconds = std::chrono::duration<double>(Clock::now() - started).count();
  for (size_t i = 0; i < metadata; ++i) {
    int32_t canonical = bank.registry[labels[i]];
    std::memcpy(result + 2 * vectors + metadata + i, &canonical, sizeof(canonical));
  }
  if (bank.capture) {
    if (bank.captured.size() >= bank.capture_limit) {
      bank.capture_overflow = true;
      throw std::runtime_error("Native diagnostic query capture limit exceeded");
    }
    bank.captured.emplace_back(query, query + count * bank.dim);
  }
  return seconds;
}

extern "C" int NativeAbiVersion() { return 2; }
extern "C" const char* NativeFaissVersion() {
#if FAISS_VERSION_MAJOR > 1 || FAISS_VERSION_MINOR >= 15
  static const std::string version = faiss::get_version();
#else
  static const std::string version = std::to_string(FAISS_VERSION_MAJOR) + "." +
      std::to_string(FAISS_VERSION_MINOR) + "." + std::to_string(FAISS_VERSION_PATCH);
#endif
  return version.c_str();
}
extern "C" const char* NativeFaissCompileOptions() {
  static const std::string options = faiss::get_compile_options();
  return options.c_str();
}
extern "C" const char* NativeLastError() { return last_error.c_str(); }

extern "C" int64_t NativeCreateBank(const uint8_t* serialized, size_t serialized_bytes,
    const float* keys, const float* values, const int32_t* registry,
    size_t rows, size_t dim, int threads) {
  try {
    if (!serialized || !serialized_bytes || !keys || !values || !registry || threads < 1 ||
        !rows || rows > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        !dim || dim > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Invalid native bank arguments");
    size_t elements = Product({rows, dim});
    auto bank = std::make_shared<Bank>();
    faiss::VectorIOReader reader;
    reader.data.assign(serialized, serialized + serialized_bytes);
    bank->index.reset(faiss::read_index(&reader));
    if (!bank->index || reader.rp != serialized_bytes ||
        bank->index->ntotal != static_cast<faiss::idx_t>(rows) ||
        bank->index->d != static_cast<int>(dim) || !bank->index->is_trained ||
        bank->index->metric_type != faiss::METRIC_INNER_PRODUCT ||
        !(dynamic_cast<faiss::IndexFlatIP*>(bank->index.get()) ||
          dynamic_cast<faiss::IndexHNSWFlat*>(bank->index.get())))
      throw std::invalid_argument("Serialized index must be matching FlatIP or HNSWFlat IP");
    bank->keys.assign(keys, keys + elements);
    bank->values.assign(values, values + elements);
    bank->registry.assign(registry, registry + rows);
    std::vector<bool> seen(rows, false);
    for (int32_t id : bank->registry) {
      if (id < 0 || static_cast<size_t>(id) >= rows || seen[id])
        throw std::invalid_argument("Registry mapping must be a permutation");
      seen[id] = true;
    }
    for (size_t i = 0; i < elements; ++i)
      if (!std::isfinite(keys[i]) || !std::isfinite(values[i]))
        throw std::invalid_argument("Native bank vectors must be finite");
    bank->rows = rows; bank->dim = dim; bank->threads = threads;
    std::lock_guard<std::mutex> lock(registry_mutex);
    int64_t id = NextId();
    banks.emplace(id, std::move(bank));
    return id;
  } catch (const std::exception& error) { last_error = error.what(); return -1; }
    catch (...) { last_error = "Unknown native bank creation failure"; return -1; }
}
extern "C" int NativeInvalidateBank(int64_t id) {
  auto bank = Find(banks, id);
  if (!bank) return -1;
  bank->valid.store(false);
  return 0;
}
extern "C" int NativeCloseBank(int64_t id) {
  std::shared_ptr<Bank> bank;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto found = banks.find(id);
    if (found == banks.end()) return 0;
    bank = found->second;
    bank->valid.store(false);
    banks.erase(found);
  }
  std::lock_guard<std::mutex> lock(bank->mutex);
  return 0;
}
static int64_t CreateContext(int64_t bank_id, size_t count, size_t matches,
                             int device, int event_return) {
  try {
    auto bank = RequireBank(bank_id);
    if (!count || count > static_cast<size_t>(std::numeric_limits<faiss::idx_t>::max()) ||
        !matches || matches > bank->rows || device < 0 ||
        (event_return != 0 && event_return != 1))
      throw std::invalid_argument("Invalid native context arguments");
    auto context = std::make_shared<Context>();
    context->bank = bank; context->count = count; context->matches = matches; context->device = device;
    context->query_bytes = Product({count, bank->dim, sizeof(float)});
    context->result_bytes = Product({count, matches, 2 * bank->dim + 2, sizeof(float)});
    context->normalized.resize(Product({count, bank->dim}));
    context->labels.resize(Product({count, matches}));
    DeviceScope device_scope(device);
    CudaCheck(cudaMallocHost(reinterpret_cast<void**>(&context->query), context->query_bytes));
    CudaCheck(cudaMallocHost(reinterpret_cast<void**>(&context->result), context->result_bytes));
    if (event_return)
      CudaCheck(cudaEventCreateWithFlags(&context->return_done, cudaEventDisableTiming));
    std::lock_guard<std::mutex> lock(registry_mutex);
    int64_t id = NextId();
    contexts.emplace(id, std::move(context));
    return id;
  } catch (const std::exception& error) { last_error = error.what(); return -1; }
    catch (...) { last_error = "Unknown native context creation failure"; return -1; }
}
extern "C" int64_t NativeCreate(int64_t bank_id, size_t count, size_t matches, int device) {
  return CreateContext(bank_id, count, matches, device, 0);
}
extern "C" int64_t NativeCreateWithReturnMode(
    int64_t bank_id, size_t count, size_t matches, int device, int event_return) {
  return CreateContext(bank_id, count, matches, device, event_return);
}
extern "C" int NativeClose(int64_t id) {
  auto context = Find(contexts, id);
  if (!context) return 0;
  std::lock_guard<std::mutex> lock(context->mutex);
  if (context->closed) return 0;
  try {
    DeviceScope device_scope(context->device);
    Drain(*context);
    CudaCheck(cudaFreeHost(context->query));
    context->query = nullptr;
    CudaCheck(cudaFreeHost(context->result));
    context->result = nullptr;
    if (context->return_done) {
      CudaCheck(cudaEventDestroy(context->return_done));
      context->return_done = nullptr;
    }
    for (auto& event : context->transfer_events) {
      if (event) { CudaCheck(cudaEventDestroy(event)); event = nullptr; }
    }
    context->closed = true;
    std::lock_guard<std::mutex> registry_lock(registry_mutex);
    contexts.erase(id);
    return 0;
  } catch (const std::exception& error) {
    context->unusable = true;
    context->error = last_error = error.what();
    return -1;
  }
}
extern "C" int NativeStats(int64_t id, double* output, int reset) {
  auto context = Find(contexts, id);
  if (!context || !output) return -1;
  std::lock_guard<std::mutex> lock(context->mutex);
  if (context->closed) return -1;
  std::memcpy(output, context->stats, sizeof(context->stats));
  if (reset) std::memset(context->stats, 0, sizeof(context->stats));
  return 0;
}

extern "C" int NativeSetTransferTiming(int64_t id, int enabled) {
  auto context = Find(contexts, id);
  if (!context || (enabled != 0 && enabled != 1)) return -1;
  std::lock_guard<std::mutex> lock(context->mutex);
  try {
    if (context->closed || context->unusable)
      throw std::invalid_argument("Transfer timing context is closed or unusable");
    DeviceScope device_scope(context->device);
    Drain(*context);
    CollectTransfers(*context);
    if (enabled) {
      for (auto& event : context->transfer_events)
        if (!event) CudaCheck(cudaEventCreateWithFlags(&event, cudaEventDefault));
      std::memset(context->transfer_stats, 0, sizeof(context->transfer_stats));
    }
    context->transfer_timing = enabled != 0;
    return 0;
  } catch (const std::exception& error) {
    context->error = last_error = error.what();
    return -1;
  }
}

extern "C" int NativeTransferStats(int64_t id, double* output, int reset) {
  auto context = Find(contexts, id);
  if (!context || !output) return -1;
  std::lock_guard<std::mutex> lock(context->mutex);
  try {
    if (context->closed || context->unusable)
      throw std::invalid_argument("Transfer timing context is closed or unusable");
    DeviceScope device_scope(context->device);
    Drain(*context);
    CollectTransfers(*context);
    std::memcpy(output, context->transfer_stats, sizeof(context->transfer_stats));
    if (reset) std::memset(context->transfer_stats, 0, sizeof(context->transfer_stats));
    return 0;
  } catch (const std::exception& error) {
    context->error = last_error = error.what();
    return -1;
  }
}
extern "C" int NativeContextError(int64_t id, char* output, size_t size) {
  auto context = Find(contexts, id);
  if (!context || !output || !size) return -1;
  std::lock_guard<std::mutex> lock(context->mutex);
  if (context->error.size() >= size) return -1;
  std::memcpy(output, context->error.c_str(), context->error.size() + 1);
  return 0;
}

// Host-only parity entry; the GPU FFI below uses Context's reusable scratch.
extern "C" int NativeSearchHost(int64_t id, size_t count, size_t matches,
    const float* query, float* output, double* seconds) {
  try {
    auto bank = RequireBank(id);
    if (!count || count > static_cast<size_t>(std::numeric_limits<faiss::idx_t>::max()) ||
        !matches || matches > bank->rows || !query || !output || !seconds)
      throw std::invalid_argument("Invalid host search arguments");
    Product({count, matches, 2 * bank->dim + 2, sizeof(float)});
    std::vector<float> normalized(Product({count, bank->dim}));
    std::vector<faiss::idx_t> labels(Product({count, matches}));
    std::lock_guard<std::mutex> lock(bank->mutex);
    *seconds = Search(*bank, count, matches, query, output, normalized.data(), labels.data());
    return 0;
  } catch (const std::exception& error) { last_error = error.what(); return -1; }
    catch (...) { last_error = "Unknown native CPU search failure"; return -1; }
}
extern "C" int NativeCaptureStart(int64_t id, size_t limit) {
  try {
    auto bank = RequireBank(id);
    std::lock_guard<std::mutex> lock(bank->mutex);
    if (!limit || bank->capture) throw std::invalid_argument("Invalid or nested query capture");
    bank->captured.clear(); bank->capture = true;
    bank->capture_overflow = false; bank->capture_limit = limit;
    return 0;
  } catch (const std::exception& error) { last_error = error.what(); return -1; }
}
extern "C" int64_t NativeCaptureStop(int64_t id) {
  auto bank = Find(banks, id);
  if (!bank) return -1;
  std::lock_guard<std::mutex> lock(bank->mutex);
  bank->capture = false;
  return bank->capture_overflow ? -1 : static_cast<int64_t>(bank->captured.size());
}
extern "C" int64_t NativeCaptureSize(int64_t id, size_t index) {
  auto bank = Find(banks, id);
  if (!bank) return -1;
  std::lock_guard<std::mutex> lock(bank->mutex);
  if (bank->capture || index >= bank->captured.size()) return -1;
  return static_cast<int64_t>(bank->captured[index].size());
}
extern "C" int NativeCaptureRead(int64_t id, size_t index, float* output, size_t elements) {
  auto bank = Find(banks, id);
  if (!bank || !output) return -1;
  std::lock_guard<std::mutex> lock(bank->mutex);
  if (bank->capture || index >= bank->captured.size() || bank->captured[index].size() != elements) return -1;
  std::memcpy(output, bank->captured[index].data(), elements * sizeof(float));
  return 0;
}

struct Output {
  void* data;
  size_t bytes;
};

static ffi::Error RetrieveInto(cudaStream_t stream, int64_t id,
                              ffi::Buffer<ffi::F32> query,
                              std::initializer_list<Output> outputs) {
  auto context = Find(contexts, id);
  if (!context) return ffi::Error::InvalidArgument("Native retrieval context is closed or unknown");
  std::lock_guard<std::mutex> lock(context->mutex);
  try {
    if (context->closed || context->unusable ||
        query.element_count() * sizeof(float) != context->query_bytes)
      throw std::invalid_argument("Native retrieval context/buffer mismatch");
    DeviceScope device_scope(context->device);
    Drain(*context);
    CollectTransfers(*context);
    size_t vectors = Product({context->count, context->matches, context->bank->dim, sizeof(float)});
    size_t metadata = Product({context->count, context->matches, sizeof(float)});
    const size_t expected[] = {vectors, vectors, metadata, metadata};
    if (outputs.size() != 1 && outputs.size() != 4)
      throw std::invalid_argument("Invalid native output count");
    size_t index = 0;
    for (const auto& output : outputs) {
      size_t bytes = outputs.size() == 1 ? context->result_bytes : expected[index++];
      if (!output.data || output.bytes != bytes)
        throw std::invalid_argument("Native retrieval output/buffer mismatch");
    }
    std::lock_guard<std::mutex> bank_lock(context->bank->mutex);
    if (!context->bank->valid.load())
      throw std::invalid_argument("Native bank generation changed or closed");
    cudaPointerAttributes attributes;
    CudaCheck(cudaPointerGetAttributes(&attributes, query.typed_data()));
    if (attributes.device != context->device)
      throw std::invalid_argument("Native retrieval query is on the wrong CUDA device");
    auto started = Clock::now();
    context->unfenced_stream = stream;
    context->unfenced = true;
    if (context->transfer_timing)
      CudaCheck(cudaEventRecord(context->transfer_events[0], stream));
    CudaCheck(cudaMemcpyAsync(context->query, query.typed_data(), context->query_bytes,
                             cudaMemcpyDeviceToHost, stream));
    if (context->transfer_timing)
      CudaCheck(cudaEventRecord(context->transfer_events[1], stream));
    CudaCheck(cudaStreamSynchronize(stream));
    context->unfenced = false;
    auto copied = Clock::now();
    double index_seconds = Search(*context->bank, context->count, context->matches,
        context->query, context->result, context->normalized.data(), context->labels.data());
    auto searched = Clock::now();
    const char* source = reinterpret_cast<const char*>(context->result);
    context->unfenced_stream = stream;
    context->unfenced = true;
    if (context->transfer_timing)
      CudaCheck(cudaEventRecord(context->transfer_events[2], stream));
    for (const auto& output : outputs) {
      CudaCheck(cudaMemcpyAsync(output.data, source, output.bytes,
                               cudaMemcpyHostToDevice, stream));
      source += output.bytes;
    }
    if (context->transfer_timing)
      CudaCheck(cudaEventRecord(context->transfer_events[3], stream));
    if (context->return_done) {
      // GPU consumers follow these copies on XLA's stream. Fence host reuse only.
      CudaCheck(cudaEventRecord(context->return_done, stream));
      context->return_pending = true;
    } else {
      CudaCheck(cudaStreamSynchronize(stream));
    }
    context->unfenced = false;
    context->transfer_pending = context->transfer_timing;
    auto done = Clock::now();
    context->error.clear();
    context->stats[0] += 1;
    context->stats[1] += std::chrono::duration<double>(copied - started).count();
    context->stats[2] += std::chrono::duration<double>(searched - copied).count();
    context->stats[3] += std::chrono::duration<double>(done - searched).count();
    context->stats[4] += index_seconds;
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    context->error = error.what();
    try {
      DeviceScope device_scope(context->device);
      Drain(*context);
    } catch (const std::exception& cleanup) {
      context->unusable = true;
      context->error += std::string("; CUDA drain failed: ") + cleanup.what();
    }
    return ffi::Error::Internal(context->error);
  } catch (...) {
    context->error = "Unknown native retrieval failure";
    try {
      DeviceScope device_scope(context->device);
      Drain(*context);
    } catch (...) {
      context->unusable = true;
      context->error += "; CUDA drain failed";
    }
    return ffi::Error::Internal(context->error);
  }
}

ffi::Error Retrieve(cudaStream_t stream, int64_t id, ffi::Buffer<ffi::F32> query,
                    ffi::ResultBuffer<ffi::F32> output) {
  return RetrieveInto(stream, id, query,
                      {{output->typed_data(), output->element_count() * sizeof(float)}});
}

ffi::Error RetrieveDirect(cudaStream_t stream, int64_t id, ffi::Buffer<ffi::F32> query,
    ffi::ResultBuffer<ffi::F32> keys, ffi::ResultBuffer<ffi::F32> values,
    ffi::ResultBuffer<ffi::F32> scores, ffi::ResultBuffer<ffi::S32> indices) {
  return RetrieveInto(stream, id, query, {
      {keys->typed_data(), keys->element_count() * sizeof(float)},
      {values->typed_data(), values->element_count() * sizeof(float)},
      {scores->typed_data(), scores->element_count() * sizeof(float)},
      {indices->typed_data(), indices->element_count() * sizeof(int32_t)}});
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NsllmNativeRetrieveV2, Retrieve,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("context_id")
        .Arg<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NsllmNativeRetrieveDirectV2, RetrieveDirect,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("context_id").Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>().Ret<ffi::Buffer<ffi::S32>>());
