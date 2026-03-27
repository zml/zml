#include "tools/traceme.h"

#include <cstdlib>
#include <string>

#include "xla/tsl/profiler/backends/cpu/host_tracer_utils.h"
#include "xla/tsl/profiler/backends/cpu/traceme_recorder.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "tsl/profiler/lib/traceme.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

#if defined(ZML_RUNTIME_CUDA)
#include "nvtx3/nvToolsExt.h"
#endif

namespace {

struct LocalHostTraceSession {
  bool active = false;
  uint64_t start_timestamp_ns = 0;
};

LocalHostTraceSession g_local_host_trace_session;

tensorflow::profiler::XPlane* FindOrAddHostPlane(
    tensorflow::profiler::XSpace* space) {
  for (auto& plane : *space->mutable_planes()) {
    if (plane.name() == "/host:CPU") {
      return &plane;
    }
  }
  tensorflow::profiler::XPlane* plane = space->add_planes();
  plane->set_name("/host:CPU");
  return plane;
}

zig_slice CopyToOwnedSlice(const std::string& data) {
  if (data.empty()) {
    return {};
  }
  void* ptr = std::malloc(data.size());
  if (ptr == nullptr) {
    return {};
  }
  std::memcpy(ptr, data.data(), data.size());
  return {ptr, data.size()};
}

#if defined(ZML_RUNTIME_CUDA)
struct CudaNvtxRangeBackend {
  explicit CudaNvtxRangeBackend(const std::string& name)
      : domain(TslNvtxDomain()), active(domain != nullptr) {
    if (!active) {
      return;
    }

    nvtxEventAttributes_t attrs{};
    attrs.version = NVTX_VERSION;
    attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attrs.message.ascii = name.c_str();
    nvtxDomainRangePushEx(domain, &attrs);
  }

  ~CudaNvtxRangeBackend() {
    if (active) {
      nvtxDomainRangePop(domain);
    }
  }

  static bool IsEnabled() { return TslNvtxDomain() != nullptr; }

 private:
  static nvtxDomainHandle_t TslNvtxDomain() {
    static nvtxDomainHandle_t domain = nvtxDomainCreateA("TSL");
    return domain;
  }

  nvtxDomainHandle_t domain = nullptr;
  bool active = false;
};
#else
struct CudaNvtxRangeBackend {
  explicit CudaNvtxRangeBackend(const std::string&) {}
  static bool IsEnabled() { return false; }
};
#endif

}  // namespace

struct zml_traceme {
  explicit zml_traceme(std::string encoded_name)
      : name(std::move(encoded_name)),
        traceme([&] { return name; }),
        cuda_backend(name) {}

  std::string name;
  tsl::profiler::TraceMe traceme;
  CudaNvtxRangeBackend cuda_backend;
};

extern "C" zml_traceme* zml_traceme_start(zig_slice name) {
  return new zml_traceme(
      std::string(static_cast<const char*>(name.ptr), name.len));
}

extern "C" void zml_traceme_stop(zml_traceme* traceme) { delete traceme; }

extern "C" bool zml_traceme_enabled(void) {
  return tsl::profiler::TraceMe::Active() || CudaNvtxRangeBackend::IsEnabled();
}

extern "C" void zml_traceme_session_start(int level, uint64_t filter_mask,
                                          bool enable_filter) {
  g_local_host_trace_session.start_timestamp_ns =
      tsl::profiler::GetCurrentTimeNanos();
  g_local_host_trace_session.active =
      enable_filter ? tsl::profiler::TraceMeRecorder::Start(level, filter_mask)
                    : tsl::profiler::TraceMeRecorder::Start(level);
}

extern "C" zig_slice zml_traceme_session_merge(zig_slice xspace) {
  tensorflow::profiler::XSpace space;
  if (xspace.len != 0 &&
      !space.ParseFromArray(xspace.ptr, static_cast<int>(xspace.len))) {
    return CopyToOwnedSlice(
        std::string(static_cast<const char*>(xspace.ptr), xspace.len));
  }

  if (g_local_host_trace_session.active) {
    auto events = tsl::profiler::TraceMeRecorder::Stop();
    g_local_host_trace_session.active = false;
    if (!events.empty()) {
      tsl::profiler::ConvertCompleteEventsToXPlane(
          g_local_host_trace_session.start_timestamp_ns, std::move(events),
          FindOrAddHostPlane(&space));
    }
  }

  return CopyToOwnedSlice(space.SerializeAsString());
}

extern "C" void zml_traceme_str_free(zig_slice text) { std::free(text.ptr); }
