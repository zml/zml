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

struct TraceMeSession {
  bool active = false;
  uint64_t start_timestamp_ns = 0;
};

TraceMeSession g_traceme_session;

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

}  // namespace

struct zml_traceme {
  explicit zml_traceme(std::string encoded_name)
      : name(std::move(encoded_name)),
        traceme([&] { return name; }) {
    PushCudaRange();
  }

  ~zml_traceme() {
    PopCudaRange();
  }

  std::string name;
  tsl::profiler::TraceMe traceme;

#if defined(ZML_RUNTIME_CUDA)
  static nvtxDomainHandle_t TslNvtxDomain() {
    static nvtxDomainHandle_t domain = nvtxDomainCreateA("TSL");
    return domain;
  }

  void PushCudaRange() {
    nvtx_domain = TslNvtxDomain();
    pushed_nvtx_range = nvtx_domain != nullptr;
    if (!pushed_nvtx_range) {
      return;
    }

    nvtxEventAttributes_t attrs{};
    attrs.version = NVTX_VERSION;
    attrs.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attrs.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attrs.message.ascii = name.c_str();
    nvtxDomainRangePushEx(nvtx_domain, &attrs);
  }

  void PopCudaRange() {
    if (pushed_nvtx_range) {
      nvtxDomainRangePop(nvtx_domain);
    }
  }

  nvtxDomainHandle_t nvtx_domain = nullptr;
  bool pushed_nvtx_range = false;
#else
  void PushCudaRange() {}
  void PopCudaRange() {}
#endif
};

extern "C" zml_traceme* zml_traceme_start(zig_slice name) {
  return new zml_traceme(
      std::string(static_cast<const char*>(name.ptr), name.len));
}

extern "C" void zml_traceme_stop(zml_traceme* traceme) { delete traceme; }

extern "C" bool zml_traceme_enabled(void) {
  return tsl::profiler::TraceMe::Active()
#if defined(ZML_RUNTIME_CUDA)
         || zml_traceme::TslNvtxDomain() != nullptr
#endif
      ;
}

extern "C" void zml_traceme_session_start(int level, uint64_t filter_mask,
                                          bool enable_filter) {
  g_traceme_session.start_timestamp_ns = tsl::profiler::GetCurrentTimeNanos();
  g_traceme_session.active =
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

  if (g_traceme_session.active) {
    auto events = tsl::profiler::TraceMeRecorder::Stop();
    g_traceme_session.active = false;
    if (!events.empty()) {
      tsl::profiler::ConvertCompleteEventsToXPlane(
          g_traceme_session.start_timestamp_ns, std::move(events),
          FindOrAddHostPlane(&space));
    }
  }

  return CopyToOwnedSlice(space.SerializeAsString());
}

extern "C" void zml_traceme_str_free(zig_slice text) { std::free(text.ptr); }
