#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "json/json.h"
#include "tools/xspace_to_perfetto.h"
#include "xla/tsl/profiler/utils/format_utils.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace {

bool ParseXSpace(std::string_view xspace_proto,
                 tensorflow::profiler::XSpace* xspace,
                 std::string* error) {
  if (xspace_proto.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    *error = "XSpace protobuf is too large to parse";
    return false;
  }
  if (!xspace->ParseFromArray(xspace_proto.data(),
                              static_cast<int>(xspace_proto.size()))) {
    *error = "Failed to parse XSpace protobuf";
    return false;
  }
  return true;
}

inline std::string PicosToMicrosString(uint64_t ps) {
  return tsl::profiler::MaxPrecision(tsl::profiler::PicoToMicro(ps));
}

inline std::string JsonString(std::string_view s) {
  const std::string owned(s);
  return Json::valueToQuotedString(owned.c_str());
}

class JsonArrayWriter {
 public:
  explicit JsonArrayWriter(std::ostream* output) : output_(output) {}

  bool NextItem() {
    if (needs_comma_) {
      *output_ << ',';
    }
    needs_comma_ = true;
    return output_->good();
  }

  bool WriteItemSequence(std::string_view items) {
    if (items.empty()) {
      return output_->good();
    }
    if (needs_comma_) {
      *output_ << ',';
    }
    output_->write(items.data(), static_cast<std::streamsize>(items.size()));
    needs_comma_ = true;
    return output_->good();
  }

 private:
  std::ostream* output_;
  bool needs_comma_ = false;
};

struct PlaneWorkItem {
  uint32_t device_id;
  tsl::profiler::XPlaneVisitor visitor;
};

bool StreamResourceMetadata(uint32_t device_id,
                            uint32_t resource_id,
                            std::string_view resource_name,
                            uint32_t sort_index,
                            std::ostream* output,
                            JsonArrayWriter* writer) {
  if (!resource_name.empty()) {
    writer->NextItem();
    *output << R"({"ph":"M","pid":)" << device_id << R"(,"tid":)"
            << resource_id << R"(,"name":"thread_name","args":{"name":)"
            << JsonString(resource_name) << "}}";
  }

  writer->NextItem();
  *output << R"({"ph":"M","pid":)" << device_id << R"(,"tid":)"
          << resource_id
          << R"(,"name":"thread_sort_index","args":{"sort_index":)"
          << sort_index << "}}";
  return output->good();
}

bool StreamPlaneMetadata(uint32_t device_id,
                         const tsl::profiler::XPlaneVisitor& plane,
                         std::ostream* output,
                         JsonArrayWriter* writer) {
  if (!plane.Name().empty()) {
    writer->NextItem();
    *output << R"({"ph":"M","pid":)" << device_id
            << R"(,"name":"process_name","args":{"name":)"
            << JsonString(plane.Name()) << "}}";
  }

  writer->NextItem();
  *output << R"({"ph":"M","pid":)" << device_id
          << R"(,"name":"process_sort_index","args":{"sort_index":)"
          << device_id << "}}";

  const bool sort_by_ordinal = (device_id == tsl::profiler::kHostThreadsDeviceId);
  uint32_t ordinal = 0;
  plane.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    const uint32_t sort_index = sort_by_ordinal ? ++ordinal : line.DisplayId();
    StreamResourceMetadata(device_id, line.DisplayId(), line.DisplayName(),
                           sort_index, output, writer);
  });
  return output->good();
}

bool StreamEvent(uint32_t device_id,
                 uint32_t resource_id,
                 const tsl::profiler::XEventVisitor& xevent,
                 std::ostream* output,
                 JsonArrayWriter* writer) {
  std::string event_name;
  std::map<std::string, std::string> args;
  if (xevent.HasDisplayName()) {
    event_name = std::string(xevent.DisplayName());
    args["long_name"] = std::string(xevent.Name());
  } else {
    event_name = std::string(xevent.Name());
  }

  auto for_each_stat = [&](const tsl::profiler::XStatVisitor& stat) {
    if (stat.ValueCase() == tensorflow::profiler::XStat::VALUE_NOT_SET) {
      return;
    }
    if (tsl::profiler::IsInternalStat(stat.Type())) {
      return;
    }
    std::string value = stat.ToString();
    if (stat.Type() == tsl::profiler::StatType::kStepName) {
      event_name = value;
    }
    args[std::string(stat.Name())] = std::move(value);
  };
  xevent.Metadata().ForEachStat(for_each_stat);
  xevent.ForEachStat(for_each_stat);

  writer->NextItem();
  *output << R"({"ph":"X","pid":)" << device_id << R"(,"tid":)"
          << resource_id << R"(,"ts":)"
          << PicosToMicrosString(xevent.TimestampPs()) << R"(,"dur":)"
          << PicosToMicrosString(std::max<uint64_t>(xevent.DurationPs(), 1))
          << R"(,"name":)" << JsonString(event_name);

  if (!args.empty()) {
    *output << R"(,"args":{)";
    bool first_arg = true;
    for (const auto& [name, value] : args) {
      if (!first_arg) {
        *output << ',';
      }
      first_arg = false;
      *output << JsonString(name) << ':' << JsonString(value);
    }
    *output << '}';
  }
  *output << '}';
  return output->good();
}

bool StreamPlaneEvents(uint32_t device_id,
                       const tsl::profiler::XPlaneVisitor& plane,
                       std::ostream* output,
                       JsonArrayWriter* writer) {
  plane.ForEachLine([&](const tsl::profiler::XLineVisitor& line) {
    if (line.DisplayName() == tsl::profiler::kXlaAsyncOpLineName) {
      return;
    }
    const uint32_t resource_id = line.DisplayId();
    line.ForEachEvent([&](const tsl::profiler::XEventVisitor& xevent) {
      const int64_t event_type = xevent.Type().value_or(
          tsl::profiler::HostEventType::kUnknownHostEventType);
      if (tsl::profiler::IsInternalEvent(event_type)) {
        return;
      }
      StreamEvent(device_id, resource_id, xevent, output, writer);
    });
  });
  return output->good();
}

bool StreamXSpaceToTraceJson(const tensorflow::profiler::XSpace& xspace,
                             std::ostream* output,
                             std::string* error) {
  *output << R"({"displayTimeUnit":"ns","metadata":{"highres-ticks":true},"traceEvents":[)";
  JsonArrayWriter writer(output);

  std::vector<const tensorflow::profiler::XPlane*> raw_planes;
  const tensorflow::profiler::XPlane* host_plane =
      tsl::profiler::FindPlaneWithName(xspace, tsl::profiler::kHostThreadsPlaneName);
  if (host_plane != nullptr) {
    raw_planes.push_back(host_plane);
  }

  std::vector<const tensorflow::profiler::XPlane*> device_planes =
      tsl::profiler::FindPlanesWithPrefix(xspace, tsl::profiler::kGpuPlanePrefix);
  if (device_planes.empty()) {
    device_planes =
        tsl::profiler::FindPlanesWithPrefix(xspace, tsl::profiler::kTpuPlanePrefix);
  }
  if (device_planes.empty()) {
    device_planes = tsl::profiler::FindPlanesWithPrefix(
        xspace, tsl::profiler::kCustomPlanePrefix);
  }
  raw_planes.insert(raw_planes.end(), device_planes.begin(), device_planes.end());

  std::vector<PlaneWorkItem> planes;
  planes.reserve(raw_planes.size());
  for (const tensorflow::profiler::XPlane* plane : raw_planes) {
    const tsl::profiler::XPlaneVisitor visitor =
        tsl::profiler::CreateTfXPlaneVisitor(plane);
    const uint32_t device_id =
        plane == host_plane ? tsl::profiler::kHostThreadsDeviceId
                            : tsl::profiler::kFirstDeviceId + visitor.Id();
    planes.push_back(PlaneWorkItem{device_id, visitor});
  }

  for (const PlaneWorkItem& plane : planes) {
    if (!StreamPlaneMetadata(plane.device_id, plane.visitor, output, &writer)) {
      *error = "Failed to write trace metadata";
      return false;
    }
  }
  for (const PlaneWorkItem& plane : planes) {
    if (!StreamPlaneEvents(plane.device_id, plane.visitor, output, &writer)) {
      *error = "Failed to write trace events";
      return false;
    }
  }

  *output << "]}";
  if (!output->good()) {
    *error = "Failed to write trace JSON output";
    return false;
  }
  return true;
}

bool DumpXSpaceToTraceJsonFile(const tensorflow::profiler::XSpace& xspace,
                               std::string_view output_path,
                               std::string* error) {
  std::ofstream output(std::string(output_path), std::ios::binary);
  if (!output.is_open()) {
    *error = "Failed to open output file: " + std::string(output_path);
    return false;
  }

  if (!StreamXSpaceToTraceJson(xspace, &output, error)) {
    return false;
  }
  output.close();
  if (!output) {
    *error = "Failed to write output file: " + std::string(output_path);
    return false;
  }
  return true;
}

}

namespace zml::tools {

bool ConvertXSpaceProtoToTraceJson(std::string_view xspace_proto,
                                   std::string* json,
                                   std::string* error) {
  tensorflow::profiler::XSpace xspace;
  if (!ParseXSpace(xspace_proto, &xspace, error)) {
    return false;
  }

  std::ostringstream output;
  if (!StreamXSpaceToTraceJson(xspace, &output, error)) {
    return false;
  }
  *json = std::move(output).str();
  return true;
}

bool DumpXSpaceProtoToTraceJsonFile(std::string_view xspace_proto,
                                    std::string_view output_path,
                                    std::string* error) {
  tensorflow::profiler::XSpace xspace;
  if (!ParseXSpace(xspace_proto, &xspace, error)) {
    return false;
  }

  return DumpXSpaceToTraceJsonFile(xspace, output_path, error);
}

bool DumpXSpaceFileToTraceJsonFile(std::string_view input_path,
                                   std::string_view output_path,
                                   std::string* error) {
  std::ifstream input(std::string(input_path), std::ios::binary);
  if (!input.is_open()) {
    *error = "Failed to open input file: " + std::string(input_path);
    return false;
  }

  tensorflow::profiler::XSpace xspace;
  if (!xspace.ParseFromIstream(&input)) {
    *error = "Failed to parse XSpace protobuf from: " + std::string(input_path);
    return false;
  }

  return DumpXSpaceToTraceJsonFile(xspace, output_path, error);
}

}

extern "C" zig_slice zml_xspace_to_perfetto_dump(zig_slice xspace_proto,
                                                 zig_slice output_path) {
  std::string error;
  if (zml::tools::DumpXSpaceProtoToTraceJsonFile(
          std::string_view(static_cast<const char*>(xspace_proto.ptr),
                           xspace_proto.len),
          std::string_view(static_cast<const char*>(output_path.ptr),
                           output_path.len),
          &error)) {
    return {.ptr = nullptr, .len = 0};
  }

  if (error.empty()) {
    return {.ptr = nullptr, .len = 0};
  }

  void* buffer = std::malloc(error.size());
  if (buffer == nullptr) {
    return {.ptr = nullptr, .len = 0};
  }

  std::memcpy(buffer, error.data(), error.size());
  return {.ptr = buffer, .len = error.size()};
}

extern "C" void zml_xspace_to_perfetto_str_free(zig_slice text) {
  std::free(text.ptr);
}
