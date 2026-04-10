#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>

#include "tools/xspace_to_perfetto.h"
#include "xla/tsl/profiler/convert/trace_events_to_json.h"
#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"
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

bool WriteFile(std::string_view output_path,
               std::string_view contents,
               std::string* error) {
  std::ofstream output(std::string(output_path), std::ios::binary);
  if (!output.is_open()) {
    *error = "Failed to open output file: " + std::string(output_path);
    return false;
  }

  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
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

  const tsl::profiler::TraceContainer container =
      tsl::profiler::ConvertXSpaceToTraceContainer(xspace);
  *json = tsl::profiler::TraceContainerToJson(container);
  return true;
}

bool DumpXSpaceProtoToTraceJsonFile(std::string_view xspace_proto,
                                    std::string_view output_path,
                                    std::string* error) {
  std::string json;
  if (!ConvertXSpaceProtoToTraceJson(xspace_proto, &json, error)) {
    return false;
  }
  return WriteFile(output_path, json, error);
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

  const tsl::profiler::TraceContainer container =
      tsl::profiler::ConvertXSpaceToTraceContainer(xspace);
  const std::string json = tsl::profiler::TraceContainerToJson(container);
  return WriteFile(output_path, json, error);
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
