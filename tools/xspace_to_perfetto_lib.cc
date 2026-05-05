#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <unistd.h>

#include "tools/xspace_to_perfetto.h"
#include "xla/tsl/profiler/convert/post_process_single_host_xplane.h"
#include "xla/tsl/profiler/convert/trace_events_to_json.h"
#include "xla/tsl/profiler/convert/xplane_to_trace_events.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace {

zig_slice EmptySlice() { return {.ptr = nullptr, .len = 0}; }

std::string_view ToStringView(zig_slice slice) {
  return std::string_view(static_cast<const char*>(slice.ptr), slice.len);
}

bool SetError(zig_allocator* error_allocator,
              zig_slice* error,
              std::string_view message) {
  char* buffer = error_allocator->allocate<char>(message.size());
  if (buffer == nullptr) {
    *error = EmptySlice();
    return false;
  }

  std::memcpy(buffer, message.data(), message.size());
  *error = {.ptr = buffer, .len = message.size()};
  return true;
}

bool SetErrorWithPath(zig_allocator* error_allocator,
                      zig_slice* error,
                      std::string_view prefix,
                      std::string_view path) {
  const size_t len = prefix.size() + path.size();
  char* buffer = error_allocator->allocate<char>(len);
  if (buffer == nullptr) {
    *error = EmptySlice();
    return false;
  }

  std::memcpy(buffer, prefix.data(), prefix.size());
  std::memcpy(buffer + prefix.size(), path.data(), path.size());
  *error = {.ptr = buffer, .len = len};
  return true;
}

void PostProcessXSpace(tensorflow::profiler::XSpace* xspace,
                       uint64_t start_time_ns,
                       uint64_t stop_time_ns) {
  tsl::profiler::SetXSpacePidIfNotSet(*xspace, getpid());
  tsl::profiler::PostProcessSingleHostXSpace(
      xspace, start_time_ns, stop_time_ns);
}

bool SerializeXSpace(const tensorflow::profiler::XSpace& xspace,
                     std::string* serialized_xspace,
                     zig_allocator* error_allocator,
                     zig_slice* error) {
  if (!xspace.SerializeToString(serialized_xspace)) {
    SetError(error_allocator, error,
             "Failed to serialize normalized XSpace protobuf");
    return false;
  }
  return true;
}

bool ParseXSpace(std::string_view xspace_proto,
                 tensorflow::profiler::XSpace* xspace,
                 zig_allocator* error_allocator,
                 zig_slice* error) {
  if (xspace_proto.size() >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    SetError(error_allocator, error, "XSpace protobuf is too large to parse");
    return false;
  }
  if (!xspace->ParseFromArray(xspace_proto.data(),
                              static_cast<int>(xspace_proto.size()))) {
    SetError(error_allocator, error, "Failed to parse XSpace protobuf");
    return false;
  }
  return true;
}

bool WriteFile(std::string_view output_path,
               std::string_view contents,
               zig_allocator* error_allocator,
               zig_slice* error) {
  std::ofstream output(std::string(output_path), std::ios::binary);
  if (!output.is_open()) {
    SetErrorWithPath(error_allocator, error, "Failed to open output file: ",
                     output_path);
    return false;
  }

  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output) {
    SetErrorWithPath(error_allocator, error, "Failed to write output file: ",
                     output_path);
    return false;
  }
  return true;
}

}  // namespace

namespace zml::tools {

bool ConvertXSpaceProtoToTraceJson(std::string_view xspace_proto,
                                   std::string* json,
                                   zig_allocator* error_allocator,
                                   zig_slice* error) {
  tensorflow::profiler::XSpace xspace;
  if (!ParseXSpace(xspace_proto, &xspace, error_allocator, error)) {
    return false;
  }

  const tsl::profiler::TraceContainer container =
      tsl::profiler::ConvertXSpaceToTraceContainer(xspace);
  *json = tsl::profiler::TraceContainerToJson(container);
  return true;
}

bool DumpXSpaceProtoToTraceJsonFile(std::string_view xspace_proto,
                                    std::string_view output_path,
                                    zig_allocator* error_allocator,
                                    zig_slice* error) {
  std::string json;
  if (!ConvertXSpaceProtoToTraceJson(xspace_proto, &json, error_allocator,
                                     error)) {
    return false;
  }
  return WriteFile(output_path, json, error_allocator, error);
}

bool DumpXSpaceFileToTraceJsonFile(std::string_view input_path,
                                   std::string_view output_path,
                                   zig_allocator* error_allocator,
                                   zig_slice* error) {
  std::ifstream input(std::string(input_path), std::ios::binary);
  if (!input.is_open()) {
    SetErrorWithPath(error_allocator, error, "Failed to open input file: ",
                     input_path);
    return false;
  }

  tensorflow::profiler::XSpace xspace;
  if (!xspace.ParseFromIstream(&input)) {
    SetErrorWithPath(error_allocator, error,
                     "Failed to parse XSpace protobuf from: ", input_path);
    return false;
  }

  const tsl::profiler::TraceContainer container =
      tsl::profiler::ConvertXSpaceToTraceContainer(xspace);
  const std::string json = tsl::profiler::TraceContainerToJson(container);
  return WriteFile(output_path, json, error_allocator, error);
}

}  // namespace zml::tools

extern "C" zig_slice zml_xspace_to_perfetto_dump(
    zig_allocator* error_allocator,
    zig_slice xspace_proto,
    zig_slice output_path) {
  zig_slice error = EmptySlice();
  if (zml::tools::DumpXSpaceProtoToTraceJsonFile(
          ToStringView(xspace_proto),
          ToStringView(output_path),
          error_allocator,
          &error)) {
    return EmptySlice();
  }
  return error;
}

extern "C" zig_slice zml_xspace_normalize_and_dump(
    zig_allocator* error_allocator, zig_slice xspace_proto,
    uint64_t start_time_ns, uint64_t stop_time_ns, zig_slice output_path) {
  zig_slice error = EmptySlice();
  tensorflow::profiler::XSpace xspace;
  if (!ParseXSpace(ToStringView(xspace_proto), &xspace, error_allocator,
                   &error)) {
    return error;
  }

  PostProcessXSpace(&xspace, start_time_ns, stop_time_ns);
  std::string normalized_xspace;
  if (!SerializeXSpace(xspace, &normalized_xspace, error_allocator, &error) ||
      !WriteFile(ToStringView(output_path), normalized_xspace, error_allocator,
                 &error)) {
    return error;
  }

  return EmptySlice();
}
