#include "zml/profiling/traceme.h"

#include <climits>
#include <cstddef>
#include <iostream>

#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xla/tsl/profiler/backends/cpu/traceme_recorder.h"

namespace {

int failures = 0;

void Check(bool condition, const char* name) {
  if (!condition) {
    std::cerr << "FAIL: " << name << '\n';
    ++failures;
  }
}

zig_slice Slice(const char* bytes, size_t size) {
  return {const_cast<char*>(bytes), size};
}

void CheckAccepted(zig_slice input, const char* name) {
  const bool modes[] = {false, true};
  for (bool postprocess : modes) {
    zig_slice result = postprocess
                           ? zml_traceme_session_postprocess(input, 0, 1)
                           : zml_traceme_session_merge(input);
    tensorflow::profiler::XSpace space;
    Check(result.len > 0 && result.len <= INT_MAX &&
              space.ParseFromArray(result.ptr, static_cast<int>(result.len)) &&
              space.planes_size() > 0 && space.planes(0).id() == 0,
          name);
    zml_traceme_str_free(result);
  }
}

void CheckRejected(zig_slice input, const char* name) {
  zig_slice merged = zml_traceme_session_merge(input);
  Check(merged.len == 0, name);
  zml_traceme_str_free(merged);
  zig_slice normalized = zml_traceme_session_postprocess(input, 0, 1);
  Check(normalized.len == 0, name);
  zml_traceme_str_free(normalized);
}

}  // namespace

int main() {
  // A real XSpace wire message: planes[0] has an explicit int64 id = 0.
  const char terminal_zero[] = {0x0a, 0x02, 0x08, 0x00};
  const char sentinel_zero[] = {0x0a, 0x02, 0x08, 0x00, 0x00};
  const char two_sentinels[] = {0x0a, 0x02, 0x08, 0x00, 0x00, 0x00};
  const char malformed[] = {0x0a, 0x02, 0x08};
  const char malformed_zero[] = {static_cast<char>(0xff), 0x00};
  CheckAccepted(Slice(terminal_zero, sizeof(terminal_zero)),
                "valid binary terminal zero survives");
  CheckAccepted(Slice(sentinel_zero, sizeof(sentinel_zero)),
                "one provider sentinel is accepted after validated retry");
  CheckRejected(Slice(two_sentinels, sizeof(two_sentinels)),
                "two sentinels are rejected");
  CheckRejected(Slice(malformed, sizeof(malformed)), "truncated protobuf rejected");
  CheckRejected(Slice(malformed_zero, sizeof(malformed_zero)),
                "invalid bytes remain invalid after sentinel retry");
  CheckRejected(Slice(terminal_zero, static_cast<size_t>(INT_MAX) + 1),
                "oversized input rejected before reading or int conversion");
  CheckRejected({nullptr, 1}, "nonempty null input rejected");

  const bool modes[] = {false, true};
  for (bool postprocess : modes) {
    zml_traceme_session_start(1, 0, false);
    zig_slice input = Slice(malformed, sizeof(malformed));
    zig_slice result = postprocess
                           ? zml_traceme_session_postprocess(input, 0, 1)
                           : zml_traceme_session_merge(input);
    Check(result.len == 0, "malformed active session rejected");
    zml_traceme_str_free(result);
    const bool restarted = tsl::profiler::TraceMeRecorder::Start(1);
    Check(restarted, "parse failure releases local host TraceMe session");
    if (restarted) (void)tsl::profiler::TraceMeRecorder::Stop();
  }

  if (failures == 0) std::cout << "All XSpace compatibility checks passed.\n";
  return failures == 0 ? 0 : 1;
}
