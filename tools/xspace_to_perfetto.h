#pragma once

#include <stdint.h>

#include "ffi/zig_allocator.h"
#include "ffi/zig_slice.h"

#ifdef __cplusplus
extern "C" {
#endif

zig_slice zml_xspace_to_perfetto_dump(zig_allocator* error_allocator,
                                      zig_slice xspace_proto,
                                      zig_slice output_path);
zig_slice zml_xspace_normalize_and_dump(zig_allocator* error_allocator,
                                        zig_slice xspace_proto,
                                        uint64_t start_time_ns,
                                        uint64_t stop_time_ns,
                                        zig_slice output_path);

#ifdef __cplusplus
}
#endif
