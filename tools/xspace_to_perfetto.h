#pragma once

#include "ffi/zig_slice.h"

#ifdef __cplusplus
extern "C" {
#endif

zig_slice zml_xspace_to_perfetto_dump(zig_slice xspace_proto,
                                      zig_slice output_path);
void zml_xspace_to_perfetto_str_free(zig_slice text);

#ifdef __cplusplus
}
#endif
