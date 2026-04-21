#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "ffi/zig_slice.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zml_traceme zml_traceme;

zml_traceme *zml_traceme_start(zig_slice name);
void zml_traceme_stop(zml_traceme *traceme);
bool zml_traceme_enabled(void);
void zml_traceme_session_start(int level, uint64_t filter_mask,
                               bool enable_filter);
zig_slice zml_traceme_session_merge(zig_slice xspace);
void zml_traceme_str_free(zig_slice text);

#ifdef __cplusplus
}
#endif
