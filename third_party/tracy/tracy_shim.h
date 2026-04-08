#ifndef ZML_TRACY_SHIM_H_
#define ZML_TRACY_SHIM_H_

#include <stddef.h>
#include <stdint.h>

#include "tracy/TracyC.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef TracyCZoneCtx ZmlTracyZoneCtx;

void zml_tracy_set_thread_name(const char* name);
int32_t zml_tracy_is_connected(void);
ZmlTracyZoneCtx zml_tracy_zone_begin(
    const char* name,
    const char* function,
    const char* file,
    uint32_t line,
    uint32_t color);
void zml_tracy_zone_end(ZmlTracyZoneCtx ctx);
void zml_tracy_zone_text(ZmlTracyZoneCtx ctx, const char* text, size_t size);
void zml_tracy_zone_value(ZmlTracyZoneCtx ctx, uint64_t value);
void zml_tracy_frame_mark(void);
void zml_tracy_frame_mark_named(const char* name);

#ifdef __cplusplus
}
#endif

#endif
