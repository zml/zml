#include <os/log.h>
#include <os/signpost.h>

void zml_os_signpost_event(
    os_log_t log,
    os_signpost_id_t signpost_id,
    const char *message);

void zml_os_signpost_interval_begin(
    os_log_t log,
    os_signpost_id_t signpost_id,
    const char *message);

void zml_os_signpost_interval_end(
    os_log_t log,
    os_signpost_id_t signpost_id,
    const char *message);
