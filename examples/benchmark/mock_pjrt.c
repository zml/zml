#include "xla/pjrt/c/pjrt_c_api.h"

// These handles only encode host pointers for verification outside the timed loop.
// They must never be dereferenced or passed to a real PJRT API.
static PJRT_Error* buffer_from_host_buffer(
    PJRT_Client_BufferFromHostBuffer_Args* args) {
  args->buffer = (PJRT_Buffer*)args->data;
  args->done_with_host_buffer = NULL;
  return NULL;
}

const PJRT_Api* zml_benchmark_pjrt_api(void) {
  static const PJRT_Api api = {
      .struct_size = PJRT_Api_STRUCT_SIZE,
      .PJRT_Client_BufferFromHostBuffer = buffer_from_host_buffer,
  };
  return &api;
}
