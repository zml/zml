#ifndef ZML_PLATFORMS_NEURON_NKI_SIMULATOR_H_
#define ZML_PLATFORMS_NEURON_NKI_SIMULATOR_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zml_nki_simulator_buffer {
  void* data;
  size_t byte_size;
  const int64_t* dims;
  size_t rank;
  const char* dtype;
  size_t dtype_len;
} zml_nki_simulator_buffer;

// Returns NULL on success and a thread-local error message on failure.
const char* zml_nki_simulator_initialize(const char* python_home,
                                         const char* site_packages,
                                         const char* bridge_directory);

// Executes one NKI kernel against host-addressable XLA FFI buffers. Returns
// NULL on success and a thread-local error message on failure.
const char* zml_nki_simulator_execute(
    const char* source, size_t source_len, const char* entrypoint,
    size_t entrypoint_len, const char* compiler_target,
    size_t compiler_target_len, int64_t grid,
    const zml_nki_simulator_buffer* inputs, size_t input_count,
    const zml_nki_simulator_buffer* outputs, size_t output_count);

#ifdef __cplusplus
}
#endif

#endif  // ZML_PLATFORMS_NEURON_NKI_SIMULATOR_H_
