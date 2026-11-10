#ifndef PHONON_AUDIO_H
#define PHONON_AUDIO_H
#include <stddef.h>
#include <stdint.h>
typedef struct phonon_audio phonon_audio;
/* null_backend is for device-independent tests only. */
int phonon_audio_create(uint32_t sample_rate, int null_backend, phonon_audio **out);
int phonon_audio_write(phonon_audio *audio, const float *samples, size_t count);
/* PCM samples consumed by the callback; excludes underrun silence. */
uint64_t phonon_audio_consumed_frames(phonon_audio *audio);
int phonon_audio_finish(phonon_audio *audio);
void phonon_audio_destroy(phonon_audio *audio);
#endif
