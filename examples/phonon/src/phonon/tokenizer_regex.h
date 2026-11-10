#ifndef PHONON_TOKENIZER_REGEX_H_
#define PHONON_TOKENIZER_REGEX_H_

#include <stddef.h>
#include <stdbool.h>

void *phonon_regex_compile(const char *pattern, size_t length, bool literal);
void phonon_regex_free(void *pattern);
// Returns 0 on success, 1 if capacity is too small (updates length), -1 on error.
int phonon_regex_replace(void *pattern, const char *text, size_t text_length,
                         const char *replacement, size_t replacement_length,
                         char *output, size_t *length);

#endif
