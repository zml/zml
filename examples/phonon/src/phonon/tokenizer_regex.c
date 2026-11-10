#include "tokenizer_regex.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

void *phonon_regex_compile(const char *pattern, size_t length, bool literal) {
    int error;
    PCRE2_SIZE offset;
    uint32_t options = PCRE2_UTF | (literal ? PCRE2_LITERAL : PCRE2_UCP);
    return pcre2_compile((PCRE2_SPTR)pattern, length, options, &error, &offset, NULL);
}

void phonon_regex_free(void *pattern) {
    pcre2_code_free(pattern);
}

int phonon_regex_replace(void *pattern, const char *text, size_t text_length,
                         const char *replacement, size_t replacement_length,
                         char *output, size_t *length) {
    PCRE2_SIZE size = *length;
    int result = pcre2_substitute(
        pattern, (PCRE2_SPTR)text, text_length, 0,
        PCRE2_SUBSTITUTE_GLOBAL | PCRE2_SUBSTITUTE_LITERAL |
            PCRE2_SUBSTITUTE_OVERFLOW_LENGTH,
        NULL, NULL, (PCRE2_SPTR)replacement, replacement_length,
        (PCRE2_UCHAR *)output, &size);
    *length = size;
    if (result == PCRE2_ERROR_NOMEMORY) return 1;
    return result < 0 ? -1 : 0;
}
