#pragma once

#include <stdlib.h>

typedef struct
{
    char *ptr;
    size_t len;
} zig_slice;
typedef void hf_tokenizers;

hf_tokenizers *hf_tokenizers_new(zig_slice);
void hf_tokenizers_drop(void *tokenizer);
zig_slice hf_tokenizers_encode(hf_tokenizers *tokenizer, zig_slice text);
void hf_tokenizers_tokens_drop(zig_slice tokens);
zig_slice hf_tokenizers_decode(hf_tokenizers *tokenizer, zig_slice tokens);
void hf_tokenizers_str_drop(zig_slice text);
