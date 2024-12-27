#pragma once

#include <stdlib.h>

#include "ffi/zig_slice.h"

typedef struct hftokenizers hftokenizers;

hftokenizers *hftokenizers_new(zig_slice);
void hftokenizers_drop(void *tokenizer);
zig_slice hftokenizers_encode(hftokenizers *tokenizer, zig_slice text);
void hftokenizers_tokens_drop(zig_slice tokens);
zig_slice hftokenizers_decode(hftokenizers *tokenizer, zig_slice tokens);
void hftokenizers_str_drop(zig_slice text);
