const std = @import("std");
const zml = @import("zml");

const Qwen3VLTextRotaryEmbedding = @This();

eps: f32,
tag: zml.Shape.Tag,
