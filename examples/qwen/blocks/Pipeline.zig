const std = @import("std");
const zml = @import("zml");

const tokenizer = @import("./tokenizer.zig");

const Pipeline = @This();

// _get_qwen_prompt_embeds in diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py
const PromptEmbeds = struct {
    prompt_embeds: zml.Tensor,
    encoder_attention_mask: zml.Tensor,
    image_pad_mask: zml.Tensor,
};

pub fn runOnce(
    allocator: std.mem.Allocator,
    io: std.Io,
    prompt: []const u8,
) !void {
    const embeds = try tokenizer.encode_prompt(allocator, io, prompt);
    _ = embeds; // autofix
}
