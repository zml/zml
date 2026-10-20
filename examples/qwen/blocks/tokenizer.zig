const std = @import("std");
const zml = @import("zml");

const image_token = "<|image_pad|>";
const video_token = "<|video_pad|>";
const vision_start_token = "<|vision_start|>";
const vision_end_token = "<|vision_end|>";

const line1 = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
const line2_start_t2i = "<|im_start|>user\n";
// const line2_start_ti2i = "image1><|vision_start|><|image_pad|><|vision_end|>";
const line2_end = "<|im_end|>\n";
const line3 = "<|im_start|>assistant\n";

// <|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\nA neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement<|im_end|>\n<|im_start|>assistant\n

pub const TokenizedPrompt = struct {
    tokens: []u32,

    pub fn deinit(self: *TokenizedPrompt, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
    }
};

// Takes in a string and returns the u32 token representation
pub fn tokenize(
    allocator: std.mem.Allocator,
    io: std.Io,
    model_dir: []const u8,
    prompt: []const u8,
    opts: struct { templated: bool = false },
) !TokenizedPrompt {
    var buffer: [256]u8 = undefined;
    const tokenizer_path = try std.fmt.bufPrint(&buffer, "{s}/{s}", .{ model_dir, "processor/tokenizer.json" });
    var tokenizer: zml.tokenizer.Tokenizer = try .fromFile(allocator, io, tokenizer_path);
    defer tokenizer.deinit();

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var tokens: []u32 = undefined;
    if (opts.templated) {
        const templated_prompt = try prompt_template_t2i(allocator, prompt);
        defer allocator.free(templated_prompt);
        tokens = try encoder.encodeAlloc(allocator, templated_prompt);
    } else {
        tokens = try encoder.encodeAlloc(allocator, prompt);
    }

    return TokenizedPrompt{ .tokens = tokens };
}

// It should not be hardcoded but actually the value is always the same (length of the tokenized system prompt)
// "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
// -> [151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198]
pub const DROP_IDX = 14;

const BufferizedPrompt = struct {
    input_ids: zml.Buffer,
    // attention_mask: zml.Buffer,

    pub fn deinit(self: *BufferizedPrompt) void {
        self.input_ids.deinit();
        // self.attention_mask.deinit();
    }
};

/// Moves the input_ids to Tensor and creates the attention mask + token_type_ids Tensor
pub fn bufferize(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    tokenized: TokenizedPrompt,
) !BufferizedPrompt {
    // In theory, attention_mask should be computed using this image_tokens_id
    // const image_tokens_id = try tokenized.encoder.encodeAlloc(allocator, image_token);
    // std.debug.assert(image_tokens_id.len == 1, "image_token_id should only be one token");
    // const image_token_id = image_tokens_id[0];
    // _ = image_token_id; // autofix

    // const sliced_tokens = tokenized.tokens[DROP_IDX..];

    const tokens_slice = try zml.Slice.alloc(allocator, .init(.{ .b = 1, .s = tokenized.tokens.len }, .u32));
    defer tokens_slice.free(allocator);
    // 0 represents text token "<|pad|>"
    // @memset(tokens_slice.items(u32), 0);
    @memcpy(tokens_slice.items(u32)[0..tokenized.tokens.len], tokenized.tokens);

    // const attention_mask_slice = try zml.Slice.alloc(allocator, .init(.{ .b = 1, .s = tokenized.tokens.len }, .bool));
    // defer attention_mask_slice.free(allocator);
    // @memset(attention_mask_slice.items(bool)[0..sliced_tokens.len], true);
    // @memset(attention_mask_slice.items(bool)[sliced_tokens.len..], false);

    const tokens_buffer = try zml.Buffer.fromSlice(io, platform, tokens_slice, .replicated);
    // const attention_mask_buffer = try zml.Buffer.fromSlice(io, platform, attention_mask_slice, .replicated);

    return BufferizedPrompt{
        .input_ids = tokens_buffer,
        // .attention_mask = attention_mask_buffer,
    };
}

fn prompt_template_t2i(allocator: std.mem.Allocator, str: []const u8) ![]const u8 {
    const total_size = line1.len + line2_start_t2i.len + str.len + line2_end.len + line3.len;
    const buffer: []u8 = try allocator.alloc(u8, total_size);
    return try std.fmt.bufPrint(buffer, "{s}{s}{s}{s}{s}", .{ line1, line2_start_t2i, str, line2_end, line3 });
}

// fn prompt_template_ti2i(allocator: std.mem.Allocator, str: []const u8) ![]const u8 {
//     const total_size = line1.len + line2_start_t2i.len + line2_start_ti2i.len + str.len + line2_end.len + line3.len;
//     const buffer: []u8 = try allocator.alloc(u8, total_size);
//     return try std.fmt.bufPrint(buffer, "{s}{s}{s}{s}{s}{s}", .{ line1, line2_start_t2i, line2_start_ti2i, str, line2_end, line3 });
// }

test "Templated" {
    const prompt = "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement";
    const expected = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\nA neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement<|im_end|>\n<|im_start|>assistant\n";

    const templated_prompt = try prompt_template_t2i(std.testing.allocator, prompt);
    defer std.testing.allocator.free(templated_prompt);

    try std.testing.expectEqualSlices(u8, templated_prompt, expected);
}

test "Default prompt templated" {
    const TEST_MODEL_PATH = "/home/erwan/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/d26bb61231c349cf6b7896fa83353113880e1ba3";

    const prompt = "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement";
    const expected = .{ 151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198, 151644, 872, 198, 32, 46652, 8061, 1841, 429, 15804, 330, 48, 54, 953, 33669, 220, 17, 13, 16, 497, 62757, 3729, 11, 62751, 389, 14401, 64342, 151645, 198, 151644, 77091, 198 };

    var result = try tokenize(std.testing.allocator, std.testing.io, TEST_MODEL_PATH, prompt, .{ .templated = true });
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(u32, &expected, result.tokens);
}

test "Default prompt non templated" {
    const TEST_MODEL_PATH = "/home/erwan/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/d26bb61231c349cf6b7896fa83353113880e1ba3";

    const prompt = "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement";
    const expected = .{ 32, 46652, 8061, 1841, 429, 15804, 330, 48, 54, 953, 33669, 220, 17, 13, 16, 497, 62757, 3729, 11, 62751, 389, 14401, 64342 };

    var result = try tokenize(std.testing.allocator, std.testing.io, TEST_MODEL_PATH, prompt, .{ .templated = false });
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqualSlices(u32, &expected, result.tokens);
}
