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

const PromptEmbeds = struct {
    /// You own tokenizer, don't forget to deinit
    tokenizer: zml.tokenizer.Tokenizer,
    tokens: []u32,
    // attention_mask: zml.Tensor,
    // mm_token_type_ids: zml.Tensor,
};

pub fn encode_prompt(allocator: std.mem.Allocator, io: std.Io, prompt: []const u8) !PromptEmbeds {
    const tokenizer_path = "/home/erwan/.cache/huggingface/hub/models--Qwen--Qwen-Image-2.1/snapshots/d26bb61231c349cf6b7896fa83353113880e1ba3/processor/tokenizer.json";
    const tokenizer: zml.tokenizer.Tokenizer = try .fromFile(allocator, io, tokenizer_path);

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    const templated_prompt = try prompt_template_t2i(allocator, prompt);
    defer allocator.free(templated_prompt);
    const tokens = try encoder.encodeAlloc(allocator, templated_prompt);

    // const attention_mask = zml.Tensor.scalar(1, .bool);
    // const mm_token_type_ids = zml.Tensor.scalar(0, .i32);

    return PromptEmbeds{
        .tokenizer = tokenizer,
        .tokens = tokens,
        // .attention_mask = attention_mask,
        // .mm_token_type_ids = mm_token_type_ids,
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

test "Default prompt" {
    const prompt =
        \\A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement
    ;
    const expected = .{ 151644, 8948, 198, 1092, 30782, 408, 323, 23643, 279, 3897, 9934, 13, 151645, 198, 151644, 872, 198, 32, 46652, 8061, 1841, 429, 15804, 330, 48, 54, 953, 33669, 220, 17, 13, 16, 497, 62757, 3729, 11, 62751, 389, 14401, 64342, 151645, 198, 151644, 77091, 198 };

    var result = try encode_prompt(std.testing.allocator, std.testing.io, prompt);
    defer result.tokenizer.deinit();
    defer std.testing.allocator.free(result.tokens);

    try std.testing.expectEqualSlices(u32, &expected, result.tokens);
}
