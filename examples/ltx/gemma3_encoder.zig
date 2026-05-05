// Gemma-3 Text Encoder for LTX-2.3
//
// Simplified Gemma-3 encoder that produces stacked hidden states [B, S, D, L]
// for the LTX text embedding pipeline.  Only the tokenization + left-padding
// logic is implemented so far; the model forward pass will follow.
//
// Reference: Python LTXVGemmaTokenizer (ltx_core/text_encoders/gemma/tokenizer.py)
//   - HuggingFace AutoTokenizer with max_length=1024, padding="max_length",
//     padding_side="left", truncation=True.
//   - BOS token (id=2) is prepended explicitly (IREE tokenizer backend does
//     not apply the HuggingFace post_processor TemplateProcessing rule).
//   - Pad token id = 0.

const std = @import("std");
const zml = @import("zml");

const log = std.log.scoped(.@"ltx/gemma3");

/// Fixed sequence length matching the Python LTXVGemmaTokenizer.
pub const MAX_SEQ_LEN: usize = 1024;

/// Gemma-3 pad token id.
pub const PAD_TOKEN_ID: u32 = 0;

/// Gemma-3 BOS token id.
pub const BOS_TOKEN_ID: u32 = 2;

/// Result of tokenizing and padding a prompt.
pub const TokenizeResult = struct {
    /// Token IDs, left-padded to MAX_SEQ_LEN.
    input_ids: [MAX_SEQ_LEN]u32,
    /// Attention mask: 0 for padding, 1 for real tokens.
    attention_mask: [MAX_SEQ_LEN]u32,
    /// Number of real (non-pad) tokens.
    real_token_count: usize,
};

/// Tokenize a prompt and left-pad to MAX_SEQ_LEN.
///
/// Matches the Python behavior:
///   - BOS is prepended if the tokenizer doesn't add it (IREE backend
///     does not apply the HuggingFace post_processor TemplateProcessing rule).
///   - Truncation to MAX_SEQ_LEN if the prompt is too long.
///   - Left-padding with PAD_TOKEN_ID.
///   - Attention mask: 0 for pad positions (left), 1 for real tokens (right).
pub fn tokenizeAndPad(
    allocator: std.mem.Allocator,
    tokenizer: *const zml.tokenizer.Tokenizer,
    text: []const u8,
) !TokenizeResult {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const raw_ids = try encoder.encodeAlloc(allocator, text);
    defer allocator.free(raw_ids);

    // Prepend BOS if not already present (IREE tokenizer may not apply
    // the HuggingFace post_processor TemplateProcessing rule).
    const has_bos = raw_ids.len > 0 and raw_ids[0] == BOS_TOKEN_ID;
    const total_len = if (has_bos) raw_ids.len else raw_ids.len + 1;

    // Truncate if needed (matching Python truncation=True).
    const effective_len = @min(total_len, MAX_SEQ_LEN);
    const pad_len = MAX_SEQ_LEN - effective_len;

    var result: TokenizeResult = .{
        .input_ids = undefined,
        .attention_mask = undefined,
        .real_token_count = effective_len,
    };

    // Left-pad: fill pad positions first, then copy real tokens.
    @memset(result.input_ids[0..pad_len], PAD_TOKEN_ID);

    if (has_bos) {
        @memcpy(result.input_ids[pad_len..], raw_ids[0..effective_len]);
    } else {
        result.input_ids[pad_len] = BOS_TOKEN_ID;
        const tokens_to_copy = effective_len - 1;
        @memcpy(result.input_ids[pad_len + 1 ..][0..tokens_to_copy], raw_ids[0..tokens_to_copy]);
    }

    @memset(result.attention_mask[0..pad_len], 0);
    @memset(result.attention_mask[pad_len..], 1);

    return result;
}
