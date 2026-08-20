const std = @import("std");

const zml = @import("zml");

const base_vocab_size = 163_584;

const Piece = struct {
    start: usize,
    end: usize,
    token_id: u32,
};

fn isDirectByte(byte: u8) bool {
    return (byte >= 33 and byte <= 126) or
        (byte >= 161 and byte <= 172) or
        (byte >= 174);
}

/// GPT-2/tiktoken reversible byte-to-Unicode mapping used by ByteLevel BPE.
fn byteLevelCodepoint(byte: u8) u21 {
    if (isDirectByte(byte)) return byte;
    var offset: u21 = 0;
    for (0..byte) |candidate| {
        if (!isDirectByte(@intCast(candidate))) offset += 1;
    }
    return 256 + offset;
}

fn tokenIdForBytes(
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    bytes: []const u8,
) !?u32 {
    var spelling: std.Io.Writer.Allocating = .init(allocator);
    defer spelling.deinit();
    for (bytes) |byte| {
        var encoded: [4]u8 = undefined;
        const len = std.unicode.utf8Encode(byteLevelCodepoint(byte), &encoded) catch unreachable;
        try spelling.writer.writeAll(encoded[0..len]);
    }
    const token_id = tokenizer.tokenId(spelling.written()) orelse return null;
    return if (token_id < base_vocab_size) token_id else null;
}

/// Run the original rank-ordered byte-pair merge on one whitespace regex
/// piece. Mergeable token IDs are their tiktoken ranks in this vocabulary.
fn encodeWhitespace(
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    whitespace: []const u8,
    output: *std.ArrayList(u32),
) !void {
    var pieces = try std.ArrayList(Piece).initCapacity(allocator, whitespace.len);
    defer pieces.deinit(allocator);
    for (whitespace, 0..) |_, index| {
        const token_id = try tokenIdForBytes(allocator, tokenizer, whitespace[index .. index + 1]) orelse
            return error.KimiK3MissingByteToken;
        try pieces.append(allocator, .{ .start = index, .end = index + 1, .token_id = token_id });
    }

    while (pieces.items.len > 1) {
        var best_index: ?usize = null;
        var best_rank: u32 = std.math.maxInt(u32);
        for (0..pieces.items.len - 1) |index| {
            const candidate = try tokenIdForBytes(
                allocator,
                tokenizer,
                whitespace[pieces.items[index].start..pieces.items[index + 1].end],
            ) orelse continue;
            if (candidate < best_rank) {
                best_rank = candidate;
                best_index = index;
            }
        }
        const index = best_index orelse break;
        pieces.items[index].end = pieces.items[index + 1].end;
        pieces.items[index].token_id = best_rank;
        _ = pieces.orderedRemove(index + 1);
    }
    for (pieces.items) |piece| try output.append(allocator, piece.token_id);
}

/// Reproduce the official regex's ordered whitespace alternatives:
///   1. all whitespace through the last CR/LF is one BPE piece;
///   2. before following text, all but the final whitespace codepoint is one
///      piece and the final codepoint is a second piece;
///   3. terminal whitespace is one piece.
fn encodeOfficialWhitespace(
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    whitespace: []const u8,
    has_following_text: bool,
    output: *std.ArrayList(u32),
) !void {
    var after_last_newline: usize = 0;
    for (whitespace, 0..) |byte, index| {
        if (byte == '\r' or byte == '\n') after_last_newline = index + 1;
    }
    if (after_last_newline > 0) {
        try encodeWhitespace(allocator, tokenizer, whitespace[0..after_last_newline], output);
    }

    const trailing = whitespace[after_last_newline..];
    if (trailing.len == 0) return;
    if (!has_following_text) {
        try encodeWhitespace(allocator, tokenizer, trailing, output);
        return;
    }

    var final_start = trailing.len - 1;
    while (final_start > 0 and trailing[final_start] & 0xC0 == 0x80) final_start -= 1;
    if (final_start > 0) {
        try encodeWhitespace(allocator, tokenizer, trailing[0..final_start], output);
    }
    try encodeWhitespace(allocator, tokenizer, trailing[final_start..], output);
}

fn isWhitespace(bytes: []const u8) bool {
    if (bytes.len == 0) return false;
    for (bytes) |byte| if (!std.ascii.isWhitespace(byte)) return false;
    return true;
}

/// Encode literal K3 text. IREE handles the compatible word/number/punctuation
/// segmentation; contiguous ASCII whitespace is re-merged with exact
/// tiktoken ranks to compensate for IREE's streaming regex commit boundaries.
pub fn encodeText(
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    text: []const u8,
) ![]u32 {
    var encoder = try tokenizer.encoderWithoutSpecialTokenMatching();
    defer encoder.deinit();
    const initial = try encoder.encodeAlloc(allocator, text);
    defer allocator.free(initial);

    var output = try std.ArrayList(u32).initCapacity(allocator, initial.len);
    errdefer output.deinit(allocator);
    var index: usize = 0;
    while (index < initial.len) {
        const run_start = index;
        var decoder = try tokenizer.decoder();
        defer decoder.deinit();
        var decoded = try decoder.decodeAlloc(allocator, initial[index .. index + 1]);
        defer decoded.deinit(allocator);
        if (!isWhitespace(decoded.items)) {
            try output.append(allocator, initial[index]);
            index += 1;
            continue;
        }

        var run: std.Io.Writer.Allocating = .init(allocator);
        defer run.deinit();
        while (index < initial.len) : (index += 1) {
            var piece_decoder = try tokenizer.decoder();
            defer piece_decoder.deinit();
            var piece = try piece_decoder.decodeAlloc(allocator, initial[index .. index + 1]);
            defer piece.deinit(allocator);
            if (!isWhitespace(piece.items)) break;
            try run.writer.writeAll(piece.items);
        }
        if (std.mem.indexOfAny(u8, run.written(), "\r\n") == null) {
            try output.appendSlice(allocator, initial[run_start..index]);
            continue;
        }
        try encodeOfficialWhitespace(
            allocator,
            tokenizer,
            run.written(),
            index < initial.len,
            &output,
        );
    }
    return output.toOwnedSlice(allocator);
}
