const std = @import("std");
const zml = @import("zml");

/// Convert Tekken's ranked byte vocabulary to the equivalent HF BPE schema
/// consumed by ZML's IREE tokenizer. No Python or tokenizer conversion file is
/// needed at runtime. Special speech markers are assembled separately.
pub fn load(allocator: std.mem.Allocator, bytes: []const u8) !zml.tokenizer.Tokenizer {
    const Entry = struct { rank: u32, token_bytes: []const u8 };
    const File = struct {
        config: struct { pattern: []const u8, default_vocab_size: u32, default_num_special_tokens: u32 },
        vocab: []const Entry,
    };
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try std.json.parseFromSliceLeaky(File, a, bytes, .{ .ignore_unknown_fields = true });
    const reserved = parsed.config.default_num_special_tokens;
    if (reserved != 1000 or parsed.config.default_vocab_size != 131072) return error.UnsupportedTokenizer;
    const count = parsed.config.default_vocab_size - reserved;
    const tokens = try a.alloc([]const u8, count);
    const encoded = try a.alloc([]const u8, count);
    @memset(tokens, "");
    var ranks: std.StringHashMap(u32) = .init(a);
    for (parsed.vocab) |entry| {
        if (entry.rank >= count) continue;
        const data = try a.alloc(u8, try std.base64.standard.Decoder.calcSizeForSlice(entry.token_bytes));
        try std.base64.standard.Decoder.decode(data, entry.token_bytes);
        if (data.len == 0 or tokens[entry.rank].len != 0) return error.InvalidTokenizer;
        tokens[entry.rank] = data;
        encoded[entry.rank] = try byteLevel(a, data);
        try ranks.put(data, entry.rank);
    }
    var json: std.Io.Writer.Allocating = .init(a);
    const w = &json.writer;
    try w.writeAll("{\"version\":\"1.0\",\"added_tokens\":[],\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Split\",\"pattern\":{\"Regex\":");
    try std.json.Stringify.value(parsed.config.pattern, .{}, w);
    try w.writeAll("},\"behavior\":\"Isolated\",\"invert\":false},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}]},\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false},\"model\":{\"type\":\"BPE\",\"unk_token\":null,\"byte_fallback\":false,\"ignore_merges\":true,\"vocab\":{");
    for (0..reserved) |id| {
        if (id != 0) try w.writeByte(',');
        try w.print("\"\u{e000}{d}\":{d}", .{ id, id });
    }
    for (tokens, encoded, 0..) |token, name, rank| {
        if (token.len == 0) return error.InvalidTokenizer;
        try w.writeByte(',');
        try std.json.Stringify.value(name, .{}, w);
        try w.print(":{d}", .{rank + reserved});
    }
    try w.writeAll("},\"merges\":[");
    var first = true;
    var cuts: std.ArrayList(usize) = .empty;
    for (tokens, 0..) |token, rank| {
        if (token.len == 1) continue;
        cuts.clearRetainingCapacity();
        for (0..token.len + 1) |i| try cuts.append(a, i);
        while (cuts.items.len > 3) {
            var best: u32 = @intCast(rank);
            var best_index: ?usize = null;
            for (0..cuts.items.len - 2) |i| {
                const pair_rank = ranks.get(token[cuts.items[i]..cuts.items[i + 2]]) orelse continue;
                if (pair_rank < best) {
                    best = pair_rank;
                    best_index = i + 1;
                }
            }
            _ = cuts.orderedRemove(best_index orelse return error.InvalidBpeMerge);
        }
        const left = ranks.get(token[0..cuts.items[1]]) orelse return error.InvalidBpeMerge;
        const right = ranks.get(token[cuts.items[1]..]) orelse return error.InvalidBpeMerge;
        if (!first) try w.writeByte(',');
        first = false;
        try w.writeByte('[');
        try std.json.Stringify.value(encoded[left], .{}, w);
        try w.writeByte(',');
        try std.json.Stringify.value(encoded[right], .{}, w);
        try w.writeByte(']');
    }
    try w.writeAll("]}}");
    return zml.tokenizer.Tokenizer.fromBytes(allocator, json.written());
}

fn byteLevel(allocator: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var mapping: [256]u21 = undefined;
    var extra: u21 = 256;
    for (&mapping, 0..) |*entry, i| {
        if ((i >= 33 and i <= 126) or (i >= 161 and i <= 172) or i >= 174) {
            entry.* = @intCast(i);
        } else {
            entry.* = extra;
            extra += 1;
        }
    }
    var out: std.Io.Writer.Allocating = .init(allocator);
    for (bytes) |b| {
        var utf8: [4]u8 = undefined;
        const len = try std.unicode.utf8Encode(mapping[b], &utf8);
        try out.writer.writeAll(utf8[0..len]);
    }
    return out.toOwnedSlice();
}

pub fn prompt(allocator: std.mem.Allocator, text: []const u32, voice_frames: usize) ![]u32 {
    const ids = try allocator.alloc(u32, voice_frames + text.len + 5);
    ids[0] = 1; // BOS
    ids[1] = 25; // BEGIN_AUDIO
    @memset(ids[2..][0..voice_frames], 24);
    ids[2 + voice_frames] = 36; // TEXT_TO_AUDIO / NEXT_AUDIO_TEXT
    @memcpy(ids[3 + voice_frames ..][0..text.len], text);
    ids[ids.len - 2] = 35; // AUDIO_TO_TEXT / REPEAT_AUDIO_TEXT
    ids[ids.len - 1] = 25;
    return ids;
}

test "speech prompt wraps voice and text with the Mistral markers" {
    const ids = try prompt(std.testing.allocator, &.{ 1234, 5678 }, 2);
    defer std.testing.allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 1, 25, 24, 24, 36, 1234, 5678, 35, 25 }, ids);
}
