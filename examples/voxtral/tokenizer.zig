const std = @import("std");

/// Voxtral only needs special-token lookup and decoding: the prompt contains no text.
/// Tekken token IDs reserve the first N entries for specials; ordinary entries
/// contain base64-encoded bytes and can split a UTF-8 codepoint across tokens.
pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    tokens: []const []const u8,
    specials: []const Special,

    const Special = struct { rank: u32, token_str: []const u8, is_control: bool };
    const VocabEntry = struct { rank: u32, token_bytes: []const u8 };
    const Config = struct { default_vocab_size: u32, default_num_special_tokens: u32 };
    const File = struct { config: Config, vocab: []const VocabEntry, special_tokens: []const Special };

    pub fn fromBytes(allocator: std.mem.Allocator, bytes: []const u8) !Tokenizer {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer arena.deinit();
        const a = arena.allocator();
        const file = try std.json.parseFromSliceLeaky(File, a, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        if (file.config.default_vocab_size < file.config.default_num_special_tokens) return error.InvalidTokenizer;
        const tokens = try a.alloc([]const u8, file.config.default_vocab_size);
        @memset(tokens, "");
        for (file.special_tokens) |special| {
            if (special.rank >= file.config.default_num_special_tokens) return error.InvalidTokenizer;
            if (!special.is_control) tokens[special.rank] = special.token_str;
        }
        const decoder_ = std.base64.standard.Decoder;
        for (file.vocab) |entry| {
            if (entry.rank >= tokens.len - file.config.default_num_special_tokens) continue;
            const decoded = try a.alloc(u8, try decoder_.calcSizeForSlice(entry.token_bytes));
            try decoder_.decode(decoded, entry.token_bytes);
            tokens[file.config.default_num_special_tokens + entry.rank] = decoded;
        }
        return .{ .arena = arena, .tokens = tokens, .specials = file.special_tokens };
    }

    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
    }

    pub fn tokenToId(self: *const Tokenizer, token: []const u8) ?u32 {
        for (self.specials) |special| {
            if (std.mem.eql(u8, special.token_str, token)) return special.rank;
        }
        return null;
    }

    pub fn decoder(self: *const Tokenizer) !Decoder {
        return .{ .tokens = self.tokens };
    }

    pub const Decoder = struct {
        tokens: []const []const u8,

        pub fn deinit(_: *Decoder) void {}

        pub fn next(self: *Decoder, id: u32) !?[]const u8 {
            if (id >= self.tokens.len) return error.InvalidTokenId;
            return if (self.tokens[id].len == 0) null else self.tokens[id];
        }
    };
};

test "Tekken IDs include reserved specials and preserve split UTF-8 bytes" {
    var tokenizer = try Tokenizer.fromBytes(std.testing.allocator,
        \\{"config":{"default_vocab_size":6,"default_num_special_tokens":3},
        \\"special_tokens":[{"rank":1,"token_str":"<s>","is_control":true}],
        \\"vocab":[{"rank":0,"token_bytes":"SGk="},{"rank":1,"token_bytes":"ww=="},{"rank":2,"token_bytes":"qQ=="}]}
    );
    defer tokenizer.deinit();
    try std.testing.expectEqual(@as(?u32, 1), tokenizer.tokenToId("<s>"));
    var decoder = try tokenizer.decoder();
    try std.testing.expectEqual(@as(?[]const u8, null), try decoder.next(1));
    try std.testing.expectEqualStrings("Hi", (try decoder.next(3)).?);
    try std.testing.expectEqualStrings("\xc3", (try decoder.next(4)).?);
    try std.testing.expectEqualStrings("\xa9", (try decoder.next(5)).?);
    try std.testing.expectError(error.InvalidTokenId, decoder.next(6));
}
