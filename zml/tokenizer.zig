//! Text tokenizer implementations
const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

const log = std.log.scoped(.zml_tokenizer);

const helpers = @import("helpers.zig");
const meta = @import("meta.zig");

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Normalizer);
    std.testing.refAllDecls(Tokenizer);
}

/// Byte Pair Encoding tokenizer generally used for LLM.
pub const Tokenizer = struct {
    tokens: [][]const u8,
    token_lookup: std.StringHashMapUnmanaged(u32),
    special_tokens: SpecialTokens,

    scores: []f32,
    max_token_len: u32,
    normalizer: ?Normalizer,
    // Allows to split unknown unicode characters into bytes.
    byte_fallback: bool = false,

    arena_state: std.heap.ArenaAllocator,
    vocab_size: u32,
    next_token_id: u32 = 0,

    pub const SpecialTokens = struct {
        eos: u32,
        bos: u32,
        unk: u32,
        pad: u32 = std.math.maxInt(u32),
        hard_space: u32 = std.math.maxInt(u32),
    };

    pub fn init(
        allocator: std.mem.Allocator,
        vocab_size: u32,
        max_token_len: u32,
        normalizer: ?Normalizer,
        special_tokens: SpecialTokens,
        alloc_tokens: bool,
    ) !Tokenizer {
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        errdefer arena_state.deinit();
        const arena = arena_state.allocator();

        var token_lookup: std.StringHashMapUnmanaged(u32) = .{};
        errdefer token_lookup.deinit(arena);

        try token_lookup.ensureTotalCapacity(arena, @intCast(vocab_size));

        const tokens: [][]const u8 = if (alloc_tokens) try arena.alloc([]u8, vocab_size) else &.{};
        errdefer if (alloc_tokens) arena.free(tokens);

        const scores: []f32 = if (alloc_tokens) try arena.alloc(f32, vocab_size) else &.{};
        errdefer if (alloc_tokens) arena.free(scores);

        return .{
            .tokens = tokens,
            .scores = scores,
            .max_token_len = max_token_len,
            .token_lookup = token_lookup,
            .arena_state = arena_state,
            .normalizer = normalizer,
            .vocab_size = vocab_size,
            .special_tokens = special_tokens,
        };
    }

    pub fn deinit(self: Tokenizer) void {
        self.arena_state.deinit();
    }

    /// Reads a new word directly into the tokenizer arena.
    pub fn readTokenInto(self: *Tokenizer, score: f32, len: usize, tok_reader: anytype) !void {
        const arena = self.arena_state.allocator();

        const token = try arena.alloc(u8, len);
        const n = try tok_reader.read(token);
        std.debug.assert(n == len);

        return self.addOwnedToken(score, token);
    }

    /// Adds a new token (and copy it)
    pub fn addToken(self: *Tokenizer, score: f32, token: []const u8) !void {
        const arena = self.arena_state.allocator();
        return self.addOwnedToken(score, try arena.dupe(u8, token));
    }

    /// Adds a new token (without copying it)
    pub fn addOwnedToken(self: *Tokenizer, score: f32, token: []const u8) void {
        const i = self.next_token_id;
        std.debug.assert(i < self.vocab_size);
        self.next_token_id += 1;

        self.scores[i] = score;
        self.tokens[i] = token;
        const v = self.token_lookup.getOrPutAssumeCapacity(token);
        if (!v.found_existing) {
            v.value_ptr.* = i;
        }
    }

    pub fn addOwnedTokenByIndex(self: *Tokenizer, i: u32, score: f32, token: []const u8) void {
        std.debug.assert(i < self.vocab_size);
        self.next_token_id += 1;
        self.scores[i] = score;
        self.tokens[i] = token;
        const v = self.token_lookup.getOrPutAssumeCapacity(token);
        if (!v.found_existing) {
            v.value_ptr.* = @intCast(i);
        }
    }

    fn lookup(self: *const Tokenizer, str: []const u8) ?u32 {
        return self.token_lookup.get(str);
    }

    pub const EncodeOptions = struct {
        /// Should the beginning of sentence '<s>' token be added.
        add_bos: bool = true,
        add_eos: bool = false,
        pad_to: u32 = 0,
        // Print tokenization intermediary steps.
        debug: bool = false,
    };

    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, raw: []const u8, options: EncodeOptions) ![]u32 {
        if (options.debug) log.debug("Tokenizer.encode('{s}')", .{raw});
        const input = if (self.normalizer) |n| try n.normalize(allocator, raw) else raw;
        defer if (self.normalizer) |_| allocator.free(input);
        if (options.debug) log.debug("Tokenizer.encode.normalize -> '{s}'", .{input});

        // Allocate a buffer that can fit all indices as well as extra character if requested.
        // We then slice it so that the token merging code doesn't see the bos token.
        const tok_buff_alloc = try allocator.alloc(u32, @max(options.pad_to, input.len + 2));
        const tok_buff = if (options.add_bos) tok_buff_alloc[1..] else tok_buff_alloc;

        const MergeState = union(enum) { ready: u32, nope, hard_space, idk };
        const mergeable = try allocator.alloc(MergeState, tok_buff.len);

        var num_tokens: usize = 0;
        var it: CharTokenIterator = .{ .input = input };
        while (try it.nextCodepointToken(self)) |token| : (num_tokens += 1) {
            tok_buff[num_tokens] = token;
            mergeable[num_tokens] = if (token == self.special_tokens.hard_space)
                .hard_space
            else
                .idk;
        }

        var stable_prefix: usize = 0;
        var stable_off: usize = 0;
        while (true) {
            // Step by step visualization of the progress.
            if (options.debug) {
                var _debug_buf: [256]u8 = undefined;
                var debug_progress: std.ArrayList(u8) = .{ .items = _debug_buf[0..0], .capacity = _debug_buf.len, .allocator = undefined };
                self.decodeWithOpts(&debug_progress, tok_buff[0..num_tokens], .{ .sep = "|" }) catch {};
                log.debug("tokens: {d} -> {s}", .{ tok_buff[0..num_tokens], debug_progress.items });
            }
            var best_score: f32 = -1e10;
            var best_token: u32 = 0;
            var best_idx: ?usize = null;
            var input_off: usize = stable_off;

            // Find best tokens to merge in all available tokens
            for (stable_prefix..num_tokens - 1) |i| {
                if (tok_buff[i] == self.special_tokens.unk) {
                    input_off += 1;
                    continue;
                }
                const cur_tok = self.tokens[tok_buff[i]];
                defer input_off += cur_tok.len;

                // Lookup merge for current token, if not already done.
                switch (mergeable[i]) {
                    .nope => continue,
                    .ready => {},
                    .hard_space => {
                        // Since tokens are not allowed to merge through hard sep,
                        // we don't need to merge the sentence-wide best token.
                        // We can just merge the best token since beginning.
                        if (best_idx != null) break;
                        // OTOH if there was no merge possible since beginning,
                        // we can skip the beginning in future iterations.
                        stable_prefix = i + 1;
                        stable_off = input_off + cur_tok.len;
                        continue;
                    },
                    .idk => {

                        // Special tokens can't be concatenated.
                        if (builtin.mode == .Debug and tok_buff[i] != self.special_tokens.unk) {
                            // Detects memory corruption of tokens.
                            if (cur_tok.len == 0 or cur_tok.len > self.max_token_len) @panic("Token looks corrupted !");

                            meta.assert(std.mem.eql(u8, cur_tok, input[input_off..][0..cur_tok.len]), "current token '{s}' not found in input string '{s}' !", .{ cur_tok, input[input_off..] });
                        }
                        const next_tok = self.tokens[tok_buff[i + 1]];
                        // if `next_tok` is `.unk`, length is 1; otherwise, it's the length of the token.
                        const next_tok_len = if (tok_buff[i + 1] == self.special_tokens.unk) 1 else next_tok.len;
                        const concat_tokens = input[input_off..][0 .. cur_tok.len + next_tok_len];
                        // Save the result
                        mergeable[i] = if (self.lookup(concat_tokens)) |tok|
                            .{ .ready = tok }
                        else
                            .nope;
                    },
                }

                switch (mergeable[i]) {
                    .idk, .hard_space => unreachable,
                    .nope => continue,
                    .ready => |tok| {
                        if (self.scores[tok] > best_score) {
                            best_score = self.scores[tok];
                            best_token = tok;
                            best_idx = i;
                        }
                    },
                }
            }

            if (best_idx) |bidx| {
                // Apply the merge.
                tok_buff[bidx] = best_token;
                std.mem.copyForwards(u32, tok_buff[bidx + 1 ..], tok_buff[bidx + 2 .. num_tokens]);
                std.mem.copyForwards(MergeState, mergeable[bidx + 1 ..], mergeable[bidx + 2 .. num_tokens]);
                num_tokens -= 1;
                // We got two new merge lookups to do.
                mergeable[bidx] = .idk;
                if (bidx > 0 and mergeable[bidx - 1] != .hard_space) mergeable[bidx - 1] = .idk;
            } else {
                // No merge candidate => we are done !
                break;
            }
        }

        if (options.add_eos) {
            tok_buff[num_tokens] = self.special_tokens.eos;
            num_tokens += 1;
        }
        if (options.add_bos) {
            tok_buff_alloc[0] = self.special_tokens.bos;
            num_tokens += 1;
        }
        if (num_tokens < options.pad_to) {
            for (num_tokens..options.pad_to) |i| {
                tok_buff_alloc[i] = self.special_tokens.pad;
            }
            num_tokens = options.pad_to;
        }

        // Release extra memory we don't need anymore.
        allocator.free(mergeable);
        _ = allocator.resize(tok_buff_alloc, num_tokens);
        return tok_buff_alloc[0..num_tokens];
    }

    /// Returns a slice corresponding to the given id. Handles unknown ids and special ids.
    pub fn lookupPiece(self: *const Tokenizer, id: usize) []const u8 {
        return if (id == self.special_tokens.bos or id == self.special_tokens.eos or id == self.special_tokens.pad)
            ""
        else if (id == self.special_tokens.unk)
            "<unk>"
        else if (id > self.tokens.len)
            std.debug.panic("Unexpected token id: {d}, vocab_size: {d}", .{ id, self.vocab_size })
        else
            self.tokens[id];
    }

    /// Converts the given slice of tokens back into bytes.
    /// Note that if the tokenizer allows sub-unicode bytes, it's possible
    /// the output is not valid utf8.
    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, input: []const u32) error{OutOfMemory}![]u8 {
        var output = std.ArrayList(u8).init(allocator);
        errdefer output.deinit();

        try self.decodeWithOpts(&output, input, .{});
        return output.toOwnedSlice();
    }

    pub fn decodeWithOpts(
        self: *const Tokenizer,
        output: *std.ArrayList(u8),
        input: []const u32,
        opts: struct { sep: []const u8 = "" },
    ) error{OutOfMemory}!void {
        // Flag used to indicate if the first dummy whitespace has been consumed.
        for (input) |id| {
            // Retrieve the slice corresponding to the id.
            var piece = self.lookupPiece(id);

            // Convert `▁` to a regular space.
            if (std.mem.startsWith(u8, piece, Normalizer.space_symbol)) {
                piece = piece[Normalizer.space_symbol.len..];

                // don't output a space at beginning of text.
                if (output.items.len > 0) try output.append(' ');
            }

            try output.appendSlice(piece);
            if (opts.sep.len > 0) try output.appendSlice(opts.sep);
        }
    }

    /// Some tokenizers have bytes encoded in hex like this: "<0x40>".
    /// This break the tokenization algorithm because the input text
    /// will contain "@" not "<0x40>",
    /// and if the input contains "<0x40>" it needs to not be treated as a single byte.
    /// So we replace byte fallbacks strings, by their corresponding character.
    /// This enables the normal tokenization algorithm to work.
    pub fn rewriteByteFallbackTokens(tokenizer: *Tokenizer) !void {
        tokenizer.byte_fallback = true;
        var single_bytes = try tokenizer.arena_state.allocator().alloc(u8, 256);
        var byte_fallback_buf = "<0x00>".*;

        for (0..256) |i| {
            const c: u8 = @truncate(i);
            single_bytes[i] = c;

            // First lookup the byte fallback entry.
            // Note: we assume upper case, but we could try both upper and lower case if needed.
            _ = std.fmt.bufPrintIntToSlice(byte_fallback_buf[3..5], c, 16, .upper, .{ .fill = '0', .width = 2 });
            const entry = tokenizer.token_lookup.getEntry(&byte_fallback_buf) orelse {
                log.err("Tokenizer has \"byte_fallback\" = true, but doesn't contains the byte fallback token {s}", .{byte_fallback_buf});
                return error.InvalidInput;
            };

            // Check if the character is already present in the vocab.
            // In that case, nothing to do,
            // but note that the fallback token will be "unreachable",
            // ie there is no way the tokenizer can produce it.
            if (tokenizer.token_lookup.get(&.{c})) |_| continue;

            const idx: u32 = entry.value_ptr.*;
            tokenizer.token_lookup.removeByPtr(entry.key_ptr);
            tokenizer.addOwnedTokenByIndex(idx, tokenizer.scores[idx], single_bytes[i .. i + 1]);
        }
    }
};

test Tokenizer {
    const allocator = std.testing.allocator;
    const special_tokens: Tokenizer.SpecialTokens = .{
        .unk = 0,
        .bos = 1,
        .eos = 2,
    };

    var tokenizer = try Tokenizer.init(allocator, 10, 5, .{}, special_tokens, true);
    defer tokenizer.deinit();

    try tokenizer.addToken(10, "hello");
    try tokenizer.addToken(3.5, "world");

    try testing.expect(tokenizer.lookup("hello") == 0);
    try testing.expect(tokenizer.lookup("world") == 1);

    // TODO: test Tokenizer.decode, Tokenizer.encode, Tokenizer.readTokenInto
}

/// Given a slice, split it in the most simple tokens using the given tokenizer tokens.
/// The output of this can be used to initialize the tokenization algorithm.
/// Normally we split the input text into utf8 codepoint,
/// but if we find an unknown codepoint we either split it in bytes, or use the special "unknown" token,
/// depending on the tokenizer configuration.
const CharTokenIterator = struct {
    state: union(enum) { by_codepoint, by_byte: u8 } = .by_codepoint,
    input: []const u8,

    fn nextCodepointToken(self: *CharTokenIterator, tokenizer: *const Tokenizer) error{ TruncatedInput, Utf8InvalidStartByte }!?u32 {
        if (self.input.len == 0) return null;
        return switch (self.state) {
            .by_byte => |*byte_left| {
                const idx = tokenizer.lookup(self.input[0..1]) orelse {
                    // Normally this has been caught when calling `rewriteByteFallbackTokens`.
                    std.debug.panic("Tokenizer has \"byte_fallback\" = true, but doesn't contains the byte fallback for token '<0x{X:02}>'", .{self.input[0]});
                };

                self.input = self.input[1..];
                byte_left.* -|= 1;
                if (byte_left.* == 0) self.state = .by_codepoint;
                return idx;
            },
            .by_codepoint => {
                // Try to lookup valid utf8 codepoint first.
                const utf8_len = try std.unicode.utf8ByteSequenceLength(self.input[0]);
                if (self.input.len < utf8_len) return error.TruncatedInput;
                if (tokenizer.lookup(self.input[0..utf8_len])) |idx| {
                    self.input = self.input[utf8_len..];
                    return idx;
                }

                // Otherwise split in bytes if it's allowed.
                if (tokenizer.byte_fallback) {
                    // TODO: replace this by a continue statement next time we bump Zig.
                    self.state = .{ .by_byte = utf8_len };
                    return self.nextCodepointToken(tokenizer);
                }

                // Or mark the full utf8 codepoint as unknown.
                log.debug("Token not found for char '{s}'", .{self.input[0..utf8_len]});
                self.input = self.input[utf8_len..];
                return tokenizer.special_tokens.unk;
            },
        };
    }
};

test CharTokenIterator {
    const special_tokens: Tokenizer.SpecialTokens = .{ .unk = 0, .bos = 1, .eos = 2 };
    var tokenizer = try Tokenizer.init(std.testing.allocator, 16, 4, null, special_tokens, true);
    defer tokenizer.deinit();

    tokenizer.addOwnedToken(1.0, "<unk>"); // 0
    tokenizer.addOwnedToken(1.0, "<s>"); // 1
    tokenizer.addOwnedToken(1.0, "</s>"); // 2
    tokenizer.addOwnedToken(1.0, "ζ"); // 3
    tokenizer.addOwnedToken(1.0, &.{0xE2}); // 4: ℳ, first byte
    tokenizer.addOwnedToken(1.0, &.{0x84}); // 5: ℳ, second byte
    tokenizer.addOwnedToken(1.0, &.{0xB3}); // 6: ℳ, third byte
    tokenizer.addOwnedToken(1.0, "L"); // 7

    // No byte fallback
    {
        tokenizer.byte_fallback = false;
        var it: CharTokenIterator = .{ .input = "ζℳL" };
        var res: std.BoundedArray(u32, 8) = .{};
        while (try it.nextCodepointToken(&tokenizer)) |token| {
            res.appendAssumeCapacity(token);
        }
        try std.testing.expectEqualSlices(u32, &[_]u32{ 3, 0, 7 }, res.constSlice());
    }

    // with byte fallback
    {
        tokenizer.byte_fallback = true;
        var it: CharTokenIterator = .{ .input = "ζℳL" };
        var res: std.BoundedArray(u32, 8) = .{};
        while (try it.nextCodepointToken(&tokenizer)) |token| {
            res.appendAssumeCapacity(token);
        }
        try std.testing.expectEqualSlices(u32, &[_]u32{ 3, 4, 5, 6, 7 }, res.constSlice());
    }
}

/// Text normalizer. Most tokenizer assumes the input text have been prepocessed
/// with on of those.
pub const Normalizer = struct {
    pub const space_symbol = "▁"; // \xe2\x96\x81

    flags: packed struct {
        escape_whitespaces: bool = true,
        remove_extra_whitespaces: bool = true,
        add_dummy_prefix: bool = true,
        add_dummy_suffix: bool = false,
        /// Cheap lower casing.
        /// TODO: try to match Python "lower"
        lower_case_ascii: bool = false,
        /// cheap ascii punct splitting.
        // doing this processing ahead of time simplifies the logic
        split_on_punct_ascii: bool = false,
    } = .{},

    fn addSlice(data: []const u8, consumed: usize, normalized: *std.ArrayList(u8), normalized_to_origin: *std.ArrayList(usize)) !void {
        try normalized.appendSlice(data);
        for (data) |_| try normalized_to_origin.append(consumed);
    }

    pub const Result = struct {
        /// Normalized string
        normalized: []const u8,
        /// Mapping between chars in the original string and chars in the new string
        normalized_to_origin: []const usize,

        pub fn deinit(self: Result, allocator: std.mem.Allocator) void {
            allocator.free(self.normalized);
            allocator.free(self.normalized_to_origin);
        }
    };

    /// Simplifed version of Sentencepiece normalizer.
    ///
    /// Llama2 uses a normalizer called "identity" so this basically only handles trailing
    /// whitespaces and replaces whitespace with the "▁" (U+2581) character.
    pub fn normalize(self: Normalizer, allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
        const res = try self.normalizeWithMapping(allocator, input);
        allocator.free(res.normalized_to_origin);
        return res.normalized;
    }

    /// Returns both the normalized string and a mapping between the normalized string and the original.
    pub fn normalizeWithMapping(self: Normalizer, allocator: std.mem.Allocator, input: []const u8) !Result {
        // Number of bytes consumed from the input.
        var consumed: usize = 0;
        var trimmed_input = input;

        // Skip leading whitespaces.
        if (self.flags.remove_extra_whitespaces) {
            while (trimmed_input.len != 0) {
                if (trimmed_input[0] != ' ') break;
                trimmed_input = trimmed_input[1..];
                consumed += 1;
            }
        }

        // If the trimmed input is empty, we are done.
        if (trimmed_input.len == 0) {
            return .{ .normalized = &.{}, .normalized_to_origin = &.{} };
        }

        // Pre-allocate outputs
        const space = if (self.flags.escape_whitespaces) Normalizer.space_symbol else " ";
        const overhead = if (self.flags.split_on_punct_ascii) space.len + 1 else space.len;
        var normalized = try std.ArrayList(u8).initCapacity(allocator, trimmed_input.len * overhead + 2 * space.len);
        errdefer normalized.deinit();
        var normalized_to_origin = try std.ArrayList(usize).initCapacity(allocator, normalized.capacity);
        errdefer normalized_to_origin.deinit();

        // If the spec asks for it, add a whitespace at the beginning.
        if (self.flags.add_dummy_prefix) try addSlice(space, consumed, &normalized, &normalized_to_origin);

        var is_prev_space: bool = true;
        var is_prev_word: bool = false;

        while (trimmed_input.len != 0) {
            // NOTE(Corendos): This might feel weird but normally the slice we get comes from a normalizing process and can contain multiple codepoints.
            // Since we have an "identity" normalizer, each slice is actually a unicode character.
            const multibyte_length = try std.unicode.utf8ByteSequenceLength(trimmed_input[0]);
            var slice = trimmed_input[0..multibyte_length];
            const origin = consumed;
            consumed += multibyte_length;
            trimmed_input = trimmed_input[multibyte_length..];

            if (self.flags.remove_extra_whitespaces and is_prev_space) {
                while (slice.len > 0 and slice[0] == ' ') {
                    slice = slice[1..];
                }
                if (slice.len == 0) continue;
            }
            is_prev_space = slice[slice.len - 1] == ' ';

            if (slice.len == 1) ascii: {
                // The more advanced logic only works with ascii atm
                var byte = slice[0];
                if (self.flags.escape_whitespaces and byte == ' ') {
                    // replace the space token by the special token
                    try addSlice(space, origin, &normalized, &normalized_to_origin);
                    is_prev_word = false;
                    break :ascii;
                } else if (self.flags.split_on_punct_ascii) {
                    if (is_prev_word and isPunct(slice)) {
                        // Insert a space, but continue handling the rest
                        try addSlice(space, origin, &normalized, &normalized_to_origin);
                    }
                }
                if (self.flags.lower_case_ascii) {
                    byte = std.ascii.toLower(byte);
                }
                try normalized.append(byte);
                try normalized_to_origin.append(origin);
            } else {
                // we can safely copy to the output.
                try addSlice(slice, origin, &normalized, &normalized_to_origin);
            }
            is_prev_word = !is_prev_space and !isPunct(slice);
        }

        // Skip trailing whitespaces
        if (self.flags.remove_extra_whitespaces) {
            while (std.mem.endsWith(u8, normalized.items, space)) {
                const length = normalized.items.len - space.len;
                consumed = normalized_to_origin.items[length];
                try normalized.resize(length);
                try normalized_to_origin.resize(length);
            }
        }

        try normalized_to_origin.append(consumed);

        std.debug.assert(normalized_to_origin.items.len == normalized.items.len + 1);

        if (self.flags.add_dummy_suffix) try addSlice(space, consumed, &normalized, &normalized_to_origin);

        return .{
            .normalized = try normalized.toOwnedSlice(),
            .normalized_to_origin = try normalized_to_origin.toOwnedSlice(),
        };
    }

    pub fn wellKnown(impl: KnownImplementation) Normalizer {
        return switch (impl) {
            .sentencepiece => .{ .flags = .{
                .escape_whitespaces = true,
                .remove_extra_whitespaces = true,
                .add_dummy_prefix = true,
                .add_dummy_suffix = false,
                .lower_case_ascii = false,
                .split_on_punct_ascii = false,
            } },
            .gpt2 => .{ .flags = .{
                .escape_whitespaces = false,
                .remove_extra_whitespaces = true,
                .add_dummy_prefix = true,
                .add_dummy_suffix = false,
                .lower_case_ascii = false,
                .split_on_punct_ascii = false,
            } },
        };
    }
};

pub const KnownImplementation = enum(u8) {
    sentencepiece,
    gpt2,
};

fn isPunct(unicode_char: []const u8) bool {
    // TODO use unicode categories
    if (unicode_char.len > 1) return false;

    return switch (unicode_char[0]) {
        ' ', '\t' => false,
        0...8 => true,
        10...31 => true,
        '!'...'/' => true,
        ':'...'@' => true,
        '['...'`' => true,
        '{'...'~' => true,
        else => false,
    };
}

test Normalizer {
    try testing.expectEqualSlices(u8, "▁", Normalizer.space_symbol);

    {
        const n: Normalizer = .{ .flags = .{
            .escape_whitespaces = false,
            .remove_extra_whitespaces = true,
            .add_dummy_prefix = true,
            .add_dummy_suffix = false,
        } };
        const res = try n.normalizeWithMapping(testing.allocator, "Hellŏ  world!");
        defer res.deinit(testing.allocator);

        try testing.expectEqualSlices(u8, " Hellŏ world!", res.normalized);
        try testing.expectEqualSlices(
            usize,
            // H     e  l  l  ŏ     ␣  w  o  r   l   d   !
            &.{ 0, 0, 1, 2, 3, 4, 4, 6, 8, 9, 10, 11, 12, 13, 14 },
            res.normalized_to_origin,
        );
    }

    {
        const n: Normalizer = .{ .flags = .{
            .escape_whitespaces = false,
            .remove_extra_whitespaces = true,
            .add_dummy_prefix = true,
            .add_dummy_suffix = true,
        } };
        const res = try n.normalize(testing.allocator, "Hello  world!");
        defer testing.allocator.free(res);

        try testing.expectEqualSlices(u8, " Hello world! ", res);
    }

    {
        const n: Normalizer = .{ .flags = .{
            .escape_whitespaces = true,
            .remove_extra_whitespaces = false,
            .add_dummy_prefix = true,
            .add_dummy_suffix = false,
        } };
        const res = try n.normalize(testing.allocator, "Hello  world!");
        defer testing.allocator.free(res);

        try testing.expectEqualSlices(u8, "▁Hello▁▁world!", res);
    }

    {
        const n: Normalizer = .{ .flags = .{
            .escape_whitespaces = false,
            .remove_extra_whitespaces = true,
            .add_dummy_prefix = false,
            .add_dummy_suffix = true,
            .lower_case_ascii = true,
        } };
        const res = try n.normalize(testing.allocator, "Hello  world!");
        defer testing.allocator.free(res);

        try testing.expectEqualSlices(u8, "hello world! ", res);
    }

    {
        const n: Normalizer = .{ .flags = .{
            .escape_whitespaces = false,
            .remove_extra_whitespaces = true,
            .add_dummy_prefix = false,
            .add_dummy_suffix = true,
            .split_on_punct_ascii = true,
        } };
        const res = try n.normalize(testing.allocator, "Hello  world!");
        defer testing.allocator.free(res);

        try testing.expectEqualSlices(u8, "Hello world ! ", res);
    }
}

/// gpt2 had their own way of storing text.
/// Unfortunately this has contaminated other models.
/// This implementation precompupte a mapping between bytes encoded with GPT2 algorithm,
/// into utf8 bytes, and do lookups at runtime.
pub const Gpt2TextDecoder = struct {
    const Code = std.BoundedArray(u8, 2);

    // TODO: benchmark this is more efficient than doing the conversion at runtime.
    code_to_byte: std.AutoArrayHashMap(Code, u8),

    pub fn init(allocator: std.mem.Allocator) !Gpt2TextDecoder {
        var self = Gpt2TextDecoder{
            .code_to_byte = std.AutoArrayHashMap(Code, u8).init(allocator),
        };
        try self.code_to_byte.ensureTotalCapacity(256);
        errdefer unreachable;

        var n: usize = 0;
        for (0..256) |index| {
            var code: Code = .{ .buffer = .{ 0, 0 }, .len = 0 }; // 0-init
            const i: u8 = @intCast(index);
            if (isPrintableByte(i)) {
                if (std.ascii.isASCII(i)) {
                    code.appendAssumeCapacity(i);
                } else {
                    const codepoint: u21 = @as(u21, @intCast(i));
                    code.len = @intCast(std.unicode.utf8Encode(codepoint, &code.buffer) catch unreachable);
                }
            } else {
                const codepoint: u21 = 256 + @as(u21, @intCast(n));
                code.len = @intCast(std.unicode.utf8Encode(codepoint, &code.buffer) catch unreachable);
                n += 1;
            }

            self.code_to_byte.putAssumeCapacityNoClobber(code, i);
        }
        return self;
    }

    pub fn deinit(self: *Gpt2TextDecoder) void {
        self.code_to_byte.deinit();
    }

    /// Transform bytes representing text under the gpt2 encoding,
    /// and write to the `unicode` buffer utf-8 bytes.
    pub fn decode(self: Gpt2TextDecoder, unicode: *std.ArrayList(u8), bytes: []const u8) ![]const u8 {
        const start = unicode.items.len;
        var it = std.unicode.Utf8Iterator{ .i = 0, .bytes = bytes };
        while (it.nextCodepointSlice()) |codepoint| {
            const code: Code = switch (codepoint.len) {
                1 => .{ .buffer = .{ codepoint[0], 0 }, .len = 1 }, // 0-init
                2 => .{ .buffer = .{ codepoint[0], codepoint[1] }, .len = 2 },
                else => return error.InvalidInput,
            };
            const byte = self.code_to_byte.get(code) orelse return error.InvalidInput;
            try unicode.append(byte);
        }
        return unicode.items[start..];
    }

    inline fn isPrintableByte(c: u8) bool {
        return ('!' <= c and c <= '~') or (0xa1 <= c and c <= 0xac) or (0xae <= c and c <= 0xff);
    }
};

test Gpt2TextDecoder {
    var decoder = try Gpt2TextDecoder.init(testing.allocator);
    defer decoder.deinit();

    var out = std.ArrayList(u8).init(testing.allocator);
    defer out.deinit();

    // Ascii is not changed.
    try testing.expectEqualStrings("getTitle", try decoder.decode(&out, "getTitle"));
    // Leading space are represented with 'Ġ'
    try testing.expectEqualStrings(" UINavigationController", try decoder.decode(&out, "ĠUINavigationController"));
    // Russian is wild
    try testing.expectEqualStrings(" работ", try decoder.decode(&out, "ĠÑĢÐ°Ð±Ð¾ÑĤ"));
}

/// Open a json file in HF format and load the vocab from it.
pub fn fromHfJson(allocator: std.mem.Allocator, tokenizer_path: []const u8) !Tokenizer {
    const file = try std.fs.cwd().openFile(tokenizer_path, .{});
    defer file.close();

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const file_content = try file.readToEndAlloc(arena, 32 * 1024 * 1024);

    const info = try std.json.parseFromSliceLeaky(std.json.Value, arena, file_content, .{
        .duplicate_field_behavior = .use_last,
    });
    const main_object = switch (info) {
        .object => |obj| if (obj.get("added_tokens") == null or obj.get("model") == null) {
            return error.InvalidFormat;
        } else obj,
        else => return error.InvalidFormat,
    };

    // TODO: remove all panics
    const added_tokens = main_object.get("added_tokens").?.array;
    const vocab = main_object.get("model").?.object.get("vocab").?.object;
    const vocab_size: u32 = @intCast(vocab.count() + added_tokens.items.len);

    const normalizer = Normalizer.wellKnown(.gpt2);
    var gpt2_decoder = try Gpt2TextDecoder.init(allocator);
    defer gpt2_decoder.deinit();

    var tokenizer = try Tokenizer.init(allocator, vocab_size, 256, normalizer, undefined, true);
    errdefer tokenizer.deinit();

    // Buffer containing all concatenated tokens.
    // Reserve a big chunk, to avoid grow event, but release over-allocated memory.
    var all_tokens = try std.ArrayList(u8).initCapacity(tokenizer.arena_state.allocator(), file_content.len);
    const original_alloc = all_tokens.items.ptr;
    // A re-alloc event here means we have invalidated all slices inside the tokenizer.
    // If this is too annoying we could switch to a custom type instead of slices.
    defer {
        std.debug.assert(all_tokens.items.ptr == original_alloc);
    }

    var it = vocab.iterator();
    // gpt2 based tokenizer got a special way of encoding unicode.
    // we don't know in advance if this will be used by this tokenizer or not.
    // so we assume it is the case, but if we find some unicode character,
    // outside of the range used by gpt2 we know it was wrong, and start over.
    var is_gpt2_vocab: bool = true;
    while (it.next()) |kv| {
        const token = gpt2_decoder.decode(&all_tokens, kv.key_ptr.*) catch |err| {
            switch (err) {
                error.InvalidInput => {
                    is_gpt2_vocab = false;
                    break;
                },
                else => return err,
            }
        };
        const idx: u32 = @intCast(kv.value_ptr.*.integer);
        tokenizer.addOwnedTokenByIndex(idx, @floatFromInt(vocab_size - idx), token);
    }

    if (!is_gpt2_vocab) {
        // We where wrong, this is not a gpt2 vocab, start over,
        // and reset the tokenizer state.
        tokenizer.next_token_id = 0;
        tokenizer.token_lookup.clearRetainingCapacity();
        all_tokens.clearRetainingCapacity();
        it = vocab.iterator();
        while (it.next()) |kv| {
            const idx: u32 = @intCast(kv.value_ptr.*.integer);
            const token = try dup(&all_tokens, kv.key_ptr.*);
            tokenizer.addOwnedTokenByIndex(idx, @floatFromInt(vocab_size - idx), token);
        }
    }

    // More tokens, typically added during fine tuning of the model.
    for (added_tokens.items) |token_obj| {
        const v = token_obj.object.get("content").?.string;
        const id: u32 = @intCast(token_obj.object.get("id").?.integer);
        const token = try if (is_gpt2_vocab)
            gpt2_decoder.decode(&all_tokens, v)
        else
            dup(&all_tokens, v);

        tokenizer.addOwnedTokenByIndex(id, 0, token);
    }
    // We won't add more tokens here, let release.
    all_tokens.shrinkAndFree(all_tokens.items.len);

    tokenizer.special_tokens = .{
        .bos = tokenizer.lookup("<s>") orelse tokenizer.lookup("<|begin_of_text|>") orelse @panic("bos token not found !"),
        .eos = tokenizer.lookup("</s>") orelse tokenizer.lookup("<|end_of_text|>") orelse @panic("eos token not found !"),
        .unk = tokenizer.lookup("<unk>") orelse std.math.maxInt(u32),
    };

    if (main_object.get("model").?.object.get("byte_fallback")) |byte_fallback| {
        if (byte_fallback == .bool and byte_fallback.bool) {
            try tokenizer.rewriteByteFallbackTokens();
        }
    }
    return tokenizer;
}

/// Returns a copy of the given string, stored inside the given ArrayList.
fn dup(buffer: *std.ArrayList(u8), str: []const u8) ![]const u8 {
    const n = buffer.items.len;
    try buffer.appendSlice(str);
    return buffer.items[n..];
}
