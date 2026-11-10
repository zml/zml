//! Host-side Phonon tokenizer, including its JSON-defined text normalization.

// TODO: all allocations in here are a mess of wastefulness, but very low prio

const std = @import("std");
const zml = @import("zml");

const util = @import("../util.zig");

const c = @import("c");

const Tokenizer = @This();

inner: zml.tokenizer.Tokenizer,
definition: std.json.Parsed(std.json.Value),
normalizers: []Normalizer,
allocator: std.mem.Allocator,

/// Loads a local tokenizer JSON definition through the supplied IO.
pub fn init(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Tokenizer {
    const here = @typeName(@This()) ++ ".init";

    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    const bytes = blk: {
        var span_read = zml.tracer.span_at(@src(), here ++ "/read", .{});
        defer span_read.end();

        break :blk try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    };
    defer allocator.free(bytes);

    var span_process = zml.tracer.span_at(@src(), here ++ "/process", .{});
    defer span_process.end();

    var definition = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{ .allocate = .alloc_always });
    errdefer definition.deinit();

    const steps = definition.value.object.get("normalizer").?.object.get("normalizers").?.array.items;
    const normalizers = try allocator.alloc(Normalizer, steps.len);
    errdefer allocator.free(normalizers);

    var initialized: usize = 0;
    errdefer for (normalizers[0..initialized]) |normalizer| normalizer.deinit();
    for (steps, normalizers) |step, *normalizer| {
        normalizer.* = try .init(step);
        initialized += 1;
    }

    // Match the checkpoint's literal special tokens before normalizing text.
    for (definition.value.object.get("added_tokens").?.array.items) |token| {
        for ([_][]const u8{ "normalized", "single_word", "lstrip", "rstrip" }) |flag| {
            if (token.object.get(flag).?.bool) return error.UnsupportedAddedToken;
        }
    }

    // IREE cannot load the 191-step pipeline or its lookaround regexes.
    // Apply those rules here; retain IREE for Unigram segmentation and IDs.
    var backend_definition = definition.value;

    const original_normalizer = backend_definition.object.get("normalizer").?;
    backend_definition.object.getPtr("normalizer").?.* = .null;

    const backend_json = try std.json.Stringify.valueAlloc(allocator, backend_definition, .{});
    defer allocator.free(backend_json);

    backend_definition.object.getPtr("normalizer").?.* = original_normalizer;

    const inner = try zml.tokenizer.Tokenizer.fromBytes(allocator, backend_json);
    return .{ .inner = inner, .definition = definition, .normalizers = normalizers, .allocator = allocator };
}

pub fn deinit(self: *Tokenizer) void {
    const here = @typeName(@This()) ++ ".deinit";

    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    self.inner.deinit();
    for (self.normalizers) |normalizer| normalizer.deinit();
    self.allocator.free(self.normalizers);
    self.definition.deinit();
}

/// Tokenizes one UTF-8 string on the CPU. The string must already contain
/// any conversation formatting and expanded audio placeholders.
///
/// Returns an owned i64 tensor matching Python's input_ids dtype:
/// - `.b`: batch, fixed to 1 because this method accepts a single string.
/// - `.s`: sequence, one integer token ID per position, with no padding.
/// Shape: `[.b=1, .s=number_of_tokens]`. Free with `result.free(allocator)`.
pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) !zml.Slice {
    const here = @typeName(@This()) ++ ".encode";

    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    var ids: std.ArrayList(i64) = .empty;
    defer ids.deinit(allocator);

    var start: usize = 0;
    var position: usize = 0;
    const added_tokens = self.definition.value.object.get("added_tokens").?.array.items;
    while (position < text.len) {
        var match: ?std.json.Value = null;
        var match_length: usize = 0;
        for (added_tokens) |token| {
            const content = token.object.get("content").?.string;
            if (content.len > match_length and std.mem.startsWith(u8, text[position..], content)) {
                match = token;
                match_length = content.len;
            }
        }
        if (match) |token| {
            try self.encode_text(allocator, text[start..position], &ids);
            try ids.append(allocator, token.object.get("id").?.integer);
            position += match_length;
            start = position;
        } else position += 1;
    }
    try self.encode_text(allocator, text[start..], &ids);

    const result = try zml.Slice.alloc(allocator, .init(.{ .b = 1, .s = ids.items.len }, .i64));
    @memcpy(result.items(i64), ids.items);

    return result;
}

fn encode_text(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, ids: *std.ArrayList(i64)) !void {
    const here = @typeName(@This()) ++ ".encode_text";

    if (text.len == 0) return;

    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    var normalized = try allocator.dupe(u8, text);
    defer allocator.free(normalized);
    for (self.normalizers) |normalizer| {
        const next = try normalizer.apply(allocator, normalized);
        allocator.free(normalized);
        normalized = next;
    }

    var encoder = try self.inner.encoder();
    defer encoder.deinit();
    const tokens = try encoder.encodeAlloc(allocator, normalized);
    defer allocator.free(tokens);
    for (tokens) |id| try ids.append(allocator, @intCast(id));
}

/// One normalized, decoded text chunk. num_tokens excludes voice and BOS markers.
pub const Text_Chunk = struct {
    text: []const u8,
    num_tokens: usize,
};

pub const Text_Chunks = struct {
    items: []Text_Chunk,
    /// Index of the first chunk with the largest token count; null for empty input.
    longest_chunk: ?usize,
    max_text_tokens: usize,

    pub fn deinit(self: Text_Chunks, allocator: std.mem.Allocator) void {
        for (self.items) |chunk| allocator.free(chunk.text);
        allocator.free(self.items);
    }
};

/// PocketTTSProcessor.split_text's sentence boundaries and greedy grouping,
/// with max_tokens as a soft target: sentences are never split into clauses.
/// Returns owned chunks in speaking order. Empty text produces no chunks.
pub fn split_text(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, max_tokens: usize) !Text_Chunks {
    const here = @typeName(@This()) ++ ".split_text";
    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();
    std.debug.assert(max_tokens > 0);

    // The text-only chat template trims its input. The processor removes audio
    // placeholders before tokenizing it; do the same before normalization.
    var cleaned: std.Io.Writer.Allocating = .init(allocator);
    defer cleaned.deinit();
    const trimmed = std.mem.trim(u8, text, " \t\r\n\x0b\x0c");
    var position: usize = 0;
    while (position < trimmed.len) {
        var placeholder_len: usize = 0;
        for (self.definition.value.object.get("added_tokens").?.array.items) |token| {
            const id = token.object.get("id").?.integer;
            if (id != 4 and id != 5 and id != 6) continue;
            const content = token.object.get("content").?.string;
            if (std.mem.startsWith(u8, trimmed[position..], content)) placeholder_len = @max(placeholder_len, content.len);
        }
        if (placeholder_len > 0) {
            position += placeholder_len;
        } else {
            try cleaned.writer.writeByte(trimmed[position]);
            position += 1;
        }
    }

    const encoded = try self.encode(allocator, cleaned.written());
    defer encoded.free(allocator);
    const ids = try allocator.alloc(u32, encoded.constItems(i64).len);
    defer allocator.free(ids);
    for (ids, encoded.constItems(i64)) |*dst, src| dst.* = @intCast(src);

    const unknown_id: u32 = @intCast(self.definition.value.object.get("model").?.object.get("unk_id").?.integer);
    const sentence_ends = [_]?u32{
        self.inner.tokenId("."), self.inner.tokenId("!"), self.inner.tokenId("..."), self.inner.tokenId("?"),
    };
    var sentences: std.ArrayList([]const u32) = .empty;
    defer sentences.deinit(allocator);
    var start: usize = 0;
    for (ids, 0..) |id, i| {
        if (i == 0) continue;
        if (is_sentence_end(ids[i - 1], &sentence_ends, unknown_id) and
            !is_sentence_end(id, &sentence_ends, unknown_id) and
            !try self.is_decimal_point(allocator, ids, i))
        {
            try sentences.append(allocator, ids[start..i]);
            start = i;
        }
    }
    if (ids.len > 0) try sentences.append(allocator, ids[start..]);

    var chunks: std.ArrayList(Text_Chunk) = .empty;
    errdefer {
        for (chunks.items) |chunk| allocator.free(chunk.text);
        chunks.deinit(allocator);
    }
    var current: std.ArrayList(u8) = .empty;
    defer current.deinit(allocator);
    var current_tokens: usize = 0;
    for (sentences.items) |sentence| {
        const decoded = try self.decode_text(allocator, sentence);
        defer allocator.free(decoded);
        if (current.items.len > 0 and current_tokens + sentence.len > max_tokens) {
            try self.append_chunk(allocator, &chunks, current.items);
            current.clearRetainingCapacity();
            current_tokens = 0;
        }
        if (current.items.len > 0) try current.append(allocator, ' ');
        try current.appendSlice(allocator, decoded);
        current_tokens += sentence.len;
    }
    if (current.items.len > 0) try self.append_chunk(allocator, &chunks, current.items);

    var longest_chunk: ?usize = null;
    var max_text_tokens: usize = 0;
    for (chunks.items, 0..) |chunk, i| {
        if (longest_chunk == null or chunk.num_tokens > max_text_tokens) {
            longest_chunk = i;
            max_text_tokens = chunk.num_tokens;
        }
    }
    return .{
        .items = try chunks.toOwnedSlice(allocator),
        .longest_chunk = longest_chunk,
        .max_text_tokens = max_text_tokens,
    };
}

fn is_sentence_end(id: u32, boundaries: []const ?u32, unknown_id: u32) bool {
    if (id == unknown_id) return false;
    for (boundaries) |boundary| if (boundary == id) return true;
    return false;
}

fn is_decimal_point(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32, index: usize) !bool {
    const prefix = try self.decode_text(allocator, ids[0..index]);
    defer allocator.free(prefix);
    if (prefix.len < 2 or prefix[prefix.len - 1] != '.' or !std.ascii.isDigit(prefix[prefix.len - 2])) return false;
    const suffix = try self.decode_text(allocator, ids[index..]);
    defer allocator.free(suffix);
    return suffix.len > 0 and std.ascii.isDigit(suffix[0]);
}

fn decode_text(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
    var decoder = try self.inner.decoder();
    defer decoder.deinit();
    var decoded = try decoder.decodeAlloc(allocator, ids);
    defer decoded.deinit(allocator);
    return try decoded.toOwnedSlice(allocator);
}

fn append_chunk(self: *const Tokenizer, allocator: std.mem.Allocator, chunks: *std.ArrayList(Text_Chunk), text: []const u8) !void {
    const owned = try allocator.dupe(u8, std.mem.trim(u8, text, " \t\r\n\x0b\x0c"));
    errdefer allocator.free(owned);
    // Decode/join/normalize can change tokenization. Record the actual count
    // that encode_prompt will use, rather than the grouping estimate above.
    const encoded = try self.encode(allocator, owned);
    defer encoded.free(allocator);
    try chunks.append(allocator, .{ .text = owned, .num_tokens = encoded.constItems(i64).len });
}

pub fn encode_prompt(self: *const Tokenizer, gpa: std.mem.Allocator, text: []const u8, voice_frames: usize) !zml.Slice {
    var prompt: std.Io.Writer.Allocating = .init(gpa);
    defer prompt.deinit();
    for (0..voice_frames) |_| try prompt.writer.writeAll("<|audio_prompt|>");
    try prompt.writer.writeAll(std.mem.trim(u8, text, " \t\r\n\x0b\x0c"));
    try prompt.writer.writeAll("<|audio_bos|>");
    return self.encode(gpa, prompt.written());
}

/// Tokenizes on the CPU and transfers the `[.b=1, .s]` i64 IDs to the
/// execution platform. The caller owns the returned buffer and must deinit it.
pub fn encode_buffer(self: *const Tokenizer, gpa: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, text: []const u8) !zml.Buffer {
    const tokens = try self.encode(gpa, text);
    defer tokens.free(gpa);
    return .fromSlice(io, platform, tokens);
}

// ---

/// One host-side tokenizer JSON normalization step; no neural tensors or weights.
/// init consumes a parsed Replace, Strip or Prepend rule. State is either a compiled
/// PCRE2 pattern plus replacement UTF-8 bytes, or a UTF-8 prefix. Text slices borrow
/// storage from Tokenizer.definition or static literals; compiled patterns are owned
/// and freed by deinit.
/// apply consumes UTF-8 text []const u8 and an allocator, returning owned []u8 of
/// variable byte length after replacement/stripping/prepending. The caller frees it.
/// Regex scratch/output storage is allocated on the host during apply.
const Normalizer = union(enum) {
    /// Host replacement payload: pattern is an owned compiled PCRE2 expression;
    /// content is replacement UTF-8 text borrowed from the parsed JSON definition.
    /// Normalizer.apply maps input text bytes to replacement-result bytes.
    replace: struct { pattern: *anyopaque, content: []const u8 },
    prepend: []const u8,

    fn init(value: std.json.Value) !Normalizer {
        const object = value.object;
        const kind = object.get("type").?.string;
        if (std.mem.eql(u8, kind, "Prepend")) return .{ .prepend = object.get("prepend").?.string };
        var pattern: []const u8 = undefined;
        var literal = false;
        var content: []const u8 = "";
        if (std.mem.eql(u8, kind, "Replace")) {
            const spec = object.get("pattern").?.object;
            literal = spec.contains("String");
            pattern = (spec.get(if (literal) "String" else "Regex") orelse return error.InvalidNormalizer).string;
            content = object.get("content").?.string;
        } else if (std.mem.eql(u8, kind, "Strip")) {
            const left = object.get("strip_left").?.bool;
            const right = object.get("strip_right").?.bool;
            pattern = if (left and right) "^\\s+|\\s+$" else if (left) "^\\s+" else if (right) "\\s+$" else "(?!)";
        } else return error.UnsupportedNormalizer;
        const compiled = c.phonon_regex_compile(pattern.ptr, pattern.len, literal) orelse return error.InvalidNormalizerRegex;
        return .{ .replace = .{ .pattern = compiled, .content = content } };
    }

    fn deinit(self: Normalizer) void {
        switch (self) {
            .replace => |rule| c.phonon_regex_free(rule.pattern),
            .prepend => {},
        }
    }

    fn apply(self: Normalizer, allocator: std.mem.Allocator, text: []const u8) ![]u8 {
        switch (self) {
            .prepend => |prefix| return std.mem.concat(allocator, u8, &.{ prefix, text }),
            .replace => |rule| {
                var capacity = text.len + rule.content.len + 16;
                while (true) {
                    const output = try allocator.alloc(u8, capacity);
                    defer allocator.free(output);
                    var length = capacity;
                    switch (c.phonon_regex_replace(rule.pattern, text.ptr, text.len, rule.content.ptr, rule.content.len, output.ptr, &length)) {
                        0 => return allocator.dupe(u8, output[0..length]),
                        1 => capacity = length,
                        else => return error.NormalizationFailed,
                    }
                }
            },
        }
    }
};
