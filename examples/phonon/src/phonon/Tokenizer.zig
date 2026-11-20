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

    const normalized = try self.normalize_text(allocator, text);
    defer allocator.free(normalized);

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

/// Lazy sentence grouping over a streaming Unigram encoder. Owns the normalized
/// document, but retains only pending token IDs and at most one lookahead sentence.
/// next transfers ownership of each returned chunk's text to the caller.
/// The tokenizer must outlive this iterator. deinit releases unread input/state.
pub const Text_Splitter = struct {
    tokenizer: *const Tokenizer,
    allocator: std.mem.Allocator,
    normalized: []u8,
    encoder: zml.tokenizer.Tokenizer.Encoder,
    tokens: std.Io.Writer.Allocating,
    position: usize = 0,
    eof: bool = false,
    sentence_start: usize = 0,
    scan_index: usize = 1,
    sentence_ends: [4]?u32,
    unknown_id: u32,
    max_tokens: usize,
    pending: ?Text_Chunk = null,
    current: std.ArrayList(u8) = .empty,

    pub fn deinit(self: *Text_Splitter) void {
        if (self.pending) |sentence| self.allocator.free(sentence.text);
        self.current.deinit(self.allocator);
        self.tokens.deinit();
        self.encoder.deinit();
        self.allocator.free(self.normalized);
    }

    pub fn next(self: *Text_Splitter, io: std.Io) !?Text_Chunk {
        var span = zml.tracer.span_at(@src(), @typeName(@This()) ++ ".next", .{});
        defer span.end();
        var current_tokens: usize = 0;
        while (true) {
            try io.checkCancel();
            const sentence = if (self.pending) |pending| blk: {
                self.pending = null;
                break :blk pending;
            } else (try self.next_sentence(io)) orelse break;
            if (self.current.items.len > 0 and current_tokens + sentence.num_tokens > self.max_tokens) {
                self.pending = sentence;
                break;
            }
            defer self.allocator.free(sentence.text);
            if (self.current.items.len > 0) try self.current.append(self.allocator, ' ');
            try self.current.appendSlice(self.allocator, sentence.text);
            current_tokens += sentence.num_tokens;
        }
        if (self.current.items.len == 0) return null;
        const chunk = try self.tokenizer.make_chunk(self.allocator, self.current.items);
        self.current.clearRetainingCapacity();
        return chunk;
    }

    fn ids(self: *Text_Splitter) []const u32 {
        return std.mem.bytesAsSlice(u32, @as([]align(@alignOf(u32)) u8, @alignCast(self.tokens.written())));
    }

    fn next_sentence(self: *Text_Splitter, io: std.Io) !?Text_Chunk {
        while (true) {
            const buffered = self.ids();
            // A decimal check needs two suffix tokens. Do not decide a boundary
            // at the end of a feed block until that lookahead arrives (or EOF).
            while (self.scan_index < buffered.len and (self.eof or self.scan_index + 1 < buffered.len)) {
                const i = self.scan_index;
                self.scan_index += 1;
                if (is_sentence_end(buffered[i - 1], &self.sentence_ends, self.unknown_id) and
                    !is_sentence_end(buffered[i], &self.sentence_ends, self.unknown_id) and
                    !try self.tokenizer.is_decimal_point(self.allocator, buffered, i))
                {
                    const text = try self.tokenizer.decode_text(self.allocator, buffered[self.sentence_start..i]);
                    const count = i - self.sentence_start;
                    self.sentence_start = i;
                    return .{ .text = text, .num_tokens = count };
                }
            }
            if (self.eof) {
                if (self.sentence_start == buffered.len) return null;
                const text = try self.tokenizer.decode_text(self.allocator, buffered[self.sentence_start..]);
                const count = buffered.len - self.sentence_start;
                self.sentence_start = buffered.len;
                return .{ .text = text, .num_tokens = count };
            }
            try io.checkCancel();
            try self.feed();
        }
    }

    fn feed(self: *Text_Splitter) !void {
        // Discard consumed sentences before requesting more input. Keeping their
        // IDs would turn this iterator back into an allocation of the whole book.
        const buffered = self.ids();
        const remaining = buffered[self.sentence_start..];
        std.mem.copyForwards(u8, self.tokens.written(), std.mem.sliceAsBytes(remaining));
        self.tokens.shrinkRetainingCapacity(remaining.len * @sizeOf(u32));
        self.scan_index -= self.sentence_start;
        self.sentence_start = 0;

        const encoder = &self.encoder.iree;
        if (self.position == self.normalized.len) {
            try encoder.finalize(&self.tokens.writer);
            self.eof = true;
            return;
        }
        // Bound speculative tokenization and never split a UTF-8 code point.
        var end = @min(self.normalized.len, self.position + 1024);
        while (end < self.normalized.len and self.normalized[end] & 0xc0 == 0x80) end += 1;
        try encoder.feed(self.normalized[self.position..end], &self.tokens.writer);
        self.position = end;
    }
};

/// PocketTTSProcessor.split_text's sentence boundaries and greedy grouping,
/// with max_tokens as a soft target: sentences are never split into clauses.
/// Normalization is one document-wide pass to preserve anchored rules. Encoding
/// and grouping happen on demand in next(), rather than before the first chunk.
pub fn split_text(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8, max_tokens: usize) !Text_Splitter {
    var span = zml.tracer.span_at(@src(), @typeName(@This()) ++ ".split_text", .{});
    defer span.end();
    std.debug.assert(max_tokens > 0);

    // Remove only the processor's three voice/audio placeholders before
    // normalization. Collect them once, instead of searching 204 added tokens
    // at every byte of a potentially book-sized input.
    var placeholders: [3][]const u8 = undefined;
    var num_placeholders: usize = 0;
    const added_tokens = self.definition.value.object.get("added_tokens").?.array.items;
    var special_prefix: [256]bool = @splat(false);
    for (added_tokens) |token| {
        const id = token.object.get("id").?.integer;
        const content = token.object.get("content").?.string;
        special_prefix[content[0]] = true;
        if (id != 4 and id != 5 and id != 6) continue;
        placeholders[num_placeholders] = content;
        num_placeholders += 1;
    }
    var cleaned: std.Io.Writer.Allocating = .init(allocator);
    defer cleaned.deinit();
    const trimmed = std.mem.trim(u8, text, " \t\r\n\x0b\x0c");
    var position: usize = 0;
    while (position < trimmed.len) {
        var placeholder_len: usize = 0;
        for (placeholders[0..num_placeholders]) |content| {
            if (std.mem.startsWith(u8, trimmed[position..], content)) placeholder_len = @max(placeholder_len, content.len);
        }
        if (placeholder_len > 0) {
            position += placeholder_len;
        } else {
            try cleaned.writer.writeByte(trimmed[position]);
            position += 1;
        }
    }

    // As in encode(), literal added tokens bypass normalization. Ordinary text
    // takes the fast path without scanning every added-token string per byte.
    var normalized: std.Io.Writer.Allocating = .init(allocator);
    defer normalized.deinit();
    const input = cleaned.written();
    var start: usize = 0;
    position = 0;
    while (position < input.len) {
        if (!special_prefix[input[position]]) {
            position += 1;
            continue;
        }
        var match_len: usize = 0;
        for (added_tokens) |token| {
            const content = token.object.get("content").?.string;
            if (std.mem.startsWith(u8, input[position..], content)) match_len = @max(match_len, content.len);
        }
        if (match_len == 0) {
            position += 1;
            continue;
        }
        const segment = try self.normalize_text(allocator, input[start..position]);
        defer allocator.free(segment);
        try normalized.writer.writeAll(segment);
        try normalized.writer.writeAll(input[position..][0..match_len]);
        position += match_len;
        start = position;
    }
    const segment = try self.normalize_text(allocator, input[start..]);
    defer allocator.free(segment);
    try normalized.writer.writeAll(segment);

    const owned = try normalized.toOwnedSlice();
    errdefer allocator.free(owned);
    var encoder = try self.inner.encoder();
    errdefer encoder.deinit();
    return .{
        .tokenizer = self,
        .allocator = allocator,
        .normalized = owned,
        .encoder = encoder,
        .tokens = .initAligned(allocator, .of(u32)),
        .sentence_ends = .{ self.inner.tokenId("."), self.inner.tokenId("!"), self.inner.tokenId("..."), self.inner.tokenId("?") },
        .unknown_id = @intCast(self.definition.value.object.get("model").?.object.get("unk_id").?.integer),
        .max_tokens = max_tokens,
    };
}

fn normalize_text(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var span = zml.tracer.span_at(@src(), @typeName(@This()) ++ ".normalize_text", .{});
    defer span.end();
    var normalized = try allocator.dupe(u8, text);
    errdefer allocator.free(normalized);
    if (text.len == 0) return normalized;
    for (self.normalizers) |normalizer| {
        const next = try normalizer.apply(allocator, normalized);
        allocator.free(normalized);
        normalized = next;
    }
    return normalized;
}

fn is_sentence_end(id: u32, boundaries: []const ?u32, unknown_id: u32) bool {
    if (id == unknown_id) return false;
    for (boundaries) |boundary| if (boundary == id) return true;
    return false;
}

fn is_decimal_point(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32, index: usize) !bool {
    // Phonon's boundary tokens are standalone punctuation. The preceding token
    // suffices to check for an ASCII digit before '.', including byte fallback.
    // Never decode the growing prefix: doing so at every boundary is quadratic.
    const prefix = try self.decode_text(allocator, ids[index -| 2..index]);
    defer allocator.free(prefix);
    if (prefix.len < 2 or prefix[prefix.len - 1] != '.' or !std.ascii.isDigit(prefix[prefix.len - 2])) return false;
    // Metaspace can remove the first token's leading space, leaving it empty.
    // Include the next token too; further whitespace or non-ASCII bytes already
    // rule out an ASCII digit at the beginning of the decoded suffix.
    const suffix = try self.decode_text(allocator, ids[index..@min(ids.len, index + 2)]);
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

fn make_chunk(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) !Text_Chunk {
    const owned = try allocator.dupe(u8, std.mem.trim(u8, text, " \t\r\n\x0b\x0c"));
    errdefer allocator.free(owned);
    // Decode/join/normalize can change tokenization. Record the actual count
    // that encode_prompt will use, rather than the grouping estimate above.
    const encoded = try self.encode(allocator, owned);
    defer encoded.free(allocator);
    return .{ .text = owned, .num_tokens = encoded.constItems(i64).len };
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
