const std = @import("std");

const c = @import("c");

const log = std.log.scoped(.@"zml/tokenizer");

pub const Error = error{
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
};

const StatusCodeMask = 0x1F;

inline fn stringView(s: []const u8) c.iree_string_view_t {
    return .{ .data = s.ptr, .size = s.len };
}

inline fn mutableStringView(s: []u8) c.iree_mutable_string_view_t {
    return .{ .data = s.ptr, .size = s.len };
}

inline fn byteSpan(bytes: []u8) c.iree_byte_span_t {
    return .{ .data = bytes.ptr, .data_length = bytes.len };
}

inline fn tokenOutput(ids: []u32) c.iree_tokenizer_token_output_t {
    comptime {
        if (@sizeOf(c.iree_tokenizer_token_id_t) != @sizeOf(u32) or @alignOf(c.iree_tokenizer_token_id_t) != @alignOf(u32)) {
            @compileError("iree_tokenizer_token_id_t must match u32 layout");
        }
    }
    return .{
        .capacity = ids.len,
        .token_ids = @ptrCast(ids.ptr),
        .token_offsets = null,
        .type_ids = null,
    };
}

inline fn tokenIdList(ids: []const u32) c.iree_tokenizer_token_id_list_t {
    comptime {
        if (@sizeOf(c.iree_tokenizer_token_id_t) != @sizeOf(u32) or @alignOf(c.iree_tokenizer_token_id_t) != @alignOf(u32)) {
            @compileError("iree_tokenizer_token_id_t must match u32 layout");
        }
    }
    return .{ .count = ids.len, .values = @ptrCast(ids.ptr) };
}

inline fn statusCode(status: c.iree_status_t) u32 {
    return @intCast(@intFromPtr(status.?) & StatusCodeMask);
}

fn checkOk(status: c.iree_status_t) !void {
    if (status == null) return;
    const code = statusCode(status);
    var buffer: [1024]u8 = undefined;
    var buffer_length: usize = 0;
    if (c.iree_status_format(status, buffer.len, &buffer, &buffer_length)) {
        log.err("IREE tokenizer error: {s}", .{buffer[0..@min(buffer_length, buffer.len - 1)]});
    }
    _ = c.iree_status_ignore(status);
    return switch (code) {
        c.IREE_STATUS_CANCELLED => Error.Cancelled,
        c.IREE_STATUS_UNKNOWN => Error.Unknown,
        c.IREE_STATUS_INVALID_ARGUMENT => Error.InvalidArgument,
        c.IREE_STATUS_DEADLINE_EXCEEDED => Error.DeadlineExceeded,
        c.IREE_STATUS_NOT_FOUND => Error.NotFound,
        c.IREE_STATUS_ALREADY_EXISTS => Error.AlreadyExists,
        c.IREE_STATUS_PERMISSION_DENIED => Error.PermissionDenied,
        c.IREE_STATUS_RESOURCE_EXHAUSTED => Error.ResourceExhausted,
        c.IREE_STATUS_FAILED_PRECONDITION => Error.FailedPrecondition,
        c.IREE_STATUS_ABORTED => Error.Aborted,
        c.IREE_STATUS_OUT_OF_RANGE => Error.OutOfRange,
        c.IREE_STATUS_UNIMPLEMENTED => Error.Unimplemented,
        c.IREE_STATUS_INTERNAL => Error.Internal,
        c.IREE_STATUS_UNAVAILABLE => Error.Unavailable,
        c.IREE_STATUS_DATA_LOSS => Error.DataLoss,
        c.IREE_STATUS_UNAUTHENTICATED => Error.Unauthenticated,
        else => Error.Internal,
    };
}

pub const Tokenizer = struct {
    // TODO: use the ArrayList pattern. Stop copying allocator everywhere
    allocator: std.mem.Allocator,
    inner: *c.iree_tokenizer_t,

    pub fn fromHuggingFaceJson(allocator: std.mem.Allocator, json: []const u8) !Tokenizer {
        // TODO: wrap allocator into an iree allocator
        var raw: ?*c.iree_tokenizer_t = null;
        try checkOk(c.iree_tokenizer_from_huggingface_json(stringView(json), c.iree_allocator_system(), &raw));
        return .{
            .allocator = allocator,
            .inner = raw.?,
        };
    }

    pub fn fromBytes(allocator: std.mem.Allocator, bytes: []const u8) !Tokenizer {
        return fromHuggingFaceJson(allocator, bytes);
    }

    pub fn fromFile(allocator: std.mem.Allocator, io: std.Io, model: []const u8) !Tokenizer {
        const json = try std.Io.Dir.cwd().readFileAlloc(io, model, allocator, .unlimited);
        defer allocator.free(json);
        return try fromHuggingFaceJson(allocator, json);
    }

    pub fn deinit(self: *Tokenizer) void {
        c.iree_tokenizer_free(self.inner);
    }

    pub fn encoder(self: *const Tokenizer) !Encoder {
        return Encoder.init(self);
    }

    pub fn decoder(self: *const Tokenizer) !Decoder {
        return Decoder.init(self, self.allocator);
    }

    pub fn tokenId(self: *const Tokenizer, token: []const u8) ?u32 {
        const vocab = c.iree_tokenizer_vocab(self.inner);
        if (vocab == null) return null;

        // Looks up a string in the vocabulary.
        // Returns the token ID if found, or -1 if not found.
        const id = c.iree_tokenizer_vocab_lookup(vocab, stringView(token));
        return if (id < 0) null else @intCast(id);
    }

    pub const Encoder = struct {
        tokenizer: *const Tokenizer,
        state_storage: []u8,
        transform_buffer: []u8,
        state: *c.iree_tokenizer_encode_state_t,

        fn init(tokenizer: *const Tokenizer) !Encoder {
            var state_size: usize = undefined;
            try checkOk(c.iree_tokenizer_encode_state_calculate_size(tokenizer.inner, &state_size));
            const state_storage = try tokenizer.allocator.alloc(u8, state_size);
            errdefer tokenizer.allocator.free(state_storage);

            const transform_size: usize = c.IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE;
            const transform_buffer = try tokenizer.allocator.alloc(u8, transform_size);
            errdefer tokenizer.allocator.free(transform_buffer);

            var state: ?*c.iree_tokenizer_encode_state_t = null;
            try checkOk(c.iree_tokenizer_encode_state_initialize(
                tokenizer.inner,
                byteSpan(state_storage),
                byteSpan(transform_buffer),
                c.iree_tokenizer_offset_run_list_empty(),
                c.IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START,
                &state,
            ));

            return .{
                .tokenizer = tokenizer,
                .state_storage = state_storage,
                .transform_buffer = transform_buffer,
                .state = state.?,
            };
        }

        pub fn deinit(self: *Encoder) void {
            c.iree_tokenizer_encode_state_deinitialize(self.state);
            self.tokenizer.allocator.free(self.state_storage);
            self.tokenizer.allocator.free(self.transform_buffer);
        }

        pub fn reset(self: *Encoder) void {
            c.iree_tokenizer_encode_state_reset(self.state, c.IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START);
        }

        pub fn encodeAlloc(self: *Encoder, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
            var token_ids = std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
            errdefer token_ids.deinit();

            try self.feed(text, &token_ids.writer);
            try self.finalize(&token_ids.writer);

            const bytes = try token_ids.toOwnedSlice();
            const aligned_bytes: []align(4) u8 = @alignCast(bytes[0..bytes.len]);
            return std.mem.bytesAsSlice(u32, aligned_bytes);
        }

        /// /!\ Assumes the writer provides buffers with proper alignment.
        /// /!\ Remember to call the finalize() method
        pub fn writer(self: *Encoder, buffer: []u8, out: *std.Io.Writer) Writer {
            return Writer.init(self, buffer, out);
        }

        pub const Writer = struct {
            encoder: *Encoder,
            out: *std.Io.Writer,
            interface: std.Io.Writer,

            fn init(encoder_: *Encoder, buffer: []u8, out: *std.Io.Writer) Writer {
                return .{
                    .encoder = encoder_,
                    .out = out,
                    .interface = .{
                        .buffer = buffer,
                        .vtable = &.{
                            .drain = drain,
                            .flush = flush,
                            .rebase = std.Io.Writer.failingRebase,
                        },
                    },
                };
            }

            pub fn finalize(self: *Writer) std.Io.Writer.Error!void {
                try self.interface.flush();
                self.encoder.finalize(self.out) catch |e| {
                    log.err("Finalize failed with {s}", .{@errorName(e)});
                    return error.WriteFailed;
                };
            }

            fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
                if (w.end == 0) return;
                const self: *Writer = @alignCast(@fieldParentPtr("interface", w));
                try self.feed(w.buffer[0..w.end]);
                w.end = 0;
            }

            fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
                const self: *Writer = @alignCast(@fieldParentPtr("interface", w));

                if (w.end != 0) {
                    try self.feed(w.buffer[0..w.end]);
                    w.end = 0;
                }

                var total: usize = 0;
                // Prefix chunks: written once each.
                for (data[0 .. data.len - 1]) |chunk| {
                    try self.feed(chunk);
                    total += chunk.len;
                }
                // Last chunk is the splat pattern.
                const pattern = data[data.len - 1];
                if (pattern.len == 0) return total;
                for (0..splat) |_| {
                    try self.feed(pattern);
                    total += pattern.len;
                }
                return total;
            }

            fn feed(self: *Writer, chunk: []const u8) std.Io.Writer.Error!void {
                self.encoder.feed(chunk, self.out) catch |e| {
                    log.err("Feed failed with {s}", .{@errorName(e)});
                    return error.WriteFailed;
                };
            }
        };

        /// /!\ Assumes the writer provides buffers with proper alignment.
        pub fn feed(self: *Encoder, text: []const u8, out: *std.Io.Writer) !void {
            var min_tokens: usize = @max(text.len / 4, 32);
            var remaining = text;
            while (remaining.len > 0) {
                var consumed: usize = 0;
                var produced: usize = 0;

                checkOk(c.iree_tokenizer_encode_state_feed(
                    self.state,
                    stringView(remaining),
                    tokenOutput(try writableTokenSliceGreedy(out, min_tokens)),
                    &consumed,
                    &produced,
                )) catch |e| switch (e) {
                    error.ResourceExhausted => {
                        out.advance(produced * @sizeOf(u32));
                        min_tokens = @max(32, min_tokens -| produced);
                        remaining = remaining[consumed..];
                        produced = 0;
                        consumed = 0;
                    },
                    else => return e,
                };

                if (produced == 0 and consumed == 0) {
                    min_tokens *= 2;
                    std.debug.assert(min_tokens < 1 << 20);
                    continue;
                }

                out.advance(produced * @sizeOf(u32));
                min_tokens = @max(32, min_tokens -| produced);
                remaining = remaining[consumed..];
            }
        }

        /// /!\ Assumes the writer provides buffers with proper alignment.
        pub fn finalize(self: *Encoder, out: *std.Io.Writer) !void {
            var min_tokens: usize = 4;
            while (true) {
                var produced: usize = 0;

                checkOk(c.iree_tokenizer_encode_state_finalize(
                    self.state,
                    tokenOutput(try writableTokenSliceGreedy(out, min_tokens)),
                    &produced,
                )) catch |e| switch (e) {
                    error.ResourceExhausted => {
                        out.advance(produced * @sizeOf(u32));
                        min_tokens *= 2;
                        std.debug.assert(min_tokens < 1 << 10);
                        continue;
                    },
                    else => return e,
                };

                out.advance(produced * @sizeOf(u32));
                if (!c.iree_tokenizer_encode_state_has_pending(self.state)) break;
                if (produced == 0) {
                    min_tokens *= 2;
                    std.debug.assert(min_tokens < 1 << 20);
                }
            }
        }

        fn writableTokenSliceGreedy(out: *std.Io.Writer, min_tokens: usize) std.Io.Writer.Error![]u32 {
            std.debug.assert(@mod(out.end, @sizeOf(u32)) == 0);
            const bytes = try out.writableSliceGreedy(min_tokens * @sizeOf(u32));
            std.debug.assert(@mod(@intFromPtr(bytes.ptr), 4) == 0);
            const aligned_len = bytes.len - @mod(bytes.len, @sizeOf(u32));
            const aligned_bytes: []align(4) u8 = @alignCast(bytes[0..aligned_len]);
            return std.mem.bytesAsSlice(u32, aligned_bytes);
        }
    };

    pub const Decoder = struct {
        tokenizer: *const Tokenizer,
        state_storage: []u8,
        state: *c.iree_tokenizer_decode_state_t,

        fn init(tokenizer: *const Tokenizer, allocator: std.mem.Allocator) !Decoder {
            var state_size: usize = undefined;
            try checkOk(c.iree_tokenizer_decode_state_calculate_size(tokenizer.inner, &state_size));
            const state_storage = try allocator.alloc(u8, state_size);
            errdefer allocator.free(state_storage);

            var state: ?*c.iree_tokenizer_decode_state_t = null;
            try checkOk(c.iree_tokenizer_decode_state_initialize(
                tokenizer.inner,
                c.IREE_TOKENIZER_DECODE_FLAG_NONE,
                byteSpan(state_storage),
                &state,
            ));

            return .{
                .tokenizer = tokenizer,
                .state_storage = state_storage,
                .state = state.?,
            };
        }

        pub fn deinit(self: *Decoder) void {
            c.iree_tokenizer_decode_state_deinitialize(self.state);
            self.tokenizer.allocator.free(self.state_storage);
        }

        pub fn reset(self: *Decoder) !void {
            c.iree_tokenizer_decode_state_deinitialize(self.state);
            var state: ?*c.iree_tokenizer_decode_state_t = null;
            try checkOk(c.iree_tokenizer_decode_state_initialize(
                self.tokenizer.inner,
                c.IREE_TOKENIZER_DECODE_FLAG_NONE,
                byteSpan(self.state_storage),
                &state,
            ));
            self.state = state.?;
        }

        pub fn feedOne(self: *Decoder, token_id: u32, out: []u8) ![]u8 {
            std.debug.assert(out.len != 0);
            var consumed: usize = 0;
            var produced: usize = 0;
            try checkOk(c.iree_tokenizer_decode_state_feed(
                self.state,
                tokenIdList(&.{token_id}),
                mutableStringView(out),
                &consumed,
                &produced,
            ));
            std.debug.assert(consumed == 1);
            return out[0..produced];
        }

        pub fn finalize(self: *Decoder, out: []u8) ![]u8 {
            std.debug.assert(out.len != 0);
            var produced: usize = 0;
            try checkOk(c.iree_tokenizer_decode_state_finalize(
                self.state,
                mutableStringView(out),
                &produced,
            ));
            return out[0..produced];
        }

        pub fn decode(self: *Decoder, token_ids: []const u32, out: *std.Io.Writer) !void {
            var min_output: usize = @max(token_ids.len * 4, 128);
            var remaining = token_ids;
            while (remaining.len > 0) {
                var consumed: usize = 0;
                var produced: usize = 0;

                checkOk(c.iree_tokenizer_decode_state_feed(
                    self.state,
                    tokenIdList(remaining),
                    mutableStringView(try out.writableSliceGreedy(min_output)),
                    &consumed,
                    &produced,
                )) catch |e| switch (e) {
                    error.ResourceExhausted => {
                        out.advance(produced);
                        min_output = @max(128, min_output -| produced);
                        remaining = remaining[consumed..];
                        produced = 0;
                        consumed = 0;
                    },
                    else => return e,
                };

                if (produced == 0 and consumed == 0) {
                    @branchHint(.unlikely);
                    min_output *= 2;
                    std.debug.assert(min_output < 1 << 20);
                    continue;
                }

                out.advance(produced);
                min_output = @max(128, min_output -| produced);
                remaining = remaining[consumed..];
            }

            min_output = 32;
            while (true) {
                var produced: usize = 0;

                checkOk(c.iree_tokenizer_decode_state_finalize(
                    self.state,
                    mutableStringView(try out.writableSliceGreedy(min_output)),
                    &produced,
                )) catch |e| switch (e) {
                    error.ResourceExhausted => {
                        out.advance(produced);
                        min_output *= 2;
                        std.debug.assert(min_output < 1 << 10);
                        continue;
                    },
                    else => return e,
                };

                out.advance(produced);
                break;
            }
        }
    };
};

pub const Normalizer = struct {
    pub const Kind = union(enum) {
        /// Unicode Normalization Form C.
        NFC,
        /// Unicode Normalization Form D.
        NFD,
        /// Unicode Normalization Form KD.
        NFKD,
        /// Removes Unicode combining marks.
        /// filters out characters in the Unicode Mark category (Mn, Mc, Me).
        StripAccents,
        /// Converts text to lowercase.
        Lowercase,
        /// Substitutes all occurrences of a literal pattern with the specified content.
        Replace: struct {
            kind: enum {
                regex,
                exact,
            },
            pattern: []const u8,
            content: []const u8,
        },
        /// Removes leading and/or trailing whitespace.
        Strip: enum(u8) {
            left = 0b10,
            right = 0b01,
            both = 0b11,
        },
    };

    sequence: *c.iree_tokenizer_normalizer_t,
    sequence_size: usize,

    // Supported normalizer types as of `4d4e97d00f099a21f38eeff26f82a6d9e3643a11`:
    //   - Sequence: Chains multiple normalizers in order
    //   - Lowercase: Unicode case folding
    //   - Strip: Leading/trailing whitespace removal
    //   - Prepend: Prefix string insertion
    //   - StripAccents: Removes combining marks (without NFD)
    //   - BertNormalizer: Combined BERT normalization pipeline
    pub fn fromHuggingFaceJson(json: []const u8) !Normalizer {
        var normalizer: ?*c.iree_tokenizer_normalizer_t = null;
        errdefer if (normalizer) |ptr| c.iree_tokenizer_normalizer_free(ptr);

        try checkOk(c.iree_tokenizer_huggingface_parse_normalizer(.{ .data = json.ptr, .size = json.len }, c.iree_allocator_system(), &normalizer));
        if (normalizer == null) return error.NormalizerAllocationFailed;

        const sequence_size: usize = @intCast(c.iree_tokenizer_normalizer_state_size(normalizer.?));

        return .{
            .sequence = normalizer.?,
            .sequence_size = sequence_size,
        };
    }

    fn createNormalizer(normalizer_kind: Kind) !?*c.iree_tokenizer_normalizer_t {
        var norm: ?*c.iree_tokenizer_normalizer_t = null;

        const iree_allocator = c.iree_allocator_system();

        const status = switch (normalizer_kind) {
            .NFC => c.iree_tokenizer_normalizer_nfc_allocate(iree_allocator, &norm),
            .NFD => c.iree_tokenizer_normalizer_nfd_allocate(iree_allocator, &norm),
            .NFKD => c.iree_tokenizer_normalizer_nfkd_allocate(iree_allocator, &norm),
            .StripAccents => c.iree_tokenizer_normalizer_strip_accents_allocate(iree_allocator, &norm),
            .Lowercase => c.iree_tokenizer_normalizer_lowercase_allocate(iree_allocator, &norm),
            .Replace => |rule| switch (rule.kind) {
                .regex => c.iree_tokenizer_normalizer_regex_replace_allocate(
                    .{ .data = rule.pattern.ptr, .size = rule.pattern.len },
                    .{ .data = rule.content.ptr, .size = rule.content.len },
                    iree_allocator,
                    &norm,
                ),
                .exact => c.iree_tokenizer_normalizer_replace_allocate(
                    .{ .data = rule.pattern.ptr, .size = rule.pattern.len },
                    .{ .data = rule.content.ptr, .size = rule.content.len },
                    iree_allocator,
                    &norm,
                ),
            },
            .Strip => |direction| b: {
                const strip_left = @intFromEnum(direction) & 0b10 != 0;
                const strip_right = @intFromEnum(direction) & 0b01 != 0;
                break :b c.iree_tokenizer_normalizer_strip_allocate(strip_left, strip_right, iree_allocator, &norm);
            },
        };

        try checkOk(status);
        if (norm == null) return error.NormalizerAllocationFailed;
        return norm;
    }

    fn init(allocator: std.mem.Allocator, sequence: []const Kind) !Normalizer {
        if (sequence.len < 0) return error.EmptySequence;

        if (sequence.len == 1) {
            // IREE requires a sequence of at least two normalizer.
            const norm = try createNormalizer(sequence[0]) orelse unreachable;
            const sequence_size: usize = @intCast(c.iree_tokenizer_normalizer_state_size(norm));

            return .{
                .sequence = norm,
                .sequence_size = sequence_size,
            };
        }

        var normalizers: std.ArrayList(*c.iree_tokenizer_normalizer_t) = try .initCapacity(allocator, sequence.len);
        defer normalizers.deinit(allocator);
        errdefer for (normalizers.items) |normalizer| c.iree_tokenizer_normalizer_free(normalizer);

        for (sequence) |normalizer| {
            normalizers.appendAssumeCapacity(try createNormalizer(normalizer) orelse unreachable);
        }

        var normalizer_sequence: ?*c.iree_tokenizer_normalizer_t = null;
        // IREE takes ownership of all the normalizers.
        try checkOk(c.iree_tokenizer_normalizer_sequence_allocate(normalizers.items.ptr, normalizers.items.len, c.iree_allocator_system(), &normalizer_sequence));
        if (normalizer_sequence == null) return error.NormalizerAllocationFailed;

        const sequence_size: usize = @intCast(c.iree_tokenizer_normalizer_state_size(normalizer_sequence.?));
        if (sequence_size == 0) return error.NormalizerAllocationFailed;

        return .{
            .sequence = normalizer_sequence.?,
            .sequence_size = sequence_size,
        };
    }

    pub fn deinit(self: *Normalizer) void {
        c.iree_tokenizer_normalizer_free(self.sequence);
    }

    pub fn normalize(self: *Normalizer, allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
        const storage: []align(16) u8 = try allocator.alignedAlloc(u8, .@"16", self.sequence_size);
        defer allocator.free(storage);

        var state: ?*c.iree_tokenizer_normalizer_state_t = null;
        defer if (state) |ptr| c.iree_tokenizer_normalizer_state_deinitialize(ptr);

        try checkOk(c.iree_tokenizer_normalizer_state_initialize(self.sequence, storage.ptr, &state));
        if (state == null) return error.NormalizationFailed;

        var buffer: [2048]u8 = undefined;
        var remaining = text;

        var writer: std.Io.Writer.Allocating = .init(allocator);
        defer writer.deinit();

        while (remaining.len > 0) {
            var written: usize = 0;
            var consumed: usize = 0;

            try checkOk(c.iree_tokenizer_normalizer_state_process(
                state.?,
                .{ .data = remaining.ptr, .size = remaining.len },
                .{ .data = &buffer, .size = buffer.len },
                c.IREE_TOKENIZER_NORMALIZER_FLAG_SEGMENT_END,
                &consumed,
                &written,
            ));

            if (consumed == 0 and written == 0) return error.NormalizationFailed;

            remaining = remaining[consumed..];
            try writer.writer.writeAll(buffer[0..written]);
        }

        var has_pending = true;
        while (has_pending) {
            var written: usize = 0;

            try checkOk(c.iree_tokenizer_normalizer_state_finalize(state.?, .{ .data = &buffer, .size = buffer.len }, &written));
            has_pending = c.iree_tokenizer_normalizer_state_has_pending(state.?);

            try writer.writer.writeAll(buffer[0..written]);

            if (written == 0 and has_pending) return error.NormalizationFailed;
        }

        return writer.toOwnedSlice();
    }
};

test "normalizer from Hugging Face" {
    const allocator = std.testing.allocator;

    const json =
        \\{
        \\  "type": "Sequence",
        \\  "normalizers": [
        \\    {"type":"NFKD"},
        \\    {"type":"StripAccents"},
        \\    {"type":"Lowercase"},
        \\    {"type":"Replace","pattern":{"Regex":"[ \\t\\r\\n]+"},"content":" "},
        \\    {"type":"Replace","pattern":{"Regex":"^ $"},"content":"\ue000"},
        \\    {"type":"Strip","strip_left":true,"strip_right":true},
        \\    {"type":"Replace","pattern":{"String":"\ue000"},"content":" "}
        \\  ]
        \\}
    ;

    var normalizer = try Normalizer.fromHuggingFaceJson(json);
    defer normalizer.deinit();

    const TestCase = struct {
        input: []const u8,
        expected: []const u8,
    };

    for ([_]TestCase{
        .{ .input = "  hello", .expected = "hello" },
        .{ .input = "hello   ", .expected = "hello" },
        .{ .input = "   hello   ", .expected = "hello" },
        .{ .input = "\nworld\r\n", .expected = "world" },
        .{ .input = "ÉCOLE", .expected = "ecole" },
        .{ .input = "  CAFÉ\tAU\nLAIT  ", .expected = "cafe au lait" },
        .{ .input = "\t \r\n", .expected = " " },
        .{ .input = "", .expected = "" },
    }) |case| {
        const res = try normalizer.normalize(allocator, case.input);
        defer allocator.free(res);

        try std.testing.expectEqualSlices(u8, case.expected, res);
    }
}

test "normalizer sequence" {
    const allocator = std.testing.allocator;

    var normalizer = try Normalizer.init(allocator, &.{.{ .Strip = .both }});
    defer normalizer.deinit();

    const res = try normalizer.normalize(allocator, "hello ");
    defer allocator.free(res);

    try std.testing.expectEqualSlices(u8, "hello", res);
}

test "normalizer preserves whitespace across output and sequence boundaries" {
    const allocator = std.testing.allocator;
    for ([_][]const Normalizer.Kind{
        &.{.{ .Strip = .both }},
        &.{ .Lowercase, .{ .Strip = .both } },
        &.{ .Lowercase, .{ .Strip = .both }, .NFC },
    }) |steps| {
        var normalizer = try Normalizer.init(allocator, steps);
        defer normalizer.deinit();
        for ([_]usize{ 63, 64, 2047, 2048, 4096 }) |prefix_len| {
            const input = try allocator.alloc(u8, prefix_len + 4);
            defer allocator.free(input);
            @memset(input[0..prefix_len], 'a');
            @memcpy(input[prefix_len..], " b  ");
            const output = try normalizer.normalize(allocator, input);
            defer allocator.free(output);
            try std.testing.expectEqualSlices(u8, input[0 .. prefix_len + 2], output);
        }
    }
}

test "normalizer sequence preserves Unicode across tiles" {
    const allocator = std.testing.allocator;
    var normalizer = try Normalizer.fromHuggingFaceJson(
        \\{"type":"Sequence","normalizers":[{"type":"Lowercase"},{"type":"Strip","strip_left":true,"strip_right":true}]}
    );
    defer normalizer.deinit();
    for ([_][]const u8{ "é", "€", "😀" }) |codepoint| {
        for (61..65) |prefix_len| {
            const input = try allocator.alloc(u8, prefix_len + codepoint.len + 1);
            defer allocator.free(input);
            @memset(input[0..prefix_len], 'a');
            @memcpy(input[prefix_len..][0..codepoint.len], codepoint);
            input[input.len - 1] = 'b';
            const output = try normalizer.normalize(allocator, input);
            defer allocator.free(output);
            try std.testing.expectEqualSlices(u8, input, output);
        }
    }
}

test "normalizer sequence preserves composition across tiles" {
    const allocator = std.testing.allocator;
    var normalizer = try Normalizer.init(allocator, &.{ .NFC, .Lowercase });
    defer normalizer.deinit();
    const input = "a" ** 63 ++ "e\u{301}b";
    const output = try normalizer.normalize(allocator, input);
    defer allocator.free(output);
    try std.testing.expectEqualSlices(u8, "a" ** 63 ++ "éb", output);
}

test "normalizer sequence allocation failure retains child ownership" {
    const FailAllocator = struct {
        fn control(_: ?*anyopaque, _: c.iree_allocator_command_t, _: ?*const anyopaque, _: [*c]?*anyopaque) callconv(.c) c.iree_status_t {
            return @ptrFromInt(c.IREE_STATUS_RESOURCE_EXHAUSTED);
        }
    };
    // Include a nested sequence: failure must preserve its shell and children.
    var nested = try Normalizer.init(std.testing.allocator, &.{ .NFC, .Lowercase });
    defer nested.deinit();
    var strip = try Normalizer.init(std.testing.allocator, &.{.{ .Strip = .both }});
    defer strip.deinit();
    const children = [_]*c.iree_tokenizer_normalizer_t{ nested.sequence, strip.sequence };
    var sequence: ?*c.iree_tokenizer_normalizer_t = null;
    const status = c.iree_tokenizer_normalizer_sequence_allocate(&children, children.len, .{
        .self = null,
        .ctl = FailAllocator.control,
    }, &sequence);
    defer c.iree_status_free(status);
    try std.testing.expect(status != null);
    try std.testing.expectEqual(@as(u32, c.IREE_STATUS_RESOURCE_EXHAUSTED), statusCode(status));
    try std.testing.expect(sequence == null);
    const output = try nested.normalize(std.testing.allocator, "HELLO");
    defer std.testing.allocator.free(output);
    try std.testing.expectEqualStrings("hello", output);
}

fn tokenOutputWriter(allocator: std.mem.Allocator) std.Io.Writer.Allocating {
    return std.Io.Writer.Allocating.initAligned(allocator, .of(u32));
}

fn writtenTokenIds(out: *std.Io.Writer.Allocating) []const u32 {
    const bytes = out.written();
    const aligned_bytes: []align(4) u8 = @alignCast(bytes[0..bytes.len]);
    return std.mem.bytesAsSlice(u32, aligned_bytes);
}

fn decodedTextWriter(allocator: std.mem.Allocator) std.Io.Writer.Allocating {
    return std.Io.Writer.Allocating.init(allocator);
}

test "huggingface json encode/decode" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##",
        \\    "max_input_chars_per_word": 100,
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "hello": 1,
        \\      "world": 2
        \\    }
        \\  },
        \\  "pre_tokenizer": {"type": "Whitespace"},
        \\  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
        \\}
    ;

    var tokenizer = try Tokenizer.fromHuggingFaceJson(allocator, json);
    defer tokenizer.deinit();

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const text = "hello world";

    const token_ids = try encoder.encodeAlloc(allocator, text);
    defer allocator.free(token_ids);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, token_ids);

    const token_hello = tokenizer.tokenId("hello") orelse unreachable;
    try std.testing.expectEqual(@as(u32, 1), token_hello);

    const unknown = tokenizer.tokenId("missing");
    try std.testing.expect(unknown == null);

    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    var decoded = decodedTextWriter(allocator);
    defer decoded.deinit();
    try decoder.decode(token_ids, &decoded.writer);
    try std.testing.expectEqualStrings("hello world", decoded.written());
}

test "loads bpe tokenizer with newline lookahead split and added unk" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "model": {
        \\    "type": "BPE",
        \\    "unk_token": "[UNK]",
        \\    "vocab": {
        \\      "<|UNK|>": 0,
        \\      "h": 1,
        \\      "e": 2,
        \\      "l": 3,
        \\      "o": 4,
        \\      "he": 5,
        \\      "ll": 6,
        \\      "lo": 7,
        \\      "hel": 8,
        \\      "hello": 9
        \\    },
        \\    "merges": ["h e", "l l", "l o", "he l", "hel lo"]
        \\  },
        \\  "added_tokens": [
        \\    {
        \\      "id": 0,
        \\      "content": "<|UNK|>",
        \\      "single_word": false,
        \\      "lstrip": false,
        \\      "rstrip": false,
        \\      "normalized": false,
        \\      "special": true
        \\    }
        \\  ],
        \\  "pre_tokenizer": {
        \\    "type": "Sequence",
        \\    "pretokenizers": [
        \\      {
        \\        "type": "Split",
        \\        "pattern": {"Regex": "(?:\\r?\\n)+(?!\\r?\\n)"},
        \\        "behavior": "MergedWithNext",
        \\        "invert": false
        \\      },
        \\      {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true}
        \\    ]
        \\  },
        \\  "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true}
        \\}
    ;

    var tokenizer = try Tokenizer.fromBytes(allocator, json);
    defer tokenizer.deinit();

    try std.testing.expectEqual(@as(u32, 0), tokenizer.tokenId("<|UNK|>").?);

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const ids = try encoder.encodeAlloc(allocator, "hello");
    defer allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 5, 6, 4 }, ids);
}

test "writer" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "model": {
        \\    "type": "WordPiece",
        \\    "unk_token": "[UNK]",
        \\    "continuing_subword_prefix": "##",
        \\    "max_input_chars_per_word": 100,
        \\    "vocab": {
        \\      "[UNK]": 0,
        \\      "hello": 1,
        \\      "world": 2
        \\    }
        \\  },
        \\  "pre_tokenizer": {"type": "Whitespace"},
        \\  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
        \\}
    ;

    var tokenizer = try Tokenizer.fromBytes(allocator, json);
    defer tokenizer.deinit();

    const text = "hello world";

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const token_ids_alloc = try encoder.encodeAlloc(allocator, text);
    defer allocator.free(token_ids_alloc);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, token_ids_alloc);

    encoder.reset();
    var buf: [1024]u8 = undefined;
    var out = tokenOutputWriter(allocator);
    defer out.deinit();
    var writer = encoder.writer(&buf, &out.writer);
    try writer.interface.writeAll(text);
    try writer.finalize();
    const token_ids = writtenTokenIds(&out);

    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, token_ids);

    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    var decoded = decodedTextWriter(allocator);
    defer decoded.deinit();
    try decoder.decode(token_ids, &decoded.writer);
    try std.testing.expectEqualStrings("hello world", decoded.written());
}

/// Shared tokenizer JSON config used by multiple tests.
const test_tokenizer_json =
    \\{
    \\  "model": {
    \\    "type": "WordPiece",
    \\    "unk_token": "[UNK]",
    \\    "continuing_subword_prefix": "##",
    \\    "max_input_chars_per_word": 100,
    \\    "vocab": {
    \\      "[UNK]": 0,
    \\      "hello": 1,
    \\      "world": 2,
    \\      "foo": 3,
    \\      "bar": 4,
    \\      "a": 5,
    \\      "b": 6,
    \\      "c": 7
    \\    }
    \\  },
    \\  "pre_tokenizer": {"type": "Whitespace"},
    \\  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false}
    \\}
;

test "writer with tiny buffer forces drain" {
    // A buffer smaller than the input forces the drain path (not just flush).
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.fromBytes(allocator, test_tokenizer_json);
    defer tokenizer.deinit();

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    // Buffer of 4 bytes; "hello world" is 11 bytes, so drain must be invoked.
    var buf: [4]u8 = undefined;
    var out = tokenOutputWriter(allocator);
    defer out.deinit();
    var writer = encoder.writer(&buf, &out.writer);
    try writer.interface.writeAll("hello world");
    try writer.finalize();

    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, writtenTokenIds(&out));
}

test "writer byte-by-byte" {
    // Feed one byte at a time through the writer to stress buffer/drain boundaries.
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.fromBytes(allocator, test_tokenizer_json);
    defer tokenizer.deinit();

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var buf: [1]u8 = undefined;
    var out = tokenOutputWriter(allocator);
    defer out.deinit();
    var writer = encoder.writer(&buf, &out.writer);
    for ("hello world") |byte| {
        try writer.interface.writeByte(byte);
    }
    try writer.finalize();

    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, writtenTokenIds(&out));
}

test "encoder reset between multiple encodes" {
    // Ensure reset properly clears state so successive encodes are independent.
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.fromBytes(allocator, test_tokenizer_json);
    defer tokenizer.deinit();

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    {
        const ids = try encoder.encodeAlloc(allocator, "foo bar");
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(u32, &.{ 3, 4 }, ids);
    }

    encoder.reset();

    {
        const ids = try encoder.encodeAlloc(allocator, "hello world");
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, ids);
    }

    encoder.reset();

    // Third encode to verify repeated resets work.
    {
        const ids = try encoder.encodeAlloc(allocator, "foo");
        defer allocator.free(ids);
        try std.testing.expectEqualSlices(u32, &.{3}, ids);
    }
}

test "writer and encodeAlloc produce same results" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.fromBytes(allocator, test_tokenizer_json);
    defer tokenizer.deinit();

    const inputs = [_][]const u8{
        "hello",
        "hello world",
        "foo bar",
        "a b c",
        "foo bar hello world a b c",
    };

    for (inputs) |text| {
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();

        const expected = try encoder.encodeAlloc(allocator, text);
        defer allocator.free(expected);

        encoder.reset();

        var buf: [8]u8 = undefined;
        var out = tokenOutputWriter(allocator);
        defer out.deinit();
        var writer = encoder.writer(&buf, &out.writer);
        try writer.interface.writeAll(text);
        try writer.finalize();

        try std.testing.expectEqualSlices(u32, expected, writtenTokenIds(&out));
    }
}

test "encode empty string" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.fromBytes(allocator, test_tokenizer_json);
    defer tokenizer.deinit();

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    // encodeAlloc with empty string should produce no tokens.
    {
        const ids = try encoder.encodeAlloc(allocator, "");
        defer allocator.free(ids);
        try std.testing.expectEqual(@as(usize, 0), ids.len);
    }

    encoder.reset();

    // Writer with empty string should also produce no tokens.
    {
        var buf: [16]u8 = undefined;
        var out = tokenOutputWriter(allocator);
        defer out.deinit();
        var writer = encoder.writer(&buf, &out.writer);
        try writer.interface.writeAll("");
        try writer.finalize();
        try std.testing.expectEqual(@as(usize, 0), writtenTokenIds(&out).len);
    }
}

test "unknown tokens" {
    // Words not in the vocabulary should map to the [UNK] token (id 0).
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.fromBytes(allocator, test_tokenizer_json);
    defer tokenizer.deinit();

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const ids = try encoder.encodeAlloc(allocator, "unknown");
    defer allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{0}, ids);
}

test "decode round-trip" {
    // Encode then decode several strings and verify round-trip fidelity.
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.fromBytes(allocator, test_tokenizer_json);
    defer tokenizer.deinit();

    const inputs = [_][]const u8{
        "hello",
        "hello world",
        "foo bar",
        "a b c",
    };

    for (inputs) |text| {
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();

        const ids = try encoder.encodeAlloc(allocator, text);
        defer allocator.free(ids);

        var decoder = try tokenizer.decoder();
        defer decoder.deinit();

        var decoded = decodedTextWriter(allocator);
        defer decoded.deinit();
        try decoder.decode(ids, &decoded.writer);

        try std.testing.expectEqualStrings(text, decoded.written());
    }
}

test "decoder reset between decodes" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.fromBytes(allocator, test_tokenizer_json);
    defer tokenizer.deinit();

    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    {
        var text = decodedTextWriter(allocator);
        defer text.deinit();
        try decoder.decode(&.{ 1, 2 }, &text.writer);
        try std.testing.expectEqualStrings("hello world", text.written());
    }

    try decoder.reset();

    {
        var text = decodedTextWriter(allocator);
        defer text.deinit();
        try decoder.decode(&.{ 3, 4 }, &text.writer);
        try std.testing.expectEqualStrings("foo bar", text.written());
    }
}

test "tokenId lookup" {
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.fromBytes(allocator, test_tokenizer_json);
    defer tokenizer.deinit();

    try std.testing.expectEqual(@as(u32, 0), tokenizer.tokenId("[UNK]").?);
    try std.testing.expectEqual(@as(u32, 1), tokenizer.tokenId("hello").?);
    try std.testing.expectEqual(@as(u32, 2), tokenizer.tokenId("world").?);
    try std.testing.expectEqual(@as(u32, 3), tokenizer.tokenId("foo").?);
    try std.testing.expectEqual(@as(u32, 4), tokenizer.tokenId("bar").?);
    try std.testing.expect(tokenizer.tokenId("nonexistent") == null);
    try std.testing.expect(tokenizer.tokenId("") == null);
}

test "writer with pre-allocated output writer" {
    // Pass an already-allocated aligned output writer to the encoder stream.
    const allocator = std.testing.allocator;

    var tokenizer = try Tokenizer.fromBytes(allocator, test_tokenizer_json);
    defer tokenizer.deinit();

    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var out = tokenOutputWriter(allocator);
    defer out.deinit();
    try out.ensureUnusedCapacity(64 * @sizeOf(u32));

    var buf: [32]u8 = undefined;
    var writer = encoder.writer(&buf, &out.writer);
    try writer.interface.writeAll("hello world");
    try writer.finalize();

    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, writtenTokenIds(&out));
}
