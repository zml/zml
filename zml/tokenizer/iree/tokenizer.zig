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
                break;
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
