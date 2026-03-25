const std = @import("std");

const c = @import("c");

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
    return .{ .data = @ptrCast(s.ptr), .size = s.len };
}

inline fn byteSpan(bytes: []u8) c.iree_byte_span_t {
    return .{ .data = bytes.ptr, .data_length = bytes.len };
}

inline fn tokenOutput(ids: []c.iree_tokenizer_token_id_t) c.iree_tokenizer_token_output_t {
    return .{
        .capacity = ids.len,
        .token_ids = ids.ptr,
        .token_offsets = null,
        .type_ids = null,
    };
}

inline fn tokenIdList(ids: []const c.iree_tokenizer_token_id_t) c.iree_tokenizer_token_id_list_t {
    return .{ .count = ids.len, .values = ids.ptr };
}

inline fn statusCode(status: c.iree_status_t) u32 {
    return @intCast(@intFromPtr(status.?) & StatusCodeMask);
}

fn assertOk(status: c.iree_status_t) Error!void {
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
    const EncodeBatchSize = 256;
    const DecodeOutputBatchSize = c.IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE;
    const DecodeTokenBatchSize = 256;

    allocator: std.mem.Allocator,
    inner: *c.iree_tokenizer_t,

    pub fn fromHuggingFaceJson(allocator: std.mem.Allocator, json: []const u8) Error!Tokenizer {
        var raw: ?*c.iree_tokenizer_t = null;
        try assertOk(c.iree_tokenizer_from_huggingface_json(stringView(json), c.iree_allocator_system(), &raw));
        return .{
            .allocator = allocator,
            .inner = raw.?,
        };
    }

    pub fn fromBytes(allocator: std.mem.Allocator, bytes: []const u8) Error!Tokenizer {
        return fromHuggingFaceJson(allocator, bytes);
    }

    pub fn fromFile(allocator: std.mem.Allocator, io: std.Io, model: []const u8) (Error || std.mem.Allocator.Error || std.fs.File.OpenError || std.fs.File.ReadError)!Tokenizer {
        _ = io;
        const json = try std.fs.cwd().readFileAlloc(allocator, model, std.math.maxInt(usize));
        defer allocator.free(json);
        return try fromHuggingFaceJson(allocator, json);
    }

    pub fn deinit(self: *Tokenizer) void {
        c.iree_tokenizer_free(self.inner);
    }

    pub fn encoder(self: *const Tokenizer) (Error || std.mem.Allocator.Error)!Encoder {
        return Encoder.init(self);
    }

    pub fn decoder(self: *const Tokenizer) (Error || std.mem.Allocator.Error)!Decoder {
        return Decoder.init(self);
    }

    pub fn findTokenId(self: *const Tokenizer, token: []const u8) ?u32 {
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
        output: [EncodeBatchSize]c.iree_tokenizer_token_id_t,
        output_token_ids: [EncodeBatchSize]u32,

        fn init(tokenizer: *const Tokenizer) (Error || std.mem.Allocator.Error)!Encoder {
            var state_size: usize = undefined;
            try assertOk(c.iree_tokenizer_encode_state_calculate_size(tokenizer.inner, &state_size));
            const state_storage = try tokenizer.allocator.alloc(u8, state_size);
            errdefer tokenizer.allocator.free(state_storage);

            const transform_size: usize = c.IREE_TOKENIZER_TRANSFORM_BUFFER_MAX_SIZE;
            const transform_buffer = try tokenizer.allocator.alloc(u8, transform_size);
            errdefer tokenizer.allocator.free(transform_buffer);

            var state: ?*c.iree_tokenizer_encode_state_t = null;
            try assertOk(c.iree_tokenizer_encode_state_initialize(
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
                .output = undefined,
                .output_token_ids = undefined,
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

        pub fn feed(self: *Encoder, chunk: []const u8) Error!struct {
            usize,
            []const u32,
        } {
            var consumed: usize = 0;
            var produced: usize = 0;

            // Feeds an input |chunk| to the encoder. Tokens that can be definitively
            // produced are written to |output|. Additional tokens may be produced by
            // subsequent feed calls or by finalize once all input has been provided.
            //
            // Pull-based model: Output buffer capacity drives processing. The encoder
            // pulls data through the pipeline (normalizer → segmenter → model) to
            // fill the output buffer. Processing stops when output is full or input
            // is exhausted.
            //
            // Returns:
            // - |out_bytes_consumed|: Bytes consumed from |chunk|. May be less than
            //   chunk.size if internal buffers (transform_buffer) fill up. Caller should
            //   retry with remaining bytes: chunk.data + *out_bytes_consumed.
            // - |out_token_count|: Tokens written to |output|.
            //
            // The function always makes progress if possible: it will consume input
            // and/or produce tokens. Returns iree_ok_status() even if not all input
            // was consumed (check out_bytes_consumed).
            try assertOk(c.iree_tokenizer_encode_state_feed(
                self.state,
                stringView(chunk),
                tokenOutput(&self.output),
                &consumed,
                &produced,
            ));

            return .{
                consumed,
                @as([*]const u32, @ptrCast(&self.output))[0..produced],
            };
        }

        pub fn finalize(self: *Encoder) Error![]const u32 {
            var produced: usize = 0;

            // Finalizes encoding by flushing any buffered data through the pipeline. Must
            // be called after all input chunks have been fed. Any remaining tokens are
            // written to |output|.
            //
            // Returns the number of tokens written in |out_token_count|. Returns
            // IREE_STATUS_RESOURCE_EXHAUSTED if the output would exceed output.capacity.
            try assertOk(c.iree_tokenizer_encode_state_finalize(
                self.state,
                tokenOutput(&self.output),
                &produced,
            ));

            return @as([*]const u32, @ptrCast(&self.output))[0..produced];
        }
    };

    pub const Decoder = struct {
        tokenizer: *const Tokenizer,
        state_storage: []u8,
        state: *c.iree_tokenizer_decode_state_t,
        output: [DecodeOutputBatchSize]u8,
        input_ids: [DecodeTokenBatchSize]c.iree_tokenizer_token_id_t,

        fn init(tokenizer: *const Tokenizer) (Error || std.mem.Allocator.Error)!Decoder {
            var state_size: usize = undefined;
            try assertOk(c.iree_tokenizer_decode_state_calculate_size(tokenizer.inner, &state_size));
            const state_storage = try tokenizer.allocator.alloc(u8, state_size);
            errdefer tokenizer.allocator.free(state_storage);

            var state: ?*c.iree_tokenizer_decode_state_t = null;
            try assertOk(c.iree_tokenizer_decode_state_initialize(
                tokenizer.inner,
                c.IREE_TOKENIZER_DECODE_FLAG_NONE,
                byteSpan(state_storage),
                &state,
            ));

            return .{
                .tokenizer = tokenizer,
                .state_storage = state_storage,
                .state = state.?,
                .output = undefined,
                .input_ids = undefined,
            };
        }

        pub fn deinit(self: *Decoder) void {
            c.iree_tokenizer_decode_state_deinitialize(self.state);
            self.tokenizer.allocator.free(self.state_storage);
        }

        pub fn reset(self: *Decoder) Error!void {
            c.iree_tokenizer_decode_state_deinitialize(self.state);
            var state: ?*c.iree_tokenizer_decode_state_t = null;
            try assertOk(c.iree_tokenizer_decode_state_initialize(
                self.tokenizer.inner,
                c.IREE_TOKENIZER_DECODE_FLAG_NONE,
                byteSpan(self.state_storage),
                &state,
            ));
            self.state = state.?;
        }

        pub fn feed(self: *Decoder, token_ids_: []const u32) Error!struct {
            usize,
            []const u8,
        } {
            const token_ids: []const c.iree_tokenizer_token_id_t =
                @as([*]const i32, @ptrCast(token_ids_.ptr))[0..token_ids_.len];

            var consumed: usize = 0;
            var written: usize = 0;

            // Feeds |tokens| to the decoder. Text that can be definitively produced is
            // written to |text_output| as raw UTF-8 bytes (not NUL-terminated). Additional
            // text may be produced by subsequent feed calls or by finalize once all tokens
            // have been provided.
            //
            // Pull-based model: Output buffer capacity drives processing. The decoder
            // pulls tokens through the pipeline to fill the text output. Processing
            // stops when output is full or tokens are exhausted.
            //
            // Returns iree_ok_status() always (errors are only from internal pipeline
            // failures). Progress is indicated by the output parameters:
            // - |out_tokens_consumed|: Tokens consumed from |tokens|. May be less than
            //   tokens.count if text output fills up. Caller should retry with remaining
            //   tokens: tokens.values + *out_tokens_consumed.
            // - |out_text_length|: Bytes written to |text_output|.
            //
            // Zero-progress case: If both out_tokens_consumed and out_text_length are 0,
            // the output buffer is genuinely exhausted (the next token's text does not fit
            // in |text_output|). Callers must check for this to avoid infinite loops.
            // Provide a larger output buffer or consume the already-written text.
            try assertOk(c.iree_tokenizer_decode_state_feed(
                self.state,
                tokenIdList(token_ids),
                mutableStringView(&self.output),
                &consumed,
                &written,
            ));

            return .{
                consumed,
                self.output[0..written],
            };
        }

        pub fn finalize(self: *Decoder) Error![]const u8 {
            var written: usize = 0;

            // Finalizes decoding by flushing any buffered data through the pipeline. Must
            // be called after all tokens have been fed. Any remaining text is written to
            // |text_output| as raw UTF-8 bytes (not NUL-terminated).
            //
            // Returns the number of bytes written in |out_text_length|. Returns
            // IREE_STATUS_RESOURCE_EXHAUSTED if the output would exceed text_output.size.
            try assertOk(c.iree_tokenizer_decode_state_finalize(
                self.state,
                mutableStringView(&self.output),
                &written,
            ));
            return self.output[0..written];
        }
    };
};

test "huggingface json streaming encode/decode" {
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

    var token_ids: std.ArrayList(u32) = .empty;
    defer token_ids.deinit(allocator);
    var remaining_txt: []const u8 = text;
    while (remaining_txt.len > 0) {
        const consumed, const ids = try encoder.feed(remaining_txt);
        try token_ids.appendSlice(allocator, ids);
        remaining_txt = remaining_txt[consumed..];
    }
    try token_ids.appendSlice(allocator, try encoder.finalize());

    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, token_ids.items);

    const token_hello = tokenizer.findTokenId("hello") orelse unreachable;
    try std.testing.expectEqual(@as(u32, 1), token_hello);

    const unknown = tokenizer.findTokenId("missing");
    try std.testing.expect(unknown == null);

    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    var tokens: std.ArrayList(u8) = .empty;
    defer tokens.deinit(allocator);
    var remaining_ids: []const u32 = token_ids.items;
    while (remaining_ids.len > 0) {
        const consumed, const chunk = try decoder.feed(remaining_ids);
        try tokens.appendSlice(allocator, chunk);
        remaining_ids = remaining_ids[consumed..];
    }
    try tokens.appendSlice(allocator, try decoder.finalize());

    try std.testing.expectEqualStrings("hello world", tokens.items);
}
