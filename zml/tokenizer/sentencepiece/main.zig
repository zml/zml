const std = @import("std");
const c = @import("c");
const ffi = @import("ffi");

pub const SentencePieceError = error{
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

pub const DecoderStream = struct {
    const TokensSize = 4;
    const StringSize = 128;
    decoder: SentencePieceProcessor.Decoder,
    buffer: [StringSize]u8 = undefined,
    last_tokens: []u8 = &.{},

    pub fn init(decoder: SentencePieceProcessor.Decoder) DecoderStream {
        var ret: DecoderStream = .{
            .decoder = decoder,
        };
        ret.decoder.reserve_tokens(TokensSize);
        ret.decoder.reserve_string(StringSize);
        return ret;
    }

    pub fn next(self: *DecoderStream, next_token: u32) !?[]const u8 {
        if (self.decoder.tokens().len >= TokensSize) {
            const tokens = self.decoder.tokens();
            inline for (0..TokensSize - 1) |i| {
                tokens[i] = tokens[i + 1];
            }
            tokens[TokensSize - 1] = next_token;
        } else {
            self.decoder.append(next_token);
        }
        const new_tokens = try self.decoder.decode();
        if (self.last_tokens.len == 0) {
            self.last_tokens = self.buffer[0..new_tokens.len];
            @memcpy(self.last_tokens, new_tokens);
            return new_tokens;
        }
        for (1..self.last_tokens.len) |i| {
            if (std.mem.startsWith(u8, new_tokens, self.last_tokens[i..])) {
                const toks = new_tokens[self.last_tokens.len - i ..];
                self.last_tokens = self.buffer[0..new_tokens.len];
                @memcpy(self.last_tokens, new_tokens);
                return toks;
            }
        }
        return null;
    }
};

pub const SentencePieceProcessor = opaque {
    pub const Encoder = struct {
        inner: *SentencePieceProcessor,
        vec: *c.std_vector_int,

        fn init(inner: *SentencePieceProcessor) Encoder {
            return .{
                .inner = inner,
                .vec = c.std_vector_int_new() orelse unreachable,
            };
        }

        pub fn deinit(self: *Encoder) void {
            c.std_vector_int_delete(self.vec);
        }

        pub fn reserve(self: *Encoder, size: usize) void {
            c.std_vector_int_reserve(self.vec, size);
        }

        pub fn reset(self: *Encoder) void {
            c.std_vector_int_clear(self.vec);
        }

        pub fn encode(self: *Encoder, input: []const u8) ![]const u32 {
            try assertOk(c.SentencePieceProcessor_Encode(@ptrCast(self.inner), ffi.ZigSlice.from(input), self.vec));
            return ffi.ZigSlice.to(u32, .{
                .ptr = c.std_vector_int_data(self.vec),
                .len = c.std_vector_int_size(self.vec),
            });
        }
    };

    pub const Decoder = struct {
        inner: *SentencePieceProcessor,
        vec: *c.std_vector_int,
        str: *c.std_string,

        fn init(inner: *SentencePieceProcessor) Decoder {
            return .{
                .inner = inner,
                .vec = c.std_vector_int_new() orelse unreachable,
                .str = c.std_string_new() orelse unreachable,
            };
        }

        pub fn append(self: *Decoder, token: u32) void {
            c.std_vector_int_push_back(self.vec, @intCast(token));
        }

        pub fn deinit(self: *Decoder) void {
            c.std_vector_int_delete(self.vec);
            c.std_string_delete(self.str);
        }

        pub fn reserve_tokens(self: *Decoder, size: usize) void {
            c.std_vector_int_reserve(self.vec, size);
        }

        pub fn reserve_string(self: *Decoder, size: usize) void {
            c.std_string_reserve(self.str, size);
        }

        pub fn reset(self: *Decoder) void {
            c.std_vector_int_clear(self.vec);
            c.std_string_clear(self.str);
        }

        pub fn decode(self: *Decoder) ![]const u8 {
            try assertOk(c.SentencePieceProcessor_Decode(@ptrCast(self.inner), self.vec, self.str));
            return self.string();
        }

        pub fn string(self: *const Decoder) []const u8 {
            const res = c.std_string_data(self.str);
            return ffi.ZigSlice.to(u8, res);
        }

        pub fn tokens(self: *const Decoder) []u32 {
            const ptr: [*c]u32 = @ptrCast(c.std_vector_int_data(self.vec));
            return ptr[0..c.std_vector_int_size(self.vec)];
        }
    };

    fn assertOk(code: c.sentencepiece_util_StatusCode) SentencePieceError!void {
        return switch (code) {
            c.sentencepiece_util_StatusCode_kOk => {},
            c.sentencepiece_util_StatusCode_kCancelled => error.Cancelled,
            c.sentencepiece_util_StatusCode_kUnknown => error.Unknown,
            c.sentencepiece_util_StatusCode_kInvalidArgument => error.InvalidArgument,
            c.sentencepiece_util_StatusCode_kDeadlineExceeded => error.DeadlineExceeded,
            c.sentencepiece_util_StatusCode_kNotFound => error.NotFound,
            c.sentencepiece_util_StatusCode_kAlreadyExists => error.AlreadyExists,
            c.sentencepiece_util_StatusCode_kPermissionDenied => error.PermissionDenied,
            c.sentencepiece_util_StatusCode_kResourceExhausted => error.ResourceExhausted,
            c.sentencepiece_util_StatusCode_kFailedPrecondition => error.FailedPrecondition,
            c.sentencepiece_util_StatusCode_kAborted => error.Aborted,
            c.sentencepiece_util_StatusCode_kOutOfRange => error.OutOfRange,
            c.sentencepiece_util_StatusCode_kUnimplemented => error.Unimplemented,
            c.sentencepiece_util_StatusCode_kInternal => error.Internal,
            c.sentencepiece_util_StatusCode_kUnavailable => error.Unavailable,
            c.sentencepiece_util_StatusCode_kDataLoss => error.DataLoss,
            c.sentencepiece_util_StatusCode_kUnauthenticated => error.Unauthenticated,
            else => unreachable,
        };
    }

    pub fn load(model: []const u8) !*SentencePieceProcessor {
        const sp: *SentencePieceProcessor = @ptrCast(c.SentencePieceProcessor_new());
        errdefer sp.deinit();
        try sp.load_from(model);
        return sp;
    }

    pub fn deinit(self: *SentencePieceProcessor) void {
        c.SentencePieceProcessor_delete(@ptrCast(self));
    }

    fn load_from(self: *SentencePieceProcessor, model: []const u8) !void {
        try assertOk(c.SentencePieceProcessor_Load(@ptrCast(self), ffi.ZigSlice.from(model)));
    }

    pub fn encoder(self: *SentencePieceProcessor) Encoder {
        return Encoder.init(self);
    }

    pub fn decoder(self: *SentencePieceProcessor) Decoder {
        return Decoder.init(self);
    }
};

pub fn as_path(path: []const u8) [std.fs.max_path_bytes:0]u8 {
    var result: [std.fs.max_path_bytes:0]u8 = undefined;
    @memcpy(result[0..path.len], path);
    result[path.len] = 0;
    return result;
}

pub fn main() !void {
    const sp = try SentencePieceProcessor.load("/Users/steeve/Downloads/poolside.sp.pb");
    defer sp.deinit();

    std.debug.print("Loaded model\n", .{});

    var encoder = sp.encoder();
    defer encoder.deinit();

    var decoder = sp.decoder();
    defer decoder.deinit();

    const ss = @embedFile("main.zig");
    // \\String class
    // \\Strings are objects that represent sequences of characters.
    // \\
    // \\The standard string class provides support for such objects with an interface similar to that of a standard container of bytes, but adding features specifically designed to operate with strings of single-byte characters.
    // \\
    // \\The string class is an instantiation of the basic_string class template that uses char (i.e., bytes) as its character type, with its default char_traits and allocator types (see basic_string for more info on the template).
    // \\
    // \\Note that this class handles bytes independently of the encoding used: If used to handle sequences of multi-byte or variable-length characters (such as UTF-8), all members of this class (such as length or size), as well as its iterators, will still operate in terms of bytes (not actual encoded characters).
    // \\
    // ;
    const tokens = try encoder.encode(ss);

    // const ss2 = 128;
    // var buf = [_]u8{0} ** ss2;
    // // _ = buf; // autofix
    // var last_tokens: []u8 = &.{};
    // // _ = last_tokens; // autofix
    // decoder.reserve_tokens(4);
    // decoder.reserve_string(128);

    var stream = DecoderStream.init(decoder);

    var start = try std.time.Timer.start();
    for (tokens) |token| {
        if (try stream.next(token)) |chunk| {
            // std.debug.print("{s}", .{chunk});
            std.debug.print("{d}us - {s}\n", .{ start.lap() / std.time.ns_per_us, chunk });
        }
    }

    // var start = try std.time.Timer.start();
    // var it = std.mem.window(u32, tokens, 3, 1);
    // while (it.next()) |slice| {
    //     if (decoder.tokens().len >= 4) {
    //         const kept_tokens = decoder.tokens()[1..];
    //         std.mem.copyForwards(u32, decoder.tokens()[0..kept_tokens.len], kept_tokens);
    //         kept_tokens[kept_tokens.len - 1] = slice[2];
    //     } else {
    //         for (slice) |token| {
    //             decoder.append(token);
    //         }
    //     }
    //     const new_tokens = try decoder.decode();
    //     for (0..ss2) |i| {
    //         if (std.mem.startsWith(u8, new_tokens, last_tokens[i..])) {
    //             const toks = new_tokens[last_tokens.len - i..];
    //             // std.debug.print("{s}", .{toks});
    //             if (toks.len == 0) {
    //                 // std.debug.print("WESH\n", .{});
    //             }
    //             break;
    //         }
    //     }
    //     last_tokens = buf[0..new_tokens.len];
    //     @memcpy(last_tokens, new_tokens);
    //     std.debug.print("{d}us\n", .{start.lap() / std.time.ns_per_us});
    // }

    // for (tokens) |token| {
    //     decoder.append(token);
    // }
    // const decoded = try decoder.decode();
    // std.debug.print("Decoded: {s}\n", .{decoded});

    // const model = "/Users/steeve/Downloads/poolside.sp.pb";

    // c.SentencePieceProcessor_LoadOrDie(sp, c.zig_slice{ .ptr = model.ptr, .len = model.len });

    // const piece = c.SentencePieceProcessor_IdToPiece(sp, 10999);
    // std.debug.print("{s}\n", .{piece.ptr[0..piece.len]});
}
