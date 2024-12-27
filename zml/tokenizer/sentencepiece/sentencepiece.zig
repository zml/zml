const std = @import("std");
const c = @import("c");
const ffi = @import("ffi");

const StringToTokenRatio = 3;

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

fn assertOk(code: c.sentencepiece_util_StatusCode) Error!void {
    return switch (code) {
        c.sentencepiece_util_StatusCode_kOk => {},
        c.sentencepiece_util_StatusCode_kCancelled => Error.Cancelled,
        c.sentencepiece_util_StatusCode_kUnknown => Error.Unknown,
        c.sentencepiece_util_StatusCode_kInvalidArgument => Error.InvalidArgument,
        c.sentencepiece_util_StatusCode_kDeadlineExceeded => Error.DeadlineExceeded,
        c.sentencepiece_util_StatusCode_kNotFound => Error.NotFound,
        c.sentencepiece_util_StatusCode_kAlreadyExists => Error.AlreadyExists,
        c.sentencepiece_util_StatusCode_kPermissionDenied => Error.PermissionDenied,
        c.sentencepiece_util_StatusCode_kResourceExhausted => Error.ResourceExhausted,
        c.sentencepiece_util_StatusCode_kFailedPrecondition => Error.FailedPrecondition,
        c.sentencepiece_util_StatusCode_kAborted => Error.Aborted,
        c.sentencepiece_util_StatusCode_kOutOfRange => Error.OutOfRange,
        c.sentencepiece_util_StatusCode_kUnimplemented => Error.Unimplemented,
        c.sentencepiece_util_StatusCode_kInternal => Error.Internal,
        c.sentencepiece_util_StatusCode_kUnavailable => Error.Unavailable,
        c.sentencepiece_util_StatusCode_kDataLoss => Error.DataLoss,
        c.sentencepiece_util_StatusCode_kUnauthenticated => Error.Unauthenticated,
        else => unreachable,
    };
}

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

    pub fn reset(self: *Encoder) void {
        c.std_vector_int_clear(self.vec);
    }

    pub fn encode(self: *Encoder, input: []const u8) ![]const u32 {
        c.std_vector_int_reserve(self.vec, input.len / StringToTokenRatio);
        try assertOk(c.SentencePieceProcessor_Encode(@ptrCast(self.inner), ffi.ZigSlice.from(input), self.vec));
        return self.ids();
    }

    pub fn ids(self: *const Encoder) []const u32 {
        return ffi.ZigSlice.to(u32, .{
            .ptr = c.std_vector_int_data(self.vec),
            .len = c.std_vector_int_size(self.vec),
        });
    }
};

pub const Decoder = struct {
    const TokensBufferSize = 4;
    const StringBufferSize = 128;

    inner: *SentencePieceProcessor,
    vec: *c.std_vector_int,
    str: *c.std_string,
    str_buffer: [StringBufferSize]u8 = undefined,
    last_tokens: []u8 = &.{},

    fn init(inner: *SentencePieceProcessor) !Decoder {
        const vec = try (c.std_vector_int_new() orelse std.mem.Allocator.Error.OutOfMemory);
        c.std_vector_int_reserve(vec, TokensBufferSize);
        errdefer c.std_vector_int_delete(vec);

        const str = try (c.std_string_new() orelse std.mem.Allocator.Error.OutOfMemory);
        c.std_string_reserve(str, StringBufferSize);
        errdefer c.std_string_delete(str);

        return .{
            .inner = inner,
            .vec = vec,
            .str = str,
        };
    }

    pub fn deinit(self: *Decoder) void {
        c.std_vector_int_delete(self.vec);
        c.std_string_delete(self.str);
    }

    pub fn reset(self: *Decoder) void {
        c.std_vector_int_clear(self.vec);
        c.std_string_clear(self.str);
    }

    pub fn decode(self: *Decoder, ids_: []const u32) ![]const u8 {
        c.std_vector_int_reserve(self.vec, ids_.len);
        c.std_string_reserve(self.str, ids_.len * StringToTokenRatio);
        for (ids_) |id| {
            c.std_vector_int_push_back(self.vec, @intCast(id));
        }
        try assertOk(c.SentencePieceProcessor_Decode(@ptrCast(self.inner), self.vec, self.str));
        return self.string();
    }

    pub fn string(self: *const Decoder) []const u8 {
        const res = c.std_string_data(self.str);
        return ffi.ZigSlice.to(u8, res);
    }

    fn ids(self: *const Decoder) []u32 {
        const ptr: [*c]u32 = @ptrCast(c.std_vector_int_data(self.vec));
        return ptr[0..c.std_vector_int_size(self.vec)];
    }
};

pub const SentencePieceProcessor = opaque {
    pub fn from_file(model: []const u8) !*SentencePieceProcessor {
        const sp: *SentencePieceProcessor = @ptrCast(c.SentencePieceProcessor_new());
        errdefer sp.deinit();
        try assertOk(c.SentencePieceProcessor_Load(@ptrCast(sp), ffi.ZigSlice.from(model)));
        return sp;
    }

    pub fn deinit(self: *SentencePieceProcessor) void {
        c.SentencePieceProcessor_delete(@ptrCast(self));
    }

    pub fn encoder(self: *SentencePieceProcessor) !Encoder {
        return Encoder.init(self);
    }

    pub fn decoder(self: *SentencePieceProcessor) !Decoder {
        return try Decoder.init(self);
    }
};
