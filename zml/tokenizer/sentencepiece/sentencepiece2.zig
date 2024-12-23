const std = @import("std");
const ffi = @import("ffi_zig");
const c = @import("c");

const log = std.log.scoped(.sentencepiece);

// ref: https://github.com/google/sentencepiece/blob/d8f741853847553169444afc12c00f4bbff3e9ce/src/sentencepiece_processor.h#L34
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

pub const SentencePieceProcessor = struct {
    arena: std.heap.ArenaAllocator,
    _sp_processor: *c.SentencePieceProcessor,

    fn assertOk(code: c.sentencepiece_util_StatusCode) SentencePieceError!void {
        return switch (code) {
            c.kOk => {},
            c.kCancelled => error.Cancelled,
            c.kUnknown => error.Unknown,
            c.kInvalidArgument => error.InvalidArgument,
            c.kDeadlineExceeded => error.DeadlineExceeded,
            c.kNotFound => error.NotFound,
            c.kAlreadyExists => error.AlreadyExists,
            c.kPermissionDenied => error.PermissionDenied,
            c.kResourceExhausted => error.ResourceExhausted,
            c.kFailedPrecondition => error.FailedPrecondition,
            c.kAborted => error.Aborted,
            c.kOutOfRange => error.OutOfRange,
            c.kUnimplemented => error.Unimplemented,
            c.kInternal => error.Internal,
            c.kUnavailable => error.Unavailable,
            c.kDataLoss => error.DataLoss,
            c.kUnauthenticated => error.Unauthenticated,
            else => unreachable,
        };
    }

    pub fn init(allocator: std.mem.Allocator, model: []const u8) !SentencePieceProcessor {
        var arena = std.heap.ArenaAllocator.init(allocator);

        const sp_processor = try (c.SentencePieceProcessor_new(ffi.ZigAllocator.from(arena.allocator())) orelse std.mem.Allocator.Error.OutOfMemory);
        errdefer c.SentencePieceProcessor_delete(ffi.ZigAllocator.from(arena.allocator()), sp_processor);

        try assertOk(c.SentencePieceProcessor_Load(sp_processor, ffi.as_path(model)[0..].ptr));

        return .{
            .arena = arena,
            ._sp_processor = sp_processor,
        };
    }

    pub fn deinit(self: *const SentencePieceProcessor) void {
        c.SentencePieceProcessor_delete(self.arena.allocator(), self._sp_processor);
        self.arena.deinit();
    }

    pub fn encode(self: *const SentencePieceProcessor, allocator: std.mem.Allocator, input: []const u8) ![]i32 {
        var ids: c.zig_slice = undefined;
        try assertOk(c.SentencePieceProcessor_Encode(self._sp_processor, ffi.ZigAllocator.from(allocator), ffi.ZigSlice.from(input), &ids));
        return ffi.ZigSlice.to(i32, ids);
    }

    pub fn decode(self: *const SentencePieceProcessor, allocator: std.mem.Allocator, input: []const i32) ![]u8 {
        var detokenized: c.zig_slice = undefined;
        try assertOk(c.SentencePieceProcessor_Decode(self._sp_processor, ffi.ZigAllocator.from(allocator), ffi.ZigSlice.from(input), &detokenized));
        return ffi.ZigSlice.to(u8, detokenized);
    }

    pub fn decode2(self: *const SentencePieceProcessor, allocator: std.mem.Allocator, input: []const i32) ![]const u8 {
        var detokenized = try ffi.std_string.ArrayList(u8).init(allocator);

        try assertOk(c.SentencePieceProcessor_Decode2(self._sp_processor, ffi.ZigSlice.from(input), @ptrCast(detokenized.str)));
        detokenized.refresh();
        try detokenized.array_list().appendSlice("hello hellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohellohello");

        log.debug("detokenized: {d}", .{detokenized.array_list().items.len});
        return detokenized.array_list().items;
    }

    pub fn decode3(self: *const SentencePieceProcessor, allocator: std.mem.Allocator, input: []const i32) ![]const u8 {
        var pieces = try std.ArrayList(u8).initCapacity(allocator, input.len);
        defer pieces.deinit();

        for (input) |token_id| {
            var piece_slice: c.zig_slice = undefined;
            try assertOk(c.SentencePieceProcessor_DecodeId(self._sp_processor, token_id, &piece_slice));
            const piece = ffi.ZigSlice.to(u8, piece_slice);
            try pieces.appendSlice(piece);
        }
        return pieces.toOwnedSlice();
    }

    pub fn decodeId(self: *const SentencePieceProcessor, id: i32) ![]u8 {
        var buffer: c.zig_slice = undefined;
        try assertOk(c.SentencePieceProcessor_DecodeId(self._sp_processor, id, &buffer));
        return ffi.ZigSlice.to(u8, buffer);
    }

    pub fn pieceToId(self: *const SentencePieceProcessor, input: []const u8) i32 {
        return c.SentencePieceProcessor_PieceToId(self._sp_processor, ffi.ZigSlice.from(input));
    }

    pub fn eosId(self: *const SentencePieceProcessor) i32 {
        return c.SentencePieceProcessor_eos_id(self._sp_processor);
    }

    pub fn bosId(self: *const SentencePieceProcessor) i32 {
        return c.SentencePieceProcessor_bos_id(self._sp_processor);
    }

    pub fn unkId(self: *const SentencePieceProcessor) i32 {
        return c.SentencePieceProcessor_unk_id(self._sp_processor);
    }
};
