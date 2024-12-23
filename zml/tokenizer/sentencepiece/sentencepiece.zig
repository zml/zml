pub const std = @import("std");
pub const c = @import("c");

pub const SentencePieceProcessor = opaque {
    pub const create() *SentencePieceProcessor {
        return @ptrCast(c.SentencePieceProcessor_new());
    }

    pub const deinit(self: *SentencePieceProcessor) {
        c.SentencePieceProcessor_delete(@ptrCast(self));
    }
};
