const std = @import("std");

pub const hftokenizers = @import("hftokenizers");
pub const iree = @import("iree");
pub const sentencepiece = @import("sentencepiece");

pub const homemade = @import("homemade.zig");

const log = std.log.scoped(.@"zml/tokenizer");

const Tokenizers = enum {
    hftokenizers,
    iree,
    sentencepiece,
    homemade,
};

pub const Tokenizer = union(Tokenizers) {
    pub const Encoder = union(Tokenizers) {
        hftokenizers: hftokenizers.Encoder,
        iree: iree.Tokenizer.Encoder,
        sentencepiece: sentencepiece.Encoder,
        homemade: homemade.Encoder,

        pub fn deinit(self: *Encoder) void {
            switch (self.*) {
                inline else => |*v| v.deinit(),
            }
        }

        pub fn reset(self: *Encoder) void {
            switch (self.*) {
                inline else => |*v| v.reset(),
            }
        }

        pub fn encodeAlloc(self: *Encoder, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
            return switch (self.*) {
                .iree => |*encoder_| encoder_.encodeAlloc(allocator, text),
                inline else => |*encoder_| try allocator.dupe(u32, try encoder_.encode(text)),
            };
        }
    };

    pub const Decoder = union(Tokenizers) {
        hftokenizers: hftokenizers.Decoder,
        iree: iree.Tokenizer.Decoder,
        sentencepiece: sentencepiece.Decoder,
        homemade: homemade.Decoder,

        pub fn deinit(self: *Decoder) void {
            switch (self.*) {
                inline else => |*v| v.deinit(),
            }
        }

        pub fn reset(self: *Decoder) void {
            switch (self.*) {
                inline else => |*v| v.reset(),
            }
        }

        pub fn decodeAlloc(self: *Decoder, allocator: std.mem.Allocator, token_ids: []const u32) !std.ArrayList(u8) {
            return switch (self.*) {
                .iree => |*d| {
                    var writer: std.Io.Writer.Allocating = .init(allocator);
                    errdefer writer.deinit();
                    try d.decode(token_ids, &writer.writer);
                    return writer.toArrayList();
                },
                inline else => |*d| .fromOwnedSlice(try allocator.dupe(u8, try d.decode(token_ids))),
            };
        }

        // Only valid until the next decode and while the decoder AND buffer is alive.
        pub fn feedOne(self: *Decoder, token_id: u32, buffer: []u8) ![]const u8 {
            return switch (self.*) {
                .iree => |*d| d.feedOne(token_id, buffer),
                inline else => |*d| try d.next(token_id) orelse &.{},
            };
        }

        /// Output is only valid until the next feed/finalize call and while the decoder AND buffer is alive.
        pub fn finalize(self: *Decoder, buffer: []u8) ![]const u8 {
            return switch (self.*) {
                .iree => |*v| v.finalize(buffer),
                inline else => &.{},
            };
        }
    };

    hftokenizers: *hftokenizers.HFTokenizer,
    iree: iree.Tokenizer,
    sentencepiece: *sentencepiece.SentencePieceProcessor,
    homemade: *homemade.Tokenizer,

    pub fn fromFile(allocator: std.mem.Allocator, io: std.Io, model: []const u8) !Tokenizer {
        if (std.mem.endsWith(u8, model, ".pb")) {
            return .{ .sentencepiece = try .fromFile(model) };
        }
        if (std.mem.endsWith(u8, model, ".json")) {
            return .{ .iree = try .fromFile(allocator, io, model) };
        }

        return error.InvalidArgument;
    }

    pub fn fromBytes(allocator: std.mem.Allocator, bytes: []const u8) !Tokenizer {
        if (bytes.len == 0) return error.InvalidArgument;

        var trimmed = bytes;
        // Strip UTF-8 BOM if present
        if (trimmed.len >= 3 and std.mem.eql(u8, trimmed[0..3], "\xEF\xBB\xBF")) {
            trimmed = trimmed[3..];
        }
        // Strip leading whitespace
        var start: usize = 0;
        while (start < trimmed.len and std.ascii.isWhitespace(trimmed[start])) : (start += 1) {}
        trimmed = trimmed[start..];
        if (trimmed.len == 0) return error.InvalidArgument;

        if (trimmed[0] == '{') {
            return .{ .iree = try .fromBytes(allocator, trimmed) };
        }
        return .{ .sentencepiece = try .fromBytes(trimmed) };
    }

    pub fn deinit(self: *Tokenizer) void {
        switch (self.*) {
            inline else => |*t| t.*.deinit(),
        }
    }

    pub fn encoder(self: *const Tokenizer) !Encoder {
        return switch (self.*) {
            inline else => |*v, tag| @unionInit(Encoder, @tagName(tag), try v.*.encoder()),
        };
    }

    pub fn decoder(self: *const Tokenizer) !Decoder {
        return switch (self.*) {
            inline else => |*v, tag| @unionInit(Decoder, @tagName(tag), try v.*.decoder()),
        };
    }

    pub fn tokenId(self: *const Tokenizer, token: []const u8) ?u32 {
        return switch (self.*) {
            inline else => |v| v.tokenId(token),
        };
    }
};

test {
    std.testing.refAllDecls(@This());
}
