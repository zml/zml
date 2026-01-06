const std = @import("std");

pub const hftokenizers = @import("hftokenizers");
pub const sentencepiece = @import("sentencepiece");

pub const homemade = @import("homemade.zig");

test {
    std.testing.refAllDecls(@This());
}

const Tokenizers = enum {
    hftokenizers,
    sentencepiece,
    homemade,
};

pub const Tokenizer = union(Tokenizers) {
    pub const Encoder = union(Tokenizers) {
        hftokenizers: hftokenizers.Encoder,
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

        pub fn encode(self: *Encoder, input: []const u8) ![]const u32 {
            return switch (self.*) {
                inline else => |*v| v.encode(input),
            };
        }

        pub fn ids(self: Encoder) []const u32 {
            return switch (self) {
                inline else => |v| v.ids(),
            };
        }
    };

    pub const Decoder = union(Tokenizers) {
        hftokenizers: hftokenizers.Decoder,
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

        pub fn decode(self: *Decoder, ids_: []const u32) ![]const u8 {
            return switch (self.*) {
                inline else => |*v| v.decode(ids_),
            };
        }

        pub fn string(self: *Decoder) []const u8 {
            return switch (self.*) {
                inline else => |v| v.string(),
            };
        }

        //pub fn ids(self: *Decoder) []u32 {
        //    return switch (self.*) {
        //        inline else => |v| v.ids(),
        //    };
        //}

        pub fn next(self: *Decoder, token_id: u32) !?[]const u8 {
            return switch (self.*) {
                inline else => |*v| v.next(token_id),
            };
        }
    };

    hftokenizers: *hftokenizers.HFTokenizer,
    sentencepiece: *sentencepiece.SentencePieceProcessor,
    homemade: *homemade.Tokenizer,

    pub fn fromFile(allocator: std.mem.Allocator, io: std.Io, model: []const u8) !Tokenizer {
        _ = allocator; // autofix
        _ = io; // autofix
        if (std.mem.endsWith(u8, model, ".pb")) {
            return .{ .sentencepiece = try .fromFile(model) };
        }
        if (std.mem.endsWith(u8, model, ".json")) {
            return .{ .hftokenizers = try .fromFile(model) };
        }

        return error.InvalidArgument;
    }

    pub fn fromBytes(allocator: std.mem.Allocator, io: std.Io, bytes: []const u8) !Tokenizer {
        _ = allocator; // autofix
        _ = io; // autofix
        if (bytes[0] == '{') {
            return .{ .hftokenizers = try .fromBytes(bytes) };
        }
        return .{ .sentencepiece = try .fromBytes(bytes) };
    }

    pub fn deinit(self: *Tokenizer) void {
        switch (self.*) {
            inline else => |t| t.deinit(),
        }
    }

    pub fn encoder(self: Tokenizer) !Encoder {
        return switch (self) {
            inline else => |v, tag| @unionInit(Encoder, @tagName(tag), try v.encoder()),
        };
    }

    pub fn decoder(self: Tokenizer) !Decoder {
        return switch (self) {
            inline else => |v, tag| @unionInit(Decoder, @tagName(tag), try v.decoder()),
        };
    }

    pub fn tokenToId(self: Tokenizer, token: []const u8) ?u32 {
        return switch (self) {
            inline else => |v| v.tokenToId(token),
        };
    }
};

test {}
test {}
