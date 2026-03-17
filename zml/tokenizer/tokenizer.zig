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

        pub const FeedOutput = struct {
            consumed: usize,
            /// Only valid until the next feed/finalize call and while the encoder is alive.
            token_ids: []const u32,
        };

        pub fn feed(self: *Encoder, tokens: []const u8) !FeedOutput {
            return switch (self.*) {
                .iree => |*v| {
                    const consumed, const token_ids = try v.feed(tokens);
                    return .{ .consumed = consumed, .token_ids = token_ids };
                },
                inline else => |*v| {
                    const token_ids = try v.encode(tokens);
                    return .{ .consumed = tokens.len, .token_ids = token_ids };
                },
            };
        }

        /// Output is only valid until the next feed/finalize call and while the encoder is alive.
        pub fn finalize(self: *Encoder) ![]const u32 {
            return switch (self.*) {
                .iree => |*v| v.finalize(),
                inline else => &.{},
            };
        }

        pub fn encodeAlloc(self: *Encoder, allocator: std.mem.Allocator, tokens: []const u8) !std.ArrayList(u32) {
            var token_ids: std.ArrayList(u32) = try .initCapacity(allocator, tokens.len / 4);
            var remaining = tokens;
            while (remaining.len > 0) {
                const out = try self.feed(remaining);
                try token_ids.appendSlice(allocator, out.token_ids);
                remaining = remaining[out.consumed..];
            }
            try token_ids.appendSlice(allocator, try self.finalize());
            return token_ids;
        }

        pub fn writerAlloc(self: *Encoder, allocator: std.mem.Allocator, buffer: []u8, init_out_capacity: usize) !WriterAllocating {
            return try WriterAllocating.init(allocator, self, buffer, init_out_capacity);
        }

        pub const WriterAllocating = struct {
            _allocator: std.mem.Allocator,
            _encoder: *Tokenizer.Encoder,
            _token_ids: std.ArrayList(u32),
            interface: std.Io.Writer,

            pub fn init(allocator: std.mem.Allocator, encoder_: *Encoder, buffer: []u8, init_out_capacity: usize) !WriterAllocating {
                return .{
                    ._allocator = allocator,
                    ._encoder = encoder_,
                    ._token_ids = try .initCapacity(allocator, init_out_capacity),
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

            pub fn deinit(self: *WriterAllocating) void {
                self._token_ids.deinit(self._allocator);
            }

            pub fn finish(self: *WriterAllocating) std.Io.Writer.Error![]u32 {
                try self.interface.flush();
                self._token_ids.appendSlice(
                    self._allocator,
                    self._encoder.finalize() catch return error.WriteFailed,
                ) catch return error.WriteFailed;
                return self._token_ids.toOwnedSlice(self._allocator) catch return error.WriteFailed;
            }

            fn appendChunk(self: *WriterAllocating, chunk: []const u8) std.Io.Writer.Error!void {
                var remaining = chunk;
                while (remaining.len > 0) {
                    const fout = self._encoder.feed(remaining) catch |err| {
                        log.err("writer drain feed failed: {s}", .{@errorName(err)});
                        return error.WriteFailed;
                    };
                    self._token_ids.appendSlice(self._allocator, fout.token_ids) catch |err| {
                        log.err("writer drain append failed: {s}", .{@errorName(err)});
                        return error.WriteFailed;
                    };
                    remaining = remaining[fout.consumed..];
                }
            }

            fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
                if (w.end == 0) return;
                const self: *WriterAllocating = @alignCast(@fieldParentPtr("interface", w));
                try self.appendChunk(w.buffer[0..w.end]);
                w.end = 0;
            }

            fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
                const self: *WriterAllocating = @alignCast(@fieldParentPtr("interface", w));

                if (w.end != 0) {
                    try self.appendChunk(w.buffer[0..w.end]);
                    w.end = 0;
                }

                var total: usize = 0;
                // Prefix chunks: written once each.
                for (data[0 .. data.len - 1]) |chunk| {
                    try self.appendChunk(chunk);
                    total += chunk.len;
                }
                // Last chunk is the splat pattern.
                const pattern = data[data.len - 1];
                if (pattern.len == 0) return total;
                for (0..splat) |_| {
                    try self.appendChunk(pattern);
                    total += pattern.len;
                }
                return total;
            }
        };
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

        pub const FeedOutput = struct {
            consumed: usize,
            /// Only valid until the next feed/finalize call and while the decoder is alive.
            tokens: []const u8,
        };

        pub fn feed(self: *Decoder, token_ids: []const u32) !FeedOutput {
            return switch (self.*) {
                .iree => |*v| {
                    const consumed, const tokens = try v.feed(token_ids);
                    return .{ .consumed = consumed, .tokens = tokens };
                },
                inline else => |*v| {
                    const tokens = try v.decode(token_ids);
                    return .{ .consumed = token_ids.len, .tokens = tokens };
                },
            };
        }

        /// Output is only valid until the next feed/finalize call and while the decoder is alive.
        pub fn finalize(self: *Decoder) ![]const u8 {
            return switch (self.*) {
                .iree => |*v| v.finalize(),
                inline else => &.{},
            };
        }

        pub fn decodeAlloc(self: *Decoder, allocator: std.mem.Allocator, token_ids: []const u32) !std.ArrayList(u8) {
            var tokens: std.ArrayList(u8) = try .initCapacity(allocator, token_ids.len * 4);
            var remaining = token_ids;
            while (remaining.len > 0) {
                const fout = try self.feed(remaining);
                try tokens.appendSlice(allocator, fout.tokens);
                remaining = remaining[fout.consumed..];
            }
            try tokens.appendSlice(allocator, try self.finalize());
            return tokens;
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
        if (bytes[0] == '{') {
            return .{ .iree = try .fromBytes(allocator, bytes) };
        }
        return .{ .sentencepiece = try .fromBytes(bytes) };
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

    pub fn findTokenId(self: *const Tokenizer, token: []const u8) ?u32 {
        return switch (self.*) {
            inline else => |v| v.findTokenId(token),
        };
    }
};

test {
    std.testing.refAllDecls(@This());
}

test "iree tokenizer smoke: alloc and writer" {
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

    var token_ids_alloc = try encoder.encodeAlloc(allocator, text);
    defer token_ids_alloc.deinit(allocator);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, token_ids_alloc.items);

    encoder.reset();
    var write_buffer: [64]u8 = undefined;
    var writer = try encoder.writerAlloc(allocator, &write_buffer, 4);
    defer writer.deinit();
    try writer.interface.writeAll(text);
    const token_ids_writer = try writer.finish();
    defer allocator.free(token_ids_writer);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, token_ids_writer);

    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    var tokens = try decoder.decodeAlloc(allocator, token_ids_writer);
    defer tokens.deinit(allocator);
    try std.testing.expectEqualStrings("hello world", tokens.items);
}
