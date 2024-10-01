const asynk = @import("async");
const core = @import("gguf/core.zig");
const std = @import("std");
const zml = @import("../zml.zig");

const HostBuffer = @import("../hostbuffer.zig").HostBuffer;

const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const log = std.log.scoped(.zml_io);

pub fn open(allocator: Allocator, path: []const u8) !zml.aio.BufferStore {
    var file = try core.GgufFile.open(path);
    errdefer file.close();

    var res: zml.aio.BufferStore = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    errdefer res.arena.deinit();
    const arena = res.arena.allocator();

    res.files = try arena.dupe(zml.aio.MemoryMappedFile, &.{file.file});

    // metadata must be read in order to read tensors
    try loadMetadata(arena, &res, &file);
    try loadBuffers(arena, &res, &file);
    if (res.buffers.count() != file.header.tensor_count) {
        log.warn("Expected to find {d} tensors in {s}, only found {d}", .{ file.header.tensor_count, path, res.buffers.count() });
    }
    return res;
}

pub fn getGgufTokenizer(self: zml.aio.BufferStore, allocator: std.mem.Allocator) !zml.tokenizer.Tokenizer {
    const tokens = self.metadataSlice("tokenizer.ggml.tokens", .string) orelse {
        log.err("GGUF File: Tokens not found", .{});
        return error.TokensNotFound;
    };
    const scores = self.metadataSlice("tokenizer.ggml.scores", .float32) orelse {
        log.err("GGUF File: Scores not found", .{});
        return error.ScoresNotFound;
    };
    assert(tokens.len == scores.len);
    const tokenizer_type = self.metadata("tokenizer.ggml.model", .string) orelse "llama";
    const tokenizer_impl: zml.tokenizer.KnownImplementation = if (std.mem.eql(u8, tokenizer_type, "gpt2")) .gpt2 else .sentencepiece;
    const bos = self.metadata("tokenizer.ggml.bos_token_id", .uint32);
    const eos = self.metadata("tokenizer.ggml.eos_token_id", .uint32);
    const unk = self.metadata("tokenizer.ggml.unknown_token_id", .uint32);
    const pad = self.metadata("tokenizer.ggml.padding_token_id", .uint32);

    const NOT_FOUND = std.math.maxInt(u32);
    const special_tokens: zml.tokenizer.Tokenizer.SpecialTokens = .{
        .bos = bos.?,
        .eos = eos.?,
        .unk = unk orelse NOT_FOUND,
        .pad = pad orelse NOT_FOUND,
    };

    // default options
    const gguf_normalizer: zml.tokenizer.Normalizer = .{ .flags = .{
        .escape_whitespaces = true,
        .remove_extra_whitespaces = true,
        .add_dummy_prefix = true,
        .add_dummy_suffix = false,
        .lower_case_ascii = false,
        .split_on_punct_ascii = false,
    } };

    const extra_tokens: u8 = if (tokenizer_impl == .gpt2) 1 else 0;
    const n_tokens: u32 = @intCast(tokens.len + extra_tokens);

    var tokenizer = try zml.tokenizer.Tokenizer.init(
        allocator,
        n_tokens,
        32,
        gguf_normalizer,
        special_tokens,
        true,
    );

    var gpt2_unicode = if (tokenizer_impl == .gpt2)
        try zml.tokenizer.Gpt2TextDecoder.init(allocator)
    else
        null;
    defer if (gpt2_unicode) |*gpt2| gpt2.deinit();
    var decoded = std.ArrayList(u8).init(allocator);
    defer decoded.deinit();

    // copy the tokens to the tokenizer arena.
    for (tokens, 0..tokens.len) |t, i| {
        if (tokenizer_impl == .gpt2) {
            decoded.clearRetainingCapacity();
            try tokenizer.addToken(scores[i], try gpt2_unicode.?.decode(&decoded, t));
            // log.debug("token: {s} -> {s}", .{t, decoded.items});
        } else {
            try tokenizer.addToken(scores[i], t);
        }
    }

    // Gpt2 tokenizer always splits on spaces.
    if (tokenizer_impl == .gpt2) {
        tokenizer.normalizer.?.flags.add_dummy_prefix = true;
        tokenizer.normalizer.?.flags.escape_whitespaces = false;
        tokenizer.special_tokens.hard_space = tokenizer.next_token_id;
        tokenizer.addOwnedToken(0, " ");
    }

    return tokenizer;
}

fn loadMetadata(allocator: Allocator, store: *zml.aio.BufferStore, file: *core.GgufFile) !void {
    try store._metadata.ensureTotalCapacity(allocator, @intCast(file.header.metadata_kv_count));

    while (file.readMetadata(allocator)) |entry| {
        log.info("Loading MetaData: {s}", .{entry.name});
        const res = store._metadata.getOrPutAssumeCapacity(entry.name);
        if (res.found_existing) {
            // This file seems invalid. Since most metadatas aren't required, continue ahead.
            log.warn("Found duplicated metadata key: {s}", .{entry.name});
            continue;
        }
        res.value_ptr.* = entry.val.asLoaderValue();
    } else |err| switch (err) {
        error.EndOfMetadata => {},
        else => return err,
    }
}

fn loadBuffers(allocator: Allocator, store: *zml.aio.BufferStore, file: *core.GgufFile) !void {
    try store.buffers.ensureTotalCapacity(allocator, @intCast(file.header.tensor_count));
    while (file.readTensorInfo(allocator)) |info| {
        const res = store.buffers.getOrPutAssumeCapacity(info.name);
        if (res.found_existing) {
            // This file seems invalid. Try to continue anyway.
            log.warn("Found duplicated tensor: {s}", .{info.name});
            continue;
        }

        // TODO: handle quantized types
        const dtype: zml.DataType = info.t.toDtype() orelse return error.UnsupportedGgufType;
        const buffer = HostBuffer.fromBytes(zml.Shape.init(info.shape(), dtype), file.file.mappedSlice(info.start, info.byte_len));
        res.value_ptr.* = buffer;
        // store the info index.
    } else |err| switch (err) {
        error.EndOfMetadata => {},
        else => return err,
    }
}
