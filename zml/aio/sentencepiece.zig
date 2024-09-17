const std = @import("std");
const asynk = @import("async");
const zml = @import("../zml.zig");

const sentencepiece_proto = @import("//sentencepiece:model_proto");
const Normalizer = zml.tokenizer.Normalizer;
const Tokenizer = zml.tokenizer.Tokenizer;

pub fn loadTokenizerFromPath(allocator: std.mem.Allocator, path: []const u8) !Tokenizer {
    const file = try asynk.File.open(path, .{});
    defer file.close() catch unreachable;

    return loadTokenizerFromFile(allocator, file);
}

pub fn loadTokenizerFromFile(allocator: std.mem.Allocator, file: asynk.File) !Tokenizer {
    const reader = file.reader();
    const input = try reader.readAllAlloc(allocator, 16 * 1024 * 1024);
    defer allocator.free(input);

    var proto_arena = std.heap.ArenaAllocator.init(allocator);
    defer proto_arena.deinit();

    const model = try sentencepiece_proto.ModelProto.decode(input, proto_arena.allocator());
    // no deinit, memory will be freed by the proto_arena

    return loadTokenizerFromModelProto(allocator, model);
}

pub fn loadTokenizerFromModelProto(allocator: std.mem.Allocator, model: sentencepiece_proto.ModelProto) !Tokenizer {
    std.debug.assert(model.trainer_spec.?.model_type.? == .BPE);
    const special_tokens: Tokenizer.SpecialTokens = .{
        .unk = @intCast(model.trainer_spec.?.unk_id.?),
        .bos = @intCast(model.trainer_spec.?.bos_id.?),
        .eos = @intCast(model.trainer_spec.?.eos_id.?),
        .pad = parseTokenId(model.trainer_spec.?.pad_id),
    };

    var tokenizer = try Tokenizer.init(
        allocator,
        @intCast(model.pieces.items.len),
        @intCast(model.trainer_spec.?.max_sentencepiece_length.?),
        normalizerFromSpec(model.normalizer_spec.?),
        special_tokens,
        true,
    );
    errdefer tokenizer.deinit();

    for (model.pieces.items) |*piece| {
        try tokenizer.addToken(piece.score.?, piece.piece.?.getSlice());
    }

    return tokenizer;
}

fn parseTokenId(id: ?i32) u32 {
    if (id) |idx| {
        if (idx > 0) return @intCast(idx);
    }

    return std.math.maxInt(u32);
}

pub fn normalizerFromSpec(spec: sentencepiece_proto.NormalizerSpec) Normalizer {
    std.log.info("NormalizerSpec: {}", .{spec});
    if (spec.normalization_rule_tsv) |rule_tsv| {
        if (!rule_tsv.isEmpty()) {
            std.debug.panic("SentencePiece model with normalization rules not supported: model.normalizer_spec.normalization_rule_tsv: {s}", .{spec.normalization_rule_tsv.?.getSlice()});
        }
    }
    if (!std.mem.eql(u8, spec.name.?.getSlice(), "identity")) std.debug.panic("Normalizer only supports NormalizerSpec with name \"identity\", got \"{s}\"", .{spec.name.?.getSlice()});
    if (!spec.escape_whitespaces.?) std.debug.panic("Normalizer only supports NormalizerSpec with \"escape_whitespaces\" flag set", .{});
    if (spec.remove_extra_whitespaces) |_| {} else std.debug.panic("Normalizer only supports NormalizerSpec with \"remove_extra_whitespaces\" flag set", .{});
    if (spec.add_dummy_prefix) |_| {} else std.debug.panic("Normalizer only supports NormalizerSpec with \"add_dummy_prefix\" flag set", .{});
    return .{ .flags = .{
        .remove_extra_whitespaces = spec.remove_extra_whitespaces orelse false,
        .add_dummy_prefix = spec.add_dummy_prefix orelse true,
    } };
}
