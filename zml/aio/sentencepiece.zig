const std = @import("std");
const asynk = @import("async");
const zml = @import("../zml.zig");

const sentencepiece_proto = @import("//sentencepiece:model_proto");
const Normalizer = zml.tokenizer.Normalizer;

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

    return Normalizer.init(
        .{
            .remove_extra_whitespaces = spec.remove_extra_whitespaces orelse false,
            .add_dummy_prefix = spec.add_dummy_prefix orelse false,
            .add_dummy_suffix = false,
            .lower_case_ascii = false,
            .split_on_punct_ascii = false,
        },
        if (spec.escape_whitespaces orelse false) Normalizer.sentencepiece_space else null,
    );
}
