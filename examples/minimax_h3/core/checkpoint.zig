const std = @import("std");

pub const AdalnKind = enum { full, curve, missing };

pub const LinearStorage = enum {
    int8_convrot,
    fp8,
    nvfp4_awq,
    unknown,
};

pub const Report = struct {
    adaln: AdalnKind = .missing,
    dit_storage: LinearStorage = .unknown,
};

fn hasKey(keys: []const []const u8, suffix: []const u8) bool {
    for (keys) |key| {
        if (std.mem.endsWith(u8, key, suffix) or std.mem.eql(u8, key, suffix)) return true;
    }
    return false;
}

fn detectAdaln(keys: []const []const u8) AdalnKind {
    const has_table = hasKey(keys, "adaln_t_table");
    const has_full = hasKey(keys, "blocks.0.adaln_proj.linear.weight") or hasKey(keys, "adaln_proj.linear.weight");
    if (has_table and !has_full) return .curve;
    if (has_full) return .full;
    if (has_table) return .curve;
    return .missing;
}

fn detectLinearStorage(keys: []const []const u8) LinearStorage {
    if (hasKey(keys, "pre_quant_scale") and (hasKey(keys, "weight.nvfp4") or hasKey(keys, "weight_scale")))
        return .nvfp4_awq;
    if (hasKey(keys, "convrot") or hasKey(keys, "hadamard") or hasKey(keys, "input_rotation"))
        return .int8_convrot;
    if (hasKey(keys, "weight_scale") and hasKey(keys, "weight")) {
        if (hasKey(keys, "pre_quant_scale")) return .nvfp4_awq;
        return .int8_convrot;
    }
    if (hasKey(keys, "weight_scale_inv")) return .fp8;
    return .unknown;
}

pub fn inspect(keys: []const []const u8) Report {
    return .{
        .adaln = detectAdaln(keys),
        .dit_storage = detectLinearStorage(keys),
    };
}

pub fn refuseReason(report: Report) ?[]const u8 {
    return switch (report.adaln) {
        .missing => "AdaLN projection weights missing; not a recognized H3 DiT",
        .curve => null,
        .full => switch (report.dit_storage) {
            .int8_convrot => "INT8 ConvRot DiT weights are not implemented",
            .fp8 => "scaled FP8 DiT weights are not implemented",
            .nvfp4_awq => "NVFP4/AWQ DiT weights are not implemented",
            .unknown => null,
        },
    };
}
