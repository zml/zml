const std = @import("std");

pub const Report = struct {
    has_adaln_proj: bool = false,
    has_time: bool = false,
};

fn hasKey(keys: []const []const u8, suffix: []const u8) bool {
    for (keys) |key| {
        if (std.mem.endsWith(u8, key, suffix) or std.mem.eql(u8, key, suffix)) return true;
    }
    return false;
}

pub fn inspect(keys: []const []const u8) Report {
    return .{
        .has_adaln_proj = hasKey(keys, "adaln_proj.linear.weight"),
        .has_time = hasKey(keys, "time_embedder.proj_in.weight") or hasKey(keys, "time_embedder.linear_1.weight") or hasKey(keys, "adaln_t_table"),
    };
}

pub fn refuseReason(report: Report) ?[]const u8 {
    if (!report.has_adaln_proj) return "AdaLN projection weights missing; not a recognized H3 DiT";
    if (!report.has_time) return "neither time_embedder nor adaln_t_table; not a recognized H3 DiT";
    return null;
}

pub const bundle_leaves = [_][]const u8{ "diffusion_models", "text_encoders", "vae" };

fn containsIgnoreCase(hay: []const u8, needle: []const u8) bool {
    if (hay.len < needle.len) return false;
    var i: usize = 0;
    while (i + needle.len <= hay.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(hay[i..][0..needle.len], needle)) return true;
    }
    return false;
}

/// `needles` are all required (AND). Empty `needles` matches any `.safetensors` file.
pub fn safetensorsContains(name: []const u8, needles: []const []const u8) bool {
    if (!std.mem.endsWith(u8, name, ".safetensors")) return false;
    for (needles) |needle| {
        if (!containsIgnoreCase(name, needle)) return false;
    }
    return true;
}

pub fn isBundleLeaf(name: []const u8) bool {
    for (bundle_leaves) |leaf| {
        if (std.mem.eql(u8, name, leaf)) return true;
    }
    return false;
}
