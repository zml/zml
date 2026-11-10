const std = @import("std");
const zml = @import("zml");

// --- Loading ---

// Logical runfiles names. Repositories, revisions and checksums live in MODULE.bazel.
pub const gradium_phonon_mimi = "phonon_mimi/model.safetensors";

pub const gradium_phonon_31m_en = "phonon_31m_en/model.safetensors";
pub const gradium_phonon_31m_en_tokenizer = "phonon_31m_en/tokenizer.5.0.0.json";
pub const gradium_phonon_voice = "phonon_voices/voices/Tek4tJXiX6_yvXq7.safetensors";

pub fn assetPath(allocator: std.mem.Allocator, name: []const u8) ![]const u8 {
    const bazel_builtin = @import("bazel_builtin");
    const runfiles = try zml.bazel.runfiles(bazel_builtin.current_repository);
    return try runfiles.rlocationAlloc(allocator, name) orelse error.AssetNotFound;
}

/// Backend buffer-plan statistics for one executable; shared inputs must not be
/// counted again when estimating the footprint of several executables together.
pub fn dumpMemoryStats(name: []const u8, exe: *const zml.Exe) void {
    const stats = exe.memoryStats() catch |err| {
        std.log.warn("Memory stats unavailable for {s}: {t}", .{ name, err });
        return;
    };
    const device_data = @as(u128, stats.argument_size_in_bytes) + stats.output_size_in_bytes + stats.temp_size_in_bytes - stats.alias_size_in_bytes;
    const host_data = @as(u128, stats.host_argument_size_in_bytes) + stats.host_output_size_in_bytes + stats.host_temp_size_in_bytes - stats.host_alias_size_in_bytes;
    std.log.info(
        \\Memory stats: {s}
        \\                          device           host
        \\  generated code  {Bi:>14.2} {Bi:>14.2}
        \\  arguments       {Bi:>14.2} {Bi:>14.2}
        \\  outputs         {Bi:>14.2} {Bi:>14.2}
        \\  aliases         {Bi:>14.2} {Bi:>14.2}
        \\  temporaries     {Bi:>14.2} {Bi:>14.2}
        \\  planned data    {Bi:>14.2} {Bi:>14.2}
        \\  planned data = arguments + outputs + temporaries - aliases
    , .{
        name,
        stats.generated_code_size_in_bytes,
        stats.host_generated_code_size_in_bytes,
        stats.argument_size_in_bytes,
        stats.host_argument_size_in_bytes,
        stats.output_size_in_bytes,
        stats.host_output_size_in_bytes,
        stats.alias_size_in_bytes,
        stats.host_alias_size_in_bytes,
        stats.temp_size_in_bytes,
        stats.host_temp_size_in_bytes,
        @as(u64, @intCast(device_data)),
        @as(u64, @intCast(host_data)),
    });
}
