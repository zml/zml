const std = @import("std");

// Map the upstream Neuron bridge callback contract onto the target strings
// expected by `neuronx-cc`.
pub fn resolveTarget(platform_version: []const u8) ![]const u8 {
    return std.StaticStringMap([]const u8).initComptime(.{
        .{ "1.0", "inf1" },
        .{ "2.0", "trn1" },
        .{ "3.0", "trn2" },
    }).get(platform_version) orelse error.UnknownPlatformVersion;
}

// Translate the Zig log level into the Neuron compiler verbosity flag.
pub fn compilerVerbosity() []const u8 {
    return switch (std.options.log_level) {
        .debug => "debug",
        .info => "info",
        .warn => "warning",
        .err => "error",
    };
}
