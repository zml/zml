const std = @import("std");

pub const version_string = "0.17.0-dev+4e83599939e9";
pub const version = std.SemanticVersion.parse(version_string) catch unreachable;
pub const minimum_runtime_zig_version_string = "0.17.0";
