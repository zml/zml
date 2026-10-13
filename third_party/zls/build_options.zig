const std = @import("std");

pub const version_string = "0.17.0-dev+eab2be0fd744";
pub const version = std.SemanticVersion.parse(version_string) catch unreachable;
pub const minimum_runtime_zig_version_string = "0.17.0";
