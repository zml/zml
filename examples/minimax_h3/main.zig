const std = @import("std");

const boot = @import("boot.zig");

// =============================================================================
// main.zig — process entry
// =============================================================================

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main(init: std.process.Init) !void {
    return boot.run(init);
}
