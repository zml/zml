const std = @import("std");
pub const std_options: std.Options = .{ .log_level = .info };
pub fn main(init: std.process.Init) !void {
    try @import("main.zig").run(init, false);
}
