const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const dspark = @import("dspark_model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,

    pub const help =
        \\Options:
        \\  --model=<path>          Path to the DeepSeek V4 model repository
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const args = stdx.flags.parse(init.minimal.args, Args);
    _ = args;
    _ = dspark.Model;
}
