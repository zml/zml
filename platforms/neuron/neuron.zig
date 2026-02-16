const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/neuron");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_NEURON");
}

fn hasNeuronDevice(io: std.Io) bool {
    std.Io.Dir.accessAbsolute(io, "/dev/neuron0", .{ .read = true }) catch return false;
    return true;
}

fn isRunningOnEC2(io: std.Io) !bool {
    const AmazonEC2 = "Amazon EC2";
    var buffer: [AmazonEC2.len]u8 = undefined;
    return std.mem.eql(
        u8,
        AmazonEC2,
        try std.Io.Dir.readFile(
            .cwd(),
            io,
            "/sys/devices/virtual/dmi/id/sys_vendor",
            &buffer,
        ),
    );
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator; // autofix
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!(isRunningOnEC2(io) catch false)) {
        return error.Unavailable;
    }
    if (!hasNeuronDevice(io)) {
        return error.Unavailable;
    }

    const r = try bazel.runfiles(io, bazel_builtin.current_repository);

    var sandbox_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_neuron/sandbox", &sandbox_path_buf) orelse {
        log.err("Failed to find sandbox path for NEURON runtime", .{});
        return error.FileNotFound;
    };

    return blk: {
        var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_neuron.so" });
        break :blk .loadFrom(path);
    };
}
