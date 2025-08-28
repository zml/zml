const std = @import("std");
const builtin = @import("builtin");

const asynk = @import("async");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/runtime/neuron");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_NEURON");
}

fn hasNeuronDevice() bool {
    asynk.File.access("/dev/neuron0", .{ .mode = .read_only }) catch return false;
    return true;
}

fn isRunningOnEC2() !bool {
    const AmazonEC2 = "Amazon EC2";

    var f = try asynk.File.open("/sys/devices/virtual/dmi/id/sys_vendor", .{ .mode = .read_only });
    defer f.close() catch {};

    var content: [AmazonEC2.len]u8 = undefined;
    const n_read = try f.pread(&content, 0);

    return std.mem.eql(u8, content[0..n_read], AmazonEC2);
}

pub fn load() !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!(isRunningOnEC2() catch false)) {
        return error.Unavailable;
    }
    if (!hasNeuronDevice()) {
        return error.Unavailable;
    }

    var arena = std.heap.ArenaAllocator.init(std.heap.smp_allocator);
    defer arena.deinit();

    var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
        stdx.debug.panic("Unable to find runfiles", .{});
    };

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);

    var sandbox_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_neuron/sandbox", &sandbox_path_buf) orelse {
        log.err("Failed to find sandbox path for NEURON runtime", .{});
        return error.FileNotFound;
    };

    return blk: {
        var lib_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try stdx.fs.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_neuron.so" });
        break :blk asynk.callBlocking(pjrt.Api.loadFrom, .{path});
    };
}
