const std = @import("std");
const builtin = @import("builtin");

const asynk = @import("async");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const libneuronxla_pyenv = @import("libneuronxla_pyenv");
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

    var buf = [_]u8{0} ** AmazonEC2.len;
    _ = try f.reader().readAll(&buf);

    return std.mem.eql(u8, &buf, AmazonEC2);
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

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
        stdx.debug.panic("Unable to find runfiles", .{});
    };

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("libpjrt_neuron/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for NEURON runtime", .{});
        return error.FileNotFound;
    };

    // setNeuronCCFlags();
    // try initialize(arena.allocator(), &r_);

    return blk: {
        var lib_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try stdx.fs.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_neuron.so" });
        break :blk asynk.callBlocking(pjrt.Api.loadFrom, .{path});
    };
}
