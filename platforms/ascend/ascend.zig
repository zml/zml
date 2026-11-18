const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/ascend");

const plugin_library = "build/code/xla/bazel-bin/xla/xla_npu/pjrt/c/pjrt_c_api_npu_plugin.so";

pub fn isEnabled() bool {
    return platforms_options.ascend_enabled;
}

fn hasAscendDevice(io: std.Io) bool {
    std.Io.Dir.accessAbsolute(io, "/dev/davinci0", .{ .read = true }) catch return false;
    return true;
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    _ = allocator;
    if (comptime !isEnabled() or builtin.os.tag != .linux) return error.Unavailable;
    if (comptime builtin.cpu.arch != .x86_64 and builtin.cpu.arch != .aarch64) return error.Unavailable;
    if (!hasAscendDevice(io)) return error.Unavailable;

    if (std.c.getenv("ZML_ASCEND_PJRT_PLUGIN")) |library| {
        const path = std.mem.span(library);
        log.info("Loading Ascend plugin: {s}", .{path});
        return .loadFrom(path);
    }

    if (std.c.getenv("XLA_NPU_ROOT_DIR")) |root| {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const path = try stdx.Io.Dir.path.bufJoinZ(&path_buf, &.{ std.mem.span(root), plugin_library });
        log.info("Loading Ascend plugin: {s}", .{path});
        return .loadFrom(path);
    }

    log.err(
        "Ascend NPU found but no PJRT plugin was configured. Set ZML_ASCEND_PJRT_PLUGIN to the " ++
            "XLA-NPU pjrt_c_api_npu_plugin.so path, or source xla-npu's build/xla_npu_env so " ++
            "XLA_NPU_ROOT_DIR is set.",
        .{},
    );
    return error.FileNotFound;
}
