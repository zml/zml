const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");

pub const LaunchDeviceFn = *const fn (
    stream: ?*anyopaque,
    activations_bf16: ?*const anyopaque,
    weights_fp4_e2m1_packed: ?*const u8,
    scales_ue8m0: ?*const u8,
    expert_first_token_offset_device: ?*const i64,
    num_experts: c_int,
    hidden_size: c_int,
    out_features: c_int,
    output_bf16: ?*anyopaque,
) callconv(.c) c_int;

pub var moe_cutlass_sm90_bf16_mxfp4_launch_device: LaunchDeviceFn = undefined;
var flashinfer_lib: ?std.DynLib = null;

const default_candidates = [_][]const u8{
    "/home/louis/flashinfer-gptoss-sm90-bf16-mxfp4-standalone/build_gptoss/libgptoss_moe_cutlass_sm90.so",
    // "/home/louis/flashinfer-gptoss-sm90-bf16-mxfp4-standalone/build_device/libgptoss_moe_cutlass_sm90.so",
    // "/home/louis/flashinfer-gptoss-sm90-bf16-mxfp4-standalone/build/libgptoss_moe_cutlass_sm90.so",
};

fn openLibrary(path: []const u8) !void {
    var lib = std.DynLib.open(path) catch |err| {
        std.log.warn("Failed to open FlashInfer MoE library at {s}: {any}", .{ path, err });
        return err;
    };
    moe_cutlass_sm90_bf16_mxfp4_launch_device =
        lib.lookup(LaunchDeviceFn, "moe_cutlass_sm90_bf16_mxfp4_launch_device") orelse return error.NotFound;
    flashinfer_lib = lib;
}

pub fn load(allocator: std.mem.Allocator) !void {
    _ = allocator;

    if (flashinfer_lib != null) return;

    if (std.c.getenv("FLASHINFER_MOE_KERNEL_SO")) |value_z| {
        const env_path = std.mem.span(value_z);
        openLibrary(env_path) catch |err| {
            std.log.err("Failed to load FLASHINFER_MOE_KERNEL_SO={s}: {any}", .{ env_path, err });
            return err;
        };
        return;
    }

    const r = try bazel.runfiles(bazel_builtin.current_repository);
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    if (try r.rlocation("flashinfer-gptoss-sm90-bf16-mxfp4-standalone/build_gptoss/libgptoss_moe_cutlass_sm90.so", &buffer)) |so_path| {
        if (openLibrary(so_path)) |_| {
            return;
        } else |_| {}
    }

    inline for (default_candidates) |candidate| {
        if (openLibrary(candidate)) |_| {
            return;
        } else |_| {}
    }

    return error.NotFound;
}
