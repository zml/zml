//! Public types shared by the harness binary, per-backend drivers, and
//! macro-generated kernel entry shims. Imported as `@import("harness")`.

const std = @import("std");

const mlir = @import("mlir");
const zml = @import("zml");
const bazel_runfiles = @import("runfiles");

pub const diff = @import("diff.zig");
pub const normalize = @import("normalize.zig");
pub const py_runner = @import("py_runner.zig");
pub const report = @import("report.zig");

pub const KernelKind = enum { triton, mosaic_tpu };

/// Pumps a TTIR string through `zml.module.compile` with XLA's Triton
/// emitter dumping per-pass IR to `out_dir`.
pub const CompileFn = *const fn (
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    ttir: [:0]const u8,
    out_dir: []const u8,
) anyerror!void;

pub fn Sweep(comptime CfgT: type) type {
    return struct {
        label: [:0]const u8,
        cfg: CfgT,
    };
}

/// Type-erased sweep — the concrete `cfg` is only reachable via `EmitFn`,
/// which closes over the comptime `Config` type.
pub const SweepRef = struct {
    label: [:0]const u8,
};

pub const EmitFn = *const fn (
    allocator: std.mem.Allocator,
    ctx: *mlir.Context,
    cfg_idx: usize,
) anyerror![:0]const u8;

pub const CfgJsonFn = *const fn (
    allocator: std.mem.Allocator,
    cfg_idx: usize,
) anyerror![]const u8;

pub const KernelEntry = struct {
    name: [:0]const u8,
    kind: KernelKind,
    sweeps: []const SweepRef,
    emitFn: EmitFn,
    cfgJsonFn: CfgJsonFn,
    py_runfile: [:0]const u8,
    py_module: [:0]const u8,
    py_kernel_fn: [:0]const u8,
    /// Null for Mosaic kernels — they don't go through the XLA pipeline.
    compileFn: ?CompileFn,

    pub fn spawnRunner(
        entry: *const KernelEntry,
        gpa: std.mem.Allocator,
        io: std.Io,
        arena: std.mem.Allocator,
        repo_rf: *const bazel_runfiles.Runfiles.WithSourceRepo,
        environ_map: *const std.process.Environ.Map,
    ) !*py_runner.Runner {
        var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const py_path = (repo_rf.rlocation(entry.py_runfile, &path_buf) catch return error.PyRunfileResolveFailed) orelse return error.PyRunfileNotFound;
        const py_args = [_][]const u8{
            try std.fmt.allocPrint(arena, "--kernel-module={s}", .{entry.py_module}),
            try std.fmt.allocPrint(arena, "--kernel-fn={s}", .{entry.py_kernel_fn}),
        };
        return py_runner.Runner.spawn(gpa, io, py_path, &py_args, environ_map);
    }
};

/// Erase the comptime `Config` type out of a typed `SWEEPS` slice; called
/// by the kernel macro so the runtime-visible entry struct stays plain.
pub fn projectSweepRefs(comptime CfgT: type, comptime sweeps: []const Sweep(CfgT)) []const SweepRef {
    comptime {
        var refs: [sweeps.len]SweepRef = undefined;
        for (sweeps, 0..) |s, i| refs[i] = .{ .label = s.label };
        const out = refs;
        return &out;
    }
}
