//! Mosaic-TPU verification entry point. The Zig DSL emits already-
//! canonicalized MLIR; we canonicalize the Pallas side here so the diff
//! is apples-to-apples. Per-step failures fold into a `pipeline_error` row.

const std = @import("std");

const mlir = @import("mlir");
const bazel_runfiles = @import("runfiles");

const harness = @import("harness");

const canonicalize_lib = @import("canonicalize.zig");

const errorRow = harness.report.errorRow;

const Allocator = std.mem.Allocator;
const Io = std.Io;

const PyMosaicTask = struct {
    arena: std.heap.ArenaAllocator,
    runner: *harness.py_runner.Runner,
    cfg_json: []const u8,
    progress: std.Progress.Node,
    mlir: ?[]const u8 = null,
    err: ?anyerror = null,
    reason: []const u8 = "",

    fn run(self: *PyMosaicTask) void {
        var step = self.progress.start("pallas lower", 0);
        defer step.end();

        const response = self.runner.requestCompile(self.arena.allocator(), self.cfg_json) catch |err| {
            self.err = err;
            return;
        };
        self.mlir = response.mosaic orelse {
            self.err = error.MissingMosaicMlir;
            self.reason = "missing mosaic mlir in py response";
            return;
        };
    }
};

fn taskReason(err: anyerror, reason: []const u8) []const u8 {
    return if (reason.len > 0) reason else @errorName(err);
}

pub const Driver = struct {
    gpa: Allocator,
    arena: Allocator,
    io: Io,
    ctx: *mlir.Context,
    repo_rf: *const bazel_runfiles.Runfiles.WithSourceRepo,
    environ_map: *const std.process.Environ.Map,
    normalizer: *const harness.normalize.Normalizer,
    show_diffs: bool = false,

    pub fn compareSweep(
        self: Driver,
        entry: *const harness.KernelEntry,
        sref: harness.SweepRef,
        idx: usize,
        runner: *harness.py_runner.Runner,
        progress: std.Progress.Node,
    ) harness.report.KernelRow {
        const cfg_json = entry.cfgJsonFn(self.arena, idx) catch |err|
            return errorRow(entry, sref.label, @errorName(err));

        var py_task: PyMosaicTask = .{
            .arena = std.heap.ArenaAllocator.init(self.gpa),
            .runner = runner,
            .cfg_json = cfg_json,
            .progress = progress,
        };
        defer py_task.arena.deinit();

        var lower_group: std.Io.Group = .init;
        lower_group.concurrent(self.io, PyMosaicTask.run, .{&py_task}) catch |err|
            return errorRow(entry, sref.label, @errorName(err));

        var zig_arena = std.heap.ArenaAllocator.init(self.gpa);
        defer zig_arena.deinit();
        const zig_mlir = blk: {
            var step = progress.start("zig emit", 0);
            defer step.end();
            break :blk entry.emitFn(zig_arena.allocator(), self.ctx, idx) catch |err| {
                lower_group.await(self.io) catch {};
                return errorRow(entry, sref.label, @errorName(err));
            };
        };

        lower_group.await(self.io) catch |err|
            return errorRow(entry, sref.label, @errorName(err));
        if (py_task.err) |err| {
            return errorRow(entry, sref.label, taskReason(err, py_task.reason));
        }
        const py_mlir_raw = py_task.mlir.?;

        const py_mlir = canonicalize_lib.canonicalize(self.arena, self.ctx, py_mlir_raw) catch |err|
            return errorRow(entry, sref.label, @errorName(err));

        const norm: struct { zig: []const u8, py: []const u8 } = blk: {
            var step = progress.start("strip", 0);
            defer step.end();
            const zig_norm = self.normalizer.normalize(self.arena, zig_mlir, .mosaic_mlir) catch |err|
                return errorRow(entry, sref.label, @errorName(err));
            const py_norm = self.normalizer.normalize(self.arena, py_mlir, .mosaic_mlir) catch |err|
                return errorRow(entry, sref.label, @errorName(err));
            break :blk .{ .zig = zig_norm, .py = py_norm };
        };

        const r = blk: {
            var step = progress.start("diff", 0);
            defer step.end();
            break :blk harness.diff.diffText(self.arena, norm.zig, norm.py, 2, self.show_diffs) catch |err|
                return errorRow(entry, sref.label, @errorName(err));
        };

        var stages = self.arena.alloc(harness.report.StageRow, 1) catch |err|
            return errorRow(entry, sref.label, @errorName(err));
        stages[0] = if (r.is_match)
            .{ .name = "mlir", .verdict = .match }
        else
            .{ .name = "mlir", .verdict = .diff, .added = r.added, .removed = r.removed, .diff_blob = r.body };

        return .{
            .kernel = entry.name,
            .sweep = sref.label,
            .kind = .mosaic_tpu,
            .classification = harness.report.classify(stages),
            .stages = stages,
        };
    }
};
