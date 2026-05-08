//! Triton-side verification entry point. `Driver.compareSweep` emits both
//! Zig and Python TTIR, pushes them through XLA, and returns a populated
//! `KernelRow`. Per-step failures fold into a `pipeline_error` row.

const std = @import("std");

const mlir = @import("mlir");
const zml = @import("zml");
const bazel_runfiles = @import("runfiles");

const harness = @import("harness");

const pipeline = @import("pipeline.zig");

const errorRow = harness.report.errorRow;

const Allocator = std.mem.Allocator;
const Io = std.Io;

const PyTtirTask = struct {
    arena: std.heap.ArenaAllocator,
    runner: *harness.py_runner.Runner,
    cfg_json: []const u8,
    progress: std.Progress.Node,
    ttir: ?[]const u8 = null,
    err: ?anyerror = null,
    reason: []const u8 = "",

    fn run(self: *PyTtirTask) void {
        var step = self.progress.start("python lower", 0);
        defer step.end();

        const response = self.runner.requestCompile(self.arena.allocator(), self.cfg_json) catch |err| {
            self.err = err;
            return;
        };
        self.ttir = response.ttir orelse {
            self.err = error.MissingTtir;
            self.reason = "missing ttir in py response";
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
    platform: *const zml.Platform,
    repo_rf: *const bazel_runfiles.Runfiles.WithSourceRepo,
    environ_map: *const std.process.Environ.Map,
    normalizer: *const harness.normalize.Normalizer,
    /// Each sweep gets a fresh `<out_dir>/<kernel>__<sweep>/{zig,py}` subtree.
    out_dir: []const u8,
    show_diffs: bool = false,
    /// Empty = all stages; else one of ttir|ttgir|llir|ptx.
    stage_filter: []const u8 = "",

    pub fn compareSweep(
        self: Driver,
        entry: *const harness.KernelEntry,
        sref: harness.SweepRef,
        idx: usize,
        runner: *harness.py_runner.Runner,
        progress: std.Progress.Node,
    ) harness.report.KernelRow {
        if (entry.compileFn == null) {
            return errorRow(entry, sref.label, "registration file not wired for the XLA pipeline; add forward+args+setActiveTtir");
        }

        const cfg_json = entry.cfgJsonFn(self.arena, idx) catch |err|
            return errorRow(entry, sref.label, @errorName(err));

        var py_task: PyTtirTask = .{
            .arena = std.heap.ArenaAllocator.init(self.gpa),
            .runner = runner,
            .cfg_json = cfg_json,
            .progress = progress,
        };
        defer py_task.arena.deinit();

        var lower_group: std.Io.Group = .init;
        lower_group.concurrent(self.io, PyTtirTask.run, .{&py_task}) catch |err|
            return errorRow(entry, sref.label, @errorName(err));

        var zig_arena = std.heap.ArenaAllocator.init(self.gpa);
        defer zig_arena.deinit();
        const zig_ttir = blk: {
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
        const py_ttir_raw = py_task.ttir.?;

        const zig_ttir_z = self.arena.dupeZ(u8, zig_ttir) catch |err|
            return errorRow(entry, sref.label, @errorName(err));
        const py_ttir_z = self.arena.dupeZ(u8, py_ttir_raw) catch |err|
            return errorRow(entry, sref.label, @errorName(err));

        var run_result = pipeline.runBothSides(
            self.gpa,
            self.io,
            self.platform,
            entry,
            sref.label,
            zig_ttir_z,
            py_ttir_z,
            self.out_dir,
            progress,
        ) catch |err| return errorRow(entry, sref.label, @errorName(err));
        defer run_result.deinit();

        const stages = buildStageRows(self.arena, self.normalizer, &run_result, progress, .{
            .stage_filter = self.stage_filter,
            .capture_body = self.show_diffs,
        }) catch |err| return errorRow(entry, sref.label, @errorName(err));

        return .{
            .kernel = entry.name,
            .sweep = sref.label,
            .kind = .triton,
            .classification = harness.report.classify(stages),
            .stages = stages,
        };
    }
};

/// `name` doubles as a `pipeline.PerStage` field accessed via `@field`
/// below — keep the two in sync. `is_cosmetic` flags the pre-LLIR
/// stages that XLA canonicalizes downstream.
const STAGES = [_]struct { name: []const u8, lang: harness.normalize.Lang, is_cosmetic: bool }{
    .{ .name = "ttir", .lang = .triton_mlir, .is_cosmetic = true },
    .{ .name = "ttgir", .lang = .triton_mlir, .is_cosmetic = true },
    .{ .name = "llir", .lang = .llvm_ir, .is_cosmetic = false },
    .{ .name = "ptx", .lang = .ptx, .is_cosmetic = false },
};

const DiffOptions = struct {
    stage_filter: []const u8 = "",
    capture_body: bool = false,
};

fn buildStageRows(
    arena: Allocator,
    normalizer: *const harness.normalize.Normalizer,
    rr: *const pipeline.RunResult,
    progress: std.Progress.Node,
    opts: DiffOptions,
) ![]const harness.report.StageRow {
    var rows = try arena.alloc(harness.report.StageRow, STAGES.len);
    inline for (STAGES, 0..) |s, i| {
        if (opts.stage_filter.len > 0 and !std.mem.eql(u8, opts.stage_filter, s.name)) {
            rows[i] = .{ .name = s.name, .verdict = .skipped };
        } else if (@field(rr.zig, s.name)) |zig_blob| if (@field(rr.py, s.name)) |py_blob| {
            const norm: struct { zig: []const u8, py: []const u8 } = blk: {
                var step = progress.startFmt(0, "strip {s}", .{s.name});
                defer step.end();
                break :blk .{
                    .zig = try normalizer.normalize(arena, zig_blob, s.lang),
                    .py = try normalizer.normalize(arena, py_blob, s.lang),
                };
            };
            const r = blk: {
                var step = progress.startFmt(0, "diff {s}", .{s.name});
                defer step.end();
                break :blk try harness.diff.diffText(arena, norm.zig, norm.py, 2, opts.capture_body);
            };
            const is_cosmetic = opts.stage_filter.len == 0 and s.is_cosmetic;
            rows[i] = if (r.is_match)
                .{ .name = s.name, .verdict = .match, .is_cosmetic = is_cosmetic }
            else
                .{ .name = s.name, .verdict = .diff, .added = r.added, .removed = r.removed, .diff_blob = r.body, .is_cosmetic = is_cosmetic };
        } else {
            rows[i] = .{ .name = s.name, .verdict = .missing_py };
        } else {
            rows[i] = .{ .name = s.name, .verdict = .missing_zig };
        }
    }
    return rows;
}
