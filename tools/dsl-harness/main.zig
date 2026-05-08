//! Unified Triton + Mosaic-TPU kernel verification harness. Kernels are
//! registered via macros in `kernel.bzl`; the build-time
//! `aggregator.zig` is the source of truth this file iterates over.

const std = @import("std");

const stdx = @import("stdx");
const mlir = @import("mlir");
const zml = @import("zml");
const bazel_runfiles = @import("runfiles");
const bazel_builtin = @import("bazel_builtin");

const harness = @import("harness");
const triton = @import("harness/triton");
const mosaic_tpu = @import("harness/mosaic_tpu");
const aggregator = @import("aggregator");

pub const std_options: std.Options = .{ .log_level = .info };

const log = std.log.scoped(.@"dsl-harness");

var runfiles_global: ?bazel_runfiles.Runfiles = null;

fn initRunfiles(io: std.Io) !*const bazel_runfiles.Runfiles {
    if (runfiles_global) |*rf| return rf;
    var exe_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const exe_size = try std.process.executablePath(io, &exe_path_buf);
    const rf = (try bazel_runfiles.Runfiles.create(.{
        .allocator = std.heap.c_allocator,
        .io = io,
        .argv0 = exe_path_buf[0..exe_size],
        .directory = if (std.c.getenv("RUNFILES_DIR")) |d| std.mem.span(d) else null,
        .manifest = if (std.c.getenv("RUNFILES_MANIFEST_FILE")) |m| std.mem.span(m) else null,
    })) orelse return error.NoRunfiles;
    runfiles_global = rf;
    return &runfiles_global.?;
}

/// Point per-kernel `py_binary` subprocesses at the harness's own runfiles
/// tree: their adjacent `<py_binary>.runfiles/` only materializes for
/// top-level targets, not `data` deps, so the bash bootstrap needs the
/// override.
fn injectRunfilesDir(init: std.process.Init) !void {
    const env = init.environ_map;
    if (env.get("RUNFILES_DIR") != null or env.get("RUNFILES_MANIFEST_FILE") != null) return;
    var exe_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const exe_size = try std.process.executablePath(init.io, &exe_path_buf);
    const rf_dir = try std.fmt.allocPrint(init.arena.allocator(), "{s}.runfiles", .{exe_path_buf[0..exe_size]});
    try env.put("RUNFILES_DIR", rf_dir);
}

const CompareFlags = struct {
    backend: []const u8 = "all",
    kernel: []const u8 = "",
    sweep: []const u8 = "",
    stage: []const u8 = "",
    show_diffs: bool = false,
    json: bool = false,
    out_dir: []const u8 = "",

    pub const help =
        \\dsl-harness compare [flags]
        \\
        \\Push both Zig-DSL and Python kernels through XLA's Triton pipeline
        \\and diff each post-pipeline stage. Requires CUDA or ROCm.
        \\
        \\  --backend=B       triton | mosaic_tpu | all (default: all)
        \\  --kernel=NAME     only compare this kernel (default: all)
        \\  --sweep=LABELS    comma-separated sweep labels (default: all)
        \\  --stage=S         only compare this stage (ttir|ttgir|llir|ptx)
        \\  --show-diffs      print full unified-diff bodies for non-matches
        \\  --json            machine-readable output
        \\  --out-dir=DIR     where to write per-stage artifacts (default: tmpdir)
        \\
    ;
};

const ListFlags = struct {
    json: bool = false,

    pub const help =
        \\dsl-harness list [--json]
        \\
        \\Print every registered kernel and its sweeps.
        \\
    ;
};

const CliArgs = union(enum) {
    compare: CompareFlags,
    list: ListFlags,

    pub const help =
        \\dsl-harness <subcommand> [flags]
        \\
        \\Subcommands:
        \\  compare   diff Zig-DSL kernels against their Python references
        \\  list      print every registered kernel
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    switch (args) {
        .compare => |flags| try cmdCompare(init, flags),
        .list => |flags| try cmdList(init, flags),
    }
}

fn cmdList(init: std.process.Init, flags: ListFlags) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    var stdout_buf: [16 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    defer stdout.interface.flush() catch {};
    const w = &stdout.interface;

    if (flags.json) {
        try harness.report.renderListJson(arena, w, aggregator.TRITON_KERNELS, aggregator.MOSAIC_TPU_KERNELS);
        return;
    }
    const c = harness.report.Colors.init(std.Io.File.stdout(), io);
    try harness.report.renderListHuman(w, c, aggregator.TRITON_KERNELS, aggregator.MOSAIC_TPU_KERNELS);
}

fn setupMlirContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    inline for (.{ "func", "arith", "scf", "math", "memref", "vector", "tt", "tpu", "llvm" }) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }
    mlir.registerPasses("Transforms");
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

const Backends = struct {
    triton: bool,
    mosaic_tpu: bool,
};

fn parseBackend(name: []const u8) ?Backends {
    if (std.mem.eql(u8, name, "triton")) return .{ .triton = true, .mosaic_tpu = false };
    if (std.mem.eql(u8, name, "mosaic_tpu")) return .{ .triton = false, .mosaic_tpu = true };
    if (std.mem.eql(u8, name, "all")) return .{ .triton = true, .mosaic_tpu = true };
    return null;
}

fn cmdCompare(init: std.process.Init, flags: CompareFlags) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;

    const backends = parseBackend(flags.backend) orelse {
        log.err("compare: unknown --backend '{s}' — expected triton, mosaic_tpu, or all", .{flags.backend});
        std.process.exit(2);
    };

    try injectRunfilesDir(init);

    const ctx = try setupMlirContext();
    defer ctx.deinit();

    const rf = try initRunfiles(io);
    const repo_rf = rf.withSourceRepo(bazel_builtin.current_repository);

    const out_dir = if (flags.out_dir.len > 0)
        flags.out_dir
    else
        try std.fmt.allocPrint(arena, "/tmp/dsl-harness-{d}", .{std.posix.system.getpid()});
    try std.Io.Dir.createDirPath(.cwd(), io, out_dir);

    var stdout_buf: [64 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    defer stdout.interface.flush() catch {};
    const w = &stdout.interface;

    const colors = harness.report.Colors.init(std.Io.File.stdout(), io);
    const fmt: harness.report.Format = if (flags.json) .json else .human;
    if (fmt == .human) try w.print("dump dir: {s}\n", .{out_dir});

    // Collect rows during the run; render after `progress.end()` so the
    // live progress UI never competes with the report output for the
    // same terminal lines.
    var results_arena_state = std.heap.ArenaAllocator.init(gpa);
    defer results_arena_state.deinit();
    const results_arena = results_arena_state.allocator();
    var rows: std.ArrayList(harness.report.KernelRow) = .empty;
    defer rows.deinit(gpa);

    var progress = std.Progress.start(io, .{
        .root_name = "compare",
        .estimated_total_items = totalSweeps(backends, flags),
    });

    var normalizer = try harness.normalize.Normalizer.init(gpa);
    defer normalizer.deinit();

    var agg: harness.report.Aggregate = .{};
    const env: CompareEnv = .{
        .gpa = gpa,
        .arena = arena,
        .io = io,
        .ctx = ctx,
        .repo_rf = &repo_rf,
        .environ_map = init.environ_map,
        .normalizer = &normalizer,
        .out_dir = out_dir,
        .r = .{
            .results_arena = results_arena,
            .rows = &rows,
            .agg = &agg,
            .flags = flags,
            .progress = progress,
        },
    };

    if (backends.triton) try compareTritonAll(env);
    if (backends.mosaic_tpu) try compareMosaicTpuAll(env);

    progress.end();

    for (rows.items) |row| {
        if (fmt == .json) {
            try harness.report.renderKernelJson(w, row);
        } else {
            try harness.report.renderKernelHuman(w, colors, row, flags.show_diffs);
        }
    }
    if (fmt == .json) {
        try harness.report.renderSummaryJson(w, agg);
    } else {
        try harness.report.renderSummaryHuman(w, colors, agg);
    }
    try w.flush();
    if (agg.anyFailures()) std.process.exit(1);
}

const CompareEnv = struct {
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: std.Io,
    ctx: *mlir.Context,
    repo_rf: *const bazel_runfiles.Runfiles.WithSourceRepo,
    environ_map: *const std.process.Environ.Map,
    normalizer: *const harness.normalize.Normalizer,
    out_dir: []const u8,
    r: RenderCtx,
};

fn totalSweeps(backends: Backends, flags: CompareFlags) usize {
    var n: usize = 0;
    if (backends.triton) n += countSelectedSweeps(aggregator.TRITON_KERNELS, flags);
    if (backends.mosaic_tpu) n += countSelectedSweeps(aggregator.MOSAIC_TPU_KERNELS, flags);
    return n;
}

const RenderCtx = struct {
    /// All collected rows live here so they outlive the per-sweep arena.
    results_arena: std.mem.Allocator,
    rows: *std.ArrayList(harness.report.KernelRow),
    agg: *harness.report.Aggregate,
    flags: CompareFlags,
    progress: std.Progress.Node,
};

fn collectRow(r: RenderCtx, gpa: std.mem.Allocator, row: harness.report.KernelRow) !void {
    r.agg.add(row.classification);
    const cloned = try harness.report.cloneKernelRow(r.results_arena, row);
    try r.rows.append(gpa, cloned);
}

fn kernelSelected(filter: []const u8, label: []const u8) bool {
    return filter.len == 0 or std.mem.eql(u8, filter, label);
}

fn sweepSelected(filter: []const u8, label: []const u8) bool {
    if (filter.len == 0) return true;
    var it = std.mem.splitScalar(u8, filter, ',');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t");
        if (std.mem.eql(u8, trimmed, label)) return true;
    }
    return false;
}

fn anySweepSelected(entry: *const harness.KernelEntry, filter: []const u8) bool {
    for (entry.sweeps) |sref| {
        if (sweepSelected(filter, sref.label)) return true;
    }
    return false;
}

fn collectEntryError(r: RenderCtx, gpa: std.mem.Allocator, entry: *const harness.KernelEntry, reason: []const u8) !void {
    for (entry.sweeps) |sref| {
        if (!sweepSelected(r.flags.sweep, sref.label)) continue;
        try collectRow(r, gpa, harness.report.errorRow(entry, sref.label, reason));
    }
}

fn countSelectedSweeps(kernels: []const *const harness.KernelEntry, flags: CompareFlags) usize {
    var n: usize = 0;
    for (kernels) |entry| {
        if (!kernelSelected(flags.kernel, entry.name)) continue;
        for (entry.sweeps) |sref| {
            if (!sweepSelected(flags.sweep, sref.label)) continue;
            n += 1;
        }
    }
    return n;
}

fn iterateAndCompare(
    driver: anytype,
    kernels: []const *const harness.KernelEntry,
    r: RenderCtx,
) !void {
    for (kernels) |entry| {
        if (!kernelSelected(r.flags.kernel, entry.name)) continue;
        if (!anySweepSelected(entry, r.flags.sweep)) continue;
        if (entry.kind == .triton and entry.compileFn == null) {
            try collectEntryError(r, driver.gpa, entry, "registration file not wired for the XLA pipeline; add forward+args+setActiveTtir");
            continue;
        }

        // Don't wrap spawn in its own Progress.Node — it'd be a child of
        // root and bump completed_count, throwing off the [N/total] tally
        // (total counts sweeps, not kernels). The first sweep's node
        // appears within a few hundred ms of the spawn start anyway.
        var runner = entry.spawnRunner(driver.gpa, driver.io, driver.arena, driver.repo_rf, driver.environ_map) catch |err| {
            try collectEntryError(r, driver.gpa, entry, @errorName(err));
            continue;
        };
        defer runner.deinit();

        for (entry.sweeps, 0..) |sref, idx| {
            if (!sweepSelected(r.flags.sweep, sref.label)) continue;
            var node = r.progress.startFmt(0, "{s} :: {s} [{s}]", .{ entry.name, sref.label, @tagName(entry.kind) });
            defer node.end();

            var sweep_arena = std.heap.ArenaAllocator.init(driver.gpa);
            defer sweep_arena.deinit();

            var sweep_driver = driver;
            sweep_driver.arena = sweep_arena.allocator();
            try collectRow(r, driver.gpa, sweep_driver.compareSweep(entry, sref, idx, runner, node));
        }
    }
}

fn compareTritonAll(env: CompareEnv) !void {
    // Skip CUDA/ROCm init when filters rule out every Triton sweep.
    const have_triton = for (aggregator.TRITON_KERNELS) |entry| {
        if (kernelSelected(env.r.flags.kernel, entry.name) and anySweepSelected(entry, env.r.flags.sweep)) break true;
    } else false;
    if (!have_triton) return;

    const platform = zml.Platform.auto(env.gpa, env.io, .{}) catch |err| {
        log.err("compare: cannot init zml.Platform ({s}). Triton path needs CUDA or ROCm.", .{@errorName(err)});
        std.process.exit(2);
    };
    defer platform.deinit(env.gpa, env.io);
    if (platform.target != .cuda and platform.target != .rocm) {
        log.err("compare triton: detected {s} platform; need cuda or rocm.", .{@tagName(platform.target)});
        std.process.exit(2);
    }

    const driver: triton.Driver = .{
        .gpa = env.gpa,
        .arena = env.arena,
        .io = env.io,
        .ctx = env.ctx,
        .platform = platform,
        .repo_rf = env.repo_rf,
        .environ_map = env.environ_map,
        .normalizer = env.normalizer,
        .out_dir = env.out_dir,
        .show_diffs = env.r.flags.show_diffs,
        .stage_filter = env.r.flags.stage,
    };
    try iterateAndCompare(driver, aggregator.TRITON_KERNELS, env.r);
}

fn compareMosaicTpuAll(env: CompareEnv) !void {
    const driver: mosaic_tpu.Driver = .{
        .gpa = env.gpa,
        .arena = env.arena,
        .io = env.io,
        .ctx = env.ctx,
        .repo_rf = env.repo_rf,
        .environ_map = env.environ_map,
        .normalizer = env.normalizer,
        .show_diffs = env.r.flags.show_diffs,
    };
    try iterateAndCompare(driver, aggregator.MOSAIC_TPU_KERNELS, env.r);
}
