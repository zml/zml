//! Drive both Zig and Python TTIR through the XLA Triton pipeline and
//! slice each post-pipeline stage (TTIR/TTGIR/LLIR/PTX) out of the dump
//! dir. Pushing both sides through the same lowering is what makes
//! pre-XLA cosmetic divergence cancel.

const std = @import("std");

const Allocator = std.mem.Allocator;
const Io = std.Io;

const harness = @import("harness");
const triton_extract = @import("extract.zig");
const zml = @import("zml");

const log = std.log.scoped(.@"harness/triton/pipeline");

pub const PerStage = struct {
    ttir: ?[]const u8 = null,
    ttgir: ?[]const u8 = null,
    llir: ?[]const u8 = null,
    ptx: ?[]const u8 = null,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *PerStage) void {
        self.arena.deinit();
    }
};

fn extractStages(
    parent: Allocator,
    io: Io,
    dump_dir: []const u8,
    kernel_name: []const u8,
) !PerStage {
    var stages: PerStage = .{ .arena = std.heap.ArenaAllocator.init(parent) };
    errdefer stages.deinit();
    const a = stages.arena.allocator();

    var dir = std.Io.Dir.cwd().openDir(io, dump_dir, .{ .iterate = true }) catch |err| {
        log.warn("cannot open dump dir '{s}': {s}", .{ dump_dir, @errorName(err) });
        return stages;
    };
    defer dir.close(io);

    const tritonToLlvmSuffix = ".triton-to-llvm.txt";
    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        switch (entry.kind) {
            .file => {},
            else => continue,
        }
        const fname = entry.name;
        if (std.mem.endsWith(u8, fname, tritonToLlvmSuffix)) {
            // Stem is `<program>.<kernel>` — match the kernel name after the dot.
            const stem = fname[0 .. fname.len - tritonToLlvmSuffix.len];
            if (std.mem.indexOfScalar(u8, stem, '.') == null) continue;
            const after_dot = stem[std.mem.indexOfScalar(u8, stem, '.').? + 1 ..];
            if (!std.mem.eql(u8, after_dot, kernel_name)) continue;

            const path = try std.fs.path.join(a, &.{ dump_dir, fname });
            const text = try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .unlimited, .of(u8), 0);
            const snaps = try triton_extract.splitSnapshots(a, text);
            if (triton_extract.extractTtir(snaps)) |body| {
                stages.ttir = body;
            }
            if (triton_extract.extractTtgir(snaps)) |body| {
                stages.ttgir = body;
            }
            continue;
        }
        if (triton_extract.matchLlirFilename(fname)) |k| {
            if (!std.mem.eql(u8, k, kernel_name)) continue;
            const path = try std.fs.path.join(a, &.{ dump_dir, fname });
            stages.llir = try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .unlimited, .of(u8), 0);
            continue;
        }
        if (triton_extract.matchPtxFilename(fname)) |k| {
            if (!std.mem.eql(u8, k, kernel_name)) continue;
            const path = try std.fs.path.join(a, &.{ dump_dir, fname });
            stages.ptx = try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .unlimited, .of(u8), 0);
            continue;
        }
    }

    return stages;
}

pub const RunResult = struct {
    zig: PerStage,
    py: PerStage,

    pub fn deinit(self: *RunResult) void {
        self.zig.deinit();
        self.py.deinit();
    }
};

const CompileTask = struct {
    parent_alloc: Allocator,
    io: Io,
    platform: *const zml.Platform,
    compileFn: harness.CompileFn,
    ttir: [:0]const u8,
    out_dir: []const u8,
    progress: std.Progress.Node,
    label: []const u8,
    err: ?anyerror = null,

    fn run(self: *CompileTask) void {
        var step = self.progress.start(self.label, 0);
        defer step.end();

        var arena = std.heap.ArenaAllocator.init(self.parent_alloc);
        defer arena.deinit();

        self.compileFn(arena.allocator(), self.io, self.platform, self.ttir, self.out_dir) catch |err| {
            self.err = err;
        };
    }
};

/// Each side gets its own `parent_out_dir/<kernel>__<sweep>/{zig,py}`
/// dump tree before stages get sliced out.
pub fn runBothSides(
    parent_alloc: Allocator,
    io: Io,
    platform: *const zml.Platform,
    entry: *const harness.KernelEntry,
    sweep_label: []const u8,
    zig_ttir: [:0]const u8,
    py_ttir: [:0]const u8,
    parent_out_dir: []const u8,
    progress: std.Progress.Node,
) !RunResult {
    const compileFn = entry.compileFn orelse return error.XlaDriverNotWired;

    const stem = try std.fmt.allocPrint(parent_alloc, "{s}__{s}", .{ entry.name, sweep_label });
    defer parent_alloc.free(stem);
    const zig_dir = try std.fs.path.join(parent_alloc, &.{ parent_out_dir, stem, "zig" });
    defer parent_alloc.free(zig_dir);
    const py_dir = try std.fs.path.join(parent_alloc, &.{ parent_out_dir, stem, "py" });
    defer parent_alloc.free(py_dir);

    try std.Io.Dir.createDirPath(.cwd(), io, zig_dir);
    try std.Io.Dir.createDirPath(.cwd(), io, py_dir);

    var group: std.Io.Group = .init;
    var zig_compile: CompileTask = .{
        .parent_alloc = parent_alloc,
        .io = io,
        .platform = platform,
        .compileFn = compileFn,
        .ttir = zig_ttir,
        .out_dir = zig_dir,
        .progress = progress,
        .label = "xla pipeline (zig)",
    };
    var py_compile: CompileTask = .{
        .parent_alloc = parent_alloc,
        .io = io,
        .platform = platform,
        .compileFn = compileFn,
        .ttir = py_ttir,
        .out_dir = py_dir,
        .progress = progress,
        .label = "xla pipeline (py)",
    };
    try group.concurrent(io, CompileTask.run, .{&zig_compile});
    group.concurrent(io, CompileTask.run, .{&py_compile}) catch |err| {
        group.cancel(io);
        group.await(io) catch {};
        return err;
    };
    try group.await(io);
    if (zig_compile.err) |err| return err;
    if (py_compile.err) |err| return err;

    const zig = try extractStages(parent_alloc, io, zig_dir, entry.name);
    errdefer {
        var z = zig;
        z.deinit();
    }
    const py = try extractStages(parent_alloc, io, py_dir, entry.name);

    return .{ .zig = zig, .py = py };
}
