//! Select explicit workloads, compile each graph once, then measure its runtime variants.
const std = @import("std");
const zml = @import("zml");
const cases = @import("cases.zig");
const runner = @import("runner.zig");
const results = @import("results.zig");
const log = std.log.scoped(.moe_bench);

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, cases.Args);
    args.validate() catch |err| zml.stdx.flags.fatal("invalid benchmark arguments: {s}; use --help", .{@errorName(err)});
    const platform = if (args.platform) |target| try zml.Platform.init(allocator, io, target, .{ .cpu = .{ .device_count = args.cpu_devices } }) else try zml.Platform.auto(allocator, io, .{ .cpu = .{ .device_count = args.cpu_devices } });
    defer platform.deinit(allocator, io);
    const experts_mesh = try platform.registerMesh("experts", .mesh(.{ .experts = .high_bandwidth }));
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const plan = try Plan.init(arena.allocator(), args, platform, experts_mesh);
    if ((args.dump != null or args.compare != null) and plan.runnable != 1) zml.stdx.flags.fatal("--dump/--compare require one runnable case, selected {d}; narrow axes or use --list", .{plan.runnable});

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const stdout = &stdout_file.interface;
    const json_file: ?std.Io.File = if (args.json) |path| try std.Io.Dir.createFile(.cwd(), io, path, .{}) else null;
    defer if (json_file) |f| f.close(io);
    var json_buffer: [8192]u8 = undefined;
    var json_file_writer = if (json_file) |f| f.writer(io, &json_buffer) else null;
    const json: ?*std.Io.Writer = if (json_file_writer) |*writer| &writer.interface else null;
    log.info("{s}: {d} runnable cases, {d} graphs, {d} skipped; {d} samples of at least {d} ms", .{ @tagName(platform.target), plan.runnable, plan.groups.len, plan.skipped.len, args.samples, args.sample_ms });
    const devices = try arena.allocator().alloc([]const u8, platform.devices.len);
    for (platform.devices, devices) |device, *name| {
        name.* = device.kind();
        log.info("device: {s}", .{name.*});
    }
    if (json) |writer| try results.writeJsonLine(writer, .{
        .kind = "run",
        .schema_version = 4,
        .timestamp_ns = std.Io.Timestamp.now(io, .real).nanoseconds,
        .source = results.source(),
        .platform = platform.target,
        .devices = devices,
        .device_details = try results.deviceDetails(arena.allocator(), platform),
        .environment = try results.environment(arena.allocator(), init.environ_map),
        .pjrt_version = platform.pjrt_api.version(),
        .expert_partitions = experts_mesh.numPartitionsForLogicalAxis(.experts),
        .args = args,
        .timing = "synchronous public API wall time including dispatch, output disposal and collectives; no compilation overlap",
        .padding_tiles = [_]i64{ 8, 16, 32, 64, 128 },
    });
    for (plan.skipped) |record| try results.report(stdout, json, record);
    if (args.list) {
        for (plan.groups) |group| for (group.cases.items) |selected| {
            try results.report(stdout, json, group.record(selected, .listed));
        };
        return;
    }
    if (plan.runnable == 0) return error.NoRunnableCases;

    var failures: usize = 0;
    var setup: ?runner.Setup = null;
    defer if (setup) |*s| s.deinit();
    var prepared: ?runner.Prepared = null;
    defer if (prepared) |*p| p.deinit();
    for (plan.groups) |group| failures += try runGroup(allocator, io, platform, experts_mesh, args, group, &setup, &prepared, stdout, json);
    if (failures != 0) return error.BenchmarkFailed;
}

const Group = struct {
    id: []const u8,
    cases: std.ArrayList(cases.Selection) = .empty,

    fn record(group: Group, selected: cases.Selection, status: results.Status) results.Record {
        return .{ .id = selected.id, .graph_id = group.id, .workload = selected.workload, .status = status };
    }
};

const Plan = struct {
    groups: []const Group,
    skipped: []const results.Record,
    runnable: usize,

    // Run-scoped arena owns the selection, graph keys, and grouped lists.
    fn init(allocator: std.mem.Allocator, args: cases.Args, platform: *const zml.Platform, experts_mesh: *const zml.Mesh) !Plan {
        var selection_args = args;
        if (args.backend == null) {
            var names: std.ArrayList([]const u8) = .empty;
            for (std.meta.tags(zml.moe.Backend)) |backend| {
                if (backend != .stablehlo and backend.isAvailable(platform)) try names.append(allocator, @tagName(backend));
            }
            selection_args.backend = if (names.items.len == 0) "stablehlo" else try std.mem.join(allocator, ",", names.items);
        }
        const selected = try cases.select(allocator, selection_args, platform.target, experts_mesh.numPartitionsForLogicalAxis(.experts));
        var groups: std.ArrayList(Group) = .empty;
        var skipped: std.ArrayList(results.Record) = .empty;
        var graph_indices: std.StringHashMapUnmanaged(usize) = .empty;
        defer graph_indices.deinit(allocator);
        var runnable: usize = 0;
        for (selected) |s| {
            const graph_id = try s.workload.graphId(allocator);
            if (runner.unavailable(s.workload, platform, experts_mesh)) |reason| {
                try skipped.append(allocator, .{ .id = s.id, .graph_id = graph_id, .workload = s.workload, .status = .skipped, .reason = reason });
                continue;
            }
            const entry = try graph_indices.getOrPut(allocator, graph_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = groups.items.len;
                try groups.append(allocator, .{ .id = graph_id });
            }
            try groups.items[entry.value_ptr.*].cases.append(allocator, s);
            runnable += 1;
        }
        return .{ .groups = try groups.toOwnedSlice(allocator), .skipped = try skipped.toOwnedSlice(allocator), .runnable = runnable };
    }
};

fn runGroup(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, experts_mesh: *const zml.Mesh, args: cases.Args, group: Group, setup: *?runner.Setup, prepared: *?runner.Prepared, stdout: *std.Io.Writer, json: ?*std.Io.Writer) !usize {
    log.info("compiling {s} ({d} runtime variants)", .{ group.id, group.cases.items.len });
    const w = group.cases.items[0].workload;
    if (setup.* != null and !setup.*.?.compatible(w)) {
        if (prepared.*) |*p| p.deinit();
        prepared.* = null;
        setup.*.?.deinit();
        setup.* = null;
    }
    var setup_compile_ns: i64 = 0;
    if (setup.* == null) {
        log.info("compiling expert setup and reference", .{});
        setup.* = runner.compileSetup(allocator, io, platform, experts_mesh, w, args.check) catch |err| {
            if (err == error.Canceled) return error.Canceled;
            return reportCompilationFailure(group, err, stdout, json);
        };
        setup_compile_ns = setup.*.?.compile_ns;
    }
    const compiled = runner.compile(allocator, io, platform, experts_mesh, w, &setup.*.?) catch |err| {
        if (err == error.Canceled) return error.Canceled;
        return reportCompilationFailure(group, err, stdout, json);
    };
    defer compiled.deinit();
    var failures: usize = 0;
    for (group.cases.items, 0..) |selected, index| {
        var record = group.record(selected, .measured);
        if (index == 0) record.compile_ns = compiled.compile_ns + setup_compile_ns;
        // Weights stay on device across routing variants with the same seed.
        if (prepared.* != null and prepared.*.?.seed != selected.workload.seed) {
            prepared.*.?.deinit();
            prepared.* = null;
        }
        if (prepared.* == null) {
            prepared.* = runner.Prepared.init(allocator, io, platform, &setup.*.?, selected.workload) catch |err| {
                if (err == error.Canceled) return error.Canceled;
                record.status = .execution_failed;
                record.reason = @errorName(err);
                try results.report(stdout, json, record);
                failures += 1;
                continue;
            };
        }
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();
        runner.runCase(arena.allocator(), io, platform, args, &compiled, &setup.*.?, &prepared.*.?, &record) catch |err| {
            if (err == error.Canceled) return error.Canceled;
            record.status = .execution_failed;
            record.reason = @errorName(err);
        };
        failures += @intFromBool(record.status.failed());
        try results.report(stdout, json, record);
    }
    return failures;
}

fn reportCompilationFailure(group: Group, err: anyerror, stdout: *std.Io.Writer, json: ?*std.Io.Writer) !usize {
    for (group.cases.items) |selected| {
        var record = group.record(selected, .compilation_failed);
        record.reason = @errorName(err);
        try results.report(stdout, json, record);
    }
    return group.cases.items.len;
}

test {
    _ = zml.bazel;
    _ = cases;
    _ = @import("workload.zig");
    _ = runner;
    _ = results;
}
