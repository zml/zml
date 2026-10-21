//! CuTe implementation settings over the shared MoE workload benchmark.
const std = @import("std");
const zml = @import("zml");
const bench = @import("moe_bench");
const cute = zml.moe.cute;
const log = std.log.scoped(.moe_tune);

const Args = struct {
    pub const help =
        \\CuTe tuning: kernel settings are explicit compilation arguments.
        \\  --output=<new directory> --tiles=8,16,32,64,128
        \\  --quantize-lanes=1,4 --persistent-ctas=128,148,152
        \\  --configuration=tile-64-single
        \\
        \\Lists vary one setting at a time against the production default.
        \\Wide tiles also run with CTA pairing disabled. Use 'none' to omit an axis.
        \\--configuration selects one candidate; the default baseline always runs first.
        \\The tuner owns backend and output/validation options; --check is optional.
        \\--list, --json, --dump*, and --compare* are not accepted by the tuner.
        \\Workload options are the same as the main benchmark:
        \\
    ;
    output: ?[]const u8 = null,
    tiles: []const u8 = "8,16,32,64,128",
    quantize_lanes: []const u8 = "1,4",
    persistent_ctas: []const u8 = "none",
    configuration: ?[]const u8 = null,

    fn validate(self: Args, workload: bench.cases.Args) !void {
        if (self.output == null or self.output.?.len == 0) return error.MissingOutputDirectory;
        if (workload.backend) |backend| if (!std.mem.eql(u8, backend, "cute")) return error.CuteBackendRequired;
        if (workload.platform) |platform| if (platform != .cuda) return error.CudaPlatformRequired;
        if (workload.list or workload.json != null or workload.dump != null or workload.dump_dir != null or
            workload.compare != null or workload.compare_dir != null or workload.compare_exact)
            return error.TunerOwnsOutputOptions;
        try workload.validate();
    }
};

// Compose the CLI fields rather than maintain a second copy of workload axes.
const CliArgs = blk: {
    const w = @typeInfo(bench.cases.Args).@"struct";
    const t = @typeInfo(Args).@"struct";
    break :blk @Struct(.auto, null, w.field_names ++ t.field_names, w.field_types ++ t.field_types, w.field_attrs ++ t.field_attrs);
};

fn fields(comptime T: type, args: CliArgs) T {
    var result: T = undefined;
    inline for (@typeInfo(T).@"struct".field_names) |name| @field(result, name) = @field(args, name);
    return result;
}

const Setting = struct {
    name: []const u8,
    // Null measures the production public API, not an explicitly tuned copy.
    tuning: ?cute.kernels.Tuning,
};

fn append(allocator: std.mem.Allocator, list: *std.ArrayList(Setting), setting: Setting) !void {
    for (list.items) |existing| if (std.mem.eql(u8, existing.name, setting.name)) return;
    try list.append(allocator, setting);
}

/// The caller supplies an arena for names and selection storage.
fn settings(allocator: std.mem.Allocator, args: Args) ![]const Setting {
    var result: std.ArrayList(Setting) = .empty;
    try result.append(allocator, .{ .name = "default", .tuning = null });
    inline for (.{ "tiles", "quantize_lanes", "persistent_ctas" }) |axis| {
        const values = @field(args, axis);
        var it = std.mem.splitScalar(u8, values, ',');
        while (it.next()) |text| {
            if (values.len == 0 or std.mem.eql(u8, values, "none")) break;
            const value = try std.fmt.parseInt(i64, text, 10);
            if (comptime std.mem.eql(u8, axis, "tiles")) {
                switch (value) {
                    8, 16, 32, 64, 128 => {},
                    else => return error.InvalidCuteTileN,
                }
                try append(allocator, &result, .{ .name = try std.fmt.allocPrint(allocator, "tile-{d}-auto", .{value}), .tuning = .{ .tile_n = value } });
                if (value >= 64) try append(allocator, &result, .{ .name = try std.fmt.allocPrint(allocator, "tile-{d}-single", .{value}), .tuning = .{ .tile_n = value, .two_cta = false } });
            } else if (comptime std.mem.eql(u8, axis, "quantize_lanes")) {
                if (value != 1 and value != 4) return error.InvalidCuteQuantizeLanes;
                try append(allocator, &result, .{ .name = try std.fmt.allocPrint(allocator, "quantize-{d}", .{value}), .tuning = .{ .quantize_lanes = value } });
            } else {
                if (value <= 0) return error.InvalidCutePersistentCtas;
                try append(allocator, &result, .{ .name = try std.fmt.allocPrint(allocator, "ctas-{d}", .{value}), .tuning = .{ .persistent_ctas = value } });
            }
        }
    }
    if (args.configuration) |name| {
        for (result.items) |setting| if (std.mem.eql(u8, name, setting.name)) {
            if (setting.tuning == null) return result.items[0..1];
            result.items[1] = setting;
            return result.items[0..2];
        };
        return error.UnknownConfiguration;
    }
    return result.toOwnedSlice(allocator);
}

const Compiler = struct {
    tuning: ?cute.kernels.Tuning,

    pub fn compile(self: Compiler, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mesh: *const zml.Mesh, w: bench.cases.Workload, setup: *const bench.runner.Setup) !bench.runner.Compiled {
        const tuning = self.tuning orelse return bench.runner.compile(allocator, io, platform, mesh, w, setup);
        std.debug.assert(w.case.backend == .cute and setup.compatible(w));
        const start: std.Io.Timestamp = .now(io, .awake);
        const c = w.case;
        const plan = try tuning.resolveFor(platform, c.tokens, c.width, c.intermediate, w.global_experts, c.topk);
        const timed = try platform.compileFn(allocator, io, cute.fusedExpertsWithPlan, .{
            zml.Tensor.init(.{ .t = c.tokens, .d = c.width }, c.activation_dtype),
            zml.Tensor.init(.{ .t = c.tokens, .topk = c.topk }, .i32),
            zml.Tensor.init(.{ .t = c.tokens, .topk = c.topk }, .f32),
            setup.template.prepared.gate_up,
            setup.template.prepared.down,
            c.options(),
            plan,
        }, .{ .meshes = &.{mesh} });
        return .{ .timed = timed, .compile_ns = @intCast(start.untilNow(io, .awake).toNanoseconds()), .cute_launch = plan };
    }
};

pub fn main(init: std.process.Init) !void {
    var arguments = init.minimal.args.iterate();
    while (arguments.next()) |arg| if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
        std.debug.print(Args.help ++ bench.cases.Args.help, .{});
        return;
    };
    const cli = zml.stdx.flags.parse(init.minimal.args, CliArgs);
    const args = fields(Args, cli);
    var workload = fields(bench.cases.Args, cli);
    args.validate(workload) catch |err| zml.stdx.flags.fatal("invalid tuning arguments: {s}; use --help", .{@errorName(err)});
    workload.backend = "cute";
    var arena: std.heap.ArenaAllocator = .init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();
    const selected = settings(allocator, args) catch |err| zml.stdx.flags.fatal("invalid tuning selection: {s}", .{@errorName(err)});
    const platform = try zml.Platform.init(init.gpa, init.io, .cuda, .{});
    defer platform.deinit(init.gpa, init.io);
    if (!cute.isAvailable(platform)) return error.CuteUnavailable;
    const mesh = try platform.registerMesh("experts", .mesh(.{ .experts = .high_bandwidth }));
    try std.Io.Dir.createDir(.cwd(), init.io, args.output.?, .default_dir);
    const baseline = try std.fmt.allocPrint(allocator, "{s}/baseline", .{args.output.?});
    var failures: usize = 0;
    for (selected) |setting| {
        var run_args = workload;
        run_args.json = try std.fmt.allocPrint(allocator, "{s}/{s}.jsonl", .{ args.output.?, setting.name });
        if (setting.tuning == null) {
            run_args.dump_dir = baseline;
        } else {
            run_args.compare_dir = baseline;
            run_args.compare_exact = true;
        }
        log.info("running {s}", .{setting.name});
        bench.suite.run(init, platform, mesh, run_args, setting.name, Compiler{ .tuning = setting.tuning }) catch |err| {
            if (err == error.Canceled) return err;
            log.err("{s}: {s}", .{ setting.name, @errorName(err) });
            // No candidate can be validated without a complete baseline.
            if (setting.tuning == null) return err;
            failures += 1;
        };
    }
    log.info("{d} failed configurations; results: {s}", .{ failures, args.output.? });
    if (failures != 0) return error.TuningFailed;
}

test {
    _ = zml.bazel;
}

test "tuning selection preserves baseline, deduplicates and selects explicit settings" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const all = try settings(allocator, .{ .persistent_ctas = "128,148,152" });
    try std.testing.expectEqual(@as(usize, 13), all.len);
    try std.testing.expect(all[0].tuning == null);
    const filtered = try settings(allocator, .{ .configuration = "tile-64-single" });
    try std.testing.expectEqual(@as(usize, 2), filtered.len);
    try std.testing.expectEqual(@as(i64, 64), filtered[1].tuning.?.tile_n.?);
    try std.testing.expect(!filtered[1].tuning.?.two_cta.?);
    const baseline_only = try settings(allocator, .{ .tiles = "none", .quantize_lanes = "none", .persistent_ctas = "none" });
    try std.testing.expectEqual(@as(usize, 1), baseline_only.len);
    const repeated = try settings(allocator, .{ .tiles = "8,8", .quantize_lanes = "", .persistent_ctas = "" });
    try std.testing.expectEqual(@as(usize, 2), repeated.len);
    try std.testing.expectError(error.UnknownConfiguration, settings(allocator, .{ .configuration = "typo" }));
    try std.testing.expectError(error.InvalidCuteTileN, settings(allocator, .{ .tiles = "7" }));
    try std.testing.expectError(error.InvalidCuteQuantizeLanes, settings(allocator, .{ .quantize_lanes = "2" }));
    try std.testing.expectError(error.InvalidCutePersistentCtas, settings(allocator, .{ .persistent_ctas = "0" }));
}

test "tuner owns output and backend options without duplicating workload fields" {
    const args: Args = .{ .output = "results" };
    try args.validate(.{ .check = true });
    try std.testing.expectError(error.MissingOutputDirectory, (Args{}).validate(.{}));
    try std.testing.expectError(error.CuteBackendRequired, args.validate(.{ .backend = "triton" }));
    try std.testing.expectError(error.TunerOwnsOutputOptions, args.validate(.{ .list = true }));
    try std.testing.expectError(error.TunerOwnsOutputOptions, args.validate(.{ .compare_exact = true }));
    try std.testing.expectError(error.TunerOwnsOutputOptions, args.validate(.{ .json = "owned.jsonl" }));
    const cli: CliArgs = .{ .tokens = "1,17", .input_scale = "16", .tiles = "64", .output = "results" };
    try std.testing.expectEqualStrings("1,17", fields(bench.cases.Args, cli).tokens.?);
    try std.testing.expectEqualStrings("16", fields(bench.cases.Args, cli).input_scale);
    try std.testing.expectEqualStrings("64", fields(Args, cli).tiles);
}
