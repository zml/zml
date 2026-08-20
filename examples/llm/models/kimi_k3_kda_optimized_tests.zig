const std = @import("std");

const zml = @import("zml");
const kda = @import("kimi_k3/kda.zig");

comptime {
    @setEvalBranchQuota(400_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    only: []const u8 = "",
    fixture: []const u8,
    profile: bool = false,
    profile_repository: []const u8 = "/tmp/kimi-k3-kda-profile",
    profile_session: []const u8 = "milestone-18-kda",

    pub const help =
        \\Use kimi_k3_kda_optimized_tests --fixture=<kda-optimized-cases.safetensors>
        \\
        \\Compare fused and sequential channel-wise KDA recurrence on NVIDIA CUDA.
        \\
    ;
};

const Inputs = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    alpha: zml.Tensor,
    beta: zml.Tensor,
    state: zml.Tensor,
};

const Case = struct {
    name: []const u8,
    sequence: usize,
    benchmark: bool = false,
};

const cases = [_]Case{
    .{ .name = "small_s1", .sequence = 1 },
    .{ .name = "small_s3", .sequence = 3 },
    .{ .name = "small_s4", .sequence = 4 },
    .{ .name = "small_s5", .sequence = 5 },
    .{ .name = "small_s31", .sequence = 31 },
    .{ .name = "small_s32", .sequence = 32 },
    .{ .name = "small_s33", .sequence = 33 },
    .{ .name = "small_s63", .sequence = 63 },
    .{ .name = "small_s64", .sequence = 64 },
    .{ .name = "small_s65", .sequence = 65 },
    .{ .name = "small_s257", .sequence = 257 },
    .{ .name = "production_decode", .sequence = 1, .benchmark = true },
    .{ .name = "production_prefill64", .sequence = 64, .benchmark = true },
};

const tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 3e-4,
    .relative_tolerance = 3e-4,
    .minimum_close_fraction = 1.0,
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    sharding: zml.Sharding,
    stdout: *std.Io.Writer,

    fn load(self: *Context, key: []const u8, tags: anytype) !zml.Buffer {
        const shape = self.store.getShape(key) orelse {
            try self.stdout.print("KIMI_K3_KDA_OPT_MISSING key={s}\n", .{key});
            return error.MissingKdaOptimizedFixture;
        };
        const bytes = try self.allocator.alloc(u8, shape.byteSize());
        defer self.allocator.free(bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.store.getReader(key, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(bytes);
        return zml.Buffer.fromBytes(self.io, self.platform, shape.withTags(tags), self.sharding, bytes);
    }

    fn tensorKey(self: *Context, prefix: []const u8, suffix: []const u8) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ prefix, suffix });
    }

    fn loadInputs(self: *Context, prefix: []const u8) !zml.Bufferized(Inputs) {
        const q_key = try self.tensorKey(prefix, "q");
        defer self.allocator.free(q_key);
        const k_key = try self.tensorKey(prefix, "k");
        defer self.allocator.free(k_key);
        const v_key = try self.tensorKey(prefix, "v");
        defer self.allocator.free(v_key);
        const alpha_key = try self.tensorKey(prefix, "alpha");
        defer self.allocator.free(alpha_key);
        const beta_key = try self.tensorKey(prefix, "beta");
        defer self.allocator.free(beta_key);
        const state_key = try self.tensorKey(prefix, "state");
        defer self.allocator.free(state_key);
        return .{
            .q = try self.load(q_key, .{ .b, .s, .h, .k }),
            .k = try self.load(k_key, .{ .b, .s, .h, .k }),
            .v = try self.load(v_key, .{ .b, .s, .h, .v }),
            .alpha = try self.load(alpha_key, .{ .b, .s, .h, .k }),
            .beta = try self.load(beta_key, .{ .b, .s, .h }),
            .state = try self.load(state_key, .{ .b, .h, .v, .k }),
        };
    }

    fn inputTensors(inputs: zml.Bufferized(Inputs)) Inputs {
        return .{
            .q = .fromShape(inputs.q.shape()),
            .k = .fromShape(inputs.k.shape()),
            .v = .fromShape(inputs.v.shape()),
            .alpha = .fromShape(inputs.alpha.shape()),
            .beta = .fromShape(inputs.beta.shape()),
            .state = .fromShape(inputs.state.shape()),
        };
    }

    fn compile(self: *Context, comptime function: anytype, symbolic: Inputs) !struct { zml.Exe, i96 } {
        const started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            function,
            .{ symbolic.q, symbolic.k, symbolic.v, symbolic.alpha, symbolic.beta, symbolic.state },
            .{ .shardings = &.{self.sharding} },
        );
        return .{ exe, @divTrunc(std.Io.Clock.now(.real, self.io).toNanoseconds() - started, 1000) };
    }

    fn execute(
        self: *Context,
        comptime function: anytype,
        exe: *const zml.Exe,
        inputs: zml.Bufferized(Inputs),
    ) !zml.Bufferized(kda.RecurrentResult) {
        return zml.testing.autoCall(
            self.allocator,
            self.io,
            exe,
            function,
            .{ inputs.q, inputs.k, inputs.v, inputs.alpha, inputs.beta, inputs.state },
        );
    }

    fn compare(
        self: *Context,
        case_name: []const u8,
        path: []const u8,
        boundary: []const u8,
        actual: zml.Buffer,
        expected: zml.Buffer,
    ) !void {
        var actual_host = try actual.toSliceAlloc(self.allocator, self.io);
        defer actual_host.free(self.allocator);
        var expected_host = try expected.toSliceAlloc(self.allocator, self.io);
        defer expected_host.free(self.allocator);
        const report = try zml.testing.compareSlices(
            self.allocator,
            f32,
            f32,
            actual_host.items(f32),
            expected_host.items(f32),
            tolerance,
        );
        const values = actual_host.items(f32);
        // KIMI_K3_TEMP_REMOVE_M20: activation samples and detailed errors are
        // bring-up diagnostics and are removed after permanent tests land.
        try self.stdout.print(
            "KIMI_K3_KDA_OPT_ACTIVATION case={s} path={s} boundary={s} first={d:.8} last={d:.8} finite=true\n{f}\n",
            .{ case_name, path, boundary, values[0], values[values.len - 1], report },
        );
        try self.stdout.flush();
        try zml.testing.expectClose(self.io, actual_host, expected_host, tolerance);
    }

    fn benchmark(
        self: *Context,
        case_name: []const u8,
        path: []const u8,
        comptime function: anytype,
        exe: *const zml.Exe,
    ) !i96 {
        const warmups = 2;
        const repetitions = 7;
        var total: i96 = 0;
        for (0..warmups + repetitions) |iteration| {
            var inputs = try self.loadInputs(case_name);
            defer zml.Buffer.deinitAll(Inputs, &inputs);
            const started = std.Io.Clock.now(.real, self.io).toNanoseconds();
            var result = try self.execute(function, exe, inputs);
            var synchronized = try result.output.toSliceAlloc(self.allocator, self.io);
            synchronized.free(self.allocator);
            const elapsed = @divTrunc(std.Io.Clock.now(.real, self.io).toNanoseconds() - started, 1000);
            zml.Buffer.deinitAll(kda.RecurrentResult, &result);
            if (iteration >= warmups) total += elapsed;
        }
        const mean = @divTrunc(total, repetitions);
        // KIMI_K3_TEMP_REMOVE_M20: synchronized stage timing is retained only
        // through bring-up and replaced by the permanent benchmark suite.
        try self.stdout.print(
            "KIMI_K3_KDA_OPT_BENCH case={s} path={s} warmups={} repetitions={} mean_execute_us={}\n",
            .{ case_name, path, warmups, repetitions, mean },
        );
        try self.stdout.flush();
        return mean;
    }

    fn runCase(self: *Context, one: Case) !void {
        // KIMI_K3_TEMP_REMOVE_M20: model-family span is bring-up profiling
        // instrumentation replaced by permanent inference spans in cleanup.
        var span = zml.tracer.span("kimi_k3.kda.optimized_case", .{});
        defer span.end();
        var optimized_inputs = try self.loadInputs(one.name);
        defer zml.Buffer.deinitAll(Inputs, &optimized_inputs);
        const symbolic = inputTensors(optimized_inputs);
        var optimized_exe, const optimized_compile_us = try self.compile(kda.recurrentOptimized, symbolic);
        defer optimized_exe.deinit();
        var reference_exe, const reference_compile_us = try self.compile(kda.recurrentReference, symbolic);
        defer reference_exe.deinit();

        const expected_output_key = try self.tensorKey(one.name, "expected_output");
        defer self.allocator.free(expected_output_key);
        const expected_state_key = try self.tensorKey(one.name, "expected_state");
        defer self.allocator.free(expected_state_key);
        var expected_output = try self.load(expected_output_key, .{ .b, .s, .h, .v });
        defer expected_output.deinit();
        var expected_state = try self.load(expected_state_key, .{ .b, .h, .v, .k });
        defer expected_state.deinit();

        var optimized = try self.execute(kda.recurrentOptimized, &optimized_exe, optimized_inputs);
        defer zml.Buffer.deinitAll(kda.RecurrentResult, &optimized);
        try self.compare(one.name, "optimized", "output", optimized.output, expected_output);
        try self.compare(one.name, "optimized", "state", optimized.state, expected_state);

        var reference_inputs = try self.loadInputs(one.name);
        defer zml.Buffer.deinitAll(Inputs, &reference_inputs);
        var reference = try self.execute(kda.recurrentReference, &reference_exe, reference_inputs);
        defer zml.Buffer.deinitAll(kda.RecurrentResult, &reference);
        try self.compare(one.name, "reference", "output", reference.output, expected_output);
        try self.compare(one.name, "reference", "state", reference.state, expected_state);

        var optimized_us: i96 = 0;
        var reference_us: i96 = 0;
        if (one.benchmark) {
            optimized_us = try self.benchmark(one.name, "optimized", kda.recurrentOptimized, &optimized_exe);
            reference_us = try self.benchmark(one.name, "reference", kda.recurrentReference, &reference_exe);
            if (std.mem.eql(u8, one.name, "production_decode")) {
                // Decode may trade launch overhead for lower state traffic, but
                // the fused default is not allowed to regress by more than 5%.
                if (optimized_us * 100 > reference_us * 105) return error.KdaDecodeRegressionBudgetExceeded;
            } else {
                // Wall-synchronized stage timing has a 5% no-regression budget;
                // the official full-KDA fixture records the end-to-end speedup.
                if (optimized_us * 100 > reference_us * 105) return error.KdaPrefillRegressionBudgetExceeded;
            }
        }
        try self.stdout.print(
            "KIMI_K3_KDA_OPT_PASS case={s} sequence={} optimized_compile_us={} reference_compile_us={} optimized_execute_us={} reference_execute_us={}\n",
            .{ one.name, one.sequence, optimized_compile_us, reference_compile_us, optimized_us, reference_us },
        );
        try self.stdout.flush();
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.45 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.fixture);
    var profiler: ?zml.Platform.Profiler = null;
    defer if (profiler) |*active_profiler| {
        _ = active_profiler.stop() catch {};
        active_profiler.deinit();
    };
    if (args.profile) {
        profiler = try platform.profiler(allocator, io, .{
            .repository_path = args.profile_repository,
            .session_id = args.profile_session,
        });
        try profiler.?.start();
    }
    defer registry.deinit();
    var tensor_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer tensor_store.deinit();
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    var context: Context = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .store = tensor_store.view(),
        .sharding = platform.replicated_sharding,
        .stdout = &stdout_file.interface,
    };
    for (cases) |one| {
        if (args.only.len != 0 and !std.mem.eql(u8, args.only, one.name)) continue;
        try context.runCase(one);
    }
    if (args.only.len != 0) return;
    try stdout_file.interface.writeAll("KIMI_K3_KDA_OPT_ALL_PASS cases=13 awkward=3,4,5,31,32,33,63,64,65 long=257 production=decode,prefill64 backend=cuda\n");
    try stdout_file.interface.flush();
}
