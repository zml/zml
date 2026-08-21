const std = @import("std");

const zml = @import("zml");
const kda = @import("kimi_k3/kda.zig");

comptime {
    @setEvalBranchQuota(200_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_kda_tests --fixture=<kda-decode-reference.safetensors>
        \\
        \\Run four-step Kimi K3 KDA decode differential tests on NVIDIA CUDA only.
        \\
    ;
};

const core_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 5e-4,
    .relative_tolerance = 5e-4,
    .minimum_close_fraction = 1.0,
};

const output_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 5e-3,
    .relative_tolerance = 5e-3,
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
            try self.stdout.print("KIMI_K3_KDA_MISSING key={s}\n", .{key});
            return error.MissingKdaFixture;
        };
        const bytes = try self.allocator.alloc(u8, shape.byteSize());
        defer self.allocator.free(bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.store.getReader(key, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(bytes);
        return zml.Buffer.fromBytes(
            self.io,
            self.platform,
            shape.withTags(tags),
            self.sharding,
            bytes,
        );
    }

    fn loadWeights(self: *Context) !zml.Bufferized(kda.Weights) {
        return .{
            .q_weight = try self.load("weights.q_weight", .{ .out, .d }),
            .k_weight = try self.load("weights.k_weight", .{ .out, .d }),
            .v_weight = try self.load("weights.v_weight", .{ .out, .d }),
            .q_conv_weight = try self.load("weights.q_conv_weight", .{ .channel, .kernel }),
            .k_conv_weight = try self.load("weights.k_conv_weight", .{ .channel, .kernel }),
            .v_conv_weight = try self.load("weights.v_conv_weight", .{ .channel, .kernel }),
            .decay_a_weight = try self.load("weights.decay_a_weight", .{ .out, .d }),
            .decay_b_weight = try self.load("weights.decay_b_weight", .{ .channel, .rank }),
            .a_log = try self.load("weights.a_log", .{.h}),
            .dt_bias = try self.load("weights.dt_bias", .{ .h, .k }),
            .beta_weight = try self.load("weights.beta_weight", .{ .out, .d }),
            .gate_weight = try self.load("weights.gate_weight", .{ .out, .d }),
            .norm_weight = try self.load("weights.norm_weight", .{.v}),
            .output_weight = try self.load("weights.output_weight", .{ .d, .out }),
        };
    }

    fn loadInitialCache(self: *Context) !zml.Bufferized(kda.Cache) {
        return .{
            .q_conv = try self.load("inputs.initial_q_cache", .{ .b, .channel, .kernel }),
            .k_conv = try self.load("inputs.initial_k_cache", .{ .b, .channel, .kernel }),
            .v_conv = try self.load("inputs.initial_v_cache", .{ .b, .channel, .kernel }),
            .recurrent_state = try self.load("inputs.initial_recurrent_state", .{ .b, .h, .v, .k }),
        };
    }

    fn expectedKey(self: *Context, step: usize, name: []const u8) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "step.{}.expected.{s}", .{ step, name });
    }

    fn compare(
        self: *Context,
        step: usize,
        name: []const u8,
        actual: zml.Buffer,
        opts: zml.testing.CompareOpts,
    ) !void {
        const key = try self.expectedKey(step, name);
        defer self.allocator.free(key);
        const expected_shape = self.store.getShape(key) orelse return error.MissingKdaExpectedBoundary;
        var expected = try self.load(key, expected_shape.tags());
        defer expected.deinit();
        try zml.testing.expectClose(self.io, actual, expected, opts);
    }

    fn compareStep(self: *Context, step: usize, actual: zml.Bufferized(kda.DecodeResult)) !void {
        try self.compare(step, "q_proj", actual.q_proj, core_tolerance);
        try self.compare(step, "k_proj", actual.k_proj, core_tolerance);
        try self.compare(step, "v_proj", actual.v_proj, core_tolerance);
        try self.compare(step, "q_conv", actual.q_conv, core_tolerance);
        try self.compare(step, "k_conv", actual.k_conv, core_tolerance);
        try self.compare(step, "v_conv", actual.v_conv, core_tolerance);
        try self.compare(step, "q", actual.q, core_tolerance);
        try self.compare(step, "k", actual.k, core_tolerance);
        try self.compare(step, "v", actual.v, core_tolerance);
        try self.compare(step, "q_norm", actual.q_norm, core_tolerance);
        try self.compare(step, "k_norm", actual.k_norm, core_tolerance);
        try self.compare(step, "raw_decay", actual.raw_decay, core_tolerance);
        try self.compare(step, "log_alpha", actual.log_alpha, core_tolerance);
        try self.compare(step, "alpha", actual.alpha, core_tolerance);
        try self.compare(step, "raw_beta", actual.raw_beta, core_tolerance);
        try self.compare(step, "beta", actual.beta, core_tolerance);
        try self.compare(step, "prediction", actual.prediction, core_tolerance);
        try self.compare(step, "error", actual.error_value, core_tolerance);
        try self.compare(step, "recurrent_state", actual.cache.recurrent_state, core_tolerance);
        try self.compare(step, "recurrent_output", actual.recurrent_output, core_tolerance);
        try self.compare(step, "output_gate", actual.output_gate, core_tolerance);
        try self.compare(step, "norm_gated", actual.norm_gated, output_tolerance);
        try self.compare(step, "projection_output", actual.projection_output, output_tolerance);
        try self.compare(step, "q_cache", actual.cache.q_conv, core_tolerance);
        try self.compare(step, "k_cache", actual.cache.k_conv, core_tolerance);
        try self.compare(step, "v_cache", actual.cache.v_conv, core_tolerance);
    }
};

fn deinitDiagnostics(actual: *zml.Bufferized(kda.DecodeResult)) void {
    actual.q_proj.deinit();
    actual.k_proj.deinit();
    actual.v_proj.deinit();
    actual.q_conv.deinit();
    actual.k_conv.deinit();
    actual.v_conv.deinit();
    actual.q.deinit();
    actual.k.deinit();
    actual.v.deinit();
    actual.q_norm.deinit();
    actual.k_norm.deinit();
    actual.raw_decay.deinit();
    actual.log_alpha.deinit();
    actual.alpha.deinit();
    actual.raw_beta.deinit();
    actual.beta.deinit();
    actual.prediction.deinit();
    actual.error_value.deinit();
    actual.recurrent_output.deinit();
    actual.output_gate.deinit();
    actual.norm_gated.deinit();
    actual.projection_output.deinit();
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.25 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.fixture);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    var context: Context = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .store = store.view(),
        .sharding = platform.replicated_sharding,
        .stdout = &stdout_file.interface,
    };
    var weights = try context.loadWeights();
    defer zml.Buffer.deinitAll(kda.Weights, &weights);
    var cache = try context.loadInitialCache();
    defer zml.Buffer.deinitAll(kda.Cache, &cache);

    const hidden_shape = context.store.getShape("step.0.input.hidden") orelse return error.MissingKdaInput;
    const hidden = zml.Tensor.fromShape(hidden_shape).withTags(.{ .b, .d });
    const weight_tensors: kda.Weights = .{
        .q_weight = .fromShape(weights.q_weight.shape()),
        .k_weight = .fromShape(weights.k_weight.shape()),
        .v_weight = .fromShape(weights.v_weight.shape()),
        .q_conv_weight = .fromShape(weights.q_conv_weight.shape()),
        .k_conv_weight = .fromShape(weights.k_conv_weight.shape()),
        .v_conv_weight = .fromShape(weights.v_conv_weight.shape()),
        .decay_a_weight = .fromShape(weights.decay_a_weight.shape()),
        .decay_b_weight = .fromShape(weights.decay_b_weight.shape()),
        .a_log = .fromShape(weights.a_log.shape()),
        .dt_bias = .fromShape(weights.dt_bias.shape()),
        .beta_weight = .fromShape(weights.beta_weight.shape()),
        .gate_weight = .fromShape(weights.gate_weight.shape()),
        .norm_weight = .fromShape(weights.norm_weight.shape()),
        .output_weight = .fromShape(weights.output_weight.shape()),
    };
    const cache_tensors: kda.Cache = .{
        .q_conv = .fromShape(cache.q_conv.shape()),
        .k_conv = .fromShape(cache.k_conv.shape()),
        .v_conv = .fromShape(cache.v_conv.shape()),
        .recurrent_state = .fromShape(cache.recurrent_state.shape()),
    };
    const exe = try platform.compileFn(
        allocator,
        io,
        kda.decode,
        .{ hidden, weight_tensors, cache_tensors },
        .{ .shardings = &.{context.sharding} },
    );
    defer exe.deinit();

    for (0..4) |step| {
        const input_key = try std.fmt.allocPrint(allocator, "step.{}.input.hidden", .{step});
        defer allocator.free(input_key);
        var hidden_buffer = try context.load(input_key, .{ .b, .d });
        defer hidden_buffer.deinit();
        const started = std.Io.Clock.now(.real, io).toNanoseconds();
        var actual = try zml.testing.autoCall(
            allocator,
            io,
            &exe,
            kda.decode,
            .{ hidden_buffer, weights, cache },
        );
        const elapsed_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - started, 1000);
        try context.compareStep(step, actual);

        zml.Buffer.deinitAll(kda.Cache, &cache);
        cache = actual.cache;
        deinitDiagnostics(&actual);
        try stdout_file.interface.print(
            "KIMI_K3_KDA_STEP_PASS step={} boundaries=26 elapsed_us={} output={f} recurrent_cache={f} conv_cache={f}\n",
            .{ step, elapsed_us, actual.projection_output.shape(), cache.recurrent_state.shape(), cache.q_conv.shape() },
        );
        try stdout_file.interface.flush();
    }
    try stdout_file.interface.writeAll("KIMI_K3_KDA_ALL_PASS steps=4 boundaries_per_step=26 backend=cuda\n");
    try stdout_file.interface.flush();
}
