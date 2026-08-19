const std = @import("std");

const zml = @import("zml");
const kda = @import("kimi_k3/kda.zig");

comptime {
    @setEvalBranchQuota(300_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_kda_prefill_tests --fixture=<kda-prefill-reference.safetensors>
        \\
        \\Run Kimi K3 sequential prefill/cache differential tests on NVIDIA CUDA only.
        \\
    ;
};

const cache_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 8e-4,
    .relative_tolerance = 8e-4,
    .minimum_close_fraction = 1.0,
};

const output_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 8e-3,
    .relative_tolerance = 8e-3,
    .minimum_close_fraction = 1.0,
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    sharding: zml.Sharding,
    stdout: *std.Io.Writer,
    weights: zml.Bufferized(kda.Weights),

    fn load(self: *Context, key: []const u8, tags: anytype) !zml.Buffer {
        const shape = self.store.getShape(key) orelse {
            try self.stdout.print("KIMI_K3_KDA_PREFILL_MISSING key={s}\n", .{key});
            return error.MissingKdaPrefillFixture;
        };
        const bytes = try self.allocator.alloc(u8, shape.byteSize());
        defer self.allocator.free(bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.store.getReader(key, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(bytes);
        return zml.Buffer.fromBytes(self.io, self.platform, shape.withTags(tags), self.sharding, bytes);
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

    fn weightsTensors(self: *Context) kda.Weights {
        return .{
            .q_weight = .fromShape(self.weights.q_weight.shape()),
            .k_weight = .fromShape(self.weights.k_weight.shape()),
            .v_weight = .fromShape(self.weights.v_weight.shape()),
            .q_conv_weight = .fromShape(self.weights.q_conv_weight.shape()),
            .k_conv_weight = .fromShape(self.weights.k_conv_weight.shape()),
            .v_conv_weight = .fromShape(self.weights.v_conv_weight.shape()),
            .decay_a_weight = .fromShape(self.weights.decay_a_weight.shape()),
            .decay_b_weight = .fromShape(self.weights.decay_b_weight.shape()),
            .a_log = .fromShape(self.weights.a_log.shape()),
            .dt_bias = .fromShape(self.weights.dt_bias.shape()),
            .beta_weight = .fromShape(self.weights.beta_weight.shape()),
            .gate_weight = .fromShape(self.weights.gate_weight.shape()),
            .norm_weight = .fromShape(self.weights.norm_weight.shape()),
            .output_weight = .fromShape(self.weights.output_weight.shape()),
        };
    }

    fn loadCache(self: *Context, prefix: []const u8) !zml.Bufferized(kda.Cache) {
        const q_key = try std.fmt.allocPrint(self.allocator, "{s}.q_cache", .{prefix});
        defer self.allocator.free(q_key);
        const k_key = try std.fmt.allocPrint(self.allocator, "{s}.k_cache", .{prefix});
        defer self.allocator.free(k_key);
        const v_key = try std.fmt.allocPrint(self.allocator, "{s}.v_cache", .{prefix});
        defer self.allocator.free(v_key);
        const state_key = try std.fmt.allocPrint(self.allocator, "{s}.recurrent_state", .{prefix});
        defer self.allocator.free(state_key);
        return .{
            .q_conv = try self.load(q_key, .{ .b, .channel, .kernel }),
            .k_conv = try self.load(k_key, .{ .b, .channel, .kernel }),
            .v_conv = try self.load(v_key, .{ .b, .channel, .kernel }),
            .recurrent_state = try self.load(state_key, .{ .b, .h, .v, .k }),
        };
    }

    fn cacheTensors(cache: zml.Bufferized(kda.Cache)) kda.Cache {
        return .{
            .q_conv = .fromShape(cache.q_conv.shape()),
            .k_conv = .fromShape(cache.k_conv.shape()),
            .v_conv = .fromShape(cache.v_conv.shape()),
            .recurrent_state = .fromShape(cache.recurrent_state.shape()),
        };
    }

    fn compare(self: *Context, expected_key: []const u8, actual: zml.Buffer, opts: zml.testing.CompareOpts) !void {
        const shape = self.store.getShape(expected_key) orelse return error.MissingKdaPrefillExpected;
        var expected = try self.load(expected_key, shape.tags());
        defer expected.deinit();
        try zml.testing.expectClose(self.io, actual, expected, opts);
    }

    fn compareCache(self: *Context, prefix: []const u8, actual: zml.Bufferized(kda.Cache)) !void {
        const q_key = try std.fmt.allocPrint(self.allocator, "{s}.q_cache", .{prefix});
        defer self.allocator.free(q_key);
        const k_key = try std.fmt.allocPrint(self.allocator, "{s}.k_cache", .{prefix});
        defer self.allocator.free(k_key);
        const v_key = try std.fmt.allocPrint(self.allocator, "{s}.v_cache", .{prefix});
        defer self.allocator.free(v_key);
        const state_key = try std.fmt.allocPrint(self.allocator, "{s}.recurrent_state", .{prefix});
        defer self.allocator.free(state_key);
        try self.compare(q_key, actual.q_conv, cache_tolerance);
        try self.compare(k_key, actual.k_conv, cache_tolerance);
        try self.compare(v_key, actual.v_conv, cache_tolerance);
        try self.compare(state_key, actual.recurrent_state, cache_tolerance);
    }

    fn run(
        self: *Context,
        comptime function: anytype,
        input_key: []const u8,
        cache: zml.Bufferized(kda.Cache),
    ) !struct { zml.Bufferized(kda.CompactResult), i96 } {
        var input = try self.load(input_key, if (function == kda.prefill) .{ .b, .s, .d } else .{ .b, .d });
        defer input.deinit();
        const input_tensor = zml.Tensor.fromShape(input.shape());
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            function,
            .{ input_tensor, self.weightsTensors(), cacheTensors(cache) },
            .{ .shardings = &.{self.sharding} },
        );
        defer exe.deinit();
        const started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        const actual = try zml.testing.autoCall(
            self.allocator,
            self.io,
            &exe,
            function,
            .{ input, self.weights, cache },
        );
        return .{ actual, std.Io.Clock.now(.real, self.io).toNanoseconds() - started };
    }

    fn runFull(self: *Context, length: usize) !void {
        const input_key = try std.fmt.allocPrint(self.allocator, "len{}.input.hidden", .{length});
        defer self.allocator.free(input_key);
        const expected_output = try std.fmt.allocPrint(self.allocator, "len{}.expected.output", .{length});
        defer self.allocator.free(expected_output);
        const expected_cache = try std.fmt.allocPrint(self.allocator, "len{}.expected", .{length});
        defer self.allocator.free(expected_cache);
        var initial = try self.loadCache("inputs.initial");
        defer zml.Buffer.deinitAll(kda.Cache, &initial);
        var actual, const elapsed = try self.run(kda.prefill, input_key, initial);
        defer actual.output.deinit();
        defer zml.Buffer.deinitAll(kda.Cache, &actual.cache);
        try self.compare(expected_output, actual.output, output_tolerance);
        try self.compareCache(expected_cache, actual.cache);
        try self.stdout.print("KIMI_K3_KDA_PREFILL_PASS kind=full length={} elapsed_us={}\n", .{ length, @divTrunc(elapsed, 1000) });
    }

    fn runDecode(self: *Context, length: usize) !void {
        var cache = try self.loadCache("inputs.initial");
        defer zml.Buffer.deinitAll(kda.Cache, &cache);
        for (0..length) |token| {
            const input_key = try std.fmt.allocPrint(self.allocator, "len{}.token{}.input.hidden", .{ length, token });
            defer self.allocator.free(input_key);
            const output_key = try std.fmt.allocPrint(self.allocator, "len{}.token{}.expected.output", .{ length, token });
            defer self.allocator.free(output_key);
            const cache_prefix = try std.fmt.allocPrint(self.allocator, "len{}.token{}.expected", .{ length, token });
            defer self.allocator.free(cache_prefix);
            var actual, _ = try self.run(kda.decodeCompact, input_key, cache);
            try self.compare(output_key, actual.output, output_tolerance);
            try self.compareCache(cache_prefix, actual.cache);
            actual.output.deinit();
            zml.Buffer.deinitAll(kda.Cache, &cache);
            cache = actual.cache;
        }
        try self.stdout.print("KIMI_K3_KDA_PREFILL_PASS kind=token_decode length={} steps={}\n", .{ length, length });
    }

    fn runSplit(self: *Context, length: usize, split: usize) !void {
        var initial = try self.loadCache("inputs.initial");
        defer zml.Buffer.deinitAll(kda.Cache, &initial);
        const first_input = try std.fmt.allocPrint(self.allocator, "len{}.split{}.input.first", .{ length, split });
        defer self.allocator.free(first_input);
        const second_input = try std.fmt.allocPrint(self.allocator, "len{}.split{}.input.second", .{ length, split });
        defer self.allocator.free(second_input);
        const first_output = try std.fmt.allocPrint(self.allocator, "len{}.split{}.expected.first_output", .{ length, split });
        defer self.allocator.free(first_output);
        const second_output = try std.fmt.allocPrint(self.allocator, "len{}.split{}.expected.second_output", .{ length, split });
        defer self.allocator.free(second_output);
        const intermediate = try std.fmt.allocPrint(self.allocator, "len{}.split{}.expected.intermediate", .{ length, split });
        defer self.allocator.free(intermediate);
        const final_prefix = try std.fmt.allocPrint(self.allocator, "len{}.expected", .{length});
        defer self.allocator.free(final_prefix);
        var first, const first_elapsed = try self.run(kda.prefill, first_input, initial);
        try self.compare(first_output, first.output, output_tolerance);
        try self.compareCache(intermediate, first.cache);
        first.output.deinit();
        var second, const second_elapsed = try self.run(kda.prefill, second_input, first.cache);
        try self.compare(second_output, second.output, output_tolerance);
        try self.compareCache(final_prefix, second.cache);
        second.output.deinit();
        zml.Buffer.deinitAll(kda.Cache, &first.cache);
        zml.Buffer.deinitAll(kda.Cache, &second.cache);
        try self.stdout.print(
            "KIMI_K3_KDA_PREFILL_PASS kind=split length={} split={} elapsed_us={}\n",
            .{ length, split, @divTrunc(first_elapsed + second_elapsed, 1000) },
        );
    }

    fn runContinuation(self: *Context) !void {
        var initial = try self.loadCache("continuation.initial");
        defer zml.Buffer.deinitAll(kda.Cache, &initial);
        var actual, const elapsed = try self.run(kda.prefill, "continuation.input.hidden", initial);
        defer actual.output.deinit();
        defer zml.Buffer.deinitAll(kda.Cache, &actual.cache);
        try self.compare("continuation.expected.output", actual.output, output_tolerance);
        try self.compareCache("continuation.expected", actual.cache);
        try self.stdout.print("KIMI_K3_KDA_PREFILL_PASS kind=moonshot_continuation length=4 elapsed_us={}\n", .{@divTrunc(elapsed, 1000)});
    }
};

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
    var context: Context = undefined;
    context = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .store = store.view(),
        .sharding = platform.replicated_sharding,
        .stdout = &stdout_file.interface,
        .weights = undefined,
    };
    context.weights = try context.loadWeights();
    defer zml.Buffer.deinitAll(kda.Weights, &context.weights);

    const lengths = [_]usize{ 1, 4, 8, 16 };
    for (lengths) |length| {
        try context.runFull(length);
        try context.runDecode(length);
    }
    for (1..4) |split| try context.runSplit(4, split);
    for (1..8) |split| try context.runSplit(8, split);
    for ([_]usize{ 1, 4, 8, 12, 15 }) |split| try context.runSplit(16, split);
    try context.runContinuation();
    try stdout_file.interface.writeAll("KIMI_K3_KDA_PREFILL_ALL_PASS full=4 token_decode=4 splits=15 continuation=1 backend=cuda\n");
    try stdout_file.interface.flush();
}
