const std = @import("std");

const zml = @import("zml");
const mla = @import("kimi_k3/mla.zig");
const support = @import("kimi_k3_layer0_tests.zig");

comptime {
    @setEvalBranchQuota(300_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_mla_tests --fixture=<expanded-mla-reference.safetensors>
        \\
        \\Run expanded Gated NoPE MLA differentials on NVIDIA CUDA only.
        \\
    ;
};

const projection_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 5e-2,
    .relative_tolerance = 2e-2,
    .minimum_close_fraction = 0.995,
};

const attention_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 2e-2,
    .relative_tolerance = 2e-2,
    .minimum_close_fraction = 0.995,
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    sharding: zml.Sharding,
    stdout: *std.Io.Writer,
    weights: zml.Bufferized(mla.Weights),

    fn load(self: *Context, key: []const u8, tags: anytype) !zml.Buffer {
        return support.loadBuffer(
            self.allocator,
            self.io,
            self.platform,
            self.store,
            key,
            tags,
            self.sharding,
        );
    }

    fn loadWeights(self: *Context) !zml.Bufferized(mla.Weights) {
        return .{
            .q_a_proj = try self.load("weights.q_a_proj", .{ .rank, .d }),
            .q_a_norm = try self.load("weights.q_a_layernorm", .{.rank}),
            .q_b_proj = try self.load("weights.q_b_proj", .{ .mix, .rank }),
            .kv_a_proj = try self.load("weights.kv_a_proj_with_mqa", .{ .kv_mix, .d }),
            .kv_a_norm = try self.load("weights.kv_a_layernorm", .{.kv_rank}),
            .kv_b_proj = try self.load("weights.kv_b_proj", .{ .kv_mix, .kv_rank }),
            .gate_proj = try self.load("weights.g_proj", .{ .out, .d }),
            .output_proj = try self.load("weights.o_proj", .{ .d, .out }),
        };
    }

    fn weightTensors(self: *Context) mla.Weights {
        return .{
            .q_a_proj = .fromShape(self.weights.q_a_proj.shape()),
            .q_a_norm = .fromShape(self.weights.q_a_norm.shape()),
            .q_b_proj = .fromShape(self.weights.q_b_proj.shape()),
            .kv_a_proj = .fromShape(self.weights.kv_a_proj.shape()),
            .kv_a_norm = .fromShape(self.weights.kv_a_norm.shape()),
            .kv_b_proj = .fromShape(self.weights.kv_b_proj.shape()),
            .gate_proj = .fromShape(self.weights.gate_proj.shape()),
            .output_proj = .fromShape(self.weights.output_proj.shape()),
        };
    }

    fn compare(self: *Context, prefix: []const u8, suffix: []const u8, actual: zml.Buffer, opts: zml.testing.CompareOpts) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ prefix, suffix });
        defer self.allocator.free(key);
        try support.compare(
            self.allocator,
            self.io,
            self.platform,
            self.store,
            key,
            actual,
            opts,
            self.sharding,
        );
    }

    fn compareResult(self: *Context, prefix: []const u8, actual: zml.Bufferized(mla.Result)) !void {
        try self.compare(prefix, "q_a", actual.q_a, projection_tolerance);
        try self.compare(prefix, "q_norm", actual.q_norm, projection_tolerance);
        try self.compare(prefix, "q_b", actual.q_b, projection_tolerance);
        try self.compare(prefix, "q_pass", actual.q_pass, projection_tolerance);
        try self.compare(prefix, "q_extra", actual.q_extra, projection_tolerance);
        try self.compare(prefix, "kv_a", actual.kv_a, projection_tolerance);
        try self.compare(prefix, "compressed_kv", actual.compressed_kv, projection_tolerance);
        try self.compare(prefix, "k_extra", actual.k_extra, projection_tolerance);
        try self.compare(prefix, "kv_norm", actual.kv_norm, projection_tolerance);
        try self.compare(prefix, "kv_b", actual.kv_b, projection_tolerance);
        try self.compare(prefix, "k_pass", actual.k_pass, projection_tolerance);
        try self.compare(prefix, "value_new", actual.value_new, projection_tolerance);
        try self.compare(prefix, "query", actual.query, projection_tolerance);
        try self.compare(prefix, "key_new", actual.key_new, projection_tolerance);
        try self.compare(prefix, "cache_key", actual.cache.key, projection_tolerance);
        try self.compare(prefix, "cache_value", actual.cache.value, projection_tolerance);
        try self.compare(prefix, "scores", actual.scores, attention_tolerance);
        // masked_scores contains intentional -inf values. Probabilities verify
        // the causal masking without treating those sentinels as numeric errors.
        try self.compare(prefix, "probabilities", actual.probabilities, attention_tolerance);
        try self.compare(prefix, "aggregation", actual.aggregation, projection_tolerance);
        try self.compare(prefix, "flattened", actual.flattened, projection_tolerance);
        try self.compare(prefix, "gate_logits", actual.gate_logits, projection_tolerance);
        try self.compare(prefix, "gate", actual.gate, attention_tolerance);
        try self.compare(prefix, "gated", actual.gated, projection_tolerance);
        try self.compare(prefix, "output", actual.output, projection_tolerance);
        try self.compare(prefix, "official_output", actual.output, projection_tolerance);
    }

    fn runPrefill(self: *Context, length: usize) !void {
        const prefix = try std.fmt.allocPrint(self.allocator, "len{}", .{length});
        defer self.allocator.free(prefix);
        const input_key = try std.fmt.allocPrint(self.allocator, "{s}.input", .{prefix});
        defer self.allocator.free(input_key);
        var input = try self.load(input_key, .{ .b, .s, .d });
        defer input.deinit();
        const compile_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            mla.prefill,
            .{ zml.Tensor.fromShape(input.shape()), self.weightTensors() },
            .{ .shardings = &.{self.sharding} },
        );
        defer exe.deinit();
        const compile_us = @divTrunc(std.Io.Clock.now(.real, self.io).toNanoseconds() - compile_started, 1000);
        const execute_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        var actual = try zml.testing.autoCall(
            self.allocator,
            self.io,
            &exe,
            mla.prefill,
            .{ input, self.weights },
        );
        defer zml.Buffer.deinitAll(mla.Result, &actual);
        const execute_us = @divTrunc(std.Io.Clock.now(.real, self.io).toNanoseconds() - execute_started, 1000);
        try self.compareResult(prefix, actual);
        try self.stdout.print(
            "KIMI_K3_MLA_PASS kind=prefill length={} boundaries=25 compile_us={} execute_us={}\n",
            .{ length, compile_us, execute_us },
        );
    }

    fn runDecode(self: *Context) !void {
        var input = try self.load("decode.input", .{ .b, .s, .d });
        defer input.deinit();
        var cache: zml.Bufferized(mla.ExpandedCache) = .{
            .key = try self.load("decode.past_key", .{ .b, .h, .k, .hd }),
            .value = try self.load("decode.past_value", .{ .b, .h, .k, .v }),
        };
        defer zml.Buffer.deinitAll(mla.ExpandedCache, &cache);
        const cache_tensors: mla.ExpandedCache = .{
            .key = .fromShape(cache.key.shape()),
            .value = .fromShape(cache.value.shape()),
        };
        const compile_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            mla.decode,
            .{ zml.Tensor.fromShape(input.shape()), self.weightTensors(), cache_tensors },
            .{ .shardings = &.{self.sharding} },
        );
        defer exe.deinit();
        const compile_us = @divTrunc(std.Io.Clock.now(.real, self.io).toNanoseconds() - compile_started, 1000);
        const execute_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        var actual = try zml.testing.autoCall(
            self.allocator,
            self.io,
            &exe,
            mla.decode,
            .{ input, self.weights, cache },
        );
        defer zml.Buffer.deinitAll(mla.Result, &actual);
        const execute_us = @divTrunc(std.Io.Clock.now(.real, self.io).toNanoseconds() - execute_started, 1000);
        try self.compareResult("decode", actual);
        try self.stdout.print(
            "KIMI_K3_MLA_PASS kind=decode past=4 length=1 boundaries=25 compile_us={} execute_us={}\n",
            .{ compile_us, execute_us },
        );
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.85 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.fixture);
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
        .weights = undefined,
    };
    context.weights = try context.loadWeights();
    defer zml.Buffer.deinitAll(mla.Weights, &context.weights);
    for ([_]usize{ 1, 4, 8, 16 }) |length| try context.runPrefill(length);
    try context.runDecode();
    try stdout_file.interface.writeAll("KIMI_K3_MLA_ALL_PASS prefill=4 decode=1 boundaries_per_case=25 backend=cuda\n");
    try stdout_file.interface.flush();
}
