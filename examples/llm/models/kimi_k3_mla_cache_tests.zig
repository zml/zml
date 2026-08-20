const std = @import("std");

const zml = @import("zml");
const mla = @import("kimi_k3/mla.zig");
const support = @import("kimi_k3_layer0_tests.zig");

comptime {
    @setEvalBranchQuota(300_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    reference: []const u8,
    cases: []const u8,

    pub const help =
        \\Use kimi_k3_mla_cache_tests --reference=<M12.safetensors> --cases=<M13.safetensors>
        \\
        \\Compare production-shaped latent MLA caches to the expanded CUDA oracle.
        \\
    ;
};

const output_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 7e-2,
    .relative_tolerance = 3e-2,
    .minimum_close_fraction = 0.995,
};

const probability_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 3e-2,
    .relative_tolerance = 3e-2,
    .minimum_close_fraction = 0.995,
};

const cache_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 5e-2,
    .relative_tolerance = 2e-2,
    .minimum_close_fraction = 0.995,
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    reference: zml.io.TensorStore.View,
    cases: zml.io.TensorStore.View,
    sharding: zml.Sharding,
    stdout: *std.Io.Writer,
    weights: zml.Bufferized(mla.Weights),

    fn loadFrom(self: *Context, store: zml.io.TensorStore.View, key: []const u8, tags: anytype) !zml.Buffer {
        return support.loadBuffer(
            self.allocator,
            self.io,
            self.platform,
            store,
            key,
            tags,
            self.sharding,
        );
    }

    fn loadWeights(self: *Context) !zml.Bufferized(mla.Weights) {
        return .{
            .q_a_proj = try self.loadFrom(self.reference, "weights.q_a_proj", .{ .rank, .d }),
            .q_a_norm = try self.loadFrom(self.reference, "weights.q_a_layernorm", .{.rank}),
            .q_b_proj = try self.loadFrom(self.reference, "weights.q_b_proj", .{ .mix, .rank }),
            .kv_a_proj = try self.loadFrom(self.reference, "weights.kv_a_proj_with_mqa", .{ .kv_mix, .d }),
            .kv_a_norm = try self.loadFrom(self.reference, "weights.kv_a_layernorm", .{.kv_rank}),
            .kv_b_proj = try self.loadFrom(self.reference, "weights.kv_b_proj", .{ .kv_mix, .kv_rank }),
            .gate_proj = try self.loadFrom(self.reference, "weights.g_proj", .{ .out, .d }),
            .output_proj = try self.loadFrom(self.reference, "weights.o_proj", .{ .d, .out }),
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

    fn cacheTensors(cache: zml.Bufferized(mla.LatentCache)) mla.LatentCache {
        return .{
            .compressed = .fromShape(cache.compressed.shape()),
            .extra_key = .fromShape(cache.extra_key.shape()),
        };
    }

    fn sessionCacheTensors(cache: zml.Bufferized(mla.SessionCache)) mla.SessionCache {
        return .{
            .compressed = .fromShape(cache.compressed.shape()),
            .extra_key = .fromShape(cache.extra_key.shape()),
        };
    }

    fn zeroSessionCache(self: *Context, capacity: usize) !zml.Bufferized(mla.SessionCache) {
        return .{
            .compressed = try support.zeroBuffer(
                self.allocator,
                self.io,
                self.platform,
                zml.Shape.init(.{ .b = 1, .k = capacity, .kv_rank = 512 }, .bf16),
                self.sharding,
            ),
            .extra_key = try support.zeroBuffer(
                self.allocator,
                self.io,
                self.platform,
                zml.Shape.init(.{ .b = 1, .k = capacity, .hd = 64 }, .bf16),
                self.sharding,
            ),
        };
    }

    fn compareExpected(self: *Context, key: []const u8, actual: zml.Buffer, opts: zml.testing.CompareOpts) !void {
        try support.compare(
            self.allocator,
            self.io,
            self.platform,
            self.cases,
            key,
            actual,
            opts,
            self.sharding,
        );
    }

    fn compareCase(self: *Context, prefix: []const u8, actual: zml.Bufferized(mla.LatentResult)) !void {
        const output_key = try std.fmt.allocPrint(self.allocator, "{s}.expected.output", .{prefix});
        defer self.allocator.free(output_key);
        const probabilities_key = try std.fmt.allocPrint(self.allocator, "{s}.expected.probabilities", .{prefix});
        defer self.allocator.free(probabilities_key);
        const compressed_key = try std.fmt.allocPrint(self.allocator, "{s}.expected.cache.compressed", .{prefix});
        defer self.allocator.free(compressed_key);
        const extra_key = try std.fmt.allocPrint(self.allocator, "{s}.expected.cache.extra_key", .{prefix});
        defer self.allocator.free(extra_key);
        try self.compareExpected(output_key, actual.output, output_tolerance);
        try self.compareExpected(probabilities_key, actual.probabilities, probability_tolerance);
        try self.compareExpected(compressed_key, actual.cache.compressed, cache_tolerance);
        try self.compareExpected(extra_key, actual.cache.extra_key, cache_tolerance);
    }

    fn callPrefill(self: *Context, prefix: []const u8) !struct { zml.Bufferized(mla.LatentResult), i96, i96 } {
        const input_key = try std.fmt.allocPrint(self.allocator, "{s}.input", .{prefix});
        defer self.allocator.free(input_key);
        var input = try self.loadFrom(self.cases, input_key, .{ .b, .s, .d });
        defer input.deinit();
        const compile_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            mla.latentPrefill,
            .{ zml.Tensor.fromShape(input.shape()), self.weightTensors() },
            .{ .shardings = &.{self.sharding} },
        );
        defer exe.deinit();
        const compile_ns = std.Io.Clock.now(.real, self.io).toNanoseconds() - compile_started;
        const execute_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        const actual = try zml.testing.autoCall(
            self.allocator,
            self.io,
            &exe,
            mla.latentPrefill,
            .{ input, self.weights },
        );
        return .{ actual, compile_ns, std.Io.Clock.now(.real, self.io).toNanoseconds() - execute_started };
    }

    fn callContinue(
        self: *Context,
        prefix: []const u8,
        cache: zml.Bufferized(mla.LatentCache),
    ) !struct { zml.Bufferized(mla.LatentResult), i96, i96 } {
        const input_key = try std.fmt.allocPrint(self.allocator, "{s}.input", .{prefix});
        defer self.allocator.free(input_key);
        var input = try self.loadFrom(self.cases, input_key, .{ .b, .s, .d });
        defer input.deinit();
        const compile_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            mla.latentContinue,
            .{ zml.Tensor.fromShape(input.shape()), self.weightTensors(), cacheTensors(cache) },
            .{ .shardings = &.{self.sharding} },
        );
        defer exe.deinit();
        const compile_ns = std.Io.Clock.now(.real, self.io).toNanoseconds() - compile_started;
        const execute_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        const actual = try zml.testing.autoCall(
            self.allocator,
            self.io,
            &exe,
            mla.latentContinue,
            .{ input, self.weights, cache },
        );
        return .{ actual, compile_ns, std.Io.Clock.now(.real, self.io).toNanoseconds() - execute_started };
    }

    fn deinitNonCache(result: *zml.Bufferized(mla.LatentResult)) void {
        result.output.deinit();
        result.probabilities.deinit();
        result.q_absorbed.deinit();
        result.latent_aggregation.deinit();
    }

    fn runFull(self: *Context, length: usize) !void {
        const prefix = try std.fmt.allocPrint(self.allocator, "full.len{}", .{length});
        defer self.allocator.free(prefix);
        var actual, const compile_ns, const execute_ns = try self.callPrefill(prefix);
        defer zml.Buffer.deinitAll(mla.LatentResult, &actual);
        try self.compareCase(prefix, actual);
        try self.stdout.print(
            "KIMI_K3_MLA_CACHE_PASS kind=full length={} compile_us={} execute_us={}\n",
            .{ length, @divTrunc(compile_ns, 1000), @divTrunc(execute_ns, 1000) },
        );
    }

    fn runSplit(self: *Context, split: usize) !void {
        const first_prefix = try std.fmt.allocPrint(self.allocator, "split4.at{}.first", .{split});
        defer self.allocator.free(first_prefix);
        const second_prefix = try std.fmt.allocPrint(self.allocator, "split4.at{}.second", .{split});
        defer self.allocator.free(second_prefix);
        var first, const first_compile, const first_execute = try self.callPrefill(first_prefix);
        defer zml.Buffer.deinitAll(mla.LatentResult, &first);
        try self.compareCase(first_prefix, first);
        var second, const second_compile, const second_execute = try self.callContinue(second_prefix, first.cache);
        defer zml.Buffer.deinitAll(mla.LatentResult, &second);
        try self.compareCase(second_prefix, second);
        try self.stdout.print(
            "KIMI_K3_MLA_CACHE_PASS kind=split length=4 split={} compile_us={} execute_us={}\n",
            .{
                split,
                @divTrunc(first_compile + second_compile, 1000),
                @divTrunc(first_execute + second_execute, 1000),
            },
        );
    }

    fn runRepeatedDecode(self: *Context) !void {
        var cache: ?zml.Bufferized(mla.LatentCache) = null;
        defer if (cache) |*value| zml.Buffer.deinitAll(mla.LatentCache, value);
        var compile_ns: i96 = 0;
        var execute_ns: i96 = 0;
        for (0..4) |token| {
            const prefix = try std.fmt.allocPrint(self.allocator, "decode4.token{}", .{token});
            defer self.allocator.free(prefix);
            var actual, const current_compile, const current_execute = if (cache) |previous|
                try self.callContinue(prefix, previous)
            else
                try self.callPrefill(prefix);
            try self.compareCase(prefix, actual);
            compile_ns += current_compile;
            execute_ns += current_execute;
            Context.deinitNonCache(&actual);
            if (cache) |*previous| zml.Buffer.deinitAll(mla.LatentCache, previous);
            cache = actual.cache;
        }
        try self.stdout.print(
            "KIMI_K3_MLA_CACHE_PASS kind=repeated_decode steps=4 compile_us={} execute_us={}\n",
            .{ @divTrunc(compile_ns, 1000), @divTrunc(execute_ns, 1000) },
        );
    }

    fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
        if (a._shards.len != b._shards.len) return false;
        for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
            if (a_shard != b_shard) return false;
        }
        return true;
    }

    fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
        if (!sameBufferHandle(dst.*, src.*)) dst.deinit();
        dst.* = src.*;
    }

    fn runReusableSessionDecode(self: *Context) !void {
        const capacity: usize = 4;
        var first_input = try self.loadFrom(self.cases, "decode4.token0.input", .{ .b, .s, .d });
        defer first_input.deinit();
        var cache = try self.zeroSessionCache(capacity);
        defer zml.Buffer.deinitAll(mla.SessionCache, &cache);
        var token_index = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32);
        defer token_index.deinit();

        const compile_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            mla.latentSession,
            .{
                zml.Tensor.fromShape(first_input.shape()),
                self.weightTensors(),
                sessionCacheTensors(cache),
                zml.Tensor.fromShape(token_index.shape()),
            },
            .{ .shardings = &.{self.sharding} },
        );
        defer exe.deinit();
        const compile_ns = std.Io.Clock.now(.real, self.io).toNanoseconds() - compile_started;

        for (0..2) |reset_index| {
            if (reset_index != 0) {
                zml.Buffer.deinitAll(mla.SessionCache, &cache);
                cache = try self.zeroSessionCache(capacity);
            }
            const execute_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
            for (0..capacity) |position| {
                const prefix = try std.fmt.allocPrint(self.allocator, "decode4.token{}", .{position});
                defer self.allocator.free(prefix);
                const input_key = try std.fmt.allocPrint(self.allocator, "{s}.input", .{prefix});
                defer self.allocator.free(input_key);
                var input = try self.loadFrom(self.cases, input_key, .{ .b, .s, .d });
                defer input.deinit();
                var position_buffer = try zml.Buffer.scalar(
                    self.io,
                    self.platform,
                    @as(u32, @intCast(position)),
                    .u32,
                );
                defer position_buffer.deinit();
                var actual = try zml.testing.autoCall(
                    self.allocator,
                    self.io,
                    &exe,
                    mla.latentSession,
                    .{ input, self.weights, cache, position_buffer },
                );
                const output_key = try std.fmt.allocPrint(self.allocator, "{s}.expected.output", .{prefix});
                defer self.allocator.free(output_key);
                try self.compareExpected(output_key, actual.output, output_tolerance);
                Context.deinitNonCache(&actual);
                replaceBuffer(&cache.compressed, &actual.cache.compressed);
                replaceBuffer(&cache.extra_key, &actual.cache.extra_key);
            }
            try self.compareExpected("decode4.token3.expected.cache.compressed", cache.compressed, cache_tolerance);
            try self.compareExpected("decode4.token3.expected.cache.extra_key", cache.extra_key, cache_tolerance);
            const execute_ns = std.Io.Clock.now(.real, self.io).toNanoseconds() - execute_started;
            try self.stdout.print(
                "KIMI_K3_MLA_SESSION_CACHE_PASS reset={} steps={} compile_us={} execute_us={} capacity={}\n",
                .{ reset_index, capacity, @divTrunc(compile_ns, 1000), @divTrunc(execute_ns, 1000), capacity },
            );
            try self.stdout.flush();
        }
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
    var reference_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.reference);
    defer reference_registry.deinit();
    var reference_store: zml.io.TensorStore = .fromRegistry(allocator, &reference_registry);
    defer reference_store.deinit();
    var case_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.cases);
    defer case_registry.deinit();
    var case_store: zml.io.TensorStore = .fromRegistry(allocator, &case_registry);
    defer case_store.deinit();
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    var context: Context = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .reference = reference_store.view(),
        .cases = case_store.view(),
        .sharding = platform.replicated_sharding,
        .stdout = &stdout_file.interface,
        .weights = undefined,
    };
    context.weights = try context.loadWeights();
    defer zml.Buffer.deinitAll(mla.Weights, &context.weights);
    for ([_]usize{ 1, 4, 8, 16 }) |length| try context.runFull(length);
    for ([_]usize{ 1, 2, 3 }) |split| try context.runSplit(split);
    try context.runRepeatedDecode();
    try context.runReusableSessionDecode();
    try stdout_file.interface.writeAll("KIMI_K3_MLA_CACHE_ALL_PASS full=4 splits=3 repeated_decode_steps=4 values_per_token=576 backend=cuda\n");
    try stdout_file.interface.flush();
}
