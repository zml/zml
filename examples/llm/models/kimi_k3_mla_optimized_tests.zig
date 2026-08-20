const std = @import("std");

const zml = @import("zml");
const mla = @import("kimi_k3/mla.zig");

comptime {
    @setEvalBranchQuota(400_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    only: []const u8 = "",
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_mla_optimized_tests --fixture=<mla-optimized-cases.safetensors>
        \\
        \\Validate absorbed latent MLA at cache/page boundaries on NVIDIA CUDA.
        \\
    ;
};

const Inputs = struct {
    q_absorbed: zml.Tensor,
    q_extra: zml.Tensor,
    compressed: zml.Tensor,
    extra_key: zml.Tensor,
    valid_tokens: zml.Tensor,
};

const Case = struct {
    name: []const u8,
    capacity: usize,
    valid_tokens: usize,
    benchmark_ceiling_us: ?i96 = null,
};

const cases = [_]Case{
    .{ .name = "capacity1_valid1", .capacity = 1, .valid_tokens = 1 },
    .{ .name = "capacity32_valid31", .capacity = 32, .valid_tokens = 31 },
    .{ .name = "capacity32_valid32", .capacity = 32, .valid_tokens = 32 },
    .{ .name = "capacity64_valid33", .capacity = 64, .valid_tokens = 33 },
    .{ .name = "capacity64_valid63", .capacity = 64, .valid_tokens = 63 },
    .{ .name = "capacity64_valid64", .capacity = 64, .valid_tokens = 64, .benchmark_ceiling_us = 750 },
    .{ .name = "capacity128_valid65", .capacity = 128, .valid_tokens = 65 },
    .{ .name = "capacity128_valid127", .capacity = 128, .valid_tokens = 127 },
    .{ .name = "capacity128_valid128", .capacity = 128, .valid_tokens = 128 },
    .{ .name = "capacity4096_valid4096", .capacity = 4096, .valid_tokens = 4096, .benchmark_ceiling_us = 900 },
};

const tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 1.2e-2,
    .relative_tolerance = 1.2e-2,
    .minimum_close_fraction = 1.0,
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    sharding: zml.Sharding,
    stdout: *std.Io.Writer,

    fn key(self: *Context, prefix: []const u8, suffix: []const u8) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ prefix, suffix });
    }

    fn load(self: *Context, key_name: []const u8, tags: anytype) !zml.Buffer {
        const shape = self.store.getShape(key_name) orelse return error.MissingMlaOptimizedFixture;
        const bytes = try self.allocator.alloc(u8, shape.byteSize());
        defer self.allocator.free(bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.store.getReader(key_name, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(bytes);
        return zml.Buffer.fromBytes(self.io, self.platform, shape.withTags(tags), self.sharding, bytes);
    }

    fn loadInputs(self: *Context, prefix: []const u8) !zml.Bufferized(Inputs) {
        const q_absorbed_key = try self.key(prefix, "q_absorbed");
        defer self.allocator.free(q_absorbed_key);
        const q_extra_key = try self.key(prefix, "q_extra");
        defer self.allocator.free(q_extra_key);
        const compressed_key = try self.key(prefix, "compressed");
        defer self.allocator.free(compressed_key);
        const extra_key = try self.key(prefix, "extra");
        defer self.allocator.free(extra_key);
        const valid_key = try self.key(prefix, "valid_tokens");
        defer self.allocator.free(valid_key);
        return .{
            .q_absorbed = try self.load(q_absorbed_key, .{ .b, .h, .q, .kv_rank }),
            .q_extra = try self.load(q_extra_key, .{ .b, .h, .q, .hd }),
            .compressed = try self.load(compressed_key, .{ .b, .k, .kv_rank }),
            .extra_key = try self.load(extra_key, .{ .b, .k, .hd }),
            .valid_tokens = try self.load(valid_key, .{.one}),
        };
    }

    fn tensorInputs(inputs: zml.Bufferized(Inputs)) Inputs {
        return .{
            .q_absorbed = .fromShape(inputs.q_absorbed.shape()),
            .q_extra = .fromShape(inputs.q_extra.shape()),
            .compressed = .fromShape(inputs.compressed.shape()),
            .extra_key = .fromShape(inputs.extra_key.shape()),
            .valid_tokens = .fromShape(inputs.valid_tokens.shape()),
        };
    }

    fn compile(self: *Context, symbolic: Inputs) !struct { zml.Exe, i96 } {
        const started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            mla.latentAttentionStableHlo,
            .{
                symbolic.q_absorbed,
                symbolic.q_extra,
                mla.LatentCache{ .compressed = symbolic.compressed, .extra_key = symbolic.extra_key },
                symbolic.valid_tokens,
            },
            .{ .shardings = &.{self.sharding} },
        );
        return .{ exe, @divTrunc(std.Io.Clock.now(.real, self.io).toNanoseconds() - started, 1000) };
    }

    fn execute(self: *Context, exe: *const zml.Exe, inputs: zml.Bufferized(Inputs)) !zml.Buffer {
        return zml.testing.autoCall(
            self.allocator,
            self.io,
            exe,
            mla.latentAttentionStableHlo,
            .{
                inputs.q_absorbed,
                inputs.q_extra,
                zml.Bufferized(mla.LatentCache){ .compressed = inputs.compressed, .extra_key = inputs.extra_key },
                inputs.valid_tokens,
            },
        );
    }

    fn compare(self: *Context, case_name: []const u8, actual: zml.Buffer) !void {
        const expected_key = try self.key(case_name, "expected");
        defer self.allocator.free(expected_key);
        var expected = try self.load(expected_key, .{ .b, .h, .q, .kv_rank });
        defer expected.deinit();
        try zml.testing.expectClose(self.io, actual, expected, tolerance);

        var host = try actual.toSliceAlloc(self.allocator, self.io);
        defer host.free(self.allocator);
        const values = host.items(zml.floats.BFloat16);
        const first = values[0].toF32();
        const last = values[values.len - 1].toF32();
        for (values) |value| if (!std.math.isFinite(value.toF32())) return error.NonfiniteMlaActivation;
        // KIMI_K3_TEMP_REMOVE_M20: activation samples are bring-up diagnostics
        // and are removed after permanent differential coverage lands.
        try self.stdout.print(
            "KIMI_K3_MLA_OPT_ACTIVATION case={s} boundary=latent_aggregation first={d:.8} last={d:.8} finite=true\n",
            .{ case_name, first, last },
        );
        try self.stdout.flush();
    }

    fn benchmark(self: *Context, case_name: []const u8, exe: *const zml.Exe) !i96 {
        const warmups = 2;
        const repetitions = 7;
        var total: i96 = 0;
        for (0..warmups + repetitions) |iteration| {
            var inputs = try self.loadInputs(case_name);
            defer zml.Buffer.deinitAll(Inputs, &inputs);
            const started = std.Io.Clock.now(.real, self.io).toNanoseconds();
            var output = try self.execute(exe, inputs);
            var synchronized = try output.toSliceAlloc(self.allocator, self.io);
            synchronized.free(self.allocator);
            output.deinit();
            const elapsed = @divTrunc(std.Io.Clock.now(.real, self.io).toNanoseconds() - started, 1000);
            if (iteration >= warmups) total += elapsed;
        }
        const mean = @divTrunc(total, repetitions);
        // KIMI_K3_TEMP_REMOVE_M20: synchronized stage timing is retained only
        // through bring-up and replaced by the permanent benchmark suite.
        try self.stdout.print(
            "KIMI_K3_MLA_OPT_BENCH case={s} warmups={} repetitions={} mean_execute_us={}\n",
            .{ case_name, warmups, repetitions, mean },
        );
        try self.stdout.flush();
        return mean;
    }

    fn runCase(self: *Context, one: Case) !void {
        var inputs = try self.loadInputs(one.name);
        defer zml.Buffer.deinitAll(Inputs, &inputs);
        var exe, const compile_us = try self.compile(tensorInputs(inputs));
        defer exe.deinit();
        var output = try self.execute(&exe, inputs);
        defer output.deinit();
        try self.compare(one.name, output);
        const execute_us = if (one.benchmark_ceiling_us) |ceiling| blk: {
            const measured = try self.benchmark(one.name, &exe);
            if (measured > ceiling) return error.MlaStableHloRegressionBudgetExceeded;
            break :blk measured;
        } else 0;
        try self.stdout.print(
            "KIMI_K3_MLA_OPT_PASS case={s} capacity={} valid_tokens={} compile_us={} execute_us={} cache_values_per_token=576 expanded_kv=false\n",
            .{ one.name, one.capacity, one.valid_tokens, compile_us, execute_us },
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
    try stdout_file.interface.writeAll(
        "KIMI_K3_MLA_OPT_ALL_PASS cases=10 boundaries=31,32,33,63,64,65,127,128 long=4096 cache_values_per_token=576 expanded_kv=false backend=cuda\n",
    );
    try stdout_file.interface.flush();
}
