const std = @import("std");

const zml = @import("zml");
const primitives = @import("kimi_k3/primitives.zig");

comptime {
    @setEvalBranchQuota(100_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_primitives_tests --fixture=<primitive-reference.safetensors>
        \\
        \\Run Kimi K3 primitive differential tests on NVIDIA CUDA only.
        \\
    ;
};

const strict: zml.testing.CompareOpts = .{
    .absolute_tolerance = 1e-5,
    .relative_tolerance = 1e-5,
    .minimum_close_fraction = 1.0,
};

const Forward = struct {
    fn rms(input: zml.Tensor, weight: zml.Tensor) zml.Tensor {
        return primitives.rmsNorm(input, weight, 1e-6);
    }

    fn l2(input: zml.Tensor) zml.Tensor {
        return primitives.normalizeL2(input, 1e-6);
    }

    fn situ(gate: zml.Tensor, up: zml.Tensor) zml.Tensor {
        return primitives.situGlu(gate, up);
    }

    fn slowMxfp4Bf16(input: zml.Tensor, packed_values: zml.Tensor, scale: zml.Tensor) zml.Tensor {
        return primitives.slowMxfp4Linear(input.convert(.bf16), packed_values, scale).convert(.f32);
    }

    fn nativeMxfp4(input: zml.Tensor, packed_values: zml.Tensor, scale: zml.Tensor) zml.Tensor {
        return primitives.nativeMxfp4Linear(input.convert(.bf16), packed_values, scale).convert(.f32);
    }
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    sharding: zml.Sharding,
    stdout: *std.Io.Writer,

    fn compareUnary(
        self: *Context,
        name: []const u8,
        comptime function: anytype,
        input_key: []const u8,
        expected_key: []const u8,
        tags: anytype,
        opts: zml.testing.CompareOpts,
    ) !void {
        @setEvalBranchQuota(100_000);
        var input_buffer = try loadBuffer(self, input_key);
        defer input_buffer.deinit();
        var expected = try loadBuffer(self, expected_key);
        defer expected.deinit();
        const input = zml.Tensor.fromShape(input_buffer.shape()).withTags(tags);
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            function,
            .{input},
            .{ .shardings = &.{self.sharding} },
        );
        defer exe.deinit();
        const started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        var actual = try zml.testing.autoCall(
            self.allocator,
            self.io,
            &exe,
            function,
            .{input_buffer},
        );
        defer actual.deinit();
        try zml.testing.expectClose(self.io, actual, expected, opts);
        try self.pass(name, started, input.shape());
    }

    fn compareBinary(
        self: *Context,
        name: []const u8,
        comptime function: anytype,
        lhs_key: []const u8,
        rhs_key: []const u8,
        expected_key: []const u8,
        lhs_tags: anytype,
        rhs_tags: anytype,
        opts: zml.testing.CompareOpts,
    ) !void {
        @setEvalBranchQuota(100_000);
        var lhs_buffer = try loadBuffer(self, lhs_key);
        defer lhs_buffer.deinit();
        var rhs_buffer = try loadBuffer(self, rhs_key);
        defer rhs_buffer.deinit();
        var expected = try loadBuffer(self, expected_key);
        defer expected.deinit();
        const lhs = zml.Tensor.fromShape(lhs_buffer.shape()).withTags(lhs_tags);
        const rhs = zml.Tensor.fromShape(rhs_buffer.shape()).withTags(rhs_tags);
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            function,
            .{ lhs, rhs },
            .{ .shardings = &.{self.sharding} },
        );
        defer exe.deinit();
        const started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        var actual = try zml.testing.autoCall(
            self.allocator,
            self.io,
            &exe,
            function,
            .{ lhs_buffer, rhs_buffer },
        );
        defer actual.deinit();
        try zml.testing.expectClose(self.io, actual, expected, opts);
        try self.pass(name, started, lhs.shape());
    }

    fn compareMxfp4Linear(
        self: *Context,
        name: []const u8,
        comptime function: anytype,
        first_key: []const u8,
        second_key: []const u8,
        third_key: []const u8,
        expected_key: []const u8,
        opts: zml.testing.CompareOpts,
    ) !void {
        @setEvalBranchQuota(100_000);
        var first_buffer = try loadBuffer(self, first_key);
        defer first_buffer.deinit();
        var second_buffer = try loadBuffer(self, second_key);
        defer second_buffer.deinit();
        var third_buffer = try loadBuffer(self, third_key);
        defer third_buffer.deinit();
        var expected = try loadBuffer(self, expected_key);
        defer expected.deinit();
        const first = zml.Tensor.fromShape(first_buffer.shape()).withTags(.{ .token, .d });
        const second = zml.Tensor.fromShape(second_buffer.shape()).withTags(.{ .out, .kw });
        const third = zml.Tensor.fromShape(third_buffer.shape()).withTags(.{ .out, .block });
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            function,
            .{ first, second, third },
            .{ .shardings = &.{self.sharding} },
        );
        defer exe.deinit();
        const started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        var actual = try zml.testing.autoCall(
            self.allocator,
            self.io,
            &exe,
            function,
            .{ first_buffer, second_buffer, third_buffer },
        );
        defer actual.deinit();
        try zml.testing.expectClose(self.io, actual, expected, opts);
        try self.pass(name, started, first.shape());
    }

    fn compareNativeMxfp4Linear(
        self: *Context,
        name: []const u8,
        first_key: []const u8,
        second_key: []const u8,
        third_key: []const u8,
        opts: zml.testing.CompareOpts,
    ) !void {
        @setEvalBranchQuota(100_000);
        var first_buffer = try loadBuffer(self, first_key);
        defer first_buffer.deinit();
        var second_buffer = try loadBuffer(self, second_key);
        defer second_buffer.deinit();
        var third_buffer = try loadBuffer(self, third_key);
        defer third_buffer.deinit();
        const first = zml.Tensor.fromShape(first_buffer.shape()).withTags(.{ .token, .d });
        const second = zml.Tensor.fromShape(second_buffer.shape()).withTags(.{ .out, .kw });
        const third = zml.Tensor.fromShape(third_buffer.shape()).withTags(.{ .out, .block });
        const shardings = &.{self.sharding};
        const slow_exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            Forward.slowMxfp4Bf16,
            .{ first, second, third },
            .{ .shardings = shardings },
        );
        defer slow_exe.deinit();
        const native_exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            Forward.nativeMxfp4,
            .{ first, second, third },
            .{ .shardings = shardings },
        );
        defer native_exe.deinit();
        var expected = try zml.testing.autoCall(
            self.allocator,
            self.io,
            &slow_exe,
            Forward.slowMxfp4Bf16,
            .{ first_buffer, second_buffer, third_buffer },
        );
        defer expected.deinit();
        const started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        var actual = try zml.testing.autoCall(
            self.allocator,
            self.io,
            &native_exe,
            Forward.nativeMxfp4,
            .{ first_buffer, second_buffer, third_buffer },
        );
        defer actual.deinit();
        var actual_host = try actual.toSliceAlloc(self.allocator, self.io);
        defer actual_host.free(self.allocator);
        var expected_host = try expected.toSliceAlloc(self.allocator, self.io);
        defer expected_host.free(self.allocator);
        try zml.testing.expectClose(self.io, actual_host, expected_host, opts);
        try self.pass(name, started, first.shape());
    }

    fn loadBuffer(self: *Context, key: []const u8) !zml.Buffer {
        const shape = self.store.getShape(key) orelse {
            try self.stdout.print("KIMI_K3_PRIMITIVE_MISSING key={s}\n", .{key});
            return error.MissingPrimitiveFixture;
        };
        const host_bytes = try self.allocator.alloc(u8, shape.byteSize());
        defer self.allocator.free(host_bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.store.getReader(key, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(host_bytes);
        return zml.Buffer.fromBytes(self.io, self.platform, shape, self.sharding, host_bytes);
    }

    fn pass(self: *Context, name: []const u8, started_ns: i96, shape: zml.Shape) !void {
        const elapsed_ns = std.Io.Clock.now(.real, self.io).toNanoseconds() - started_ns;
        try self.stdout.print(
            "KIMI_K3_PRIMITIVE_PASS name={s} elapsed_us={} input_shape={f}\n",
            .{ name, @divTrunc(elapsed_ns, 1000), shape },
        );
        try self.stdout.flush();
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    // Fail closed: this executable is a CUDA differential test and must never
    // silently select the CPU PJRT plugin when CUDA support was not built in.
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
    var ctx: Context = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .store = store.view(),
        .sharding = platform.replicated_sharding,
        .stdout = &stdout_file.interface,
    };

    try ctx.compareBinary("rms", Forward.rms, "rms.input", "rms.weight", "rms.expected", .{ .batch, .d }, .{.d}, strict);
    try ctx.compareBinary("rms_real_slice", Forward.rms, "rms_real.input", "rms_real.weight", "rms_real.expected", .{ .batch, .d }, .{.d}, strict);
    try ctx.compareUnary("l2", Forward.l2, "l2.input", "l2.expected", .{ .batch, .d }, strict);
    try ctx.compareBinary("situ_glu", Forward.situ, "situ.gate", "situ.up", "situ.expected", .{ .batch, .d }, .{ .batch, .d }, strict);
    try ctx.compareUnary("sigmoid", primitives.sigmoid, "sigmoid.input", "sigmoid.expected", .{ .batch, .d }, strict);
    try ctx.compareUnary("softmax", primitives.softmax, "softmax.input", "softmax.expected", .{ .batch, .d }, strict);
    try ctx.compareUnary("topk_values", primitives.topKValues, "topk.input", "topk.expected_values", .{ .batch, .d }, zml.testing.CompareOpts.exact_match);
    try ctx.compareUnary("topk_ids", primitives.topKIndices, "topk.input", "topk.expected_ids", .{ .batch, .d }, zml.testing.CompareOpts.exact_match);
    try ctx.compareBinary("causal_depthwise_conv", primitives.causalDepthwiseConv1d, "conv.input", "conv.kernel", "conv.expected", .{ .batch, .sequence, .channel }, .{ .channel, .one, .kernel }, strict);
    try ctx.compareUnary("conv_tail", primitives.causalConvTail3, "conv.input", "conv.expected_tail", .{ .batch, .sequence, .channel }, zml.testing.CompareOpts.exact_match);
    try ctx.compareUnary("conv_short_tail", primitives.causalConvTail3, "conv.short_input", "conv.short_expected_tail", .{ .batch, .sequence, .channel }, zml.testing.CompareOpts.exact_match);
    try ctx.compareBinary("mla_nope_join", primitives.mlaNopeJoin, "mla.content", "mla.extra", "mla.expected_join", .{ .batch, .head, .sequence, .head_dim }, .{ .batch, .head, .sequence, .head_dim }, zml.testing.CompareOpts.exact_match);
    try ctx.compareUnary("mla_scale", primitives.mlaScale, "mla.scores", "mla.expected_scaled", .{ .batch, .head, .sequence, .key }, strict);
    try ctx.compareUnary("e2m1_unpack", primitives.unpackE2m1, "mxfp4.packed", "mxfp4.expected_unpacked", .{ .out, .kw }, zml.testing.CompareOpts.exact_match);
    try ctx.compareUnary("e8m0_decode", primitives.decodeE8m0, "mxfp4.scale_e8m0", "mxfp4.expected_scale", .{ .out, .block }, strict);
    try ctx.compareUnary("block32_expand", primitives.expandBlock32Scale, "mxfp4.scale_e8m0", "mxfp4.expected_expanded", .{ .out, .block }, strict);
    try ctx.compareBinary("mxfp4_dequant", primitives.dequantizeMxfp4, "mxfp4.packed", "mxfp4.scale_e8m0", "mxfp4.expected_weight", .{ .out, .kw }, .{ .out, .block }, strict);
    try ctx.compareMxfp4Linear("mxfp4_slow_linear", primitives.slowMxfp4Linear, "mxfp4.linear_input", "mxfp4.packed", "mxfp4.scale_e8m0", "mxfp4.expected_linear", strict);
    try ctx.compareNativeMxfp4Linear("mxfp4_native_linear", "mxfp4.linear_input", "mxfp4.packed", "mxfp4.scale_e8m0", .{ .absolute_tolerance = 0.25, .relative_tolerance = 0.025, .minimum_close_fraction = 1.0 });
    try ctx.compareBinary("mxfp4_real_dequant", primitives.dequantizeMxfp4, "mxfp4_real.packed", "mxfp4_real.scale_e8m0", "mxfp4_real.expected_weight", .{ .out, .kw }, .{ .out, .block }, strict);
    try ctx.compareMxfp4Linear("mxfp4_real_slow_linear", primitives.slowMxfp4Linear, "mxfp4_real.linear_input", "mxfp4_real.packed", "mxfp4_real.scale_e8m0", "mxfp4_real.expected_linear", strict);
    try ctx.compareNativeMxfp4Linear("mxfp4_real_native_linear", "mxfp4_real.linear_input", "mxfp4_real.packed", "mxfp4_real.scale_e8m0", .{});

    try stdout_file.interface.writeAll("KIMI_K3_PRIMITIVES_ALL_PASS count=22 backend=cuda\n");
    try stdout_file.interface.flush();
}
