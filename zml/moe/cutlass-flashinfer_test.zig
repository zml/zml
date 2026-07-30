const std = @import("std");

const platforms = @import("platforms");
const zml = @import("zml");

const cutlass = zml.moe.cutlass_flashinfer;
const BFloat16 = zml.floats.BFloat16;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.flashinfer_cutlass_moe_test);

const Args = struct {
    model: []const u8,
    tokens: u32 = 4,
    layer: usize = 0,
    seed: u64 = 0,

    pub const help =
        \\Use flashinfer_cutlass_moe_gemma4_test --model=<path>
        \\
        \\ Load one Gemma 4 MoE layer, route random BF16 hidden states with
        \\ the model's real router, and compare Triton with FlashInfer CUTLASS.
        \\
        \\ Options:
        \\   --model=<path>   Path to the Gemma 4 model repository
        \\   --tokens=<n>     Number of random tokens (default: 4)
        \\   --layer=<n>      Zero-based MoE layer index (default: 0)
        \\   --seed=<n>       Random hidden-state seed (default: 0)
        \\
    ;
};

const Gemma4Router = struct {
    scale: zml.Tensor,
    proj: zml.Tensor,
    per_expert_scale: zml.Tensor,
    rms_norm_eps: f32 = 1e-6,
    top_k_experts: u32 = 8,

    fn forward(self: Gemma4Router, hidden_states_: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        // This intentionally mirrors Gemma4Router.forward in llmd. Gemma 4's
        // router RMSNorm has no learned weight.
        var hidden_states = hidden_states_.convert(.f32);
        hidden_states = hidden_states.mul(
            zml.Tensor.rsqrt(
                hidden_states.powByConst(2).mean(.d).addConstant(self.rms_norm_eps),
            ),
        ).convert(hidden_states_.dtype());

        hidden_states = hidden_states
            .mul(self.scale.convert(hidden_states.dtype()).broad(hidden_states.shape()))
            .scale(std.math.pow(
            f32,
            @as(f32, @floatFromInt(hidden_states.dim(.d))),
            -0.5,
        ));

        const projection: zml.nn.Linear = .init(self.proj, null, .d);
        const expert_scores = projection.forward(hidden_states)
            .withPartialTags(.{.expert})
            .convert(.f32);
        const router_probabilities = expert_scores.softmax(.expert);
        const routing = router_probabilities.topK(
            .{ .top_expert = .expert },
            self.top_k_experts,
            .{},
        );

        const top_k_indices = routing.indices.convert(.i64);
        var top_k_weights = routing.values.div(
            routing.values.sum(.top_expert).addConstant(1e-20),
        );
        const expert_scale = self.per_expert_scale.convert(.f32).gather(
            .{ .expert = top_k_indices },
            .{},
        );
        top_k_weights = top_k_weights.mul(expert_scale);

        return .{ top_k_weights, top_k_indices.convert(.i32) };
    }
};

const Gemma4Moe = struct {
    router: Gemma4Router,
    gate_up: zml.Tensor,
    down: zml.Tensor,

    fn init(store: zml.io.TensorStore.View, layer_index: usize) !Gemma4Moe {
        const layer = store
            .withPrefix("model.language_model.layers")
            .withLayer(layer_index);
        const router = layer.withPrefix("router");
        const experts = layer.withPrefix("experts");

        const result: Gemma4Moe = .{
            .router = .{
                .scale = router.maybeCreateTensor(
                    "scale",
                    .{.d},
                    .replicated,
                ) orelse return error.MissingGemma4MoeTensor,
                .proj = router.maybeCreateTensor(
                    "proj.weight",
                    .{ .expert, .d },
                    .replicated,
                ) orelse return error.MissingGemma4MoeTensor,
                .per_expert_scale = router.maybeCreateTensor(
                    "per_expert_scale",
                    .{.expert},
                    .replicated,
                ) orelse return error.MissingGemma4MoeTensor,
            },
            .gate_up = experts.maybeCreateTensor(
                "gate_up_proj",
                .{ .expert, .dout, .d },
                .replicated,
            ) orelse return error.MissingGemma4MoeTensor,
            .down = experts.maybeCreateTensor(
                "down_proj",
                .{ .expert, .d, .dout },
                .replicated,
            ) orelse return error.MissingGemma4MoeTensor,
        };

        if (result.gate_up.dtype() != .bf16 or
            result.down.dtype() != .bf16 or
            result.router.proj.dtype() != .bf16)
        {
            return error.ExpectedBf16Gemma4Moe;
        }
        if (result.router.proj.dim(.expert) != result.gate_up.dim(.expert) or
            result.down.dim(.expert) != result.gate_up.dim(.expert) or
            result.router.proj.dim(.d) != result.gate_up.dim(.d) or
            result.down.dim(.d) != result.gate_up.dim(.d) or
            result.gate_up.dim(.dout) != 2 * result.down.dim(.dout))
        {
            return error.InvalidGemma4MoeShape;
        }

        return result;
    }

    fn deinitBuffers(buffers: *zml.Bufferized(Gemma4Moe)) void {
        buffers.router.scale.deinit();
        buffers.router.proj.deinit();
        buffers.router.per_expert_scale.deinit();
        buffers.gate_up.deinit();
        buffers.down.deinit();
    }
};

const TestMoe = struct {
    fn cutlassForward(
        hidden: zml.Tensor,
        topk_ids: zml.Tensor,
        topk_weights: zml.Tensor,
        gate_up: zml.Tensor,
        down: zml.Tensor,
    ) zml.Tensor {
        const backend: zml.moe.Backend = .flashinfer_cutlass;
        const metadata = zml.moe.Metadata.init(.fromBackend(backend));
        const parameters = zml.moe.Parameters.init(
            .fromBackend(
                backend,
                @intCast(topk_ids.dim(.top_expert)),
                .gelu,
            ),
        );
        return zml.moe.forwardMoe(
            hidden,
            topk_ids,
            topk_weights,
            gate_up,
            null,
            null,
            down,
            null,
            null,
            null,
            metadata,
            parameters,
        ) catch |err| std.debug.panic(
            "FlashInfer CUTLASS backend graph construction failed: {}",
            .{err},
        );
    }

    fn tritonForward(
        hidden: zml.Tensor,
        topk_ids: zml.Tensor,
        topk_weights: zml.Tensor,
        gate_up: zml.Tensor,
        down: zml.Tensor,
    ) zml.Tensor {
        const backend: zml.moe.Backend = .triton;
        const metadata = zml.moe.Metadata.init(.fromBackend(backend));
        const parameters = zml.moe.Parameters.init(
            .fromBackend(
                backend,
                @intCast(topk_ids.dim(.top_expert)),
                .gelu,
            ),
        );
        return zml.moe.forwardMoe(
            hidden,
            topk_ids,
            topk_weights,
            gate_up,
            null,
            null,
            down,
            null,
            null,
            null,
            metadata,
            parameters,
        ) catch |err| std.debug.panic(
            "Triton backend graph construction failed: {}",
            .{err},
        );
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    if (args.tokens == 0) return error.ExpectedAtLeastOneToken;

    const platform: *zml.Platform = try .auto(allocator, io, .{
        .xla_gpu = .{
            .allocator = .{
                .bfc = .{
                    .preallocate = false,
                    .memory_fraction = 0.85,
                },
            },
        },
    });
    defer platform.deinit(allocator, io);

    if (platform.target != .cuda) return error.CudaRequired;
    if (!cutlass.isAvailable(platform)) {
        return error.FlashInferCutlassMoeUnavailable;
    }

    const devices = platform.pjrt_client.devices(platform.pjrt_api);
    const compute_capability =
        zml.platform.cuda.tryGetComputeCapabilities(platform, devices[0]) orelse
        return error.UnknownCudaComputeCapability;
    log.info(
        "testing Gemma 4 layer {d} on SM{s} with {d} random token(s)",
        .{ args.layer, compute_capability, args.tokens },
    );

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    var registry: zml.safetensors.TensorRegistry =
        try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const moe = try Gemma4Moe.init(store.view(), args.layer);
    const hidden_size = moe.gate_up.dim(.d);
    const num_experts = moe.gate_up.dim(.expert);
    const intermediate_size = moe.down.dim(.dout);
    log.info(
        "loaded graph metadata: {d} experts, hidden={d}, intermediate={d}, top-k={d}",
        .{
            num_experts,
            hidden_size,
            intermediate_size,
            moe.router.top_k_experts,
        },
    );

    const hidden: zml.Tensor = .init(
        .{ .b = 1, .s = args.tokens, .d = hidden_size },
        .bf16,
    );

    var router_exe = try platform.compileFn(
        allocator,
        io,
        Gemma4Router.forward,
        .{ moe.router, hidden },
        .{ .program_name = "gemma4_layer_router" },
    );
    defer router_exe.deinit();

    const topk_weights: zml.Tensor = .init(
        .{
            .b = 1,
            .s = args.tokens,
            .top_expert = moe.router.top_k_experts,
        },
        .f32,
    );
    const topk_ids: zml.Tensor = .init(
        .{
            .b = 1,
            .s = args.tokens,
            .top_expert = moe.router.top_k_experts,
        },
        .i32,
    );

    var cutlass_exe = try platform.compileFn(
        allocator,
        io,
        TestMoe.cutlassForward,
        .{ hidden, topk_ids, topk_weights, moe.gate_up, moe.down },
        .{ .program_name = "gemma4_flashinfer_cutlass_moe_bf16" },
    );
    defer cutlass_exe.deinit();
    var triton_exe = try platform.compileFn(
        allocator,
        io,
        TestMoe.tritonForward,
        .{ hidden, topk_ids, topk_weights, moe.gate_up, moe.down },
        .{ .program_name = "gemma4_triton_moe_bf16_reference" },
    );
    defer triton_exe.deinit();

    var moe_buffers = try zml.mem.bufferize(allocator, Gemma4Moe, &moe);
    var loader: zml.io.Loader = try .init(allocator, platform, .{
        .parallelism = 4,
        .dma_chunks = 2,
        .dma_chunk_size = 16 * zml.MiB,
    });
    defer loader.deinit();

    var progress = std.Progress.start(io, .{
        .root_name = "Loading Gemma 4 MoE layer",
    });
    loader.load(
        io,
        Gemma4Moe,
        &moe,
        &moe_buffers,
        &store,
        &.{platform.replicated_sharding},
        .{ .progress = &progress },
    );
    try loader.await(io);
    progress.end();
    defer Gemma4Moe.deinitBuffers(&moe_buffers);

    const hidden_host = try allocator.alloc(
        BFloat16,
        @intCast(args.tokens * @as(u32, @intCast(hidden_size))),
    );
    defer allocator.free(hidden_host);
    var prng = std.Random.DefaultPrng.init(args.seed);
    const random = prng.random();
    for (hidden_host) |*value| {
        value.* = BFloat16.fromF32(2 * random.float(f32) - 1);
    }

    var hidden_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        hidden.shape(),
        .replicated,
        std.mem.sliceAsBytes(hidden_host),
    );
    defer hidden_buffer.deinit();

    var routing = try zml.testing.autoCall(
        allocator,
        io,
        &router_exe,
        Gemma4Router.forward,
        .{ moe_buffers.router, hidden_buffer },
    );
    defer routing[0].deinit();
    defer routing[1].deinit();
    try validateRouting(
        allocator,
        io,
        routing[0],
        routing[1],
        @intCast(num_experts),
        moe.router.top_k_experts,
    );

    var cutlass_output = try zml.testing.autoCall(
        allocator,
        io,
        &cutlass_exe,
        TestMoe.cutlassForward,
        .{
            hidden_buffer,
            routing[1],
            routing[0],
            moe_buffers.gate_up,
            moe_buffers.down,
        },
    );
    defer cutlass_output.deinit();
    var triton_output = try zml.testing.autoCall(
        allocator,
        io,
        &triton_exe,
        TestMoe.tritonForward,
        .{
            hidden_buffer,
            routing[1],
            routing[0],
            moe_buffers.gate_up,
            moe_buffers.down,
        },
    );
    defer triton_output.deinit();

    try compareNonzeroOutputs(
        allocator,
        io,
        triton_output,
        cutlass_output,
    );
}

fn validateRouting(
    allocator: std.mem.Allocator,
    io: std.Io,
    weights_buffer: zml.Buffer,
    ids_buffer: zml.Buffer,
    num_experts: i32,
    top_k: u32,
) !void {
    var weights = try weights_buffer.toSliceAlloc(allocator, io);
    defer weights.free(allocator);
    var ids = try ids_buffer.toSliceAlloc(allocator, io);
    defer ids.free(allocator);

    const weights_data = weights.constItems(f32);
    const ids_data = ids.constItems(i32);
    if (weights_data.len != ids_data.len or
        weights_data.len == 0 or
        weights_data.len % top_k != 0)
    {
        return error.InvalidRoutingOutput;
    }

    for (ids_data, weights_data) |expert, weight| {
        if (expert < 0 or expert >= num_experts) {
            return error.InvalidRoutedExpert;
        }
        if (!std.math.isFinite(weight)) return error.NonFiniteRoutingWeight;
    }
    for (0..weights_data.len / top_k) |token| {
        var weight_magnitude: f32 = 0;
        const offset = token * top_k;
        for (weights_data[offset..][0..top_k]) |weight| {
            weight_magnitude += @abs(weight);
        }
        if (weight_magnitude == 0) return error.ZeroRoutingWeights;
    }

    log.info(
        "first token route: experts={any}, weights={any}",
        .{ ids_data[0..top_k], weights_data[0..top_k] },
    );
}

fn compareNonzeroOutputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    triton_buffer: zml.Buffer,
    cutlass_buffer: zml.Buffer,
) !void {
    var triton = try triton_buffer.toSliceAlloc(allocator, io);
    defer triton.free(allocator);
    var cutlass_output = try cutlass_buffer.toSliceAlloc(allocator, io);
    defer cutlass_output.free(allocator);

    const triton_data = triton.constItems(BFloat16);
    const cutlass_data = cutlass_output.constItems(BFloat16);
    if (triton_data.len != cutlass_data.len or triton_data.len == 0) {
        return error.InvalidMoeOutput;
    }

    const triton_stats = try outputStats(triton_data);
    const cutlass_stats = try outputStats(cutlass_data);
    if (triton_stats.max_abs <= 1e-6 or cutlass_stats.max_abs <= 1e-6) {
        return error.AllZeroMoeOutput;
    }

    log.info(
        "Triton output: max_abs={d:.6}, rms={d:.6}; CUTLASS output: max_abs={d:.6}, rms={d:.6}",
        .{
            triton_stats.max_abs,
            triton_stats.rms,
            cutlass_stats.max_abs,
            cutlass_stats.rms,
        },
    );

    const compare_opts: zml.testing.CompareOpts = .{
        .absolute_tolerance = 2e-2,
        .relative_tolerance = 2e-2,
        .epsilon_relative = 1e-3,
        .minimum_close_fraction = 0.99,
    };
    const report = try zml.testing.compareSlices(
        allocator,
        BFloat16,
        BFloat16,
        triton_data,
        cutlass_data,
        compare_opts,
    );
    log.info("Triton/CUTLASS comparison:\n{f}", .{report});
    if (report.nan_or_inf or
        report.close_fraction < compare_opts.minimum_close_fraction)
    {
        return error.MoeOutputMismatch;
    }
}

const OutputStats = struct {
    max_abs: f32,
    rms: f32,
};

fn outputStats(values: []const BFloat16) !OutputStats {
    var max_abs: f32 = 0;
    var sum_squared: f64 = 0;
    for (values) |value| {
        const value_f32 = value.toF32();
        if (!std.math.isFinite(value_f32)) return error.NonFiniteMoeOutput;
        max_abs = @max(max_abs, @abs(value_f32));
        sum_squared +=
            @as(f64, @floatCast(value_f32)) *
            @as(f64, @floatCast(value_f32));
    }
    return .{
        .max_abs = max_abs,
        .rms = @floatCast(std.math.sqrt(
            sum_squared / @as(f64, @floatFromInt(values.len)),
        )),
    };
}

test "BF16 backend matches Triton after routing" {
    @setEvalBranchQuota(10_000);
    if (comptime !platforms.isEnabled(.cuda)) return;

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda) return;
    const devices = platform.pjrt_client.devices(platform.pjrt_api);
    const cc = zml.platform.cuda.tryGetComputeCapabilities(platform, devices[0]) orelse
        return;
    const supported_architecture = std.mem.eql(u8, cc, "9.0") or
        std.mem.eql(u8, cc, "10.0") or
        std.mem.eql(u8, cc, "12.0");
    if (!supported_architecture) return;
    try std.testing.expect(cutlass.isAvailable(platform));

    try std.testing.expectEqual(
        zml.moe.Backend.flashinfer_cutlass,
        try zml.moe.Backend.auto(platform, .bf16),
    );
    const blackwell = std.mem.eql(u8, cc, "10.0") or std.mem.eql(u8, cc, "12.0");
    if (blackwell) {
        try std.testing.expectEqual(
            zml.moe.Backend.flashinfer_cutlass,
            try zml.moe.Backend.auto(platform, .f4e2m1),
        );
    }
    const tactics = try cutlass.tacticCounts(0, .bf16xbf16);
    try std.testing.expect(tactics.gemm1 > 0);
    try std.testing.expect(tactics.gemm2 > 0);

    const num_experts = 2;
    const num_tokens = 4;
    const hidden_size = 128;
    const intermediate_size = 128;

    const hidden = zml.Tensor.init(
        .{ .b = 1, .s = num_tokens, .d = hidden_size },
        .bf16,
    );
    const topk_ids = zml.Tensor.init(
        .{ .b = 1, .s = num_tokens, .top_expert = 1 },
        .i32,
    );
    const topk_weights = zml.Tensor.init(
        .{ .b = 1, .s = num_tokens, .top_expert = 1 },
        .f32,
    );
    const gate_up = zml.Tensor.init(
        .{
            .expert = num_experts,
            .dout = 2 * intermediate_size,
            .d = hidden_size,
        },
        .bf16,
    );
    const down = zml.Tensor.init(
        .{
            .expert = num_experts,
            .d = hidden_size,
            .dout = intermediate_size,
        },
        .bf16,
    );

    var cutlass_exe = try platform.compileFn(
        allocator,
        io,
        TestMoe.cutlassForward,
        .{ hidden, topk_ids, topk_weights, gate_up, down },
        .{ .program_name = "flashinfer_cutlass_moe_bf16_test" },
    );
    defer cutlass_exe.deinit();
    var triton_exe = try platform.compileFn(
        allocator,
        io,
        TestMoe.tritonForward,
        .{ hidden, topk_ids, topk_weights, gate_up, down },
        .{ .program_name = "triton_moe_bf16_reference" },
    );
    defer triton_exe.deinit();

    const hidden_host = try allocator.alloc(BFloat16, num_tokens * hidden_size);
    defer allocator.free(hidden_host);
    for (hidden_host, 0..) |*value, i| {
        const signed: i32 = @intCast(i % 17);
        value.* = BFloat16.fromF32(@as(f32, @floatFromInt(signed - 8)) / 16.0);
    }

    const gate_up_host = try allocator.alloc(
        BFloat16,
        num_experts * 2 * intermediate_size * hidden_size,
    );
    defer allocator.free(gate_up_host);
    @memset(gate_up_host, BFloat16.fromF32(0));
    for (0..num_experts) |expert| {
        const expert_offset = expert * 2 * intermediate_size * hidden_size;
        const expert_scale = 1.0 + @as(f32, @floatFromInt(expert)) * 0.25;
        for (0..intermediate_size) |i| {
            gate_up_host[expert_offset + i * hidden_size + i] =
                BFloat16.fromF32(0.5 * expert_scale);
            gate_up_host[
                expert_offset + (intermediate_size + i) * hidden_size + i
            ] = BFloat16.fromF32(0.25 * expert_scale);
        }
    }

    const down_host = try allocator.alloc(
        BFloat16,
        num_experts * hidden_size * intermediate_size,
    );
    defer allocator.free(down_host);
    @memset(down_host, BFloat16.fromF32(0));
    for (0..num_experts) |expert| {
        const expert_offset = expert * hidden_size * intermediate_size;
        for (0..hidden_size) |i| {
            down_host[expert_offset + i * intermediate_size + i] =
                BFloat16.fromF32(1);
        }
    }

    const ids_host = [num_tokens]i32{ 0, 1, 0, 1 };
    const weights_host = [num_tokens]f32{ 1, 1, 1, 1 };

    var hidden_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        hidden.shape(),
        .replicated,
        std.mem.sliceAsBytes(hidden_host),
    );
    defer hidden_buffer.deinit();
    var ids_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        topk_ids.shape(),
        .replicated,
        std.mem.asBytes(&ids_host),
    );
    defer ids_buffer.deinit();
    var weights_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        topk_weights.shape(),
        .replicated,
        std.mem.asBytes(&weights_host),
    );
    defer weights_buffer.deinit();
    var gate_up_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        gate_up.shape(),
        .replicated,
        std.mem.sliceAsBytes(gate_up_host),
    );
    defer gate_up_buffer.deinit();
    var down_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        down.shape(),
        .replicated,
        std.mem.sliceAsBytes(down_host),
    );
    defer down_buffer.deinit();

    var cutlass_output = try zml.testing.autoCall(
        allocator,
        io,
        &cutlass_exe,
        TestMoe.cutlassForward,
        .{
            hidden_buffer,
            ids_buffer,
            weights_buffer,
            gate_up_buffer,
            down_buffer,
        },
    );
    defer cutlass_output.deinit();
    var triton_output = try zml.testing.autoCall(
        allocator,
        io,
        &triton_exe,
        TestMoe.tritonForward,
        .{
            hidden_buffer,
            ids_buffer,
            weights_buffer,
            gate_up_buffer,
            down_buffer,
        },
    );
    defer triton_output.deinit();

    try zml.testing.expectClose(io, triton_output, cutlass_output, .{
        .absolute_tolerance = 2e-2,
        .relative_tolerance = 2e-2,
        .epsilon_relative = 1e-3,
        .minimum_close_fraction = 0.99,
    });
}
