const std = @import("std");

const zml = @import("zml");
const moe = @import("kimi_k3/moe.zig");
const primitives = @import("kimi_k3/primitives.zig");
const support = @import("kimi_k3_layer0_tests.zig");

comptime {
    @setEvalBranchQuota(500_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_moe_tests --fixture=<selected-moe-reference.safetensors>
        \\
        \\Run the bounded selected-expert Gate B differential on NVIDIA CUDA only.
        \\
    ;
};

const FixtureWeights = struct {
    experts: moe.ExpertBank,
    dense: moe.DenseWeights,
};

const ProbeInputs = struct {
    w13: zml.Tensor,
    w2: zml.Tensor,
};

const Result = struct {
    routed_down: zml.Tensor,
    route_outputs: zml.Tensor,
    combined_latent: zml.Tensor,
    routed_norm: zml.Tensor,
    routed_up: zml.Tensor,
    shared_gate: zml.Tensor,
    shared_up: zml.Tensor,
    shared_situ: zml.Tensor,
    shared_output: zml.Tensor,
    final: zml.Tensor,
    probe_w1: zml.Tensor,
    probe_w2: zml.Tensor,
    probe_w3: zml.Tensor,
    partitioned_route_outputs: zml.Tensor,
    partitioned_combined_latent: zml.Tensor,
};

fn partitionedBankLinear(
    input: zml.Tensor,
    expert_ids: zml.Tensor,
    bank: moe.Mxfp4Bank,
    comptime output_tag: zml.Shape.Tag,
) zml.Tensor {
    const expert_count = bank.values.dim(.expert);
    const split = @divTrunc(expert_count, 2);
    const right_count = expert_count - split;
    const left_valid = expert_ids.cmp(.LT, zml.Tensor.scalar(split, expert_ids.dtype()));
    const right_valid = expert_ids.cmp(.GE, zml.Tensor.scalar(split, expert_ids.dtype()));
    const left_ids = left_valid.select(expert_ids, zml.Tensor.scalar(split, expert_ids.dtype()));
    const right_local = expert_ids.addConstant(-split);
    const right_ids = right_valid.select(right_local, zml.Tensor.scalar(right_count, expert_ids.dtype()));
    const left_bank: moe.Mxfp4Bank = .{
        .values = bank.values.slice1d(.expert, .{ .end = split }),
        .scale = bank.scale.slice1d(.expert, .{ .end = split }),
    };
    const right_bank: moe.Mxfp4Bank = .{
        .values = bank.values.slice1d(.expert, .{ .start = split }),
        .scale = bank.scale.slice1d(.expert, .{ .start = split }),
    };
    return moe.nativeBankLinear(input, left_ids, left_bank, output_tag)
        .add(moe.nativeBankLinear(input, right_ids, right_bank, output_tag));
}

fn forwardSelectedImpl(
    hidden: zml.Tensor,
    local_route_ids: zml.Tensor,
    route_weights: zml.Tensor,
    weights: FixtureWeights,
    probes: ProbeInputs,
    comptime check_partitions: bool,
) Result {
    const token_hidden = hidden.merge(.{ .token = .{ .b, .s } });
    const local_ids = local_route_ids.merge(.{ .token = .{ .b, .s } });
    const weights_flat = route_weights.merge(.{ .token = .{ .b, .s } });
    const routed_down = token_hidden.dot(weights.dense.routed_down, .d);
    const expert_input = routed_down.rename(.{ .latent = .d });
    const gate = moe.nativeBankLinear(expert_input, local_ids, weights.experts.w1, zml.Shape.toTag(.intermediate));
    const up = moe.nativeBankLinear(expert_input, local_ids, weights.experts.w3, zml.Shape.toTag(.intermediate));
    const activated = primitives.situGlu(gate, up);
    const route_outputs = moe.nativeBankLinear(
        activated.rename(.{ .intermediate = .d }),
        local_ids,
        weights.experts.w2,
        zml.Shape.toTag(.latent),
    );
    const partitioned_gate = if (check_partitions) partitionedBankLinear(expert_input, local_ids, weights.experts.w1, zml.Shape.toTag(.intermediate)) else gate;
    const partitioned_up = if (check_partitions) partitionedBankLinear(expert_input, local_ids, weights.experts.w3, zml.Shape.toTag(.intermediate)) else up;
    const partitioned_activated = primitives.situGlu(partitioned_gate, partitioned_up);
    const partitioned_route_outputs = if (check_partitions) partitionedBankLinear(
        partitioned_activated.rename(.{ .intermediate = .d }),
        local_ids,
        weights.experts.w2,
        zml.Shape.toTag(.latent),
    ) else route_outputs;
    const partitioned_combined_latent = partitioned_route_outputs.convert(.f32)
        .mul(weights_flat.convert(.f32).broad(partitioned_route_outputs.shape()))
        .sum(.route).squeeze(.route).convert(hidden.dtype());
    const combined_latent = route_outputs.convert(.f32)
        .mul(weights_flat.convert(.f32).broad(route_outputs.shape()))
        .sum(.route).squeeze(.route).convert(hidden.dtype());
    const routed = moe.finishRouted(combined_latent, weights.dense);
    const shared = moe.sharedMlp(token_hidden, weights.dense);
    const final = routed.output.add(shared.output).reshape(.{
        .b = hidden.dim(.b),
        .s = hidden.dim(.s),
        .d = hidden.dim(.d),
    });
    return .{
        .routed_down = routed_down,
        .route_outputs = route_outputs,
        .combined_latent = combined_latent,
        .routed_norm = routed.normalized,
        .routed_up = routed.output,
        .shared_gate = shared.gate,
        .shared_up = shared.up,
        .shared_situ = shared.activated,
        .shared_output = shared.output,
        .final = final,
        .probe_w1 = moe.probeLinear(probes.w13, weights.experts.w1),
        .probe_w2 = moe.probeLinear(probes.w2, weights.experts.w2),
        .probe_w3 = moe.probeLinear(probes.w13, weights.experts.w3),
        .partitioned_route_outputs = partitioned_route_outputs,
        .partitioned_combined_latent = partitioned_combined_latent,
    };
}

fn forwardSelectedNativeOnly(
    hidden: zml.Tensor,
    local_route_ids: zml.Tensor,
    route_weights: zml.Tensor,
    weights: FixtureWeights,
    probes: ProbeInputs,
) Result {
    return forwardSelectedImpl(hidden, local_route_ids, route_weights, weights, probes, false);
}

fn forwardSelected(
    hidden: zml.Tensor,
    local_route_ids: zml.Tensor,
    route_weights: zml.Tensor,
    weights: FixtureWeights,
    probes: ProbeInputs,
) Result {
    return forwardSelectedImpl(hidden, local_route_ids, route_weights, weights, probes, true);
}

const fp32_tolerance: zml.testing.CompareOpts = .{
    .absolute_tolerance = 2e-4,
    .relative_tolerance = 2e-4,
    .minimum_close_fraction = 1.0,
};

fn load(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    key: []const u8,
    tags: anytype,
    sharding: zml.Sharding,
) !zml.Buffer {
    return support.loadBuffer(allocator, io, platform, store, key, tags, sharding);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.85 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;
    const sharding = platform.replicated_sharding;

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.fixture);
    defer registry.deinit();
    var tensor_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer tensor_store.deinit();
    const store = tensor_store.view();

    var hidden = try load(allocator, io, platform, store, "moe.input", .{ .b, .s, .d }, sharding);
    defer hidden.deinit();
    var local_ids = try load(allocator, io, platform, store, "moe.local_route_ids", .{ .b, .s, .route }, sharding);
    defer local_ids.deinit();
    var route_weights = try load(allocator, io, platform, store, "moe.route_weights", .{ .b, .s, .route }, sharding);
    defer route_weights.deinit();
    var weights: zml.Bufferized(FixtureWeights) = .{
        .experts = .{
            .w1 = .{
                .values = try load(allocator, io, platform, store, "selected.w1.packed", .{ .expert, .intermediate, .kw }, sharding),
                .scale = try load(allocator, io, platform, store, "selected.w1.scale", .{ .expert, .intermediate, .block }, sharding),
            },
            .w2 = .{
                .values = try load(allocator, io, platform, store, "selected.w2.packed", .{ .expert, .latent, .kw }, sharding),
                .scale = try load(allocator, io, platform, store, "selected.w2.scale", .{ .expert, .latent, .block }, sharding),
            },
            .w3 = .{
                .values = try load(allocator, io, platform, store, "selected.w3.packed", .{ .expert, .intermediate, .kw }, sharding),
                .scale = try load(allocator, io, platform, store, "selected.w3.scale", .{ .expert, .intermediate, .block }, sharding),
            },
        },
        .dense = .{
            .routed_down = try load(allocator, io, platform, store, "dense.routed_down", .{ .latent, .d }, sharding),
            .routed_norm = try load(allocator, io, platform, store, "dense.routed_norm", .{.latent}, sharding),
            .routed_up = try load(allocator, io, platform, store, "dense.routed_up", .{ .d, .latent }, sharding),
            .shared_gate = try load(allocator, io, platform, store, "dense.shared_gate", .{ .intermediate, .d }, sharding),
            .shared_up = try load(allocator, io, platform, store, "dense.shared_up", .{ .intermediate, .d }, sharding),
            .shared_down = try load(allocator, io, platform, store, "dense.shared_down", .{ .d, .intermediate }, sharding),
        },
    };
    defer zml.Buffer.deinitAll(FixtureWeights, &weights);
    var probes: zml.Bufferized(ProbeInputs) = .{
        .w13 = try load(allocator, io, platform, store, "probe.w13.input", .{ .expert, .d }, sharding),
        .w2 = try load(allocator, io, platform, store, "probe.w2.input", .{ .expert, .d }, sharding),
    };
    defer zml.Buffer.deinitAll(ProbeInputs, &probes);

    const weight_tensors: FixtureWeights = .{
        .experts = .{
            .w1 = .{ .values = .fromShape(weights.experts.w1.values.shape()), .scale = .fromShape(weights.experts.w1.scale.shape()) },
            .w2 = .{ .values = .fromShape(weights.experts.w2.values.shape()), .scale = .fromShape(weights.experts.w2.scale.shape()) },
            .w3 = .{ .values = .fromShape(weights.experts.w3.values.shape()), .scale = .fromShape(weights.experts.w3.scale.shape()) },
        },
        .dense = .{
            .routed_down = .fromShape(weights.dense.routed_down.shape()),
            .routed_norm = .fromShape(weights.dense.routed_norm.shape()),
            .routed_up = .fromShape(weights.dense.routed_up.shape()),
            .shared_gate = .fromShape(weights.dense.shared_gate.shape()),
            .shared_up = .fromShape(weights.dense.shared_up.shape()),
            .shared_down = .fromShape(weights.dense.shared_down.shape()),
        },
    };
    const probe_tensors: ProbeInputs = .{
        .w13 = .fromShape(probes.w13.shape()),
        .w2 = .fromShape(probes.w2.shape()),
    };
    const compile_started = std.Io.Clock.now(.real, io).toNanoseconds();
    const exe = try platform.compileFn(
        allocator,
        io,
        forwardSelectedNativeOnly,
        .{
            zml.Tensor.fromShape(hidden.shape()),
            zml.Tensor.fromShape(local_ids.shape()),
            zml.Tensor.fromShape(route_weights.shape()),
            weight_tensors,
            probe_tensors,
        },
        .{ .shardings = &.{sharding} },
    );
    defer exe.deinit();
    const compile_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - compile_started, 1000);
    const cold_execute_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var cold_actual = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        forwardSelectedNativeOnly,
        .{ hidden, local_ids, route_weights, weights, probes },
    );
    const cold_execute_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - cold_execute_started, 1000);
    zml.Buffer.deinitAll(Result, &cold_actual);
    const execute_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var actual = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        forwardSelectedNativeOnly,
        .{ hidden, local_ids, route_weights, weights, probes },
    );
    defer zml.Buffer.deinitAll(Result, &actual);
    const execute_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - execute_started, 1000);

    const boundaries = [_][]const u8{
        "moe.routed_down", "moe.route_outputs", "moe.combined_latent",
        "moe.routed_norm", "moe.routed_up",     "moe.shared_gate",
        "moe.shared_up",   "moe.shared_situ",   "moe.shared_output",
        "moe.final",
    };
    const values = .{
        actual.routed_down, actual.route_outputs, actual.combined_latent,
        actual.routed_norm, actual.routed_up,     actual.shared_gate,
        actual.shared_up,   actual.shared_situ,   actual.shared_output,
        actual.final,
    };
    inline for (boundaries, values) |key, value| {
        try support.compare(allocator, io, platform, store, key, value, support.bf16_tolerance, sharding);
    }
    try support.compare(allocator, io, platform, store, "probe.w1.output", actual.probe_w1, fp32_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "probe.w2.output", actual.probe_w2, fp32_tolerance, sharding);
    try support.compare(allocator, io, platform, store, "probe.w3.output", actual.probe_w3, fp32_tolerance, sharding);
    const partition_compile_started = std.Io.Clock.now(.real, io).toNanoseconds();
    const partition_exe = try platform.compileFn(
        allocator,
        io,
        forwardSelected,
        .{
            zml.Tensor.fromShape(hidden.shape()),
            zml.Tensor.fromShape(local_ids.shape()),
            zml.Tensor.fromShape(route_weights.shape()),
            weight_tensors,
            probe_tensors,
        },
        .{ .shardings = &.{sharding} },
    );
    defer partition_exe.deinit();
    const partition_compile_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - partition_compile_started, 1000);
    const partition_execute_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var partition_actual = try zml.testing.autoCall(
        allocator,
        io,
        &partition_exe,
        forwardSelected,
        .{ hidden, local_ids, route_weights, weights, probes },
    );
    defer zml.Buffer.deinitAll(Result, &partition_actual);
    const partition_execute_us = @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - partition_execute_started, 1000);
    try zml.testing.expectClose(io, partition_actual.route_outputs, partition_actual.partitioned_route_outputs, .exact_match);
    try zml.testing.expectClose(io, partition_actual.combined_latent, partition_actual.partitioned_combined_latent, .exact_match);

    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    try stdout_file.interface.print(
        "KIMI_K3_MOE_PASS experts=61 routes=64 matrices=183 boundaries=13 partition_shards=2 partition_exact=true compile_us={} cold_execute_us={} execute_us={} partition_compile_us={} partition_execute_us={}\n",
        .{ compile_us, cold_execute_us, execute_us, partition_compile_us, partition_execute_us },
    );
    try stdout_file.interface.flush();
}
