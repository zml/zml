const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../zml.zig");
const stdx = zml.stdx;
pub const cutlass_flashinfer = @import("cutlass_flashinfer.zig");
pub const metal = @import("metal.zig");
pub const mosaic_tpu = @import("mosaic_tpu.zig");
pub const triton = @import("triton.zig");
pub const triton_kernels = @import("triton_kernels/triton_kernels.zig");

test {
    std.testing.refAllDecls(@This());
}

pub const ActivationMode = enum {
    silu,
    relu,
    gelu,
};

pub const Backend = enum {
    flashinfer_cutlass,
    triton,
    mosaic_tpu,
    metal,

    pub fn auto(platform: *const zml.Platform, weights_dtype: zml.DataType) !Backend {
        return switch (platform.target) {
            .cuda => switch (weights_dtype) {
                .bf16 => if (cutlass_flashinfer.isAvailable(platform))
                    .flashinfer_cutlass
                else
                    .triton,
                .f8e4m3fn => .triton,
                .f4e2m1 => if (cutlass_flashinfer.isNvfp4Supported(platform))
                    .flashinfer_cutlass
                else
                    return error.UnsupportedDataType,
                .f8e8m0, .f16, .f32 => .triton,
                else => error.UnsupportedDataType,
            },
            .rocm, .oneapi => switch (weights_dtype) {
                .bf16, .f16, .f32 => .triton,
                else => error.UnsupportedDataType,
            },
            .tpu => switch (weights_dtype) {
                .bf16, .f16, .f32 => .mosaic_tpu,
                else => error.UnsupportedDataType,
            },
            .metal => switch (weights_dtype) {
                .bf16, .f16, .f32, .f4e2m1, .u8, .f8e4m3fn => .metal,
                else => error.UnsupportedDataType,
            },
            else => error.UnimplementedMoEBackend,
        };
    }

    pub fn isAvailable(backend: Backend, platform: *const zml.Platform) bool {
        return switch (backend) {
            .flashinfer_cutlass => cutlass_flashinfer.isAvailable(platform),
            .triton => switch (platform.target) {
                .cuda, .rocm, .oneapi => true,
                else => false,
            },
            .mosaic_tpu => platform.target == .tpu,
            .metal => platform.target == .metal,
        };
    }

    pub fn register(backend: Backend, platform: *zml.Platform) !void {
        return switch (backend) {
            .flashinfer_cutlass => cutlass_flashinfer.register(platform),
            .triton => {},
            .mosaic_tpu => {},
            .metal => {},
        };
    }
};

test "Backend.auto selects Triton for CUDA E4M3FN experts" {
    var platform: zml.Platform = undefined;
    platform.target = .cuda;

    try std.testing.expectEqual(Backend.triton, try Backend.auto(&platform, .f8e4m3fn));
}

pub const Parameters = union(Backend) {
    flashinfer_cutlass: cutlass_flashinfer.Parameters,
    triton: triton.Parameters,
    mosaic_tpu: mosaic_tpu.Parameters,
    metal: metal.Parameters,

    pub const InitOptions = union(Backend) {
        flashinfer_cutlass: cutlass_flashinfer.Parameters.InitOptions,
        triton: triton.Parameters.InitOptions,
        mosaic_tpu: mosaic_tpu.Parameters.InitOptions,
        metal: metal.Parameters.InitOptions,

        pub fn fromBackend(backend: Backend, num_experts_per_tok: u32, activation: ActivationMode) InitOptions {
            return switch (backend) {
                .flashinfer_cutlass => .{ .flashinfer_cutlass = .{
                    .num_experts_per_tok = num_experts_per_tok,
                    .activation = switch (activation) {
                        .silu => .silu,
                        .relu => .relu,
                        .gelu => .gelu,
                    },
                } },
                .triton => .{ .triton = .{
                    .num_experts_per_tok = num_experts_per_tok,
                    .activation = switch (activation) {
                        .silu => .silu,
                        .relu => .relu,
                        .gelu => .gelu,
                    },
                } },
                .mosaic_tpu => .{ .mosaic_tpu = .{
                    .num_experts_per_tok = num_experts_per_tok,
                    .activation = switch (activation) {
                        .silu => .silu,
                        .relu => .relu,
                        .gelu => .gelu,
                    },
                } },
                .metal => .{ .metal = .{
                    .num_experts_per_tok = num_experts_per_tok,
                    .activation = switch (activation) {
                        .silu => .silu,
                        .relu => .relu,
                        .gelu => .gelu,
                    },
                } },
            };
        }
    };

    pub fn init(opts: InitOptions) Parameters {
        return switch (opts) {
            .flashinfer_cutlass => |v| .{ .flashinfer_cutlass = cutlass_flashinfer.Parameters.init(v) },
            .triton => |v| .{ .triton = triton.Parameters.init(v) },
            .mosaic_tpu => |v| .{ .mosaic_tpu = mosaic_tpu.Parameters.init(v) },
            .metal => |v| .{ .metal = metal.Parameters.init(v) },
        };
    }
};

pub const Metadata = union(Backend) {
    flashinfer_cutlass: cutlass_flashinfer.Metadata,
    triton: triton.Metadata,
    mosaic_tpu: mosaic_tpu.Metadata,
    metal: metal.Metadata,

    pub const InitOptions = union(Backend) {
        flashinfer_cutlass: cutlass_flashinfer.Metadata.InitOptions,
        triton: triton.Metadata.InitOptions,
        mosaic_tpu: mosaic_tpu.Metadata.InitOptions,
        metal: metal.Metadata.InitOptions,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .flashinfer_cutlass => .{ .flashinfer_cutlass = .{} },
                .triton => .{ .triton = .{} },
                .mosaic_tpu => .{ .mosaic_tpu = .{} },
                .metal => .{ .metal = .{} },
            };
        }
    };

    pub fn init(opts: InitOptions) Metadata {
        return switch (opts) {
            .flashinfer_cutlass => |v| .{ .flashinfer_cutlass = cutlass_flashinfer.Metadata.init(v) },
            .triton => |v| .{ .triton = triton.Metadata.init(v) },
            .mosaic_tpu => |v| .{ .mosaic_tpu = mosaic_tpu.Metadata.init(v) },
            .metal => |v| .{ .metal = metal.Metadata.init(v) },
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        return switch (self) {
            .flashinfer_cutlass => |metadata| .{ .flashinfer_cutlass = try metadata.initBuffer(io, platform) },
            .triton => |metadata| .{ .triton = try metadata.initBuffer(io, platform) },
            .mosaic_tpu => |metadata| .{ .mosaic_tpu = try metadata.initBuffer(io, platform) },
            .metal => |metadata| .{ .metal = try metadata.initBuffer(io, platform) },
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        switch (self.*) {
            .flashinfer_cutlass => |*metadata| cutlass_flashinfer.deinitBuffer(metadata),
            .triton => |*metadata| triton.deinitBuffer(metadata),
            .mosaic_tpu => |*metadata| mosaic_tpu.deinitBuffer(metadata),
            .metal => |*metadata| metal.deinitBuffer(metadata),
        }
    }
};

pub const Options = struct {
    activation_threshold: ?f32 = null,
    quant_scheme: ?zml.Quantization.Scheme = null,
};

fn hasOnlyUnshardedInnerAxes(tensor: zml.Tensor) bool {
    for (1..tensor.rank()) |axis| {
        switch (tensor.shape().partition(axis)) {
            .open, .replicated => {},
            .axis, .unknown => return false,
        }
    }
    return true;
}

fn hasOnlyUnshardedAxes(tensor: zml.Tensor) bool {
    for (0..tensor.rank()) |axis| {
        switch (tensor.shape().partition(axis)) {
            .open, .replicated => {},
            .axis, .unknown => return false,
        }
    }
    return true;
}

const TritonManual = struct {
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    weights_gate_up: zml.Tensor,
    weights_down: zml.Tensor,
    gate_up_scale: ?zml.Tensor,
    down_scale: ?zml.Tensor,
    activation: triton.Parameters.ActivationMode,
    global_num_experts: i64,
    quant_scheme: ?zml.Quantization.Scheme,
    activation_threshold: ?f32,

    fn expertMap(self: TritonManual) zml.Tensor {
        const local_num_experts = self.weights_gate_up.dim(.expert);
        const expert_start = zml.ops.partitionId().convert(.i32).scale(local_num_experts).convert(.i32);
        const global_expert_ids = zml.Tensor.arange(.{ .end = self.global_num_experts }, .i32).withTags(.{.expert});
        const local_expert_mask = global_expert_ids.cmp(.GE, expert_start)
            .logical(.AND, global_expert_ids.cmp(.LT, expert_start.addConstant(local_num_experts)));
        return local_expert_mask.select(global_expert_ids.sub(expert_start), zml.Tensor.scalar(-1, .i32));
    }

    fn forward(self: TritonManual, expert_map: ?zml.Tensor, prepared_a1: ?zml.quantization.QuantizedInput) zml.Tensor {
        const output = triton.fusedExpertsImpl(
            self.input,
            self.weights_gate_up,
            self.weights_down,
            self.topk_weights,
            self.topk_ids,
            .{},
            .{
                .activation = self.activation,
                .global_num_experts = self.global_num_experts,
                .expert_map = expert_map,
                .w1_scale = self.gate_up_scale,
                .w2_scale = self.down_scale,
                .quant_scheme = self.quant_scheme,
                .activation_threshold = self.activation_threshold,
                .prepared_a1 = prepared_a1,
            },
        ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});
        return output.reshape(self.input.shape().dims()).withTags(.{ .b, .s, .d });
    }

    fn expertParallel(self: TritonManual, _: zml.Shape) zml.Tensor {
        return zml.ops.allReduce(self.forward(self.expertMap(), null), zml.Tensor.add);
    }

    fn tensorParallel(self: TritonManual, _: zml.Shape) zml.Tensor {
        return zml.ops.allReduce(self.forward(null, null), zml.Tensor.add);
    }
};

/// Run a Triton expert-parallel MoE and combine an additional shard-local
/// contribution with its routed output before the single cross-rank sum.
///
/// `epilogue` contains the tensors and configuration needed by its
/// `forward(self, local_input, prepared_a1, local_routed) Tensor` method.
/// Its tensors are localized along with the MoE inputs before it runs.
///
/// This deliberately narrow entry point currently supports only block-FP8
/// Triton experts sharded on the `.experts` mesh.
pub fn forwardMoeWithReduceEpilogue(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    weights_gate_up: zml.Tensor,
    scales_gate_up: ?zml.Tensor,
    bias_gate_up: ?zml.Tensor,
    weights_down: zml.Tensor,
    scales_down: ?zml.Tensor,
    bias_down: ?zml.Tensor,
    w1_global_scale: ?zml.Tensor,
    w2_global_scale: ?zml.Tensor,
    opts: Options,
    metadata: Metadata,
    parameters: Parameters,
    epilogue: anytype,
) !zml.Tensor {
    const activation = switch (parameters) {
        .triton => |value| value.activation,
        else => return error.UnsupportedBackend,
    };
    switch (parameters) {
        .triton => if (metadata != .triton) return error.InvalidMetadata,
        else => unreachable,
    }
    if (!weights_gate_up.shape().partition(.expert).eql(.init(.experts)) or
        !weights_down.shape().partition(.expert).eql(.init(.experts)))
    {
        return error.ExpectedExpertParallelSharding;
    }
    if (!hasOnlyUnshardedInnerAxes(weights_gate_up) or
        !hasOnlyUnshardedInnerAxes(weights_down))
    {
        return error.UnsupportedTensorParallelSharding;
    }
    const gate_up_scale = scales_gate_up orelse return error.MissingWeightScale;
    const down_scale = scales_down orelse return error.MissingWeightScale;
    if (!gate_up_scale.shape().partition(.expert).eql(.init(.experts)) or
        !down_scale.shape().partition(.expert).eql(.init(.experts)))
    {
        return error.InconsistentExpertSharding;
    }
    if (!hasOnlyUnshardedInnerAxes(gate_up_scale) or
        !hasOnlyUnshardedInnerAxes(down_scale))
    {
        return error.UnsupportedTensorParallelSharding;
    }
    if (bias_gate_up != null or bias_down != null) return error.UnsupportedBias;
    if (w1_global_scale != null or w2_global_scale != null) return error.UnsupportedQuantization;
    if (opts.quant_scheme != .fp8_block128) return error.UnsupportedQuantization;

    return zml.ops.manualComputation(
        (struct {
            moe: TritonManual,
            epilogue: @TypeOf(epilogue),

            fn body(self: @This(), _: zml.Shape) zml.Tensor {
                const local_input = self.moe.input;
                const prepared_a1 = triton.prepareBlock128Fp8Activation(
                    local_input.reshape(.{
                        .token = local_input.dim(.b) * local_input.dim(.s),
                        .in = local_input.dim(.d),
                    }),
                    zml.Compiler.current().platform.target == .rocm,
                );
                const routed = self.moe.forward(self.moe.expertMap(), prepared_a1);
                const epilogue_input: zml.quantization.QuantizedInput = .{
                    .values = prepared_a1.values.reshape(local_input.shape().withDtype(prepared_a1.values.dtype())),
                    .scales = prepared_a1.scales.reshape(local_input.shape().setDim(.d, @divExact(local_input.dim(.d), 128)).withDtype(.f32)),
                };
                const combined = self.epilogue.forward(local_input, epilogue_input, routed);
                stdx.debug.assert(combined.shape().eql(routed.shape()), "MoE reduce epilogue returned shape {f}, expected {f}", .{ combined.shape(), routed.shape() });
                return zml.ops.allReduce(combined, zml.Tensor.add);
            }
        }).body,
        .{
            .moe = .{
                .input = input,
                .topk_ids = topk_ids,
                .topk_weights = topk_weights,
                .weights_gate_up = weights_gate_up,
                .weights_down = weights_down,
                .gate_up_scale = gate_up_scale,
                .down_scale = down_scale,
                .activation = activation,
                .global_num_experts = weights_gate_up.dim(.expert),
                .quant_scheme = opts.quant_scheme,
                .activation_threshold = opts.activation_threshold,
            },
            .epilogue = epilogue,
        },
        input.shape(),
    );
}

pub fn forwardMoe(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    opts: Options,
    metadata: Metadata,
    parameters: Parameters,
) !zml.Tensor {
    const gate_up_scheme: ?zml.Quantization.Scheme = if (gate_up.quantization) |q| q.scheme else null;
    const down_scheme: ?zml.Quantization.Scheme = if (down.quantization) |q| q.scheme else null;
    if (gate_up_scheme != down_scheme) return error.UnsupportedQuantization;

    const gate_up_scales: ?zml.Tensor = if (gate_up.quantization) |q| q.scales else null;
    const down_scales: ?zml.Tensor = if (down.quantization) |q| q.scales else null;
    const gate_up_global_scale: ?zml.Tensor = if (gate_up.quantization) |q| (if (q.global_scale) |scale| scale.asMultiplier() else null) else null;
    const down_global_scale: ?zml.Tensor = if (down.quantization) |q| (if (q.global_scale) |scale| scale.asMultiplier() else null) else null;
    const quant_scheme: ?zml.Quantization.Scheme = if (gate_up.quantization) |q| q.scheme else null;

    return switch (parameters) {
        .flashinfer_cutlass => b: {
            if (comptime !platforms.isEnabled(.cuda)) {
                return error.UnsupportedPlatform;
            }
            if (gate_up.bias != null or down.bias != null) {
                return error.UnsupportedBias;
            }

            const runner_options = try parameters.flashinfer_cutlass.runnerOptions();
            const expert_partition = gate_up.weight.shape().partition(.expert);

            if (quant_scheme != null and quant_scheme == .nvfp4) {
                const gate_up_weight_unpacked = unpackedWeight(gate_up);
                const down_weight_unpacked = unpackedWeight(down);

                // TODO(Corentin): Do error checking on nvfp4
                // Also, maybe pass `zml.nn.Linear` directly
                if (expert_partition.eql(.init(.experts))) {
                    break :b zml.ops.manualComputation(
                        (struct {
                            input: zml.Tensor,
                            topk_ids: zml.Tensor,
                            topk_weights: zml.Tensor,
                            gate_up_weight_unpacked: zml.Tensor,
                            down_weight_unpacked: zml.Tensor,
                            gate_up_input_scale: zml.Tensor,
                            gate_up_scales: zml.Tensor,
                            gate_up_global_scale: zml.Tensor,
                            down_input_scale: zml.Tensor,
                            down_scales: zml.Tensor,
                            down_global_scale: zml.Tensor,
                            activation: cutlass_flashinfer.Activation,
                            enable_pdl: bool,
                            gemm1_tactic: i32,
                            gemm2_tactic: i32,
                            workspace_query_device: i32,

                            fn body(
                                self: @This(),
                                _: zml.Shape,
                            ) zml.Tensor {
                                const local_num_experts = self.gate_up_weight_unpacked.dim(.expert);
                                const partition_id = zml.ops.partitionId().convert(.i32);
                                const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                                const expert_end = expert_start.addConstant(local_num_experts);

                                const local_route_mask = self.topk_ids
                                    .cmp(.GE, expert_start)
                                    .logical(.AND, self.topk_ids.cmp(.LT, expert_end));
                                const local_topk_ids = local_route_mask.select(
                                    self.topk_ids.sub(expert_start),
                                    zml.Tensor.scalar(0, .i32),
                                );
                                const local_topk_weights = local_route_mask.select(
                                    self.topk_weights,
                                    zml.Tensor.scalar(0, self.topk_weights.dtype()),
                                );

                                const local_output = cutlass_flashinfer.fusedExpertsNvfp4(
                                    self.input,
                                    self.gate_up_weight_unpacked,
                                    self.down_weight_unpacked,
                                    local_topk_weights,
                                    local_topk_ids,
                                    self.gate_up_input_scale,
                                    self.gate_up_scales,
                                    self.gate_up_global_scale,
                                    self.down_input_scale,
                                    self.down_scales,
                                    self.down_global_scale,
                                    .{
                                        .workspace_query_device = self.workspace_query_device,
                                        .activation = self.activation,
                                        .enable_pdl = self.enable_pdl,
                                        .gemm1_tactic = self.gemm1_tactic,
                                        .gemm2_tactic = self.gemm2_tactic,
                                    },
                                ) catch |err| stdx.debug.panic(
                                    "FlashInfer CUTLASS NVFP4 MoE backend failed: {}",
                                    .{err},
                                );
                                const local_reshaped = local_output
                                    .reshape(self.input.shape().dims())
                                    .withTags(.{ .b, .s, .d });
                                return zml.ops.allReduce(local_reshaped, zml.Tensor.add);
                            }
                        }).body,
                        .{
                            .input = input,
                            .topk_ids = topk_ids,
                            .topk_weights = topk_weights,
                            .gate_up_weight_unpacked = gate_up_weight_unpacked,
                            .down_weight_unpacked = down_weight_unpacked,
                            .gate_up_input_scale = gate_up.quantization.?.input_scale.?.asMultiplier(),
                            .gate_up_scales = gate_up.quantization.?.scales,
                            .gate_up_global_scale = gate_up.quantization.?.global_scale.?.asMultiplier(),
                            .down_input_scale = down.quantization.?.input_scale.?.asMultiplier(),
                            .down_scales = down.quantization.?.scales,
                            .down_global_scale = down.quantization.?.global_scale.?.asMultiplier(),
                            .activation = runner_options.activation,
                            .enable_pdl = runner_options.enable_pdl,
                            .gemm1_tactic = runner_options.gemm1_tactic,
                            .gemm2_tactic = runner_options.gemm2_tactic,
                            .workspace_query_device = runner_options.workspace_query_device,
                        },
                        input.shape(),
                    );
                }

                break :b try cutlass_flashinfer.fusedExpertsNvfp4(
                    input,
                    gate_up_weight_unpacked,
                    down_weight_unpacked,
                    topk_weights,
                    topk_ids,
                    gate_up.quantization.?.input_scale.?.asMultiplier(),
                    gate_up.quantization.?.scales,
                    gate_up.quantization.?.global_scale.?.asMultiplier(),
                    down.quantization.?.input_scale.?.asMultiplier(),
                    down.quantization.?.scales,
                    down.quantization.?.global_scale.?.asMultiplier(),
                    runner_options,
                );
            }

            if (expert_partition.eql(.init(.experts))) {
                break :b zml.ops.manualComputation(
                    (struct {
                        input: zml.Tensor,
                        topk_ids: zml.Tensor,
                        topk_weights: zml.Tensor,
                        weights_gate_up: zml.Tensor,
                        weights_down: zml.Tensor,
                        activation: cutlass_flashinfer.Activation,
                        enable_pdl: bool,
                        gemm1_tactic: i32,
                        gemm2_tactic: i32,
                        workspace_query_device: i32,

                        fn body(
                            self: @This(),
                            _: zml.Shape,
                        ) zml.Tensor {
                            const local_num_experts = self.weights_gate_up.dim(.expert);
                            const partition_id = zml.ops.partitionId().convert(.i32);
                            const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                            const expert_end = expert_start.addConstant(local_num_experts);

                            const local_route_mask = self.topk_ids
                                .cmp(.GE, expert_start)
                                .logical(.AND, self.topk_ids.cmp(.LT, expert_end));
                            const local_topk_ids = local_route_mask.select(
                                self.topk_ids.sub(expert_start),
                                zml.Tensor.scalar(0, .i32),
                            );
                            const local_topk_weights = local_route_mask.select(
                                self.topk_weights,
                                zml.Tensor.scalar(0, self.topk_weights.dtype()),
                            );

                            const local_output = cutlass_flashinfer.fusedExpertsBf16(
                                self.input,
                                self.weights_gate_up,
                                self.weights_down,
                                local_topk_weights,
                                local_topk_ids,
                                .{
                                    .workspace_query_device = self.workspace_query_device,
                                    .activation = self.activation,
                                    .enable_pdl = self.enable_pdl,
                                    .gemm1_tactic = self.gemm1_tactic,
                                    .gemm2_tactic = self.gemm2_tactic,
                                },
                            ) catch |err| stdx.debug.panic(
                                "FlashInfer CUTLASS MoE backend failed: {}",
                                .{err},
                            );
                            const local_reshaped = local_output
                                .reshape(self.input.shape().dims())
                                .withTags(.{ .b, .s, .d });
                            return zml.ops.allReduce(local_reshaped, zml.Tensor.add);
                        }
                    }).body,
                    .{
                        .input = input,
                        .topk_ids = topk_ids,
                        .topk_weights = topk_weights,
                        .weights_gate_up = gate_up.weight,
                        .weights_down = down.weight,
                        .activation = runner_options.activation,
                        .enable_pdl = runner_options.enable_pdl,
                        .gemm1_tactic = runner_options.gemm1_tactic,
                        .gemm2_tactic = runner_options.gemm2_tactic,
                        .workspace_query_device = runner_options.workspace_query_device,
                    },
                    input.shape(),
                );
            }

            break :b try cutlass_flashinfer.fusedExpertsBf16(
                input,
                gate_up.weight,
                down.weight,
                topk_weights,
                topk_ids,
                runner_options,
            );
        },
        .triton => b: {
            const triton_metadata = switch (metadata) {
                .triton => |v| v,
                else => return error.InvalidMetadata,
            };

            const weights_gate_up = gate_up.weight;
            const weights_down = down.weight;
            const scales_gate_up = gate_up_scales;
            const scales_down = down_scales;
            const bias_gate_up = gate_up.bias;
            const bias_down = down.bias;
            const global_num_experts = weights_gate_up.dim(.expert);
            const expert_partition = weights_gate_up.shape().partition(.expert);
            const model_partition = zml.Shape.PartitionSpec.init(.model);
            const gate_up_tensor_parallel = weights_gate_up.shape().partition(1).eql(model_partition);
            const down_tensor_parallel = weights_down.shape().partition(2).eql(model_partition);

            if (gate_up_tensor_parallel != down_tensor_parallel) {
                return error.InconsistentTensorParallelSharding;
            }

            if (gate_up_tensor_parallel) {
                if (bias_gate_up != null or bias_down != null) return error.UnsupportedBias;
                if ((scales_gate_up == null) != (scales_down == null)) return error.MissingWeightScale;

                break :b zml.ops.manualComputation(
                    TritonManual.tensorParallel,
                    .{
                        .input = input,
                        .topk_ids = topk_ids,
                        .topk_weights = topk_weights,
                        .weights_gate_up = weights_gate_up,
                        .weights_down = weights_down,
                        .gate_up_scale = scales_gate_up,
                        .down_scale = scales_down,
                        .activation = parameters.triton.activation,
                        .global_num_experts = global_num_experts,
                        .quant_scheme = quant_scheme,
                        .activation_threshold = opts.activation_threshold,
                    },
                    input.shape(),
                );
            }

            if (!expert_partition.eql(.init(.experts))) {
                break :b try triton.fusedExpertsImpl(
                    input,
                    gate_up.weight,
                    down.weight,
                    topk_weights,
                    topk_ids,
                    triton_metadata,
                    .{
                        .activation = parameters.triton.activation,
                        .global_num_experts = global_num_experts,
                        .w1_scale = gate_up_scales,
                        .w2_scale = down_scales,
                        .w1_bias = gate_up.bias,
                        .w2_bias = down.bias,
                        .quant_scheme = quant_scheme,
                        .activation_threshold = opts.activation_threshold,
                    },
                );
            }

            if (bias_gate_up != null or bias_down != null) return error.UnsupportedBias;
            break :b zml.ops.manualComputation(
                TritonManual.expertParallel,
                .{
                    .input = input,
                    .topk_ids = topk_ids,
                    .topk_weights = topk_weights,
                    .weights_gate_up = weights_gate_up,
                    .weights_down = weights_down,
                    .gate_up_scale = scales_gate_up,
                    .down_scale = scales_down,
                    .activation = parameters.triton.activation,
                    .global_num_experts = global_num_experts,
                    .quant_scheme = quant_scheme,
                    .activation_threshold = opts.activation_threshold,
                },
                input.shape(),
            );
        },
        .mosaic_tpu => b: {
            const tpu_metadata = switch (metadata) {
                .mosaic_tpu => |v| v,
                else => return error.InvalidMetadata,
            };

            const expert_partition = gate_up.weight.shape().partition(.expert);

            if (expert_partition.eql(.init(.experts))) {
                const global_num_experts = down.weight.dim(.expert);
                const partial_output = zml.ops.manualComputation(
                    (struct {
                        input: zml.Tensor,
                        topk_ids: zml.Tensor,
                        topk_weights: zml.Tensor,
                        weights_gate_up: zml.Tensor,
                        weights_down: zml.Tensor,
                        activation: mosaic_tpu.ActivationMode,
                        global_num_experts: i64,
                        gate_up_scales: ?zml.Tensor,
                        bias_gate_up: ?zml.Tensor,
                        down_scales: ?zml.Tensor,
                        bias_down: ?zml.Tensor,

                        fn body(self: @This(), _: zml.Shape) zml.Tensor {
                            const local_num_experts = self.weights_gate_up.dim(.expert);
                            const partition_id = zml.ops.partitionId().convert(.i32);
                            const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                            const global_expert_ids = zml.Tensor.arange(.{ .end = self.global_num_experts }, .i32).withTags(.{.expert});

                            const local_expert_mask = global_expert_ids.cmp(.GE, expert_start)
                                .logical(.AND, global_expert_ids.cmp(.LT, expert_start.addConstant(local_num_experts)));
                            const expert_map = local_expert_mask.select(
                                global_expert_ids.sub(expert_start),
                                zml.Tensor.scalar(-1, .i32),
                            );
                            const local_output = mosaic_tpu.fusedExpertsImpl(
                                self.input,
                                self.weights_gate_up,
                                self.weights_down,
                                self.topk_weights,
                                self.topk_ids,
                                .{},
                                .{
                                    .activation = self.activation,
                                    .global_num_experts = self.global_num_experts,
                                    .expert_map = expert_map,
                                    .w1_scale = self.gate_up_scales,
                                    .w2_scale = self.down_scales,
                                    .w1_bias = self.bias_gate_up,
                                    .w2_bias = self.bias_down,
                                },
                            ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});
                            return local_output.reshape(self.input.shape().dims()).withTags(.{ .b, .s, .d });
                        }
                    }).body,
                    .{
                        .input = input,
                        .topk_ids = topk_ids,
                        .topk_weights = topk_weights,
                        .weights_gate_up = gate_up.weight,
                        .weights_down = down.weight,
                        .activation = parameters.mosaic_tpu.activation,
                        .global_num_experts = global_num_experts,
                        .gate_up_scales = gate_up_scales,
                        .bias_gate_up = gate_up.bias,
                        .down_scales = down_scales,
                        .bias_down = down.bias,
                    },
                    input.shape(),
                );
                break :b zml.ops.allReduce(partial_output, zml.Tensor.add);
            }

            break :b try mosaic_tpu.fusedExpertsImpl(
                input,
                gate_up.weight,
                down.weight,
                topk_weights,
                topk_ids,
                tpu_metadata,
                .{
                    .activation = parameters.mosaic_tpu.activation,
                    .global_num_experts = gate_up.weight.dim(.expert),
                    .w1_scale = gate_up_scales,
                    .w2_scale = down_scales,
                    .w1_bias = gate_up.bias,
                    .w2_bias = down.bias,
                },
            );
        },
        .metal => b: {
            const gate_up_weight_unpacked = unpackedWeight(gate_up);
            const down_weight_unpacked = unpackedWeight(down);
            const metal_metadata = switch (metadata) {
                .metal => |v| v,
                else => return error.InvalidMetadata,
            };

            break :b try metal.fusedExpertsImpl(
                input,
                gate_up_weight_unpacked,
                down_weight_unpacked,
                topk_weights,
                topk_ids,
                metal_metadata,
                .{
                    .activation = parameters.metal.activation,
                    .global_num_experts = gate_up_weight_unpacked.dim(.expert),
                    .w1_scale = gate_up_scales,
                    .w2_scale = down_scales,
                    .w1_global_scale = gate_up_global_scale,
                    .w2_global_scale = down_global_scale,
                    .w1_bias = gate_up.bias,
                    .w2_bias = down.bias,
                },
            );
        },
    };
}

fn unpackedWeight(linear: zml.nn.Linear) zml.Tensor {
    const quantization = linear.quantization orelse return linear.weight;
    return if (zml.nn.isPackedFp4(quantization.scheme, linear.weight.dtype()))
        zml.nn.unpackFp4(linear.weight, linear.tag, linear.tag)
    else
        linear.weight;
}
