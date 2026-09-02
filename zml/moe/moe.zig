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
};

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
                        .{
                            input,
                            topk_ids,
                            topk_weights,
                            gate_up_weight_unpacked,
                            down_weight_unpacked,
                            gate_up.quantization.?.input_scale.?.asMultiplier(),
                            gate_up.quantization.?.scales,
                            gate_up.quantization.?.global_scale.?.asMultiplier(),
                            down.quantization.?.input_scale.?.asMultiplier(),
                            down.quantization.?.scales,
                            down.quantization.?.global_scale.?.asMultiplier(),
                        },
                        input.shape(),
                        .{
                            .activation = runner_options.activation,
                            .enable_pdl = runner_options.enable_pdl,
                            .gemm1_tactic = runner_options.gemm1_tactic,
                            .gemm2_tactic = runner_options.gemm2_tactic,
                            .workspace_query_device = runner_options.workspace_query_device,
                        },
                        (struct {
                            fn body(
                                ctx: anytype,
                                _: std.mem.Allocator,
                                sharded_inputs: []const zml.Tensor,
                                _: zml.Shape,
                            ) zml.Tensor {
                                const local_num_experts = sharded_inputs[3].dim(.expert);
                                const partition_id = zml.ops.partitionId().convert(.i32);
                                const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                                const expert_end = expert_start.addConstant(local_num_experts);

                                const local_route_mask = sharded_inputs[1]
                                    .cmp(.GE, expert_start)
                                    .logical(.AND, sharded_inputs[1].cmp(.LT, expert_end));
                                const local_topk_ids = local_route_mask.select(
                                    sharded_inputs[1].sub(expert_start),
                                    zml.Tensor.scalar(0, .i32),
                                );
                                const local_topk_weights = local_route_mask.select(
                                    sharded_inputs[2],
                                    zml.Tensor.scalar(0, sharded_inputs[2].dtype()),
                                );

                                const local_output = cutlass_flashinfer.fusedExpertsNvfp4(
                                    sharded_inputs[0],
                                    sharded_inputs[3],
                                    sharded_inputs[4],
                                    local_topk_weights,
                                    local_topk_ids,
                                    sharded_inputs[5],
                                    sharded_inputs[6],
                                    sharded_inputs[7],
                                    sharded_inputs[8],
                                    sharded_inputs[9],
                                    sharded_inputs[10],
                                    .{
                                        .workspace_query_device = ctx.workspace_query_device,
                                        .activation = ctx.activation,
                                        .enable_pdl = ctx.enable_pdl,
                                        .gemm1_tactic = ctx.gemm1_tactic,
                                        .gemm2_tactic = ctx.gemm2_tactic,
                                    },
                                ) catch |err| stdx.debug.panic(
                                    "FlashInfer CUTLASS NVFP4 MoE backend failed: {}",
                                    .{err},
                                );
                                const local_reshaped = local_output
                                    .reshape(sharded_inputs[0].shape().dims())
                                    .withTags(.{ .b, .s, .d });
                                return zml.ops.allReduce(local_reshaped, zml.Tensor.add);
                            }
                        }).body,
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
                    .{ input, topk_ids, topk_weights, gate_up.weight, down.weight },
                    input.shape(),
                    .{
                        .activation = runner_options.activation,
                        .enable_pdl = runner_options.enable_pdl,
                        .gemm1_tactic = runner_options.gemm1_tactic,
                        .gemm2_tactic = runner_options.gemm2_tactic,
                        .workspace_query_device = runner_options.workspace_query_device,
                    },
                    (struct {
                        fn body(
                            ctx: anytype,
                            _: std.mem.Allocator,
                            sharded_inputs: []const zml.Tensor,
                            _: zml.Shape,
                        ) zml.Tensor {
                            const local_num_experts = sharded_inputs[3].dim(.expert);
                            const partition_id = zml.ops.partitionId().convert(.i32);
                            const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                            const expert_end = expert_start.addConstant(local_num_experts);

                            const local_route_mask = sharded_inputs[1]
                                .cmp(.GE, expert_start)
                                .logical(.AND, sharded_inputs[1].cmp(.LT, expert_end));
                            const local_topk_ids = local_route_mask.select(
                                sharded_inputs[1].sub(expert_start),
                                zml.Tensor.scalar(0, .i32),
                            );
                            const local_topk_weights = local_route_mask.select(
                                sharded_inputs[2],
                                zml.Tensor.scalar(0, sharded_inputs[2].dtype()),
                            );

                            const local_output = cutlass_flashinfer.fusedExpertsBf16(
                                sharded_inputs[0],
                                sharded_inputs[3],
                                sharded_inputs[4],
                                local_topk_weights,
                                local_topk_ids,
                                .{
                                    .workspace_query_device = ctx.workspace_query_device,
                                    .activation = ctx.activation,
                                    .enable_pdl = ctx.enable_pdl,
                                    .gemm1_tactic = ctx.gemm1_tactic,
                                    .gemm2_tactic = ctx.gemm2_tactic,
                                },
                            ) catch |err| stdx.debug.panic(
                                "FlashInfer CUTLASS MoE backend failed: {}",
                                .{err},
                            );
                            const local_reshaped = local_output
                                .reshape(sharded_inputs[0].shape().dims())
                                .withTags(.{ .b, .s, .d });
                            return zml.ops.allReduce(local_reshaped, zml.Tensor.add);
                        }
                    }).body,
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

            const global_num_experts = gate_up.weight.dim(.expert);
            const expert_partition = gate_up.weight.shape().partition(.expert);

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

            const manual_inputs: []const zml.Tensor = if (quant_scheme == .mxfp4)
                &.{ input, topk_ids, topk_weights, gate_up.weight, down.weight, gate_up_scales.?, down_scales.? }
            else
                &.{ input, topk_ids, topk_weights, gate_up.weight, down.weight };
            break :b zml.ops.manualComputation(
                manual_inputs,
                input.shape(),
                .{
                    .activation = parameters.triton.activation,
                    .global_num_experts = global_num_experts,
                    .bias_gate_up = gate_up.bias,
                    .bias_down = down.bias,
                    .quant_scheme = quant_scheme,
                    .activation_threshold = opts.activation_threshold,
                },
                (struct {
                    fn body(ctx: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                        const local_num_experts = sharded_inputs[3].dim(.expert);
                        const partition_id = zml.ops.partitionId().convert(.i32);
                        const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                        // List of global expert ids
                        const global_expert_ids = zml.Tensor.arange(.{ .end = ctx.global_num_experts }, .i32).withTags(.{.expert});

                        // Mapping of local experts to global expert ids, -1 if the global expert is not present in the local partition
                        const local_expert_mask = global_expert_ids.cmp(.GE, expert_start)
                            .logical(.AND, global_expert_ids.cmp(.LT, expert_start.addConstant(local_num_experts)));
                        const expert_map = local_expert_mask.select(
                            global_expert_ids.sub(expert_start),
                            zml.Tensor.scalar(-1, .i32),
                        );

                        const local_output = triton.fusedExpertsImpl(
                            sharded_inputs[0],
                            sharded_inputs[3],
                            sharded_inputs[4],
                            sharded_inputs[2],
                            sharded_inputs[1],
                            .{},
                            .{
                                .activation = ctx.activation,
                                .global_num_experts = ctx.global_num_experts,
                                .expert_map = expert_map,
                                .w1_scale = if (ctx.quant_scheme == .mxfp4) sharded_inputs[5] else null,
                                .w2_scale = if (ctx.quant_scheme == .mxfp4) sharded_inputs[6] else null,
                                .w1_bias = ctx.bias_gate_up,
                                .w2_bias = ctx.bias_down,
                                .quant_scheme = ctx.quant_scheme,
                                .activation_threshold = ctx.activation_threshold,
                            },
                        ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});
                        const local_reshaped = local_output.reshape(sharded_inputs[0].shape().dims()).withTags(.{ .b, .s, .d });
                        return zml.ops.allReduce(local_reshaped, zml.Tensor.add);
                    }
                }).body,
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
                    .{ input, topk_ids, topk_weights, gate_up.weight, down.weight },
                    input.shape(),
                    .{
                        .activation = parameters.mosaic_tpu.activation,
                        .global_num_experts = global_num_experts,
                        .gate_up_scales = gate_up_scales,
                        .bias_gate_up = gate_up.bias,
                        .down_scales = down_scales,
                        .bias_down = down.bias,
                    },
                    (struct {
                        fn body(ctx: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                            const local_num_experts = sharded_inputs[3].dim(.expert);
                            const partition_id = zml.ops.partitionId().convert(.i32);
                            const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                            const global_expert_ids = zml.Tensor.arange(.{ .end = ctx.global_num_experts }, .i32).withTags(.{.expert});

                            const local_expert_mask = global_expert_ids.cmp(.GE, expert_start)
                                .logical(.AND, global_expert_ids.cmp(.LT, expert_start.addConstant(local_num_experts)));
                            const expert_map = local_expert_mask.select(
                                global_expert_ids.sub(expert_start),
                                zml.Tensor.scalar(-1, .i32),
                            );
                            const local_output = mosaic_tpu.fusedExpertsImpl(
                                sharded_inputs[0],
                                sharded_inputs[3],
                                sharded_inputs[4],
                                sharded_inputs[2],
                                sharded_inputs[1],
                                .{},
                                .{
                                    .activation = ctx.activation,
                                    .global_num_experts = ctx.global_num_experts,
                                    .expert_map = expert_map,
                                    .w1_scale = ctx.gate_up_scales,
                                    .w2_scale = ctx.down_scales,
                                    .w1_bias = ctx.bias_gate_up,
                                    .w2_bias = ctx.bias_down,
                                },
                            ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});
                            return local_output.reshape(sharded_inputs[0].shape().dims()).withTags(.{ .b, .s, .d });
                        }
                    }).body,
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
        zml.nn.unpackFp4(linear.weight, linear.tag)
    else
        linear.weight;
}
