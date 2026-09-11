const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../zml.zig");
const stdx = zml.stdx;
pub const triton_mxfp4 = @import("triton_mxfp4.zig");
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
    triton_mxfp4,
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
                .f8e4m3fn, .f8e4m3fnuz, .f8e8m0, .f16, .f32 => .triton,
                else => error.UnsupportedDataType,
            },
            .rocm => switch (weights_dtype) {
                .bf16, .f16, .f32, .f8e4m3fn, .f8e4m3fnuz, .f8e8m0 => .triton,
                else => error.UnsupportedDataType,
            },
            .oneapi => switch (weights_dtype) {
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
            .triton_mxfp4 => triton_mxfp4.isAvailable(platform),
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
            .triton_mxfp4 => {},
            .flashinfer_cutlass => cutlass_flashinfer.register(platform),
            .triton => {},
            .mosaic_tpu => {},
            .metal => {},
        };
    }
};

pub const Parameters = union(Backend) {
    triton_mxfp4: triton_mxfp4.Parameters,
    flashinfer_cutlass: cutlass_flashinfer.Parameters,
    triton: triton.Parameters,
    mosaic_tpu: mosaic_tpu.Parameters,
    metal: metal.Parameters,

    pub const InitOptions = union(Backend) {
        triton_mxfp4: triton_mxfp4.Parameters.InitOptions,
        flashinfer_cutlass: cutlass_flashinfer.Parameters.InitOptions,
        triton: triton.Parameters.InitOptions,
        mosaic_tpu: mosaic_tpu.Parameters.InitOptions,
        metal: metal.Parameters.InitOptions,

        pub fn fromBackend(backend: Backend, num_experts_per_tok: u32, activation: ActivationMode) InitOptions {
            return switch (backend) {
                .triton_mxfp4 => .{ .triton_mxfp4 = .{ .num_experts_per_tok = num_experts_per_tok, .activation = activation } },
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
            .triton_mxfp4 => |v| .{ .triton_mxfp4 = triton_mxfp4.Parameters.init(v) },
            .flashinfer_cutlass => |v| .{ .flashinfer_cutlass = cutlass_flashinfer.Parameters.init(v) },
            .triton => |v| .{ .triton = triton.Parameters.init(v) },
            .mosaic_tpu => |v| .{ .mosaic_tpu = mosaic_tpu.Parameters.init(v) },
            .metal => |v| .{ .metal = metal.Parameters.init(v) },
        };
    }
};

pub const Options = struct {
    activation_threshold: ?f32 = null,
    /// Quantize activations for Triton FP8 GEMMs; false keeps BF16 activations.
    quantize_input: bool,
    /// Gate/up layout; non-Triton backends require split columns.
    gate_up_layout: triton.GateUpLayout,
    /// Where routing weights are applied; non-Triton backends require after_down.
    routing_weight_placement: triton.RoutingWeightPlacement,
};

pub fn forwardMoe(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    opts: Options,
    parameters: Parameters,
) !zml.Tensor {
    switch (parameters) {
        .triton => {},
        .flashinfer_cutlass, .mosaic_tpu, .metal => {
            stdx.debug.assert(!opts.quantize_input, "Optional FP8 input quantization requires the Triton MoE backend", .{});
            stdx.debug.assert(opts.gate_up_layout == .split, "Non-Triton MoE backends require split gate/up columns", .{});
            stdx.debug.assert(opts.routing_weight_placement == .after_down, "Non-Triton MoE backends require routing weights after the down projection", .{});
            stdx.debug.assert(opts.activation_threshold == null, "Activation thresholds require the Triton MoE backend", .{});
        },
        .triton_mxfp4 => {
            stdx.debug.assert(opts.gate_up_layout == .interleaved, "Triton MXFP4 MoE backends require interleaved gate/up columns", .{});
        },
    }

    const gate_up_scheme: ?zml.Quantization.Scheme = if (gate_up.quantization) |q| q.scheme else null;
    const down_scheme: ?zml.Quantization.Scheme = if (down.quantization) |q| q.scheme else null;
    if (gate_up_scheme != down_scheme) return error.UnsupportedQuantization;

    const gate_up_scales: ?zml.Tensor = if (gate_up.quantization) |q| q.scales else null;
    const down_scales: ?zml.Tensor = if (down.quantization) |q| q.scales else null;
    const gate_up_global_scale: ?zml.Tensor = if (gate_up.quantization) |q| (if (q.global_scale) |scale| scale.asMultiplier() else null) else null;
    const down_global_scale: ?zml.Tensor = if (down.quantization) |q| (if (q.global_scale) |scale| scale.asMultiplier() else null) else null;
    const quant_scheme: ?zml.Quantization.Scheme = if (gate_up.quantization) |q| q.scheme else null;

    return switch (parameters) {
        .triton_mxfp4 => |p| triton_mxfp4.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts, p),
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
            const args: triton.FusedExpertsArgs = .{
                .hidden_states = input,
                .gate_up = gate_up,
                .down = down,
                .topk_weights = topk_weights,
                .topk_ids = topk_ids,
                .activation = parameters.triton.activation,
                .activation_threshold = opts.activation_threshold,
                .quantize_input = opts.quantize_input,
                .gate_up_layout = opts.gate_up_layout,
                .routing_weight_placement = opts.routing_weight_placement,
            };
            const expert_partition = gate_up.weight.shape().partition(.expert);

            if (!expert_partition.eql(.init(.experts))) {
                break :b try triton.fusedExpertsImpl(args);
            }

            break :b zml.ops.manualComputation(
                (struct {
                    args: triton.FusedExpertsArgs,
                    global_num_experts: i64,

                    fn call(self: @This(), _: zml.Shape) zml.Tensor {
                        const local_args = self.args;
                        const local_num_experts = local_args.gate_up.weight.dim(.expert);
                        const partition_id = zml.ops.partitionId().convert(.i32);
                        const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                        const global_expert_ids = zml.Tensor.arange(.{ .end = self.global_num_experts }, .i32).withTags(.{.expert});

                        // Map global expert ids to local ids, or -1 for experts outside this partition.
                        const local_expert_mask = global_expert_ids.cmp(.GE, expert_start)
                            .logical(.AND, global_expert_ids.cmp(.LT, expert_start.addConstant(local_num_experts)));
                        var mapped_args = local_args;
                        mapped_args.expert_map = local_expert_mask.select(
                            global_expert_ids.sub(expert_start),
                            zml.Tensor.scalar(-1, .i32),
                        );

                        const local_output = triton.fusedExpertsImpl(mapped_args) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});
                        const local_reshaped = local_output.reshape(local_args.hidden_states.shape().dims()).withTags(.{ .b, .s, .d });
                        return zml.ops.allReduce(local_reshaped, zml.Tensor.add);
                    }
                }).call,
                .{ .args = args, .global_num_experts = gate_up.weight.dim(.expert) },
                input.shape(),
            );
        },
        .mosaic_tpu => b: {
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
            break :b try metal.fusedExpertsImpl(
                input,
                gate_up_weight_unpacked,
                down_weight_unpacked,
                topk_weights,
                topk_ids,
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
