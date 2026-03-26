const std = @import("std");

const zml = @import("../zml.zig");
const flashinfer_moe = @import("flashinfer_moe.zig");
const triton_moe = @import("triton_moe.zig");
const Tensor = zml.Tensor;
const nn = zml.nn;

const MoE = @This();

pub const Backend = enum {
    flashinfer,
    triton,

    pub fn auto(platform: *const zml.Platform, weights_dtype: zml.DataType) !Backend {
        return switch (platform.target) {
            .cuda => b: {
                const first_device = platform.pjrt_client.devices(platform.pjrt_api)[0];
                if (zml.platform.cuda.tryGetComputeCapabilities(platform, first_device)) |cc| {
                    if (std.mem.eql(u8, cc, "9.0")) {
                        break :b switch (weights_dtype) {
                            .f4e2m1 => .flashinfer,
                            .bf16, .f16 => .triton,
                            else => error.UnsupportedDataType,
                        };
                    }
                    break :b error.UnsupportedComputeCapability;
                }
                break :b error.UnsupportedComputeCapability;
            },
            else => error.UnimplementedMoEBackend,
        };
    }

    pub fn load(backend: Backend, allocator: std.mem.Allocator) !void {
        return switch (backend) {
            .flashinfer => flashinfer_moe.load(allocator),
            .triton => {},
        };
    }

    pub fn register(backend: Backend, platform: *zml.Platform) !void {
        return switch (backend) {
            .flashinfer => flashinfer_moe.register(platform),
            .triton => {},
        };
    }
};

pub const Parameters = union(Backend) {
    flashinfer: flashinfer_moe.Parameters,
    triton: flashinfer_moe.Parameters,

    pub const InitOptions = union(Backend) {
        flashinfer: flashinfer_moe.Parameters.InitOptions,
        triton: flashinfer_moe.Parameters.InitOptions,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .flashinfer => .{ .flashinfer = .{} },
                .triton => .{ .triton = .{} },
            };
        }
    };

    pub fn init(opts: InitOptions) Parameters {
        return switch (opts) {
            .flashinfer => |v| .{ .flashinfer = flashinfer_moe.Parameters.init(v) },
            .triton => |v| .{ .triton = flashinfer_moe.Parameters.init(v) },
        };
    }
};

pub const Metadata = union(Backend) {
    flashinfer: void,
    triton: void,

    pub const InitOptions = union(Backend) {
        flashinfer: void,
        triton: void,

        pub fn fromBackend(backend: Backend) InitOptions {
            return switch (backend) {
                .flashinfer => .{ .flashinfer = {} },
                .triton => .{ .triton = {} },
            };
        }
    };

    pub fn init(opts: InitOptions) Metadata {
        return switch (opts) {
            .flashinfer => .{ .flashinfer = {} },
            .triton => .{ .triton = {} },
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *zml.Platform) !zml.Bufferized(Metadata) {
        _ = self;
        _ = io;
        _ = platform;
        return {};
    }

    pub fn deinitBuffer(self: *zml.Bufferized(Metadata)) void {
        _ = self;
    }
};

pub fn moe(
    input: zml.Tensor,
    tokens_mask: ?zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    weights_gate_up: zml.Tensor,
    scales_gate_up: ?zml.Tensor,
    bias_gate_up: ?zml.Tensor,
    weights_down: zml.Tensor,
    scales_down: ?zml.Tensor,
    bias_down: ?zml.Tensor,
    num_experts_per_tok: u32,
    metadata: Metadata,
    parameters: Parameters,
) !zml.Tensor {
    return switch (parameters) {
        .flashinfer => b: {
            // No metadata buffers in flashinfer_moe
            _ = metadata;
            const dt = input.dtype();

            // Routing of the tokens to the experts
            const expert_indices_per_token = topk_ids;
            var expert_scores_per_token = topk_weights;
            var output = Tensor.zeroes(input.shape());

            const num_experts: u32 = @intCast(weights_gate_up.dim(.expert));
            const hidden_size_orig: u32 = @intCast(input.dim(.d));
            const out_dim_gate_up: u32 = @intCast(weights_gate_up.dim(.out));
            const out_dim_down: u32 = @intCast(weights_down.dim(.out));

            // Set the score of the padded tokens to 0
            if (tokens_mask) |mask| {
                const mask_b = mask.broad(expert_scores_per_token.shape()).convert(expert_scores_per_token.dtype());
                expert_scores_per_token = expert_scores_per_token.mul(mask_b);
            }

            if (tokens_mask) |mask| {

                // Set the expert indices of the padded tokens to max to keep them at the end after sorting
                var max_inf = Tensor.constant(expert_indices_per_token.dtype().maxValue());
                max_inf = max_inf.broad(expert_indices_per_token.shape());
                const expert_indices_per_token_with_max = Tensor.select(mask.broad(expert_indices_per_token.shape()), expert_indices_per_token, max_inf);
                const expert_indices_per_token_with_max_flattened = expert_indices_per_token_with_max.transpose(.{ .top_expert, .s }).reshape(.{ .seq = num_experts_per_tok * expert_indices_per_token.dim(0) });

                // Repeat the the input, mask , and flatten the expert indices and scores along seq dim to process all topk
                const input_repeated = input.repeat1d(.s, num_experts_per_tok);
                const mask_repeated = mask.repeat1d(.s, num_experts_per_tok);

                const expert_indices_per_token_repeated = expert_indices_per_token.transpose(.{ .top_expert, .s }).flatten().withTags(.{.s});
                const expert_scores_per_token_repeated = expert_scores_per_token.transpose(.{ .top_expert, .s }).flatten().withTags(.{.s});
                const counts0 = zml.Tensor.zeroes(.init(.{ .expert = num_experts }, .i32));

                // Count the number of valid tokens per expert
                const counts_tokens_per_exp = counts0.scatterSlices(
                    .{ .expert = expert_indices_per_token_repeated },
                    mask_repeated.convert(.i32),
                    .{
                        .update_fn = zml.Tensor.ScatterOpts.increment,
                        .indices_are_unique = false,
                    },
                );

                const zero_prefix = zml.Tensor.zeroes(.init(.{ .expert = 1 }, .i64));
                var expert_first_tok_offset = counts_tokens_per_exp.convert(.i64).cumulativeSum(0);
                expert_first_tok_offset = Tensor.concatenate(&.{ zero_prefix, expert_first_tok_offset }, 0);

                // Sort the expert indices so that tokens of the same expert are consecutive
                const indices_sorted = expert_indices_per_token_with_max_flattened.argsort(.seq, .{ .descending = false });
                const idx = indices_sorted;
                const input_sorted = input_repeated.gather(.{ .s = idx }, .{});

                const gate_hidden_kernel: u32 = @intCast(weights_gate_up.dim(.d) * weights_gate_up.dim(.d_block) * 2);
                var input_sorted_kernel = input_sorted;
                if (input_sorted.dim(.d) < gate_hidden_kernel) {
                    const pad_high: i64 = @as(i64, gate_hidden_kernel) - input_sorted.dim(.d);
                    input_sorted_kernel = input_sorted.pad(0, .{ .d = Tensor.Pad{ .high = pad_high } });
                }

                // Gemm 1 gate up
                var out_gate_up = flashinfer_moe.flashinferMoeForward(input_sorted_kernel, weights_gate_up, scales_gate_up.?, expert_first_tok_offset, .{
                    .num_experts = num_experts,
                    .hidden_size = gate_hidden_kernel,
                    .out_features = out_dim_gate_up,
                    .output_shape = .init(.{ .s = input_sorted_kernel.dim(.seq), .d = out_dim_gate_up }, .bf16),
                });

                // The expert indices of the sorted tokens, used to gather the bias and the expert scores
                const expert_indices_per_token_sorted = expert_indices_per_token_with_max_flattened.gather(.{ .seq = indices_sorted.rename(.{ .seq = .s }) }, .{});
                if (bias_gate_up) |bias| {
                    const bias_per_token = bias.gather(.{ .expert = expert_indices_per_token_sorted }, .{});
                    out_gate_up = out_gate_up.add(bias_per_token);
                }

                // Split the gate and the up, apply quickgelu to the gate and multiply them
                var up, var gate = zml.nn.splitRealImg(out_gate_up, .sequential);
                gate = .minimum(gate, .scalar(7, dt));
                up = .clamp(up, .scalar(-7, dt), .scalar(7, dt));
                const out = gate.quickGelu().mul(up.addConstant(1));

                // Gemm 2 moe out
                const down_hidden_kernel: u32 = @intCast(out.dim(.d));
                var moe_out = flashinfer_moe.flashinferMoeForward(out, weights_down, scales_down.?, expert_first_tok_offset, .{
                    .num_experts = num_experts,
                    .hidden_size = down_hidden_kernel,
                    .out_features = out_dim_down,
                    .output_shape = .init(.{ .s = input_sorted.dim(.seq), .d = out_dim_down }, .bf16),
                });

                // Gather the bias and add it
                if (bias_down) |bias| {
                    const bias_per_token = bias.gather(.{ .expert = expert_indices_per_token_sorted }, .{});
                    moe_out = moe_out.add(bias_per_token);
                }

                // Remove the padding if needed
                if (moe_out.dim(.d) > hidden_size_orig) {
                    moe_out = moe_out.slice1d(.d, .{ .end = hidden_size_orig });
                }

                // Mul the output by the expert scores
                const expert_scores_per_token_sorted = expert_scores_per_token_repeated.gather(.{ .s = idx }, .{}).rename(.{ .seq = .s });
                moe_out = moe_out.mul(expert_scores_per_token_sorted.convert(moe_out.dtype()).broad(moe_out.shape()));

                //Reorder, reshape and sum the output of the top k experts for each token
                const restore_indices = indices_sorted.argsort(.seq, .{ .descending = false });
                const top_output = moe_out.gather(.{ .s = restore_indices }, .{});
                output = top_output.reshape(.{ .expert = num_experts_per_tok, .s = input.dim(.s), .d = hidden_size_orig });
                output = output.sum(.expert).squeeze(0);
            } else {
                // Decode
                // Same process but without handling the padded tokens, so no need to set their score to 0 and indices to max, and no need to repeat the input and the scores
                const expert_indices_per_token_flat = expert_indices_per_token.transpose(.{ .top_expert, .s }).flatten().withTags(.{.seq});
                const expert_scores_per_token_flat = expert_scores_per_token.transpose(.{ .top_expert, .s }).flatten().withTags(.{.seq});
                const input_repeated = input.repeat1d(.s, num_experts_per_tok).rename(.{ .s = .seq });

                const counts0 = zml.Tensor.zeroes(.init(.{ .expert = num_experts }, .i32));
                const ones = zml.Tensor.constant(.{ .i32 = 1 }).broad(expert_indices_per_token_flat.shape().withDtype(.i32));
                const counts_tokens_per_exp = counts0.scatterSlices(
                    .{ .expert = expert_indices_per_token_flat },
                    ones,
                    .{
                        .update_fn = zml.Tensor.ScatterOpts.increment,
                        .indices_are_unique = false,
                    },
                );

                const zero_prefix = zml.Tensor.zeroes(.init(.{ .expert = 1 }, .i64));
                var expert_first_tok_offset = counts_tokens_per_exp.convert(.i64).cumulativeSum(0);
                expert_first_tok_offset = Tensor.concatenate(&.{ zero_prefix, expert_first_tok_offset }, 0);

                const indices_sorted = expert_indices_per_token_flat.argsort(.seq, .{ .descending = false });
                const idx = indices_sorted.rename(.{ .seq = .s });
                const input_sorted = input_repeated.gather(.{ .seq = idx }, .{});

                const gate_hidden_kernel: u32 = @intCast(weights_gate_up.dim(.d) * weights_gate_up.dim(.d_block) * 2);
                var input_sorted_kernel = input_sorted;
                if (input_sorted.dim(.d) < gate_hidden_kernel) {
                    const pad_high: i64 = @as(i64, gate_hidden_kernel) - input_sorted.dim(.d);
                    input_sorted_kernel = input_sorted.pad(0, .{ .d = Tensor.Pad{ .high = pad_high } });
                }

                var out_gate_up = flashinfer_moe.flashinferMoeForward(input_sorted_kernel, weights_gate_up, scales_gate_up.?, expert_first_tok_offset, .{
                    .num_experts = num_experts,
                    .hidden_size = gate_hidden_kernel,
                    .out_features = out_dim_gate_up,
                    .output_shape = .init(.{ .s = input_sorted_kernel.dim(.s), .d = out_dim_gate_up }, .bf16),
                });

                const expert_indices_per_token_sorted = expert_indices_per_token_flat.gather(.{ .seq = idx }, .{});
                if (bias_gate_up) |bias| {
                    const bias_per_token = bias.gather(.{ .expert = expert_indices_per_token_sorted }, .{});
                    out_gate_up = out_gate_up.add(bias_per_token);
                }

                var up, var gate = zml.nn.splitRealImg(out_gate_up, .sequential);
                gate = .minimum(gate, .scalar(7, dt));
                up = .clamp(up, .scalar(-7, dt), .scalar(7, dt));
                const out = gate.quickGelu().mul(up.addConstant(1));

                const down_hidden_kernel: u32 = @intCast(out.dim(.d));
                var moe_out = flashinfer_moe.flashinferMoeForward(out, weights_down, scales_down.?, expert_first_tok_offset, .{
                    .num_experts = num_experts,
                    .hidden_size = down_hidden_kernel,
                    .out_features = out_dim_down,
                    .output_shape = .init(.{ .s = input_sorted.dim(.s), .d = out_dim_down }, .bf16),
                });
                if (bias_down) |bias| {
                    const bias_per_token = bias.gather(.{ .expert = expert_indices_per_token_sorted }, .{});
                    moe_out = moe_out.add(bias_per_token);
                }
                if (moe_out.dim(.d) > hidden_size_orig) {
                    moe_out = moe_out.slice1d(.d, .{ .end = hidden_size_orig });
                }

                const expert_scores_per_token_sorted = expert_scores_per_token_flat.gather(.{ .seq = idx }, .{});
                moe_out = moe_out.mul(expert_scores_per_token_sorted.convert(moe_out.dtype()).broad(moe_out.shape()));

                const restore_indices = indices_sorted.argsort(.seq, .{ .descending = false });
                const top_output = moe_out.gather(.{ .s = restore_indices }, .{});
                output = top_output.reshape(.{ .expert = num_experts_per_tok, .s = input.dim(.s), .d = hidden_size_orig });
                output = output.sum(.expert).squeeze(0);
            }
            break :b output;
        },
        .triton => b: {

            // To be moved in the model

            std.log.info("input shape: {f}", .{input.shape()});
            std.log.info("weights_gate_up shape: {f}", .{weights_gate_up.shape()});
            std.log.info("weights_down shape: {f}", .{weights_down.shape()});
            std.log.info("topk_ids shape: {f}", .{topk_ids.shape()});
            std.log.info("topk_weights shape: {f}", .{topk_weights.shape()});

            break :b try triton_moe.fusedExpertsImpl(
                input,
                weights_gate_up,
                weights_down,
                topk_weights,
                topk_ids,
                .{
                    .w1_scale = scales_gate_up,
                    .w2_scale = scales_down,
                    .w1_bias = bias_gate_up,
                    .w2_bias = bias_down,
                },
            );
        },
    };
}
