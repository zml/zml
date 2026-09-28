const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../zml.zig");
const stdx = zml.stdx;
pub const triton_mxfp4 = @import("triton_mxfp4.zig");
pub const cute_mxfp4 = @import("cute_mxfp4.zig");
pub const cutlass_flashinfer = @import("cutlass_flashinfer.zig");
pub const metal = @import("metal.zig");
pub const mosaic_tpu = @import("mosaic_tpu.zig");
pub const triton = @import("triton.zig");
pub const fly = @import("fly_kernels/moe.zig");
const fused_experts = @import("fused_experts.zig");
pub const triton_kernels = @import("triton_kernels/triton_kernels.zig");
pub const ProjectionLayout = fused_experts.ProjectionLayout;

test {
    std.testing.refAllDecls(@This());
}

/// How a backend expects the expert weights to be stored
/// The packer writes this layout and forwardMoe reads it
pub const ExpertsLayout = struct {
    /// Column order of the fused gate/up projection.
    gate_up: ProjectionLayout,
    /// Backend specific transformation of the weights and scales.
    packing: Packing,

    pub const Packing = enum {
        /// Experts stacked as stored in the checkpoint.
        plain,
        /// Block scales swizzled to the 128x4 tensor-core layout (rows tiled by 4x32, columns by 4).
        swizzled_scales,
        /// Weights, block scales and global scales in the FlashInfer CUTLASS NVFP4 layout.
        flashinfer_nvfp4,
    };
};

pub const ActivationMode = enum {
    silu,
    relu,
    gelu,
};

pub const Backend = enum {
    cute_mxfp4,
    triton_mxfp4,
    flashinfer_cutlass,
    triton,
    fly,
    mosaic_tpu,
    metal,

    pub fn auto(platform: *const zml.Platform, scheme: ?zml.Quantization.Scheme, dtype: zml.DataType) !Backend {
        // Keep the dtype as an argument because non scheme-specific backends may depend on it later
        _ = dtype;
        return switch (platform.target) {
            .cuda => b: {
                const s = scheme orelse break :b .triton;
                break :b switch (s) {
                    .mxfp4 => if (cute_mxfp4.isAvailable(platform))
                        .cute_mxfp4
                    else if (triton_mxfp4.isAvailable(platform))
                        .triton_mxfp4
                    else
                        .triton,
                    .nvfp4 => if (cutlass_flashinfer.isNvfp4Supported(platform))
                        .flashinfer_cutlass
                    else
                        error.UnsupportedQuantization,
                    .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => .triton,
                };
            },
            .rocm => b: {
                const s = scheme orelse break :b .triton;
                break :b switch (s) {
                    .mxfp4 => if (zml.platform.rocm.computeCapability(platform) == .gfx942) .fly else .triton,
                    .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => .triton,
                    .nvfp4 => error.UnsupportedQuantization,
                };
            },
            .oneapi => b: {
                const s = scheme orelse break :b .triton;
                break :b switch (s) {
                    .mxfp4 => .triton,
                    .nvfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => error.UnsupportedQuantization,
                };
            },
            .tpu => if (scheme == null) .mosaic_tpu else error.UnsupportedQuantization,
            .metal => b: {
                const s = scheme orelse break :b .metal;
                break :b switch (s) {
                    .nvfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => .metal,
                    .mxfp4 => error.UnsupportedQuantization,
                };
            },
            else => error.UnimplementedMoEBackend,
        };
    }

    pub fn isAvailable(backend: Backend, platform: *const zml.Platform) bool {
        return switch (backend) {
            .triton_mxfp4 => triton_mxfp4.isAvailable(platform),
            .cute_mxfp4 => cute_mxfp4.isAvailable(platform),
            .flashinfer_cutlass => cutlass_flashinfer.isAvailable(platform),
            .fly => switch (platform.target) {
                .rocm => zml.platform.rocm.computeCapability(platform) == .gfx942,
                else => false,
            },
            .triton => switch (platform.target) {
                .cuda, .rocm, .oneapi => true,
                else => false,
            },
            .mosaic_tpu => platform.target == .tpu,
            .metal => platform.target == .metal,
        };
    }

    /// The layout contract the expert weights must have for a given backend with scheme
    pub fn expertsLayout(backend: Backend, scheme: ?zml.Quantization.Scheme) !ExpertsLayout {
        return switch (backend) {
            .cute_mxfp4 => if (scheme == .mxfp4) .{ .gate_up = .interleaved, .packing = .swizzled_scales } else error.UnsupportedQuantization,
            .triton_mxfp4 => if (scheme == .mxfp4) .{ .gate_up = .interleaved, .packing = .plain } else error.UnsupportedQuantization,
            .flashinfer_cutlass => if (scheme == .nvfp4) .{ .gate_up = .concatenated, .packing = .flashinfer_nvfp4 } else error.UnsupportedQuantization,
            .mosaic_tpu, .metal => .{ .gate_up = .concatenated, .packing = .plain },
            .triton, .fly => if (scheme == .mxfp4) .{ .gate_up = .interleaved, .packing = .plain } else .{ .gate_up = .concatenated, .packing = .plain },
        };
    }

    pub fn register(backend: Backend, platform: *zml.Platform) !void {
        return switch (backend) {
            .cute_mxfp4, .triton_mxfp4 => {},
            .flashinfer_cutlass => cutlass_flashinfer.register(platform),
            .triton, .fly => {},
            .mosaic_tpu => {},
            .metal => {},
        };
    }
};

pub const Parameters = union(Backend) {
    cute_mxfp4: cute_mxfp4.Parameters,
    triton_mxfp4: triton_mxfp4.Parameters,
    flashinfer_cutlass: cutlass_flashinfer.Parameters,
    triton: triton.Parameters,
    fly: fly.Parameters,
    mosaic_tpu: mosaic_tpu.Parameters,
    metal: metal.Parameters,

    pub const InitOptions = union(Backend) {
        cute_mxfp4: cute_mxfp4.Parameters.InitOptions,
        triton_mxfp4: triton_mxfp4.Parameters.InitOptions,
        flashinfer_cutlass: cutlass_flashinfer.Parameters.InitOptions,
        triton: triton.Parameters.InitOptions,
        fly: fly.Parameters.InitOptions,
        mosaic_tpu: mosaic_tpu.Parameters.InitOptions,
        metal: metal.Parameters.InitOptions,

        pub fn fromBackend(backend: Backend, num_experts_per_tok: u32, activation: ActivationMode) InitOptions {
            return switch (backend) {
                inline .cute_mxfp4, .triton_mxfp4 => |backend_tag| @unionInit(InitOptions, @tagName(backend_tag), .{ .num_experts_per_tok = num_experts_per_tok, .activation = activation }),
                .flashinfer_cutlass => .{ .flashinfer_cutlass = .{
                    .num_experts_per_tok = num_experts_per_tok,
                    .activation = switch (activation) {
                        .silu => .silu,
                        .relu => .relu,
                        .gelu => .gelu,
                    },
                } },
                inline .triton, .fly => |backend_tag| @unionInit(InitOptions, @tagName(backend_tag), .{
                    .num_experts_per_tok = num_experts_per_tok,
                    .activation = switch (activation) {
                        .silu => .silu,
                        .relu => .relu,
                        .gelu => .gelu,
                    },
                }),
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
            .cute_mxfp4 => |v| .{ .cute_mxfp4 = cute_mxfp4.Parameters.init(v) },
            .triton_mxfp4 => |v| .{ .triton_mxfp4 = triton_mxfp4.Parameters.init(v) },
            .flashinfer_cutlass => |v| .{ .flashinfer_cutlass = cutlass_flashinfer.Parameters.init(v) },
            inline .triton, .fly => |v, backend_tag| @unionInit(Parameters, @tagName(backend_tag), fused_experts.Parameters.init(v)),
            .mosaic_tpu => |v| .{ .mosaic_tpu = mosaic_tpu.Parameters.init(v) },
            .metal => |v| .{ .metal = metal.Parameters.init(v) },
        };
    }
};

pub const Options = struct {
    activation_threshold: ?f32 = null,
    /// Quantize activations for Triton FP8 GEMMs; false keeps BF16 activations.
    quantize_input: bool,
    /// Where routing weights are applied; FlashInfer, Mosaic and Metal require after_down.
    routing_weight_placement: fused_experts.RoutingWeightPlacement,
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
    return switch (parameters) {
        .cute_mxfp4 => |p| cute_mxfp4.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts, p),
        .triton_mxfp4 => |p| triton_mxfp4.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts, p),
        .flashinfer_cutlass => |p| cutlass_flashinfer.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts, p),
        inline .triton, .fly => |p, backend| fused_experts.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts, p, backend),
        .mosaic_tpu => |p| mosaic_tpu.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts, p),
        .metal => |p| metal.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts, p),
    };
}

pub fn unpackedWeight(linear: zml.nn.Linear) zml.Tensor {
    const quantization = linear.quantization orelse return linear.weight;
    return if (zml.nn.isPackedFp4(quantization.scheme, linear.weight.dtype()))
        zml.nn.unpackFp4(linear.weight, linear.tag, linear.tag)
    else
        linear.weight;
}
