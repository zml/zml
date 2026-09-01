//! Quantization metadata and scheme classification.
const std = @import("std");
const stdx = @import("stdx");

const DataType = @import("dtype.zig").DataType;
const platform_mod = @import("platform.zig");
const Platform = platform_mod.Platform;
const Shape = @import("shape.zig").Shape;
const Tensor = @import("tensor.zig").Tensor;

pub const nvfp4_block_size = 16;
pub const mx_block_size = 32;

pub const Quantization = struct {
    scheme: Scheme, // decided when the checkpoint is read
    scales: Tensor,
    global_scale: ?GlobalScale = null,
    input_scale: ?GlobalScale = null,

    /// A per-tensor scale and the polarity its producer wrote it in
    pub const GlobalScale = struct {
        value: Tensor,
        operation: Operation,

        pub const Operation = enum { multiply, divide };

        pub fn asMultiplier(self: GlobalScale) Tensor {
            const s = self.value.convert(.f32);
            const flat = if (s.shape().count() == 1) s.asScalar() else s; // essentially normalization

            return switch (self.operation) {
                .multiply => flat,
                .divide => Tensor.scalar(1.0, .f32).broad(flat.shape()).div(flat),
            };
        }
    };

    pub const Scheme = enum {
        /// f4e2m1 values (`u8`-packed or native), f8e4m3fn scale per 16 contracted values.
        /// Emitted by llm-compressor and by NVIDIA's ModelOpt.
        nvfp4,
        /// f8e4m3fn values, one e8m0 (power-of-two) scale per 32 contracted values
        mxfp8,
        /// f4e2m1 values (`u8`-packed or native), e8m0 scale per 32 contracted values.
        // GPT-OSS like
        mxfp4,
        /// f8e4m3fn values, one bf16 or f32 scale per output channel, constant along the contraction.
        /// Emitted by llm-compressor, including for the layers an NVFP4 recipe leaves in FP8.
        fp8_per_channel,
        /// f8e4m3fn values, one bf16 scale per 128x128 tile. The DeepSeek-style FP8 that model
        /// vendors publish themselves, under `weight_scale_inv`.
        fp8_block128,
        /// f8e4m3fn values, one scale for the whole tensor. Spelled `[1, 1]` rather than as a
        /// scalar: XLA's composite rewriter requires the scale to have the operand's rank.
        fp8_per_tensor,

        pub fn accepts(self: Scheme, weight: Shape, scale: Shape) bool {
            if (weight.rank() != 2) return false;

            const n = weight.dim(0);
            const k = if (isPackedFp4(self, weight.dtype())) 2 * weight.dim(1) else weight.dim(1);

            return switch (self) {
                .nvfp4 => (weight.dtype() == .u8 or weight.dtype() == .f4e2m1) and
                    scale.dtype() == .f8e4m3fn and scale.rank() == 2 and
                    scale.dim(0) == n and scale.dim(1) * nvfp4_block_size == k,
                .mxfp8 => weight.dtype() == .f8e4m3fn and isMxScale(scale) and
                    scale.dim(0) == n and scale.dim(1) * mx_block_size == k,
                .mxfp4 => (weight.dtype() == .u8 or weight.dtype() == .i8 or weight.dtype() == .f4e2m1) and
                    isMxScale(scale) and
                    scale.dim(0) == n and scale.dim(1) * mx_block_size == k,
                .fp8_per_tensor => weight.dtype() == .f8e4m3fn and scale.count() == 1,
                .fp8_per_channel => weight.dtype() == .f8e4m3fn and
                    (scale.dtype() == .bf16 or scale.dtype() == .f32) and
                    scale.count() > 1 and scale.rank() == 2 and
                    scale.dim(0) == n and scale.dim(1) == 1,
                .fp8_block128 => weight.dtype() == .f8e4m3fn and scale.dtype() == .bf16 and
                    scale.count() > 1 and scale.rank() == 2 and
                    @rem(n, 128) == 0 and @rem(k, 128) == 0 and
                    scale.dim(0) == @divExact(n, 128) and scale.dim(1) == @divExact(k, 128),
            };
        }

        pub fn classify(weight: Shape, scale: Shape) ?Scheme {
            for (std.enums.values(Scheme)) |scheme| {
                if (scheme.accepts(weight, scale)) return scheme;
            }

            return null;
        }

        pub fn isMx(self: Scheme) bool {
            return self == .mxfp8 or self == .mxfp4;
        }
    };
};

pub const QuantizedInput = struct {
    values: Tensor,
    scales: Tensor,
    global_scale: ?Tensor,
};

pub fn quantizeInput(quantization: Quantization, input: Tensor, axis: Shape.Tag, platform: *const Platform) ?QuantizedInput {
    const global_scale: ?Tensor = if (quantization.input_scale) |scale| scale.asMultiplier() else null;
    return switch (quantization.scheme) {
        .nvfp4 => if (supportsNvfp4InputQuantization(platform)) quantizeNvfp4(input, global_scale, axis) else null,
        .mxfp8, .mxfp4, .fp8_per_channel, .fp8_block128, .fp8_per_tensor => null,
    };
}

pub fn quantizeNvfp4(x: Tensor, input_global_scale: ?Tensor, axis: anytype) QuantizedInput {
    const value_max = 6.0;
    const scale_min_normal = 0x1p-6;
    const scale_max = DataType.f8e4m3fn.maxValue().as(f32);

    stdx.debug.assert(x.shape().hasTag(axis) != null, "quantizeNvfp4 expects x to have {any} tag, got {f}", .{ axis, x.shape() });
    stdx.debug.assert(@rem(x.dim(axis), nvfp4_block_size) == 0, "quantizeNvfp4 expects {any} to be a multiple of {}, got {f}", .{ axis, nvfp4_block_size, x.shape() });

    const dt = x.dtype();
    const scaled = if (input_global_scale) |igs|
        x.div(igs.convert(dt).broad(x.shape()))
    else
        x;
    const grouped = scaled.splitAxis(axis, .{ .sc = -1, .blk = nvfp4_block_size });
    const amax = grouped.abs().max(.blk);

    const scales = amax.scale(1.0 / value_max)
        .clamp(.scalar(scale_min_normal, dt), .scalar(scale_max, dt))
        .convert(.f8e4m3fn);

    const divisor = scales.convert(dt)
        .maximum(.scalar(scale_min_normal, dt))
        .broad(grouped.shape());

    const values = grouped.div(divisor)
        .convert(.f4e2m1)
        .reshape(x.shape().withDtype(.f4e2m1));

    return .{
        .values = values,
        .scales = scales.squeeze(.blk),
        .global_scale = input_global_scale,
    };
}

fn supportsNvfp4InputQuantization(platform: *const Platform) bool {
    if (platform.target != .cuda) return false;

    const device = platform.pjrt_client.devices(platform.pjrt_api)[0];
    const capability = platform_mod.cuda.tryGetComputeCapabilities(platform, device) orelse return false;
    const major = std.fmt.parseInt(u8, std.mem.sliceTo(capability, '.'), 10) catch return false;

    return major >= 10;
}

fn isPackedFp4(scheme: ?Quantization.Scheme, weight_dtype: DataType) bool {
    return (scheme == .nvfp4 or scheme == .mxfp4) and (weight_dtype == .u8 or weight_dtype == .i8);
}

fn isMxScale(scale: Shape) bool {
    return scale.rank() == 2 and (scale.dtype() == .f8e8m0 or scale.dtype() == .u8);
}

// Note: this test will evolve as we support more (INT4/8 and FP8 block-128 as a
// scaled-dot are still open; see the scaledDot doc comment).
test "Quantization.Scheme.classify" {
    const expect = std.testing.expectEqual;

    // unsloth/Qwen3.6-27B-NVFP4: compressed-tensors, carrying both of its schemes under
    // `weight_scale` -- NVFP4 on the MLPs of layers 0-55, FP8 per-channel everywhere else.
    const nvfp4_packed: Shape = .init(.{ .dout = 17408, .kw = 2560 }, .u8);
    try expect(@as(?Quantization.Scheme, .nvfp4), Quantization.Scheme.classify(nvfp4_packed, .init(.{ .dout = 17408, .sc = 320 }, .f8e4m3fn)));
    try expect(@as(?Quantization.Scheme, .fp8_per_channel), Quantization.Scheme.classify(
        .init(.{ .dout = 17408, .d = 5120 }, .f8e4m3fn),
        .init(.{ .dout = 17408, .sc = 1 }, .bf16),
    ));

    // nvidia/Gemma-4-31B-IT-NVFP4: ModelOpt, packed values under the plain `weight` name.
    try expect(@as(?Quantization.Scheme, .nvfp4), Quantization.Scheme.classify(
        .init(.{ .dout = 21504, .kw = 2688 }, .u8),
        .init(.{ .dout = 21504, .sc = 336 }, .f8e4m3fn),
    ));

    // Native (unpacked) f4e2m1, with K no longer doubled.
    try expect(@as(?Quantization.Scheme, .nvfp4), Quantization.Scheme.classify(
        .init(.{ .dout = 17408, .d = 5120 }, .f4e2m1),
        .init(.{ .dout = 17408, .sc = 320 }, .f8e4m3fn),
    ));

    // Qwen/Qwen3.6-27B-FP8: block 128, spelled `weight_scale_inv`.
    try expect(@as(?Quantization.Scheme, .fp8_block128), Quantization.Scheme.classify(
        .init(.{ .dout = 17408, .d = 5120 }, .f8e4m3fn),
        .init(.{ .dout = 136, .sc = 40 }, .bf16),
    ));
    try expect(@as(?Quantization.Scheme, .fp8_block128), Quantization.Scheme.classify(
        .init(.{ .dout = 5120, .d = 6144 }, .f8e4m3fn),
        .init(.{ .dout = 40, .sc = 48 }, .bf16),
    ));

    // Mistral's per-tensor FP8: one scale for the whole tensor, rank 0 or [1].
    try expect(@as(?Quantization.Scheme, .fp8_per_tensor), Quantization.Scheme.classify(.init(.{ .dout = 4096, .d = 4096 }, .f8e4m3fn), .init(.{}, .f32)));
    try expect(@as(?Quantization.Scheme, .fp8_per_tensor), Quantization.Scheme.classify(.init(.{ .dout = 4096, .d = 4096 }, .f8e4m3fn), .init(.{ .g = 1 }, .f32)));

    // RadixArk/Muse-Glimmer-NVFP4: ModelOpt's MIXED_PRECISION recipe, whose MXFP8 half
    // lands on down_proj and lm_head. safetensors has no e8m0 dtype, so the scale arrives
    // as u8 under `weight_scale_inv` -- the shape below is verbatim from the checkpoint.
    try expect(@as(?Quantization.Scheme, .mxfp8), Quantization.Scheme.classify(
        .init(.{ .dout = 6656, .d = 19968 }, .f8e4m3fn),
        .init(.{ .dout = 6656, .sc = 624 }, .u8),
    ));
    try expect(@as(?Quantization.Scheme, .mxfp8), Quantization.Scheme.classify(
        .init(.{ .dout = 202048, .d = 6656 }, .f8e4m3fn),
        .init(.{ .dout = 202048, .sc = 208 }, .u8),
    ));
    // The same tensor from a store that types the scale natively.
    try expect(@as(?Quantization.Scheme, .mxfp8), Quantization.Scheme.classify(
        .init(.{ .dout = 6656, .d = 19968 }, .f8e4m3fn),
        .init(.{ .dout = 6656, .sc = 624 }, .f8e8m0),
    ));
    // MXFP8 rejects any group but 32: 19968 / 16 = 1248, / 64 = 312.
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(
        .init(.{ .dout = 6656, .d = 19968 }, .f8e4m3fn),
        .init(.{ .dout = 6656, .sc = 1248 }, .u8),
    ));
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(
        .init(.{ .dout = 6656, .d = 19968 }, .f8e4m3fn),
        .init(.{ .dout = 6656, .sc = 312 }, .u8),
    ));

    // MXFP4: the same u8 packing as NVFP4 (two nibbles per byte, K halved on disk), with
    // an e8m0 scale per 32 -- so K = 2 * 2560 = 5120 and 5120 / 32 = 160.
    try expect(@as(?Quantization.Scheme, .mxfp4), Quantization.Scheme.classify(nvfp4_packed, .init(.{ .dout = 17408, .sc = 160 }, .u8)));
    try expect(@as(?Quantization.Scheme, .mxfp4), Quantization.Scheme.classify(nvfp4_packed, .init(.{ .dout = 17408, .sc = 160 }, .f8e8m0)));
    // DeepSeek V4 stores the same packed FP4 bits in signed bytes.
    try expect(@as(?Quantization.Scheme, .mxfp4), Quantization.Scheme.classify(
        .init(.{ .dout = 2048, .kw = 2048 }, .i8),
        .init(.{ .dout = 2048, .sc = 128 }, .f8e8m0),
    ));
    // Native (unpacked) f4e2m1, K no longer halved.
    try expect(@as(?Quantization.Scheme, .mxfp4), Quantization.Scheme.classify(
        .init(.{ .dout = 17408, .d = 5120 }, .f4e2m1),
        .init(.{ .dout = 17408, .sc = 160 }, .f8e8m0),
    ));
    // An e8m0 scale never turns a non-fp4/fp8 weight into MX.
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(
        .init(.{ .dout = 17408, .d = 5120 }, .bf16),
        .init(.{ .dout = 17408, .sc = 160 }, .f8e8m0),
    ));

    // Rejected: everything no backend here can express.
    const fp8: Shape = .init(.{ .dout = 10240, .d = 5120 }, .f8e4m3fn);
    // Group 32 with an e4m3 scale: neither NVFP4 (wrong group) nor MX (wrong scale type).
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(nvfp4_packed, .init(.{ .dout = 17408, .sc = 160 }, .f8e4m3fn)));
    // Group 16 with an e8m0 scale: the mirror image, and equally unclaimed.
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(nvfp4_packed, .init(.{ .dout = 17408, .sc = 320 }, .f8e8m0)));
    // Per-channel in f32: what fusing per-tensor projections produces. The Metal
    // per-channel kernels have f32 entries, so this is a real scheme, not a decline.
    try expect(@as(?Quantization.Scheme, .fp8_per_channel), Quantization.Scheme.classify(fp8, .init(.{ .dout = 10240, .sc = 1 }, .f32)));
    // A block grid that is neither per-channel nor 128x128.
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(fp8, .init(.{ .dout = 160, .sc = 80 }, .bf16)));
    // Stacked MoE weights are rank 3 and go through the MoE path, not this one.
    try expect(@as(?Quantization.Scheme, null), Quantization.Scheme.classify(
        .init(.{ .expert = 128, .dout = 1408, .kw = 1408 }, .u8),
        .init(.{ .expert = 128, .dout = 1408, .sc = 176 }, .f8e4m3fn),
    ));

    // `fp8_per_tensor` never checks the scale dtype, so a [1, 1] u8 scale on a [1, 32]
    // weight satisfies it as well as `mxfp8`. Declaration order is what decides, and
    // `mxfp8` is declared first
    try expect(@as(?Quantization.Scheme, .mxfp8), Quantization.Scheme.classify(
        .init(.{ .dout = 1, .d = 32 }, .f8e4m3fn),
        .init(.{ .dout = 1, .sc = 1 }, .u8),
    ));

    // A one-tile block grid is indistinguishable from a per-tensor scale by shape alone; the
    // `count() > 1` guard on the grids is what makes per-tensor win.
    try expect(@as(?Quantization.Scheme, .fp8_per_tensor), Quantization.Scheme.classify(
        .init(.{ .dout = 128, .d = 128 }, .f8e4m3fn),
        .init(.{ .dout = 1, .sc = 1 }, .bf16),
    ));
}
