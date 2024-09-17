const std = @import("std");
const zml = @import("zml.zig");

const Allocator = std.mem.Allocator;
const module = zml.module;

pub fn Q4_0(comptime dtype: zml.DataType) type {
    return struct {
        const Self = @This();

        const QuantType = zml.io.gguf.quants.QuantType.q4_0;

        quant_buffer: zml.Tensor,

        pub fn compile(
            allocator: Allocator,
            ctx: *zml.Context,
            input: zml.Tensor,
            shape: zml.Shape,
        ) !module.CompiledModule(Self.forward) {
            std.debug.assert(input.dtype() == .u8);
            std.debug.assert(input.rank() == 1);
            return module.compile(
                allocator,
                ctx,
                Self.forward,
                Self{ .quant_buffer = input },
                .{shape},
            ) catch unreachable;
        }

        /// Each block is composed of a f16 scale and 32 4-bit values.
        const block_stride = 18;

        pub fn forward(self: Self, shape: zml.Shape) zml.Tensor {
            const input = self.quant_buffer;
            const block_count: u63 = @intCast(@divExact(input.dim(0), block_stride));
            const scales = extractScales(block_count, input);
            const weights = extractWeights(block_count, input);

            return scales.reshape(.{ block_count, 1 })
                .broadcastLeft(zml.Shape.init(.{ block_count, 32 }, .f32))
                .mul(weights)
                .convert(dtype)
                .reshape(.{block_count * 32})
                .reshape(shape);
        }

        pub fn scaleIndices(block_count: u63) zml.Tensor {
            // indices1 is the offsets of the scale bytes, repeated block_count times.
            const indices1 = zml.Tensor.arange(.{ .start = 0, .end = 2 }, .i32).repeat1d(0, block_count);

            // indices2 is the offsets of the blocks, repeated for each scale byte, repeated block_count times.
            const indices2 = zml.Tensor.arange(.{ .start = 0, .end = block_stride * block_count, .step = block_stride }, .i32)
                .reshape(.{ block_count, 1 }).broadcastLeft(zml.Shape.init(.{ block_count, 2 }, .i32)).reshape(.{2 * block_count});

            // indices is the sum of the two, which is the offsets to all the bytes we are interested in.
            return indices1.add(indices2);
        }

        pub fn weightIndices(block_count: u63) zml.Tensor {
            // indices1 is the offsets of the data bytes, repeated block_count times.
            const indices1 = zml.Tensor.arange(.{ .start = 2, .end = 18 }, .i32).repeat1d(0, block_count);

            // indices2 is the offsets of the blocks, repeated for each data byte, repeated block_count times.
            const indices2 = zml.Tensor.arange(.{ .start = 0, .end = block_stride * block_count, .step = block_stride }, .i32)
                .reshape(.{ block_count, 1 }).broadcastLeft(zml.Shape.init(.{ block_count, 16 }, .i32)).reshape(.{16 * block_count});

            // indices is the sum of the two, which is the offsets to all the bytes we are interested in.
            return indices1.add(indices2);
        }

        pub fn extractScales(block_count: u63, input: zml.Tensor) zml.Tensor {
            // The goal here is to get the first two bytes of every 18-bytes block in the input. For that,
            // we generate a list of indices that we will use to gather from the input.

            // indices1 is the offsets of the scale bytes, repeated block_count times.
            const indices1 = zml.Tensor.arange(.{ .start = 0, .end = 2 }, .i32).repeat1d(0, block_count);

            // indices2 is the offsets of the blocks, repeated for each scale byte, repeated block_count times.
            const indices2 = zml.Tensor.arange(.{ .start = 0, .end = block_stride * block_count, .step = block_stride }, .i32)
                .reshape(.{ block_count, 1 }).broadcastLeft(zml.Shape.init(.{ block_count, 2 }, .i32)).reshape(.{2 * block_count});

            // indices is the sum of the two, which is the offsets to all the bytes we are interested in.
            const indices = indices1.add(indices2);

            // We select the values we are interested in with the indices, group them by pair and bitcast them to f16, then convert them to f32.
            const scales = input.gather1d(0, indices, .{ .indices_are_sorted = true }).reshape(.{ block_count, 2 }).bitCast(.f16).convert(.f32);

            return scales;
        }

        pub fn extractWeights(block_count: u63, input: zml.Tensor) zml.Tensor {
            // The goal here is to get everything but the first two bytes of every 18-bytes block in the input. For that,
            // we generate a list of indices that we will use to gather from the input.

            // indices1 is the offsets of the data bytes, repeated block_count times.
            const indices1 = zml.Tensor.arange(.{ .start = 2, .end = 18 }, .i32).repeat1d(0, block_count);

            // indices2 is the offsets of the blocks, repeated for each data byte, repeated block_count times.
            const indices2 = zml.Tensor.arange(.{ .start = 0, .end = block_stride * block_count, .step = block_stride }, .i32)
                .reshape(.{ block_count, 1 }).broadcastLeft(zml.Shape.init(.{ block_count, 16 }, .i32)).reshape(.{16 * block_count});

            // indices is the sum of the two, which is the offsets to all the bytes we are interested in.
            const indices = indices1.add(indices2);

            // NOTE(Corendos): i4 is not supported by bitcast convert, so we need the following workaround.

            // We select the values we are interested in with the indices, these are our quantized_weights.
            const quantized_weights = input.gather1d(0, indices, .{ .indices_are_sorted = true });
            const lb_weights = quantized_weights
                .logical(.And, zml.Tensor.constant(.{16 * block_count}, zml.Data.init(.u8, 0xf)))
                .bitCast(.i8);
            const hb_weights = quantized_weights
                .shiftRightLogical(zml.Tensor.constant(.{16 * block_count}, zml.Data.init(.u8, 4))).bitCast(.i8);
            const weights = zml.Tensor.concatenate(
                &.{ lb_weights.reshape(.{ block_count, 16 }), hb_weights.reshape(.{ block_count, 16 }) },
                1,
            )
                .sub(zml.Tensor.constant(.{ block_count, 32 }, zml.Data.init(.i8, 8)))
                .convert(.f32);
            return weights;
        }
    };
}
