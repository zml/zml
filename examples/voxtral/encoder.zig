const std = @import("std");
const builtin = @import("builtin");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;


pub const Encoder = struct {
    conv0: CausalConv1d,
    conv1: CausalConv1d,

    pub fn init(store: zml.io.TensorStore.View) Encoder {
        const conv_store = store.withPrefix("conv_layers");
        return .{
            .conv0 = CausalConv1d.init(conv_store.withLayer(0).withPrefix("conv"), 1),
            .conv1 = CausalConv1d.init(conv_store.withLayer(1).withPrefix("conv"), 2),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Encoder)) void {
        CausalConv1d.unloadBuffers(&self.conv0);
        CausalConv1d.unloadBuffers(&self.conv1);
    }

    /// mel: [channels=128, time=frames]
    /// Returns: [seq, d=1280]
    pub fn forward(self: Encoder, mel: Tensor) Tensor {
        // [channels, time] -> [batch=1, channels, time]
        var h = mel.insertAxes(.channels, .{.batch});

        h = self.conv0.forward(h).gelu();
        h = self.conv1.forward(h).gelu();

        // [batch=1, channels=1280, time'] -> [time', channels=1280]
        h = h.squeeze(.batch);
        return h.transpose(.{ .time, .channels });
    }
};

/// Expects input layout [batch, channels, time] and kernel [cout, cin, k].
pub const CausalConv1d = struct {
    weight: Tensor,
    bias: Tensor,
    stride: i64,

    pub fn init(store: zml.io.TensorStore.View, stride: i64) CausalConv1d {
        return .{
            .weight = store.createTensorWithTags("weight", .{ .cout, .cin, .k }),
            .bias = store.createTensorWithTags("bias", .{.channels}),
            .stride = stride,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(CausalConv1d)) void {
        self.weight.deinit();
        self.bias.deinit();
    }

    pub fn forward(self: CausalConv1d, input: Tensor) Tensor {
        const kernel_size: i64 = @intCast(self.weight.dim(.k));
        const stride = self.stride;
        const input_len: i64 = @intCast(input.dim(.time));

        // Causal padding: padding_total goes on the left.
        // Right padding is computed to align the output length with stride.
        const padding_left = kernel_size - stride;
        const numerator = input_len - kernel_size + padding_left;
        const n_frames = @divTrunc(numerator + stride - 1, stride) + 1;
        const target_length = (n_frames - 1) * stride + kernel_size - padding_left;
        const padding_right = target_length - input_len;

        const dtype = input.dtype();
        var h = input.conv1d(self.weight.convert(dtype), .{
            .window_strides = stride,
            .padding = &.{ padding_left, padding_right },
        });

        // Bias: [channels] broadcasts to [batch, channels, time]
        h = h.add(self.bias.convert(dtype).broad(h.shape()));
        return h;
    }
};
