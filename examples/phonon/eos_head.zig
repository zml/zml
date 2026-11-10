const std = @import("std");
const zml = @import("zml");

test {
    std.testing.refAllDecls(@This());
}

/// Scores whether speech generation should end (EOS: end of speech).
///
/// A learned linear layer combines 512 final, normalized transformer features
/// into one score: `score = dot(features, weight) + bias`. A higher score favors
/// ending speech. This raw score is called a logit; it is not a probability.
///
/// Input: floating-point `[..., .d=512]`. Output: `[..., .dout=1]` in the input dtype.
/// Each input feature vector gets its own score; batch and sequence axes are
/// preserved. During generation, the caller passes only the last sequence
/// position's features, producing one score per batch row.
///
/// The generation loop compares the score directly with its EOS threshold and
/// decides when to stop. This head applies no sigmoid or stopping policy.
///
/// Matches Python's `eos_head`, a biased `Linear(512, 1)`. Parameters come from
/// `flow_lm.out_eos`: bf16 weight `[1, 512]` and bias `[1]`, converted to the input
/// dtype by forward. The generation wrapper currently supplies f32 features.
///
/// Reference:
/// ```
/// eos_head: Linear
/// ├── weight: shape=(1, 512), dtype=torch.float32
/// │   ├── loaded: eos_head.weight
/// │   └── checkpoint: flow_lm.out_eos.weight
/// └── bias: shape=(1,), dtype=torch.float32
///     ├── loaded: eos_head.bias
///     └── checkpoint: flow_lm.out_eos.bias
/// ```
pub const EOS_Head = struct {
    projection: zml.nn.Linear,

    /// Expects the checkpoint view at flow_lm.out_eos.
    pub fn init(store: zml.io.TensorStore.View) EOS_Head {
        return .{
            .projection = .init(
                store.createReplicatedTensor("weight", .{ .dout, .d }),
                store.createReplicatedTensor("bias", .{.dout}),
                .d,
            ),
        };
    }

    /// Computes one raw end-of-speech score per input feature vector.
    ///
    /// `input`: final normalized hidden states from the transformer, encoding
    /// the text, voice, and audio context visible at each supplied position.
    /// Shape: floating-point `[..., 512]`, commonly `[B, S, 512]` for a sequence or
    /// `[B, 512]` for one selected position per batch row.
    /// During generation, the prediction wrapper selects the last position
    /// after prefill, or squeezes the single position returned by cached decode,
    /// and passes `[B, 512]` here. The final axis is tagged `.d` by this method.
    ///
    /// Returns `[..., .dout=1]` in the input dtype, preserving all leading axes.
    /// Each value is a logit, not a probability; the generation loop applies the
    /// stopping threshold and policy.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension; the final axis of 512 transformer features,
    ///   contracted with the weight matrix to compute a score.
    /// - `.dout`: output dimension; one score, with its learned bias added.
    /// - Leading axes are preserved, commonly `.b` for independent sequences
    ///   and `.s` for token positions. Untagged leading axes remain untagged.
    ///
    /// Shape: `[..., .d=512]` -> `[..., .dout=1]`, preserving the floating-point dtype.
    pub fn forward(self: EOS_Head, input: zml.Tensor) zml.Tensor {
        std.debug.assert(input.dtype().isFloat());
        std.debug.assert(input.rank() >= 1);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "EOS_Head", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d}); // [...,512] -> [...,d=512]
        const output = self.projection.forward(x, x.dtype()); // [...,d=512] -> [...,dout=1], same dtype
        return output;
    }

    pub fn load(
        self: *const EOS_Head,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(EOS_Head) {
        var buffers = try zml.mem.bufferize(allocator, EOS_Head, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();
        try loader.load(io, EOS_Head, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(EOS_Head)) void {
        zml.Buffer.deinitAll(EOS_Head, self);
    }
};

test EOS_Head {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    var rng: std.Random.DefaultPrng = .init(42);

    const head: EOS_Head = .{ .projection = .init(
        .fromShape(.init(.{ .dout = 1, .d = 512 }, .bf16)),
        .fromShape(.init(.{ .dout = 1 }, .bf16)),
        .d,
    ) };

    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 2, .d = 512 }, .f32);
    const random = rng.random();
    var input: [2][2][512]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| {
                value.* = random.float(f32) * 2 - 1;
            }
        }
    }

    var weight_values: [512]zml.floats.BFloat16 = undefined;
    for (&weight_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var bias_values: [1]zml.floats.BFloat16 = undefined;
    for (&bias_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);

    var weight = try zml.Buffer.fromBytes(io, platform, head.projection.weight.shape(), std.mem.asBytes(&weight_values));
    defer weight.deinit();
    var bias = try zml.Buffer.fromBytes(io, platform, head.projection.bias.?.shape(), std.mem.asBytes(&bias_values));
    defer bias.deinit();

    const weight_buffers: zml.Bufferized(EOS_Head) = .{ .projection = .{ .weight = weight, .bias = bias, .quantization = null } };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, head, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected: [2][2][1]f32 = .{
        .{ .{-5.069765090942383}, .{-17.46717071533203} },
        .{ .{7.262748718261719}, .{-2.082882881164551} },
    };
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .s = 2, .dout = 1 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });
}

test "EOS_Head preserves floating-point input dtypes" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    const head: EOS_Head = .{ .projection = .init(
        .fromShape(.init(.{ .dout = 1, .d = 512 }, .bf16)),
        .fromShape(.init(.{ .dout = 1 }, .bf16)),
        .d,
    ) };

    for ([_]zml.DataType{ .f16, .bf16, .f32, .f64 }) |dtype| {
        const input_shape: zml.Shape = .init(.{ .b = 2, .s = 2, .d = 512 }, dtype);
        var exe = try platform.compile(allocator, io, head, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
        defer exe.deinit();
        try std.testing.expect(exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .s = 2, .dout = 1 }, dtype)));
    }
}
