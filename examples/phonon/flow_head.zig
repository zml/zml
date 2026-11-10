const std = @import("std");
const zml = @import("zml");

const test_data = @import("flow_head_test_data.zig");

test {
    std.testing.refAllDecls(@This());
}

// --- Gradium_Phonon_TTS_Flow_Head ---

/// Predicts how to move a candidate audio latent toward the next speech frame.
///
/// The sampler starts with noise in a 32-component audio latent space. This head
/// predicts a velocity (a change in those coordinates) using the candidate,
/// the transformer's contextual features, and a flow-sampling interval. The
/// sampler uses that velocity to update the candidate; the result remains a
/// latent frame that must later be decoded into waveform samples.
///
/// The candidate is projected from 32 to 512 features. Separately, the projected
/// transformer condition is added to the average of two learned time embeddings.
/// Six residual blocks use this combined condition to refine the candidate's
/// hidden features, then the final layer projects them to 32 velocity components.
///
/// Inputs: floating-point candidate `[B, 32]`, transformer condition `[B, 512]`, and
/// start/end times `[B, 1]`. Output: velocity `[B, 32]` in the candidate dtype.
/// B is the number of independent candidate frames processed together.
///
/// Matches Python's `flow_head`. Parameters come from `flow_lm.flow_net`:
/// `input_proj`, `cond_embed`, `time_embed.0..1`, `res_blocks.0..5`, and
/// `final_layer`. Stored bf16 parameters are converted to the candidate input dtype by the
/// child modules.
///
/// Reference:
/// ```
///  flow_head: GradiumPhononTTSFlowHead
///  ├── input_proj: Linear
///  │   ├── weight: shape=(512, 32), dtype=torch.float32
///  │   │   ├── loaded: flow_head.input_proj.weight
///  │   │   └── checkpoint: flow_lm.flow_net.input_proj.weight
///  │   └── bias: shape=(512,), dtype=torch.float32
///  │       ├── loaded: flow_head.input_proj.bias
///  │       └── checkpoint: flow_lm.flow_net.input_proj.bias
///  ├── cond_proj: Linear
///  │   ├── weight: shape=(512, 512), dtype=torch.float32
///  │   │   ├── loaded: flow_head.cond_proj.weight
///  │   │   └── checkpoint: flow_lm.flow_net.cond_embed.weight
///  │   └── bias: shape=(512,), dtype=torch.float32
///  │       ├── loaded: flow_head.cond_proj.bias
///  │       └── checkpoint: flow_lm.flow_net.cond_embed.bias
///  ├── timestep_embeddings: ModuleList
///  │   └── 0..1: GradiumPhononTTSTimestepEmbedding
///  │       ├── fc1: Linear
///  │       │   ├── weight: shape=(512, 256), dtype=torch.float32
///  │       │   │   ├── loaded: flow_head.timestep_embeddings.1.fc1.weight
///  │       │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.weight
///  │       │   └── bias: shape=(512,), dtype=torch.float32
///  │       │       ├── loaded: flow_head.timestep_embeddings.1.fc1.bias
///  │       │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.bias
///  │       ├── fc2: Linear
///  │       │   ├── weight: shape=(512, 512), dtype=torch.float32
///  │       │   │   ├── loaded: flow_head.timestep_embeddings.1.fc2.weight
///  │       │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.weight
///  │       │   └── bias: shape=(512,), dtype=torch.float32
///  │       │       ├── loaded: flow_head.timestep_embeddings.1.fc2.bias
///  │       │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.bias
///  │       ├── act_fn: SiLUActivation
///  │       └── norm: GradiumPhononTTSTimestepNorm
///  │           └── weight: shape=(512,), dtype=torch.float32
///  │               ├── loaded: flow_head.timestep_embeddings.1.norm.weight
///  │               └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.3.alpha
///  ├── layers: ModuleList
///  │   └── 0..5: GradiumPhononTTSFlowResBlock
///  │       ├── norm: GradiumPhononTTSFlowLayerNorm
///  │       │   ├── weight: shape=(512,), dtype=torch.float32
///  │       │   │   ├── loaded: flow_head.layers.5.norm.weight
///  │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.weight
///  │       │   └── bias: shape=(512,), dtype=torch.float32
///  │       │       ├── loaded: flow_head.layers.5.norm.bias
///  │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.bias
///  │       ├── fc1: Linear
///  │       │   ├── weight: shape=(512, 512), dtype=torch.float32
///  │       │   │   ├── loaded: flow_head.layers.5.fc1.weight
///  │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.weight
///  │       │   └── bias: shape=(512,), dtype=torch.float32
///  │       │       ├── loaded: flow_head.layers.5.fc1.bias
///  │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.bias
///  │       ├── fc2: Linear
///  │       │   ├── weight: shape=(512, 512), dtype=torch.float32
///  │       │   │   ├── loaded: flow_head.layers.5.fc2.weight
///  │       │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.weight
///  │       │   └── bias: shape=(512,), dtype=torch.float32
///  │       │       ├── loaded: flow_head.layers.5.fc2.bias
///  │       │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.bias
///  │       ├── act_fn: SiLUActivation
///  │       └── adaln_proj: Linear
///  │           ├── weight: shape=(1536, 512), dtype=torch.float32
///  │           │   ├── loaded: flow_head.layers.5.adaln_proj.weight
///  │           │   └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.weight
///  │           └── bias: shape=(1536,), dtype=torch.float32
///  │               ├── loaded: flow_head.layers.5.adaln_proj.bias
///  │               └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.bias
///  └── final_layer: GradiumPhononTTSFlowFinalLayer
///      ├── norm: GradiumPhononTTSFlowLayerNorm
///      ├── adaln_proj: Linear
///      │   ├── weight: shape=(1024, 512), dtype=torch.float32
///      │   │   ├── loaded: flow_head.final_layer.adaln_proj.weight
///      │   │   └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.weight
///      │   └── bias: shape=(1024,), dtype=torch.float32
///      │       ├── loaded: flow_head.final_layer.adaln_proj.bias
///      │       └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.bias
///      ├── act_fn: SiLUActivation
///      └── proj: Linear
///          ├── weight: shape=(32, 512), dtype=torch.float32
///          │   ├── loaded: flow_head.final_layer.proj.weight
///          │   └── checkpoint: flow_lm.flow_net.final_layer.linear.weight
///          └── bias: shape=(32,), dtype=torch.float32
///              ├── loaded: flow_head.final_layer.proj.bias
///              └── checkpoint: flow_lm.flow_net.final_layer.linear.bias
/// ```
pub const Gradium_Phonon_TTS_Flow_Head = struct {
    input_proj: Input_Projection,
    cond_proj: Condition_Projection,
    timestep_embeddings: [2]Gradium_Phonon_TTS_Timestep_Embedding,
    layers: [6]Gradium_Phonon_TTS_Flow_Res_Block,
    final_layer: Gradium_Phonon_TTS_Flow_Final_Layer,

    /// Targets the phonon-31m-en flow head: two time embeddings and six blocks.
    /// Expects a checkpoint view at flow_lm.flow_net.
    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Flow_Head {
        var self: Gradium_Phonon_TTS_Flow_Head = .{
            .input_proj = .init(store.withPrefix("input_proj")),
            .cond_proj = .init(store.withPrefix("cond_embed")),
            .timestep_embeddings = undefined,
            .layers = undefined,
            .final_layer = .init(store.withPrefix("final_layer")),
        };
        for (&self.timestep_embeddings, 0..) |*embedding, i| {
            embedding.* = .init(store.withPrefix("time_embed").withLayer(i));
        }
        for (&self.layers, 0..) |*layer, i| {
            layer.* = .init(store.withPrefix("res_blocks").withLayer(i));
        }
        return self;
    }

    /// Predicts a latent velocity for the supplied candidate and sampling interval.
    ///
    /// `input`: floating-point `[B, 32]`, the sampler's current candidate for the next audio
    /// frame in normalized latent space. The prediction wrapper passes it through
    /// from the sampler; in the current one-step sampler it is freshly drawn noise.
    /// `condition`: floating-point `[B, 512]`, final normalized transformer features encoding
    /// text, voice, and preceding audio context. The prediction wrapper selects
    /// the last position after prefill or squeezes the single cached decode position.
    /// `start_time`: floating-point `[B, 1]`, the interval's starting flow-sampling progress.
    /// `end_time`: floating-point `[B, 1]`, the interval's ending flow-sampling progress.
    /// The sampler supplies both times, normally in [0, 1]; the current one-step
    /// sampler uses 0 and 1. These values describe progress from noise toward an
    /// audio latent, rather than token positions or waveform timestamps.
    /// All inputs must have the same batch size B; this method assigns their tags.
    ///
    /// Returns floating-point `[B, 32]`, a velocity in the candidate's latent coordinates.
    /// The caller applies the numerical update; for the current interval [0, 1],
    /// that is `frame = input + velocity`.
    ///
    /// The candidate input determines the computation and output dtype.
    /// Transformer conditioning, interval times, and parameters are converted to it.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent candidate frames, shared by all four inputs.
    /// - `.ld`: latent dimension; 32 components in the candidate and output velocity.
    /// - `.d`: hidden dimension; 512 projected features in the flow blocks and condition.
    /// - `.t`: one scalar flow time per row, embedded separately for start and end.
    ///   These are sampling progress in [0, 1], not audio or token positions.
    /// Internal frequency and modulation tags are documented by the child modules.
    ///
    /// Inputs: floating-point candidate `[.b, .ld=32]`, condition `[.b, .d=512]`,
    /// start/end times each `[.b, .t=1]`. Output: floating-point `[.b, .ld=32]`.
    pub fn forward(
        self: Gradium_Phonon_TTS_Flow_Head,
        input: zml.Tensor,
        condition: zml.Tensor,
        start_time: zml.Tensor,
        end_time: zml.Tensor,
    ) zml.Tensor {
        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat() and condition.dtype().isFloat());
        std.debug.assert(start_time.dtype().isFloat() and end_time.dtype().isFloat());
        std.debug.assert(input.rank() == 2 and condition.rank() == 2);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Flow_Head", .{});
        defer compiler.popLocation();

        const x = input.withTags(.{ .b, .ld }); // [B,32] -> [b=B,ld=32]
        const c = condition.convert(dtype).withTags(.{ .b, .d }); // [B,512] -> [b=B,d=512]
        const start = start_time.convert(dtype).withTags(.{ .b, .t }); // [B,1] -> [b=B,t=1]
        const end = end_time.convert(dtype).withTags(.{ .b, .t }); // [B,1] -> [b=B,t=1]
        std.debug.assert(x.dim(.b) == c.dim(.b));
        std.debug.assert(x.dim(.b) == start.dim(.b) and x.dim(.b) == end.dim(.b));

        var hidden = self.input_proj.forward(x); // [B,ld=32] -> [B,d=512]
        const start_embedding = self.timestep_embeddings[0].forward(start); // [B,t=1] -> [B,d=512]
        const end_embedding = self.timestep_embeddings[1].forward(end); // [B,t=1] -> [B,d=512]
        const summed_time = start_embedding.add(end_embedding); // [B,512] + [B,512] -> [B,512]
        const time_embedding = summed_time.scale(0.5); // [B,512] -> [B,512]
        const projected_condition = self.cond_proj.forward(c); // [B,d=512] -> [B,d=512]
        const combined_condition = projected_condition.add(time_embedding); // [B,512] + [B,512] -> [B,512]

        for (self.layers, 0..) |layer, i| {
            compiler.pushLocationFmt(@src(), "layer[{d}]", .{i});
            hidden = layer.forward(hidden, combined_condition); // [B,512], [B,512] -> [B,512]
            compiler.popLocation();
        }
        const velocity = self.final_layer.forward(hidden, combined_condition); // [B,512], [B,512] -> [B,ld=32]
        return velocity;
    }

    pub fn load(
        self: *const Gradium_Phonon_TTS_Flow_Head,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_Flow_Head) {
        // bufferize does not traverse arrays; load their elements individually.
        var buffers: zml.Bufferized(Gradium_Phonon_TTS_Flow_Head) = undefined;
        buffers.input_proj = try self.input_proj.load(allocator, io, platform, store);
        errdefer Input_Projection.unloadBuffers(&buffers.input_proj);
        buffers.cond_proj = try self.cond_proj.load(allocator, io, platform, store);
        errdefer Condition_Projection.unloadBuffers(&buffers.cond_proj);
        buffers.final_layer = try self.final_layer.load(allocator, io, platform, store);
        errdefer Gradium_Phonon_TTS_Flow_Final_Layer.unloadBuffers(&buffers.final_layer);

        var times_loaded: usize = 0;
        errdefer for (buffers.timestep_embeddings[0..times_loaded]) |*embedding| {
            Gradium_Phonon_TTS_Timestep_Embedding.unloadBuffers(embedding);
        };
        for (&self.timestep_embeddings, &buffers.timestep_embeddings) |*embedding, *buffer| {
            buffer.* = try embedding.load(allocator, io, platform, store);
            times_loaded += 1;
        }
        var layers_loaded: usize = 0;
        errdefer for (buffers.layers[0..layers_loaded]) |*layer| {
            Gradium_Phonon_TTS_Flow_Res_Block.unloadBuffers(layer);
        };
        for (&self.layers, &buffers.layers) |*layer, *buffer| {
            buffer.* = try layer.load(allocator, io, platform, store);
            layers_loaded += 1;
        }
        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_Flow_Head)) void {
        Input_Projection.unloadBuffers(&self.input_proj);
        Condition_Projection.unloadBuffers(&self.cond_proj);
        Gradium_Phonon_TTS_Flow_Final_Layer.unloadBuffers(&self.final_layer);
        for (&self.timestep_embeddings) |*embedding| {
            Gradium_Phonon_TTS_Timestep_Embedding.unloadBuffers(embedding);
        }
        for (&self.layers) |*layer| {
            Gradium_Phonon_TTS_Flow_Res_Block.unloadBuffers(layer);
        }
    }
};

test Gradium_Phonon_TTS_Flow_Head {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    var head: Gradium_Phonon_TTS_Flow_Head = .{
        .input_proj = .{
            .projection = .init(
                .fromShape(.init(.{ .d = 512, .ld = 32 }, .bf16)),
                .fromShape(.init(.{ .d = 512 }, .bf16)),
                .ld,
            ),
        },
        .cond_proj = .{
            .projection = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
        },
        .timestep_embeddings = undefined,
        .layers = undefined,
        .final_layer = .{
            .adaln_proj = .init(
                .fromShape(.init(.{ .m = 1024, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .m = 1024 }, .bf16)),
                .d,
            ),
            .proj = .init(
                .fromShape(.init(.{ .ld = 32, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .ld = 32 }, .bf16)),
                .d,
            ),
        },
    };
    for (&head.timestep_embeddings) |*embedding| {
        embedding.* = .{
            .freqs = .fromShape(.init(.{ .f = 128 }, .bf16)),
            .fc1 = .init(
                .fromShape(.init(.{ .d = 512, .f = 256 }, .bf16)),
                .fromShape(.init(.{ .d = 512 }, .bf16)),
                .f,
            ),
            .fc2 = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
            .norm_weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
        };
    }
    for (&head.layers) |*block| {
        block.* = .{
            .norm_weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .norm_bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
            .fc1 = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
            .fc2 = .init(
                .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .dout = 512 }, .bf16)),
                .d,
            ),
            .adaln_proj = .init(
                .fromShape(.init(.{ .m = 1536, .d = 512 }, .bf16)),
                .fromShape(.init(.{ .m = 1536 }, .bf16)),
                .d,
            ),
        };
    }

    const input_shape: zml.Shape = .init(.{ .b = 2, .ld = 32 }, .f32);
    var input: [2][32]f32 = undefined;
    for (&input) |*row| {
        for (row) |*value| value.* = random.float(f32) * 2 - 1;
    }
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    const condition_shape: zml.Shape = .init(.{ .b = 2, .d = 512 }, .f32);
    var condition: [2][512]f32 = undefined;
    for (&condition) |*row| {
        for (row) |*value| value.* = random.float(f32) * 2 - 1;
    }
    var condition_buffer = try zml.Buffer.fromBytes(io, platform, condition_shape, std.mem.asBytes(&condition));
    defer condition_buffer.deinit();

    const start_shape: zml.Shape = .init(.{ .b = 2, .t = 1 }, .f32);
    var start: [2][1]f32 = undefined;
    for (&start) |*row| {
        for (row) |*value| value.* = random.float(f32);
    }
    var start_buffer = try zml.Buffer.fromBytes(io, platform, start_shape, std.mem.asBytes(&start));
    defer start_buffer.deinit();

    const end_shape = start_shape;
    var end: [2][1]f32 = undefined;
    for (&end, start) |*row, from| row[0] = from[0] + random.float(f32) * (1 - from[0]);
    var end_buffer = try zml.Buffer.fromBytes(io, platform, end_shape, std.mem.asBytes(&end));
    defer end_buffer.deinit();

    // Bufferize array elements explicitly, as in the production loader.
    var weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_Flow_Head) = .{
        .input_proj = try zml.mem.bufferize(allocator, Input_Projection, &head.input_proj),
        .cond_proj = try zml.mem.bufferize(allocator, Condition_Projection, &head.cond_proj),
        .final_layer = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Flow_Final_Layer, &head.final_layer),
        .timestep_embeddings = undefined,
        .layers = undefined,
    };
    for (&weight_buffers.timestep_embeddings, &head.timestep_embeddings) |*buffer, *embedding| {
        buffer.* = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Timestep_Embedding, embedding);
    }
    for (&weight_buffers.layers, &head.layers) |*buffer, *block| {
        buffer.* = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Flow_Res_Block, block);
    }
    const tensors = try zml.meta.collectValues(zml.Tensor, allocator, &head);
    defer allocator.free(tensors);
    const buffers = try zml.meta.collectMutPtrs(zml.Buffer, allocator, &weight_buffers);
    defer allocator.free(buffers);
    var loaded: usize = 0;
    defer for (buffers[0..loaded]) |buffer| buffer.deinit();
    for (tensors, buffers) |tensor, buffer| {
        const values = try zml.Slice.alloc(allocator, tensor.shape());
        defer values.free(allocator);
        for (values.items(zml.floats.BFloat16)) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
        buffer.* = try zml.Buffer.fromSlice(io, platform, values);
        loaded += 1;
    }

    var exe = try platform.compile(allocator, io, head, .forward, .{
        zml.Tensor.fromShape(input_shape),
        zml.Tensor.fromShape(condition_shape),
        zml.Tensor.fromShape(start_shape),
        zml.Tensor.fromShape(end_shape),
    }, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer, condition_buffer, start_buffer, end_buffer });
    defer output.deinit();

    const expected = test_data.flow_head;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .ld = 32 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, head, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
            zml.Tensor.fromShape(condition_shape),
            zml.Tensor.fromShape(start_shape),
            zml.Tensor.fromShape(end_shape),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .ld = 32 }, dtype)));
    }
}

// --- Gradium_Phonon_TTS_Timestep_Embedding ---

/// Encodes a scalar flow-sampling time as 512 conditioning features.
///
/// Multiplies the time by 128 stored frequencies and concatenates the cosines
/// and sines into 256 features. Two biased linear layers with SiLU between them
/// produce 512 features, followed by RMS normalization with a learned scale
/// and epsilon 1e-5. RMS normalization does not subtract the mean.
///
/// Input: floating-point times `[B, 1]`. Output: embeddings `[B, 512]` in the input dtype.
/// The flow head uses separate instances for interval start and end times,
/// then averages their outputs before adding the projected transformer condition.
///
/// Parameters come from `flow_lm.flow_net.time_embed.0` or `.1`: `freqs` [128],
/// `mlp.0` (256 -> 512), `mlp.2` (512 -> 512), and `mlp.3.alpha` [512].
/// Stored bf16 tensors are converted to the input dtype; frequencies retain checkpoint rounding.
///
/// Reference:
/// ```
//  GradiumPhononTTSTimestepEmbedding
//  ├── fc1: Linear
//  │   ├── weight: shape=(512, 256), dtype=torch.float32
//  │   │   ├── loaded: flow_head.timestep_embeddings.1.fc1.weight
//  │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.weight
//  │   └── bias: shape=(512,), dtype=torch.float32
//  │       ├── loaded: flow_head.timestep_embeddings.1.fc1.bias
//  │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.0.bias
//  ├── fc2: Linear
//  │   ├── weight: shape=(512, 512), dtype=torch.float32
//  │   │   ├── loaded: flow_head.timestep_embeddings.1.fc2.weight
//  │   │   └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.weight
//  │   └── bias: shape=(512,), dtype=torch.float32
//  │       ├── loaded: flow_head.timestep_embeddings.1.fc2.bias
//  │       └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.2.bias
//  ├── act_fn: SiLUActivation
//  └── norm: GradiumPhononTTSTimestepNorm
//      └── weight: shape=(512,), dtype=torch.float32
//          ├── loaded: flow_head.timestep_embeddings.1.norm.weight
//          └── checkpoint: flow_lm.flow_net.time_embed.1.mlp.3.alpha
/// ```
pub const Gradium_Phonon_TTS_Timestep_Embedding = struct {
    freqs: zml.Tensor,
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    norm_weight: zml.Tensor,
    eps: f32 = 1e-5,

    /// Expects a view at flow_lm.flow_net.time_embed.0 (or .1).
    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Timestep_Embedding {
        return .{
            .freqs = store.createReplicatedTensor("freqs", .{.f}),
            .fc1 = .init(
                store.createReplicatedTensor("mlp.0.weight", .{ .d, .f }),
                store.createReplicatedTensor("mlp.0.bias", .{.d}),
                .f,
            ),
            .fc2 = .init(
                store.createReplicatedTensor("mlp.2.weight", .{ .dout, .d }),
                store.createReplicatedTensor("mlp.2.bias", .{.dout}),
                .d,
            ),
            .norm_weight = store.createReplicatedTensor("mlp.3.alpha", .{.d}),
        };
    }

    // -- Description --

    /// Embeds a flow-sampling time using sinusoidal features and a learned MLP.
    ///
    /// `input`: floating-point `[B, 1]`, one scalar flow-sampling progress value per batch row,
    /// normally in [0, 1]. The flow head passes its `start_time` to instance 0 and
    /// its `end_time` to instance 1; both originate in the sampler. This method
    /// assigns tags `.b` and `.t`.
    ///
    /// Returns floating-point `[B, 512]`, a conditioning vector for that interval endpoint.
    /// The parent averages the two endpoint embeddings. Computation is
    /// cos/sin features -> fc1 -> SiLU -> fc2 -> learned RMS normalization.
    ///
    /// Parameters are converted to the input dtype; the output keeps that dtype.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent time values, embedded without mixing rows.
    /// - `.t`: one scalar flow time per row. Broadcast over frequencies, then removed.
    /// - `.f`: frequency features; 128 stored frequencies, expanded to 256 features
    ///   by concatenating all cosines followed by all sines. Contracted by fc1.
    /// - `.d`: hidden dimension; 512 features after fc1 and in the final embedding.
    ///   RMS normalization reduces over this axis without subtracting the mean;
    ///   the learned weight then scales each feature (no bias).
    /// - `.dout`: fc2's 512 output features, renamed to `.d` before normalization.
    ///
    /// Shape: `[.b, .t=1]` -> `[.b, .f=256]` -> `[.b, .d=512]`, preserving the input dtype.
    pub fn forward(self: Gradium_Phonon_TTS_Timestep_Embedding, input: zml.Tensor) zml.Tensor {
        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat());
        std.debug.assert(input.rank() == 2);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Timestep_Embedding", .{});
        defer compiler.popLocation();

        const times = input.withTags(.{ .b, .t }); // [B,1] -> [b=B,t=1]
        std.debug.assert(times.dim(.t) == 1);

        // Preserve the checkpoint's rounded frequencies rather than recomputing them.
        const frequencies = self.freqs.convert(dtype); // [f=128] bf16 -> [f=128]
        const angle_shape = zml.Shape.init(.{ .b = times.dim(.b), .f = frequencies.dim(.f) }, dtype);
        const frequency_times = times.rename(.{ .t = .f }); // [B,t=1] -> [B,f=1]
        const broadcast_times = frequency_times.broad(angle_shape); // [B,f=1] -> [B,f=128]
        const broadcast_frequencies = frequencies.broad(angle_shape); // [f=128] -> [B,f=128]
        const angles = broadcast_times.mul(broadcast_frequencies); // [B,128] * [B,128] -> [B,128]
        const cosines = angles.cos(); // [B,f=128] -> [B,f=128]
        const sines = angles.sin(); // [B,f=128] -> [B,f=128]
        const embedding = zml.Tensor.concatenate(&.{ cosines, sines }, .f); // 2 x [B,f=128] -> [B,f=256]

        const projected = self.fc1.forward(embedding, dtype); // [B,f=256] -> [B,d=512]
        const hidden = projected.silu(); // [B,d=512] -> [B,d=512]
        const projected_out = self.fc2.forward(hidden, dtype); // [B,d=512] -> [B,dout=512]
        const out = projected_out.rename(.{ .dout = .d }); // [B,dout=512] -> [B,d=512]

        // Match Python's executed mean-square normalization, without centering.
        const squared = out.mul(out); // [B,512] * [B,512] -> [B,512]
        const mean_square = squared.mean(.d); // [B,d=512] -> [B,d=1]
        const stabilized = mean_square.addConstant(self.eps); // [B,1] + scalar -> [B,1]
        const inverse_rms = stabilized.rsqrt(); // [B,1] -> [B,1]
        const broadcast_inverse_rms = inverse_rms.broad(out.shape()); // [B,d=1] -> [B,d=512]
        const weight = self.norm_weight.convert(dtype); // [d=512] bf16 -> [d=512]
        const broadcast_weight = weight.broad(out.shape()); // [d=512] -> [B,d=512]
        const weighted = broadcast_weight.mul(out); // [B,512] * [B,512] -> [B,512]
        const output = weighted.mul(broadcast_inverse_rms); // [B,512] * [B,512] -> [B,512]
        return output;
    }

    // -- Runtime --

    pub fn load(
        self: *const Gradium_Phonon_TTS_Timestep_Embedding,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_Timestep_Embedding) {
        var buffers = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Timestep_Embedding, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Gradium_Phonon_TTS_Timestep_Embedding, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_Timestep_Embedding)) void {
        zml.Buffer.deinitAll(Gradium_Phonon_TTS_Timestep_Embedding, self);
    }
};

test Gradium_Phonon_TTS_Timestep_Embedding {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();

    const input_shape: zml.Shape = .init(.{ .b = 2, .t = 1 }, .f32);
    var input: [2][1]f32 = undefined;
    for (&input) |*row| {
        for (row) |*value| value.* = random.float(f32);
    }
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    const embedding: Gradium_Phonon_TTS_Timestep_Embedding = .{
        .freqs = .fromShape(.init(.{ .f = 128 }, .bf16)),
        .fc1 = .init(
            .fromShape(.init(.{ .d = 512, .f = 256 }, .bf16)),
            .fromShape(.init(.{ .d = 512 }, .bf16)),
            .f,
        ),
        .fc2 = .init(
            .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
            .fromShape(.init(.{ .dout = 512 }, .bf16)),
            .d,
        ),
        .norm_weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
    };
    var frequencies: [128]zml.floats.BFloat16 = undefined;
    for (&frequencies) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var frequencies_buffer = try zml.Buffer.fromBytes(io, platform, embedding.freqs.shape(), std.mem.asBytes(&frequencies));
    defer frequencies_buffer.deinit();
    var fc1_weights: [512][256]zml.floats.BFloat16 = undefined;
    for (&fc1_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var fc1_weights_buffer = try zml.Buffer.fromBytes(io, platform, embedding.fc1.weight.shape(), std.mem.asBytes(&fc1_weights));
    defer fc1_weights_buffer.deinit();
    var fc1_bias: [512]zml.floats.BFloat16 = undefined;
    for (&fc1_bias) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var fc1_bias_buffer = try zml.Buffer.fromBytes(io, platform, embedding.fc1.bias.?.shape(), std.mem.asBytes(&fc1_bias));
    defer fc1_bias_buffer.deinit();
    var fc2_weights: [512][512]zml.floats.BFloat16 = undefined;
    for (&fc2_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var fc2_weights_buffer = try zml.Buffer.fromBytes(io, platform, embedding.fc2.weight.shape(), std.mem.asBytes(&fc2_weights));
    defer fc2_weights_buffer.deinit();
    var fc2_bias: [512]zml.floats.BFloat16 = undefined;
    for (&fc2_bias) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var fc2_bias_buffer = try zml.Buffer.fromBytes(io, platform, embedding.fc2.bias.?.shape(), std.mem.asBytes(&fc2_bias));
    defer fc2_bias_buffer.deinit();
    var norm_weights: [512]zml.floats.BFloat16 = undefined;
    for (&norm_weights) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var norm_weights_buffer = try zml.Buffer.fromBytes(io, platform, embedding.norm_weight.shape(), std.mem.asBytes(&norm_weights));
    defer norm_weights_buffer.deinit();
    const weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_Timestep_Embedding) = .{
        .freqs = frequencies_buffer,
        .fc1 = .{ .weight = fc1_weights_buffer, .bias = fc1_bias_buffer, .quantization = null },
        .fc2 = .{ .weight = fc2_weights_buffer, .bias = fc2_bias_buffer, .quantization = null },
        .norm_weight = norm_weights_buffer,
    };
    var exe = try platform.compile(allocator, io, embedding, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.timestep_embedding;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .d = 512 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, embedding, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .d = 512 }, dtype)));
    }
}

// --- Gradium_Phonon_TTS_Flow_Res_Block ---

/// Refines a candidate frame's hidden features using transformer and time context.
///
/// The condition produces three 512-feature vectors: shift, scale, and gate.
/// Shift and scale modulate the normalized input before a two-layer MLP;
/// the gate scales that MLP's update before it is added to the original input:
/// `output = input + gate * MLP(LayerNorm(input) * (1 + scale) + shift)`.
/// The gate is a raw learned multiplier, with no sigmoid.
///
/// Input and condition: floating-point `[B, 512]`. Output: `[B, 512]` in the input dtype.
/// Each batch row is processed independently. The flow head chains six blocks,
/// passing the same combined transformer/time condition to each one.
///
/// Parameters come from `flow_lm.flow_net.res_blocks.{index}`: `in_ln`
/// (learned scale and bias), `mlp.0` and `mlp.2` (512 -> 512, with SiLU between),
/// and `adaLN_modulation.1` (512 -> 1536, applied after SiLU).
/// Stored bf16 parameters are converted to the input dtype; LayerNorm epsilon is 1e-6.
///
/// Reference:
/// ```
//  GradiumPhononTTSFlowResBlock
//  ├── norm: GradiumPhononTTSFlowLayerNorm
//  │   ├── weight: shape=(512,), dtype=torch.float32
//  │   │   ├── loaded: flow_head.layers.5.norm.weight
//  │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.weight
//  │   └── bias: shape=(512,), dtype=torch.float32
//  │       ├── loaded: flow_head.layers.5.norm.bias
//  │       └── checkpoint: flow_lm.flow_net.res_blocks.5.in_ln.bias
//  ├── fc1: Linear
//  │   ├── weight: shape=(512, 512), dtype=torch.float32
//  │   │   ├── loaded: flow_head.layers.5.fc1.weight
//  │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.weight
//  │   └── bias: shape=(512,), dtype=torch.float32
//  │       ├── loaded: flow_head.layers.5.fc1.bias
//  │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.0.bias
//  ├── fc2: Linear
//  │   ├── weight: shape=(512, 512), dtype=torch.float32
//  │   │   ├── loaded: flow_head.layers.5.fc2.weight
//  │   │   └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.weight
//  │   └── bias: shape=(512,), dtype=torch.float32
//  │       ├── loaded: flow_head.layers.5.fc2.bias
//  │       └── checkpoint: flow_lm.flow_net.res_blocks.5.mlp.2.bias
//  ├── act_fn: SiLUActivation
//  └── adaln_proj: Linear
//      ├── weight: shape=(1536, 512), dtype=torch.float32
//      │   ├── loaded: flow_head.layers.5.adaln_proj.weight
//      │   └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.weight
//      └── bias: shape=(1536,), dtype=torch.float32
//          ├── loaded: flow_head.layers.5.adaln_proj.bias
//          └── checkpoint: flow_lm.flow_net.res_blocks.5.adaLN_modulation.1.bias
/// ```
pub const Gradium_Phonon_TTS_Flow_Res_Block = struct {
    norm_weight: zml.Tensor,
    norm_bias: zml.Tensor,
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    adaln_proj: zml.nn.Linear,
    eps: f32 = 1e-6,

    /// Expects a view at flow_lm.flow_net.res_blocks.{index}.
    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Flow_Res_Block {
        return .{
            .norm_weight = store.createReplicatedTensor("in_ln.weight", .{.d}),
            .norm_bias = store.createReplicatedTensor("in_ln.bias", .{.d}),
            .fc1 = .init(store.createReplicatedTensor("mlp.0.weight", .{ .dout, .d }), store.createReplicatedTensor(
                "mlp.0.bias",
                .{.dout},
            ), .d),
            .fc2 = .init(store.createReplicatedTensor("mlp.2.weight", .{ .dout, .d }), store.createReplicatedTensor(
                "mlp.2.bias",
                .{.dout},
            ), .d),
            .adaln_proj = .init(
                store.createReplicatedTensor("adaLN_modulation.1.weight", .{ .m, .d }),
                store.createReplicatedTensor("adaLN_modulation.1.bias", .{.m}),
                .d,
            ),
        };
    }

    /// Updates flow hidden features using a condition-dependent residual MLP.
    ///
    /// `input`: floating-point `[B, 512]`, the candidate's current hidden features, produced
    /// by `Input_Projection` for the first block or by the preceding residual block.
    /// `condition`: floating-point `[B, 512]`, the flow head's projected transformer context
    /// plus the average of its start/end time embeddings. The parent computes
    /// this once and passes the same tensor to all six blocks and the final layer.
    /// Batch rows correspond one-to-one; this method assigns tags `.b` and `.d`.
    ///
    /// Returns floating-point `[B, 512]`, the input plus a gated MLP update. SiLU(condition)
    /// produces shift, scale, and gate through a linear projection. Shift and scale
    /// modulate the normalized input; the gate multiplies the MLP output without
    /// a sigmoid before the residual addition.
    ///
    /// The input determines the computation and output dtype. The condition and
    /// parameters are converted to that dtype before computation.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent candidate frames, with one condition per row.
    /// - `.d`: 512 hidden features; LayerNorm reduces this axis separately per row.
    ///   Both the input and the combined transformer/time condition use this axis.
    /// - `.m`: 1536 modulation features; three consecutive groups of 512 for shift,
    ///   scale and gate. Each group is renamed to `.d` after splitting.
    /// - `.dout`: 512 output features of each MLP projection, renamed to `.d`.
    ///
    /// Shapes: input and condition floating-point `[.b, .d=512]` -> floating-point `[.b, .d=512]`.
    pub fn forward(self: Gradium_Phonon_TTS_Flow_Res_Block, input: zml.Tensor, condition: zml.Tensor) zml.Tensor {
        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat() and condition.dtype().isFloat());
        std.debug.assert(input.rank() == 2 and condition.rank() == 2);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Flow_Res_Block", .{});
        defer compiler.popLocation();

        const x = input.withTags(.{ .b, .d }); // [B,512] -> [b=B,d=512]
        const c = condition.convert(dtype).withTags(.{ .b, .d }); // [B,512] -> [b=B,d=512]
        std.debug.assert(x.dim(.b) == c.dim(.b));
        std.debug.assert(x.dim(.d) == c.dim(.d));

        const activated_condition = c.silu(); // [B,d=512] -> [B,d=512]
        const projected_condition = self.adaln_proj.forward(activated_condition, dtype); // [B,d=512] -> [B,m=1536]
        const modulation = projected_condition.chunkExact(.m, 3); // [B,m=1536] -> 3 x [B,m=512]
        const shift = modulation[0].rename(.{ .m = .d }); // [B,m=512] -> [B,d=512]
        const scale = modulation[1].rename(.{ .m = .d }); // [B,m=512] -> [B,d=512]
        const gate = modulation[2].rename(.{ .m = .d }); // [B,m=512] -> [B,d=512]

        // Population variance divides by the full feature count, matching Python.
        const mean = x.mean(.d); // [B,d=512] -> [B,d=1]
        const broadcast_mean = mean.broad(x.shape()); // [B,d=1] -> [B,d=512]
        const centered = x.sub(broadcast_mean); // [B,512] - [B,512] -> [B,512]
        const squared = centered.mul(centered); // [B,512] * [B,512] -> [B,512]
        const variance = squared.mean(.d); // [B,d=512] -> [B,d=1]
        const stabilized = variance.addConstant(self.eps); // [B,1] + scalar -> [B,1]
        const inverse_std = stabilized.rsqrt(); // [B,1] -> [B,1]
        const broadcast_inverse_std = inverse_std.broad(x.shape()); // [B,d=1] -> [B,d=512]
        const unit_variance = centered.mul(broadcast_inverse_std); // [B,512] * [B,512] -> [B,512]
        const weight = self.norm_weight.convert(dtype); // [d=512] bf16 -> [d=512]
        const broadcast_weight = weight.broad(x.shape()); // [d=512] -> [B,d=512]
        const weighted = unit_variance.mul(broadcast_weight); // [B,512] * [B,512] -> [B,512]
        const bias = self.norm_bias.convert(dtype); // [d=512] bf16 -> [d=512]
        const broadcast_bias = bias.broad(x.shape()); // [d=512] -> [B,d=512]
        const normalized = weighted.add(broadcast_bias); // [B,512] + [B,512] -> [B,512]

        const unit_scale = scale.addConstant(1); // [B,512] + scalar -> [B,512]
        const scaled = normalized.mul(unit_scale); // [B,512] * [B,512] -> [B,512]
        const modulated = scaled.add(shift); // [B,512] + [B,512] -> [B,512]

        const projected_hidden = self.fc1.forward(modulated, dtype); // [B,d=512] -> [B,dout=512]
        const renamed_hidden = projected_hidden.rename(.{ .dout = .d }); // [B,dout=512] -> [B,d=512]
        const hidden = renamed_hidden.silu(); // [B,d=512] -> [B,d=512]
        const projected_residual = self.fc2.forward(hidden, dtype); // [B,d=512] -> [B,dout=512]
        const residual = projected_residual.rename(.{ .dout = .d }); // [B,dout=512] -> [B,d=512]
        // The gate is raw modulation, not a sigmoid probability.
        const gated = gate.mul(residual); // [B,512] * [B,512] -> [B,512]
        const output = x.add(gated); // [B,512] + [B,512] -> [B,512]
        return output;
    }

    // -- Runtime --

    pub fn load(
        self: *const Gradium_Phonon_TTS_Flow_Res_Block,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_Flow_Res_Block) {
        var buffers = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Flow_Res_Block, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Gradium_Phonon_TTS_Flow_Res_Block, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_Flow_Res_Block)) void {
        zml.Buffer.deinitAll(Gradium_Phonon_TTS_Flow_Res_Block, self);
    }
};

test Gradium_Phonon_TTS_Flow_Res_Block {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 512 }, .f32);
    var input: [2][512]f32 = undefined;
    for (&input) |*row| {
        for (row) |*value| value.* = random.float(f32) * 2 - 1;
    }
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    const condition_shape: zml.Shape = .init(.{ .b = 2, .d = 512 }, .f32);
    var condition: [2][512]f32 = undefined;
    for (&condition) |*row| {
        for (row) |*value| value.* = random.float(f32) * 2 - 1;
    }
    var condition_buffer = try zml.Buffer.fromBytes(io, platform, condition_shape, std.mem.asBytes(&condition));
    defer condition_buffer.deinit();

    const block: Gradium_Phonon_TTS_Flow_Res_Block = .{
        .norm_weight = .fromShape(.init(.{ .d = 512 }, .bf16)),
        .norm_bias = .fromShape(.init(.{ .d = 512 }, .bf16)),
        .fc1 = .init(
            .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
            .fromShape(.init(.{ .dout = 512 }, .bf16)),
            .d,
        ),
        .fc2 = .init(
            .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
            .fromShape(.init(.{ .dout = 512 }, .bf16)),
            .d,
        ),
        .adaln_proj = .init(
            .fromShape(.init(.{ .m = 1536, .d = 512 }, .bf16)),
            .fromShape(.init(.{ .m = 1536 }, .bf16)),
            .d,
        ),
    };
    var norm_weights: [512]zml.floats.BFloat16 = undefined;
    for (&norm_weights) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var norm_weights_buffer = try zml.Buffer.fromBytes(io, platform, block.norm_weight.shape(), std.mem.asBytes(&norm_weights));
    defer norm_weights_buffer.deinit();
    var norm_bias: [512]zml.floats.BFloat16 = undefined;
    for (&norm_bias) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var norm_bias_buffer = try zml.Buffer.fromBytes(io, platform, block.norm_bias.shape(), std.mem.asBytes(&norm_bias));
    defer norm_bias_buffer.deinit();
    var fc1_weights: [512][512]zml.floats.BFloat16 = undefined;
    for (&fc1_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var fc1_weights_buffer = try zml.Buffer.fromBytes(io, platform, block.fc1.weight.shape(), std.mem.asBytes(&fc1_weights));
    defer fc1_weights_buffer.deinit();
    var fc1_bias: [512]zml.floats.BFloat16 = undefined;
    for (&fc1_bias) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var fc1_bias_buffer = try zml.Buffer.fromBytes(io, platform, block.fc1.bias.?.shape(), std.mem.asBytes(&fc1_bias));
    defer fc1_bias_buffer.deinit();
    var fc2_weights: [512][512]zml.floats.BFloat16 = undefined;
    for (&fc2_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var fc2_weights_buffer = try zml.Buffer.fromBytes(io, platform, block.fc2.weight.shape(), std.mem.asBytes(&fc2_weights));
    defer fc2_weights_buffer.deinit();
    var fc2_bias: [512]zml.floats.BFloat16 = undefined;
    for (&fc2_bias) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var fc2_bias_buffer = try zml.Buffer.fromBytes(io, platform, block.fc2.bias.?.shape(), std.mem.asBytes(&fc2_bias));
    defer fc2_bias_buffer.deinit();
    var modulation_weights: [1536][512]zml.floats.BFloat16 = undefined;
    for (&modulation_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var modulation_weights_buffer = try zml.Buffer.fromBytes(io, platform, block.adaln_proj.weight.shape(), std.mem.asBytes(&modulation_weights));
    defer modulation_weights_buffer.deinit();
    var modulation_bias: [1536]zml.floats.BFloat16 = undefined;
    for (&modulation_bias) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var modulation_bias_buffer = try zml.Buffer.fromBytes(io, platform, block.adaln_proj.bias.?.shape(), std.mem.asBytes(&modulation_bias));
    defer modulation_bias_buffer.deinit();
    const weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_Flow_Res_Block) = .{
        .norm_weight = norm_weights_buffer,
        .norm_bias = norm_bias_buffer,
        .fc1 = .{ .weight = fc1_weights_buffer, .bias = fc1_bias_buffer, .quantization = null },
        .fc2 = .{ .weight = fc2_weights_buffer, .bias = fc2_bias_buffer, .quantization = null },
        .adaln_proj = .{ .weight = modulation_weights_buffer, .bias = modulation_bias_buffer, .quantization = null },
    };
    var exe = try platform.compile(allocator, io, block, .forward, .{ zml.Tensor.fromShape(input_shape), zml.Tensor.fromShape(condition_shape) }, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer, condition_buffer });
    defer output.deinit();

    const expected = test_data.res_block;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .d = 512 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, block, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
            zml.Tensor.fromShape(condition_shape),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .d = 512 }, dtype)));
    }
}

// --- Input_Projection ---

/// Maps a candidate audio latent into the flow head's hidden feature space.
///
/// A biased linear layer expands 32 latent coordinates to 512 hidden features.
/// The flow head passes these features to the first residual block.
///
/// Input: floating-point `[..., .ld=32]`. Output: `[..., .d=512]` in the input dtype.
/// Leading axes are preserved; the flow head supplies one candidate per batch
/// row, shaped `[B, 32]`.
///
/// Parameters come from `flow_lm.flow_net.input_proj`: bf16 weight `[512, 32]`
/// and bias `[512]`, converted to the input dtype by forward.
///
/// Reference:
/// ```
//  input_proj: Linear
//  ├── weight: shape=(512, 32), dtype=torch.float32
//  │   ├── loaded: flow_head.input_proj.weight
//  │   └── checkpoint: flow_lm.flow_net.input_proj.weight
//  └── bias: shape=(512,), dtype=torch.float32
//      ├── loaded: flow_head.input_proj.bias
//      └── checkpoint: flow_lm.flow_net.input_proj.bias
/// ```
pub const Input_Projection = struct {
    projection: zml.nn.Linear,

    /// Expects a checkpoint view at flow_lm.flow_net.input_proj.
    pub fn init(store: zml.io.TensorStore.View) Input_Projection {
        return .{ .projection = .init(
            store.createReplicatedTensor("weight", .{ .d, .ld }),
            store.createReplicatedTensor("bias", .{.d}),
            .ld,
        ) };
    }

    // -- Description --

    /// Projects a candidate audio latent into flow hidden features.
    ///
    /// `input`: floating-point `[..., 32]`, the sampler's current candidate latent, passed
    /// by the flow head before any residual blocks. In the current one-step
    /// sampler this is fresh noise shaped `[B, 32]`. This method tags the final
    /// axis `.ld`; any leading batch or sequence axes are preserved.
    ///
    /// Returns floating-point `[..., .d=512]`, candidate features for the first residual
    /// block, computed by a biased linear layer without activation or normalization.
    ///
    /// Parameters are converted to the input dtype; the output keeps that dtype.
    ///
    /// Dimension tags:
    /// - `.ld`: latent dimension; 32 candidate audio features, contracted by the weight.
    /// - `.d`: hidden dimension; 512 output features with a learned bias.
    /// Shape: floating-point `[..., .ld=32]` -> floating-point `[..., .d=512]`.
    pub fn forward(self: Input_Projection, input: zml.Tensor) zml.Tensor {
        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat());
        std.debug.assert(input.rank() >= 1);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Input_Projection", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.ld}); // [...,32] -> [...,ld=32]
        const projected = self.projection.forward(x, dtype); // [...,ld=32] -> [...,d=512]
        return projected;
    }

    // -- Runtime --

    pub fn load(
        self: *const Input_Projection,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Input_Projection) {
        var buffers = try zml.mem.bufferize(allocator, Input_Projection, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Input_Projection, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Input_Projection)) void {
        zml.Buffer.deinitAll(Input_Projection, self);
    }
};

test Input_Projection {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    var rng: std.Random.DefaultPrng = .init(42);

    const projection: Input_Projection = .{ .projection = .init(
        .fromShape(.init(.{ .d = 512, .ld = 32 }, .bf16)),
        .fromShape(.init(.{ .d = 512 }, .bf16)),
        .ld,
    ) };

    const input_shape: zml.Shape = .init(.{ .b = 2, .s = 2, .ld = 32 }, .f32);
    const random = rng.random();
    var input: [2][2][32]f32 = undefined;
    for (&input) |*batch| {
        for (batch) |*position| {
            for (position) |*value| {
                value.* = random.float(f32) * 2 - 1;
            }
        }
    }

    var weight_values: [512][32]zml.floats.BFloat16 = undefined;
    for (&weight_values) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var bias_values: [512]zml.floats.BFloat16 = undefined;
    for (&bias_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);

    var weight = try zml.Buffer.fromBytes(io, platform, projection.projection.weight.shape(), std.mem.asBytes(&weight_values));
    defer weight.deinit();
    var bias = try zml.Buffer.fromBytes(io, platform, projection.projection.bias.?.shape(), std.mem.asBytes(&bias_values));
    defer bias.deinit();

    const weight_buffers: zml.Bufferized(Input_Projection) = .{ .projection = .{ .weight = weight, .bias = bias, .quantization = null } };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, projection, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.input_projection;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .s = 2, .d = 512 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, projection, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .s = 2, .d = 512 }, dtype)));
    }
}

// --- Condition_Projection ---

/// Maps transformer context into the flow head's conditioning feature space.
///
/// A biased linear layer transforms the final normalized transformer features.
/// The flow head adds the averaged start/end time embeddings to this projection;
/// the resulting condition guides every residual block and the final layer.
///
/// Input and output: floating-point `[..., .d=512]`, preserving dtype and leading axes.
/// During generation, the input is `[B, 512]`, selected from the last transformer
/// position by the prediction wrapper.
///
/// Parameters come from `flow_lm.flow_net.cond_embed`: bf16 weight `[512, 512]`
/// and bias `[512]`, converted to the input dtype by forward.
///
/// Reference:
/// ```
//  cond_proj: Linear
//  ├── weight: shape=(512, 512), dtype=torch.float32
//  │   ├── loaded: flow_head.cond_proj.weight
//  │   └── checkpoint: flow_lm.flow_net.cond_embed.weight
//  └── bias: shape=(512,), dtype=torch.float32
//      ├── loaded: flow_head.cond_proj.bias
//      └── checkpoint: flow_lm.flow_net.cond_embed.bias
/// ```
pub const Condition_Projection = struct {
    projection: zml.nn.Linear,

    /// Expects a checkpoint view at flow_lm.flow_net.cond_embed.
    pub fn init(store: zml.io.TensorStore.View) Condition_Projection {
        return .{ .projection = .init(
            store.createReplicatedTensor("weight", .{ .dout, .d }),
            store.createReplicatedTensor("bias", .{.dout}),
            .d,
        ) };
    }

    // -- Description --

    /// Projects transformer context into flow conditioning features.
    ///
    /// `input`: floating-point `[..., 512]`, final normalized transformer hidden states,
    /// passed through the flow head's `condition` parameter. During generation,
    /// the prediction wrapper supplies the last position's features as `[B, 512]`,
    /// encoding the text, voice, and audio context visible there. This method tags
    /// the final axis `.d` and preserves any leading batch or sequence axes.
    ///
    /// Returns floating-point `[..., .d=512]`, a biased linear projection. The flow head adds
    /// the averaged time embeddings before passing it to the blocks and final layer.
    ///
    /// Parameters are converted to the input dtype; the output keeps that dtype.
    ///
    /// Dimension tags:
    /// - `.d`: model dimension; 512 transformer features, contracted by the weight.
    /// - `.dout`: 512 projected conditioning features, renamed to `.d` on output.
    /// Shape: floating-point `[..., .d=512]` -> floating-point `[..., .d=512]`.
    pub fn forward(self: Condition_Projection, input: zml.Tensor) zml.Tensor {
        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat());
        std.debug.assert(input.rank() >= 1);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Condition_Projection", .{});
        defer compiler.popLocation();

        const x = input.withPartialTags(.{.d}); // [...,512] -> [...,d=512]
        const projected = self.projection.forward(x, dtype); // [...,d=512] -> [...,dout=512]
        const output = projected.rename(.{ .dout = .d }); // [...,dout=512] -> [...,d=512]
        return output;
    }

    // -- Runtime --

    pub fn load(
        self: *const Condition_Projection,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Condition_Projection) {
        var buffers = try zml.mem.bufferize(allocator, Condition_Projection, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Condition_Projection, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Condition_Projection)) void {
        zml.Buffer.deinitAll(Condition_Projection, self);
    }
};

test Condition_Projection {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    var rng: std.Random.DefaultPrng = .init(42);

    const projection: Condition_Projection = .{ .projection = .init(
        .fromShape(.init(.{ .dout = 512, .d = 512 }, .bf16)),
        .fromShape(.init(.{ .dout = 512 }, .bf16)),
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

    var weight_values: [512][512]zml.floats.BFloat16 = undefined;
    for (&weight_values) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var bias_values: [512]zml.floats.BFloat16 = undefined;
    for (&bias_values) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);

    var weight = try zml.Buffer.fromBytes(io, platform, projection.projection.weight.shape(), std.mem.asBytes(&weight_values));
    defer weight.deinit();
    var bias = try zml.Buffer.fromBytes(io, platform, projection.projection.bias.?.shape(), std.mem.asBytes(&bias_values));
    defer bias.deinit();

    const weight_buffers: zml.Bufferized(Condition_Projection) = .{ .projection = .{ .weight = weight, .bias = bias, .quantization = null } };
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();

    var exe = try platform.compile(allocator, io, projection, .forward, .{zml.Tensor.fromShape(input_shape)}, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer });
    defer output.deinit();

    const expected = test_data.condition_projection;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .s = 2, .d = 512 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, projection, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .s = 2, .d = 512 }, dtype)));
    }
}

// --- Gradium_Phonon_TTS_Flow_Final_Layer ---

/// Converts the final flow hidden features into an audio latent velocity.
///
/// Normalizes the last residual block's output, applies a condition-derived
/// shift and scale, and projects the result from 512 features to 32 latent
/// velocity components. The sampler uses this velocity to update its candidate.
///
/// Input and condition: floating-point `[B, 512]`. Output: `[B, 32]` in the input dtype.
/// The condition is the same combined transformer/time vector supplied to all
/// six residual blocks. LayerNorm has no learned scale or bias here; its
/// epsilon is 1e-6. Modulation uses `normalized * (1 + scale) + shift`.
///
/// Parameters come from `flow_lm.flow_net.final_layer`: `adaLN_modulation.1`
/// (512 -> 1024, producing shift and scale after SiLU) and `linear` (512 -> 32).
/// Stored bf16 parameters are converted to the input dtype.
///
/// Reference:
/// ```
// GradiumPhononTTSFlowFinalLayer
// ├── norm: GradiumPhononTTSFlowLayerNorm
// ├── adaln_proj: Linear
// │   ├── weight: shape=(1024, 512), dtype=torch.float32
// │   │   ├── loaded: flow_head.final_layer.adaln_proj.weight
// │   │   └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.weight
// │   └── bias: shape=(1024,), dtype=torch.float32
// │       ├── loaded: flow_head.final_layer.adaln_proj.bias
// │       └── checkpoint: flow_lm.flow_net.final_layer.adaLN_modulation.1.bias
// ├── act_fn: SiLUActivation
// └── proj: Linear
//     ├── weight: shape=(32, 512), dtype=torch.float32
//     │   ├── loaded: flow_head.final_layer.proj.weight
//     │   └── checkpoint: flow_lm.flow_net.final_layer.linear.weight
//     └── bias: shape=(32,), dtype=torch.float32
//         ├── loaded: flow_head.final_layer.proj.bias
//         └── checkpoint: flow_lm.flow_net.final_layer.linear.bias
/// ```
pub const Gradium_Phonon_TTS_Flow_Final_Layer = struct {
    adaln_proj: zml.nn.Linear,
    proj: zml.nn.Linear,
    eps: f32 = 1e-6,

    /// Expects a view at flow_lm.flow_net.final_layer.
    pub fn init(store: zml.io.TensorStore.View) Gradium_Phonon_TTS_Flow_Final_Layer {
        return .{
            .adaln_proj = .init(
                store.createReplicatedTensor("adaLN_modulation.1.weight", .{ .m, .d }),
                store.createReplicatedTensor("adaLN_modulation.1.bias", .{.m}),
                .d,
            ),
            .proj = .init(
                store.createReplicatedTensor("linear.weight", .{ .ld, .d }),
                store.createReplicatedTensor("linear.bias", .{.ld}),
                .d,
            ),
        };
    }

    /// Normalizes and conditions the final hidden features, then predicts velocity.
    ///
    /// `input`: floating-point `[B, 512]`, the candidate's hidden features produced by the
    /// sixth and last flow residual block.
    /// `condition`: floating-point `[B, 512]`, projected transformer context plus the averaged
    /// start/end time embeddings, computed by the flow head. This is the same
    /// condition supplied to each residual block.
    /// Batch rows correspond one-to-one; this method assigns tags `.b` and `.d`.
    ///
    /// Returns floating-point `[B, 32]`, the latent velocity consumed by the sampler.
    /// LayerNorm has no learned weight or bias here. SiLU -> Linear produces the
    /// condition-derived shift and scale; there is no gate or residual addition.
    ///
    /// The input determines the computation and output dtype. The condition and
    /// parameters are converted to that dtype before computation.
    ///
    /// Dimension tags:
    /// - `.b`: batch; independent candidate frames, with one condition per row.
    /// - `.d`: 512 hidden features in both input and combined transformer/time
    ///   condition. Normalization reduces only this axis, separately per row.
    /// - `.m`: 1024 modulation features; consecutive 512-feature shift and scale
    ///   vectors, each renamed to `.d` after splitting.
    /// - `.ld`: latent dimension; 32 velocity components produced by the final
    ///   biased projection. The sampler uses these to update the candidate latent.
    ///
    /// Shapes: two floating-point `[.b, .d=512]` inputs -> floating-point `[.b, .ld=32]` output.
    pub fn forward(self: Gradium_Phonon_TTS_Flow_Final_Layer, input: zml.Tensor, condition: zml.Tensor) zml.Tensor {
        const dtype = input.dtype();
        std.debug.assert(dtype.isFloat() and condition.dtype().isFloat());
        std.debug.assert(input.rank() == 2 and condition.rank() == 2);

        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), "Gradium_Phonon_TTS_Flow_Final_Layer", .{});
        defer compiler.popLocation();

        const x = input.withTags(.{ .b, .d }); // [B,512] -> [b=B,d=512]
        const c = condition.convert(dtype).withTags(.{ .b, .d }); // [B,512] -> [b=B,d=512]
        std.debug.assert(x.dim(.b) == c.dim(.b));
        std.debug.assert(x.dim(.d) == c.dim(.d));

        const activated_condition = c.silu(); // [B,d=512] -> [B,d=512]
        const projected_condition = self.adaln_proj.forward(activated_condition, dtype); // [B,d=512] -> [B,m=1024]
        const modulation = projected_condition.chunkExact(.m, 2); // [B,m=1024] -> 2 x [B,m=512]
        const shift = modulation[0].rename(.{ .m = .d }); // [B,m=512] -> [B,d=512]
        const scale = modulation[1].rename(.{ .m = .d }); // [B,m=512] -> [B,d=512]

        // Population variance divides by the full feature count, matching Python.
        const mean = x.mean(.d); // [B,d=512] -> [B,d=1]
        const broadcast_mean = mean.broad(x.shape()); // [B,d=1] -> [B,d=512]
        const centered = x.sub(broadcast_mean); // [B,512] - [B,512] -> [B,512]
        const squared = centered.mul(centered); // [B,512] * [B,512] -> [B,512]
        const variance = squared.mean(.d); // [B,d=512] -> [B,d=1]
        const stabilized = variance.addConstant(self.eps); // [B,1] + scalar -> [B,1]
        const inverse_std = stabilized.rsqrt(); // [B,1] -> [B,1]
        const broadcast_inverse_std = inverse_std.broad(x.shape()); // [B,d=1] -> [B,d=512]
        // The final layer has no learned normalization gain or bias.
        const normalized = centered.mul(broadcast_inverse_std); // [B,512] * [B,512] -> [B,512]

        const unit_scale = scale.addConstant(1); // [B,512] + scalar -> [B,512]
        const scaled = normalized.mul(unit_scale); // [B,512] * [B,512] -> [B,512]
        const modulated = scaled.add(shift); // [B,512] + [B,512] -> [B,512]

        const velocity = self.proj.forward(modulated, dtype); // [B,d=512] -> [B,ld=32]
        return velocity;
    }

    // -- Runtime --

    pub fn load(
        self: *const Gradium_Phonon_TTS_Flow_Final_Layer,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(Gradium_Phonon_TTS_Flow_Final_Layer) {
        var buffers = try zml.mem.bufferize(allocator, Gradium_Phonon_TTS_Flow_Final_Layer, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, Gradium_Phonon_TTS_Flow_Final_Layer, self, &buffers, store, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gradium_Phonon_TTS_Flow_Final_Layer)) void {
        zml.Buffer.deinitAll(Gradium_Phonon_TTS_Flow_Final_Layer, self);
    }
};

test Gradium_Phonon_TTS_Flow_Final_Layer {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    const layer: Gradium_Phonon_TTS_Flow_Final_Layer = .{
        .adaln_proj = .init(
            .fromShape(.init(.{ .m = 1024, .d = 512 }, .bf16)),
            .fromShape(.init(.{ .m = 1024 }, .bf16)),
            .d,
        ),
        .proj = .init(
            .fromShape(.init(.{ .ld = 32, .d = 512 }, .bf16)),
            .fromShape(.init(.{ .ld = 32 }, .bf16)),
            .d,
        ),
    };
    const input_shape: zml.Shape = .init(.{ .b = 2, .d = 512 }, .f32);
    var input: [2][512]f32 = undefined;
    for (&input) |*row| {
        for (row) |*value| value.* = random.float(f32) * 2 - 1;
    }
    var input_buffer = try zml.Buffer.fromBytes(io, platform, input_shape, std.mem.asBytes(&input));
    defer input_buffer.deinit();
    const condition_shape: zml.Shape = .init(.{ .b = 2, .d = 512 }, .f32);
    var condition: [2][512]f32 = undefined;
    for (&condition) |*row| {
        for (row) |*value| value.* = random.float(f32) * 2 - 1;
    }
    var condition_buffer = try zml.Buffer.fromBytes(io, platform, condition_shape, std.mem.asBytes(&condition));
    defer condition_buffer.deinit();
    var modulation_weights: [1024][512]zml.floats.BFloat16 = undefined;
    for (&modulation_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var modulation_weights_buffer = try zml.Buffer.fromBytes(io, platform, layer.adaln_proj.weight.shape(), std.mem.asBytes(&modulation_weights));
    defer modulation_weights_buffer.deinit();
    var modulation_bias: [1024]zml.floats.BFloat16 = undefined;
    for (&modulation_bias) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var modulation_bias_buffer = try zml.Buffer.fromBytes(io, platform, layer.adaln_proj.bias.?.shape(), std.mem.asBytes(&modulation_bias));
    defer modulation_bias_buffer.deinit();
    var projection_weights: [32][512]zml.floats.BFloat16 = undefined;
    for (&projection_weights) |*row| {
        for (row) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    }
    var projection_weights_buffer = try zml.Buffer.fromBytes(io, platform, layer.proj.weight.shape(), std.mem.asBytes(&projection_weights));
    defer projection_weights_buffer.deinit();
    var projection_bias: [32]zml.floats.BFloat16 = undefined;
    for (&projection_bias) |*value| value.* = .fromF32(random.float(f32) * 2 - 1);
    var projection_bias_buffer = try zml.Buffer.fromBytes(io, platform, layer.proj.bias.?.shape(), std.mem.asBytes(&projection_bias));
    defer projection_bias_buffer.deinit();
    const weight_buffers: zml.Bufferized(Gradium_Phonon_TTS_Flow_Final_Layer) = .{
        .adaln_proj = .{ .weight = modulation_weights_buffer, .bias = modulation_bias_buffer, .quantization = null },
        .proj = .{ .weight = projection_weights_buffer, .bias = projection_bias_buffer, .quantization = null },
    };
    var exe = try platform.compile(allocator, io, layer, .forward, .{ zml.Tensor.fromShape(input_shape), zml.Tensor.fromShape(condition_shape) }, .{});
    defer exe.deinit();
    var output = try exe.eval(allocator, io, .{ weight_buffers, input_buffer, condition_buffer });
    defer output.deinit();
    const expected = test_data.final_layer;
    const expected_slice = zml.Slice.init(.init(.{ .b = 2, .ld = 32 }, .f32), std.mem.asBytes(&expected));
    try zml.testing.expectClose(io, expected_slice, output, .{
        .absolute_tolerance = 1e-4,
        .relative_tolerance = 1e-4,
        .minimum_close_fraction = 1,
    });

    for ([_]zml.DataType{ .f16, .bf16, .f64 }) |dtype| {
        var typed_exe = try platform.compile(allocator, io, layer, .forward, .{
            zml.Tensor.fromShape(input_shape.withDtype(dtype)),
            zml.Tensor.fromShape(condition_shape),
        }, .{});
        defer typed_exe.deinit();
        try std.testing.expect(typed_exe.output_shapes[0].eqlWithTags(.init(.{ .b = 2, .ld = 32 }, dtype)));
    }
}
