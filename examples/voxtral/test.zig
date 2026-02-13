const std = @import("std");
const builtin = @import("builtin");
const log = std.log;
const cfg = @import("config.zig");
const Config = cfg.Config;

const zml = @import("zml");
const stdx = zml.stdx;
const Tensor = zml.Tensor;

const mel = @import("mel_spectrogram.zig");
const LogMelSpectrogram = mel.LogMelSpectrogram;

const enc = @import("encoder.zig");
const Encoder = enc.Encoder;

const dec = @import("decoder.zig");

const CliArgs = struct {
    reference: []const u8,
    model: []const u8,
};

pub fn main() !void {
    log.info("Start of Voxtral test", .{});

    var dbg = std.heap.DebugAllocator(.{}).init;
    defer if (builtin.mode == .Debug) std.debug.assert(dbg.deinit() == .ok);

    const allocator = switch (builtin.mode) {
	.Debug => dbg.allocator(),
	else => std.heap.c_allocator,
    };

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const args = stdx.flags.parseProcessArgs(CliArgs);

    var vfs: zml.io.VFS = try .init(allocator, threaded.io());
    defer vfs.deinit();

    var vfs_file: zml.io.VFS.File = .init(allocator, threaded.io(), .{});
    defer vfs_file.deinit();
    try vfs.register("file", vfs_file.io());

    const io = vfs.io();

    var ref_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.reference);
    defer ref_registry.deinit();

    var ref_store: zml.io.TensorStore = .fromRegistry(allocator, &ref_registry);
    defer ref_store.deinit();

    var model_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.model);
    defer model_registry.deinit();

    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    // Parse config from model directory
    const model_dir = try zml.safetensors.resolveModelRepo(io, args.model);
    var parsed_config = try cfg.parseConfig(allocator, io, model_dir);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    const encoder_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder";
    const load_opts: zml.io.LoadOpts = .{
        .store = &model_store,
        .parallelism = 16,
        .dma_chunks = 32,
        .dma_chunk_size = 128 * zml.MiB,
    };

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    log.info("Selected platform {f}\n", .{platform.fmtVerbose()});

    // -- Mel spectrogram test
    {
        var melspectro_model: LogMelSpectrogram = .init(config);
        var mel_spectrum_buffers_future = try io.concurrent(LogMelSpectrogram.load, .{&melspectro_model, io, platform});
        var mel_spectrum_buffers = try mel_spectrum_buffers_future.await(io);
        defer LogMelSpectrogram.unload(&mel_spectrum_buffers);

        try zml.testing.testLayer(allocator, io, platform, MelTestHarness{ .inner = melspectro_model }, .forward, ref_store.view(), "mel", .{ .inner = mel_spectrum_buffers }, .{});
    }

    // -- Conv stem test
    {
        const conv_store = model_store.view().withPrefix(encoder_prefix).withPrefix("conv_layers");
        const conv_stem = ConvStemTestHarness{
            .conv0 = enc.CausalConv1d.init(conv_store.withLayer(0).withPrefix("conv"), 1),
            .conv1 = enc.CausalConv1d.init(conv_store.withLayer(1).withPrefix("conv"), 2),
        };
        var conv_buffers = try zml.io.load(ConvStemTestHarness, &conv_stem, allocator, io, platform, load_opts);
        defer ConvStemTestHarness.unloadBuffers(&conv_buffers);

        try zml.testing.testLayer(allocator, io, platform, conv_stem, .forward, ref_store.view(), "conv_stem", conv_buffers, .{});
    }
    
    // -- Transformer Layer tests
    // {
    //     const transformer_store = model_store.view().withPrefix(encoder_prefix).withPrefix("transformer").withPrefix("layers");
    //     const enc_cfg = config.encoder();

    //     inline for (0..32) |i| {
    //         if (i < enc_cfg.n_layers) {
    //             const layer = TransformerLayerTestHarness{
    //                 .inner = enc.TransformerLayer.init(transformer_store.withLayer(i), config),
    //                 .config = config,
    //             };
		
    //             var layer_buffers = try zml.io.load(TransformerLayerTestHarness, &layer, allocator, io, platform, load_opts);
    //             defer TransformerLayerTestHarness.unloadBuffers(&layer_buffers);

    //             try zml.testing.testLayer(allocator, io, platform, layer, .forward, ref_store.view(), std.fmt.comptimePrint("encoder.layers.{d}", .{i}), layer_buffers, .{.minimum_close_fraction = 0.9});
    //         }
    //     }
    // }
    
    // -- Encoder test
    // {
    //     const encoder_model = Encoder.init(allocator, model_store.view().withPrefix(encoder_prefix), config);
    //     defer encoder_model.deinit(allocator);
    //     var encoder_buffers = try zml.io.load(Encoder, &encoder_model, allocator, io, platform, load_opts);
    //     defer Encoder.unloadBuffers(&encoder_buffers, allocator);

    //     const harness = EncoderTestHarness{ .inner = encoder_model };
    //     try zml.testing.testLayer(allocator, io, platform, harness, .forward, ref_store.view(), "encoder", .{ .inner = encoder_buffers }, .{});
    // }

    // -- Adapter test
    {
        const adapter = AdapterTestHarness{
            .inner = dec.Adapter.init(model_store.view()),
        };
        var adapter_buffers = try zml.io.load(AdapterTestHarness, &adapter, allocator, io, platform, load_opts);
        defer AdapterTestHarness.unloadBuffers(&adapter_buffers);

        try zml.testing.testLayer(allocator, io, platform, adapter, .forward, ref_store.view(), "adapter", adapter_buffers, .{.minimum_close_fraction = 0.8});
    }

    // -- Decoder Layer tests
    {
        const dec_layer_store = model_store.view().withPrefix("layers");

        // Load t_cond from reference activations (shared across all layers)
        const t_cond_tensor = ref_store.view().withPrefix("t_cond.out").createTensorWithTags("0", .{.d});
        const TCond = struct { t_cond: Tensor };
        const t_cond_spec = TCond{ .t_cond = t_cond_tensor };
        const ref_load_opts: zml.io.LoadOpts = .{
            .store = &ref_store,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 4096,
        };
        var t_cond_buf = try zml.io.load(TCond, &t_cond_spec, allocator, io, platform, ref_load_opts);
        defer t_cond_buf.t_cond.deinit();

        inline for (0..32) |i| {
            if (i < config.n_layers) {
                const layer = DecoderLayerTestHarness{
                    .inner = dec.DecoderLayer.init(dec_layer_store.withLayer(i), config),
                    .t_cond = t_cond_tensor,
                    .config = config,
                };

                var inner_buffers = try zml.io.load(dec.DecoderLayer, &layer.inner, allocator, io, platform, load_opts);
                defer dec.DecoderLayer.unloadBuffers(&inner_buffers);

                const layer_buffers: zml.Bufferized(DecoderLayerTestHarness) = .{
                    .inner = inner_buffers,
                    .t_cond = t_cond_buf.t_cond,
                };

                try zml.testing.testLayer(allocator, io, platform, layer, .forward, ref_store.view(), std.fmt.comptimePrint("decoder.prefill.layers.{d}", .{i}), layer_buffers, .{ .minimum_close_fraction = 0.85 });
            }
        }
    }
}

// ================================================================
// Test harnesses
// ================================================================

const MelTestHarness = struct {
    inner: LogMelSpectrogram,

    pub fn forward(self: MelTestHarness, waveform: Tensor) Tensor {
        return self.inner.forward(waveform.withTags(.{.samples}));
    }
};

const ConvStemTestHarness = struct {
    conv0: enc.CausalConv1d,
    conv1: enc.CausalConv1d,

    pub fn unloadBuffers(self: *zml.Bufferized(ConvStemTestHarness)) void {
        enc.CausalConv1d.unloadBuffers(&self.conv0);
        enc.CausalConv1d.unloadBuffers(&self.conv1);
    }

    pub fn forward(self: ConvStemTestHarness, mel_input: Tensor) Tensor {
        const dtype = self.conv0.weight.dtype();
        var h = mel_input.convert(dtype).withTags(.{ .channels, .time }).insertAxes(.channels, .{.batch});
	
        h = self.conv0.forward(h).gelu();
        h = self.conv1.forward(h).gelu();
        h = h.squeeze(.batch);
	
        return h.transpose(.{ .time, .channels }).convert(.f32);
    }
};

const TransformerLayerTestHarness = struct {
    inner: enc.TransformerLayer,
    config: Config,

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayerTestHarness)) void {
        enc.TransformerLayer.unloadBuffers(&self.inner);
    }

    pub fn forward(self: TransformerLayerTestHarness, h: Tensor) Tensor {
        const dtype = self.inner.attention.wq.weight.dtype();
        return self.inner.forward(h.convert(dtype).withTags(.{ .s, .d }), self.config).convert(.f32);
    }
};

const EncoderTestHarness = struct {
    inner: Encoder,

    pub fn forward(self: EncoderTestHarness, mel_input: Tensor) Tensor {
        return self.inner.forward(mel_input.withTags(.{ .channels, .time })).convert(.f32);
    }
};

const AdapterTestHarness = struct {
    inner: dec.Adapter,

    pub fn unloadBuffers(self: *zml.Bufferized(AdapterTestHarness)) void {
        dec.Adapter.unloadBuffers(&self.inner);
    }

    pub fn forward(self: AdapterTestHarness, encoder_out: Tensor) Tensor {
        return self.inner.forward(encoder_out.convert(.bf16).withTags(.{ .s, .d })).convert(.f32);
    }
};

const DecoderLayerTestHarness = struct {
    inner: dec.DecoderLayer,
    t_cond: Tensor,
    config: Config,

    pub fn forward(self: DecoderLayerTestHarness, h: Tensor) Tensor {
        const dtype = self.inner.attention.wq.weight.dtype();
        const h_typed = h.convert(dtype).withTags(.{ .s, .d });

        // Create zero KV cache for prefill (empty cache for this layer)
        const kv_shape = zml.Shape.init(.{
            .layer = 1,
            .k = h_typed.dim(.s),
            .h = @as(i64, @intCast(self.config.n_kv_heads)),
            .hd = @as(i64, @intCast(self.config.head_dim)),
        }, dtype);
        const kv_cache = dec.KvCache{
            .k = Tensor.zeroes(kv_shape),
            .v = Tensor.zeroes(kv_shape),
            .layer_index = Tensor.scalar(0, .u32),
        };

        const token_index = Tensor.scalar(0, .u32);
        const result = self.inner.forward(h_typed, token_index, kv_cache, self.t_cond.convert(dtype), self.config);
        return result[0].convert(.f32);
    }
};
