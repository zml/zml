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

        try zml.testing.testLayer(allocator, io, platform, MelTestHarness{ .inner = melspectro_model }, .forward, ref_store.view(), "mel", .{ .inner = mel_spectrum_buffers }, 1e-3);
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

        try zml.testing.testLayer(allocator, io, platform, conv_stem, .forward, ref_store.view(), "conv_stem", conv_buffers, 1e-3);
    }
    
    // -- Encoder test
    {
        const encoder_model = Encoder.init(allocator, model_store.view().withPrefix(encoder_prefix), config);
        defer encoder_model.deinit(allocator);
        var encoder_buffers = try zml.io.load(Encoder, &encoder_model, allocator, io, platform, load_opts);
        defer Encoder.unloadBuffers(&encoder_buffers, allocator);

        const harness = EncoderTestHarness{ .inner = encoder_model };
        try zml.testing.testLayer(allocator, io, platform, harness, .forward, ref_store.view(), "encoder", .{ .inner = encoder_buffers }, 1e-3);
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
        return self.inner.forward(mel_input.withTags(.{ .channels, .time }));
    }
};
