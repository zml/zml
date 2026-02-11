const std = @import("std");
const builtin = @import("builtin");
const log = std.log;
const cfg = @import("config.zig");
const MelSpectrumConfig = cfg.MelSpectrumConfig;

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

    // const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    // var model_registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    var model_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.model);
    defer model_registry.deinit();

    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    const melspectrum_config = MelSpectrumConfig{};

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    log.info("Selected platform {f}\n", .{platform.fmtVerbose()});

    // Mel spectrogram test
    var melspectro_model: LogMelSpectrogram = .init(melspectrum_config);
    var mel_spectrum_buffers_future = try io.concurrent(LogMelSpectrogram.load, .{&melspectro_model, io, platform});
    var mel_spectrum_buffers = try mel_spectrum_buffers_future.await(io);
    defer LogMelSpectrogram.unload(&mel_spectrum_buffers);

    try zml.testing.testLayer(allocator, io, platform, MelTestHarness{ .inner = melspectro_model }, .forward, ref_store.view(), "mel", .{ .inner = mel_spectrum_buffers }, 1e-3);

    // Encoder conv stem test
    const encoder_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder";
    const encoder_model = Encoder.init(model_store.view().withPrefix(encoder_prefix));
    var encoder_buffers = try zml.io.load(Encoder, &encoder_model, allocator, io, platform, .{
        .store = &model_store,
        .parallelism = 16,
        .dma_chunks = 32,
        .dma_chunk_size = 128 * zml.MiB,
    });
    defer Encoder.unloadBuffers(&encoder_buffers);

    try zml.testing.testLayer(allocator, io, platform, EncoderTestHarness{ .inner = encoder_model }, .forward, ref_store.view(), "conv_stem", .{ .inner = encoder_buffers }, 1e-3);
}

const MelTestHarness = struct {
    inner: LogMelSpectrogram,

    pub fn forward(self: MelTestHarness, waveform: Tensor) Tensor {
        return self.inner.forward(waveform.withTags(.{.samples}));
    }
};

const EncoderTestHarness = struct {
    inner: Encoder,

    pub fn forward(self: EncoderTestHarness, mel_input: Tensor) Tensor {
        return self.inner.forward(mel_input.withTags(.{ .channels, .time }));
    }
};
