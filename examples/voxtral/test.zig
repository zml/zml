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

const CliArgs = struct {
    reference: []const u8,
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

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.reference);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const melspectrum_config = MelSpectrumConfig{};

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    log.info("Selected platform {f}\n", .{platform.fmtVerbose()});

    var melspectro_model: LogMelSpectrogram = .init(melspectrum_config);
    var mel_spectrum_buffers_future = try io.concurrent(LogMelSpectrogram.load, .{&melspectro_model, io, platform});
    var mel_spectrum_buffers = try mel_spectrum_buffers_future.await(io);
    defer LogMelSpectrogram.unload(&mel_spectrum_buffers);

    try zml.testing.testLayer(allocator, io, platform, MelTestHarness{ .inner = melspectro_model }, .forward, store.view(), "mel", .{ .inner = mel_spectrum_buffers }, 1e-3);
}

const MelTestHarness = struct {
    inner: LogMelSpectrogram,

    pub fn forward(self: MelTestHarness, waveform: Tensor) Tensor {
        return self.inner.forward(waveform.withTags(.{.samples}));
    }
};
