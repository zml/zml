const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("audio.zig");
const media = @import("../serve/media.zig");
const pipeline = @import("pipeline.zig");
const vae = @import("geometry.zig");
const weights = @import("../recipe/weights.zig");

const log = std.log.scoped(.minimax_h3);

// =============================================================================
// draft/decode.zig — run the audio VAE on a draft
//
// Called from session.draft in parallel with TAEH3.
// =============================================================================

fn done(io: std.Io, t: std.Io.Timestamp, comptime fmt: []const u8, args: anytype) void {
    log.info(fmt ++ " [{f}]", args ++ .{t.untilNow(io, .awake)});
}

pub fn decodeAudio(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    audio_exe: *zml.FnExe(audio_vae.decode),
    loaded: *const audio_vae.LoadedModel,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    geo: pipeline.Geometry,
    packed_audio: []f32,
    progress: *std.Progress.Node,
    ready_bufs: ?*const zml.Bufferized(audio_vae.Model),
) ![]f32 {
    vae.applyLatentNorm(packed_audio, @intCast(loaded.cfg.latent_channels), &loaded.cfg.latents_mean, &loaded.cfg.latents_std, true);
    const channels: u32 = @intCast(loaded.cfg.latent_channels);
    const t = geo.audio_t;
    const batch = try allocator.alloc(f32, 2 * @as(usize, channels) * t);
    defer allocator.free(batch);
    vae.audioRowsToBct(batch, packed_audio, channels, t);

    var owned_bufs: ?zml.Bufferized(audio_vae.Model) = null;
    defer if (owned_bufs) |*b| audio_vae.Model.unloadBuffers(b, allocator);
    var clock: std.Io.Timestamp = .now(io, .awake);
    const bufs = if (ready_bufs) |b| b.* else blk: {
        log.info("audio: load", .{});
        owned_bufs = try loaded.loadBuffers(allocator, io, platform, store, shardings, progress);
        done(io, clock, "audio: loaded", .{});
        break :blk owned_bufs.?;
    };
    var runner = try zml.FnExe(audio_vae.decode).Runner(.{.model}).init(audio_exe, allocator, .{ .model = bufs });
    defer runner.deinit(allocator);

    var latent_buf = try weights.fromItems(io, platform, .init(.{
        .b = 2,
        .c = loaded.cfg.latent_channels,
        .t = geo.audio_t,
    }, .f32), batch);
    defer latent_buf.deinit();

    var wav: zml.Buffer = undefined;
    clock = .now(io, .awake);
    log.info("audio: run t={d}", .{geo.audio_t});
    runner.run(io, .{
        .inputs = .{ .latents = latent_buf },
        .outputs = .{ .wav = &wav },
        .opts = .{ .wait = true },
    });
    defer wav.deinit();
    done(io, clock, "audio: ran {f}", .{wav.shape()});

    const samples = vae.official_audio.sampleCount(geo.audio_t);
    const host = try allocator.alloc(f32, 2 * samples);
    errdefer allocator.free(host);
    clock = .now(io, .awake);
    log.info("audio: toSlice samples={d}", .{samples});
    try wav.toSlice(io, .init(zml.Shape.init(.{ .b = 2, .c = 1, .t = samples }, .f32), std.mem.sliceAsBytes(host)));
    done(io, clock, "audio: toSlice ok", .{});

    const interleaved = try media.interleaveStereo(allocator, host[0..samples], host[samples..]);
    allocator.free(host);
    return interleaved;
}
