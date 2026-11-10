const std = @import("std");
const c = @import("c");
const Audio_Player = @This();

/// Owns a stationary miniaudio device and a one-second SPSC queue. The calling
/// thread produces mono f32 samples; the device callback consumes them. miniaudio
/// converts the requested sample rate/format to the hardware's native format.
handle: *c.phonon_audio,

pub fn init(sample_rate: u32) !Audio_Player {
    return initBackend(sample_rate, false);
}

fn initBackend(sample_rate: u32, null_backend: bool) !Audio_Player {
    var handle: ?*c.phonon_audio = null;
    try check(c.phonon_audio_create(sample_rate, @intFromBool(null_backend), &handle));
    return .{ .handle = handle.? };
}

/// Copies mono waveform samples into the queue and starts playback on the first
/// nonempty write. Blocks the producer when full; the callback never waits.
pub fn write(self: Audio_Player, samples: []const f32) !void {
    try check(c.phonon_audio_write(self.handle, samples.ptr, samples.len));
}

/// Source samples handed to the device, excluding inserted underrun silence.
/// The hardware and audio server may still buffer these samples before output.
pub fn consumedFrames(self: Audio_Player) u64 {
    return c.phonon_audio_consumed_frames(self.handle);
}

/// Marks EOF, drains queued audio plus a short silent tail, then stops playback.
/// Call once after the final write. Subsequent writes are rejected.
pub fn finish(self: Audio_Player) !void {
    try check(c.phonon_audio_finish(self.handle));
}

/// Stops playback immediately and releases all resources, including on errors.
pub fn deinit(self: Audio_Player) void {
    c.phonon_audio_destroy(self.handle);
}

fn check(result: c_int) !void {
    if (result != 0) {
        std.log.err("miniaudio failed ({d})", .{result});
        return error.AudioPlaybackFailed;
    }
}

test "playback progress excludes underrun silence and resumes with new samples" {
    const io = std.testing.io;
    const player = try initBackend(24_000, true);
    defer player.deinit();
    try std.testing.expectEqual(0, player.consumedFrames());

    const samples: [1920]f32 = @splat(0.1);
    try player.write(&samples);
    const deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromSeconds(2), .clock = .awake });
    while (player.consumedFrames() < samples.len and std.Io.Clock.Timestamp.now(io, .awake).compare(.lt, deadline)) {
        try io.sleep(.fromMilliseconds(2), .awake);
    }
    try std.testing.expectEqual(samples.len, player.consumedFrames());
    try io.sleep(.fromMilliseconds(40), .awake);
    try std.testing.expectEqual(samples.len, player.consumedFrames());

    try player.write(&samples);
    try player.finish();
    try std.testing.expectEqual(2 * samples.len, player.consumedFrames());
}
