const std = @import("std");
const zml = @import("zml");

const Pipeline = @import("pipeline.zig").Pipeline;
const main = @import("main.zig");

const reference = @import("mimi/root_test_data.zig");

comptime {
    _ = zml.bazel;
}

const long_prompt = "You can open each snapshot in the UI or run SQL against it. This is useful for investigating a long-running server: capture recent history when latency spikes, without recording an enormous file or restarting tracing.";

test "prompt to waveform: empty" {
    try check_prompt("", reference.empty[0][0][0..]);
}

test "prompt to waveform: single character" {
    try check_prompt("A", reference.single_character[0][0][0..]);
}

test "prompt to waveform: Bonjour ZML" {
    try check_prompt("Bonjour ZML", reference.bonjour[0][0][0..]);
}

test "prompt to waveform: long prompt" {
    try check_prompt(long_prompt, reference.long[0][0][0..]);
}

fn check_prompt(prompt: []const u8, expected: []const f32) !void {
    const waveform = try run_pipeline(prompt);
    defer waveform.free(std.testing.allocator);
    try expect_waveform(waveform, expected);
}

fn run_pipeline(prompt: []const u8) !zml.Slice {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (std.mem.trim(u8, prompt, " \t\r\n\x0b\x0c").len == 0)
        return zml.Slice.alloc(allocator, .init(.{ .b = 1, .d = 1, .s = 0 }, .f32));
    const prepared = try main.prepare_prompt(io, allocator);
    defer prepared.deinit(allocator);
    var queue_buf: [main.chunk_queue_capacity]main.Prepared_Chunk = undefined;
    var queue: std.Io.Queue(main.Prepared_Chunk) = .init(&queue_buf);
    var producer = try io.concurrent(main.prepare_chunks, .{ io, allocator, prompt, prepared, &queue, null });
    defer {
        queue.close(io);
        producer.cancel(io) catch {};
        while (queue.getOneUncancelable(io)) |chunk| chunk.deinit(allocator) else |_| {}
    }

    const pipeline = try allocator.create(Pipeline);
    defer allocator.destroy(pipeline);
    try pipeline.init_in_place(io, allocator, platform, prepared.voice, prepared.capacity, .f32, null);
    defer pipeline.deinit();

    // Match playback: each combined device call returns the newest waveform chunk.
    var samples: std.ArrayList(f32) = .empty;
    defer samples.deinit(allocator);
    while (true) {
        const queued = queue.getOne(io) catch |err| switch (err) {
            error.Closed => break,
            else => return err,
        };
        defer queued.deinit(allocator);
        var span = zml.tracer.span_at_for(@src(), "pipeline_test.chunk", .{ .chunk = queued.index }, .{ .consumer = queued.context });
        defer span.end();
        try pipeline.start_chunk(queued.input);
        while (!pipeline.done) {
            const chunk = try pipeline.forward();
            defer chunk.free(allocator);
            try samples.appendSlice(allocator, chunk.constItems(f32));
        }
    }

    try producer.await(io);
    const waveform = try zml.Slice.alloc(allocator, .init(.{ .b = 1, .d = 1, .s = samples.items.len }, .f32));
    @memcpy(waveform.items(f32), samples.items);
    return waveform;
}

fn expect_waveform(actual: zml.Slice, expected: []const f32) !void {
    const shape: zml.Shape = .init(.{ .b = 1, .d = 1, .s = expected.len }, .f32);
    try std.testing.expect(actual.shape.eqlWithTags(shape));
    if (expected.len == 0) return;
    const report = try zml.testing.compareFloats(std.testing.allocator, f32, f32, expected, actual.constItems(f32), .{});
    try std.testing.expect(!report.nan_or_inf);

    var energy: f64 = 0;
    for (expected) |sample| energy += @as(f64, sample) * sample;
    const reference_rms: f32 = @floatCast(@sqrt(energy / @as(f64, @floatFromInt(expected.len))));

    // The accepted waveform error budget is 5% of reference RMS,
    // plus a peak-error bound to catch isolated artifacts. These are numerical
    // regression checks, not a perceptual audio-quality metric.
    try std.testing.expectApproxEqAbs(@as(f32, 0), report.rmse, reference_rms * 0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 0), report.max_absolute_error, 0.1);
}
