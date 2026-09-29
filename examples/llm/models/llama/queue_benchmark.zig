const std = @import("std");
const zml = @import("zml");
const inference = @import("inference.zig");
const model = @import("model.zig");
const Session = @import("session.zig").Session;

// Fixed-length autoregressive diagnostic, including token readback. EOS does
// not terminate the workload. Every run restarts from the same prompt/seed.
pub fn run(session: *Session, prompt: []const u32, iterations: usize) !void {
    const warmups = 4;
    if (prompt.len >= session.seqlen) return error.InvalidQueueBenchmarkLength;
    const available: usize = session.seqlen - prompt.len;
    if (iterations == 0 or available <= warmups or iterations > available - warmups) return error.InvalidQueueBenchmarkLength;
    if (session.compiled_model.params.attention_metadata != .furiosa_fa) return error.QueueBenchmarkRequiresFuriosaAttention;
    const allocator = session.allocator;
    const io = session.io;
    const platform = session.platform;
    const params = session.compiled_model.params;
    const calls = warmups + iterations;
    const cache_shape = params.kv_cache.k.shape();
    const zero_cache = try allocator.alloc(u8, cache_shape.byteSize());
    defer allocator.free(zero_cache);
    @memset(zero_cache, 0);
    var expected_tokens: ?[]u32 = null;
    defer if (expected_tokens) |v| allocator.free(v);
    var expected_state: [3]?[]u8 = @splat(null);
    defer for (expected_state) |v| {
        if (v) |bytes| allocator.free(bytes);
    };
    const vocabulary: u32 = @intCast(session.compiled_model.loaded_model.inner.model.embed_tokens.weight.dim(.voc));
    // Rotate order so each depth occupies every position once.
    const orders = [3][3]usize{ .{ 1, 2, 8 }, .{ 2, 8, 1 }, .{ 8, 1, 2 } };
    for (orders, 0..) |order, trial| {
        for (order) |depth| {
            const replacements = blk: {
                var cache: model.KvCache.Buffer = cache: {
                    var key = try zml.Buffer.fromBytes(io, platform, cache_shape, params.shardings.model, zero_cache);
                    errdefer key.deinit();
                    const value = try zml.Buffer.fromBytes(io, platform, cache_shape, params.shardings.model, zero_cache);
                    break :cache .{ .k = key, .v = value };
                };
                errdefer model.KvCache.deinitBuffer(&cache);
                const rng = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, 0);
                break :blk .{ cache, rng };
            };
            model.KvCache.deinitBuffer(&session.kv_cache_buffers);
            zml.Tensor.Rng.deinitBuffer(&session.rng_buffers);
            session.kv_cache_buffers = replacements[0];
            session.rng_buffers = replacements[1];
            try session.runPrefill(prompt);
            var token = try zml.Buffer.fromBytes(io, platform, .init(.{ .s = 1 }, .u32), .replicated, std.mem.asBytes(&session.last_generated_token));
            defer token.deinit();
            var metadata = try params.attention_metadata.initBuffer(io, platform, params.shardings.model);
            defer zml.attention.Metadata.deinitBuffer(&metadata);
            const ranks = token._shards.len;
            const values = try allocator.alloc(u32, calls * ranks);
            defer allocator.free(values);
            @memset(values, 0xffffffff);
            const events = try allocator.alloc(?*zml.pjrt.Event, calls * ranks);
            defer allocator.free(events);
            @memset(events, null);
            // Host destinations and event handles survive every queued copy,
            // including on error. Copy requests precede subsequent donation.
            defer drain(io, platform, events) catch {};
            var start: std.Io.Timestamp = undefined;
            var enqueue_ns: i96 = 0;
            for (0..calls) |i| {
                if (i == warmups) start = .now(io, .awake);
                const submitted: std.Io.Timestamp = .now(io, .awake);
                inference.run(&session.decode, .{
                    .io = io,
                    .tokens_buf = &token,
                    .token_index_buf = &session.token_index_buffers[prompt.len + i],
                    .kv_cache_buffers = &session.kv_cache_buffers,
                    .rng_buffers = &session.rng_buffers,
                    .attention_metadata_buffers = &metadata,
                });
                if (token._shards.len != ranks) return error.QueueBenchmarkShardingChanged;
                for (token._shards.constSlice(), 0..) |shard, rank| {
                    events[i * ranks + rank] = try shard.toHostBuffer(platform.pjrt_api, std.mem.asBytes(&values[i * ranks + rank]));
                }
                if (i >= warmups) enqueue_ns += submitted.untilNow(io, .awake).toNanoseconds();
                if (i < warmups or (i + 1 - warmups) % depth == 0 or i + 1 == calls) try drain(io, platform, events);
            }
            const seconds = @as(f64, @floatFromInt(start.untilNow(io, .awake).toNanoseconds())) / 1e9;
            for (values) |value| if (value >= vocabulary) return error.InvalidQueuedToken;
            if (expected_tokens) |expected| {
                try std.testing.expectEqualSlices(u32, expected, values);
            } else {
                expected_tokens = try allocator.dupe(u32, values);
                std.log.info("Queued reference token IDs (step-major, {} ranks): {any}", .{ ranks, values });
            }
            for (0..calls) |i| for (0..ranks) |rank| {
                try std.testing.expectEqual(values[i * ranks], values[i * ranks + rank]);
            };
            for ([_]zml.Buffer{ session.kv_cache_buffers.k, session.kv_cache_buffers.v, session.rng_buffers._state }, 0..) |buffer, index| {
                const host = try buffer.toSliceAlloc(allocator, io);
                defer host.free(allocator);
                if (expected_state[index]) |expected| {
                    try std.testing.expectEqualSlices(u8, expected, host.data());
                } else {
                    expected_state[index] = try allocator.dupe(u8, host.data());
                }
            }
            std.log.info("Queued decode trial {} depth {}: {} actual decode steps, {d:.3} s, {d:.3} steps/s, enqueue {d:.3} ms/step; all token ranks, final K/V and RNG exact; seed=0, EOS ignored", .{ trial + 1, depth, iterations, seconds, @as(f64, @floatFromInt(iterations)) / seconds, @as(f64, @floatFromInt(enqueue_ns)) / @as(f64, @floatFromInt(iterations)) / 1e6 });
        }
    }
}

fn drain(io: std.Io, platform: *const zml.Platform, events: []?*zml.pjrt.Event) !void {
    var first_error: ?anyerror = null;
    for (events) |*slot| {
        if (slot.*) |event| {
            event.await(platform.pjrt_api, io) catch |err| {
                if (first_error == null) first_error = err;
            };
            event.deinit(platform.pjrt_api);
            slot.* = null;
        }
    }
    if (first_error) |err| return err;
}
