const std = @import("std");
const zml = @import("zml");

const Tokenizer = @import("Tokenizer.zig");
const transformer = @import("transformer.zig");
const Model = @import("model.zig").Gradium_Phonon_TTS_For_Conditional_Generation;
const util = @import("util.zig");

pub const std_options: std.Options = .{ .log_level = .info };

// Retain the Zig half of Bazel's startup hook.
comptime {
    _ = zml.bazel;
    _ = Tokenizer;
    _ = transformer;
    _ = @import("model.zig");
}

test {
    std.testing.refAllDecls(@This());
}

// TODO: things yet to cover:
// * batch inputs (i.e. multiple prompts at once)
// * different base models (i.e. something other than phonon-31m-en)
// * different voice features
// * mimi
// * not f32?
// * now that it's ported, we could grab the asset snapshots from the zig graph instead? then there's *really* no python
// * how does one benchmark these things?
// * what's with the implicit errors all over the place?
//
// And, more generally, the big 4:
//
// * Streaming
//   We currently accumulate normalized latents and decode them
//   afterward in Python. To stream speech, the loop would send completed latents
//   to an audio decoder incrementally and deliver waveform chunks while
//   generation continues. That requires managing decoder context, chunk
//   boundaries, and buffering so playback doesn’t starve.
//
// * Batching
//   The flow head can process B > 1, but requests have different prompt lengths
//   and finish at different times. The loop must decide which requests run
//   together, track their cache positions and EOS countdowns separately, and
//   handle a finished row while others continue. Our current single-prompt loop
//   avoids those scheduling questions.
//
// * Latency
//   Prefill determines how soon the first frame can arrive; repeated decode
//   determines whether we can sustain playback. The loop controls compilation
//   reuse, device/host transfers, synchronization, and how much work happens per
//   call. For example, our current code reads velocity back and adds it to noise
//   on the host; performing that addition in the graph could reduce host work
//   and transfers.
//
// * Cancellation
//   When someone interrupts speech, the loop must stop scheduling further
//   steps, discard queued audio, and release the request’s buffers safely. If
//   a device call is already running, cancellation may mean letting that call
//   finish and ignoring its result. Cache donation makes ownership during that
//   cleanup particularly relevant.

pub fn main(init: std.process.Init) !void {
    const io_base = init.io;
    const allocator = init.gpa;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io_base, &stdout_buffer);

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("Usage: {s} <text>\n", .{args[0]});
        std.process.exit(1);
    }
    const text = try std.mem.join(allocator, " ", args[1..]);
    defer allocator.free(text);

    const io = io_base;
    const tokenizer_path = try util.assetPath(allocator, util.gradium_phonon_31m_en_tokenizer);
    defer allocator.free(tokenizer_path);
    const voice_path = try util.assetPath(allocator, util.gradium_phonon_voice);
    defer allocator.free(voice_path);
    const model_path = try util.assetPath(allocator, util.gradium_phonon_31m_en);
    defer allocator.free(model_path);

    var tokenizer = try Tokenizer.init(allocator, io, tokenizer_path);
    defer tokenizer.deinit();
    const voice = try load_voice(allocator, io, voice_path);
    defer voice.free(allocator);
    const voice_frames: usize = @intCast(voice.shape.dim(.p));
    const tokens = try tokenizer.encodePrompt(allocator, text, voice_frames);
    defer tokens.free(allocator);

    try stdout.interface.print("Token IDs: {any}\n", .{tokens.constItems(i64)});
    try stdout.interface.flush();

    var platform: *zml.Platform = try .auto(allocator, io_base, .{});
    defer platform.deinit(allocator, io_base);

    var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, model_path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &.{});
    defer store.deinit();
    const model: Model = .init(store.view());

    var weights = try model.load(allocator, io, platform, &store);
    defer Model.unloadBuffers(&weights);
    var voice_buffer = try zml.Buffer.fromSlice(io, platform, voice);
    defer voice_buffer.deinit();

    // Match this checkpoint's reference frame budget and short-text EOS tail.
    var text_count: usize = 0;
    var text_ids: std.ArrayList(u32) = .empty;
    defer text_ids.deinit(allocator);
    for (tokens.constItems(i64)) |id| {
        if (id == 4 or id == 5 or id == 6) continue;
        text_count += 1;
        try text_ids.append(allocator, @intCast(id));
    }
    var decoder = try tokenizer.inner.decoder();
    defer decoder.deinit();
    var decoded = try decoder.decodeAlloc(allocator, text_ids.items);
    defer decoded.deinit(allocator);
    var words = std.mem.tokenizeAny(u8, decoded.items, " \t\r\n\x0b\x0c");
    var word_count: usize = 0;
    while (words.next() != null) word_count += 1;
    var generation = try Generation.init(allocator, tokens.constItems(i64), .{
        .max_frames = @intFromFloat(@ceil((@as(f64, @floatFromInt(text_count)) / 3 + 2) * 12.5)),
        .frames_after_eos = if (word_count <= 4) 3 else 1,
    });
    defer generation.deinit(allocator);
    var cached = try CachedGeneration.init(allocator, io, platform, model, tokens, voice_buffer, generation.settings.max_frames);
    defer cached.deinit();
    var rng: std.Random.DefaultPrng = .init(42);
    const random = rng.random();
    while (!generation.done) {
        var noise: [32]f32 = undefined;
        for (&noise) |*value| value.* = random.floatNorm(f32) * @sqrt(@as(f32, 0.3));
        const score = try cached.step(&generation, allocator, io, platform, weights, voice_buffer, noise);
        try stdout.interface.print("Frame {d}: EOS {d:.5}, sequence length {d}\n", .{ generation.frames(), score, generation.ids.items.len });
        try stdout.interface.flush();
    }
    // Bazel runs from its runfiles tree; save beside the caller's working directory.
    const output_path = try std.fs.path.join(allocator, &.{ init.environ_map.get("BUILD_WORKING_DIRECTORY") orelse ".", util.generated_latents_path });
    defer allocator.free(output_path);
    try saveLatents(allocator, io, output_path, generation.latents.items);
    try stdout.interface.print("Generated normalized latents [1,{d},32] f32; stopped by {s}. Saved {s}\n", .{
        generation.frames(), if (generation.stopped_by_eos) "EOS" else "frame limit", output_path,
    });
    try stdout.interface.flush();
}

/// Tracks a single prompt's completed audio frames and stopping state.
///
/// `ids` begins with tokenizer-produced prompt IDs and gains one ID 6 per
/// completed frame. `latents` stores normalized f32 history `[1, A, 32]` in
/// frame order. Each uncached `step` borrows weights and voice buffers, compiles
/// a full-prefix prediction, and consumes caller-supplied noise `[1, 32]`.
/// The sampler adds velocity to noise, retains the resulting frame, and applies
/// the EOS threshold, minimum length, latched tail, and maximum-frame limit.
/// The stopping frame and requested tail remain in output history.
const Generation = struct {
    ids: std.ArrayList(i64),
    latents: std.ArrayList(f32) = .empty,
    settings: Settings,
    remaining: ?usize = null,
    done: bool = false,
    stopped_by_eos: bool = false,

    const Settings = struct {
        max_frames: usize,
        eos_threshold: f32 = 4.0,
        min_frames_before_eos: usize = 0,
        frames_after_eos: usize = 1,
    };

    fn init(allocator: std.mem.Allocator, prompt: []const i64, settings: Settings) !Generation {
        if (prompt.len == 0 or settings.max_frames == 0) return error.InvalidGenerationInputs;
        var self: Generation = .{ .ids = .empty, .settings = settings };
        try self.ids.appendSlice(allocator, prompt);
        return self;
    }

    fn deinit(self: *Generation, allocator: std.mem.Allocator) void {
        self.ids.deinit(allocator);
        self.latents.deinit(allocator);
    }

    fn frames(self: Generation) usize {
        return self.latents.items.len / 32;
    }

    // Called after retaining the sampled frame, matching Python's criterion.
    fn finishFrame(self: *Generation, score: f32) void {
        const frame_index = self.frames() - 1;
        if (self.remaining == null and frame_index >= self.settings.min_frames_before_eos and score > self.settings.eos_threshold)
            self.remaining = self.settings.frames_after_eos;
        if (self.remaining) |remaining| {
            if (remaining == 0) {
                self.done = true;
                self.stopped_by_eos = true;
            } else self.remaining = remaining - 1;
        }
        if (self.frames() >= self.settings.max_frames) self.done = true;
    }

    fn step(
        self: *Generation,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        model: Model,
        weights: zml.Bufferized(Model),
        voice: zml.Buffer,
        noise: [32]f32,
    ) !f32 {
        if (self.done) return error.GenerationFinished;
        const ids = zml.Slice.init(.init(.{ .b = 1, .s = self.ids.items.len }, .i64), std.mem.sliceAsBytes(self.ids.items));
        const history = zml.Slice.init(.init(.{ .b = 1, .a = self.frames(), .ld = 32 }, .f32), std.mem.sliceAsBytes(self.latents.items));
        const candidate = zml.Slice.init(.init(.{ .b = 1, .ld = 32 }, .f32), std.mem.asBytes(&noise));
        const start = [_]f32{0};
        const end = [_]f32{1};
        const start_time = zml.Slice.init(.init(.{ .b = 1, .t = 1 }, .f32), std.mem.asBytes(&start));
        const end_time = zml.Slice.init(.init(.{ .b = 1, .t = 1 }, .f32), std.mem.asBytes(&end));
        var executable = try platform.compile(allocator, io, model, .forward, .{
            zml.Tensor.fromShape(ids.shape),        zml.Tensor.fromShape(voice.shape()),
            zml.Tensor.fromShape(history.shape),    zml.Tensor.fromShape(candidate.shape),
            zml.Tensor.fromShape(start_time.shape), zml.Tensor.fromShape(end_time.shape),
        }, .{});
        defer executable.deinit();
        var ids_buffer = try zml.Buffer.fromSlice(io, platform, ids);
        defer ids_buffer.deinit();
        var history_buffer = try zml.Buffer.fromSlice(io, platform, history);
        defer history_buffer.deinit();
        var candidate_buffer = try zml.Buffer.fromSlice(io, platform, candidate);
        defer candidate_buffer.deinit();
        var start_buffer = try zml.Buffer.fromSlice(io, platform, start_time);
        defer start_buffer.deinit();
        var end_buffer = try zml.Buffer.fromSlice(io, platform, end_time);
        defer end_buffer.deinit();
        var args = try executable.args(allocator);
        defer args.deinit(allocator);
        var results = try executable.results(allocator);
        defer results.deinit(allocator);
        args.set(.{ weights, ids_buffer, voice, history_buffer, candidate_buffer, start_buffer, end_buffer });
        executable.call(io, args, &results, .{});
        var result = results.get(zml.Bufferized(Model.Output));
        defer zml.Buffer.deinitAll(Model.Output, &result);
        const velocity = try result.velocity.toSliceAlloc(allocator, io);
        defer velocity.free(allocator);
        const eos = try result.eos_score.toSliceAlloc(allocator, io);
        defer eos.free(allocator);
        // Default one-step Euler sampler: candidate [1,32] + velocity [1,32]
        // -> completed normalized frame [1,32] f32. Retain before checking EOS.
        var frame: [32]f32 = undefined;
        for (&frame, noise, velocity.constItems(f32)) |*value, initial, update| value.* = initial + update;
        // Reserve both growing arrays before committing the new state.
        try self.latents.ensureUnusedCapacity(allocator, 32);
        try self.ids.ensureUnusedCapacity(allocator, 1);
        self.latents.appendSliceAssumeCapacity(&frame);
        self.ids.appendAssumeCapacity(6);
        const score = eos.constItems(f32)[0];
        self.finishFrame(score);
        return score;
    }
};

/// Owns the execution state for one prompt's cached generation loop.
///
/// Prefill runs once on the complete prompt and predicts frame 0. Later calls
/// consume the latest normalized frame `[1, 1, 32]` and scalar i32 absolute
/// position through one reusable decode executable. The K/V cache retains prior
/// text, voice, and audio context with fixed prompt-plus-frame-budget capacity.
///
/// Owns both executables, the uploaded prompt, and runtime cache buffers; model
/// weights and voice are borrowed. `Generation` owns completed output history and
/// EOS state. Calls donate cache storage and replace its handles with returned
/// ones; execution/readback failure poisons the context to prevent unsafe retries.
const CachedGeneration = struct {
    prefill: zml.Exe,
    decode: zml.Exe,
    cache: zml.Bufferized(transformer.KV_Cache),
    prompt: zml.Buffer,
    prompt_length: usize,
    capacity: usize,
    calls: usize = 0,
    failed: bool = false,

    fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model: Model, prompt: zml.Slice, voice: zml.Buffer, max_frames: usize) !CachedGeneration {
        if (prompt.shape.rank() != 2 or prompt.shape.dim(0) != 1 or prompt.shape.dim(1) <= 0 or max_frames == 0)
            return error.InvalidGenerationInputs;
        const prompt_length: usize = @intCast(prompt.shape.dim(1));
        const capacity = std.math.add(usize, prompt_length, max_frames) catch return error.CacheCapacityExceeded;
        if (capacity > std.math.maxInt(i32)) return error.CacheCapacityExceeded;
        const symbolic_cache = transformer.KV_Cache.init(1, @intCast(capacity));
        const candidate = zml.Tensor.fromShape(.init(.{ .b = 1, .ld = 32 }, .f32));
        const start_time = zml.Tensor.fromShape(.init(.{ .b = 1, .t = 1 }, .f32));
        const end_time = zml.Tensor.fromShape(.init(.{ .b = 1, .t = 1 }, .f32));
        var prefill = try platform.compile(allocator, io, model, .forward_prefill, .{
            zml.Tensor.fromShape(prompt.shape), zml.Tensor.fromShape(voice.shape()), candidate, start_time, end_time, symbolic_cache,
        }, .{});
        errdefer prefill.deinit();
        // Fixed decode shapes: frame values and absolute position change at runtime.
        var decode = try platform.compile(allocator, io, model, .forward_decode, .{
            zml.Tensor.fromShape(.init(.{ .b = 1, .a = 1, .ld = 32 }, .f32)),
            zml.Tensor.fromShape(.scalar(.i32)),
            candidate,
            start_time,
            end_time,
            symbolic_cache,
        }, .{});
        errdefer decode.deinit();
        var cache = try symbolic_cache.initBuffer(allocator, io, platform);
        errdefer transformer.KV_Cache.deinitBuffer(&cache);
        var prompt_buffer = try zml.Buffer.fromSlice(io, platform, prompt);
        errdefer prompt_buffer.deinit();
        return .{ .prefill = prefill, .decode = decode, .cache = cache, .prompt = prompt_buffer, .prompt_length = prompt_length, .capacity = capacity };
    }

    fn deinit(self: *CachedGeneration) void {
        transformer.KV_Cache.deinitBuffer(&self.cache);
        self.prompt.deinit();
        self.decode.deinit();
        self.prefill.deinit();
    }

    fn step(self: *CachedGeneration, generation: *Generation, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, model_weights: zml.Bufferized(Model), voice: zml.Buffer, noise: [32]f32) !f32 {
        if (generation.done) return error.GenerationFinished;
        if (self.failed or self.calls != generation.frames() or generation.ids.items.len != self.prompt_length + self.calls)
            return error.InvalidGenerationState;
        const position = if (self.calls == 0) 0 else self.prompt_length + self.calls - 1;
        if (self.calls > 0 and position >= self.capacity) return error.CacheCapacityExceeded;
        // Reserve history before donating/mutating cache storage. A subsequent
        // execution/readback error poisons the session rather than allowing retries.
        try generation.latents.ensureUnusedCapacity(allocator, 32);
        try generation.ids.ensureUnusedCapacity(allocator, 1);
        errdefer self.failed = true;
        const candidate = zml.Slice.init(.init(.{ .b = 1, .ld = 32 }, .f32), std.mem.asBytes(&noise));
        const start_values = [_]f32{0};
        const end_values = [_]f32{1};
        const start = zml.Slice.init(.init(.{ .b = 1, .t = 1 }, .f32), std.mem.asBytes(&start_values));
        const end = zml.Slice.init(.init(.{ .b = 1, .t = 1 }, .f32), std.mem.asBytes(&end_values));
        var candidate_buffer = try zml.Buffer.fromSlice(io, platform, candidate);
        defer candidate_buffer.deinit();
        var start_buffer = try zml.Buffer.fromSlice(io, platform, start);
        defer start_buffer.deinit();
        var end_buffer = try zml.Buffer.fromSlice(io, platform, end);
        defer end_buffer.deinit();
        const executable = if (self.calls == 0) &self.prefill else &self.decode;
        var args = try executable.args(allocator);
        defer args.deinit(allocator);
        var results = try executable.results(allocator);
        defer results.deinit(allocator);
        if (self.calls == 0) {
            args.set(.{ model_weights, self.prompt, voice, candidate_buffer, start_buffer, end_buffer, self.cache });
            executable.call(io, args, &results, .{});
        } else {
            const last: [32]f32 = generation.latents.items[generation.latents.items.len - 32 ..][0..32].*;
            const frame = zml.Slice.init(.init(.{ .b = 1, .a = 1, .ld = 32 }, .f32), std.mem.asBytes(&last));
            const absolute_position: i32 = @intCast(position);
            const position_slice = zml.Slice.init(.scalar(.i32), std.mem.asBytes(&absolute_position));
            var frame_buffer = try zml.Buffer.fromSlice(io, platform, frame);
            defer frame_buffer.deinit();
            var position_buffer = try zml.Buffer.fromSlice(io, platform, position_slice);
            defer position_buffer.deinit();
            args.set(.{ model_weights, frame_buffer, position_buffer, candidate_buffer, start_buffer, end_buffer, self.cache });
            executable.call(io, args, &results, .{});
        }
        var result = results.get(zml.Bufferized(Model.CachedOutput));
        // Execution consumed the old cache contents; its raw PJRT handles are
        // still caller-owned. The returned handles now own the reused allocation.
        transformer.KV_Cache.deinitBuffer(&self.cache);
        self.cache = result.cache;
        defer result.eos_score.deinit();
        defer result.velocity.deinit();
        const velocity = try result.velocity.toSliceAlloc(allocator, io);
        defer velocity.free(allocator);
        const eos = try result.eos_score.toSliceAlloc(allocator, io);
        defer eos.free(allocator);
        var frame: [32]f32 = undefined;
        for (&frame, noise, velocity.constItems(f32)) |*value, initial, update| value.* = initial + update;
        generation.latents.appendSliceAssumeCapacity(&frame);
        generation.ids.appendAssumeCapacity(6);
        self.calls += 1;
        const score = eos.constItems(f32)[0];
        generation.finishFrame(score);
        return score;
    }
};

/// Save normalized TTS history, explicitly labelled to avoid decoder-space confusion.
fn saveLatents(allocator: std.mem.Allocator, io: std.Io, path: []const u8, values: []const f32) !void {
    const header = try std.fmt.allocPrint(allocator, "{{\"__metadata__\":{{\"latent_space\":\"normalized TTS; apply latent_std and latent_mean before decoding\"}},\"audio_latents\":{{\"dtype\":\"F32\",\"shape\":[1,{d},32],\"data_offsets\":[0,{d}]}}}}", .{ values.len / 32, values.len * 4 });
    defer allocator.free(header);
    const padding = (8 - header.len % 8) % 8;
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, @intCast(header.len + padding), .little);
    try writer.interface.writeAll(&length);
    try writer.interface.writeAll(header);
    try writer.interface.splatByteAll(' ', padding);
    // Safetensors stores little-endian values, independently of the host.
    for (values) |value| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
        try writer.interface.writeAll(&bytes);
    }
    try writer.interface.flush();
}

fn load_voice(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !zml.Slice {
    var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, path);
    defer registry.deinit();
    const tensor = registry.tensors.get("speaker_wavs") orelse return error.VoiceNotFound;
    if (tensor.shape.rank() != 3 or tensor.shape.dim(0) != 1 or tensor.shape.dim(1) != 512 or tensor.shape.dtype() != .f32)
        return error.InvalidVoiceShape;

    const raw = try zml.Slice.alloc(allocator, tensor.shape);
    defer raw.free(allocator);
    var read_buffer: [8192]u8 = undefined;
    var reader = try tensor.reader(io, &read_buffer, .{});
    defer reader.deinit();
    try reader.interface.readSliceAll(raw.data());

    const frames: usize = @intCast(tensor.shape.dim(2));
    const features: usize = @intCast(tensor.shape.dim(1));
    const voice = try zml.Slice.alloc(allocator, .init(.{ .b = 1, .p = frames, .ad = features }, .f32));
    // Match Python's speaker_wavs[0].T: [1, features, frames] -> [1, frames, features].
    const src = raw.constItems(f32);
    const dst = voice.items(f32);
    for (0..frames) |frame| {
        for (0..features) |feature| {
            dst[frame * features + feature] = src[feature * frames + frame];
        }
    }
    return voice;
}

test "uncached generation matches four Python sampler frames" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const fixtures_path = try util.assetPath(allocator, "zml/examples/phonon/assets/generation.activations.safetensors");
    defer allocator.free(fixtures_path);
    var fixtures = try zml.safetensors.TensorRegistry.fromPath(allocator, io, fixtures_path);
    defer fixtures.deinit();

    const prompt = try util.readTensorSlice(allocator, io, fixtures.tensors.get("input_ids").?);
    defer prompt.free(allocator);
    const voice = try util.readTensorSlice(allocator, io, fixtures.tensors.get("voice").?);
    defer voice.free(allocator);
    const noise = try util.readTensorSlice(allocator, io, fixtures.tensors.get("noise").?);
    defer noise.free(allocator);
    const expected = try util.readTensorSlice(allocator, io, fixtures.tensors.get("frames").?);
    defer expected.free(allocator);
    const scores = try util.readTensorSlice(allocator, io, fixtures.tensors.get("eos").?);
    defer scores.free(allocator);

    const model_path = try util.assetPath(allocator, util.gradium_phonon_31m_en);
    defer allocator.free(model_path);
    var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, model_path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &.{});
    defer store.deinit();

    const model: Model = .init(store.view());

    const platform = zml.testing.env();
    var weights = try model.load(allocator, io, platform, &store);
    defer Model.unloadBuffers(&weights);
    var voice_buffer = try zml.Buffer.fromSlice(io, platform, voice);
    defer voice_buffer.deinit();
    var generation = try Generation.init(allocator, prompt.constItems(i64), .{ .max_frames = 4 });
    defer generation.deinit(allocator);
    for (0..4) |index| {
        const sample: [32]f32 = noise.constItems(f32)[index * 32 ..][0..32].*;
        const score = try generation.step(allocator, io, platform, model, weights, voice_buffer, sample);
        try std.testing.expectApproxEqAbs(scores.constItems(f32)[index], score, 1e-4);
        // Compare the whole accumulated history: previous Zig outputs feed the
        // next call, rather than replacing history with Python fixture values.
        for (generation.latents.items, expected.constItems(f32)[0 .. (index + 1) * 32]) |actual, reference| {
            try std.testing.expect(@abs(actual - reference) <= 1e-4 + 1e-4 * @abs(reference));
        }
        try std.testing.expectEqual(@as(usize, @intCast(prompt.shape.dim(1))) + index + 1, generation.ids.items.len);
        try std.testing.expectEqual(@as(i64, 6), generation.ids.items[generation.ids.items.len - 1]);
    }
    try std.testing.expect(generation.done);
    try std.testing.expect(!generation.stopped_by_eos);
    const sample: [32]f32 = @splat(0);
    try std.testing.expectError(error.GenerationFinished, generation.step(allocator, io, platform, model, weights, voice_buffer, sample));
}

test "cached generation matches Python and preserves masked cache slots" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const fixtures_path = try util.assetPath(allocator, "zml/examples/phonon/assets/generation.activations.safetensors");
    defer allocator.free(fixtures_path);
    var fixtures = try zml.safetensors.TensorRegistry.fromPath(allocator, io, fixtures_path);
    defer fixtures.deinit();

    const prompt = try util.readTensorSlice(allocator, io, fixtures.tensors.get("input_ids").?);
    defer prompt.free(allocator);
    const voice = try util.readTensorSlice(allocator, io, fixtures.tensors.get("voice").?);
    defer voice.free(allocator);
    const noise = try util.readTensorSlice(allocator, io, fixtures.tensors.get("noise").?);
    defer noise.free(allocator);
    const expected = try util.readTensorSlice(allocator, io, fixtures.tensors.get("frames").?);
    defer expected.free(allocator);
    const scores = try util.readTensorSlice(allocator, io, fixtures.tensors.get("eos").?);
    defer scores.free(allocator);

    const model_path = try util.assetPath(allocator, util.gradium_phonon_31m_en);
    defer allocator.free(model_path);
    var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, model_path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &.{});
    defer store.deinit();

    const model: Model = .init(store.view());

    const platform = zml.testing.env();
    var weights = try model.load(allocator, io, platform, &store);
    defer Model.unloadBuffers(&weights);
    var voice_buffer = try zml.Buffer.fromSlice(io, platform, voice);
    defer voice_buffer.deinit();
    var generation = try Generation.init(allocator, prompt.constItems(i64), .{ .max_frames = 4 });
    defer generation.deinit(allocator);
    var cached = try CachedGeneration.init(allocator, io, platform, model, prompt, voice_buffer, 4);
    defer cached.deinit();
    try std.testing.expectEqual(@as(usize, 2), cached.decode.donated_input_indices.len);
    // Nonzero unused slots make a missing valid-length mask visible numerically.
    const poisoned = try zml.Slice.alloc(allocator, cached.cache.k.shape());
    defer poisoned.free(allocator);
    @memset(poisoned.items(f32), 777);
    {
        var poisoned_k = try zml.Buffer.fromSlice(io, platform, poisoned);
        errdefer poisoned_k.deinit();
        const poisoned_v = try zml.Buffer.fromSlice(io, platform, poisoned);
        transformer.KV_Cache.deinitBuffer(&cached.cache);
        cached.cache = .{ .k = poisoned_k, .v = poisoned_v };
    }
    var prefix_k: []f32 = &.{};
    defer allocator.free(prefix_k);
    var prefix_v: []f32 = &.{};
    defer allocator.free(prefix_v);

    for (0..4) |index| {
        const sample: [32]f32 = noise.constItems(f32)[index * 32 ..][0..32].*;
        const score = try cached.step(&generation, allocator, io, platform, weights, voice_buffer, sample);
        try std.testing.expectApproxEqAbs(scores.constItems(f32)[index], score, 1e-4);
        // Compare the whole accumulated history: previous Zig outputs feed the
        // next call, rather than replacing history with Python fixture values.
        for (generation.latents.items, expected.constItems(f32)[0 .. (index + 1) * 32]) |actual, reference| {
            try std.testing.expect(@abs(actual - reference) <= 1e-4 + 1e-4 * @abs(reference));
        }
        try std.testing.expectEqual(@as(usize, @intCast(prompt.shape.dim(1))) + index + 1, generation.ids.items.len);
        try std.testing.expectEqual(@as(i64, 6), generation.ids.items[generation.ids.items.len - 1]);
        const keys = try cached.cache.k.toSliceAlloc(allocator, io);
        defer keys.free(allocator);
        const values = try cached.cache.v.toSliceAlloc(allocator, io);
        defer values.free(allocator);
        if (index == 0) {
            prefix_k = try allocator.dupe(f32, keys.constItems(f32));
            prefix_v = try allocator.dupe(f32, values.constItems(f32));
        }
        const valid = cached.prompt_length + index;
        for (0..6) |layer| {
            const base = layer * cached.capacity * 512;
            const prefix_end = base + cached.prompt_length * 512;
            try std.testing.expectEqualSlices(f32, prefix_k[base..prefix_end], keys.constItems(f32)[base..prefix_end]);
            try std.testing.expectEqualSlices(f32, prefix_v[base..prefix_end], values.constItems(f32)[base..prefix_end]);
            for (keys.constItems(f32)[base + valid * 512 .. base + cached.capacity * 512]) |value|
                try std.testing.expectEqual(@as(f32, 777), value);
            for (values.constItems(f32)[base + valid * 512 .. base + cached.capacity * 512]) |value|
                try std.testing.expectEqual(@as(f32, 777), value);
        }
    }
    try std.testing.expect(generation.done);
    try std.testing.expect(!generation.stopped_by_eos);
    const sample: [32]f32 = @splat(0);
    try std.testing.expectError(error.GenerationFinished, cached.step(&generation, allocator, io, platform, weights, voice_buffer, sample));
    // Raising the host budget must not silently grow/clamp the compiled cache.
    generation.done = false;
    generation.settings.max_frames = 6;
    _ = try cached.step(&generation, allocator, io, platform, weights, voice_buffer, sample);
    generation.done = false;
    try std.testing.expectError(error.CacheCapacityExceeded, cached.step(&generation, allocator, io, platform, weights, voice_buffer, sample));
    try std.testing.expectEqual(@as(usize, 5), cached.calls);
    try std.testing.expectEqual(@as(usize, 5), generation.frames());
}

test "EOS retains stopping frame and tail, honors minimum and strict threshold" {
    const allocator = std.testing.allocator;
    var generation = try Generation.init(allocator, &.{4}, .{
        .max_frames = 10,
        .min_frames_before_eos = 1,
        .frames_after_eos = 2,
    });
    defer generation.deinit(allocator);
    const scores = [_]f32{ 10, 4, 5, -10, -10 };
    for (scores, 0..) |score, index| {
        try generation.latents.appendSlice(allocator, &@as([32]f32, @splat(0)));
        generation.finishFrame(score);
        try std.testing.expectEqual(index == 4, generation.done);
    }
    try std.testing.expect(generation.stopped_by_eos);
    try std.testing.expectEqual(@as(usize, 5), generation.frames());

    var immediate = try Generation.init(allocator, &.{4}, .{ .max_frames = 1, .frames_after_eos = 0 });
    defer immediate.deinit(allocator);
    try immediate.latents.appendSlice(allocator, &@as([32]f32, @splat(0)));
    immediate.finishFrame(5);
    try std.testing.expect(immediate.done and immediate.stopped_by_eos);
}
