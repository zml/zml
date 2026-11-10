const std = @import("std");
const zml = @import("zml");

const util = @import("util.zig");
const Audio_Player = @import("Audio_Player.zig");
const Mimi = @import("mimi/root.zig").Mimi;
const Phonon = @import("phonon/root.zig").Phonon;
const Tokenizer = @import("phonon/Tokenizer.zig");

const Args = struct {
    noop: bool = false,
    no_play: bool = false,
    trace: bool = false,
    viz: bool = false,
    dtype: enum { f32, f16, bf16 } = .f32,

    pub const help =
        \\Usage: phonon [--noop] [--no-play] [--trace] [--viz] [--dtype=f32|f16|bf16] < prompt.txt
        \\
        \\  --noop     Initialize everything, then leave.
        \\  --no-play  Generate audio without playing it.
        \\  --trace    Record and export a profiling trace.
        \\  --viz      Display a scrolling waveform in the terminal.
        \\  --dtype    Phonon transformer input and KV-cache dtype (default: f32).
        \\  --help     Show this help.
        \\
        \\Read the UTF-8 prompt from stdin until EOF, preserving line breaks.
        \\
    ;
};

pub const std_options: std.Options = .{ .log_level = .info };

// Retain the Zig half of Bazel's startup hook.
comptime {
    _ = zml.bazel;
}

// TODO: what if we windowed the frame history in Phonon (but the bootstrap history)

// TODO: things yet to cover:
// * batch inputs (i.e. multiple prompts at once)
// * different base models (i.e. something other than phonon-31m-en)
// * different voice features
// * not f32?
// * now that it's ported, we could grab the asset snapshots from the zig graph instead? then there's *really* no python
// * how does one benchmark these things?
// * what's with the implicit errors all over the place?
//
// And, more generally, the big 4:
//
// * Streaming
//   Mimi now decodes each completed latent into a waveform chunk during
//   generation. A miniaudio queue feeds the playback callback; inference
//   stalls produce silence when the queue runs dry.
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

// TODO: batching
// TODO: sharding?

// FIX: we need tests for empty and unit-length prompts
// FIX: we need benchmarks proving that we scale somewhat linearly with prompt size (..or dont)

pub fn main(init: std.process.Init) !void {
    const here = @typeName(@This()) ++ ".main";

    const gpa = init.gpa;
    const io = init.io; // TODO: we should try and see what happens with zio though
    const arena_impl = init.arena;
    const arena = arena_impl.allocator();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);

    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    var stdin_buffer: [4096]u8 = undefined;
    var stdin = std.Io.File.stdin().readerStreaming(io, &stdin_buffer);
    const text = try stdin.interface.allocRemaining(gpa, .unlimited);
    defer gpa.free(text);

    const platform: *zml.Platform = try .auto(gpa, io, .{});
    defer platform.deinit(gpa, io);

    var profiler: ?zml.Platform.Profiler = if (args.trace) try platform.profiler(gpa, io, .{}) else null;
    defer if (profiler) |*p| p.deinit();

    if (profiler) |*p| try p.start();
    defer if (profiler) |*p| {
        if ((p.stop() catch unreachable)) |profile| {
            std.log.info("Profile dumped: {s} and {s}", .{
                profile.protobuf_path,
                profile.perfetto_path,
            });
        }
    };

    var span_init = zml.tracer.span_at(@src(), here ++ "/init", .{});
    defer span_init.end();
    {}

    const prepared = try prepare_prompt(io, gpa, text);
    defer prepared.deinit(gpa);
    if (prepared.chunks.len == 0) return;
    const input = prepared.chunks[0];
    std.log.info("Prepared {d} text chunks; longest index {any}, {d} text tokens (soft target: 50).", .{
        prepared.text_chunks.items.len,
        prepared.longest_chunk,
        prepared.text_chunks.max_text_tokens,
    });
    std.log.info("Shared capacity: {d} prompt positions, {d} generated frames.", .{
        prepared.capacity.max_prompt_tokens,
        prepared.capacity.max_frames,
    });

    const phonon = try gpa.create(Phonon);
    defer gpa.destroy(phonon);
    const mimi = try gpa.create(Mimi);
    defer gpa.destroy(mimi);

    var fut_phonon = blk: {
        const context: zml.tracer.Context = .init(.generic);
        var span = zml.tracer.span_at_for(@src(), here ++ "/init/dispatch_phonon", .{}, .{
            .producer = context,
        });
        defer span.end();

        const dtype: zml.DataType = switch (args.dtype) {
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
        };
        break :blk io.async(Phonon.init_in_place, .{ phonon, io, gpa, platform, input, prepared.capacity, dtype, context });
    };
    var fut_mimi = blk: {
        const context: zml.tracer.Context = .init(.generic);
        var span = zml.tracer.span_at_for(@src(), here ++ "/init/dispatch_mimi", .{}, .{
            .producer = context,
        });
        defer span.end();

        break :blk io.async(Mimi.init_in_place, .{ mimi, io, gpa, platform, context });
    };

    fut_phonon.await(io) catch |err| {
        if (fut_mimi.await(io)) |_| mimi.deinit() else |_| {}
        return err;
    };
    defer phonon.deinit();
    try fut_mimi.await(io);
    defer mimi.deinit();

    // TODO: no reason for this to init on the same thread right?
    const player_opt: ?Audio_Player = if (args.no_play) null else try .init(Mimi.sampling_rate);
    defer if (player_opt) |audio| audio.deinit();

    var sample_queue_buf: [32]Sample = undefined;
    var sample_queue: std.Io.Queue(Sample) = .init(&sample_queue_buf);

    const display_enabled = args.viz and try std.Io.File.stdout().supportsAnsiEscapeCodes(io);
    var display_queue_buf: [32]Waveform_Sample = undefined;
    var display_queue: std.Io.Queue(Waveform_Sample) = undefined;
    var fut_display: ?std.Io.Future(anyerror!void) = null;
    if (display_enabled) {
        display_queue = .init(&display_queue_buf);
        fut_display = io.async(visual_playback, .{ io, gpa, &display_queue, player_opt });
    }
    defer if (fut_display) |*future| {
        display_queue.close(io);
        future.cancel(io) catch {};
        while (display_queue.getOneUncancelable(io)) |sample| {
            gpa.free(sample.waveform);
        } else |_| {}
    };

    var fut_waveform = io.async(audio_playback, .{ io, gpa, &sample_queue, player_opt, if (display_enabled) &display_queue else null });
    defer {
        sample_queue.close(io);
        if (fut_waveform.await(io)) |result| {
            var owned = result;
            owned.deinit(gpa);
        } else |_| {}
        while (sample_queue.getOneUncancelable(io)) |sample| {
            sample.waveform.free(gpa);
        } else |_| {}
    }

    {}
    span_init.end();

    var history: ?zml.Buffer = null;
    var active_lengths: ?zml.Buffer = null;

    var iteration: usize = 0;
    if (!args.noop) for (prepared.chunks, 0..) |chunk_input, chunk_index| {
        var chunk_span = zml.tracer.span_at(@src(), here ++ "/chunk", .{ .chunk = chunk_index });
        defer chunk_span.end();
        if (chunk_index > 0) try phonon.start_chunk(chunk_input);
        while (!phonon.done) : (iteration += 1) {
            const context: zml.tracer.Context = .init(.generic);
            var span_iteration = zml.tracer.span_at_for(@src(), here ++ "/iteration", .{ .chunk = chunk_index, .iteration = iteration }, .{
                .producer = context,
            });
            defer span_iteration.end();

            const out = try phonon.forward();

            history = out.history;
            active_lengths = out.active_lengths;

            const chunk = try mimi.forward(out.active_lengths, out.history34);
            {
                errdefer chunk.free(gpa);
                if (display_enabled) {
                    // Each consumer owns its audio; text stays alive in prepared until both finish.
                    const samples = try gpa.dupe(f32, chunk.constItems(f32));
                    display_queue.putOneUncancelable(io, .{
                        .waveform = samples,
                        .text = prepared.text_chunks.items[chunk_index].text,
                        .context = context,
                    }) catch |err| {
                        gpa.free(samples);
                        return err;
                    };
                }

                try sample_queue.putOneUncancelable(io, .{ .waveform = chunk, .context = context });
            }
        }
    };

    sample_queue.close(io);
    if (display_enabled) display_queue.close(io);
    const waveform = try fut_waveform.await(io);
    if (fut_display) |*future| try future.await(io);

    var span_shutdown = zml.tracer.span_at(@src(), here ++ "/shutdown", .{});
    defer span_shutdown.end();

    if (false) {
        if (player_opt) |player| {
            var span_player_write_all = zml.tracer.span_at(@src(), here ++ "/player_write_all", .{});
            defer span_player_write_all.end();

            {
                try player.write(waveform.items);
            }

            if (history) |hist| {
                if (active_lengths) |al| {
                    const wav = try mimi.forward(al, hist);
                    defer wav.free(gpa);
                    try player.write(wav.constItems(f32));
                }
            }
        }
    }

    if (player_opt) |player| {
        var span_player_finish = zml.tracer.span_at(@src(), here ++ "/player_finish", .{});
        defer span_player_finish.end();

        try player.finish();
    }

    if (false) {
        // Bazel runs from its runfiles tree; save beside the caller's working directory.
        const wav_path = try std.fs.path.join(arena, &.{ init.environ_map.get("BUILD_WORKING_DIRECTORY") orelse ".", "phonon.wav" });

        try save_wav(io, wav_path, waveform.items);

        try stdout.interface.print("Saved {s}: {d} mono samples at {d} Hz.\n", .{ wav_path, waveform.items.len, Mimi.sampling_rate });
        try stdout.interface.flush();
    }
}

pub const Prepared_Prompt = struct {
    /// Each input borrows the same selected voice; its token IDs are owned here.
    chunks: []Phonon.Input,
    voice: zml.Slice,
    text_chunks: Tokenizer.Text_Chunks,
    /// Longest expanded prompt, including voice markers and BOS.
    longest_chunk: ?usize,
    capacity: Phonon.Capacity,

    pub fn deinit(self: Prepared_Prompt, allocator: std.mem.Allocator) void {
        for (self.chunks) |chunk| chunk.tokens.free(allocator);
        allocator.free(self.chunks);
        self.voice.free(allocator);
        self.text_chunks.deinit(allocator);
    }
};

/// Prepare sentence chunks and their expanded prompt IDs. Capacity is the
/// maximum prompt width and frame budget across the chunks, not the full text.
/// The caller releases all owned host storage with Prepared_Prompt.deinit.
pub fn prepare_prompt(io: std.Io, gpa: std.mem.Allocator, text: []const u8) !Prepared_Prompt {
    const here = @typeName(@This()) ++ ".prepare_prompt";
    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    var arena_local_impl: std.heap.ArenaAllocator = .init(gpa);
    defer arena_local_impl.deinit();
    const arena_local = arena_local_impl.allocator();
    const tokenizer_path = try util.assetPath(arena_local, util.gradium_phonon_31m_en_tokenizer);
    var tokenizer = try Tokenizer.init(arena_local, io, tokenizer_path);
    defer tokenizer.deinit();
    const text_chunks = try tokenizer.split_text(gpa, text, 50);
    errdefer text_chunks.deinit(gpa);
    const voice_path = try util.assetPath(arena_local, util.gradium_phonon_voice);
    const voice = try load_voice(gpa, io, voice_path);
    errdefer voice.free(gpa);

    const chunks = try gpa.alloc(Phonon.Input, text_chunks.items.len);
    var initialized: usize = 0;
    errdefer {
        for (chunks[0..initialized]) |chunk| chunk.tokens.free(gpa);
        gpa.free(chunks);
    }
    var longest_chunk: ?usize = null;
    var capacity: Phonon.Capacity = .{ .max_prompt_tokens = 0, .max_frames = 0 };
    for (text_chunks.items, chunks, 0..) |chunk, *input, i| {
        const tokens = try tokenizer.encode_prompt(gpa, chunk.text, @intCast(voice.shape.dim(.p)));
        errdefer tokens.free(gpa);
        var max_frames: usize = 0;
        var frames_after_eos: usize = 0;
        {
            var span_count_frames = zml.tracer.span_at(@src(), here ++ "/count_frames", .{});
            defer span_count_frames.end();

            // Preserve the checkpoint's text-length budget and short-text EOS tail.
            var text_ids: std.ArrayList(u32) = .empty;
            defer text_ids.deinit(gpa);

            for (tokens.constItems(i64)) |id| {
                if (id == 4 or id == 5 or id == 6) continue;
                try text_ids.append(gpa, @intCast(id));
            }

            var decoder = try tokenizer.inner.decoder();
            defer decoder.deinit();
            var decoded = try decoder.decodeAlloc(gpa, text_ids.items);
            defer decoded.deinit(gpa);
            max_frames = @intFromFloat(@ceil((@as(f64, @floatFromInt(text_ids.items.len)) / 3 + 2) * 12.5));

            var words = std.mem.tokenizeAny(u8, decoded.items, " \t\r\n\x0b\x0c");
            var word_count: usize = 0;
            while (words.next() != null) word_count += 1;
            if (word_count <= 4) frames_after_eos = 3;
        }

        input.* = .{
            .tokens = tokens,
            .voice = voice,
            .max_frames = max_frames,
            .frames_after_eos = frames_after_eos,
        };
        initialized += 1;
        if (longest_chunk == null or tokens.constItems(i64).len > capacity.max_prompt_tokens) {
            longest_chunk = i;
            capacity.max_prompt_tokens = tokens.constItems(i64).len;
        }
        capacity.max_frames = @max(capacity.max_frames, max_frames);
    }

    return .{
        .chunks = chunks,
        .voice = voice,
        .text_chunks = text_chunks,
        .longest_chunk = longest_chunk,
        .capacity = capacity,
    };
}

fn load_voice(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !zml.Slice {
    const here = @typeName(@This()) ++ ".load_voice";

    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    var registry = blk: {
        var load_registry = zml.tracer.span_at(@src(), here ++ "/load_registry", .{});
        defer load_registry.end();

        break :blk try zml.safetensors.TensorRegistry.fromPath(allocator, io, path);
    };
    defer registry.deinit();

    const tensor = registry.tensors.get("speaker_wavs") orelse return error.VoiceNotFound;
    if (tensor.shape.rank() != 3 or tensor.shape.dim(0) != 1 or tensor.shape.dim(1) != 512 or !tensor.shape.dtype().isFloat())
        return error.InvalidVoiceShape;

    const raw = try zml.Slice.alloc(allocator, tensor.shape);
    defer raw.free(allocator);

    {
        var span_read_voice = zml.tracer.span_at(@src(), here ++ "/read_voice", .{});
        defer span_read_voice.end();

        var read_buffer: [8192]u8 = undefined;
        var reader = try tensor.reader(io, &read_buffer, .{});
        defer reader.deinit();
        try reader.interface.readSliceAll(raw.data());
    }

    var span_transpose_voice = zml.tracer.span_at(@src(), here ++ "/transpose_voice", .{});
    defer span_transpose_voice.end();

    const frames: usize = @intCast(tensor.shape.dim(2));
    const features: usize = @intCast(tensor.shape.dim(1));

    // Match Python's speaker_wavs[0].T: [1, features, frames] -> [1, frames, features].
    const voice = try zml.Slice.alloc(allocator, .init(.{ .b = 1, .p = frames, .ad = features }, .f32));
    const src = raw.constItems(f32);
    const dst = voice.items(f32);
    for (0..frames) |frame| {
        for (0..features) |feature| {
            dst[frame * features + feature] = src[feature * frames + frame];
        }
    }

    return voice;
}

const Sample = struct {
    waveform: zml.Slice,
    context: zml.tracer.Context,
};

const Waveform_Sample = struct {
    /// Owned by the display consumer. Text borrows Prepared_Prompt storage.
    waveform: []const f32,
    text: []const u8,
    context: zml.tracer.Context,
};

fn visual_playback(io: std.Io, gpa: std.mem.Allocator, queue: *std.Io.Queue(Waveform_Sample), player_opt: ?Audio_Player) anyerror!void {
    const here = @typeName(@This()) ++ ".visual_playback";

    var span = zml.tracer.span_at(@src(), here, .{});
    defer span.end();

    errdefer queue.close(io);
    var output_buffer: [4096]u8 = undefined;
    var output = std.Io.File.stdout().writerStreaming(io, &output_buffer);
    const writer = &output.interface;
    var previous_text: ?[]const u8 = null;
    defer if (previous_text != null) {
        writer.writeAll("\x1b[?25h\n") catch {};
        writer.flush() catch {};
    };

    const width = 72;
    const height = 8;
    const columns = width * 2;
    const dots = height * 4;
    const samples_per_column = Mimi.sampling_rate * 3 / columns;
    const Range = struct { min: f32 = 0, max: f32 = 0 };
    var history: [columns]Range = @splat(.{});
    var next: usize = 0;
    var pending: Range = .{};
    var pending_count: usize = 0;
    const dot_bits = [2][4]u3{ .{ 0, 1, 2, 6 }, .{ 3, 4, 5, 7 } };
    var next_frame = std.Io.Clock.Timestamp.now(io, .awake);
    var consumed_samples: u64 = 0;

    while (true) {
        const sample = queue.getOne(io) catch |err| switch (err) {
            error.Closed => return,
            else => return err,
        };
        defer gpa.free(sample.waveform);

        var span_iteration = zml.tracer.span_at_for(@src(), here ++ "/iteration", .{}, .{
            .consumer = sample.context,
        });
        defer span_iteration.end();

        var offset: usize = 0;
        while (offset < sample.waveform.len) {
            const values = sample.waveform[offset..][0..@min(samples_per_column, sample.waveform.len - offset)];
            if (player_opt) |player| {
                consumed_samples += values.len;
                // Follow actual callback consumption, including pauses during underruns.
                while (player.consumedFrames() < consumed_samples) {
                    try io.sleep(.fromMilliseconds(2), .awake);
                }
            } else {
                const duration: std.Io.Clock.Duration = .{
                    .raw = .fromNanoseconds(@divTrunc(@as(i96, @intCast(values.len)) * std.time.ns_per_s, Mimi.sampling_rate)),
                    .clock = .awake,
                };
                const now = std.Io.Clock.Timestamp.now(io, .awake);
                if (previous_text == null or now.compare(.gt, next_frame.addDuration(duration))) next_frame = now;
                next_frame = next_frame.addDuration(duration);
                try next_frame.wait(io);
            }

            if (previous_text == null or !std.mem.eql(u8, previous_text.?, sample.text)) {
                history = @splat(.{});
                next = 0;
                pending = .{};
                pending_count = 0;
                previous_text = sample.text;
                try writer.writeAll("\x1b[?25l\x1b[2J\x1b[H");
                try writer.print("{s}\n\n", .{sample.text});
                try writer.writeAll("\x1b7");
            }

            for (values) |value| {
                const amplitude = if (std.math.isFinite(value)) std.math.clamp(value, -1, 1) else 0;
                if (pending_count == 0) pending = .{ .min = amplitude, .max = amplitude };
                pending.min = @min(pending.min, amplitude);
                pending.max = @max(pending.max, amplitude);
                pending_count += 1;
                if (pending_count == samples_per_column) {
                    history[next] = pending;
                    next = (next + 1) % columns;
                    pending_count = 0;
                }
            }

            var cells: [height][width]u8 = @splat(@splat(0));
            for (0..columns) |x| {
                const range = history[(next + x) % columns];
                const top: usize = @intFromFloat(@round((1 - range.max) * 0.5 * (dots - 1)));
                const bottom: usize = @intFromFloat(@round((1 - range.min) * 0.5 * (dots - 1)));
                for (top..bottom + 1) |y| {
                    cells[y / 4][x / 2] |= @as(u8, 1) << dot_bits[x % 2][y % 4];
                }
            }
            try writer.writeAll("\x1b8");
            for (cells) |row| {
                for (row) |bits| {
                    var encoded: [4]u8 = undefined;
                    const len = try std.unicode.utf8Encode(@as(u21, 0x2800) + bits, &encoded);
                    try writer.writeAll(encoded[0..len]);
                }
                try writer.writeAll("\x1b[K\n");
            }
            try writer.flush();
            offset += values.len;
        }
    }
}

fn audio_playback(
    io: std.Io,
    gpa: std.mem.Allocator,
    sample_queue: *std.Io.Queue(Sample),
    player_opt: ?Audio_Player,
    display_queue: ?*std.Io.Queue(Waveform_Sample),
) anyerror!std.ArrayList(f32) {
    errdefer sample_queue.close(io);
    errdefer if (display_queue) |queue| queue.close(io);
    const here = @typeName(@This()) ++ ".audio_playback";

    var span_audio_playback = zml.tracer.span_at_for(@src(), here, .{}, .{});
    defer span_audio_playback.end();

    var waveform: std.ArrayList(f32) = .empty;
    errdefer waveform.deinit(gpa);

    while (true) {
        const sample = sample_queue.getOne(io) catch return waveform;
        const chunk = sample.waveform;
        defer chunk.free(gpa);

        var span_iteration = zml.tracer.span_at_for(@src(), here ++ "/iteration", .{}, .{
            .consumer = sample.context,
        });
        defer span_iteration.end();

        try waveform.appendSlice(gpa, chunk.constItems(f32));

        if (player_opt) |player| {
            var span_player_write = zml.tracer.span_at_for(@src(), here ++ "/player_write", .{}, .{});
            defer span_player_write.end();

            try player.write(chunk.constItems(f32));
        }
    }

    return waveform;
}

/// Save mono PCM16. Keep floating-point decoder output intact until this final
/// quantization step; clamp only to the representable WAV sample range.
fn save_wav(io: std.Io, path: []const u8, samples: []const f32) !void {
    const byte_count = try std.math.mul(u32, std.math.cast(u32, samples.len) orelse return error.AudioTooLong, 2);
    const riff_size = try std.math.add(u32, byte_count, 36);
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [8192]u8 = undefined;
    var writer = file.writer(io, &buffer);
    var header: [44]u8 = undefined;
    @memcpy(header[0..4], "RIFF");
    std.mem.writeInt(u32, header[4..8], riff_size, .little);
    @memcpy(header[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, header[16..20], 16, .little);
    std.mem.writeInt(u16, header[20..22], 1, .little);
    std.mem.writeInt(u16, header[22..24], 1, .little);
    std.mem.writeInt(u32, header[24..28], Mimi.sampling_rate, .little);
    std.mem.writeInt(u32, header[28..32], Mimi.sampling_rate * 2, .little);
    std.mem.writeInt(u16, header[32..34], 2, .little);
    std.mem.writeInt(u16, header[34..36], 16, .little);
    @memcpy(header[36..40], "data");
    std.mem.writeInt(u32, header[40..44], byte_count, .little);
    try writer.interface.writeAll(&header);
    for (samples) |sample| {
        if (!std.math.isFinite(sample)) return error.InvalidAudio;
        const quantized: i16 = @intFromFloat(@round(std.math.clamp(sample * 32768, -32768, 32767)));
        var bytes: [2]u8 = undefined;
        std.mem.writeInt(i16, &bytes, quantized, .little);
        try writer.interface.writeAll(&bytes);
    }
    try writer.interface.flush();
}
