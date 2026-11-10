const std = @import("std");
const zml = @import("zml");

const Model = @import("model.zig").Gradium_Phonon_TTS_For_Conditional_Generation;
const util = @import("../util.zig");

// ---

/// Text to latent audio.
pub const Phonon = struct {
    io: std.Io,
    gpa: std.mem.Allocator,

    platform: *const zml.Platform,

    model: Model,
    weights: zml.Bufferized(Model),

    voice: zml.Buffer,
    start_time: zml.Buffer,
    end_time: zml.Buffer,

    /// Capacity-wide prompt IDs, padded after the current chunk with audio markers.
    ids: zml.Buffer,
    /// Single audio-frame marker consumed by the decode executable.
    decode_ids: zml.Buffer,

    /// Starts at num_tokens and increments once per generated frame.
    active_token_length_host: i32,
    /// [2] i32: valid token positions in slot 0, completed audio frames in slot 1.
    /// Initialized to [num_tokens, 0]; both counters increment once per step.
    /// Gets swapped with the donated output buffer after each execution.
    active_lengths: zml.Buffer,
    /// Storage space for the generated latent frame. Written to by Phonon, read from by Mimi.
    //
    // TODO: double-buffering this would make TSS<>Decoding pipelining possible -- no idea if that's useful though.
    frame: zml.Buffer,
    history: zml.Buffer,
    history34: zml.Buffer,
    rng: zml.Tensor.Rng.Buffer,

    /// Shared backend metadata, read-only across layers and generation steps.
    attention_metadata: zml.attention.Metadata,
    attention_metadata_buffers: zml.Bufferized(zml.attention.Metadata),
    attention_parameters: zml.attention.Parameters,

    /// Fixed prefill width, including voice, text, and BOS positions.
    max_prompt_tokens: usize,
    /// Fixed latent-history budget used when compiling and allocating buffers.
    frame_capacity: usize,
    /// Length of the current chunk's fully expanded prompt.
    num_tokens: usize,
    /// Fixed cache capacity: max_prompt_tokens + frame_capacity.
    max_tokens: usize,
    /// Generation limit for the current chunk; at most frame_capacity.
    max_frames: usize,
    /// How many audio frames should we compute after EOS (some weird heuristics).
    frames_after_eos: usize,

    remaining: ?usize = null,
    done: bool = false,
    stopped_by_eos: bool = false,

    prefill_exe: zml.Exe,
    prefill_runner: zml.Exe.Runner,
    decode_exe: zml.Exe,
    decode_runner: zml.Exe.Runner,

    /// Prepared host inputs, borrowed only until init_in_place returns.
    /// The caller owns both slices; Phonon uploads its own device copies.
    pub const Input = struct {
        tokens: zml.Slice,
        voice: zml.Slice,
        max_frames: usize,
        frames_after_eos: usize,
    };

    pub const Capacity = struct {
        max_prompt_tokens: usize,
        max_frames: usize,
    };

    /// `self` can be fully undefined, as long as it's heap allocated.
    /// Accepts expanded i64 prompt IDs [1,S], f32 voice frames [1,P,512],
    /// and generation limits computed by the caller; does no text processing.
    pub fn init_in_place(
        self: *Phonon,
        io: std.Io,
        gpa: std.mem.Allocator,
        platform: *const zml.Platform,
        input: Input,
        capacity: Capacity,
        dtype: zml.DataType,
        trace_context: ?zml.tracer.Context,
    ) !void {
        const here = @typeName(@This()) ++ ".init_in_place";
        std.debug.assert(dtype == .f32 or dtype == .f16 or dtype == .bf16);

        self.io = io;
        self.gpa = gpa;
        self.platform = platform;
        self.remaining = null;
        self.done = false;
        self.stopped_by_eos = false;
        self.max_prompt_tokens = capacity.max_prompt_tokens;
        self.frame_capacity = capacity.max_frames;
        self.max_frames = input.max_frames;
        self.frames_after_eos = input.frames_after_eos;
        const tokens = input.tokens;
        const voice = input.voice;

        var span = zml.tracer.span_at_for(@src(), here, .{}, .{
            .consumer = trace_context,
        });
        defer span.end();

        var span_load_registry = zml.tracer.span_at(@src(), here ++ "/load_registry", .{});
        {}
        const model_path = try util.assetPath(gpa, util.gradium_phonon_31m_en);
        defer gpa.free(model_path);
        var registry = try zml.safetensors.TensorRegistry.fromPath(gpa, io, model_path);
        defer registry.deinit();
        var store: zml.io.TensorStore = .fromRegistry(gpa, &registry, &.{});
        defer store.deinit();
        {}
        span_load_registry.end();

        self.num_tokens = tokens.constItems(i64).len;
        if (self.num_tokens < 2 or self.num_tokens > self.max_prompt_tokens or input.max_frames == 0 or input.max_frames > self.frame_capacity)
            return error.InvalidChunkCapacity;
        self.max_tokens = try std.math.add(usize, self.max_prompt_tokens, self.frame_capacity);

        // Token and frame indices must fit the i32 counters.
        if (self.max_tokens > std.math.maxInt(i32)) return error.SequenceTooLong;

        self.model = .init(store.view(), self.max_tokens);
        for (&self.model.model.layers) |*layer| {
            const cache = &layer.self_attn.kv_cache;
            cache.k = .fromShape(cache.k.shape().withDtype(dtype));
            cache.v = .fromShape(cache.v.shape().withDtype(dtype));
        }
        try self.load(&store);
        errdefer zml.Buffer.deinitAll(Model, &self.weights);

        for (&self.model.model.layers, &self.weights.model.layers) |*layer, *buffers| {
            const spec = layer.self_attn.kv_cache;
            const cache = &buffers.self_attn.kv_cache;

            // Decode reads capacity-wide caches. Masked future V entries must be
            // finite: a zero attention weight multiplied by NaN is still NaN.
            const zeros = try gpa.alloc(u8, spec.k.shape().byteSize());
            defer gpa.free(zeros);
            @memset(zeros, 0);
            cache.k = try zml.Buffer.fromBytes(io, platform, spec.k.shape(), zeros);
            cache.v = try zml.Buffer.fromBytes(io, platform, spec.v.shape(), zeros);
        }

        // FIX: why not use a constant for this?
        // -> would be interesting to see the impact on first-token latency
        self.voice = blk: {
            var span_upload_voice = zml.tracer.span_at(@src(), here ++ "/voice", .{});
            defer span_upload_voice.end();

            if (dtype == .f32) break :blk try zml.Buffer.fromSlice(io, platform, voice);

            const shape = voice.shape.withDtype(dtype);
            const bytes = try gpa.alloc(u8, shape.byteSize());
            defer gpa.free(bytes);
            for (voice.constItems(f32), 0..) |value, i| {
                const converted = dtype.constant(value);
                const offset = i * dtype.sizeOf();
                @memcpy(bytes[offset..][0..dtype.sizeOf()], converted.asBytes());
            }
            break :blk try zml.Buffer.fromBytes(io, platform, shape, bytes);
        };
        errdefer self.voice.deinit();

        const time_shape: zml.Shape = .init(.{ .b = 1, .t = 1 }, .f32);
        self.start_time = try zml.Buffer.fromBytes(io, platform, time_shape, std.mem.asBytes(&@as([1]f32, .{0})));
        errdefer self.start_time.deinit();
        self.end_time = try zml.Buffer.fromBytes(io, platform, time_shape, std.mem.asBytes(&@as([1]f32, .{1})));
        errdefer self.end_time.deinit();

        self.active_token_length_host = @intCast(self.num_tokens);
        const initial_lengths: [2]i32 = .{ self.active_token_length_host, 0 };
        self.active_lengths = try zml.Buffer.fromBytes(io, platform, .init(.{2}, .i32), std.mem.asBytes(&initial_lengths));
        errdefer self.active_lengths.deinit();

        self.ids = try upload_ids(io, gpa, platform, tokens, self.max_prompt_tokens);

        errdefer self.ids.deinit();
        const audio_marker = [_]i64{6};
        self.decode_ids = try zml.Buffer.fromBytes(io, platform, .init(.{ .b = 1, .s = 1 }, .i64), std.mem.asBytes(&audio_marker));
        errdefer self.decode_ids.deinit();

        self.frame = try zml.Buffer.uninitialized(io, platform, .init(.{ .b = 1, .s = 1, .ld = 32 }, .f32), .{});
        errdefer self.frame.deinit();
        const rng_spec: zml.Tensor.Rng = .init();
        self.rng = try rng_spec.initBuffer(io, platform, 42);
        errdefer zml.Tensor.Rng.deinitBuffer(&self.rng);

        self.history = blk: {
            const shape: zml.Shape = .init(.{ .b = 1, .a = @max(34, self.frame_capacity), .ld = 32 }, .f32);
            const zeros = try zml.Slice.alloc(gpa, shape);
            defer zeros.free(gpa);
            @memset(zeros.items(f32), 0);
            break :blk try zml.Buffer.fromSlice(io, platform, zeros);
        };
        errdefer self.history.deinit();
        self.history34 = try zml.Buffer.uninitialized(io, platform, .init(.{ .b = 1, .a = 34, .ld = 32 }, .f32), .{});

        errdefer self.history34.deinit();

        const attention_backend = zml.attention.Backend.auto(platform);
        self.attention_parameters = .init(.fromBackend(attention_backend));
        self.attention_metadata = .init(.fromBackend(attention_backend));
        self.attention_metadata_buffers = try self.attention_metadata.initBuffer(io, platform);
        errdefer zml.attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);

        self.prefill_exe = try platform.compile(gpa, io, self.model, .forward, .{
            zml.Tensor.fromShape(self.ids.shape()),
            zml.Tensor.fromShape(self.voice.shape()),
            zml.Tensor.fromShape(self.history.shape()),
            zml.Tensor.fromShape(self.history34.shape()),
            zml.Tensor.fromShape(self.start_time.shape()),
            zml.Tensor.fromShape(self.end_time.shape()),
            zml.Tensor.fromShape(self.active_lengths.shape()),
            zml.Tensor.fromShape(self.frame.shape()),
            zml.Tensor.Rng.init(),
            self.attention_metadata,
            self.attention_parameters,
        }, .{
            .program_name = @typeName(@This()) ++ ".prefill",
        });
        errdefer self.prefill_exe.deinit();
        util.dumpMemoryStats(@typeName(@This()) ++ ".prefill", &self.prefill_exe);

        self.prefill_runner = try self.prefill_exe.runner(gpa);
        errdefer self.prefill_runner.deinit(gpa);

        self.decode_exe = try platform.compile(gpa, io, self.model, .forward, .{
            zml.Tensor.fromShape(self.decode_ids.shape()),
            zml.Tensor.fromShape(self.voice.shape()),
            zml.Tensor.fromShape(self.history.shape()),
            zml.Tensor.fromShape(self.history34.shape()),
            zml.Tensor.fromShape(self.start_time.shape()),
            zml.Tensor.fromShape(self.end_time.shape()),
            zml.Tensor.fromShape(self.active_lengths.shape()),
            zml.Tensor.fromShape(self.frame.shape()),
            zml.Tensor.Rng.init(),
            self.attention_metadata,
            self.attention_parameters,
        }, .{
            .program_name = @typeName(@This()) ++ ".decode",
        });
        errdefer self.decode_exe.deinit();
        util.dumpMemoryStats(@typeName(@This()) ++ ".decode", &self.decode_exe);

        self.decode_runner = try self.decode_exe.runner(gpa);
        errdefer self.decode_runner.deinit(gpa);
    }

    /// Start an independent utterance without loading weights or recompiling.
    /// All inputs must fit the original capacity and use the selected voice.
    /// Caches stay allocated: prefill overwrites the whole prompt-width prefix;
    /// causal attention ignores every retained entry beyond the new valid length.
    /// History and RNG are reset so previous utterances cannot condition Mimi or
    /// Phonon, and each chunk starts from the same deterministic seed (42).
    pub fn start_chunk(self: *Phonon, input: Input) !void {
        var span = zml.tracer.span_at(@src(), @typeName(@This()) ++ ".start_chunk", .{});
        defer span.end();
        const num_tokens = input.tokens.constItems(i64).len;
        if (num_tokens < 2 or num_tokens > self.max_prompt_tokens or input.max_frames == 0 or input.max_frames > self.frame_capacity)
            return error.InvalidChunkCapacity;

        var ids = try upload_ids(self.io, self.gpa, self.platform, input.tokens, self.max_prompt_tokens);
        errdefer ids.deinit();
        const counts: [2]i32 = .{ @intCast(num_tokens), 0 };
        var lengths = try zml.Buffer.fromBytes(self.io, self.platform, self.active_lengths.shape(), std.mem.asBytes(&counts));
        errdefer lengths.deinit();
        const zeros = try zml.Slice.alloc(self.gpa, self.history.shape());
        defer zeros.free(self.gpa);
        @memset(zeros.items(f32), 0);
        var history = try zml.Buffer.fromSlice(self.io, self.platform, zeros);
        errdefer history.deinit();
        const rng_spec: zml.Tensor.Rng = .init();
        var rng = try rng_spec.initBuffer(self.io, self.platform, 42);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng);

        // Replace handles only once every new allocation has succeeded.
        self.ids.deinit();
        self.active_lengths.deinit();
        self.history.deinit();
        zml.Tensor.Rng.deinitBuffer(&self.rng);
        self.ids = ids;
        self.active_lengths = lengths;
        self.history = history;
        self.rng = rng;
        self.num_tokens = num_tokens;
        self.active_token_length_host = counts[0];
        self.max_frames = input.max_frames;
        self.frames_after_eos = input.frames_after_eos;
        self.remaining = null;
        self.done = false;
        self.stopped_by_eos = false;
    }

    fn upload_ids(io: std.Io, allocator: std.mem.Allocator, platform: *const zml.Platform, tokens: zml.Slice, width: usize) !zml.Buffer {
        const padded = try zml.Slice.alloc(allocator, .init(.{ .b = 1, .s = width }, .i64));
        defer padded.free(allocator);
        @memset(padded.items(i64), 6);
        @memcpy(padded.items(i64)[0..tokens.constItems(i64).len], tokens.constItems(i64));
        return zml.Buffer.fromSlice(io, platform, padded);
    }

    /// Load all Phonon parameters through one loader; inner modules only describe tensors.
    pub fn load(self: *Phonon, store: *const zml.io.TensorStore) !void {
        var span = zml.tracer.span_at(@src(), @typeName(@This()) ++ ".load", .{});
        defer span.end();

        self.weights = try self.model.bufferize(self.gpa);
        errdefer zml.Buffer.deinitAll(Model, &self.weights);
        var loader: zml.io.Loader = try .init(self.gpa, self.platform, .default);
        defer loader.deinit();
        try loader.load(self.io, Model, &self.model, &self.weights, store, .{});
        try loader.await(self.io);
    }

    pub fn deinit(self: *Phonon) void {
        const here = @typeName(@This()) ++ ".deinit";

        const gpa = self.gpa;

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        zml.attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);
        self.end_time.deinit();
        self.start_time.deinit();
        self.voice.deinit();
        zml.Buffer.deinitAll(Model, &self.weights);

        self.ids.deinit();
        self.decode_ids.deinit();
        self.active_lengths.deinit();
        self.frame.deinit();
        self.history34.deinit();
        self.history.deinit();
        zml.Tensor.Rng.deinitBuffer(&self.rng);

        self.decode_runner.deinit(gpa);
        self.decode_exe.deinit();
        self.prefill_runner.deinit(gpa);
        self.prefill_exe.deinit();
    }

    pub fn frames(self: *const Phonon) usize {
        return @as(usize, @intCast(self.active_token_length_host)) - self.num_tokens;
    }

    pub const Step_Result = struct {
        eos_score: f32,
        /// Borrowed [2] i32 buffer: token and frame counts after this step.
        active_lengths: zml.Buffer,
        frame: zml.Buffer,
        history: zml.Buffer,
        history34: zml.Buffer,
    };

    /// Generates and retains one normalized frame, returning its raw EOS score.
    pub fn forward(self: *Phonon) !Step_Result {
        if (self.done) return error.GenerationFinished;

        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        // const gpa = self.gpa;
        const io = self.io;

        var result: zml.Bufferized(Model.Output) = undefined;
        const prefill = self.frames() == 0;
        const runner = if (prefill) &self.prefill_runner else &self.decode_runner;
        const ids = if (prefill) self.ids else self.decode_ids;
        runner.run(io, .{
            self.weights,
            ids,
            self.voice,
            self.history,
            self.history34,
            self.start_time,
            self.end_time,
            self.active_lengths,
            self.frame,
            self.rng,
            self.attention_metadata_buffers,
            self.attention_parameters,
        }, .{&result}, .{ .wait = false });

        self.active_token_length_host += 1;
        self.active_lengths = result.active_lengths;
        self.frame = result.frame;
        self.history = result.history;
        self.history34 = result.history34;
        self.rng = result.rng;
        // Runner destroyed the donated handles; retain the replacement handles for the next call.
        for (&self.weights.model.layers, result.kv_caches) |*layer, cache| {
            layer.self_attn.kv_cache = cache;
        }
        defer result.eos_score.deinit();
        // defer zml.Buffer.deinitAll(Model.Output, &result);

        const score = try result.eos_score.getValue(f32, io);

        // Retain the triggering frame and the configured tail before stopping.
        if (self.remaining == null and score > 4.0) self.remaining = self.frames_after_eos;
        if (self.remaining) |remaining| {
            if (remaining == 0) {
                self.done = true;
                self.stopped_by_eos = true;
            } else self.remaining = remaining - 1;
        }
        if (self.frames() >= self.max_frames) self.done = true;

        return .{
            .eos_score = score,
            .active_lengths = self.active_lengths,
            .frame = self.frame,
            .history = self.history,
            .history34 = self.history34,
        };
    }
};
