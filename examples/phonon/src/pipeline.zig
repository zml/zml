const std = @import("std");
const zml = @import("zml");

const phonon = @import("phonon/model.zig");
const Phonon_Model = phonon.Gradium_Phonon_TTS_For_Conditional_Generation;
const Mimi_Model = @import("mimi/model.zig").Pocket_TTS_Audio_Tokenizer_Model;
const util = @import("util.zig");

/// One device execution generates a latent frame and decodes its history window.
/// Phonon weights, six K/V cache pairs and Mimi weights are runtime model inputs.
/// Other inputs are prompt/decode IDs [B,S] i64, voice [B,P,512] in the selected
/// dtype, history [B,A,32] f32, flow endpoints [B,1] f32, counts [2] i32, RNG state
/// and backend attention metadata. The host retains the returned donated state.
/// Only the raw EOS scalar and the newest [B,1,1920] f32 PCM chunk are read back.
const Model = struct {
    phonon: Phonon_Model,
    mimi: Mimi_Model,

    pub const Output = struct {
        eos_score: zml.Tensor,
        waveform: zml.Tensor,
        active_lengths: zml.Tensor,
        history: zml.Tensor,
        rng: zml.Tensor.Rng,
        kv_caches: phonon.Gradium_Phonon_TTS_Model.KV_Caches,
    };

    pub fn bufferize(self: *const Model, allocator: std.mem.Allocator) !zml.Bufferized(Model) {
        var phonon_buffers = try self.phonon.bufferize(allocator);
        errdefer zml.Buffer.deinitAll(Phonon_Model, &phonon_buffers);
        return .{ .phonon = phonon_buffers, .mimi = try self.mimi.bufferize(allocator) };
    }

    /// .b is batch (1); .s is incoming token positions (prompt capacity or 1);
    /// .p is voice frames; .ad is 512 voice features; .a is history capacity;
    /// .ld is 32 latent coordinates; .t is one flow-time scalar per batch row.
    /// Mimi consumes the post-step counts and 34-frame window directly as graph
    /// values. Its .s output contains the newest 1920 mono waveform samples.
    pub fn forward(
        self: Model,
        token_ids: zml.Tensor,
        voice: zml.Tensor,
        history: zml.Tensor,
        start_time: zml.Tensor,
        end_time: zml.Tensor,
        active_lengths: zml.Tensor,
        rng: zml.Tensor.Rng,
        attention_metadata: zml.attention.Metadata,
        attention_parameters: zml.attention.Parameters,
    ) Output {
        var span = zml.tracer.span_at(@src(), @typeName(@This()) ++ ".forward", .{});
        defer span.end();
        const compiler = zml.Compiler.current();
        compiler.pushLocationFmt(@src(), @typeName(@This()) ++ ".forward", .{});
        defer compiler.popLocation();

        const next = self.phonon.forward(token_ids, voice, history, start_time, end_time, active_lengths, rng, attention_metadata, attention_parameters); // [B,S] IDs + device state -> updated history [B,A,32] f32 and window [B,34,32] f32
        const waveform = self.mimi.forward(next.active_lengths, next.history34); // [B,34,32] f32 -> [B,1,1920] f32
        return .{
            .eos_score = next.eos_score,
            .waveform = waveform,
            .active_lengths = next.active_lengths,
            .history = next.history,
            .rng = next.rng,
            .kv_caches = next.kv_caches,
        };
    }
};

// ---

/// Text to 24 kHz mono waveforms, with one combined device call per frame.
pub const Pipeline = struct {
    pub const sampling_rate = 24_000;
    pub const samples_per_frame = 1920;
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
    history: zml.Buffer,
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

    /// Prepared utterance inputs, borrowed only during start_chunk. The token
    /// slice is uploaded there; voice is the shared selection already uploaded
    /// by init_in_place. The caller owns both host slices.
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
    /// Loads both models and voice frames [1,P,512], then compiles combined
    /// prefill/decode graphs at fixed capacities.
    /// The instance is idle until start_chunk supplies expanded prompt IDs and
    /// generation limits. No text processing occurs here.
    pub fn init_in_place(
        self: *Pipeline,
        io: std.Io,
        gpa: std.mem.Allocator,
        platform: *const zml.Platform,
        voice: zml.Slice,
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
        self.done = true;
        self.stopped_by_eos = false;
        self.max_prompt_tokens = capacity.max_prompt_tokens;
        self.frame_capacity = capacity.max_frames;
        self.max_frames = 0;
        self.frames_after_eos = 0;

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
        const mimi_path = try util.assetPath(gpa, util.gradium_phonon_mimi);
        defer gpa.free(mimi_path);
        var mimi_registry = try zml.safetensors.TensorRegistry.fromPath(gpa, io, mimi_path);
        defer mimi_registry.deinit();
        var mimi_store: zml.io.TensorStore = .fromRegistry(gpa, &mimi_registry, &.{});
        defer mimi_store.deinit();
        {}
        span_load_registry.end();

        self.num_tokens = 0;
        if (self.max_prompt_tokens < 2 or self.frame_capacity == 0)
            return error.InvalidChunkCapacity;
        self.max_tokens = try std.math.add(usize, self.max_prompt_tokens, self.frame_capacity);

        // Token and frame indices must fit the i32 counters.
        if (self.max_tokens > std.math.maxInt(i32)) return error.SequenceTooLong;

        self.model = .{
            .phonon = .init(store.view(), self.max_tokens),
            .mimi = .init(store.view(), mimi_store.view()),
        };
        for (&self.model.phonon.model.layers) |*layer| {
            const cache = &layer.self_attn.kv_cache;
            cache.k = .fromShape(cache.k.shape().withDtype(dtype));
            cache.v = .fromShape(cache.v.shape().withDtype(dtype));
        }
        try self.load(&store, &mimi_store);
        errdefer zml.Buffer.deinitAll(Model, &self.weights);

        for (&self.model.phonon.model.layers, &self.weights.phonon.model.layers) |*layer, *buffers| {
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

        const empty_ids: zml.Slice = .initConst(.init(.{ .b = 1, .s = 0 }, .i64), &.{});
        self.ids = try upload_ids(io, gpa, platform, empty_ids, self.max_prompt_tokens);

        errdefer self.ids.deinit();
        const audio_marker = [_]i64{6};
        self.decode_ids = try zml.Buffer.fromBytes(io, platform, .init(.{ .b = 1, .s = 1 }, .i64), std.mem.asBytes(&audio_marker));
        errdefer self.decode_ids.deinit();

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

        const attention_backend = zml.attention.Backend.auto(platform);
        self.attention_parameters = .init(.fromBackend(attention_backend));
        self.attention_metadata = .init(.fromBackend(attention_backend));
        self.attention_metadata_buffers = try self.attention_metadata.initBuffer(io, platform);
        errdefer zml.attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);

        // Compile both shape variants concurrently. Runners are created only
        // after the executables have reached their stable, heap-owned addresses.
        const prefill_context: zml.tracer.Context = .init(.generic);
        var prefill_dispatch = zml.tracer.span_at_for(@src(), here ++ "/dispatch_prefill", .{}, .{ .producer = prefill_context });
        var future_prefill = io.async(Pipeline.compile_variant, .{ self, true, prefill_context });
        prefill_dispatch.end();
        const decode_context: zml.tracer.Context = .init(.generic);
        var decode_dispatch = zml.tracer.span_at_for(@src(), here ++ "/dispatch_decode", .{}, .{ .producer = decode_context });
        decode_dispatch.end();
        self.decode_exe = self.compile_variant(false, decode_context) catch |err| {
            if (future_prefill.await(io)) |compiled| {
                var owned = compiled;
                owned.deinit();
            } else |_| {}
            return err;
        };
        errdefer self.decode_exe.deinit();
        self.prefill_exe = try future_prefill.await(io);
        errdefer self.prefill_exe.deinit();
        self.prefill_runner = try self.prefill_exe.runner(gpa);
        errdefer self.prefill_runner.deinit(gpa);
        self.decode_runner = try self.decode_exe.runner(gpa);
    }

    fn compile_variant(self: *Pipeline, prefill: bool, context: zml.tracer.Context) !zml.Exe {
        const name = if (prefill) @typeName(@This()) ++ ".prefill" else @typeName(@This()) ++ ".decode";
        var span = zml.tracer.span_at_for(@src(), @typeName(@This()) ++ ".compile_variant", .{ .prefill = prefill }, .{ .consumer = context });
        defer span.end();
        var exe = try self.platform.compile(self.gpa, self.io, self.model, .forward, .{
            zml.Tensor.fromShape(if (prefill) self.ids.shape() else self.decode_ids.shape()),
            zml.Tensor.fromShape(self.voice.shape()),
            zml.Tensor.fromShape(self.history.shape()),
            zml.Tensor.fromShape(self.start_time.shape()),
            zml.Tensor.fromShape(self.end_time.shape()),
            zml.Tensor.fromShape(self.active_lengths.shape()),
            zml.Tensor.Rng.init(),
            self.attention_metadata,
            self.attention_parameters,
        }, .{ .program_name = name });
        util.dumpMemoryStats(name, &exe);
        return exe;
    }

    /// Start an independent utterance without loading weights or recompiling.
    /// All inputs must fit the original capacity and use the selected voice.
    /// Caches stay allocated: prefill overwrites the whole prompt-width prefix;
    /// causal attention ignores every retained entry beyond the new valid length.
    /// History and RNG are reset so previous utterances cannot condition Mimi or
    /// Phonon, and each chunk starts from the same deterministic seed (42).
    pub fn start_chunk(self: *Pipeline, input: Input) !void {
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

    /// Load both checkpoints once; Mimi normalization statistics belong to Phonon.
    pub fn load(self: *Pipeline, store: *const zml.io.TensorStore, mimi_store: *const zml.io.TensorStore) !void {
        var span = zml.tracer.span_at(@src(), @typeName(@This()) ++ ".load", .{});
        defer span.end();

        self.weights = try self.model.bufferize(self.gpa);
        errdefer zml.Buffer.deinitAll(Model, &self.weights);
        var loader: zml.io.Loader = try .init(self.gpa, self.platform, .default);
        defer loader.deinit();
        try loader.load(self.io, Phonon_Model, &self.model.phonon, &self.weights.phonon, store, .{});
        try loader.load(self.io, Mimi_Model, &self.model.mimi, &self.weights.mimi, mimi_store, .{});
        loader.loadBuffer(self.io, &self.weights.mimi.mean, &self.model.mimi.mean, store, .{});
        loader.loadBuffer(self.io, &self.weights.mimi.stddev, &self.model.mimi.stddev, store, .{});
        try loader.await(self.io);
    }

    pub fn deinit(self: *Pipeline) void {
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
        self.history.deinit();
        zml.Tensor.Rng.deinitBuffer(&self.rng);

        self.decode_runner.deinit(gpa);
        self.decode_exe.deinit();
        self.prefill_runner.deinit(gpa);
        self.prefill_exe.deinit();
    }

    pub fn frames(self: *const Pipeline) usize {
        return @as(usize, @intCast(self.active_token_length_host)) - self.num_tokens;
    }

    /// Generate one frame and decode it in the same device execution. Reads the
    /// EOS score only after launching the complete graph, then downloads the PCM.
    /// The caller owns the returned waveform slice; all donated state stays here.
    pub fn forward(self: *Pipeline) !zml.Slice {
        if (self.done) return error.GenerationFinished;

        const here = @typeName(@This()) ++ ".forward";

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

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
            self.start_time,
            self.end_time,
            self.active_lengths,
            self.rng,
            self.attention_metadata_buffers,
            self.attention_parameters,
        }, .{&result}, .{ .wait = false });

        self.active_token_length_host += 1;
        self.active_lengths = result.active_lengths;
        self.history = result.history;
        self.rng = result.rng;
        // Runner destroyed the donated handles; retain the replacement handles for the next call.
        for (&self.weights.phonon.model.layers, result.kv_caches) |*layer, cache| {
            layer.self_attn.kv_cache = cache;
        }
        defer result.eos_score.deinit();
        defer result.waveform.deinit();

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

        return result.waveform.toSliceAlloc(self.gpa, io);
    }
};
