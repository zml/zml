//! Loads a Laya checkpoint, compiles it once and answers typed questions.

const std = @import("std");

const zml = @import("zml");

const model = @import("model.zig");
const prompt = @import("prompt.zig");

const log = std.log.scoped(.laya);

/// Maximum number of options per question the executable is compiled for.
pub const max_options = 64;

pub const Options = struct {
    /// Compiled sequence length. Prompts are padded to it; state is truncated past it.
    seqlen: ?u32 = null,
    dtype: zml.DataType = .f32,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,

    registry: zml.safetensors.TensorRegistry,
    store: zml.io.TensorStore,
    encoder_config: std.json.Parsed(model.EncoderConfig),
    agent_config: std.json.Parsed(model.AgentConfig),
    laya: model.Laya,
    buffers: zml.Bufferized(model.Laya),

    tokenizer: zml.tokenizer.Tokenizer,
    special: prompt.SpecialTokens,
    calibration: prompt.Calibration,
    seqlen: u32,

    /// One executable per sequence length bucket, shortest first.
    buckets: []Bucket,

    const Bucket = struct { seqlen: u32, exe: zml.Exe };

    pub fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, repo: std.Io.Dir, options: Options) !*Engine {
        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.io = io;
        self.platform = platform;

        self.encoder_config = try parseJsonFile(model.EncoderConfig, allocator, io, repo, "encoder/config.json");
        errdefer self.encoder_config.deinit();
        self.agent_config = try parseJsonFile(model.AgentConfig, allocator, io, repo, "rl_agent_config.json");
        errdefer self.agent_config.deinit();

        self.seqlen = @min(options.seqlen orelse self.agent_config.value.max_len, self.agent_config.value.max_len);
        if (self.seqlen <= self.agent_config.value.head_max_len) return error.SeqlenTooShort;

        self.registry = try .fromRepo(allocator, io, repo);
        errdefer self.registry.deinit();
        self.store = .fromRegistry(allocator, &self.registry);
        errdefer self.store.deinit();

        self.laya = try .init(allocator, self.store.view(), self.encoder_config.value, self.agent_config.value, .{ .dtype = options.dtype });
        errdefer self.laya.deinit(allocator);

        self.tokenizer = try loadTokenizer(allocator, io, repo);
        errdefer self.tokenizer.deinit();
        self.special = try .init(&self.tokenizer);
        self.calibration = .init(self.agent_config.value);

        self.buckets = try compileBuckets(allocator, io, platform, self.laya, self.seqlen, options.dtype);
        errdefer deinitBuckets(allocator, self.buckets);

        {
            log.info("Loading weights...", .{});
            const start: std.Io.Timestamp = .now(io, .awake);
            self.buffers = try self.laya.load(allocator, io, platform, &self.store);
            log.info("✅ Loaded weights [{f}]", .{start.untilNow(io, .awake)});
        }

        return self;
    }

    pub fn deinit(self: *Engine) void {
        model.Laya.unloadBuffers(&self.buffers, self.allocator);
        deinitBuckets(self.allocator, self.buckets);
        self.tokenizer.deinit();
        self.laya.deinit(self.allocator);
        self.store.deinit();
        self.registry.deinit();
        self.agent_config.deinit();
        self.encoder_config.deinit();
        self.allocator.destroy(self);
    }

    pub const Result = struct {
        answers: []prompt.Answer,
        input_tokens: usize,
        latency_ms: f64,
        /// Token ids fed to the model for each question, for inspection.
        prompts: []prompt.Encoded,
    };

    /// Answers every question about `state`. All memory comes from `arena`.
    pub fn decide(self: *Engine, arena: std.mem.Allocator, state: std.json.Value, questions_json: std.json.Value) !Result {
        const questions = try prompt.parseQuestions(arena, questions_json);
        const state_text = try prompt.renderState(arena, state);

        var encoder = try self.tokenizer.encoder();
        defer encoder.deinit();
        const builder: prompt.Builder = .{
            .arena = arena,
            .encoder = &encoder,
            .special = self.special,
            .max_len = self.seqlen,
            .head_max_len = self.agent_config.value.head_max_len,
        };

        const answers = try arena.alloc(prompt.Answer, questions.len);
        const prompts = try arena.alloc(prompt.Encoded, questions.len);
        var input_tokens: usize = 0;
        const start: std.Io.Timestamp = .now(self.io, .awake);

        for (questions, answers, prompts) |question, *answer, *encoded| {
            if (question.options.len > max_options) return error.TooManyOptions;
            encoded.* = try builder.build(question, state_text);
            input_tokens += encoded.ids.len;

            const bucket = for (self.buckets) |b| {
                if (b.seqlen >= encoded.ids.len) break b;
            } else unreachable;
            var args = try bucket.exe.args(self.allocator);
            defer args.deinit(self.allocator);
            var results = try bucket.exe.results(self.allocator);
            defer results.deinit(self.allocator);

            const tokens = try arena.alloc(u32, bucket.seqlen);
            @memset(tokens, self.special.pad);
            @memcpy(tokens[0..encoded.ids.len], encoded.ids);
            var markers: [max_options]u32 = @splat(0);
            @memcpy(markers[0..encoded.markers.len], encoded.markers);

            var tokens_buf = try self.bufferFrom(.{ .s = bucket.seqlen }, std.mem.sliceAsBytes(tokens));
            defer tokens_buf.deinit();
            var length_buf = try self.scalarBuffer(@intCast(encoded.ids.len));
            defer length_buf.deinit();
            var markers_buf = try self.bufferFrom(.{ .m = max_options }, std.mem.sliceAsBytes(&markers));
            defer markers_buf.deinit();
            var n_markers_buf = try self.scalarBuffer(@intCast(encoded.markers.len));
            defer n_markers_buf.deinit();
            var qtype_buf = try self.scalarBuffer(@intFromEnum(question.type));
            defer qtype_buf.deinit();

            args.set(.{ self.buffers, tokens_buf, length_buf, markers_buf, n_markers_buf, qtype_buf });
            bucket.exe.call(args, &results);
            var logits_buf, var act_buf = results.get(struct { zml.Buffer, zml.Buffer });
            defer logits_buf.deinit();
            defer act_buf.deinit();

            const logits = try logits_buf.toSliceAlloc(arena, self.io);
            const act = try act_buf.toSliceAlloc(arena, self.io);
            answer.* = try prompt.decide(arena, self.calibration, question, logits.constItems(f32), act.constItems(f32));
        }

        const elapsed = start.untilNow(self.io, .awake);
        return .{
            .answers = answers,
            .input_tokens = input_tokens,
            .latency_ms = @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_ms,
            .prompts = prompts,
        };
    }

    /// Decoded text of one token, for display. Special tokens keep their name.
    pub fn tokenText(self: *Engine, arena: std.mem.Allocator, id: u32) []const u8 {
        var decoder = self.tokenizer.decoder() catch return "?";
        defer decoder.deinit();
        const text = decoder.decodeAlloc(arena, &.{id}) catch return "?";
        if (text.items.len > 0) return text.items;
        inline for (.{ "cls", "sep", "pad", "mask" }, .{ "[CLS]", "[SEP]", "[PAD]", "[MASK]" }) |field, name| {
            if (@field(self.special, field) == id) return name;
        }
        return "";
    }

    fn bufferFrom(self: *Engine, shape: anytype, bytes: []const u8) !zml.Buffer {
        return .fromBytes(self.io, self.platform, zml.Shape.init(shape, .u32), .replicated, bytes);
    }

    fn scalarBuffer(self: *Engine, value: u32) !zml.Buffer {
        return self.bufferFrom(.{}, std.mem.asBytes(&value));
    }
};

/// Buckets of 128, 256, ... up to `max_seqlen`: short prompts skip most of the padding.
fn compileBuckets(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, laya: model.Laya, max_seqlen: u32, dtype: zml.DataType) ![]Engine.Bucket {
    var buckets: std.ArrayList(Engine.Bucket) = .empty;
    errdefer {
        for (buckets.items) |b| b.exe.deinit();
        buckets.deinit(allocator);
    }

    var seqlen: u32 = @min(128, max_seqlen);
    while (true) : (seqlen = @min(seqlen * 2, max_seqlen)) {
        log.info("Compiling Laya (seqlen={d}, dtype={t})...", .{ seqlen, dtype });
        const start: std.Io.Timestamp = .now(io, .awake);
        const exe = try platform.compile(allocator, io, laya, .forward, .{
            zml.Tensor.init(.{ .s = seqlen }, .u32),
            zml.Tensor.init(.{}, .u32),
            zml.Tensor.init(.{ .m = max_options }, .u32),
            zml.Tensor.init(.{}, .u32),
            zml.Tensor.init(.{}, .u32),
        }, .{});
        try buckets.append(allocator, .{ .seqlen = seqlen, .exe = exe });
        log.info("✅ Compiled [{f}]", .{start.untilNow(io, .awake)});
        if (seqlen == max_seqlen) break;
    }
    return buckets.toOwnedSlice(allocator);
}

fn deinitBuckets(allocator: std.mem.Allocator, buckets: []Engine.Bucket) void {
    for (buckets) |b| b.exe.deinit();
    allocator.free(buckets);
}

fn parseJsonFile(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !std.json.Parsed(T) {
    const file = dir.openFile(io, path, .{}) catch |err| {
        log.err("Laya checkpoint is missing {s}: {t}", .{ path, err });
        return err;
    };
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const file = try dir.openFile(io, "tokenizer/tokenizer.json", .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);
    return try .fromBytes(allocator, bytes);
}
