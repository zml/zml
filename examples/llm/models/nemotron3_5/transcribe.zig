const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const frontend = @import("frontend.zig");
const model = @import("model.zig");
const perception = @import("perception.zig");
const session = @import("session.zig");

const log = std.log.scoped(.nemotron3_5);

pub const EncodingOptions = struct {
    config: model.Config,
    meshes: common.Meshes,
    output_dtype: zml.DataType,
    seqlen: u32,
};

pub const EncodedAudio = struct {
    hidden: zml.Buffer,
    length: u32,

    pub fn deinit(self: *EncodedAudio) void {
        self.hidden.deinit();
    }
};

const EncodeInput = struct {
    perception: perception.Perception,
    input: perception.Input,
    output_dtype: zml.DataType,
};

pub fn run(language_session: *session.Session, audio: *const EncodedAudio, writer: *std.Io.Writer) !void {
    const allocator = language_session.allocator;
    const io = language_session.io;

    const prefill_start: std.Io.Timestamp = .now(io, .awake);
    try language_session.runAudioPrefill(&audio.hidden, audio.length);
    log.info("Audio prompt prefill [{f}]", .{prefill_start.untilNow(io, .awake)});

    var generated: std.ArrayList(u32) = .empty;
    defer generated.deinit(allocator);

    try language_session.runDecode(&generated, writer);
    try writer.writeAll("\n");
    try writer.flush();
}

pub fn encode(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    features: frontend.Features,
    options: EncodingOptions,
    parent_progress: *std.Progress.Node,
) !EncodedAudio {
    const config = options.config;
    const audio_config = config.perception orelse return error.MissingPerceptionConfig;
    if (audio_config.output_dim != config.llm_config.hidden_size)
        return error.InvalidPerceptionConfig;

    const encoder_config = if (config.pe_encoder_config) |pe| pe.asr_encoder_cfg else audio_config.encoder;
    try encoder_config.validate();

    const factor: usize = @intCast(encoder_config.subsampling_factor);
    const audio_length = std.math.divCeil(usize, features.valid_frame_count, factor) catch return error.InvalidPerceptionInput;
    if (audio_length == 0)
        return error.EmptyAudio;

    if (audio_length >= options.seqlen) {
        log.err("Audio needs {d} embedding positions plus the transcription prompt and generated text. Increase --seqlen (currently {d})", .{ audio_length, options.seqlen });
        return error.PromptTooLong;
    }

    log.info("Running offline full-recording perception ({d} audio positions)", .{audio_length});

    const mdl = try perception.Perception.init(allocator, store.view().withPrefix("perception"), audio_config, config.pe_encoder_config, config.diarization_model_cfg);
    defer mdl.deinit(allocator);

    var inputs = try perception.Input.initBuffers(io, platform, features, encoder_config);
    defer perception.Input.deinitBuffers(&inputs);

    var progress = parent_progress.start("Audio perception", 0);
    defer progress.end();

    const compile_start: std.Io.Timestamp = .now(io, .awake);
    const exe = try zml.FnExe(encodeAudio).compile(allocator, io, platform, .{
        .meshes = &options.meshes.all(),
    }, .{
        .{
            .perception = mdl,
            .input = .{ .features = .fromShape(inputs.features.shape()), .lengths = .fromShape(inputs.lengths.shape()) },
            .output_dtype = options.output_dtype,
        },
    });
    defer exe.deinit();

    log.info("Compiled perception [{f}]", .{compile_start.untilNow(io, .awake)});

    var buffers = try zml.mem.bufferize(allocator, perception.Perception, &mdl);
    defer perception.Perception.unloadBuffers(&buffers, allocator);

    var loader = try zml.io.Loader.init(allocator, platform, .{ .parallelism = 8, .dma_chunks = 4, .dma_chunk_size = 32 * zml.MiB });
    defer loader.deinit();
    progress.increaseEstimatedTotalItems(zml.meta.count(zml.Tensor, &mdl));
    try loader.load(io, perception.Perception, &mdl, &buffers, store, .{ .progress = &progress });
    try loader.await(io);

    var runner = try zml.FnExe(encodeAudio).Runner(.{.perception}).init(&exe, allocator, .{ .perception = buffers });
    defer runner.deinit(allocator);

    var output: zml.Bufferized(perception.Output) = undefined;
    runner.run(io, .{
        .inputs = .{ .input = inputs },
        .outputs = .{ .hidden = &output.hidden, .lengths = &output.lengths },
    }, .{});

    errdefer zml.Buffer.deinitAll(perception.Output, &output);

    try output.hidden.await(io);

    const length = try output.lengths.getValue(u32, io);
    if (length != audio_length)
        return error.InvalidPerceptionOutputLength;

    output.lengths.deinit();

    return .{
        .hidden = output.hidden,
        .length = length,
    };
}

fn encodeAudio(input: EncodeInput) perception.Output {
    var output = input.perception.forward(input.input);
    output.hidden = output.hidden.convert(input.output_dtype);
    return output;
}
