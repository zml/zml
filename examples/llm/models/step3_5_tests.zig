const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const inference = @import("step3_5flash/inference.zig");
const step3p5flash = @import("step3_5flash.zig");
const model = @import("step3_5flash/model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.step3_5_tests);

const Args = struct {
    model: []const u8,
    prompt: ?[]const u8 = null,
    activations: []const u8 = "",
    generate: bool = false,
    seqlen: u32 = 64,
    start_token: u32 = 1,
    max_new_tokens: u32 = 32,
    topk: u32 = 1,

    pub const help =
        \\Use step3_5_tests --model=<path> [--generate] [--seqlen=N] ...
        \\
        \\ Step 3.5 Flash decode smoke test.
        \\
        \\ Options:
        \\   --model=<path>            Path to the model repository (safetensors + config.json)
        \\   --prompt=<text>           Optional prompt fed one token at a time before decode
        \\   --activations=<path>      (optional, unused for now) activations fixture path
        \\   --generate                Run the decode loop instead of returning early
        \\   --seqlen=<u32>            Compiled KV-cache length (default 64)
        \\   --start-token=<u32>       Token id seeded into the decoder when no --prompt (default 1)
        \\   --max-new-tokens=<u32>    Number of tokens to generate (default 32)
        \\   --topk=<u32>              Top-k sampling cutoff (default 1 = greedy)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const shardings: common.Shardings = try .init(platform);

    var repo_model = try step3p5flash.LoadedModel.init(allocator, io, repo, store.view(), shardings);
    defer repo_model.deinit(allocator);

    repo_model.inner.gen_options.sampling_strategy.topk = args.topk;

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    errdefer progress.end();

    var model_buffers = try repo_model.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer repo_model.unloadBuffers(&model_buffers, allocator);

    if (!args.generate) {
        progress.end();
        log.info("non-generate mode not implemented; pass --generate", .{});
        return;
    }

    var compiled = try repo_model.compile(allocator, io, platform, shardings, args.seqlen, &progress);
    defer compiled.deinit();

    progress.end();

    // Optional prompt: encode now so we can fail fast before touching the device loop.
    var tokenizer: ?zml.tokenizer.Tokenizer = null;
    defer if (tokenizer) |*t| t.deinit();
    var prompt_tokens: []const u32 = &.{};
    defer if (prompt_tokens.len > 0) allocator.free(prompt_tokens);
    if (args.prompt) |prompt_text| {
        tokenizer = try loadTokenizer(allocator, io, repo);
        var encoder = try tokenizer.?.encoder();
        defer encoder.deinit();
        prompt_tokens = try encoder.encodeAlloc(allocator, prompt_text);
        if (prompt_tokens.len + args.max_new_tokens > args.seqlen) {
            log.warn("prompt ({d}) + max_new_tokens ({d}) > seqlen ({d}); raise --seqlen", .{ prompt_tokens.len, args.max_new_tokens, args.seqlen });
            return;
        }
    }

    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &.{});
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    try runDecode(
        allocator,
        io,
        platform,
        &compiled,
        &model_buffers,
        tokenizer,
        prompt_tokens,
        args.start_token,
        args.max_new_tokens,
        stdout,
    );
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const file = try dir.openFile(io, "tokenizer.json", .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);
    return try .fromBytes(allocator, bytes);
}

/// Pure-decode loop: optionally walks the prompt through the single-step exe
/// (one token at a time, since the exe is compiled at s=1), then continues
/// free-running decode. No EOS handling.
fn runDecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const inference.CompiledModel,
    model_buffers: *model.Buffers,
    tokenizer: ?zml.tokenizer.Tokenizer,
    prompt_tokens: []const u32,
    start_token: u32,
    max_new_tokens: u32,
    stdout: *std.Io.Writer,
) !void {
    // KV cache is updated in-place via reuseBuffer.
    var kv_cache_buffers = try compiled.params.kv_cache.initBuffer(allocator, io, platform, compiled.params.shardings.model);
    defer model.KvCache.deinitBuffer(&kv_cache_buffers);

    // RNG state for the sampler.
    const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
    var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    // Current token buffer: shape {.b=1, .s=1}. Same buffer is both input and output every step
    var current_token: u32 = if (prompt_tokens.len > 0) prompt_tokens[0] else start_token;
    var current_token_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        .init(.{ .b = 1, .s = 1 }, .u32),
        .replicated,
        std.mem.asBytes(&current_token),
    );
    defer current_token_buffer.deinit();

    var exe_args = try compiled.exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try compiled.exe.results(allocator);
    defer exe_results.deinit(allocator);

    var detok: ?zml.tokenizer.Tokenizer.Decoder = if (tokenizer) |t| try t.decoder() else null;
    defer if (detok) |*d| d.deinit();
    var detok_buf: [4096]u8 = undefined;

    // Step 3.5 Flash advertises three terminator ids in config.json
    const eos_tokens = [_]u32{ 1, 2, 128007 };

    const total_steps: u32 = @intCast(prompt_tokens.len + max_new_tokens);
    var step: u32 = 0;
    while (step < total_steps) : (step += 1) {
        if (step >= compiled.params.seqlen) break;

        var index_value: u32 = step;
        var token_index_buffer = try zml.Buffer.fromBytes(
            io,
            platform,
            .init(.{ .s = 1 }, .u32),
            .replicated,
            std.mem.asBytes(&index_value),
        );
        defer token_index_buffer.deinit();

        exe_args.set(.{
            model_buffers.*,
            current_token_buffer,
            token_index_buffer,
            kv_cache_buffers,
            rng_buffers,
        });
        compiled.exe.call(exe_args, &exe_results);
        exe_results.fill(.{
            &current_token_buffer,
            &kv_cache_buffers,
            &rng_buffers,
        });

        const predicted = try current_token_buffer.getValue(u32, io);

        const next_step = step + 1;
        const in_prompt = next_step < prompt_tokens.len;
        const emit_token: u32 = if (in_prompt) prompt_tokens[next_step] else predicted;

        if (!in_prompt) {
            if (std.mem.indexOfScalar(u32, &eos_tokens, predicted) != null) break;

            if (detok) |*d| {
                const text = try d.feedOne(predicted, &detok_buf);
                try stdout.writeAll(text);
            } else {
                try stdout.print("{d} ", .{predicted});
            }
            try stdout.flush();
        }

        if (in_prompt) {
            current_token = emit_token;
            current_token_buffer.deinit();
            current_token_buffer = try zml.Buffer.fromBytes(
                io,
                platform,
                .init(.{ .b = 1, .s = 1 }, .u32),
                .replicated,
                std.mem.asBytes(&current_token),
            );
        }
    }

    if (detok) |*d| {
        const tail = try d.finalize(&detok_buf);
        try stdout.writeAll(tail);
    }
    try stdout.writeAll("\n");
    try stdout.flush();
}
