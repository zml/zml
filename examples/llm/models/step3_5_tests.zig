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
    activations: []const u8 = "",
    generate: bool = false,
    seqlen: u32 = 64,
    start_token: u32 = 1,
    max_new_tokens: u32 = 32,

    pub const help =
        \\Use step3_5_tests --model=<path> [--generate] [--seqlen=N] ...
        \\
        \\ Step 3.5 Flash decode smoke test.
        \\
        \\ Options:
        \\   --model=<path>            Path to the model repository (safetensors + config.json)
        \\   --activations=<path>      (optional, unused for now) activations fixture path
        \\   --generate                Run the decode loop instead of returning early
        \\   --seqlen=<u32>            Compiled KV-cache length (default 64)
        \\   --start-token=<u32>       Token id seeded into the decoder (default 1)
        \\   --max-new-tokens=<u32>    Number of tokens to generate (default 32)
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

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();

    var model_buffers = try repo_model.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer repo_model.unloadBuffers(&model_buffers, allocator);

    if (!args.generate) {
        log.info("non-generate mode not implemented; pass --generate", .{});
        return;
    }

    var compiled = try repo_model.compile(allocator, io, platform, shardings, args.seqlen, &progress);
    defer compiled.deinit();

    var stdout_writer = std.Io.File.stdout().writerStreaming(io, &.{});
    const stdout: *std.Io.Writer = &stdout_writer.interface;

    try runDecode(allocator, io, platform, &compiled, &model_buffers, args.start_token, args.max_new_tokens, stdout);
}

/// Pure-decode loop: seeds with one token, calls the single Model.forward exe without prefill, tokenizer, EOS, for testing
fn runDecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const inference.CompiledModel,
    model_buffers: *model.Buffers,
    start_token: u32,
    max_new_tokens: u32,
    stdout: *std.Io.Writer,
) !void {
    // KV cache is updated in-place via reuseBuffer.
    var kv_cache_buffers = try compiled.params.kv_cache.initBuffer(io, platform, compiled.params.shardings.model);
    defer model.KvCache.deinitBuffer(&kv_cache_buffers);

    // RNG state for the sampler.
    const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
    var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    // Current token buffer: shape {.b=1, .s=1}. Same buffer is both input and output every step
    var current_token: u32 = start_token;
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

    var step: u32 = 0;
    while (step < max_new_tokens) : (step += 1) {
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

        const next_token = try current_token_buffer.getValue(u32, io);
        try stdout.print("{d} ", .{next_token});
        try stdout.flush();
    }

    try stdout.writeAll("\n");
    try stdout.flush();
}
