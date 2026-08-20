const std = @import("std");

const zml = @import("zml");
const common = @import("common.zig");
const inference = @import("kimi_k3/inference.zig");
const model = @import("kimi_k3/model.zig");
const session_impl = @import("kimi_k3/session.zig");

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    weights: []const u8,
    tokenizer: []const u8,
    token_count: usize = 4,
    repeats: usize = 2,

    pub const help =
        \\Use kimi_k3_session_tests --weights=<S4-directory> --tokenizer=<tokenizer.json> [options]
        \\
        \\Run the fixed Moonshot prefix through the reusable NVIDIA CUDA session.
        \\
        \\Options:
        \\  --token-count=<1..4>  Prefix tokens to execute (default: 4)
        \\  --repeats=<count>     Reset-and-repeat count (default: 2)
        \\
    ;
};

const official_prefix = [_]u32{ 1, 42, 32000, 160000 };
const official_prefix4_greedy: u32 = 95385;

fn elapsedUs(io: std.Io, started: i96) i96 {
    return @divTrunc(std.Io.Clock.now(.real, io).toNanoseconds() - started, 1000);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    if (args.token_count == 0 or args.token_count > official_prefix.len) return error.InvalidTokenCount;
    if (args.repeats == 0) return error.InvalidRepeatCount;

    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.90 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;

    const repo = try zml.safetensors.resolveModelRepo(io, args.weights);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var loaded_model = try model.LoadedModel.init(
        allocator,
        io,
        repo,
        store.view(),
        .{ .kimi_k3_layer_limit = 4 },
    );
    defer loaded_model.deinit(allocator);
    const shardings: common.Shardings = try .init(platform);
    var progress = std.Progress.start(io, .{ .root_name = "Kimi K3 session gate" });
    defer progress.end();

    const compile_started = std.Io.Clock.now(.real, io).toNanoseconds();
    var compiled: inference.CompiledModel = try loaded_model.compile(
        allocator,
        io,
        platform,
        .vanilla,
        shardings,
        official_prefix.len,
        &progress,
    );
    defer compiled.deinit();
    const compile_us = elapsedUs(io, compile_started);

    var buffers = try loaded_model.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer loaded_model.unloadBuffers(&buffers, allocator);
    var tokenizer = try zml.tokenizer.Tokenizer.fromFile(allocator, io, args.tokenizer);
    defer tokenizer.deinit();
    var session = try session_impl.Session.init(allocator, io, platform, tokenizer, &compiled, &buffers);
    defer session.deinit();

    var first_greedy: ?u32 = null;
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    for (0..args.repeats) |repeat| {
        const started = std.Io.Clock.now(.real, io).toNanoseconds();
        try session.runPrefill(official_prefix[0..args.token_count]);
        const greedy = session.last_generated_token;
        if (first_greedy) |expected| {
            if (greedy != expected) return error.KimiK3SessionResetMismatch;
        } else {
            first_greedy = greedy;
        }
        if (args.token_count == official_prefix.len and greedy != official_prefix4_greedy) {
            return error.KimiK3OfficialGreedyMismatch;
        }
        try stdout_file.interface.print(
            "KIMI_K3_SESSION_PASS repeat={} tokens={} greedy={} compile_us={} session_us={} backend=cuda\n",
            .{ repeat, args.token_count, greedy, compile_us, elapsedUs(io, started) },
        );
        try stdout_file.interface.flush();
    }
    try stdout_file.interface.print(
        "KIMI_K3_SESSION_ALL_PASS reset_deterministic=true official_prefix_checked={}\n",
        .{args.token_count == official_prefix.len},
    );
    try stdout_file.interface.flush();
}
