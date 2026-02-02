const std = @import("std");

const c = @import("c");
const zml = @import("zml");
const attention = zml.attention.attention;
const stdx = zml.stdx;

const discussion = @import("discussion.zig");
const inference = @import("inference.zig");
const model = @import("model.zig");
const test_runner = @import("test_runner.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.lfm);
const Gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true });

const Args = struct {
    model: []const u8,
    prompt: ?[]const u8 = null,
    seqlen: u32 = 2048,
    backend: ?attention.Backend = null,
    run_tests: bool = false,
    single: bool = false,

    pub const help =
        \\Use lfm --model=<path> [options]
        \\
        \\ Run LFM model inference.
        \\
        \\ Options:
        \\   --model=<path>      Path to the model repository (required)
        \\   --prompt=<string>   Prompt to use for generation (default: none)
        \\   --seqlen=<number>   Sequence length (default: 1024)
        \\   --backend=<text>    Attention backend to use ([vanilla, cuda_fa2, cuda_fa3], default: auto-selection)
        \\   --run-tests         Run tests instead of inference
        \\   --single            Create a single kernel encompassing all the layers.
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .auto(allocator, io, .{ .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.8 } } } });
    defer platform.deinit(allocator);

    log.info("\n{f}", .{platform.fmtVerbose()});

    const args = stdx.flags.parse(init.minimal.args, Args);

    log.info("Resolving model repository..", .{});
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    log.info("Loading configuration...", .{});
    const parsed_config = try loadConfig(model.Config, allocator, io, repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    log.info("Initializing model...", .{});
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var mdl: model.Model = .init(allocator, store.view(), config);
    defer mdl.deinit(allocator);

    const attention_backend = args.backend orelse b: {
        const selected = attention.Backend.auto(platform);
        log.info("Selected backend: {}", .{selected});
        break :b selected;
    };

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    errdefer progress.end();

    var tokenizer_fut = try io.concurrent(loadTokenizer, .{ allocator, io, repo, &progress });
    defer if (tokenizer_fut.cancel(io)) |tokenizer| {
        var tk = tokenizer;
        tk.deinit();
    } else |_| {};

    const params: inference.CompilationOptions = .init(mdl, config, args.seqlen, attention_backend, args.single);

    var compilation_fut = try io.concurrent(inference.Inference.init, .{ allocator, io, platform, mdl, params, args.seqlen, &progress });
    defer if (compilation_fut.cancel(io)) |exe| {
        exe.deinit();
    } else |_| {};

    var load_model_fut = try io.concurrent(model.Model.loadBuffers, .{
        &mdl,
        allocator,
        io,
        platform,
        &store,
        &progress,
    });
    defer if (load_model_fut.cancel(io)) |model_buffers| {
        var mb = model_buffers;
        model.Model.unloadBuffers(&mb, allocator);
    } else |_| {};

    const exe: inference.Inference = try compilation_fut.await(io);
    var model_buffers: zml.Bufferized(model.Model) = try load_model_fut.await(io);
    const tokenizer = try tokenizer_fut.await(io);
    progress.end();

    if (args.run_tests) {
        return test_runner.run(
            allocator,
            io,
            platform,
            config,
            mdl,
            &model_buffers,
            &store,
            params.attention_metadata,
            params.attention_parameters,
        );
    }

    const interactive = args.prompt == null;
    const prompt = if (args.prompt) |p| p else blk: {
        const line = c.linenoise(discussion.DEFAULT_PROMPT) orelse return;
        defer c.linenoiseFree(line);
        break :blk try allocator.dupe(u8, std.mem.sliceTo(line, 0));
    };
    defer allocator.free(prompt);

    var ctx = try discussion.Context.init(allocator, io, platform, &model_buffers, params.cache, params.attention_metadata, exe, tokenizer, config, args.seqlen);
    defer ctx.deinit();

    if (interactive) {
        return ctx.start(prompt);
    } else {
        return ctx.runOnce(prompt);
    }
}

pub fn loadConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(model.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
}

pub fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, progress: *std.Progress.Node) !zml.tokenizer.Tokenizer {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Loading tokenizer...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Loaded tokenizer [{D}]", .{stdx.fmt.fmtDuration(now.untilNow(io, .awake))});
    const bytes = b: {
        const file = try dir.openFile(io, "tokenizer.json", .{});
        defer file.close(io);
        var reader = file.reader(io, &.{});
        break :b try reader.interface.readAlloc(allocator, try file.length(io));
    };
    defer allocator.free(bytes);

    return try .fromBytes(allocator, io, bytes);
}
