const std = @import("std");

const c = @import("c");
const zml = @import("zml");
const stdx = zml.stdx;
const tracer = zml.tracer;

const chat = @import("chat.zig");
const models = @import("models.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.llm);
const Args = struct {
    model: []const u8,
    prompt: ?[]const u8 = null,
    seqlen: u32 = 2048,
    topk: u32 = 4,
    backend: ?zml.attention.attention.Backend = null,
    single: bool = false,

    pub const help =
        \\ Use llm --model=<path> [options]
        \\
        \\ Run text generation with a model selected from `model_type` in the `config.json`.
        \\
        \\ Options:
        \\   --model=<path>      Path to the model repository (required)
        \\   --prompt=<string>   Prompt to use for generation (default: none)
        \\   --seqlen=<number>   Sequence length (default: 2048)
        \\   --topk=<number>     Top-k sampling cutoff (default: 4)
        \\   --backend=<text>    Attention backend to use ([vanilla, cuda_fa2, cuda_fa3], default: auto-selection)
        \\   --single            Create a single kernel encompassing all the layers when supported
        \\                       (only used by LFM2 which uses multiple kernels by default)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // `bazel run` executes binaries from Bazel's runfiles tree by default.
    // If available, switch back to the shell's original working directory.
    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);

    //
    // Virtual File Systems
    //
    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer http_client.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var s3_vfs: zml.io.VFS.S3 = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer s3_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());
    try vfs.register("s3", s3_vfs.io());

    const io = vfs.io();

    //
    // Platform and Backend Selection
    //
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    log.info("\n{f}", .{platform.fmtVerbose()});

    var profiler = try platform.profiler(allocator, io, .defaults);
    defer profiler.deinit();

    try profiler.start();
    defer {
        if ((profiler.stop() catch unreachable)) |profile| {
            log.info("Profile dumped: {s} and {s}", .{ profile.protobuf_path, profile.perfetto_path });
        }
    }

    const backend = args.backend orelse b: {
        const selected = zml.attention.attention.Backend.auto(platform);
        log.info("Selected backend: {}", .{selected});
        break :b selected;
    };

    //
    // Model initialization
    //
    log.info("Resolving model repository..", .{});
    const repo = b: {
        var trace = tracer.scope("llm.resolve_model_repo");
        defer trace.end();
        break :b try zml.safetensors.resolveModelRepo(io, args.model);
    };

    log.info("Initializing model..", .{});
    var registry: zml.safetensors.TensorRegistry = b: {
        var trace = tracer.scope("llm.init_registry");
        defer trace.end();
        break :b try .fromRepo(allocator, io, repo);
    };
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const generation: models.GenerationOptions = .{
        .sampling_strategy = .{
            .topk = args.topk,
        },
    };

    var model = try models.LoadedModel.load(allocator, io, repo, store.view(), generation);
    defer model.deinit(allocator);

    // Defines how the model's tensors are sharded across the available devices.
    const shardings: models.Shardings = try .init(platform);

    //
    // Load the model and compile it
    //
    var progress = std.Progress.start(io, .{ .root_name = args.model });
    errdefer progress.end();

    var tokenizer = try loadTokenizer(allocator, io, repo, &progress);
    defer tokenizer.deinit();

    var model_buffers = b: {
        var trace = tracer.scope("llm.load_buffers");
        defer trace.end();
        break :b try models.LoadedModel.loadBuffers(&model, allocator, io, platform, &store, &progress, shardings);
    };
    defer model.unloadBuffers(&model_buffers, allocator);

    var compiled_model = b: {
        var trace = try tracer.scopeWith(allocator, "llm.compile", .{
            .seqlen = args.seqlen,
            .interactive = args.prompt == null,
        });
        defer trace.end();
        break :b try models.LoadedModel.compile(&model, allocator, io, platform, backend, shardings, args.seqlen, &progress);
    };
    defer compiled_model.deinit();

    progress.end();

    //
    // We're ready to go! Let's start a chat...
    //
    try printZmlLogo(io);

    const interactive = args.prompt == null;
    const prompt = if (args.prompt) |prompt| b: {
        break :b try allocator.dupe(u8, prompt);
    } else b: {
        chat.initHistory();
        const line = c.linenoise(chat.prompt_prefix) orelse return;
        defer c.linenoiseFree(line);
        chat.rememberPrompt(line);
        break :b try allocator.dupe(u8, std.mem.sliceTo(line, 0));
    };
    defer allocator.free(prompt);

    var llm_chat = try chat.Chat.init(
        allocator,
        io,
        platform,
        tokenizer,
        &compiled_model,
        &model_buffers,
    );
    defer llm_chat.deinit();

    if (interactive) {
        var trace = tracer.scope("llm.chat.interactive");
        defer trace.end();
        try llm_chat.runInteractive(prompt);
    } else {
        var trace = tracer.scope("llm.chat.once");
        defer trace.end();
        try llm_chat.runOnce(prompt);
    }
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, progress: *std.Progress.Node) !zml.tokenizer.Tokenizer {
    var trace = tracer.scope("llm.load_tokenizer");
    defer trace.end();

    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Loading tokenizer...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Loaded tokenizer [{f}]", .{now.untilNow(io, .awake)});
    const bytes = b: {
        const file = try dir.openFile(io, "tokenizer.json", .{});
        defer file.close(io);
        var reader = file.reader(io, &.{});
        break :b try reader.interface.readAlloc(allocator, try file.length(io));
    };
    defer allocator.free(bytes);

    return try .fromBytes(allocator, io, bytes);
}

pub fn printZmlLogo(io: std.Io) !void {
    const LOGO =
        \\
        \\
        \\ ███████╗███╗   ███╗██╗
        \\ ╚══███╔╝████╗ ████║██║
        \\   ███╔╝ ██╔████╔██║██║
        \\  ███╔╝  ██║╚██╔╝██║██║  .ai
        \\ ███████╗██║ ╚═╝ ██║███████╗
        \\ ╚══════╝╚═╝     ╚═╝╚══════╝
        \\
        \\
        \\
    ;
    var writer = std.Io.File.stdout().writer(io, &.{});
    try writer.interface.writeAll(LOGO);
    try writer.interface.flush();
}
