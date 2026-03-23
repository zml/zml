const std = @import("std");

const c = @import("c");
const zml = @import("zml");
const stdx = zml.stdx;

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
    backend: ?zml.attention.attention.Backend = null,
    single: bool = false,

    pub const help =
        \\Use llm --model=<path> [options]
        \\
        \\ Run text generation with a model selected from `config.json.model_type`.
        \\
        \\ Options:
        \\   --model=<path>      Path to the model repository (required)
        \\   --prompt=<string>   Prompt to use for generation (default: none)
        \\   --seqlen=<number>   Sequence length (default: 2048)
        \\   --backend=<text>    Attention backend to use ([vanilla, cuda_fa2, cuda_fa3], default: auto-selection)
        \\   --single            Create a single kernel encompassing all the layers when supported
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    log.info("\n{f}", .{platform.fmtVerbose()});

    log.info("Resolving model repository..", .{});
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    const model_type = try models.detectModelType(allocator, io, repo);
    log.info("Detected model type: {}", .{model_type});

    log.info("Initializing model..", .{});
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var model = try models.Repository.init(model_type, allocator, io, repo, store.view());
    defer model.deinit(allocator);

    const backend = args.backend orelse b: {
        const selected = zml.attention.attention.Backend.auto(platform);
        log.info("Selected backend: {}", .{selected});
        break :b selected;
    };

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    errdefer progress.end();

    const tokenizer = try loadTokenizer(allocator, io, repo, &progress);
    defer {
        var tk = tokenizer;
        tk.deinit();
    }

    const tp_mesh: zml.sharding.LogicalMesh = try .init("tp_mesh", .{ .model = .high_bandwidth });
    const tp_strategy: zml.sharding.Strategy = try .suggest(tp_mesh, platform.physical_mesh);
    const shardings: models.Shardings = .{
        .replicated = try zml.sharding.replicatedSharding(platform),
        .model = try .initFromStrategy(platform, tp_mesh, tp_strategy),
    };

    var model_buffers = try model.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer model_buffers.deinit(allocator);

    var session = try model.initSession(
        allocator,
        io,
        platform,
        &model_buffers,
        tokenizer,
        args.seqlen,
        backend,
        args.single,
        &progress,
        shardings,
    );
    defer session.deinit();

    progress.end();

    try printLogo(io);

    const interactive = args.prompt == null;
    const prompt = if (args.prompt) |p| try allocator.dupe(u8, p) else blk: {
        chat.initHistory();
        const line = c.linenoise(chat.prompt_prefix) orelse return;
        defer c.linenoiseFree(line);
        chat.rememberPrompt(line);
        break :blk try allocator.dupe(u8, std.mem.sliceTo(line, 0));
    };
    defer allocator.free(prompt);

    var llm_chat = try chat.Chat.init(allocator, io, tokenizer, &model, &session);
    defer llm_chat.deinit();

    if (interactive) {
        try llm_chat.runInteractive(prompt);
    } else {
        try llm_chat.runOnce(prompt);
    }
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, progress: *std.Progress.Node) !zml.tokenizer.Tokenizer {
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

pub fn printLogo(io: std.Io) !void {
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
