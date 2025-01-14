const flags = @import("tigerbeetle/flags");
const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");
const log = std.log.scoped(.modernbert);
const Tensor = zml.Tensor;
const modernbert = @import("modernbert.zig");

// set this to false to disable the verbose logging
const show_mlir = true;
pub const std_options = .{
    .log_level = .warn,
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .zml_module, .level = if (show_mlir) .debug else .warn },
        .{ .scope = .modernbert, .level = .info },
    },
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const CliArgs = struct {
        model: []const u8,
        tokenizer: ?[]const u8 = null,
        text: ?[]const u8 = null, // Zig is the [MASK] programming language. Paris is the capital of [MASK].
        create_options: []const u8 = "{}",
    };

    const allocator = std.heap.c_allocator;

    const tmp = try std.fs.openDirAbsolute("/tmp", .{});
    try tmp.makePath("zml/modernbert/cache");

    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/modernbert",
        .sharding_enabled = true,
    };

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);
    const model_file = cli_args.model;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const model_arena = arena_state.allocator();

    const create_opts = try std.json.parseFromSliceLeaky(zml.Platform.CreateOptions, model_arena, cli_args.create_options, .{});
    const platform = context.autoPlatform(create_opts).withCompilationOptions(compilation_options);
    context.printAvailablePlatforms(platform);

    log.info("Model file: {s}", .{model_file});

    var ts = try zml.aio.detectFormatAndOpen(allocator, model_file);
    defer ts.deinit();

    const modern_bert_for_masked_lm = try zml.aio.populateModel(modernbert.ModernBertForMaskedLM, model_arena, ts);
    log.info("ModernBERT: {}", .{modern_bert_for_masked_lm});

    if (cli_args.tokenizer == null) {
        log.err("Model doesn't have an embbedded tokenizer, please provide a path to a tokenizer.", .{});
        @panic("No tokenizer provided");
    }
}
