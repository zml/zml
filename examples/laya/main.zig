const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const Engine = @import("engine.zig").Engine;
const prompt = @import("prompt.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.laya);

const Args = struct {
    model: []const u8,
    state: ?[]const u8 = null,
    questions: ?[]const u8 = null,
    seqlen: ?u32 = null,
    dtype: Dtype = .f32,
    show_prompt: bool = false,

    const Dtype = enum { f32, f16, bf16 };

    pub const help =
        \\ Use laya --model=<path> --state=<json|@file> --questions=<json|@file> [options]
        \\
        \\ Answer typed questions (choice / score / noul) about a state with a Laya decision model.
        \\ No tokens are generated: every question is one encoder forward pass.
        \\
        \\ Options:
        \\   --model=<path>        Laya checkpoint directory, e.g. hf://convaiinnovations/laya (required)
        \\   --state=<json|@file>  State to decide about: JSON value or text, or @path to a file
        \\   --questions=<json|@file>
        \\                         Questions keyed by id, see examples/laya/README.md
        \\   --seqlen=<number>     Compiled sequence length (default: max_len from rl_agent_config.json)
        \\   --dtype=<f32|f16|bf16>
        \\                         Activation dtype (default: f32)
        \\   --show-prompt         Log the token ids and marker positions of each question
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer http_client.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());

    const io = vfs.io();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    defer repo.close(io);

    const engine = try Engine.init(allocator, io, platform, repo, .{
        .seqlen = args.seqlen,
        .dtype = switch (args.dtype) {
            inline else => |d| @field(zml.DataType, @tagName(d)),
        },
    });
    defer engine.deinit();

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    const state = try readJsonArg(arena.allocator(), init.io, args.state orelse {
        log.err("--state is required", .{});
        return error.MissingState;
    }, .text);
    const questions = try readJsonArg(arena.allocator(), init.io, args.questions orelse {
        log.err("--questions is required", .{});
        return error.MissingQuestions;
    }, .json);

    const result = try engine.decide(arena.allocator(), state, questions);

    if (args.show_prompt) {
        for (result.prompts, result.answers) |p, a| {
            log.info("{s}: {d} tokens, markers {any}\n{any}", .{ a.question.id, p.ids.len, p.markers, p.ids });
        }
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    try prompt.writeResponse(&stdout.interface, result.answers, result.input_tokens, result.latency_ms);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

/// Reads `value`, or the file it points to when it starts with '@'.
/// Values that are not valid JSON are taken as plain text when `fallback` is `.text`.
fn readJsonArg(arena: std.mem.Allocator, io: std.Io, value: []const u8, fallback: enum { text, json }) !std.json.Value {
    const bytes = if (std.mem.startsWith(u8, value, "@"))
        try std.Io.Dir.cwd().readFileAlloc(io, value[1..], arena, .unlimited)
    else
        value;

    return std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{}) catch |err| switch (fallback) {
        .text => .{ .string = bytes },
        .json => {
            log.err("Invalid JSON in {s}: {t}", .{ value, err });
            return err;
        },
    };
}
