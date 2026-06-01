const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.test_simple);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,

    pub const help =
        \\Use test_simple --model=<path>
        \\
        \\ Open ZML machinery for a HuggingFace model.
        \\
        \\ Options:
        \\   --model=<path>   Path or URI to the model repository (e.g. hf://meta-llama/Llama-3.3-70B-Instruct)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    //
    // Virtual File Systems
    //
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

    //
    // Platform
    //
    log.info("Initializing platform..", .{});
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("\n{f}", .{platform.fmtVerbose()});

    //
    // Model repository and tensor store
    //
    log.info("Resolving model repository: {s}", .{args.model});
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    log.info("Loading tensor registry..", .{});
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    log.info("Done. Tensor store ready with {} tensors.", .{registry.tensors.count()});
    log.info("Device buffers are only allocated when explicitly requested via `loadBuffers()`.", .{});
}
