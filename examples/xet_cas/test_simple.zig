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

    // List all tensor names
    log.info("Tensor listing:", .{});
    var it = registry.tensors.iterator();
    while (it.next()) |entry| {
        const tensor = entry.value_ptr.*;
        log.info("  {s}  shape={f}  size={d} bytes", .{ tensor.name, tensor.shape, tensor.byteSize() });
    }

    // Select two non-consecutive tensors that share xorb chunks (to test caching and chunk reuse).
    // xorb: af9a1e8432dffcede6d5c00493bff0fb41bb7296059dc70603abd1507fea23fe
    const selected_names = [_][]const u8{
        "model.layers.0.self_attn.k_proj.weight", // [6] 16 MB
        "model.layers.0.self_attn.v_proj.weight", // [9] 16 MB
    };

    log.info("Selected tensors for loading:", .{});
    for (selected_names) |name| {
        const tensor = registry.tensors.get(name) orelse {
            log.err("Tensor not found: {s}", .{name});
            return error.TensorNotFound;
        };
        log.info("  {s}  shape={f}  offset={d}  size={d} bytes", .{
            tensor.name, tensor.shape, tensor.offset, tensor.byteSize(),
        });
    }
}
