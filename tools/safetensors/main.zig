const std = @import("std");

const stdx = @import("stdx");
const VFS = @import("vfs");
const zml = @import("zml");

const log = std.log.scoped(.safetensors_convert);

const Args = struct {
    output: ?[]const u8 = null,
    positional: struct { model: []const u8 },

    pub const help =
        \\Usage: convert [--output=<file>] <model>
        \\
        \\Read safetensors headers and write model.safetensors.full_index.json in the working directory
        \\Example: bazel run //tools/safetensors:convert -- hf://owner/model --output="model.safetensors.full_index.json"
    ;
};

pub fn main(init: std.process.Init) !void {
    const args = stdx.flags.parse(init.minimal.args, Args);
    const allocator = init.gpa;

    const start = std.Io.Clock.awake.now(init.io);
    var client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer client.deinit();
    try client.initDefaultProxies(allocator, init.environ_map);

    const input = args.positional.model;

    var file_vfs: VFS.File = .init(allocator, init.io, .{ .direct_io = false });
    defer file_vfs.deinit();
    var http_vfs: VFS.HTTP = try .init(allocator, init.io, &client, .http);
    defer http_vfs.deinit();
    var https_vfs: VFS.HTTP = try .init(allocator, init.io, &client, .https);
    defer https_vfs.deinit();
    var hf_vfs: ?VFS.HF = if (std.mem.startsWith(u8, input, "hf://"))
        try .auto(allocator, init.io, &client, init.environ_map)
    else
        null;
    defer if (hf_vfs) |*backend| backend.deinit();
    var s3_vfs: ?VFS.S3 = if (std.mem.startsWith(u8, input, "s3://"))
        try .auto(allocator, init.io, &client, init.environ_map)
    else
        null;
    defer if (s3_vfs) |*backend| backend.deinit();
    var gcs_vfs: ?VFS.GCS = if (std.mem.startsWith(u8, input, "gs://"))
        try .auto(allocator, init.io, &client, init.environ_map)
    else
        null;
    defer if (gcs_vfs) |*backend| backend.deinit();

    var vfs: VFS = try .init(allocator, init.io);
    const io = vfs.io();
    defer vfs.deinit();
    try vfs.register("file", file_vfs.io());
    try vfs.register("http", http_vfs.io());
    try vfs.register("https", https_vfs.io());
    if (hf_vfs) |*backend| try vfs.register("hf", backend.io());
    if (s3_vfs) |*backend| try vfs.register("s3", backend.io());
    if (gcs_vfs) |*backend| try vfs.register("gs", backend.io());

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    const repo = try zml.safetensors.resolveModelRepo(io, input);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    const loaded = std.Io.Clock.awake.now(init.io);

    const index_json_file = if (args.output) |output|
        try std.Io.Dir.cwd().createFile(io, output, .{})
    else
        try repo.createFile(io, full_index_path, .{});
    const out_buffer = try allocator.alloc(u8, 4096);
    defer allocator.free(out_buffer);
    var out_writer = index_json_file.writer(io, out_buffer);
    try registry.saveAsFullIndexJson(io, allocator, repo, &out_writer.interface);
    const done = std.Io.Clock.awake.now(init.io);

    log.info("Wrote {s}: {d} tensors; load {f}s, write {f}s, total {f}s", .{
        args.output orelse full_index_path,
        registry.tensors.count(),
        start.durationTo(loaded),
        loaded.durationTo(done),
        start.durationTo(done),
    });
}

const full_index_path = "model.safetensors.full_index.json";
