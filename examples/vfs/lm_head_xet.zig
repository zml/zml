const std = @import("std");

const zml = @import("zml");

const XET_CONCURRENCY: usize = 32;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const CountingSink = struct {
    count: u64 = 0,
    interface: std.Io.Writer,

    fn init(buffer: []u8) CountingSink {
        return .{
            .interface = .{
                .buffer = buffer,
                .vtable = &.{
                    .drain = drain,
                    .flush = flush,
                    .rebase = rebase,
                },
            },
        };
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *CountingSink = @fieldParentPtr("interface", w);
        const written = std.Io.Writer.countSplat(data, splat);
        self.count += @intCast(w.end + written);
        w.end = 0;
        return written;
    }

    fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *CountingSink = @fieldParentPtr("interface", w);
        self.count += w.end;
        w.end = 0;
    }

    fn rebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        if (preserve != 0 or capacity > w.buffer.len) return error.WriteFailed;
        try flush(w);
    }
};

const PassMode = enum {
    naive,
    xet,
};

const PassResult = struct {
    label: []const u8,
    streamed: u64,
    sink_count: u64,
    elapsed: std.Io.Duration,
    throughput: u64,
    xet_used: bool,
};

const dedup_smoke_model = "hf://hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4";
const dedup_smoke_tensor = "model.layers.0.self_attn.q_proj.qweight";

fn durationFromNs(ns: u64) std.Io.Duration {
    return .fromNanoseconds(@intCast(ns));
}

fn rateBytesPerSec(bytes: u64, ns: u64) u64 {
    if (bytes == 0 or ns == 0) return 0;
    const seconds = @as(f64, @floatFromInt(ns)) / std.time.ns_per_s;
    return @intFromFloat(@as(f64, @floatFromInt(bytes)) / seconds);
}

fn printDedupSmokeHint(writer: *std.Io.Writer) !void {
    try writer.print(
        \\
        \\dedup smoke:
        \\  bazel run //examples/vfs:lm_head_xet --config=release --config=native -- {s} {s}
        \\
    , .{ dedup_smoke_model, dedup_smoke_tensor });
}

fn runPass(
    clock_io: std.Io,
    vfs_io: std.Io,
    hf_vfs: *zml.io.VFS.HF,
    tensor: zml.safetensors.Tensor,
    sink_buffer: []u8,
    mode: PassMode,
) !PassResult {
    const label: []const u8 = switch (mode) {
        .naive => "naive resolve",
        .xet => "xet",
    };

    hf_vfs.xet.config.enabled = mode == .xet;
    hf_vfs.xet.stats.reset();

    var sink = CountingSink.init(sink_buffer);
    const started: std.Io.Timestamp = .now(clock_io, .awake);

    var reader = try zml.safetensors.TensorReader.init(vfs_io, tensor, &.{}, .{});
    defer reader.deinit();

    const streamed = try reader.interface.streamRemaining(&sink.interface);
    try sink.interface.flush();
    const elapsed = started.untilNow(clock_io, .awake);
    const stats = hf_vfs.xet.snapshotStats();

    if (streamed != sink.count) return error.ShortRead;

    const throughput: u64 = @intFromFloat(@as(f64, @floatFromInt(streamed)) / (@as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_s));
    return .{
        .label = label,
        .streamed = streamed,
        .sink_count = sink.count,
        .elapsed = elapsed,
        .throughput = throughput,
        .xet_used = stats.xorb_bytes != 0,
    };
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = init.minimal.args.iterate();
    _ = args.next();
    const model_path = args.next() orelse "hf://meta-llama/Llama-3.1-8B";
    const tensor_arg = args.next();
    const tensor_name = tensor_arg orelse "lm_head.weight";

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    defer stdout.interface.flush() catch {};

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    try http_client.initDefaultProxies(allocator, init.environ_map);
    defer http_client.deinit();

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var vfs_https: zml.io.VFS.HTTP = try .init(allocator, init.io, &http_client, .https);
    defer vfs_https.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("https", vfs_https.io());
    try vfs.register("hf", hf_vfs.io());

    const io = vfs.io();

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, model_path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    try hf_vfs.registerTensorStore(&store);

    const tensor = registry.tensors.getPtr(tensor_name) orelse {
        try stdout.interface.print("Tensor not found: {s}\n\nAvailable tensors containing \"head\":\n", .{tensor_name});
        var it = registry.iterator();
        while (it.next()) |entry| {
            if (std.mem.indexOf(u8, entry.key_ptr.*, "head") != null) {
                try stdout.interface.print("  {s}\n", .{entry.key_ptr.*});
            }
        }
        return error.TensorNotFound;
    };

    const sink_buffer_size = 128 * 1024 * 1024;
    const sink_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(4 * 1024), sink_buffer_size);
    defer allocator.free(sink_buffer);

    const naive = try runPass(init.io, io, &hf_vfs, tensor.*, sink_buffer, .naive);
    const xet = try runPass(init.io, io, &hf_vfs, tensor.*, sink_buffer, .xet);
    const stats = hf_vfs.xet.snapshotStats();
    const speedup = @as(f64, @floatFromInt(naive.elapsed.nanoseconds)) / @as(f64, @floatFromInt(xet.elapsed.nanoseconds));

    try stdout.interface.print(
        \\model: {s}
        \\tensor: {s}
        \\file: {s}
        \\offset: {d}
        \\tensor bytes: {B:.2}
        \\writer buffer: {B:.2}
        \\xet concurrency: {d}
        \\
        \\comparison:
        \\  {s}: {f}, {B:.2}/s
        \\  {s}: {f}, {B:.2}/s
        \\  speedup: {d:.2}x
        \\  streamed: {B:.2}
        \\  sink count: {B:.2}
        \\
    , .{
        model_path,
        tensor_name,
        tensor.file_uri,
        tensor.offset,
        tensor.byteSize(),
        sink_buffer_size,
        XET_CONCURRENCY,
        naive.label,
        naive.elapsed,
        naive.throughput,
        xet.label,
        xet.elapsed,
        xet.throughput,
        speedup,
        xet.streamed,
        xet.sink_count,
    });

    if (!xet.xet_used) {
        try stdout.interface.print("\nXet was not used; check that the HF tree exposes xetHash for this file.\n", .{});
        try printDedupSmokeHint(&stdout.interface);
        return error.XetNotUsed;
    }

    const decode_bytes_per_sec = rateBytesPerSec(stats.logical_bytes, stats.decode_ns);
    const xorb_to_logical_ratio: f64 = if (stats.logical_bytes == 0)
        0
    else
        @as(f64, @floatFromInt(stats.xorb_bytes)) / @as(f64, @floatFromInt(stats.logical_bytes));

    try stdout.interface.print(
        \\
        \\xet:
        \\  logical bytes: {B:.2}
        \\  xorb bytes: {B:.2}
        \\  xorb/logical: {d:.2}x
        \\  decode: {f} ({B:.2}/s)
    , .{
        stats.logical_bytes,
        stats.xorb_bytes,
        xorb_to_logical_ratio,
        durationFromNs(stats.decode_ns),
        decode_bytes_per_sec,
    });

    if (tensor_arg == null) try printDedupSmokeHint(&stdout.interface);
}
