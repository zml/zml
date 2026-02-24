const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const log = std.log.scoped(.vfs);

pub const std_options: std.Options = .{
    .log_level = .info,
};

// -- ls hf://openai/gpt-oss-20b@6cee5e8
// -- ls hf://Qwen/Qwen3-235B-A22B-Instruct-2507
// -- ls hf://meta-llama/Llama-3.1-8B-Instruct@0e9e39f
// -- cat https://iprs.fly.dev
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    const Command = enum { cat, ls, cp, stat, realpath, safetensors, load };

    var it = init.minimal.args.iterate();
    _ = it.next(); // skip program name
    const command: Command = std.meta.stringToEnum(Command, it.next() orelse return error.MissingCommand) orelse return error.CommandInvalid;
    const path = it.next() orelse return error.MissingPath;

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };

    try http_client.initDefaultProxies(allocator, init.environ_map);
    defer http_client.deinit();

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var vfs_https: zml.io.VFS.HTTP = try .init(allocator, init.io, &http_client, .https);
    defer vfs_https.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var s3_vfs: zml.io.VFS.S3 = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer s3_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("https", vfs_https.io());
    try vfs.register("hf", hf_vfs.io());
    try vfs.register("s3", s3_vfs.io());

    const io = vfs.io();

    const buffer = try allocator.alignedAlloc(u8, .fromByteUnits(4 * 1024), 16 * 1024 * 1024);
    defer allocator.free(buffer);

    var stdout_writer = std.Io.File.stdout().writer(io, buffer);
    defer stdout_writer.interface.flush() catch {};

    switch (command) {
        .cat => {
            var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
            defer file.close(io);

            var reader: std.Io.File.Reader = .initStreaming(file, io, &.{});

            const read = try reader.interface.streamRemaining(&stdout_writer.interface);
            _ = try stdout_writer.interface.write("\n");

            try stdout_writer.interface.print("Wrote {B:.2} to stdout from {s}\n", .{ read, path });
        },
        .ls => {
            var dir = try std.Io.Dir.openDir(.cwd(), io, path, .{ .iterate = true });
            defer dir.close(io);

            const dir_stat = try dir.stat(io);
            try stdout_writer.interface.print("{s} - {B:.2}\n", .{ path, dir_stat.size });

            var counts: TreeCounts = .{};
            try printTree(io, &stdout_writer.interface, dir, "", 10, &counts);
            try stdout_writer.interface.print("\n{d} directories, {d} files\n", .{ counts.dirs, counts.files });
        },
        .cp => {
            const destination_path = it.next() orelse {
                try stdout_writer.interface.print("Usage: cp <source> <destination>\n", .{});
                return error.InvalidArgument;
            };

            const source = try std.Io.Dir.openFile(.cwd(), io, path, .{});
            defer source.close(io);

            const destination = try std.Io.Dir.createFile(.cwd(), io, destination_path, .{});
            defer destination.close(io);

            var reader: std.Io.File.Reader = .initStreaming(source, io, &.{});
            var writer = destination.writer(io, buffer);

            const read = try reader.interface.streamRemaining(&writer.interface);
            try writer.interface.flush();

            try stdout_writer.interface.print("Copied {B:.2} from {s} to {s}\n", .{ read, path, destination_path });
        },
        .stat => {
            const stat = std.Io.Dir.statFile(.cwd(), io, path, .{}) catch |err| blk: {
                if (err == error.IsDir) {
                    var dir = try std.Io.Dir.openDir(.cwd(), io, path, .{});
                    defer dir.close(io);

                    break :blk try dir.stat(io);
                }
                return err;
            };

            try stdout_writer.interface.print("{s}: {B:.2} ({s})\n", .{ path, stat.size, @tagName(stat.kind) });
        },
        .realpath => {
            var dir = try std.Io.Dir.openDir(.cwd(), io, path, .{});
            defer dir.close(io);

            var real_path_buf: [256]u8 = undefined;
            const len = try dir.realPath(io, &real_path_buf);

            try stdout_writer.interface.print("{s}\n", .{real_path_buf[0..len]});
        },
        .safetensors => {
            var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
            defer registry.deinit();

            const root = try TensorNode.init(init.arena.allocator(), "");

            var registry_it = registry.iterator();
            while (registry_it.next()) |kv| {
                const name = kv.key_ptr.*;
                const tensor = kv.value_ptr.*;

                var current = root;
                var parts = std.mem.tokenizeScalar(u8, name, '.');
                while (parts.next()) |part| {
                    const gop = try current.children.getOrPut(part);
                    if (!gop.found_existing) {
                        gop.value_ptr.* = try TensorNode.init(init.arena.allocator(), part);
                    }
                    current = gop.value_ptr.*;
                }

                current.tensor = tensor;
            }

            try stdout_writer.interface.print("{s}\n", .{path});
            try printTensorTree(&stdout_writer.interface, root, "", true, true);
            try stdout_writer.interface.flush();
        },
        .load => {
            const LoadSharding = enum {
                replicated,
                model_axis0,
            };

            var load_sharding: LoadSharding = .replicated;
            var verify_shards = false;

            while (it.next()) |arg| {
                if (std.mem.eql(u8, arg, "--sharding")) {
                    const value = it.next() orelse {
                        try stdout_writer.interface.print("Missing value for --sharding\nUsage: load <path> [--sharding replicated|model-axis0] [--verify-shards]\n", .{});
                        return error.InvalidArgument;
                    };
                    if (std.mem.eql(u8, value, "replicated")) {
                        load_sharding = .replicated;
                    } else if (std.mem.eql(u8, value, "model-axis0")) {
                        load_sharding = .model_axis0;
                    } else {
                        try stdout_writer.interface.print("Invalid --sharding value: {s}\nUsage: load <path> [--sharding replicated|model-axis0] [--verify-shards]\n", .{value});
                        return error.InvalidArgument;
                    }
                } else if (std.mem.startsWith(u8, arg, "--sharding=")) {
                    const value = arg["--sharding=".len..];
                    if (std.mem.eql(u8, value, "replicated")) {
                        load_sharding = .replicated;
                    } else if (std.mem.eql(u8, value, "model-axis0")) {
                        load_sharding = .model_axis0;
                    } else {
                        try stdout_writer.interface.print("Invalid --sharding value: {s}\nUsage: load <path> [--sharding replicated|model-axis0] [--verify-shards]\n", .{value});
                        return error.InvalidArgument;
                    }
                } else if (std.mem.eql(u8, arg, "--verify-shards")) {
                    verify_shards = true;
                } else {
                    try stdout_writer.interface.print("Unknown load option: {s}\nUsage: load <path> [--sharding replicated|model-axis0] [--verify-shards]\n", .{arg});
                    return error.InvalidArgument;
                }
            }

            const platform: *zml.Platform = try .auto(allocator, io, .{});
            defer platform.deinit(allocator);

            var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
            defer registry.deinit();

            var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
            defer store.deinit();

            const AllTensorsModel = struct {
                tensors: []zml.Tensor,
            };

            const tensor_count = registry.tensors.count();
            if (tensor_count == 0) return error.NoTensorInRegistry;
            const tensors = try allocator.alloc(zml.Tensor, tensor_count);
            defer allocator.free(tensors);
            const tensor_names = try allocator.alloc([]const u8, tensor_count);
            defer allocator.free(tensor_names);

            var registry_it = registry.iterator();
            var i: usize = 0;
            while (registry_it.next()) |entry| : (i += 1) {
                tensor_names[i] = entry.key_ptr.*;
                tensors[i] = switch (load_sharding) {
                    .replicated => store.view().createTensor(entry.key_ptr.*, null, null),
                    .model_axis0 => if (entry.value_ptr.shape.rank() > 0)
                        store.view().createTensor(entry.key_ptr.*, null, .{ ._0 = .model })
                    else
                        store.view().createTensor(entry.key_ptr.*, null, null),
                };
            }

            const model: AllTensorsModel = .{ .tensors = tensors };

            const physical_mesh: zml.sharding.PhysicalMesh = platform.mesh;

            const replicated_sharding = try zml.sharding.replicatedSharding(physical_mesh);
            var sharding_buffer: [2]zml.sharding.Sharding = undefined;
            const shardings: []const zml.sharding.Sharding = switch (load_sharding) {
                .replicated => blk: {
                    sharding_buffer[0] = replicated_sharding;
                    break :blk sharding_buffer[0..1];
                },
                .model_axis0 => blk: {
                    const model_logical_mesh: zml.sharding.LogicalMesh = try .init("playground_model", .{ .model = .high_bandwidth });
                    const model_strategy: zml.sharding.Strategy = try .suggest(model_logical_mesh, physical_mesh);
                    const model_sharding = try zml.sharding.Sharding.initFromStrategy(model_logical_mesh, physical_mesh, model_strategy);
                    sharding_buffer[0] = model_sharding;
                    sharding_buffer[1] = replicated_sharding;
                    break :blk sharding_buffer[0..2];
                },
            };

            var progress = std.Progress.start(io, .{ .root_name = "zml.io.load" });
            progress.increaseEstimatedTotalItems(tensor_count);
            defer progress.end();

            const now: std.Io.Timestamp = .now(io, .awake);
            var total_bytes: usize = 0;
            defer {
                const took = now.untilNow(io, .awake);
                const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
                log.info("Loaded weights [{Bi:.2}, {D}, {Bi:.2}/s]", .{ total_bytes, stdx.fmt.fmtDuration(took), bytes_per_sec });
            }

            var model_buffers = try zml.io.load(AllTensorsModel, &model, allocator, io, platform, .{
                .store = &store,
                .shardings = shardings,
                .parallelism = 32,
                .dma_chunks = 16,
                .dma_chunk_size = 32 * zml.MiB,
                .progress = &progress,
                .total_bytes = &total_bytes,
            });
            defer {
                for (model_buffers.tensors) |*buffer_| {
                    buffer_.deinit();
                }
                allocator.free(model_buffers.tensors);
            }

            if (verify_shards) {
                const verify_now: std.Io.Timestamp = .now(io, .awake);
                const stats = try verifyLoadedShardData(allocator, io, &store, tensor_names, tensors, model_buffers.tensors);
                const took = verify_now.untilNow(io, .awake);
                const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(stats.bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
                log.info("Verified shards [{Bi:.2}, {D}, {Bi:.2}/s, tensors={d}, shards={d}]", .{
                    stats.bytes,
                    stdx.fmt.fmtDuration(took),
                    bytes_per_sec,
                    stats.tensors,
                    stats.shards,
                });
            }

            try stdout_writer.interface.print(
                "Loaded {d} tensors ({B:.2}) with zml.io.load\n",
                .{ tensor_count, total_bytes },
            );
        },
    }
}

const VerifyStats = struct {
    tensors: usize = 0,
    shards: usize = 0,
    bytes: usize = 0,
};

fn verifyLoadedShardData(
    allocator: std.mem.Allocator,
    io: std.Io,
    store: *zml.io.TensorStore,
    tensor_names: []const []const u8,
    tensors: []const zml.Tensor,
    buffers: []zml.Buffer,
) !VerifyStats {
    stdx.debug.assert(tensor_names.len == tensors.len, "Expected same len for tensor names ({}) and tensors ({})", .{ tensor_names.len, tensors.len });
    stdx.debug.assert(tensor_names.len == buffers.len, "Expected same len for tensor names ({}) and buffers ({})", .{ tensor_names.len, buffers.len });

    var reader_buffer: [64 * 1024]u8 = undefined;
    var stats: VerifyStats = .{};
    const tensor_total = tensor_names.len;

    for (buffers, tensor_names, tensors, 0..) |buffer, tensor_name, tensor, tensor_index| {
        var reader = try store.getReaderById(tensor.id, io, &reader_buffer);
        defer reader.deinit();

        const expected_shape = reader.tensor.shape;
        const actual_shape = buffer.shape();
        stdx.debug.assert(expected_shape.eql(actual_shape), "Shape mismatch for tensor {s}: expected {f}, actual {f}", .{ tensor_name, expected_shape, actual_shape });

        const expected_data = try allocator.alloc(u8, expected_shape.byteSize());
        defer allocator.free(expected_data);
        try reader.interface.readSliceAll(expected_data);

        const expected_slice = zml.Slice.init(expected_shape, expected_data);
        const placement = buffer.placement();
        const shard_total = placement.shards.len;
        log.info(
            "[verify {d}/{d}] tensor={s} shape={f} shards={d}",
            .{ tensor_index + 1, tensor_total, tensor_name, expected_shape, shard_total },
        );

        stdx.debug.assert(placement.shards.len == buffer._shards.len, "Placement shard count mismatch for tensor {s}: expected {}, actual {}", .{ tensor_name, placement.shards.len, buffer._shards.len });

        for (placement.shards.constSlice(), 0..) |shard, shard_index| {
            log.info(
                "[verify {d}/{d}] tensor={s} shard={d}/{d} dev={d} size={Bi:.2}",
                .{ tensor_index + 1, tensor_total, tensor_name, shard_index + 1, shard_total, shard.device_id, shard.shape.byteSize() },
            );
            const expected_shard_slice = shard.shardSlice(expected_slice);
            if (!expected_shard_slice.isContiguous()) {
                return error.NonContiguousShardRead;
            }

            const shard_size = shard.shape.byteSize();
            const expected_start = expected_shard_slice.offset_bytes;
            const expected_end = expected_start + shard_size;
            const expected_shard_data = expected_slice.constData()[expected_start..expected_end];

            const actual_shard_data = try allocator.alloc(u8, shard_size);
            defer allocator.free(actual_shard_data);

            const maybe_event = try buffer._shards.get(shard_index).toHostBuffer(buffer._platform.pjrt_api, actual_shard_data);
            if (maybe_event) |event| {
                try event.await(buffer._platform.pjrt_api, io);
            }

            if (findFirstMismatch(expected_shard_data, actual_shard_data)) |mismatch| {
                log.err(
                    "Shard verification failed tensor={s} shard={d} device={d} offset={d} expected=0x{x} actual=0x{x}",
                    .{
                        tensor_name,
                        shard_index,
                        shard.device_id,
                        mismatch,
                        expected_shard_data[mismatch],
                        actual_shard_data[mismatch],
                    },
                );
                return error.ShardDataMismatch;
            }

            stats.shards += 1;
            stats.bytes += shard_size;
        }

        stats.tensors += 1;
    }

    return stats;
}

fn findFirstMismatch(expected: []const u8, actual: []const u8) ?usize {
    if (expected.len != actual.len) {
        return @min(expected.len, actual.len);
    }

    for (expected, actual, 0..) |expected_byte, actual_byte, i| {
        if (expected_byte != actual_byte) return i;
    }
    return null;
}

const TreeCounts = struct {
    dirs: usize = 0,
    files: usize = 0,
};

fn printTree(io: std.Io, writer: *std.Io.Writer, dir: std.Io.Dir, prefix: []const u8, max_depth: usize, counts: *TreeCounts) !void {
    if (max_depth == 0) return;

    var entries: stdx.BoundedArray(std.Io.Dir.Entry, 1024) = .{};
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        entries.appendAssumeCapacity(entry);
    }

    for (entries.constSlice(), 0..) |entry, idx| {
        const is_last = (idx == entries.len - 1);
        const connector = if (is_last) "└── " else "├── ";
        const extension = if (is_last) "    " else "│   ";

        const size: u64 = switch (entry.kind) {
            .file => blk: {
                const stat = try dir.statFile(io, entry.name, .{});
                break :blk stat.size;
            },
            .directory => blk: {
                var sub_dir = dir.openDir(io, entry.name, .{}) catch break :blk 0;
                defer sub_dir.close(io);

                const stat = try sub_dir.stat(io);
                break :blk stat.size;
            },
            else => 0,
        };

        try writer.print("{s}{s}{s} - {B:.2}\n", .{ prefix, connector, entry.name, size });

        if (entry.kind == .directory) {
            counts.dirs += 1;
            var sub_dir = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
            defer sub_dir.close(io);

            var new_prefix_buf: [4096]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&new_prefix_buf, "{s}{s}", .{ prefix, extension }) catch continue;
            try printTree(io, writer, sub_dir, new_prefix, max_depth - 1, counts);
        } else {
            counts.files += 1;
        }
    }
}

const TensorNode = struct {
    name: []const u8,
    children: std.StringArrayHashMap(*TensorNode),
    tensor: ?zml.safetensors.Tensor = null,

    fn init(allocator: std.mem.Allocator, name: []const u8) !*TensorNode {
        const node = try allocator.create(TensorNode);
        node.* = .{
            .name = name,
            .children = std.StringArrayHashMap(*TensorNode).init(allocator),
            .tensor = null,
        };
        return node;
    }
};

fn getSortedChildren(allocator: std.mem.Allocator, node: *TensorNode) ![]*TensorNode {
    const children_nodes = try allocator.alloc(*TensorNode, node.children.count());
    var it = node.children.iterator();
    var idx: usize = 0;
    while (it.next()) |entry| : (idx += 1) {
        children_nodes[idx] = entry.value_ptr.*;
    }

    std.mem.sort(*TensorNode, children_nodes, {}, struct {
        fn lessThan(_: void, a: *TensorNode, b: *TensorNode) bool {
            const a_num = std.fmt.parseInt(usize, a.name, 10) catch null;
            const b_num = std.fmt.parseInt(usize, b.name, 10) catch null;
            if (a_num != null and b_num != null) return a_num.? < b_num.?;
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.lessThan);
    return children_nodes;
}

fn printTensorTree(
    writer: anytype,
    node: *TensorNode,
    prefix: []const u8,
    is_last: bool,
    is_root: bool,
) !void {
    const allocator = node.children.allocator;

    if (is_root) {
        const children_nodes = try getSortedChildren(allocator, node);
        for (children_nodes, 0..) |child, i| {
            try printTensorTree(writer, child, "", i == children_nodes.len - 1, false);
        }
        return;
    }

    var compacted_name = node.name;
    var walk = node;
    while (walk.children.count() == 1 and walk.tensor == null) {
        const next = walk.children.values()[0];
        compacted_name = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ compacted_name, next.name });
        walk = next;
    }

    const connector = if (is_last) "└── " else "├── ";
    try writer.print("{s}{s}{s}", .{ prefix, connector, compacted_name });

    if (walk.tensor) |t| {
        try writer.print(" [shape={f} size={B:.2}]", .{ t.shape, t.byteSize() });
    }
    try writer.print("\n", .{});

    const extension = if (is_last) "    " else "│   ";
    const child_prefix = try std.fmt.allocPrint(allocator, "{s}{s}", .{ prefix, extension });
    const children_nodes = try getSortedChildren(allocator, walk);

    const show_count = 2;
    const skip_threshold = 8;

    for (children_nodes, 0..) |child, i| {
        if (children_nodes.len > skip_threshold) {
            if (i >= show_count and i < children_nodes.len - show_count) {
                if (i == show_count) {
                    try writer.print("{s}├── ...\n", .{child_prefix});
                }
                continue;
            }
        }
        try printTensorTree(writer, child, child_prefix, i == children_nodes.len - 1, false);
    }
}
