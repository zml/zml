const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const log = std.log.scoped(.vfs);

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        .{ .scope = .@"zml/io/load", .level = .debug },
    },
};

const Command = enum { cat, tree, ls, cp, stat, realpath, safetensors, @"dma-bench", load };

// -- ls hf://openai/gpt-oss-20b@6cee5e8
// -- ls hf://Qwen/Qwen3-235B-A22B-Instruct-2507
// -- ls hf://meta-llama/Llama-3.1-8B-Instruct@0e9e39f
// -- tree s3://noaa-goes19/ABI-Flood-Day-Shapefiles/2025/08
// -- cat https://iprs.fly.dev
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var it = init.minimal.args.iterate();
    _ = it.next(); // skip program name
    const command: Command = std.meta.stringToEnum(Command, it.next() orelse return error.MissingCommand) orelse return error.CommandInvalid;
    const path = if (command == .@"dma-bench")
        ""
    else
        it.next() orelse return error.MissingPath;

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

    var gcs_vfs: zml.io.VFS.GCS = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer gcs_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.registerBackend("file", vfs_file.backend());
    try vfs.registerBackend("https", vfs_https.backend());
    try vfs.registerBackend("hf", hf_vfs.backend());
    try vfs.registerBackend("s3", s3_vfs.backend());
    try vfs.registerBackend("gs", gcs_vfs.backend());

    const io = vfs.io();

    const buffer = try allocator.alignedAlloc(u8, .fromByteUnits(4 * 1024), 16 * 1024 * 1024);
    defer allocator.free(buffer);

    var stdout_writer = std.Io.File.stdout().writerStreaming(io, buffer);
    defer stdout_writer.flush() catch {};

    switch (command) {
        .cat => {
            var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
            defer file.close(io);

            var reader: std.Io.File.Reader = .initStreaming(file, io, &.{});

            const read = try reader.interface.streamRemaining(&stdout_writer.interface);
            _ = try stdout_writer.interface.write("\n");

            try stdout_writer.interface.print("Wrote {B:.2} to stdout from {s}\n", .{ read, path });
        },
        .ls, .tree => {
            var dir = try std.Io.Dir.openDir(.cwd(), io, path, .{ .iterate = true });
            defer dir.close(io);

            const dir_stat = try dir.stat(io);
            try stdout_writer.interface.print("{s} - {B:.2}\n", .{ path, dir_stat.size });

            var counts: TreeCounts = .{};
            try printTree(io, &stdout_writer.interface, dir, "", if (command == .tree) 10 else 1, &counts);
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
                    const gop = try current.children.getOrPut(current.allocator, part);
                    if (!gop.found_existing) {
                        gop.value_ptr.* = try TensorNode.init(init.arena.allocator(), part);
                    }
                    current = gop.value_ptr.*;
                }

                current.tensor = tensor;
            }

            try stdout_writer.interface.print("{s}\n", .{path});
            try printTensorTree(&stdout_writer.interface, root, "", true, true);
            try stdout_writer.flush();
        },
        .@"dma-bench" => {
            const platform: *zml.Platform = try .auto(allocator, io, .{});
            defer platform.deinit(allocator, io);

            const option_allocator = init.arena.allocator();
            const block_sizes = try envMibList(
                option_allocator,
                init.environ_map,
                "ZML_DMA_BENCH_BLOCK_MIB",
                &zml.io.default_dma_benchmark_block_sizes,
            );
            const parallelism = try envUsizeList(
                option_allocator,
                init.environ_map,
                "ZML_DMA_BENCH_PARALLELISM",
                &zml.io.default_dma_benchmark_parallelism,
            );
            const window_ms = try envUsize(init.environ_map, "ZML_DMA_BENCH_WINDOW_MS", 2);
            const global_window_ms = try envUsize(init.environ_map, "ZML_DMA_BENCH_GLOBAL_WINDOW_MS", 2);
            try platform.benchTransfer(allocator, io, .{
                .block_sizes = block_sizes,
                .parallelism = parallelism,
                .block_parallelism = try envUsize(init.environ_map, "ZML_DMA_BENCH_BLOCK_PARALLELISM", 8),
                .duration_ns = try std.math.mul(u64, window_ms, std.time.ns_per_ms),
                .minimum_transfers_per_device = try envUsize(init.environ_map, "ZML_DMA_BENCH_MIN_TRANSFERS", 32),
                .global_duration_ns = try std.math.mul(u64, global_window_ms, std.time.ns_per_ms),
                .global_minimum_transfers_per_device = try envUsize(init.environ_map, "ZML_DMA_BENCH_GLOBAL_MIN_TRANSFERS", 32),
                .block_selection_tolerance = try envF64(init.environ_map, "ZML_DMA_BENCH_BLOCK_TOLERANCE", 0.08),
                .global_parallelism_selection_tolerance = try envF64(init.environ_map, "ZML_DMA_BENCH_GLOBAL_TOLERANCE", 0.02),
                .global_min_device_retention = try envF64(init.environ_map, "ZML_DMA_BENCH_GLOBAL_MIN_RETENTION", 0.95),
                .global_fairness_floor = try envF64(init.environ_map, "ZML_DMA_BENCH_GLOBAL_FAIRNESS", 0.98),
                .max_mapped_bytes = try envMib(init.environ_map, "ZML_DMA_BENCH_MAX_MAPPED_MIB", 2048),
                .device_numa_nodes = try dmaBenchmarkNumaNodes(option_allocator, init.environ_map),
            });
            const settings = (try platform.transferSettings()) orelse return;
            try stdout_writer.interface.print(
                "dma_settings calibrated={} block_bytes={d} parallelism={d} global_parallelism={?d} max_mapped_bytes={d} retained_mapped_bytes={d} numa_pools={d}\n",
                .{
                    settings.calibrated,
                    settings.block_size,
                    settings.max_in_flight_per_device,
                    settings.global_max_in_flight,
                    settings.max_mapped_bytes,
                    settings.retained_mapped_bytes,
                    settings.numa_pool_count,
                },
            );
            try stdout_writer.flush();
        },
        .load => {
            const ShardingType = enum { replicated, sharded };

            const sharding_type: ShardingType = std.meta.stringToEnum(ShardingType, it.next() orelse "sharded") orelse return error.InvalidShardingKind;

            const platform: *zml.Platform = try .auto(allocator, io, .{});
            defer platform.deinit(allocator, io);

            const load_dma_block_sizes = try envMibList(
                init.arena.allocator(),
                init.environ_map,
                "ZML_DMA_BENCH_BLOCK_MIB",
                &zml.io.default_dma_benchmark_block_sizes,
            );
            try platform.benchTransfer(allocator, io, .{
                .block_sizes = load_dma_block_sizes,
            });

            var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
            defer registry.deinit();

            var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
            defer store.deinit();

            const AllTensorsModel = struct {
                tensors: []zml.Tensor,
            };

            const tensor_count = registry.tensors.count();

            const tensors = try allocator.alloc(zml.Tensor, tensor_count);
            defer allocator.free(tensors);

            var registry_it = registry.iterator();
            var load_count: usize = 0;
            while (registry_it.next()) |entry| : (load_count += 1) {
                tensors[load_count] = switch (sharding_type) {
                    .replicated => store.view().createTensor(entry.key_ptr.*, null, .replicated),
                    .sharded => if (entry.value_ptr.shape.rank() > 0)
                        store.view().createTensor(entry.key_ptr.*, null, .{ ._0 = .model })
                    else
                        store.view().createTensor(entry.key_ptr.*, null, .replicated),
                };
            }

            const model: AllTensorsModel = .{ .tensors = tensors };

            const sharded_sharding: zml.Sharding = try platform.registerSharding(
                "playground_model",
                .mesh(.{ .model = .high_bandwidth }),
            );

            var progress = std.Progress.start(io, .{
                .root_name = "zml.examples.load",
                .disable_printing = true,
            });
            progress.increaseEstimatedTotalItems(load_count);
            defer progress.end();

            try platform.warmupDeviceAllocators();

            const now: std.Io.Timestamp = .now(io, .awake);
            var total_bytes: usize = 0;
            defer {
                const took = now.untilNow(io, .awake);
                const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
                log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
            }

            const load_read_parallelism: zml.io.Parallelism = if (try envOptionalUsize(init.environ_map, "ZML_LOAD_FIXED_READ_PARALLELISM")) |fixed|
                .{ .fixed = fixed }
            else
                .{ .adaptive = .{
                    .initial = try envUsize(init.environ_map, "ZML_LOAD_READ_INITIAL_PARALLELISM", 12),
                    .maximum = try envUsize(init.environ_map, "ZML_LOAD_READ_PARALLELISM", 128),
                } };

            const loaded = try zml.io.load(AllTensorsModel, &model, init.arena.allocator(), io, platform, &store, .{
                .shardings = &.{sharded_sharding},
                .read_parallelism = load_read_parallelism,
                .progress = &progress,
                .total_bytes = &total_bytes,
            });
            defer {
                for (loaded.tensors) |*buffer_| buffer_.deinit();
                init.arena.allocator().free(loaded.tensors);
            }
        },
    }
}

fn dmaBenchmarkNumaNodes(
    allocator: std.mem.Allocator,
    environ_map: *const std.process.Environ.Map,
) ![]const usize {
    if (environ_map.get("ZML_DMA_BENCH_NUMA_NODES") != null) {
        return envUsizeList(
            allocator,
            environ_map,
            "ZML_DMA_BENCH_NUMA_NODES",
            &.{},
        );
    }
    return &.{};
}

fn envUsize(environ_map: *const std.process.Environ.Map, name: []const u8, default: usize) !usize {
    const value = environ_map.get(name) orelse return default;
    return std.fmt.parseInt(usize, value, 10);
}

fn envF64(environ_map: *const std.process.Environ.Map, name: []const u8, default: f64) !f64 {
    const value = environ_map.get(name) orelse return default;
    return std.fmt.parseFloat(f64, value);
}

fn envOptionalUsize(environ_map: *const std.process.Environ.Map, name: []const u8) !?usize {
    const value = environ_map.get(name) orelse return null;
    return try std.fmt.parseInt(usize, value, 10);
}

fn envMib(environ_map: *const std.process.Environ.Map, name: []const u8, default: usize) !usize {
    return std.math.mul(usize, try envUsize(environ_map, name, default), zml.MiB);
}

fn envUsizeList(
    allocator: std.mem.Allocator,
    environ_map: *const std.process.Environ.Map,
    name: []const u8,
    defaults: []const usize,
) ![]const usize {
    const value = environ_map.get(name) orelse return allocator.dupe(usize, defaults);
    var result: std.ArrayListUnmanaged(usize) = .empty;
    var values = std.mem.tokenizeScalar(u8, value, ',');
    while (values.next()) |item| try result.append(allocator, try std.fmt.parseInt(usize, item, 10));
    if (result.items.len == 0) return error.InvalidArgument;
    return result.toOwnedSlice(allocator);
}

fn envMibList(
    allocator: std.mem.Allocator,
    environ_map: *const std.process.Environ.Map,
    name: []const u8,
    default_bytes: []const usize,
) ![]const usize {
    const value = environ_map.get(name) orelse return allocator.dupe(usize, default_bytes);
    var result: std.ArrayListUnmanaged(usize) = .empty;
    var values = std.mem.tokenizeScalar(u8, value, ',');
    while (values.next()) |item| {
        const mib = try std.fmt.parseInt(usize, item, 10);
        try result.append(allocator, try std.math.mul(usize, mib, zml.MiB));
    }
    if (result.items.len == 0) return error.InvalidArgument;
    return result.toOwnedSlice(allocator);
}

const TreeCounts = struct {
    dirs: usize = 0,
    files: usize = 0,
};

fn printTree(io: std.Io, writer: *std.Io.Writer, dir: std.Io.Dir, prefix: []const u8, max_depth: usize, counts: *TreeCounts) !void {
    if (max_depth == 0) return;

    var entries: stdx.BoundedArray(std.Io.Dir.Entry, 1024) = .empty;
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
    children: std.StringArrayHashMapUnmanaged(*TensorNode) = .empty,
    allocator: std.mem.Allocator,
    tensor: ?zml.safetensors.Tensor = null,

    fn init(allocator: std.mem.Allocator, name: []const u8) !*TensorNode {
        const node = try allocator.create(TensorNode);
        node.* = .{
            .name = name,
            .allocator = allocator,
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
    const allocator = node.allocator;

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
