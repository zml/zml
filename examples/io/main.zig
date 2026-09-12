const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const log = std.log.scoped(.vfs);

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        .{ .scope = .@"zml/io/load", .level = .debug },
        .{ .scope = .@"zml/vfs", .level = .debug },
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
                &zml.io.dma_calibration.default_block_sizes,
            );
            const window_ms = try envUsize(init.environ_map, "ZML_DMA_BENCH_WINDOW_MS", 2);
            var loader = try zml.io.Loader.init(allocator, io, platform, .{
                .dma = .{
                    .block_sizes = block_sizes,
                    .block_parallelism = try envUsize(init.environ_map, "ZML_DMA_BENCH_BLOCK_PARALLELISM", 8),
                    .minimum_duration = .fromMilliseconds(std.math.cast(i64, window_ms) orelse return error.Overflow),
                    .minimum_transfers = try envUsize(init.environ_map, "ZML_DMA_BENCH_MIN_TRANSFERS", 32),
                    .block_selection_tolerance = try envF64(init.environ_map, "ZML_DMA_BENCH_BLOCK_TOLERANCE", 0.08),
                },
            });
            defer loader.deinit();
            const calibration = loader.calibration() orelse return error.DmaBenchmarkUnsupported;
            try stdout_writer.interface.print(
                "dma_benchmark block_bytes={d} parallelism={d}\n",
                .{ calibration.block_size, calibration.max_in_flight_per_device },
            );
            try stdout_writer.flush();
        },
        .load => {
            const ShardingType = enum { replicated, sharded };

            const sharding_type: ShardingType = std.meta.stringToEnum(ShardingType, it.next() orelse "sharded") orelse return error.InvalidShardingKind;

            // A platform setting, not a loader knob: a smaller pool lets a
            // run observe the loader's admission on a device with room.
            const memory_fraction: f32 = if (init.environ_map.get("ZML_GPU_MEMORY_FRACTION")) |text|
                std.fmt.parseFloat(f32, text) catch {
                    log.err("ZML_GPU_MEMORY_FRACTION must be a number", .{});
                    return error.InvalidArgument;
                }
            else
                0.90;
            const platform: *zml.Platform = try .auto(allocator, io, .{
                .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = memory_fraction } } },
            });
            defer platform.deinit(allocator, io);

            const load_dma_block_sizes = try envMibList(
                init.arena.allocator(),
                init.environ_map,
                "ZML_DMA_BENCH_BLOCK_MIB",
                &zml.io.dma_calibration.default_block_sizes,
            );
            var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
            defer registry.deinit();

            var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
            defer store.deinit();

            const sharded_sharding: zml.Sharding = try platform.registerSharding(
                "playground_model",
                .mesh(.{ .model = .high_bandwidth }),
            );

            // Expert-pack instrument: N bindings of W same-shape rank-2 tensors,
            // each loaded through `loadExecute` with a stack executable.
            const pack_options: PackOptions = .{
                .packs = try envUsize(init.environ_map, "ZML_LOAD_PACKS", 0),
                .width = try envUsize(init.environ_map, "ZML_LOAD_PACK_WIDTH", 64),
                .pairs = try envUsize(init.environ_map, "ZML_LOAD_PACK_PAIRS", 0),
                .check = try envUsize(init.environ_map, "ZML_LOAD_PACK_CHECK", 1) != 0,
                .max_elements = try envUsize(init.environ_map, "ZML_LOAD_PACK_MAX_ELEMENTS", std.math.maxInt(i32)),
            };
            if (pack_options.width == 0) return error.InvalidArgument;
            const pack_plan = try planPacks(init.arena.allocator(), allocator, io, platform, &registry, &store, sharded_sharding, pack_options);
            defer for (pack_plan.exes) |*exe| exe.deinit();
            if (pack_options.packs > 0) {
                log.info("pack plan: packs={d} requested={d} width={d} pairs={d} executables={d} bytes={Bi:.2}", .{
                    pack_plan.packs.len,
                    pack_options.packs,
                    pack_options.width,
                    pack_options.pairs,
                    pack_plan.exes.len,
                    pack_plan.bytes,
                });
            }

            const AllTensorsModel = struct {
                tensors: []zml.Tensor,
            };

            const tensor_count = registry.tensors.count() - pack_plan.packed_count;

            const tensors = try allocator.alloc(zml.Tensor, tensor_count);
            defer allocator.free(tensors);

            var registry_it = registry.iterator();
            var registry_index: usize = 0;
            var load_count: usize = 0;
            while (registry_it.next()) |entry| : (registry_index += 1) {
                if (pack_plan.packed_mask[registry_index]) continue;
                tensors[load_count] = switch (sharding_type) {
                    .replicated => store.view().createTensor(entry.key_ptr.*, null, .replicated),
                    .sharded => if (entry.value_ptr.shape.rank() > 0)
                        store.view().createTensor(entry.key_ptr.*, null, .{ ._0 = .model })
                    else
                        store.view().createTensor(entry.key_ptr.*, null, .replicated),
                };
                load_count += 1;
            }

            const model: AllTensorsModel = .{ .tensors = tensors };

            var progress = std.Progress.start(io, .{
                .root_name = "zml.examples.load",
                .estimated_total_items = store.view().count(),
                .disable_printing = true,
            });

            defer progress.end();

            try platform.warmupDeviceAllocators(io);

            const pack_outputs = try allocator.alloc(zml.Buffer, pack_plan.packs.len);
            defer allocator.free(pack_outputs);
            var packs_loaded: usize = 0;
            defer for (pack_outputs[0..packs_loaded]) |*output| output.deinit();

            var check_error: ?anyerror = null;
            // Stamped before the read-back check so the summary excludes it.
            var load_took: ?std.Io.Duration = null;
            {
                const now: std.Io.Timestamp = .now(io, .awake);
                var total_bytes: usize = 0;
                defer {
                    const took = load_took orelse now.untilNow(io, .awake);
                    const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
                    log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
                }

                // Null takes the profile's width (16 locally, 32 remote).
                const load_read_parallelism = try envOptionalUsize(init.environ_map, "ZML_LOAD_READ_PARALLELISM");
                const load_profile = try vfs.loadProfile(path);
                const check_stride = try envUsize(init.environ_map, "ZML_LOAD_CHECK", 0);

                var loaded = try zml.mem.bufferize(init.arena.allocator(), AllTensorsModel, &model);
                errdefer zml.mem.deinitBufferized(init.arena.allocator(), AllTensorsModel, &loaded);
                var loader = try zml.io.Loader.init(init.arena.allocator(), io, platform, .{
                    .dma = .{
                        .block_sizes = load_dma_block_sizes,
                        .block_parallelism = try envUsize(init.environ_map, "ZML_DMA_BENCH_BLOCK_PARALLELISM", 8),
                    },
                    .read_parallelism = load_read_parallelism,
                    .load_profile = load_profile,
                    .direct_io = std.meta.stringToEnum(zml.io.VFS.DirectIo, init.environ_map.get("ZML_DIRECT_IO") orelse "auto") orelse {
                        log.err("ZML_DIRECT_IO must be off, on or auto", .{});
                        return error.InvalidArgument;
                    },
                });
                defer loader.deinit();

                // The loader admits the pack submissions itself; the bulk
                // queues behind them and one await retires everything.
                const packs_per_submission: usize = if (pack_options.pairs != 0) 2 else 1;
                var next_pack: usize = 0;
                while (next_pack < pack_plan.packs.len) {
                    const count = @min(packs_per_submission, pack_plan.packs.len - next_pack);
                    var bindings: [2]zml.io.Loader.Binding = undefined;
                    for (
                        bindings[0..count],
                        pack_plan.packs[next_pack..][0..count],
                        pack_outputs[next_pack..][0..count],
                    ) |*binding, pack, *output| {
                        binding.* = .{
                            .tensor = pack.tensor,
                            .output = output,
                            .exe = &pack_plan.exes[pack.exe_index],
                        };
                    }
                    try loader.loadExecute(&store, bindings[0..count], &progress);
                    next_pack += count;
                }
                try loader.load(AllTensorsModel, &model, &loaded, &store, &.{sharded_sharding}, &progress);
                try loader.awaitAll();
                packs_loaded = pack_plan.packs.len;
                total_bytes = loader.bytesLoaded();
                defer {
                    for (loaded.tensors) |*buffer_| buffer_.deinit();
                    init.arena.allocator().free(loaded.tensors);
                }

                load_took = now.untilNow(io, .awake);
                // Not `try`: the `errdefer` above and the `defer` just
                // registered both release the buffers.
                if (check_stride != 0) {
                    checkLoaded(allocator, io, &store, tensors, loaded.tensors, check_stride) catch |err| {
                        check_error = err;
                    };
                }
            }
            if (check_error) |err| return err;

            if (pack_options.check and pack_plan.packs.len > 0) {
                try checkPacks(allocator, io, &store, pack_plan.packs, pack_outputs);
            }
        },
    }
}

const PackOptions = struct {
    /// Number of pack submissions (0 disables the instrument).
    packs: usize,
    /// Sources per pack.
    width: usize,
    /// Pair two packs per submission.
    pairs: usize,
    /// Read sample packs back and compare with the source bytes.
    check: bool,
    /// Largest pack output in elements. The oneAPI PJRT plugin launches
    /// kernels with int32 ranges, so stacking more than 2^31-1 elements aborts.
    max_elements: usize,
};

const Pack = struct {
    tensor: zml.Tensor,
    exe_index: usize,
    bytes: usize,
};

const PackPlan = struct {
    packs: []const Pack,
    /// One stack executable per distinct source shape.
    exes: []zml.Exe,
    /// Indexed like the registry: true for keys owned by a pack.
    packed_mask: []const bool,
    packed_count: usize,
    bytes: usize,
};

/// Walks the registry in file order (file URI, then offset) and groups rank-2
/// tensors of identical shape and dtype, `width` at a time, into replicated
/// pack bindings. Adjacent tensors in a checkpoint rarely share a shape, so
/// grouping is per shape class; each pack still lists its sources in file
/// order. Stops after `options.packs` packs.
fn planPacks(
    arena: std.mem.Allocator,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    registry: *const zml.safetensors.TensorRegistry,
    store: *zml.io.TensorStore,
    sharding: zml.Sharding,
    options: PackOptions,
) !PackPlan {
    const keys = registry.tensors.keys();
    const entries = registry.tensors.values();
    const packed_mask = try arena.alloc(bool, entries.len);
    @memset(packed_mask, false);

    var exes: std.ArrayListUnmanaged(zml.Exe) = .empty;
    errdefer for (exes.items) |*exe| exe.deinit();
    var packs: std.ArrayListUnmanaged(Pack) = .empty;
    var packed_count: usize = 0;
    var bytes: usize = 0;

    if (options.packs > 0) {
        const order = try arena.alloc(usize, entries.len);
        for (order, 0..) |*slot, index| slot.* = index;
        const FileOrder = struct {
            entries: []const zml.safetensors.Tensor,

            fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                const lhs = ctx.entries[a];
                const rhs = ctx.entries[b];
                return switch (std.mem.order(u8, lhs.file_uri, rhs.file_uri)) {
                    .lt => true,
                    .gt => false,
                    .eq => lhs.offset < rhs.offset,
                };
            }
        };
        std.mem.sort(usize, order, FileOrder{ .entries = entries }, FileOrder.lessThan);

        const Class = struct {
            shape: zml.Shape,
            skipped: bool,
            members: std.ArrayListUnmanaged(usize) = .empty,
            exe_index: ?usize = null,
        };
        var classes: std.ArrayListUnmanaged(Class) = .empty;

        for (order) |index| {
            const entry = entries[index];
            if (entry.shape.rank() != 2) continue;
            const class = for (classes.items) |*class| {
                if (class.shape.eql(entry.shape)) break class;
            } else blk: {
                const elements = try std.math.mul(usize, entry.shape.count(), options.width);
                const skipped = elements > options.max_elements;
                if (skipped) {
                    log.warn("pack plan: skipping shape {f}: width={d} would stack {d} elements, above ZML_LOAD_PACK_MAX_ELEMENTS={d}", .{
                        entry.shape,
                        options.width,
                        elements,
                        options.max_elements,
                    });
                }
                try classes.append(arena, .{ .shape = entry.shape, .skipped = skipped });
                break :blk &classes.items[classes.items.len - 1];
            };
            if (class.skipped) continue;
            try class.members.append(arena, index);
            if (class.members.items.len < options.width) continue;

            const source_keys = try arena.alloc([]const u8, options.width);
            for (source_keys, class.members.items) |*key, member| {
                key.* = keys[member];
                packed_mask[member] = true;
            }
            class.members.clearRetainingCapacity();
            packed_count += options.width;

            const shape = packShape(entry.shape, options.width);
            const tensor = store.view().maybeCreateBinding(source_keys, shape) orelse return error.MissingPackSource;
            if (class.exe_index == null) {
                const inputs = try allocator.alloc(zml.Tensor, options.width);
                defer allocator.free(inputs);
                for (inputs) |*input| input.* = .fromShape(entry.shape);
                try exes.append(arena, try platform.compileFn(allocator, io, stackPack, .{inputs}, .{
                    .shardings = &.{sharding},
                    .program_name = "playground_pack",
                }));
                class.exe_index = exes.items.len - 1;
            }
            try packs.append(arena, .{ .tensor = tensor, .exe_index = class.exe_index.?, .bytes = shape.byteSize() });
            bytes += shape.byteSize();
            if (packs.items.len == options.packs) break;
        }
    }

    return .{
        .packs = packs.items,
        .exes = exes.items,
        .packed_mask = packed_mask,
        .packed_count = packed_count,
        .bytes = bytes,
    };
}

/// Pack output shape: a leading `expert` axis, fully replicated so that the
/// loader's expected output placement equals the executable's.
fn packShape(source: zml.Shape, width: usize) zml.Shape {
    return source
        .insert(0, .{ .expert = width })
        .withTags(.{ .expert, .rows, .cols })
        .withReplicatedPartitioning();
}

fn stackPack(inputs: []const zml.Tensor) zml.Tensor {
    return zml.Tensor.stack(inputs, 0, .expert)
        .withTags(.{ .expert, .rows, .cols })
        .withPartitioning(.{ .expert = .replicated, .rows = .replicated, .cols = .replicated });
}

/// Reads the first, middle and last pack back to host and compares each
/// expert slice with the bytes of its source tensor.
fn checkPacks(
    allocator: std.mem.Allocator,
    io: std.Io,
    store: *const zml.io.TensorStore,
    packs: []const Pack,
    outputs: []const zml.Buffer,
) !void {
    const samples = [_]usize{ 0, packs.len / 2, packs.len - 1 };
    var checked: usize = 0;
    for (samples, 0..) |sample, i| {
        if (std.mem.indexOfScalar(usize, samples[0..i], sample) != null) continue;
        const sources = (store.getSourcesById(packs[sample].tensor.id) orelse return error.NotFound).tensors;
        const slice = try outputs[sample].toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        const packed_bytes = slice.constData();
        const source_bytes: usize = @intCast(sources[0].byteSize());
        if (packed_bytes.len != source_bytes * sources.len) return error.PackContentMismatch;
        const expected = try allocator.alloc(u8, source_bytes);
        defer allocator.free(expected);
        for (sources, 0..) |source, expert| {
            var reader = try source.reader(io, &.{}, .{});
            defer reader.deinit();
            try reader.readPositionalAll(expected, 0);
            const actual = packed_bytes[expert * source_bytes ..][0..source_bytes];
            if (!std.mem.eql(u8, expected, actual)) {
                log.err("pack check: mismatch pack={d} expert={d} source={s}", .{ sample, expert, source.name });
                return error.PackContentMismatch;
            }
        }
        checked += 1;
    }
    log.info("pack check: ok packs_checked={d} of {d}", .{ checked, packs.len });
}

/// `ZML_LOAD_CHECK=n`: reads every n-th loaded tensor and the largest
/// eligible one back to host and compares the bytes with the source, so a
/// buffer that reported ready before every piece landed is caught. A fully
/// replicated buffer is compared on every replica; a partitioned one is
/// assembled with `toSliceAlloc`, which keeps the last replica of a region
/// that several devices hold. Each source file is opened once (an
/// `hf://` open is a HEAD plus a redirect). Skipped: tensors with several
/// sources, and sub-byte tensors partitioned over several devices
/// (`toSliceAlloc` places their shards by element stride).
fn checkLoaded(
    allocator: std.mem.Allocator,
    io: std.Io,
    store: *const zml.io.TensorStore,
    tensors: []const zml.Tensor,
    buffers: []const zml.Buffer,
    stride: usize,
) !void {
    std.debug.assert(tensors.len == buffers.len);
    const Eligibility = struct {
        fn source(store_: *const zml.io.TensorStore, tensor: zml.Tensor, buffer: zml.Buffer) ?*zml.safetensors.Tensor {
            const sources = (store_.getSourcesById(tensor.id) orelse return null).tensors;
            if (sources.len != 1) return null;
            const partitioned = buffer.byteSize() / buffer.numShards() != tensor.byteSize();
            if (partitioned and tensor.dtype().bitSizeOf() < 8) return null;
            return sources[0];
        }
    };
    var largest: ?usize = null;
    for (tensors, buffers, 0..) |tensor, buffer, i| {
        if (Eligibility.source(store, tensor, buffer) == null) continue;
        if (largest == null or tensor.byteSize() > tensors[largest.?].byteSize()) largest = i;
    }
    var files: std.StringHashMapUnmanaged(std.Io.File) = .empty;
    defer {
        var it = files.valueIterator();
        while (it.next()) |file| file.close(io);
        files.deinit(allocator);
    }
    var expected: []u8 = &.{};
    defer allocator.free(expected);
    var replica: []u8 = &.{};
    defer allocator.free(replica);
    const started: std.Io.Timestamp = .now(io, .awake);
    var checked: usize = 0;
    var skipped: usize = 0;
    var checked_bytes: usize = 0;
    for (tensors, buffers, 0..) |tensor, buffer, i| {
        if (i % stride != 0 and i != largest) continue;
        const source = Eligibility.source(store, tensor, buffer) orelse {
            skipped += 1;
            continue;
        };
        const file = files.get(source.file_uri) orelse blk: {
            const opened = try std.Io.Dir.openFile(.cwd(), io, source.file_uri, .{ .mode = .read_only });
            errdefer opened.close(io);
            try files.put(allocator, source.file_uri, opened);
            break :blk opened;
        };
        const source_bytes: usize = @intCast(source.byteSize());
        if (expected.len < source_bytes) {
            allocator.free(expected);
            expected = &.{};
            expected = try allocator.alloc(u8, source_bytes);
        }
        var reader = zml.safetensors.TensorReader.initBorrowedPositional(io, source.*, file);
        defer reader.deinit();
        try reader.readPositionalAll(expected[0..source_bytes], 0);

        if (buffer.byteSize() / buffer.numShards() == source_bytes) {
            // Every shard holds the whole tensor: compare each replica.
            if (replica.len < source_bytes) {
                allocator.free(replica);
                replica = &.{};
                replica = try allocator.alloc(u8, source_bytes);
            }
            var shards = buffer.shards();
            var shard_index: usize = 0;
            while (shards.next()) |shard| : (shard_index += 1) {
                try shard.toHost(io, replica[0..source_bytes]);
                try expectSameBytes(source.name, shard_index, expected[0..source_bytes], replica[0..source_bytes]);
                checked_bytes += source_bytes;
            }
        } else {
            const slice = try buffer.toSliceAlloc(allocator, io);
            defer slice.free(allocator);
            try expectSameBytes(source.name, null, expected[0..source_bytes], slice.constData());
            checked_bytes += source_bytes;
        }
        checked += 1;
    }
    log.info("load check: ok tensors_checked={d} of {d} skipped={d} bytes={Bi:.2} elapsed={f}", .{
        checked,
        tensors.len,
        skipped,
        checked_bytes,
        started.untilNow(io, .awake),
    });
}

fn expectSameBytes(name: []const u8, replica: ?usize, expected: []const u8, actual: []const u8) !void {
    if (actual.len == expected.len and std.mem.eql(u8, expected, actual)) return;
    const first_bad = std.mem.indexOfDiff(u8, expected, actual) orelse @min(expected.len, actual.len);
    log.err("load check: mismatch tensor={s} replica={?d} expected_bytes={d} actual_bytes={d} first_difference={d}", .{
        name,
        replica,
        expected.len,
        actual.len,
        first_bad,
    });
    return error.LoadContentMismatch;
}

fn gibPerSecond(bytes: usize, took: anytype) f64 {
    const seconds = @as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s;
    if (seconds <= 0) return 0;
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(zml.GiB)) / seconds;
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
