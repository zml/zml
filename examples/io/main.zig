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

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    var remotes: zml.io.VFS.Remotes = undefined;
    try remotes.init(allocator, &vfs, init.environ_map);
    defer remotes.deinit();

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
            const calibration = try zml.io.dma_calibration.measure(allocator, io, platform, .{
                .block_sizes = block_sizes,
                .block_parallelism = try envUsize(init.environ_map, "ZML_DMA_BENCH_BLOCK_PARALLELISM", 8),
                .minimum_duration = .fromMilliseconds(std.math.cast(i64, window_ms) orelse return error.Overflow),
                .minimum_transfers = try envUsize(init.environ_map, "ZML_DMA_BENCH_MIN_TRANSFERS", 32),
                .block_selection_tolerance = try envF64(init.environ_map, "ZML_DMA_BENCH_BLOCK_TOLERANCE", 0.08),
            });
            try stdout_writer.interface.print(
                "dma_benchmark block_bytes={d} parallelism={d}\n",
                .{ calibration.block_size, calibration.max_in_flight_per_device },
            );
            try stdout_writer.flush();
        },
        .load => {
            // `tp` shards like tensor parallelism: the input axis of
            // `o_proj` and `down_proj` (column-sharded, strided on every
            // device), the first axis of everything else.
            const MeshType = enum { replicated, sharded, tp };

            const mesh_type: MeshType = std.meta.stringToEnum(MeshType, it.next() orelse "sharded") orelse return error.InvalidMesheKind;

            // A platform setting, not a loader knob: a smaller pool lets a
            // run observe the loader's admission on a device with room.
            // Unset keeps the platform's allocator defaults (VMM on CUDA and
            // ROCm); set, every XLA GPU target gets a preallocated BFC pool of
            // that fraction, whose `bytes_limit` admission measures against.
            var platform_options: zml.platform.CreateOptions = .{};
            if (init.environ_map.get("ZML_GPU_MEMORY_FRACTION")) |text| {
                const memory_fraction = std.fmt.parseFloat(f32, text) catch {
                    log.err("ZML_GPU_MEMORY_FRACTION must be a number", .{});
                    return error.InvalidArgument;
                };
                const pool: zml.platform.CreateOptions.XlaGpu = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = memory_fraction } } };
                platform_options.cuda = pool;
                platform_options.rocm = pool;
                platform_options.oneapi = pool;
            }
            const platform: *zml.Platform = try .auto(allocator, io, platform_options);
            defer platform.deinit(allocator, io);

            const load_dma_block_sizes = try envMibList(
                init.arena.allocator(),
                init.environ_map,
                "ZML_DMA_BENCH_BLOCK_MIB",
                &zml.io.dma_calibration.default_block_sizes,
            );
            const tensor_parallel: *const zml.Mesh = try platform.registerMesh(
                "tp",
                .mesh(.{ .model = .high_bandwidth }),
            );
            const tp_partitioning: zml.Sharding.Partitioning = .init(&.{.sharded(@intCast(tensor_parallel.resolveLogicalAxis("model").?))});
            const column_partitioning: zml.Sharding.Partitioning = .init(&.{ .replicated, .sharded(@intCast(tensor_parallel.resolveLogicalAxis("model").?)) });

            const registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
            var store: zml.io.TensorStore = .init(allocator, registry, .{ .meshes = &.{tensor_parallel} });
            defer store.deinit();

            // Expert-pack instrument: N tensors a stack program computes
            // from W same-shape rank-2 tensors each.
            const pack_options: PackOptions = .{
                .packs = try envUsize(init.environ_map, "ZML_LOAD_PACKS", 0),
                .width = try envUsize(init.environ_map, "ZML_LOAD_PACK_WIDTH", 64),
                .group_bytes = try envOptionalUsize(init.environ_map, "ZML_LOAD_PACK_GROUP_BYTES"),
                .check = try envUsize(init.environ_map, "ZML_LOAD_PACK_CHECK", 1) != 0,
                .max_elements = try envUsize(init.environ_map, "ZML_LOAD_PACK_MAX_ELEMENTS", std.math.maxInt(i32)),
            };
            if (pack_options.width == 0) return error.InvalidArgument;
            const pack_plan = try planPacks(init.arena.allocator(), platform, &store, pack_options);
            if (pack_options.packs > 0) {
                log.info("pack plan: packs={d} requested={d} width={d} bytes={Bi:.2}", .{
                    pack_plan.packs.len,
                    pack_options.packs,
                    pack_options.width,
                    pack_plan.bytes,
                });
            }

            const AllTensorsModel = struct {
                tensors: []zml.Tensor,
                packs: []zml.Tensor,
            };

            const tensor_count = store.registry.tensors.count() - pack_plan.packed_count;

            const tensors = try allocator.alloc(zml.Tensor, tensor_count);
            defer allocator.free(tensors);

            // `ZML_LOAD_MEMORY=host_pinned` binds every tensor into host
            // pinned memory instead of device memory.
            const load_memory = std.meta.stringToEnum(zml.Memory.Kind, init.environ_map.get("ZML_LOAD_MEMORY") orelse "default") orelse {
                log.err("ZML_LOAD_MEMORY must be default, host_pinned, host_unpinned or device", .{});
                return error.InvalidArgument;
            };

            var registry_it = store.registry.iterator();
            var registry_index: usize = 0;
            var load_count: usize = 0;
            while (registry_it.next()) |entry| : (registry_index += 1) {
                if (pack_plan.packed_mask[registry_index]) continue;
                tensors[load_count] = switch (mesh_type) {
                    .replicated => store.view().createReplicatedTensor(entry.key_ptr.*, null),
                    .sharded => if (entry.value_ptr.shape.rank() > 0)
                        store.view().createTensor(entry.key_ptr.*, null, .tp, tp_partitioning)
                    else
                        store.view().createReplicatedTensor(entry.key_ptr.*, null),
                    .tp => if (entry.value_ptr.shape.rank() == 2 and
                        (std.mem.endsWith(u8, entry.key_ptr.*, "o_proj.weight") or std.mem.endsWith(u8, entry.key_ptr.*, "down_proj.weight")))
                        store.view().createTensor(entry.key_ptr.*, null, .tp, column_partitioning)
                    else if (entry.value_ptr.shape.rank() > 0)
                        store.view().createTensor(entry.key_ptr.*, null, .tp, tp_partitioning)
                    else
                        store.view().createReplicatedTensor(entry.key_ptr.*, null),
                };
                if (load_memory != .default) store.view().setMemory(tensors[load_count], load_memory);
                load_count += 1;
            }

            const model: AllTensorsModel = .{ .tensors = tensors, .packs = pack_plan.packs };

            var progress = std.Progress.start(io, .{
                .root_name = "zml.examples.load",
                .disable_printing = true,
            });
            defer progress.end();

            try platform.warmupDeviceAllocators(io);

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
                const check_stride = try envUsize(init.environ_map, "ZML_LOAD_CHECK", 0);

                var loaded = try zml.mem.bufferize(init.arena.allocator(), AllTensorsModel, &model);
                errdefer zml.mem.deinitBufferized(init.arena.allocator(), AllTensorsModel, &loaded);
                var loader = try zml.io.Loader.init(init.arena.allocator(), io, platform, .{
                    .checkpoint = path,
                    .read_parallelism = load_read_parallelism,
                    .dma_block_sizes = load_dma_block_sizes,
                    .direct_io = std.meta.stringToEnum(zml.io.VFS.DirectIo, init.environ_map.get("ZML_DIRECT_IO") orelse "auto") orelse {
                        log.err("ZML_DIRECT_IO must be off, on or auto", .{});
                        return error.InvalidArgument;
                    },
                });
                defer loader.deinit();
                if (pack_options.group_bytes) |bytes| loader.program_group_bytes = bytes;

                // One start: the bulk is published at once while the stack
                // programs compile, and wait publishes the packs in groups
                // the devices admit.
                try loader.start(&model, &loaded, &store, progress);
                try loader.wait();
                // What the loader counts: each tensor once, and a pack's
                // inputs, which are as many bytes as the pack.
                total_bytes = pack_plan.bytes;
                for (tensors) |tensor| total_bytes += tensor.shape().byteSize();
                defer zml.mem.deinitBufferized(init.arena.allocator(), AllTensorsModel, &loaded);

                load_took = now.untilNow(io, .awake);
                // Not `try`: the `errdefer` above and the `defer` just
                // registered both release the buffers.
                if (check_stride != 0) {
                    checkLoaded(allocator, io, &store, tensors, loaded.tensors, check_stride) catch |err| {
                        check_error = err;
                    };
                }
                if (check_error == null and pack_options.check and pack_plan.packs.len > 0) {
                    checkPacks(allocator, io, &store, pack_plan.packs, loaded.packs) catch |err| {
                        check_error = err;
                    };
                }
            }
            if (check_error) |err| return err;
        },
    }
}

const PackOptions = struct {
    /// Number of packs (0 disables the instrument).
    packs: usize,
    /// Sources per pack.
    width: usize,
    /// The loader's program group size; null keeps its default.
    group_bytes: ?usize,
    /// Read sample packs back and compare with the source bytes.
    check: bool,
    /// Largest pack output in elements. The oneAPI PJRT plugin launches
    /// kernels with int32 ranges, so stacking more than 2^31-1 elements aborts.
    max_elements: usize,
};

const PackPlan = struct {
    packs: []zml.Tensor,
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
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    options: PackOptions,
) !PackPlan {
    const keys = store.registry.tensors.keys();
    const entries = store.registry.tensors.values();
    const packed_mask = try arena.alloc(bool, entries.len);
    @memset(packed_mask, false);

    var packs: std.ArrayListUnmanaged(zml.Tensor) = .empty;
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

            const shape = packShape(platform, entry.shape, options.width);
            const tensor = store.view().maybeCreateBinding(source_keys, shape, StackPack{}) orelse return error.MissingPackSource;
            try packs.append(arena, tensor);
            bytes += shape.byteSize();
            if (packs.items.len == options.packs) break;
        }
    }

    return .{
        .packs = packs.items,
        .packed_mask = packed_mask,
        .packed_count = packed_count,
        .bytes = bytes,
    };
}

/// Pack output shape: a leading `expert` axis, fully replicated on the
/// platform's replicated mesh, the mesh `StackPack` names, so that the
/// loader's expected output placement equals the executable's.
fn packShape(platform: *const zml.Platform, source: zml.Shape, width: usize) zml.Shape {
    return source
        .insert(0, .{ .expert = width })
        .withTags(.{ .expert, .rows, .cols })
        .withPartitioning(platform.replicated_mesh, .{ .expert = .replicated, .rows = .replicated, .cols = .replicated });
}

/// The pack program: its sources stacked on a leading `expert` axis.
const StackPack = struct {
    pub fn forward(_: StackPack, _: zml.Shape, inputs: []const zml.Tensor) zml.Tensor {
        return zml.Tensor.stack(inputs, 0, .expert)
            .withTags(.{ .expert, .rows, .cols })
            .withPartitioning(.replicated, .{ .expert = .replicated, .rows = .replicated, .cols = .replicated });
    }
};

/// Reads the first, middle and last pack back to host and compares each
/// expert slice with the bytes of its source tensor.
fn checkPacks(
    allocator: std.mem.Allocator,
    io: std.Io,
    store: *const zml.io.TensorStore,
    packs: []const zml.Tensor,
    outputs: []const zml.Buffer,
) !void {
    const samples = [_]usize{ 0, packs.len / 2, packs.len - 1 };
    var checked: usize = 0;
    for (samples, 0..) |sample, i| {
        if (std.mem.indexOfScalar(usize, samples[0..i], sample) != null) continue;
        const inputs = (store.getSourcesById(packs[sample].id) orelse return error.NotFound).compute.inputs;
        const slice = try outputs[sample].toSliceAlloc(allocator, io);
        defer slice.free(allocator);
        const packed_bytes = slice.constData();
        const source_bytes: usize = @intCast(inputs[0].whole().?.byteSize());
        if (packed_bytes.len != source_bytes * inputs.len) return error.PackContentMismatch;
        const expected = try allocator.alloc(u8, source_bytes);
        defer allocator.free(expected);
        for (inputs, 0..) |input, expert| {
            const source = input.whole().?;
            try source.read(io, expected);
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
/// `hf://` open is a HEAD plus a redirect). Skipped: tensors that are not
/// one whole checkpoint tensor, and sub-byte tensors partitioned over
/// several devices (`toSliceAlloc` places their shards by element stride).
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
        fn source(store_: *const zml.io.TensorStore, tensor: zml.Tensor, buffer: zml.Buffer) ?*const zml.safetensors.Tensor {
            const whole = switch (store_.getSourcesById(tensor.id) orelse return null) {
                .load => |load| load.source.whole() orelse return null,
                .compute => return null,
            };
            const partitioned = buffer.byteSize() / buffer.numShards() != tensor.byteSize();
            if (partitioned and tensor.dtype().bitSizeOf() < 8) return null;
            return whole;
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
        try source.readRange(io, file, expected[0..source_bytes], 0);

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
            .file, .sym_link => blk: {
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
