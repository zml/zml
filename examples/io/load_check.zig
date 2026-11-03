//! Loads every tensor of a checkpoint through `zml.io.next` and checks the
//! bytes the devices hold against the checkpoint's. Tooling, not an
//! example: the harness the new loader's backends are validated with on
//! each host.
//!
//!     bazel run //examples/io:load_check -- <checkpoint> [replicated|sharded|tp]
//!     bazel run //examples/io:load_check -- dma-bench
//!
//! The checkpoint is a model directory, a .safetensors file or a
//! .safetensors.index.json, local or remote (hf://, s3://, gs://, https://,
//! with the credentials and proxies of the environment). Each tensor is
//! placed on the `tp` mesh, all the platform's devices on its `model` axis,
//! by the mode:
//! - `replicated`: whole on every device;
//! - `sharded`, the default: split along its first axis, which the devices
//!   must divide, whatever the model (DeepSeek-V4.x cannot bind on 2 or 4
//!   devices);
//! - `tp`: the input axis of `o_proj` and `down_proj` weights, as tensor
//!   parallelism shards them (column-sharded: every device's bytes are
//!   strided in the checkpoint), the first axis of the others.
//! A scalar is replicated in every mode.
//!
//! `dma-bench` loads nothing: it runs the direct backend's DMA calibration
//! (`zml.io.next.dma_calibration.measure`) on the platform and logs the
//! block it selects. A platform that loads through the buffered backend has
//! no DMA to measure, and CPU selects the smallest block unmeasured. Its
//! knobs, beside the platform's below, are the screen's options, at their
//! defaults when unset:
//! - `ZML_DMA_BENCH_BLOCK_MIB=a,b,...`: the candidate blocks in MiB, each
//!   at most 32 (2, 4, 8, 16 and 32); a load takes a single one (below);
//! - `ZML_DMA_BENCH_BLOCK_PARALLELISM=n`: transfers in flight per device,
//!   at most 32 (8);
//! - `ZML_DMA_BENCH_WINDOW_MS=n`: a screen window's minimum duration (2);
//! - `ZML_DMA_BENCH_MIN_TRANSFERS=n`: the transfers a window waits for on
//!   the measured device, 0 for none (32);
//! - `ZML_DMA_BENCH_BLOCK_TOLERANCE=f`: the smallest block within this
//!   fraction of the peak rate is selected (0.08).
//!
//! The loader logs the bytes it loaded, its time and rate at info, and its
//! stats at debug, on the `zml/io/load` scope; `zml/io` logs the DMA
//! calibration's pick with its measured rate on one device (`dma_bench`);
//! `zml/vfs` logs the remote read totals. This tool enables all three.
//!
//! Knobs, from the environment:
//! - `ZML_LOAD_CHECK=n`: once loaded, read every n-th tensor and the largest
//!   one back to host and compare them with the checkpoint's bytes, every
//!   replica of a tensor each device holds whole. 1 checks every tensor;
//!   unset or 0, none. A difference is `LoadContentMismatch`, logged.
//! - `ZML_LOAD_READ_PARALLELISM=n`: the loader's concurrent source reads
//!   (`Loader.Options.read_parallelism`); unset, the checkpoint's default
//!   width, 16 locally and 32 remotely.
//! - `ZML_DMA_BENCH_BLOCK_MIB=n`: the direct backend's DMA block, in MiB
//!   (`Loader.Options.dma_block_size`); unset, calibration picks one.
//! - `ZML_DIRECT_IO=off|on|auto`: direct I/O for local checkpoint files
//!   (`Loader.Options.direct_io`, `VFS.DirectIo`); `auto` when unset.
//! - `ZML_LOAD_MEMORY=default|host_pinned|host_mapped`: where every tensor
//!   is delivered (`TensorStore.Memory`, `View.setMemory`); `default`, device
//!   memory, when unset. The loader refuses a kind the platform cannot take
//!   (`InvalidBinding`).
//! - `ZML_GPU_MEMORY_FRACTION=f`: a preallocated BFC pool of that fraction
//!   of each device's memory on CUDA, ROCm and oneAPI, instead of the
//!   platform's allocator defaults, for a run that observes the loader's
//!   admission on a device with room.
//! - `ZML_CPU_DEVICE_COUNT=n`: the CPU platform's devices (4 by default).
const std = @import("std");

const zml = @import("zml");

const log = std.log.scoped(.load_check);

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        .{ .scope = .@"zml/io/load", .level = .debug },
        .{ .scope = .@"zml/io", .level = .debug },
        .{ .scope = .@"zml/vfs", .level = .debug },
    },
};

const Mode = enum { replicated, sharded, tp };

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = init.minimal.args.iterate();
    _ = args.next(); // skip program name
    const path = args.next() orelse {
        log.err("usage: load_check <checkpoint> [replicated|sharded|tp], or load_check dma-bench", .{});
        return error.MissingPath;
    };
    if (std.mem.eql(u8, path, "dma-bench")) return dmaBench(init);
    const mode = std.meta.stringToEnum(Mode, args.next() orelse "sharded") orelse {
        log.err("the mode must be replicated, sharded or tp", .{});
        return error.InvalidArgument;
    };
    const check_stride = try envNumber(usize, init.environ_map, "ZML_LOAD_CHECK") orelse 0;
    const direct_io = std.meta.stringToEnum(zml.io.VFS.DirectIo, init.environ_map.get("ZML_DIRECT_IO") orelse "auto") orelse {
        log.err("ZML_DIRECT_IO must be off, on or auto", .{});
        return error.InvalidArgument;
    };
    const dma_block_mib = try envNumber(u32, init.environ_map, "ZML_DMA_BENCH_BLOCK_MIB");
    const memory = std.meta.stringToEnum(zml.io.next.TensorStore.Memory, init.environ_map.get("ZML_LOAD_MEMORY") orelse "default") orelse {
        log.err("ZML_LOAD_MEMORY must be default, host_pinned or host_mapped", .{});
        return error.InvalidArgument;
    };

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    var remotes: zml.io.VFS.Remotes = undefined;
    try remotes.init(allocator, &vfs, init.environ_map);
    defer remotes.deinit();

    const io = vfs.io();

    const platform: *zml.Platform = try .auto(allocator, io, try platformOptions(init.environ_map));
    defer platform.deinit(allocator, io);

    // Returns at once: the loader sets its backend up while the
    // checkpoint's headers are read and its tensors bound.
    var loader: zml.io.next.Loader = try .init(allocator, io, platform, .{
        .checkpoint = path,
        .read_parallelism = try envNumber(usize, init.environ_map, "ZML_LOAD_READ_PARALLELISM"),
        .dma_block_size = if (dma_block_mib) |mib| @as(usize, mib) * zml.MiB else null,
        .direct_io = direct_io,
    });
    defer loader.deinit();

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
    defer registry.deinit();

    const tensor_parallel: *const zml.Mesh = try platform.registerMesh("tp", .mesh(.{ .model = .high_bandwidth }));
    const model_axis: u3 = @intCast(tensor_parallel.resolveLogicalAxis("model").?);
    const rows: zml.Sharding.Partitioning = .init(&.{.sharded(model_axis)});
    const columns: zml.Sharding.Partitioning = .init(&.{ .replicated, .sharded(model_axis) });

    var store: zml.io.next.TensorStore = .init(allocator, &registry, .{ .meshes = &.{tensor_parallel} });
    defer store.deinit();

    const Model = struct { tensors: []zml.Tensor };
    // The checkpoint tensor each one is bound to, at the same index.
    const sources = store.registry.tensors.values();
    const model: Model = .{ .tensors = try allocator.alloc(zml.Tensor, sources.len) };
    defer allocator.free(model.tensors);
    const view = store.view();
    for (model.tensors, store.registry.tensors.keys(), sources) |*tensor, key, source| {
        const rank = source.shape.rank();
        const column_sharded = rank == 2 and
            (std.mem.endsWith(u8, key, "o_proj.weight") or std.mem.endsWith(u8, key, "down_proj.weight"));
        tensor.* = if (rank == 0 or mode == .replicated)
            view.createReplicatedTensor(key, null)
        else if (mode == .tp and column_sharded)
            view.createTensor(key, null, .tp, columns)
        else
            view.createTensor(key, null, .tp, rows);
        if (memory != .default) view.setMemory(tensor.*, memory);
    }

    var progress = std.Progress.start(io, .{ .root_name = "load_check", .disable_printing = true });
    defer progress.end();

    var buffers = try zml.mem.bufferize(allocator, &model);
    defer zml.mem.deinitBufferized(allocator, &buffers);
    try loader.start(&model, &buffers, &store, progress);
    try loader.wait();

    if (check_stride != 0) try checkLoaded(allocator, io, sources, buffers.tensors, check_stride);
}

/// Runs `dma-bench`: the DMA calibration screen on the platform, with the
/// `ZML_DMA_BENCH_*` knobs.
fn dmaBench(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const environ_map = init.environ_map;

    const defaults: zml.io.next.dma_calibration.Options = .{};
    const window_ms = try envNumber(i64, environ_map, "ZML_DMA_BENCH_WINDOW_MS");
    const opts: zml.io.next.dma_calibration.Options = .{
        .block_sizes = try envMibList(init.arena.allocator(), environ_map, "ZML_DMA_BENCH_BLOCK_MIB") orelse defaults.block_sizes,
        .block_parallelism = try envNumber(usize, environ_map, "ZML_DMA_BENCH_BLOCK_PARALLELISM") orelse defaults.block_parallelism,
        .minimum_duration = if (window_ms) |ms| .fromMilliseconds(ms) else defaults.minimum_duration,
        .minimum_transfers = try envNumber(u64, environ_map, "ZML_DMA_BENCH_MIN_TRANSFERS") orelse defaults.minimum_transfers,
        .block_selection_tolerance = try envNumber(f64, environ_map, "ZML_DMA_BENCH_BLOCK_TOLERANCE") orelse defaults.block_selection_tolerance,
    };

    const platform: *zml.Platform = try .auto(allocator, io, try platformOptions(environ_map));
    defer platform.deinit(allocator, io);

    const block_size = try zml.io.next.dma_calibration.measure(allocator, io, platform, opts);
    log.info("dma_benchmark block_bytes={d} parallelism={d}", .{ block_size, opts.block_parallelism });
}

/// The platform settings of the environment, which are not loader knobs.
fn platformOptions(environ_map: *const std.process.Environ.Map) !zml.platform.CreateOptions {
    var options: zml.platform.CreateOptions = .{};
    if (try envNumber(u32, environ_map, "ZML_CPU_DEVICE_COUNT")) |device_count| {
        options.cpu.device_count = device_count;
    }
    // Unset keeps the platform's allocator defaults (VMM on CUDA and ROCm);
    // set, every XLA GPU target gets a preallocated BFC pool of that
    // fraction, whose `bytes_limit` admission measures against.
    if (try envNumber(f32, environ_map, "ZML_GPU_MEMORY_FRACTION")) |memory_fraction| {
        const pool: zml.platform.CreateOptions.XlaGpu = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = memory_fraction } } };
        options.cuda = pool;
        options.rocm = pool;
        options.oneapi = pool;
    }
    return options;
}

/// Reads every `stride`-th of `buffers` and the largest eligible one back to
/// host and compares the bytes with the checkpoint tensor each was bound
/// to, `sources` at the same index, so a buffer that reported ready before
/// every piece landed is caught. A buffer every device holds whole is
/// compared on every replica; a partitioned one is assembled with
/// `toSliceAlloc`, which keeps the last replica of a region that several
/// devices hold. Each source file is opened once (an `hf://` open is a HEAD
/// plus a redirect). Skipped: sub-byte tensors partitioned over several
/// devices (`toSliceAlloc` places their shards by element stride).
fn checkLoaded(
    allocator: std.mem.Allocator,
    io: std.Io,
    sources: []const zml.safetensors.Tensor,
    buffers: []const zml.Buffer,
    stride: usize,
) !void {
    std.debug.assert(sources.len == buffers.len);
    const Eligibility = struct {
        fn checkable(source: zml.safetensors.Tensor, buffer: zml.Buffer) bool {
            const partitioned = buffer.byteSize() / buffer.numShards() != source.byteSize();
            return !partitioned or source.shape.dtype().bitSizeOf() >= 8;
        }
    };
    var largest: ?usize = null;
    for (sources, buffers, 0..) |source, buffer, i| {
        if (!Eligibility.checkable(source, buffer)) continue;
        if (largest == null or source.byteSize() > sources[largest.?].byteSize()) largest = i;
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
    for (sources, buffers, 0..) |source, buffer, i| {
        if (i % stride != 0 and i != largest) continue;
        if (!Eligibility.checkable(source, buffer)) {
            skipped += 1;
            continue;
        }
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
        sources.len,
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

/// The environment variable `name` as a comma-separated list of MiB, in
/// bytes, null when unset. An entry that does not parse is
/// `InvalidArgument`, logged.
fn envMibList(allocator: std.mem.Allocator, environ_map: *const std.process.Environ.Map, name: []const u8) !?[]const usize {
    const text = environ_map.get(name) orelse return null;
    var sizes: std.ArrayList(usize) = .empty;
    errdefer sizes.deinit(allocator);
    var entries = std.mem.tokenizeScalar(u8, text, ',');
    while (entries.next()) |entry| {
        const mib = std.fmt.parseInt(u32, entry, 10) catch {
            log.err("{s} must be MiB counts separated by commas, not '{s}'", .{ name, text });
            return error.InvalidArgument;
        };
        try sizes.append(allocator, @as(usize, mib) * zml.MiB);
    }
    return try sizes.toOwnedSlice(allocator);
}

/// The environment variable `name` parsed as a `T`, null when unset. A value
/// that does not parse is `InvalidArgument`, logged.
fn envNumber(comptime T: type, environ_map: *const std.process.Environ.Map, name: []const u8) !?T {
    const text = environ_map.get(name) orelse return null;
    const value = switch (@typeInfo(T)) {
        .int => std.fmt.parseInt(T, text, 10),
        .float => std.fmt.parseFloat(T, text),
        else => @compileError("envNumber: " ++ @typeName(T)),
    };
    return value catch {
        log.err("{s} must be a number, not '{s}'", .{ name, text });
        return error.InvalidArgument;
    };
}
