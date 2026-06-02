const std = @import("std");
const zml = @import("zml");

const log = std.log.scoped(.repack_qwen_moe);

comptime {
    @setEvalBranchQuota(10_000);
}

pub const std_options: std.Options = .{
    .log_level = .info,
};

const TargetSuffixes = [_][]const u8{
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.down_proj",
};

const Cli = struct {
    src: []const u8,
    dst: []const u8,
    verbose: bool = false,
};

const TensorMeta = struct {
    dtype: []const u8,
    shape: []i64,
    data_offsets: [2]u64,
};

const HeaderEntry = struct {
    name: []const u8,
    meta: TensorMeta,
};

const ParsedShard = struct {
    arena: std.heap.ArenaAllocator,
    entries: std.ArrayList(HeaderEntry),
    metadata_json: ?[]const u8,
    data_start: u64,

    fn deinit(self: *ParsedShard) void {
        self.arena.deinit();
    }
};

const SingleTensor = struct {
    x: zml.Tensor,

    pub fn forward(self: SingleTensor) zml.Tensor {
        return self.x.withTags(.{ .expert, .n, .k }).transpose(.{ .expert, .k, .n });
    }
};

fn usage() void {
    std.debug.print(
        \\Usage:
        \\  repack_qwen3_5_moe_experts --src=<model_dir> --dst=<out_dir> [--verbose]
        \\
        \\Copies a HF safetensors model directory and rewrites only:
        \\  *.mlp.experts.gate_up_proj
        \\  *.mlp.experts.down_proj
        \\
        \\Those tensors are transposed with ZML on the selected platform:
        \\  [expert, N, K] -> [expert, K, N]
        \\
    , .{});
}

fn parseArgs(init: std.process.Init) !Cli {
    var src: ?[]const u8 = null;
    var dst: ?[]const u8 = null;
    var verbose = false;

    var it = init.minimal.args.iterate();
    _ = it.next();
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            std.process.exit(0);
        } else if (std.mem.startsWith(u8, arg, "--src=")) {
            src = arg["--src=".len..];
        } else if (std.mem.startsWith(u8, arg, "--dst=")) {
            dst = arg["--dst=".len..];
        } else if (std.mem.eql(u8, arg, "--verbose")) {
            verbose = true;
        } else {
            log.err("unknown argument: {s}", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }

    return .{
        .src = src orelse return error.MissingSrc,
        .dst = dst orelse return error.MissingDst,
        .verbose = verbose,
    };
}

fn isTarget(name: []const u8) bool {
    for (TargetSuffixes) |suffix| {
        if (std.mem.endsWith(u8, name, suffix)) return true;
    }
    return false;
}

fn dtypeSize(dtype: []const u8) !usize {
    if (std.mem.eql(u8, dtype, "BOOL") or
        std.mem.eql(u8, dtype, "U8") or
        std.mem.eql(u8, dtype, "I8") or
        std.mem.eql(u8, dtype, "F8_E5M2") or
        std.mem.eql(u8, dtype, "F8_E4M3"))
        return 1;
    if (std.mem.eql(u8, dtype, "I16") or
        std.mem.eql(u8, dtype, "U16") or
        std.mem.eql(u8, dtype, "F16") or
        std.mem.eql(u8, dtype, "BF16"))
        return 2;
    if (std.mem.eql(u8, dtype, "I32") or
        std.mem.eql(u8, dtype, "U32") or
        std.mem.eql(u8, dtype, "F32"))
        return 4;
    if (std.mem.eql(u8, dtype, "F64") or
        std.mem.eql(u8, dtype, "I64") or
        std.mem.eql(u8, dtype, "U64"))
        return 8;
    return error.UnsupportedDType;
}

fn zmlDType(dtype: []const u8) !zml.DataType {
    if (std.mem.eql(u8, dtype, "BF16")) return .bf16;
    if (std.mem.eql(u8, dtype, "F16")) return .f16;
    if (std.mem.eql(u8, dtype, "F32")) return .f32;
    if (std.mem.eql(u8, dtype, "I32")) return .i32;
    if (std.mem.eql(u8, dtype, "U32")) return .u32;
    return error.UnsupportedDTypeForZmlTranspose;
}

fn tensorBytes(meta: TensorMeta) !u64 {
    var n: u64 = @intCast(try dtypeSize(meta.dtype));
    for (meta.shape) |d| n *= @intCast(d);
    return n;
}

fn readShardHeader(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !ParsedShard {
    var file = try std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only });
    defer file.close(io);

    var len_buf: [8]u8 = undefined;
    _ = try file.readPositionalAll(io, &len_buf, 0);
    const header_len = std.mem.readInt(u64, &len_buf, .little);
    const data_start = 8 + header_len;

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const aa = arena.allocator();

    const header_bytes = try aa.alloc(u8, @intCast(header_len));
    _ = try file.readPositionalAll(io, header_bytes, 8);

    const root = try std.json.parseFromSliceLeaky(std.json.Value, aa, header_bytes, .{});
    var entries: std.ArrayList(HeaderEntry) = .empty;

    const metadata_json: ?[]const u8 = null;
    var it = root.object.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        const val = kv.value_ptr.*;
        if (std.mem.eql(u8, name, "__metadata__")) {
            continue;
        }

        const dtype = val.object.get("dtype").?.string;
        const shape_json = val.object.get("shape").?.array;
        const shape = try aa.alloc(i64, shape_json.items.len);
        for (shape_json.items, 0..) |item, i| shape[i] = item.integer;

        const offsets = val.object.get("data_offsets").?.array;
        try entries.append(aa, .{
            .name = try aa.dupe(u8, name),
            .meta = .{
                .dtype = try aa.dupe(u8, dtype),
                .shape = shape,
                .data_offsets = .{
                    @intCast(offsets.items[0].integer),
                    @intCast(offsets.items[1].integer),
                },
            },
        });
    }

    return .{
        .arena = arena,
        .entries = entries,
        .metadata_json = metadata_json,
        .data_start = data_start,
    };
}

fn appendFmt(out: *std.ArrayList(u8), allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    const s = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(s);
    try out.appendSlice(allocator, s);
}

fn writeJsonString(out: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try out.append(allocator, '"');
    for (s) |c| switch (c) {
        '"' => try out.appendSlice(allocator, "\\\""),
        '\\' => try out.appendSlice(allocator, "\\\\"),
        '\n' => try out.appendSlice(allocator, "\\n"),
        '\r' => try out.appendSlice(allocator, "\\r"),
        '\t' => try out.appendSlice(allocator, "\\t"),
        else => try out.append(allocator, c),
    };
    try out.append(allocator, '"');
}

fn buildHeader(allocator: std.mem.Allocator, shard: *ParsedShard) ![]u8 {
    var offsets = try allocator.alloc([2]u64, shard.entries.items.len);
    defer allocator.free(offsets);

    var cursor: u64 = 0;
    for (shard.entries.items, 0..) |entry, i| {
        const size = try tensorBytes(entry.meta);
        offsets[i] = .{ cursor, cursor + size };
        cursor += size;
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, '{');
    var first = true;
    if (shard.metadata_json) |metadata| {
        try writeJsonString(&out, allocator, "__metadata__");
        try out.append(allocator, ':');
        try out.appendSlice(allocator, metadata);
        first = false;
    }
    for (shard.entries.items, 0..) |entry, i| {
        if (!first) try out.append(allocator, ',');
        first = false;
        try writeJsonString(&out, allocator, entry.name);
        try out.appendSlice(allocator, ":{\"dtype\":");
        try writeJsonString(&out, allocator, entry.meta.dtype);
        try out.appendSlice(allocator, ",\"shape\":[");
        for (entry.meta.shape, 0..) |d, j| {
            if (j != 0) try out.append(allocator, ',');
            try appendFmt(&out, allocator, "{d}", .{d});
        }
        try out.appendSlice(allocator, "],\"data_offsets\":[");
        try appendFmt(&out, allocator, "{d},{d}", .{ offsets[i][0], offsets[i][1] });
        try out.appendSlice(allocator, "]}");
    }
    try out.append(allocator, '}');

    while (out.items.len % 8 != 0) try out.append(allocator, ' ');
    return out.toOwnedSlice(allocator);
}

fn copyRange(io: std.Io, src: *std.Io.File, writer: *std.Io.Writer, offset: u64, size: u64, buf: []u8) !void {
    var remaining = size;
    var pos = offset;
    while (remaining > 0) {
        const want: usize = @intCast(@min(remaining, buf.len));
        const got = try src.readPositional(io, &.{buf[0..want]}, pos);
        if (got == 0) return error.UnexpectedEof;
        try writer.writeAll(buf[0..got]);
        pos += got;
        remaining -= got;
    }
}

fn copyFile(io: std.Io, src_path: []const u8, dst_path: []const u8, buf: []u8) !void {
    var src = try std.Io.Dir.openFile(.cwd(), io, src_path, .{ .mode = .read_only });
    defer src.close(io);
    var dst = try std.Io.Dir.createFile(.cwd(), io, dst_path, .{});
    defer dst.close(io);
    var pos: u64 = 0;
    var out_pos: u64 = 0;
    while (true) {
        const got = try src.readPositional(io, &.{buf}, pos);
        if (got == 0) break;
        try dst.writePositionalAll(io, buf[0..got], out_pos);
        pos += got;
        out_pos += got;
    }
}

fn transposeOnTpu(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    registry: *zml.safetensors.TensorRegistry,
    tensor_name: []const u8,
) !zml.Slice {
    comptime {
        @setEvalBranchQuota(10_000);
    }

    var store: zml.io.TensorStore = .fromRegistry(allocator, registry);
    defer store.deinit();

    const tensor = store.view().createTensor(tensor_name, .{ .expert, .n, .k }, .replicated);
    const model: SingleTensor = .{ .x = tensor };
    const out_shape = tensor.shape().transpose(.{ .expert, .k, .n });

    var exe = try platform.compile(allocator, io, model, .forward, .{}, .{
        .shardings = &.{platform.replicated_sharding},
    });
    defer exe.deinit();

    var loaded = try zml.io.load(SingleTensor, &model, allocator, io, platform, &store, .{
        .parallelism = 1,
        .shardings = &.{platform.replicated_sharding},
        .dma_chunks = 8,
        .dma_chunk_size = 128 * zml.MiB,
    });
    defer loaded.x.deinit();

    var args = try zml.exe.Exe.Arguments.init(allocator, &.{tensor.shape()}, &.{platform.replicated_sharding}, platform.devices.len);
    defer args.deinit(allocator);
    args.set(.{loaded.x});

    var results = try zml.exe.Exe.Results.init(allocator, &.{out_shape}, &.{platform.replicated_sharding}, platform, platform.devices.len);
    defer results.deinit(allocator);

    exe.callOpts(io, args, &results, .{ .wait = true });
    var out_buf = results.get(zml.Buffer);
    defer out_buf.deinit();

    return try out_buf.toSliceAlloc(allocator, io);
}

fn repackShard(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    registry: *zml.safetensors.TensorRegistry,
    src_path: []const u8,
    dst_path: []const u8,
    verbose: bool,
) !usize {
    var shard = try readShardHeader(allocator, io, src_path);
    defer shard.deinit();

    var target_count: usize = 0;
    for (shard.entries.items) |*entry| {
        if (isTarget(entry.name)) {
            if (entry.meta.shape.len != 3) return error.ExpectedRank3;
            const old = entry.meta.shape;
            entry.meta.shape = try shard.arena.allocator().dupe(i64, &.{ old[0], old[2], old[1] });
            target_count += 1;
        }
    }
    if (target_count == 0) {
        const copy_buf = try allocator.alloc(u8, 64 * zml.MiB);
        defer allocator.free(copy_buf);
        try copyFile(io, src_path, dst_path, copy_buf);
        return 0;
    }

    const header = try buildHeader(allocator, &shard);
    defer allocator.free(header);

    var src = try std.Io.Dir.openFile(.cwd(), io, src_path, .{ .mode = .read_only });
    defer src.close(io);
    var dst = try std.Io.Dir.createFile(.cwd(), io, dst_path, .{});
    defer dst.close(io);
    const write_buf = try allocator.alloc(u8, 1024 * 1024);
    defer allocator.free(write_buf);
    var writer = dst.writer(io, write_buf);
    defer writer.interface.flush() catch {};

    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try writer.interface.writeAll(&len_buf);
    try writer.interface.writeAll(header);

    const copy_buf = try allocator.alloc(u8, 64 * zml.MiB);
    defer allocator.free(copy_buf);

    for (shard.entries.items) |entry| {
        const original_shape = registry.tensors.get(entry.name).?.shape;
        if (isTarget(entry.name)) {
            if (verbose) log.info("transpose on TPU: {s} {f}", .{ entry.name, original_shape });
            var out = try transposeOnTpu(allocator, io, platform, registry, entry.name);
            defer out.free(allocator);
            try writer.interface.writeAll(out.constData());
        } else {
            const meta = registry.tensors.get(entry.name).?;
            try copyRange(io, &src, &writer.interface, meta.offset, meta.byteSize(), copy_buf);
        }
    }

    return target_count;
}

fn pathJoin(allocator: std.mem.Allocator, a: []const u8, b: []const u8) ![]u8 {
    return try std.fs.path.join(allocator, &.{ a, b });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const cli = try parseArgs(init);

    try std.Io.Dir.cwd().createDirPath(io, cli.dst);

    var repo = try zml.safetensors.resolveModelRepo(io, cli.src);
    defer repo.close(io);
    var registry = try zml.safetensors.TensorRegistry.fromRepo(allocator, io, repo);
    defer registry.deinit();

    var platform = try zml.Platform.auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    var src_dir = try std.Io.Dir.openDir(.cwd(), io, cli.src, .{ .iterate = true });
    defer src_dir.close(io);
    var it = src_dir.iterate();
    var rewritten: usize = 0;
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        const src_path = try pathJoin(allocator, cli.src, entry.name);
        defer allocator.free(src_path);
        const dst_path = try pathJoin(allocator, cli.dst, entry.name);
        defer allocator.free(dst_path);

        if (std.mem.endsWith(u8, entry.name, ".safetensors")) {
            rewritten += try repackShard(allocator, io, platform, &registry, src_path, dst_path, cli.verbose);
        } else {
            const copy_buf = try allocator.alloc(u8, 64 * zml.MiB);
            defer allocator.free(copy_buf);
            try copyFile(io, src_path, dst_path, copy_buf);
        }
    }

    log.info("done: copied {s} -> {s}; transposed {d} tensors", .{ cli.src, cli.dst, rewritten });
}
