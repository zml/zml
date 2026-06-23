const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

const log = std.log.scoped(.gemma4_transpose);
const SafetensorsHeaderBytes = 8;

const CliArgs = struct {
    pub const help =
        \\Convert a Gemma 4 MoE checkpoint folder into an otherwise identical
        \\folder where MoE expert gate_up_proj/down_proj tensors have their
        \\last two dimensions transposed by a compiled ZML executable.
        \\
        \\Usage:
        \\  gemma4_transpose --input=/path/gemma4-26b-a4b --output=/path/gemma4-26b-a4b-transposed [--overwrite] [--parallelism=2]
        \\
    ;

    input: []const u8 = "",
    output: []const u8 = "",
    overwrite: bool = false,
    parallelism: usize = 2,
};

const TensorEntry = struct {
    name: []const u8,
    dtype_name: []const u8,
    dtype: zml.DataType,
    shape: []i64,
    data_start: u64,
    data_end: u64,
    transpose: bool,
};

const Converter = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    compiled: std.AutoHashMapUnmanaged(ShapeKey, *zml.Exe) = .empty,
    compiled_mutex: std.Io.Mutex = .init,
    transformed: std.atomic.Value(usize) = .init(0),
    copied_tensors: std.atomic.Value(usize) = .init(0),

    fn deinit(self: *Converter) void {
        var it = self.compiled.valueIterator();
        while (it.next()) |exe| {
            exe.*.deinit();
            self.allocator.destroy(exe.*);
        }
        self.compiled.deinit(self.allocator);
    }

    fn transposeBytes(self: *Converter, dtype: zml.DataType, input_shape: []const i64, bytes: []const u8) ![]u8 {
        if (input_shape.len < 2) return error.InvalidRank;

        var output_dims_buf: [zml.Shape.MAX_RANK]i64 = undefined;
        @memcpy(output_dims_buf[0..input_shape.len], input_shape);
        std.mem.swap(i64, &output_dims_buf[input_shape.len - 1], &output_dims_buf[input_shape.len - 2]);

        const input_zml_shape = zml.Shape.init(input_shape, dtype);
        const exe = try self.getOrCompile(input_zml_shape, ShapeKey.init(input_shape, dtype));

        var input_buffer = try zml.Buffer.fromBytes(self.io, self.platform, input_zml_shape, .replicated, bytes);
        defer input_buffer.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);

        args.set(.{input_buffer});
        exe.call(args, &results);

        var output_buffer = results.get(zml.Buffer);
        defer output_buffer.deinit();

        const output_slice = try output_buffer.toSliceAlloc(self.allocator, self.io);
        defer output_slice.free(self.allocator);

        const result = try self.allocator.alloc(u8, output_slice.constData().len);
        @memcpy(result, output_slice.constData());

        const expected_output_shape = zml.Shape.init(output_dims_buf[0..input_shape.len], dtype);
        std.debug.assert(result.len == expected_output_shape.byteSize());
        return result;
    }

    fn getOrCompile(self: *Converter, input_zml_shape: zml.Shape, key: ShapeKey) !*zml.Exe {
        try self.compiled_mutex.lock(self.io);
        defer self.compiled_mutex.unlock(self.io);

        const gop = try self.compiled.getOrPut(self.allocator, key);
        if (!gop.found_existing) {
            const t = zml.Tensor.fromShape(input_zml_shape);
            const exe = try self.allocator.create(zml.Exe);
            errdefer self.allocator.destroy(exe);
            exe.* = try self.platform.compileFn(
                self.allocator,
                self.io,
                transposeLastTwo,
                .{t},
                .{ .program_name = "gemma4_moe_weight_transpose" },
            );
            gop.value_ptr.* = exe;
        }

        return gop.value_ptr.*;
    }
};

const ShapeKey = struct {
    dtype: zml.DataType,
    rank: u8,
    dims: [zml.Shape.MAX_RANK]i64,

    fn init(dims: []const i64, dtype: zml.DataType) ShapeKey {
        var res: ShapeKey = .{
            .dtype = dtype,
            .rank = @intCast(dims.len),
            .dims = [_]i64{0} ** zml.Shape.MAX_RANK,
        };
        @memcpy(res.dims[0..dims.len], dims);
        return res;
    }
};

pub fn transposeLastTwo(x: zml.Tensor) zml.Tensor {
    const rank = x.rank();
    switch (rank) {
        2 => return x.transpose(.{ 1, 0 }),
        3 => return x.transpose(.{ 0, 2, 1 }),
        4 => return x.transpose(.{ 0, 1, 3, 2 }),
        5 => return x.transpose(.{ 0, 1, 2, 4, 3 }),
        else => stdx.debug.panic("unsupported tensor rank for transpose: {d}", .{rank}),
    }
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);
    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    if (args.input.len == 0 or args.output.len == 0) {
        log.err("missing --input or --output\n{s}", .{CliArgs.help});
        return error.InvalidArguments;
    }

    if (std.mem.eql(u8, args.input, args.output)) {
        log.err("--input and --output must be different folders", .{});
        return error.InvalidArguments;
    }

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    var converter: Converter = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
    };
    defer converter.deinit();

    try prepareOutput(io, args.output, args.overwrite);
    try copyNonSafetensors(allocator, io, args.input, args.output);
    try rewriteSafetensorsTree(&converter, args.input, args.output, @max(args.parallelism, 1));

    log.info("done: transposed {d} tensors, copied {d} tensors", .{
        converter.transformed.load(.monotonic),
        converter.copied_tensors.load(.monotonic),
    });
}

fn prepareOutput(io: std.Io, output: []const u8, overwrite: bool) !void {
    const cwd = std.Io.Dir.cwd();
    if (overwrite) {
        if (exists(io, output)) try std.Io.Dir.deleteTree(cwd, io, output);
    } else if (exists(io, output)) {
        return error.OutputAlreadyExists;
    }
    _ = try cwd.createDirPath(io, output);
}

fn copyNonSafetensors(allocator: std.mem.Allocator, io: std.Io, input_root: []const u8, output_root: []const u8) !void {
    var input_dir = try std.Io.Dir.openDir(.cwd(), io, input_root, .{ .iterate = true });
    defer input_dir.close(io);

    try copyNonSafetensorsInner(allocator, io, input_dir, output_root, "");
}

fn copyNonSafetensorsInner(allocator: std.mem.Allocator, io: std.Io, input_dir: std.Io.Dir, output_root: []const u8, rel: []const u8) !void {
    var it = input_dir.iterate();
    while (try it.next(io)) |entry| {
        const entry_rel = if (rel.len == 0)
            try allocator.dupe(u8, entry.name)
        else
            try std.Io.Dir.path.join(allocator, &.{ rel, entry.name });
        defer allocator.free(entry_rel);

        switch (entry.kind) {
            .directory => {
                const out_dir = try std.Io.Dir.path.join(allocator, &.{ output_root, entry_rel });
                defer allocator.free(out_dir);
                _ = try std.Io.Dir.cwd().createDirPath(io, out_dir);

                var child = try input_dir.openDir(io, entry.name, .{ .iterate = true });
                defer child.close(io);
                try copyNonSafetensorsInner(allocator, io, child, output_root, entry_rel);
            },
            .file => {
                if (std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
                const out_path = try std.Io.Dir.path.join(allocator, &.{ output_root, entry_rel });
                defer allocator.free(out_path);
                if (std.fs.path.dirname(out_path)) |parent| _ = try std.Io.Dir.cwd().createDirPath(io, parent);
                try copyFile(io, input_dir, entry.name, std.Io.Dir.cwd(), out_path);
            },
            else => {},
        }
    }
}

fn rewriteSafetensorsTree(converter: *Converter, input_root: []const u8, output_root: []const u8, parallelism: usize) !void {
    var input_dir = try std.Io.Dir.openDir(.cwd(), converter.io, input_root, .{ .iterate = true });
    defer input_dir.close(converter.io);

    var paths: std.ArrayList([]const u8) = .empty;
    defer {
        for (paths.items) |path| converter.allocator.free(path);
        paths.deinit(converter.allocator);
    }

    try collectSafetensors(converter.allocator, converter.io, input_dir, "", &paths);

    var group: stdx.Io.LimitedGroup = .init(parallelism);
    for (paths.items) |entry_rel| {
        try group.concurrent(converter.io, rewriteSafetensorsPath, .{ converter, input_root, output_root, entry_rel });
    }
    try group.await(converter.io);
}

fn collectSafetensors(
    allocator: std.mem.Allocator,
    io: std.Io,
    input_dir: std.Io.Dir,
    rel: []const u8,
    paths: *std.ArrayList([]const u8),
) !void {
    var it = input_dir.iterate();
    while (try it.next(io)) |entry| {
        const entry_rel = if (rel.len == 0)
            try allocator.dupe(u8, entry.name)
        else
            try std.Io.Dir.path.join(allocator, &.{ rel, entry.name });
        errdefer allocator.free(entry_rel);

        switch (entry.kind) {
            .directory => {
                var child = try input_dir.openDir(io, entry.name, .{ .iterate = true });
                defer child.close(io);
                try collectSafetensors(allocator, io, child, entry_rel, paths);
                allocator.free(entry_rel);
            },
            .file => {
                if (std.mem.endsWith(u8, entry.name, ".safetensors")) {
                    try paths.append(allocator, entry_rel);
                } else {
                    allocator.free(entry_rel);
                }
            },
            else => allocator.free(entry_rel),
        }
    }
}

fn rewriteSafetensorsPath(
    converter: *Converter,
    input_root: []const u8,
    output_root: []const u8,
    entry_rel: []const u8,
) void {
    rewriteSafetensorsPathInner(converter, input_root, output_root, entry_rel) catch |err| {
        std.debug.panic("failed to rewrite {s}: {}", .{ entry_rel, err });
    };
}

fn rewriteSafetensorsPathInner(
    converter: *Converter,
    input_root: []const u8,
    output_root: []const u8,
    entry_rel: []const u8,
) !void {
    const input_path = try std.Io.Dir.path.join(converter.allocator, &.{ input_root, entry_rel });
    defer converter.allocator.free(input_path);
    const out_path = try std.Io.Dir.path.join(converter.allocator, &.{ output_root, entry_rel });
    defer converter.allocator.free(out_path);
    if (std.fs.path.dirname(out_path)) |parent| _ = try std.Io.Dir.cwd().createDirPath(converter.io, parent);
    log.info("rewriting {s}", .{entry_rel});
    try rewriteSafetensorsFile(converter, input_path, out_path);
}

fn rewriteSafetensorsFile(converter: *Converter, input_path: []const u8, output_path: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(converter.allocator);
    defer arena.deinit();
    const arena_allocator = arena.allocator();

    var input = try std.Io.Dir.openFile(.cwd(), converter.io, input_path, .{ .mode = .read_only });
    defer input.close(converter.io);

    var header_len_buf: [SafetensorsHeaderBytes]u8 = undefined;
    _ = try input.readPositionalAll(converter.io, header_len_buf[0..], 0);
    const header_len = std.mem.readInt(u64, &header_len_buf, .little);
    const data_start = SafetensorsHeaderBytes + header_len;

    const header_bytes = try arena_allocator.alloc(u8, @intCast(header_len));
    _ = try input.readPositionalAll(converter.io, header_bytes, SafetensorsHeaderBytes);

    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena_allocator, header_bytes, .{});
    const root = parsed.object;

    var metadata: ?std.json.Value = null;
    var entries: std.ArrayList(TensorEntry) = .empty;
    defer entries.deinit(converter.allocator);

    var json_it = root.iterator();
    while (json_it.next()) |field| {
        const name = field.key_ptr.*;
        if (std.mem.eql(u8, name, "__metadata__")) {
            metadata = field.value_ptr.*;
            continue;
        }

        const obj = field.value_ptr.object;
        const dtype_name = obj.get("dtype").?.string;
        const shape_array = obj.get("shape").?.array;
        const offsets_array = obj.get("data_offsets").?.array;

        const dims = try arena_allocator.alloc(i64, shape_array.items.len);
        for (shape_array.items, 0..) |dim_value, i| {
            dims[i] = @intCast(dim_value.integer);
        }

        const start: u64 = @intCast(offsets_array.items[0].integer);
        const end: u64 = @intCast(offsets_array.items[1].integer);
        try entries.append(converter.allocator, .{
            .name = name,
            .dtype_name = dtype_name,
            .dtype = try parseDType(dtype_name),
            .shape = dims,
            .data_start = start,
            .data_end = end,
            .transpose = shouldTranspose(name),
        });
    }

    var header: std.Io.Writer.Allocating = .init(converter.allocator);
    defer header.deinit();
    try writeSafetensorsHeader(&header.writer, metadata, entries.items);

    var output = try std.Io.Dir.createFile(.cwd(), converter.io, output_path, .{});
    defer output.close(converter.io);
    var out_buf: [1024 * 1024]u8 = undefined;
    var output_writer = output.writer(converter.io, &out_buf);
    defer output_writer.interface.flush() catch {};
    try output_writer.interface.writeInt(u64, header.written().len, .little);
    try output_writer.interface.writeAll(header.written());

    const scratch = try converter.allocator.alloc(u8, 16 * 1024 * 1024);
    defer converter.allocator.free(scratch);

    for (entries.items) |entry| {
        const bytes_len: usize = @intCast(entry.data_end - entry.data_start);
        const abs_offset = data_start + entry.data_start;
        if (entry.transpose) {
            const raw = try converter.allocator.alloc(u8, bytes_len);
            defer converter.allocator.free(raw);
            _ = try input.readPositionalAll(converter.io, raw, abs_offset);

            const transposed = try converter.transposeBytes(entry.dtype, entry.shape, raw);
            defer converter.allocator.free(transposed);
            try output_writer.interface.writeAll(transposed);
            _ = converter.transformed.fetchAdd(1, .monotonic);
            log.info("  transposed {s}: {any} -> {any}", .{ entry.name, entry.shape, transposedShapeScratch(entry.shape) });
        } else {
            try copyRange(converter.io, input, &output_writer.interface, abs_offset, bytes_len, scratch);
            _ = converter.copied_tensors.fetchAdd(1, .monotonic);
        }
    }
    try output_writer.interface.flush();
}

fn writeSafetensorsHeader(writer: anytype, metadata: ?std.json.Value, entries: []const TensorEntry) !void {
    var offset: u64 = 0;

    try writer.writeByte('{');
    var first = true;
    if (metadata) |meta| {
        try writer.writeAll("\"__metadata__\":");
        try writeJson(writer, meta);
        first = false;
    }

    for (entries) |entry| {
        if (!first) try writer.writeByte(',');
        first = false;

        const out_shape = transposedShapeScratch(entry.shape);
        const shape = if (entry.transpose) out_shape[0..entry.shape.len] else entry.shape;
        const byte_len = entry.data_end - entry.data_start;

        try writeJson(writer, entry.name);
        try writer.writeAll(":{\"dtype\":");
        try writeJson(writer, entry.dtype_name);
        try writer.writeAll(",\"shape\":[");
        for (shape, 0..) |dim, i| {
            if (i != 0) try writer.writeByte(',');
            try writer.print("{d}", .{dim});
        }
        try writer.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + byte_len });
        offset += byte_len;
    }
    try writer.writeByte('}');
}

fn writeJson(writer: *std.Io.Writer, value: anytype) !void {
    var jw: std.json.Stringify = .{ .writer = writer };
    try jw.write(value);
}

fn copyRange(io: std.Io, input: std.Io.File, writer: *std.Io.Writer, offset: u64, len: usize, scratch: []u8) !void {
    var remaining = len;
    var current = offset;
    while (remaining > 0) {
        const n = @min(remaining, scratch.len);
        _ = try input.readPositionalAll(io, scratch[0..n], current);
        try writer.writeAll(scratch[0..n]);
        remaining -= n;
        current += n;
    }
}

fn transposedShapeScratch(shape: []const i64) [zml.Shape.MAX_RANK]i64 {
    var out = [_]i64{0} ** zml.Shape.MAX_RANK;
    @memcpy(out[0..shape.len], shape);
    if (shape.len >= 2) std.mem.swap(i64, &out[shape.len - 1], &out[shape.len - 2]);
    return out;
}

fn shouldTranspose(name: []const u8) bool {
    const is_moe_expert = std.mem.indexOf(u8, name, ".experts.") != null or
        std.mem.indexOf(u8, name, ".moe.") != null or
        std.mem.indexOf(u8, name, ".moe_block.") != null;
    if (!is_moe_expert) return false;

    return std.mem.endsWith(u8, name, "gate_up_proj") or
        std.mem.endsWith(u8, name, "gate_up_proj.weight") or
        std.mem.endsWith(u8, name, "down_proj") or
        std.mem.endsWith(u8, name, "down_proj.weight");
}

fn parseDType(dtype_name: []const u8) !zml.DataType {
    if (std.mem.eql(u8, dtype_name, "BF16")) return .bf16;
    if (std.mem.eql(u8, dtype_name, "F16")) return .f16;
    if (std.mem.eql(u8, dtype_name, "F32")) return .f32;
    if (std.mem.eql(u8, dtype_name, "F64")) return .f64;
    if (std.mem.eql(u8, dtype_name, "I8")) return .i8;
    if (std.mem.eql(u8, dtype_name, "I16")) return .i16;
    if (std.mem.eql(u8, dtype_name, "I32")) return .i32;
    if (std.mem.eql(u8, dtype_name, "I64")) return .i64;
    if (std.mem.eql(u8, dtype_name, "U8")) return .u8;
    if (std.mem.eql(u8, dtype_name, "U16")) return .u16;
    if (std.mem.eql(u8, dtype_name, "U32")) return .u32;
    if (std.mem.eql(u8, dtype_name, "U64")) return .u64;
    if (std.mem.eql(u8, dtype_name, "BOOL")) return .bool;
    return error.UnsupportedDType;
}

fn exists(io: std.Io, path: []const u8) bool {
    std.Io.Dir.access(.cwd(), io, path, .{}) catch return false;
    return true;
}

fn copyFile(io: std.Io, src_dir: std.Io.Dir, src_name: []const u8, dst_dir: std.Io.Dir, dst_path: []const u8) !void {
    var source = try src_dir.openFile(io, src_name, .{ .mode = .read_only });
    defer source.close(io);

    var destination = try dst_dir.createFile(io, dst_path, .{});
    defer destination.close(io);

    var read_buf: [1024 * 1024]u8 = undefined;
    var write_buf: [1024 * 1024]u8 = undefined;
    var reader: std.Io.File.Reader = .initStreaming(source, io, &read_buf);
    var writer = destination.writer(io, &write_buf);
    defer writer.interface.flush() catch {};

    _ = try reader.interface.streamRemaining(&writer.interface);
    try writer.interface.flush();
}
