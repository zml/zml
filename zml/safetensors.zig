const std = @import("std");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;
const testing = @import("testing.zig");

pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);

const log = std.log.scoped(.@"zml/safetensors");

const BYTES_HEADER = 8;

pub const FileType = enum {
    index,
    safetensors,
    unknown,
};

pub fn fetchRegistry(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo: std.Io.Dir,
    model: std.Io.File,
    model_path: []const u8,
) !TensorRegistry {
    var registry: TensorRegistry = .init(allocator);
    switch (resolveFiletype(model_path)) {
        .index => {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const scratch = arena.allocator();

            var index_uri_buf: [1024]u8 = undefined;
            const index_uri = index_uri_buf[0..try model.realPath(io, &index_uri_buf)];

            var index_reader = model.reader(io, try scratch.alloc(u8, 64 * 1024));
            var json_reader: std.json.Reader = .init(scratch, &index_reader.interface);
            // The default allocation, `.alloc_always` over a `json.Reader`,
            // copies every string out of `index_reader`'s buffer, which the
            // next refill overwrites.
            const Index = struct { weight_map: std.json.ArrayHashMap([]const u8) };
            const index = std.json.parseFromTokenSourceLeaky(Index, scratch, &json_reader, .{ .ignore_unknown_fields = true }) catch |err| switch (err) {
                error.OutOfMemory => |e| return e,
                error.ReadFailed => return index_reader.err orelse err,
                else => {
                    log.warn("Invalid safetensors index {s}: needs a weight_map from tensor names to file names ({t})", .{ index_uri, err });
                    return error.InvalidSafetensorsHeader;
                },
            };

            // The files the weight map names, each read once.
            var files: std.StringArrayHashMapUnmanaged(void) = .empty;
            for (index.weight_map.map.values()) |file_name| try files.put(scratch, file_name, {});
            for (files.keys()) |file_name| {
                const file = try repo.openFile(io, file_name, .{});
                defer file.close(io);
                try parseSafetensors(allocator, io, &registry, file);
            }
        },
        .safetensors => try parseSafetensors(allocator, io, &registry, model),
        .unknown => return error.InvalidPath,
    }
    return registry;
}

pub const TensorReader = struct {
    tensor: Tensor,
    file: std.Io.File,
    file_reader: std.Io.File.Reader,
    alignment: ?std.mem.Alignment,
    remaining: u64,
    padding_remaining: u64,

    io: std.Io,
    interface: std.Io.Reader,

    pub const InitOpts = struct {
        alignment: ?std.mem.Alignment = null,
    };

    pub fn init(
        io: std.Io,
        tensor: Tensor,
        buffer: []u8,
        opts: InitOpts,
    ) !TensorReader {
        const file = try std.Io.Dir.openFile(.cwd(), io, tensor.file_uri, .{ .mode = .read_only });
        errdefer file.close(io);

        const offset, const padding_remaining = if (opts.alignment) |alignment| blk: {
            const aligned_offset = std.mem.alignBackward(u64, tensor.offset, alignment.toByteUnits());
            break :blk .{ aligned_offset, tensor.offset - aligned_offset };
        } else blk: {
            break :blk .{ tensor.offset, 0 };
        };

        var file_reader = file.reader(io, buffer);
        try file_reader.seekTo(offset);

        return .{
            .tensor = tensor,
            .file = file,
            .file_reader = file_reader,
            .alignment = opts.alignment,
            .remaining = tensor.byteSize(),
            .padding_remaining = padding_remaining,
            .io = io,
            .interface = .{
                .vtable = if (opts.alignment) |_| blk: {
                    break :blk &.{
                        .stream = streamAligned,
                        .discard = discardAligned,
                    };
                } else blk: {
                    break :blk &.{
                        .stream = stream,
                        .discard = discard,
                    };
                },
                .buffer = &.{},
                .seek = 0,
                .end = 0,
            },
        };
    }

    pub fn deinit(self: *TensorReader) void {
        self.file.close(self.io);
    }

    // Limit single reads to 1GB to avoid issues on macOS where pread returns
    // INT_MAX when the requested size is larger than 2GB
    const max_single_read_bytes: u64 = 1 << 30;

    fn stream(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *TensorReader = @fieldParentPtr("interface", r);
        if (self.remaining == 0) {
            return error.EndOfStream;
        }

        const combined_limit = limit.min(.limited64(self.remaining)).min(.limited64(max_single_read_bytes));
        const n = try self.file_reader.interface.stream(w, combined_limit);
        self.remaining -= n;
        return n;
    }

    fn discard(r: *std.Io.Reader, limit: std.Io.Limit) std.Io.Reader.Error!usize {
        const self: *TensorReader = @fieldParentPtr("interface", r);
        if (self.remaining == 0) {
            return error.EndOfStream;
        }

        const combined_limit = limit.min(.limited64(self.remaining));
        const n = try self.file_reader.interface.discard(combined_limit);
        self.remaining -= n;
        return n;
    }

    fn streamAligned(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *TensorReader = @fieldParentPtr("interface", r);
        if (self.remaining == 0) {
            return error.EndOfStream;
        }

        const requested_bytes = @min(limit.minInt64(self.remaining), max_single_read_bytes);

        if (self.padding_remaining > 0) {
            _ = try self.file_reader.interface.discard(.limited64(self.padding_remaining));
            self.padding_remaining = 0;
        }

        const align_bytes = self.alignment.?.toByteUnits();
        const file_offset = self.file_reader.pos - std.mem.alignBackward(u64, self.file_reader.pos, align_bytes);
        const writer_addr = @intFromPtr(w.buffer.ptr) + w.end;
        const writer_offset = writer_addr - std.mem.alignBackward(usize, writer_addr, align_bytes);

        if (file_offset != writer_offset) {
            const skip = if (file_offset >= writer_offset) blk: {
                break :blk file_offset - writer_offset;
            } else blk: {
                break :blk align_bytes - writer_offset + file_offset;
            };

            w.advance(skip);
        }

        if (w.writableSliceGreedy(align_bytes)) |dest| {
            const phys_buf = dest.ptr - file_offset;
            const phys_pos = self.file_reader.pos - file_offset;
            const phys_size = std.mem.alignBackward(usize, @min(dest.len, requested_bytes) + file_offset, align_bytes);

            if (phys_size > file_offset) {
                const phys_read = self.file.readPositional(self.io, &.{phys_buf[0..phys_size]}, phys_pos) catch unreachable;
                if (phys_read > file_offset) {
                    const bytes_read = phys_read - file_offset;
                    w.advance(bytes_read);
                    self.file_reader.pos += bytes_read;
                    self.remaining -= bytes_read;
                    return bytes_read;
                }
            }
        } else |_| {}

        const bytes_read = try self.file_reader.interface.stream(w, .limited64(requested_bytes));
        self.remaining -= bytes_read;

        return bytes_read;
    }

    fn discardAligned(r: *std.Io.Reader, limit: std.Io.Limit) std.Io.Reader.Error!usize {
        const self: *TensorReader = @fieldParentPtr("interface", r);
        if (self.remaining == 0) return 0;

        if (self.padding_remaining > 0) {
            _ = try self.file_reader.interface.discard(.limited64(self.padding_remaining));
            self.padding_remaining = 0;
        }

        const combined_limit = limit.min(.limited64(self.remaining));
        const n = try self.file_reader.interface.discard(combined_limit);
        self.remaining -= n;
        return n;
    }
};

pub const Tensor = struct {
    file_uri: []const u8,
    name: []const u8,
    shape: Shape,
    offset: u64,

    pub fn byteSize(self: Tensor) u64 {
        return self.shape.byteSize();
    }

    pub fn reader(self: Tensor, io: std.Io, buffer: []u8, opts: TensorReader.InitOpts) !TensorReader {
        return TensorReader.init(io, self, buffer, opts);
    }

    pub fn format(self: Tensor, writer: *std.Io.Writer) !void {
        try writer.print("Tensor(name={s} shape={f} size={d}, offset={d}, file_uri={s})", .{
            self.name,
            self.shape,
            self.byteSize(),
            self.offset,
            self.file_uri,
        });
    }
};

pub const TensorRegistry = struct {
    arena: std.heap.ArenaAllocator,

    tensors: Tensors,

    pub fn init(allocator: std.mem.Allocator) TensorRegistry {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tensors = .{},
        };
    }

    pub fn fromRepo(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
    ) !TensorRegistry {
        const entrypoint_file, const entrypoint_path = try resolveModelEntrypoint(io, repo);
        defer entrypoint_file.close(io);
        return try fetchRegistry(allocator, io, repo, entrypoint_file, entrypoint_path);
    }

    pub fn fromPath(
        allocator: std.mem.Allocator,
        io: std.Io,
        path: []const u8,
    ) !TensorRegistry {
        var repo = try resolveModelRepo(io, path);
        defer repo.close(io);

        const entrypoint_file, const entrypoint_path = if (std.mem.endsWith(u8, path, ".safetensors.index.json") or std.mem.endsWith(u8, path, ".safetensors"))
            .{ try repo.openFile(io, path, .{ .mode = .read_only }), path }
        else
            try resolveModelEntrypoint(io, repo);
        defer entrypoint_file.close(io);

        return try fetchRegistry(allocator, io, repo, entrypoint_file, entrypoint_path);
    }

    pub fn deinit(self: *TensorRegistry) void {
        const allocator = self.arena.allocator();
        self.tensors.deinit(allocator);
        self.arena.deinit();
    }

    pub fn registerTensor(
        self: *TensorRegistry,
        tensor: Tensor,
    ) !void {
        const allocator = self.arena.allocator();

        var tensor_copy = tensor;

        tensor_copy.name = try allocator.dupe(u8, tensor.name);
        tensor_copy.file_uri = try allocator.dupe(u8, tensor.file_uri);

        try self.tensors.put(allocator, tensor_copy.name, tensor_copy);
    }

    pub fn reader(
        self: *TensorRegistry,
        io: std.Io,
        tensor_name: []const u8,
        buffer: []u8,
    ) !TensorReader {
        const tensor = self.tensors.get(tensor_name) orelse {
            log.err("Tensor {s} not found in registry", .{tensor_name});
            return error.TensorNotFound;
        };

        return try .init(io, tensor, buffer, .{});
    }

    pub fn iterator(self: *TensorRegistry) Tensors.Iterator {
        return self.tensors.iterator();
    }

    pub fn totalBytes(self: *TensorRegistry) u64 {
        var total: u64 = 0;

        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            total += entry.value_ptr.byteSize();
        }

        return total;
    }
};

/// Adds the tensors of the safetensors `file` to `registry`. A header that
/// is not an object of tensors, each with a `dtype`, a `shape` and
/// `data_offsets` spanning exactly its bytes within the file, is
/// `InvalidSafetensorsHeader`, logged.
pub fn parseSafetensors(
    allocator: std.mem.Allocator,
    io: std.Io,
    registry: *TensorRegistry,
    file: std.Io.File,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var reader = file.reader(io, try arena.allocator().alloc(u8, 64 * 1024));

    var file_uri_buf: [1024]u8 = undefined;
    const file_uri_len = try file.realPath(io, &file_uri_buf);
    const file_uri = file_uri_buf[0..file_uri_len];

    // `File.Reader` returns `ReadFailed` with its cause in `err`, which a
    // failed seek leaves unset.
    const json_header_length: u64 = reader.interface.takeInt(u64, .little) catch |err| switch (err) {
        error.EndOfStream => {
            log.warn("Invalid safetensors header in {s}: shorter than its {d}-byte header length", .{ file_uri, BYTES_HEADER });
            return error.InvalidSafetensorsHeader;
        },
        error.ReadFailed => return reader.err orelse err,
    };
    // The header and every tensor's data must lie inside the file.
    const file_length = try file.length(io);
    if (json_header_length > file_length -| BYTES_HEADER) {
        log.warn("Invalid safetensors header in {s}: {d} bytes of header in a file of {d}", .{ file_uri, json_header_length, file_length });
        return error.InvalidSafetensorsHeader;
    }
    const data_start_offset = BYTES_HEADER + json_header_length;
    const json_data = try arena.allocator().alloc(u8, @intCast(json_header_length));

    reader.interface.readSliceAll(json_data) catch |err| return switch (err) {
        error.ReadFailed => reader.err orelse err,
        error.EndOfStream => err,
    };

    const headers = std.json.parseFromSliceLeaky(std.json.ArrayHashMap(std.json.Value), arena.allocator(), json_data, .{}) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        else => {
            log.warn("Invalid safetensors header in {s}: not a JSON object ({t})", .{ file_uri, err });
            return error.InvalidSafetensorsHeader;
        },
    };
    const data_len = file_length - data_start_offset;

    // `data_offsets` are `[start, end)` after the header.
    const Entry = struct { dtype: []const u8, shape: []const i64, data_offsets: [2]u64 };
    var it = headers.map.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;

        // Free-form strings, which nothing reads.
        if (std.mem.eql(u8, key, "__metadata__")) continue;

        const entry = std.json.parseFromValueLeaky(Entry, arena.allocator(), kv.value_ptr.*, .{ .ignore_unknown_fields = true }) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            else => {
                log.warn("Invalid safetensors header in {s}: tensor {s} needs a string dtype, an integer shape and two unsigned data_offsets ({t})", .{ file_uri, key, err });
                return error.InvalidSafetensorsHeader;
            },
        };

        if (entry.shape.len > Shape.MAX_RANK) {
            log.warn("Cannot load {s}: tensor {s} has {d} dims, more than {d}", .{ file_uri, key, entry.shape.len, Shape.MAX_RANK });
            return error.TooManyDimensions;
        }
        const dtype = stringToDtype(entry.dtype) orelse {
            log.warn("Cannot load {s}: tensor {s} has dtype {s}, which zml does not support", .{ file_uri, key, entry.dtype });
            return error.UnsupportedDataType;
        };

        const start, const end = entry.data_offsets;
        const byte_size = checkedByteSize(entry.shape, dtype);
        if (byte_size == null or start > end or end > data_len or end - start != byte_size.?) {
            log.warn("Invalid safetensors header in {s}: tensor {s} has data_offsets [{d},{d}], which must span the bytes of its {s} shape {any} within the file's {d} data bytes", .{ file_uri, key, start, end, entry.dtype, entry.shape, data_len });
            return error.InvalidSafetensorsHeader;
        }

        const tensor: Tensor = .{
            .file_uri = file_uri,
            .name = key,
            .shape = .init(entry.shape, dtype),
            .offset = data_start_offset + start,
        };

        try registry.registerTensor(tensor);
    }
}

/// The bytes of `dims` of `dtype`, as `Shape.byteSize` counts them, or null
/// on a negative dim or a bit count past u64.
fn checkedByteSize(dims: []const i64, dtype: DataType) ?u64 {
    var bits: u64 = dtype.bitSizeOf();
    for (dims) |d| bits = std.math.mul(u64, bits, std.math.cast(u64, d) orelse return null) catch return null;
    return std.math.divCeil(u64, bits, 8) catch unreachable;
}

test "parseSafetensors rejects a malformed header or data_offsets that do not span their tensor within the file" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const max = "18446744073709551615";
    // One U8 tensor over the file's 4 data bytes. `header` overrides the
    // whole header, `header_len` the header's length field.
    const cases = [_]struct { offsets: []const u8 = "[0,4]", shape: []const u8 = "[4]", header: ?[]const u8 = null, header_len: ?u64 = null, valid: bool = false }{
        .{ .valid = true },
        .{ .offsets = "[4,0]" },
        .{ .offsets = "[0,3]" },
        .{ .offsets = "[0,5]" },
        .{ .offsets = "[-4,4]" },
        .{ .offsets = "[0," ++ max ++ "]" },
        .{ .offsets = "[" ++ max ++ "," ++ max ++ "]" },
        .{ .offsets = "[0,18446744073709551616]" },
        .{ .offsets = "[0]" },
        .{ .shape = "[-4]" },
        .{ .shape = "[4294967296,4294967296]" },
        .{ .header = "{\"x\":{\"dtype\":\"U8\",\"shape\":[4]}}" },
        .{ .header = "{\"x\":{\"dtype\":\"U8\",\"shape\":4,\"data_offsets\":[0,4]}}" },
        .{ .header = "[]" },
        .{ .header = "{\"x\":" },
        .{ .header_len = 1 << 20 },
        .{ .header_len = std.math.maxInt(u64) },
    };
    for (cases, 0..) |case, index| {
        var header_buffer: [128]u8 = undefined;
        const header = case.header orelse try std.fmt.bufPrint(&header_buffer, "{{\"x\":{{\"dtype\":\"U8\",\"shape\":{s},\"data_offsets\":{s}}}}}", .{ case.shape, case.offsets });
        var name_buffer: [32]u8 = undefined;
        const file = try tmp.dir.createFile(io, try std.fmt.bufPrint(&name_buffer, "{d}.safetensors", .{index}), .{ .read = true });
        defer file.close(io);
        var length: [BYTES_HEADER]u8 = undefined;
        std.mem.writeInt(u64, &length, case.header_len orelse header.len, .little);
        try file.writePositionalAll(io, &length, 0);
        try file.writePositionalAll(io, header, length.len);
        try file.writePositionalAll(io, &.{ 1, 2, 3, 4 }, length.len + header.len);

        var registry: TensorRegistry = .init(allocator);
        defer registry.deinit();
        const result = parseSafetensors(allocator, io, &registry, file);
        if (case.valid) {
            try result;
            try std.testing.expectEqual(@as(u64, BYTES_HEADER + header.len), registry.tensors.get("x").?.offset);
        } else {
            try std.testing.expectError(error.InvalidSafetensorsHeader, result);
        }
    }
}

test "fetchRegistry refuses a file shorter than its header length or an index without a weight_map" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const files = [_]struct { []const u8, []const u8 }{
        .{ "short.safetensors", &.{ 1, 2, 3 } },
        .{ "list.safetensors.index.json", "[]" },
        .{ "unmapped.safetensors.index.json", "{\"metadata\":{\"total_size\":4}}" },
        .{ "unnamed.safetensors.index.json", "{\"weight_map\":{\"w\":4}}" },
    };
    for (files) |file| {
        const name, const data = file;
        try tmp.dir.writeFile(io, .{ .sub_path = name, .data = data });
        const model = try tmp.dir.openFile(io, name, .{});
        defer model.close(io);
        try std.testing.expectError(error.InvalidSafetensorsHeader, fetchRegistry(allocator, io, tmp.dir, model, name));
    }
}

test "fromRepo reads every file an index longer than its read buffer names" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    // About 90 KiB of weight map, read through a 64 KiB buffer: file names
    // straddle refills.
    const shards = 16;
    const tensors_per_shard = 64;
    var index: std.Io.Writer.Allocating = .init(scratch);
    try index.writer.writeAll("{\"metadata\":{\"total_size\":1024},\"weight_map\":{");
    for (0..shards) |shard| {
        const file_name = try std.fmt.allocPrint(scratch, "model-{d:0>5}-of-{d:0>5}.safetensors", .{ shard + 1, shards });
        var tensors: [tensors_per_shard]testing.TensorBytes = undefined;
        for (&tensors, 0..) |*tensor, expert| {
            const name = try std.fmt.allocPrint(scratch, "model.layers.{d}.mlp.experts.{d}.down_proj.weight", .{ shard, expert });
            tensor.* = .{ .name = name, .dtype = "U8", .dims = &.{1}, .bytes = &.{0} };
            try index.writer.print("{s}\"{s}\":\"{s}\"", .{ if (shard + expert == 0) "" else ",", name, file_name });
        }
        try testing.writeSafetensorsFile(allocator, io, tmp.dir, file_name, &tensors);
    }
    try index.writer.writeAll("}}");
    try std.testing.expect(index.written().len > 64 * 1024);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.written() });

    var registry: TensorRegistry = try .fromRepo(allocator, io, tmp.dir);
    defer registry.deinit();
    // Each file holds its own tensors: a misread file name would lose some.
    try std.testing.expectEqual(@as(usize, shards * tensors_per_shard), registry.tensors.count());
}

pub fn resolveFiletype(path: []const u8) FileType {
    if (std.mem.endsWith(u8, path, ".safetensors.index.json")) {
        return .index;
    } else if (std.mem.endsWith(u8, path, ".safetensors")) {
        return .safetensors;
    } else {
        return .unknown;
    }
}

pub const ModelPathResolutionError = error{
    FileNotFound,
    InvalidPath,
} || std.mem.Allocator.Error;

pub fn resolveModelRepo(
    io: std.Io,
    path: []const u8,
) ModelPathResolutionError!std.Io.Dir {
    if (std.mem.endsWith(u8, path, ".safetensors.index.json") or
        std.mem.endsWith(u8, path, ".safetensors"))
    {
        const dir_path = std.fs.path.dirname(path) orelse return ModelPathResolutionError.InvalidPath;

        const dir = std.Io.Dir.openDir(.cwd(), io, dir_path, .{}) catch |e| {
            return switch (e) {
                error.FileNotFound => ModelPathResolutionError.FileNotFound,
                else => ModelPathResolutionError.InvalidPath,
            };
        };

        return dir;
    }

    const dir = std.Io.Dir.openDir(.cwd(), io, path, .{}) catch |e| {
        return switch (e) {
            error.FileNotFound => ModelPathResolutionError.FileNotFound,
            else => ModelPathResolutionError.InvalidPath,
        };
    };

    return dir;
}

pub fn resolveModelEntrypoint(io: std.Io, repo: std.Io.Dir) ModelPathResolutionError!struct { std.Io.File, []const u8 } {
    {
        const index_name = "model.safetensors.index.json";
        if (repo.openFile(io, index_name, .{ .mode = .read_only })) |index_file| {
            return .{ index_file, "model.safetensors.index.json" };
        } else |_| {}
    }

    {
        const model_name = "model.safetensors";
        if (repo.openFile(io, model_name, .{ .mode = .read_only })) |model_file| {
            return .{ model_file, "model.safetensors" };
        } else |_| {}
    }

    return ModelPathResolutionError.FileNotFound;
}

fn stringToDtype(safetensor_type: []const u8) ?DataType {
    const map = std.StaticStringMap(DataType).initComptime(.{
        .{ "F64", .f64 },
        .{ "F32", .f32 },
        .{ "F16", .f16 },
        .{ "BF16", .bf16 },
        .{ "F8_E8M0", .f8e8m0 },
        .{ "F8_E4M3", .f8e4m3fn },
        .{ "F4_E2M1", .f4e2m1 },
        .{ "I64", .i64 },
        .{ "I32", .i32 },
        .{ "I16", .i16 },
        .{ "I8", .i8 },
        .{ "U64", .u64 },
        .{ "U32", .u32 },
        .{ "U16", .u16 },
        .{ "U8", .u8 },
        .{ "BOOL", .bool },
    });

    return map.get(safetensor_type);
}
