const std = @import("std");

const DataType = @import("dtype.zig").DataType;
const Shape = @import("shape.zig").Shape;
const Dims = Shape.DimsArray;

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
    const file_type = resolveFiletype(model_path);

    return switch (file_type) {
        .index => blk: {
            const file_reader_buf = try allocator.alloc(u8, 64 * 1024);
            defer allocator.free(file_reader_buf);

            var index_reader = model.reader(io, file_reader_buf);

            var safetensors_index = try parseSafetensorsIndex(
                allocator,
                &index_reader.interface,
            );
            defer safetensors_index.deinit();

            var registry: TensorRegistry = .init(allocator);

            try parseSafetensorsIndexFiles(
                allocator,
                io,
                &registry,
                safetensors_index,
                repo,
            );

            break :blk registry;
        },
        .safetensors => blk: {
            var registry: TensorRegistry = .init(allocator);

            try parseSafetensors(allocator, io, &registry, model);

            break :blk registry;
        },
        else => return error.InvalidPath,
    };
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

pub fn parseSafetensors(
    allocator: std.mem.Allocator,
    io: std.Io,
    registry: *TensorRegistry,
    file: std.Io.File,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const header_buf = try allocator.alloc(u8, 64 * 1024);
    defer allocator.free(header_buf);

    var reader = file.reader(io, header_buf);

    var file_uri_buf: [1024]u8 = undefined;
    const file_uri_len = try file.realPath(io, &file_uri_buf);
    const file_uri = file_uri_buf[0..file_uri_len];

    const json_header_length: u64 = try reader.interface.takeInt(u64, .little);
    const data_start_offset = BYTES_HEADER + json_header_length;
    const json_data = try arena.allocator().alloc(u8, @intCast(json_header_length));

    try reader.interface.readSliceAll(json_data);

    const headers: std.json.Value = try std.json.parseFromSliceLeaky(
        std.json.Value,
        arena.allocator(),
        json_data,
        .{},
    );

    var it = headers.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        // Free-form strings, which nothing reads.
        if (std.mem.eql(u8, key, "__metadata__")) continue;

        const shape_field = value.object.get("shape").?.array;

        if (shape_field.items.len > Shape.MAX_RANK) {
            log.warn("Can't load tensor {s}, too many dims: {}", .{ key, shape_field.items.len });
            return error.TooManyDimensions;
        }

        const offset_field = value.object.get("data_offsets").?;
        const start: u64 = @intCast(offset_field.array.items[0].integer);
        const dtype = try stringToDtype(value.object.get("dtype").?.string);

        var dims: Dims = .empty;
        for (shape_field.items) |d| {
            dims.appendAssumeCapacity(d.integer);
        }

        const tensor: Tensor = .{
            .file_uri = file_uri,
            .name = key,
            .shape = .init(dims.slice(), dtype),
            .offset = data_start_offset + start,
        };

        try registry.registerTensor(tensor);
    }
}

const SafetensorsIndex = struct {
    pub const Map = std.StringArrayHashMapUnmanaged(std.ArrayList([]const u8));

    arena: std.heap.ArenaAllocator,

    map: Map,

    pub fn init(allocator: std.mem.Allocator) SafetensorsIndex {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .map = .empty,
        };
    }

    pub fn iterator(self: *SafetensorsIndex) Map.Iterator {
        return self.map.iterator();
    }

    pub fn deinit(self: *SafetensorsIndex) void {
        const allocator = self.arena.allocator();
        self.map.deinit(allocator);
        self.arena.deinit();
    }
};

fn parseSafetensorsIndex(
    allocator: std.mem.Allocator,
    reader: *std.Io.Reader,
) !SafetensorsIndex {
    var safetensors_index: SafetensorsIndex = .init(allocator);
    errdefer safetensors_index.deinit();

    const arena_allocator = safetensors_index.arena.allocator();
    var json_reader: std.json.Reader = .init(arena_allocator, reader);

    const index = try std.json.parseFromTokenSourceLeaky(
        std.json.Value,
        arena_allocator,
        &json_reader,
        .{ .allocate = .alloc_if_needed },
    );

    const weight_map = index.object.get("weight_map");

    if (weight_map) |wm| {
        var it = wm.object.iterator();

        while (it.next()) |entry| {
            const weight_name = entry.key_ptr.*;
            const filename = entry.value_ptr.string;

            const map_entry = try safetensors_index.map.getOrPut(arena_allocator, filename);

            if (!map_entry.found_existing) {
                map_entry.value_ptr.* = .empty;
            }

            try map_entry.value_ptr.append(arena_allocator, try arena_allocator.dupe(u8, weight_name));
        }
    } else {
        log.warn("No weight_map attribute found in index", .{});
    }

    return safetensors_index;
}

fn parseSafetensorsIndexFiles(
    allocator: std.mem.Allocator,
    io: std.Io,
    registry: *TensorRegistry,
    safetensors_index: SafetensorsIndex,
    repo: std.Io.Dir,
) !void {
    const filenames = safetensors_index.map.keys();
    for (filenames) |filename| {
        const file = try repo.openFile(io, filename, .{});
        defer file.close(io);
        try parseSafetensors(allocator, io, registry, file);
    }
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

fn stringToDtype(safetensor_type: []const u8) !DataType {
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

    return map.get(safetensor_type) orelse {
        log.err("Unsupported safetensor data type: {s}", .{safetensor_type});
        return error.UnsupportedDataType;
    };
}
