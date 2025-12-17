const std = @import("std");

const Shape = @import("shape.zig").Shape;
const DataType = @import("dtype.zig").DataType;

const Dims = Shape.DimsArray;
const StringBuilder = std.ArrayListUnmanaged(u8);

pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);
pub const Metadatas = std.StringArrayHashMapUnmanaged(Metadata);

const log = std.log.scoped(.@"zml/safetensors");

const BYTES_HEADER = 8;

pub fn parseFromPath(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) !TensorRegistry {
    var entrypoint = try resolveModelEntrypoint(allocator, io, path);
    defer entrypoint.deinit(allocator, io);
    log.debug("Resolved model path entrypoint: {s}", .{entrypoint.path});

    const repo_path = try resolveModelRepoPath(allocator, io, path);
    defer allocator.free(repo_path);
    log.debug("Resolved model repo path: {s}", .{repo_path});

    const repo = try std.Io.Dir.openDir(.cwd(), io, repo_path, .{});
    defer repo.close(io);

    const file_type = resolveFiletype(entrypoint.path);
    log.debug("Resolved file type: {any}", .{file_type});

    return switch (file_type) {
        .index => blk: {
            const index_file = try std.Io.Dir.openFile(.cwd(), io, entrypoint.path, .{ .mode = .read_only });
            defer index_file.close(io);

            const file_reader_buf = try allocator.alloc(u8, 64 * 1024);
            defer allocator.free(file_reader_buf);

            var index_reader = index_file.reader(io, file_reader_buf);

            var safetensors_index = try parseSafetensorsIndex(
                allocator,
                &index_reader.interface,
            );
            defer safetensors_index.deinit();

            var registry: TensorRegistry = try .initWithMetadata(allocator, safetensors_index.metadata);

            try parseSafetensorsIndexFiles(
                allocator,
                io,
                safetensors_index,
                &registry,
                repo,
                repo_path,
            );

            break :blk registry;
        },
        .safetensors => blk: {
            var registry: TensorRegistry = .init(allocator);

            const file = try repo.openFile(io, entrypoint.path, .{});
            defer file.close(io);

            const header_buf = try allocator.alloc(u8, 16 * 1024);
            defer allocator.free(header_buf);

            var reader = file.reader(io, header_buf);

            try parseSafetensors(allocator, entrypoint.path, &reader.interface, &registry);

            break :blk registry;
        },
        else => return error.InvalidPath,
    };
}

pub const FileType = enum {
    index,
    safetensors,
    unknown,
};

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

pub fn resolveModelRepoPath(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ModelPathResolutionError![]const u8 {
    const resolved_path = if (std.mem.startsWith(u8, path, "/"))
        try std.fmt.allocPrint(allocator, "file://{s}", .{path})
    else
        try allocator.dupe(u8, path);
    defer allocator.free(resolved_path);

    if (std.mem.endsWith(u8, resolved_path, ".safetensors.index.json") or
        std.mem.endsWith(u8, resolved_path, ".safetensors"))
    {
        const dir_path = std.fs.path.dirname(resolved_path) orelse return ModelPathResolutionError.InvalidPath;
        return try allocator.dupe(u8, dir_path);
    }

    if (std.Io.Dir.openDir(.cwd(), io, resolved_path, .{})) |dir| {
        defer dir.close(io);
        return try allocator.dupe(u8, resolved_path);
    } else |_| {}

    return ModelPathResolutionError.FileNotFound;
}

pub const ModelEntrypoint = struct {
    file: std.Io.File,
    path: []const u8,

    pub fn deinit(self: *ModelEntrypoint, allocator: std.mem.Allocator, io: std.Io) void {
        self.file.close(io);
        allocator.free(self.path);
    }
};

pub fn resolveModelEntrypoint(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ModelPathResolutionError!ModelEntrypoint {
    const resolved_path = if (std.mem.startsWith(u8, path, "/"))
        try std.fmt.allocPrint(allocator, "file://{s}", .{path})
    else
        try allocator.dupe(u8, path);

    if (std.mem.endsWith(u8, resolved_path, ".safetensors.index.json") or
        std.mem.endsWith(u8, resolved_path, ".safetensors"))
    {
        const file = std.Io.Dir.openFile(.cwd(), io, resolved_path, .{ .mode = .read_only }) catch |e| {
            allocator.free(resolved_path);
            log.err("Error opening model entrypoint file '{s}': {any}", .{ resolved_path, e });

            return switch (e) {
                error.FileNotFound => ModelPathResolutionError.FileNotFound,
                else => ModelPathResolutionError.InvalidPath,
            };
        };

        return .{ .file = file, .path = resolved_path };
    }

    const repo = std.Io.Dir.openDir(.cwd(), io, resolved_path, .{}) catch |e| {
        allocator.free(resolved_path);

        return switch (e) {
            error.FileNotFound => ModelPathResolutionError.FileNotFound,
            else => ModelPathResolutionError.InvalidPath,
        };
    };
    defer repo.close(io);

    {
        const index_name = "model.safetensors.index.json";
        if (repo.openFile(io, index_name, .{ .mode = .read_only })) |index_file| {
            const index_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ resolved_path, index_name });
            allocator.free(resolved_path);
            return .{ .file = index_file, .path = index_path };
        } else |_| {}
    }

    {
        const model_name = "model.safetensors";
        if (repo.openFile(io, model_name, .{ .mode = .read_only })) |model_file| {
            const model_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ resolved_path, model_name });
            allocator.free(resolved_path);
            return .{ .file = model_file, .path = model_path };
        } else |_| {}
    }

    allocator.free(resolved_path);
    return ModelPathResolutionError.FileNotFound;
}

pub const TensorReader = struct {
    file: std.Io.File,
    file_reader: std.Io.File.Reader,
    remaining: u64,
    io: std.Io,
    interface: std.Io.Reader,

    pub const Error = error{TensorNotFound} || std.Io.File.OpenError || std.Io.File.Reader.SeekError || std.mem.Allocator.Error;

    pub fn init(
        io: std.Io,
        tensor: Tensor,
        buffer: []u8,
    ) Error!TensorReader {
        const file = try std.Io.Dir.openFile(.cwd(), io, tensor.file_uri, .{ .mode = .read_only });
        errdefer file.close(io);

        var file_reader = file.reader(io, buffer);
        try file_reader.seekTo(tensor.offset);

        return .{
            .file = file,
            .file_reader = file_reader,
            .remaining = tensor.byteSize(),
            .io = io,
            .interface = .{
                .vtable = &.{
                    .stream = stream,
                    .discard = discard,
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

    fn stream(r: *std.Io.Reader, w: *std.Io.Writer, limit: std.Io.Limit) std.Io.Reader.StreamError!usize {
        const self: *TensorReader = @fieldParentPtr("interface", r);
        if (self.remaining == 0) return error.EndOfStream;

        const combined_limit = limit.min(.limited64(self.remaining));
        const n = try self.file_reader.interface.stream(w, combined_limit);
        self.remaining -= n;
        return n;
    }

    fn discard(r: *std.Io.Reader, limit: std.Io.Limit) std.Io.Reader.Error!usize {
        const self: *TensorReader = @fieldParentPtr("interface", r);
        if (self.remaining == 0) return error.EndOfStream;

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
    metadata: Metadatas,

    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator) TensorRegistry {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tensors = .{},
            .metadata = .{},
            .mutex = .{},
        };
    }

    pub fn initWithMetadata(
        allocator: std.mem.Allocator,
        metadata: Metadatas,
    ) !TensorRegistry {
        var self: TensorRegistry = .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tensors = .{},
            .metadata = .empty,
            .mutex = .{},
        };

        try self.mergeMetadata(metadata);

        return self;
    }

    pub fn deinit(self: *TensorRegistry) void {
        const allocator = self.arena.allocator();
        self.tensors.deinit(allocator);
        self.metadata.deinit(allocator);
        self.arena.deinit();
    }

    pub fn mergeMetadata(
        self: *TensorRegistry,
        other: Metadatas,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const allocator = self.arena.allocator();

        var it = other.iterator();
        while (it.next()) |entry| {
            const key = try allocator.dupe(u8, entry.key_ptr.*);
            const value = try entry.value_ptr.*.clone(allocator);

            const gop = try self.metadata.getOrPut(allocator, key);
            if (gop.found_existing) {
                gop.value_ptr.*.deinit(allocator);
                gop.value_ptr.* = value;
                log.warn("Overwrote existing metadata key={s} with value={f}", .{ key, value });
            } else {
                gop.value_ptr.* = value;
                log.debug("Added new metadata key={s} with value={f}", .{ key, value });
            }
        }
    }

    pub fn registerTensor(
        self: *TensorRegistry,
        tensor: Tensor,
    ) !void {
        const allocator = self.arena.allocator();

        self.mutex.lock();
        defer self.mutex.unlock();

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
    ) TensorReader.Error!TensorReader {
        const tensor = self.tensors.get(tensor_name) orelse {
            log.err("Tensor {s} not found in registry", .{tensor_name});
            return TensorReader.Error.TensorNotFound;
        };

        return try .init(io, tensor, buffer);
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
    file_uri: []const u8,
    reader: *std.Io.Reader,
    registry: *TensorRegistry,
) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const json_header_length: u64 = try reader.takeInt(u64, .little);

    const data_start_offset = BYTES_HEADER + json_header_length;

    const json_data = try arena.allocator().alloc(u8, @intCast(json_header_length));

    try reader.readSliceAll(json_data);

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

        if (std.mem.eql(u8, key, "__metadata__")) {
            // Parse metadata into a local arena to avoid concurrent allocations
            // against registry.arena (which is not thread-safe).
            var local_arena = std.heap.ArenaAllocator.init(allocator);
            defer local_arena.deinit();

            const metas = try parseMetadata(local_arena.allocator(), value);
            try registry.mergeMetadata(metas);

            continue;
        }

        const shape_field = value.object.get("shape").?.array;

        if (shape_field.items.len > Shape.MAX_RANK) {
            log.warn("Can't load tensor {s}, too many dims: {}", .{ key, shape_field.items.len });
            return error.TooManyDimensions;
        }

        const offset_field = value.object.get("data_offsets").?;
        const start: u64 = @intCast(offset_field.array.items[0].integer);
        const dtype = try stringToDtype(value.object.get("dtype").?.string);

        var dims: Dims = .{};
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

pub const SafetensorsIndex = struct {
    pub const Map = std.StringArrayHashMapUnmanaged(std.ArrayList([]const u8));

    arena: std.heap.ArenaAllocator,

    map: Map,
    metadata: Metadatas,

    pub fn init(allocator: std.mem.Allocator) SafetensorsIndex {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .map = .empty,
            .metadata = .empty,
        };
    }

    pub fn iterator(self: *SafetensorsIndex) Map.Iterator {
        return self.map.iterator();
    }

    pub fn deinit(self: *SafetensorsIndex) void {
        const allocator = self.arena.allocator();
        self.map.deinit(allocator);
        self.metadata.deinit(allocator);
        self.arena.deinit();
    }
};

pub fn parseSafetensorsIndex(
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
                map_entry.value_ptr.* = .{};
            }

            try map_entry.value_ptr.append(arena_allocator, try arena_allocator.dupe(u8, weight_name));
        }
    } else {
        log.warn("No weight_map attribute found in index", .{});
    }

    if (index.object.get("metadata")) |metadata_val| {
        safetensors_index.metadata = try parseMetadata(arena_allocator, metadata_val);
    }

    return safetensors_index;
}

pub fn parseSafetensorsIndexFiles(
    allocator: std.mem.Allocator,
    io: std.Io,
    safetensors_index: SafetensorsIndex,
    registry: *TensorRegistry,
    repo: std.Io.Dir,
    repo_path: []const u8,
) !void {
    var group: std.Io.Group = .init;
    defer group.cancel(io);

    var err: ?anyerror = null;

    const filenames = safetensors_index.map.keys();

    for (filenames) |filename| {
        group.async(io, AsyncParseSafetensorsIndexFile.run, .{
            allocator,
            io,
            registry,
            repo,
            repo_path,
            filename,
            &err,
        });
    }

    group.wait(io);

    if (err) |e| {
        log.err("Error parsing safetensors index files: {any}", .{e});
        return e;
    }
}

const AsyncParseSafetensorsIndexFile = struct {
    pub fn run(
        allocator: std.mem.Allocator,
        io: std.Io,
        registry: *TensorRegistry,
        repo: std.Io.Dir,
        repo_path: []const u8, // todo: Dir.realPath when ready
        filename: []const u8,
        err_ptr: *?anyerror,
    ) void {
        parse(allocator, io, registry, repo, repo_path, filename) catch |err| {
            err_ptr.* = err;
        };
    }

    fn parse(
        allocator: std.mem.Allocator,
        io: std.Io,
        registry: *TensorRegistry,
        repo: std.Io.Dir,
        repo_path: []const u8, // todo: Dir.realPath when ready
        filename: []const u8,
    ) !void {
        const file_uri = try std.fs.path.join(allocator, &.{ repo_path, filename });
        defer allocator.free(file_uri);

        const file = try repo.openFile(io, filename, .{});
        defer file.close(io);

        const header_buf = try allocator.alloc(u8, 64 * 1024);
        defer allocator.free(header_buf);

        var reader = file.reader(io, header_buf);

        try parseSafetensors(allocator, file_uri, &reader.interface, registry);
    }
};

pub const Metadata = union(enum) {
    null: void,
    int: i64,
    float: f64,
    bool: bool,
    string: []const u8,

    array_bool: []const bool,
    array_int: []const i64,
    array_float: []const f64,
    array_string: []const []const u8,

    pub const ItemType = enum {
        int,
        float,
        bool,
        string,

        pub fn toZigType(comptime kind: ItemType) type {
            return switch (kind) {
                .int => i64,
                .float => f64,
                .bool => bool,
                .string => []const u8,
            };
        }
    };

    pub fn wrap(x: anytype) Metadata {
        return switch (@TypeOf(x)) {
            inline u8, i8, u16, i16, u32, i32, u64, i64 => .{ .int = @intCast(x) },
            inline f16, f32, f64 => .{ .float = @floatCast(x) },
            bool => .{ .bool = x },
            []const u8 => .{ .string = x },
            else => @panic("Unsupported type for Value: " ++ @typeName(@TypeOf(x))),
        };
    }

    pub fn copySlice(allocator: std.mem.Allocator, any_slice: anytype) !Metadata {
        return switch (@TypeOf(any_slice[0])) {
            inline u8, i8, u16, i16, u32, i32, u64, i64 => {
                const res = try allocator.alloc(i64, any_slice.len);
                for (res, any_slice) |*r, val| r.* = @intCast(val);
                return .{ .array_int = res };
            },
            inline f16, f32, f64 => {
                const res = try allocator.alloc(f64, any_slice.len);
                for (res, any_slice) |*r, val| r.* = @floatCast(val);
                return .{ .array_float = res };
            },
            bool => .{ .array_bool = try allocator.dupe(bool, any_slice) },
            []const u8 => .{ .array_string = try allocator.dupe([]const u8, @alignCast(any_slice)) },
            else => @panic("Unsupported type for Value: " ++ @typeName(@TypeOf(any_slice))),
        };
    }

    pub fn format(
        self: Metadata,
        writer: *std.Io.Writer,
    ) !void {
        switch (self) {
            .null => _ = try writer.write("null"),
            .string => |s| try writer.print("{s}", .{s}),
            .bool => |b| try writer.print("{}", .{b}),
            .int => |i| try writer.print("{d}", .{i}),
            .float => |f| try writer.print("{d}", .{f}),
            .array_bool => |arr| {
                try writer.writeByte('[');
                for (arr, 0..) |v, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{}", .{v});
                }
                try writer.writeByte(']');
            },
            .array_int => |arr| {
                try writer.writeByte('[');
                for (arr, 0..) |v, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{d}", .{v});
                }
                try writer.writeByte(']');
            },
            .array_float => |arr| {
                try writer.writeByte('[');
                for (arr, 0..) |v, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{d}", .{v});
                }
                try writer.writeByte(']');
            },
            .array_string => |arr| {
                try writer.writeByte('[');
                for (arr, 0..) |v, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("\"{s}\"", .{v});
                }
                try writer.writeByte(']');
            },
        }
    }

    pub fn deinit(self: Metadata, allocator: std.mem.Allocator) void {
        switch (self) {
            .string => |s| allocator.free(s),
            .array_bool => |s| allocator.free(s),
            .array_int => |s| allocator.free(s),
            .array_float => |s| allocator.free(s),
            .array_string => |s| {
                for (s) |str| allocator.free(str);
                allocator.free(s);
            },
            else => {},
        }
    }

    pub fn clone(self: Metadata, allocator: std.mem.Allocator) !Metadata {
        return switch (self) {
            .null => .null,
            .int => |v| .{ .int = v },
            .float => |v| .{ .float = v },
            .bool => |v| .{ .bool = v },
            .string => |s| .{ .string = try allocator.dupe(u8, s) },
            .array_bool => |s| .{ .array_bool = try allocator.dupe(bool, s) },
            .array_int => |s| .{ .array_int = try allocator.dupe(i64, s) },
            .array_float => |s| .{ .array_float = try allocator.dupe(f64, s) },
            .array_string => |s| blk: {
                const new_slice = try allocator.alloc([]const u8, s.len);
                errdefer allocator.free(new_slice);
                for (s, 0..) |str, i| {
                    new_slice[i] = try allocator.dupe(u8, str);
                }
                break :blk .{ .array_string = new_slice };
            },
        };
    }
};

pub fn stringToDtype(safetensor_type: []const u8) !DataType {
    const map = std.StaticStringMap(DataType).initComptime(.{
        .{ "F64", .f64 },
        .{ "F32", .f32 },
        .{ "F16", .f16 },
        .{ "BF16", .bf16 },
        .{ "F8_E4M3", .f8e4m3fn },
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

fn parseMetadata(allocator: std.mem.Allocator, val: std.json.Value) !Metadatas {
    var metadatas: Metadatas = .{};
    var prefix_buf: [1024]u8 = undefined;
    var prefix = StringBuilder.initBuffer(&prefix_buf);

    try populateMetadata(allocator, &prefix, val, &metadatas);

    return metadatas;
}

fn populateMetadata(allocator: std.mem.Allocator, prefix: *StringBuilder, val: std.json.Value, metadatas: *Metadatas) !void {
    const key = prefix.items;
    return switch (val) {
        .null => try metadatas.put(allocator, try allocator.dupe(u8, key), .null),
        .bool => |v| try metadatas.put(allocator, try allocator.dupe(u8, key), .{ .bool = v }),
        .integer => |v| try metadatas.put(allocator, try allocator.dupe(u8, key), .{ .int = v }),
        .float => |v| try metadatas.put(allocator, try allocator.dupe(u8, key), .{ .float = v }),
        .number_string, .string => |v| try metadatas.put(allocator, try allocator.dupe(u8, key), .{ .string = try allocator.dupe(u8, v) }),
        .array => |v| {
            if (v.items.len == 0) return;
            if (validSlice(v)) |item_type| {
                const data: Metadata = switch (item_type) {
                    .bool => blk: {
                        const values = try allocator.alloc(bool, v.items.len);
                        for (v.items, 0..) |item, i| values[i] = item.bool;
                        break :blk .{ .array_bool = values };
                    },
                    .integer => blk: {
                        const values = try allocator.alloc(i64, v.items.len);
                        for (v.items, 0..) |item, i| values[i] = item.integer;
                        break :blk .{ .array_int = values };
                    },
                    .float => blk: {
                        const values = try allocator.alloc(f64, v.items.len);
                        for (v.items, 0..) |item, i| values[i] = item.float;
                        break :blk .{ .array_float = values };
                    },
                    inline .string, .number_string => |tag| blk: {
                        const values = try allocator.alloc([]const u8, v.items.len);
                        for (v.items, 0..) |item, i| {
                            values[i] = try allocator.dupe(u8, @field(item, @tagName(tag)));
                        }
                        break :blk .{ .array_string = values };
                    },
                    .null, .array, .object => unreachable,
                };
                try metadatas.put(allocator, try allocator.dupe(u8, key), data);
            } else {
                for (v.items, 0..) |item, i| {
                    const old_len = prefix.items.len;
                    if (prefix.items.len > 0) {
                        prefix.appendAssumeCapacity('.');
                    }
                    prefix.items.len += std.fmt.printInt(prefix.unusedCapacitySlice(), i, 10, .lower, .{});
                    try populateMetadata(allocator, prefix, item, metadatas);
                    prefix.items.len = old_len;
                }
            }
        },
        .object => |v| {
            var obj_iter = v.iterator();
            while (obj_iter.next()) |entry| {
                const old_len = prefix.items.len;
                if (prefix.items.len > 0) {
                    prefix.appendAssumeCapacity('.');
                }
                prefix.appendSliceAssumeCapacity(entry.key_ptr.*);
                try populateMetadata(allocator, prefix, entry.value_ptr.*, metadatas);
                prefix.items.len = old_len;
            }
        },
    };
}

/// We can only create a Zig slice out of json array, if all values
/// in the array have the same type.
fn validSlice(v: std.json.Array) ?std.meta.Tag(std.json.Value) {
    if (v.items.len == 0) return null;

    const item_type: std.meta.Tag(std.json.Value) = v.items[0];
    switch (item_type) {
        .null, .array, .object => return null,
        else => {},
    }

    for (v.items[1..]) |item| {
        if (item != item_type)
            return null;
    }

    return item_type;
}
