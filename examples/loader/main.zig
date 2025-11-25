// bazel run --@zml//runtimes:cpu=false --@zml//runtimes:cuda=true --run_under="sudo -E nsys profile -t cuda,syscall,nvtx,cublas,cublas-verbose,cusparse,cusparse-verbose,cudnn,osrt --inherit-environment=true --gpu-metrics-devices='cuda-visible' --cuda-memory-usage true --cuda-event-trace=false --backtrace=dwarf --cuda-graph-trace=node" //examples/loader -- /home/hugo/Llama-3.1-8B-Instruct/model.safetensors.index.json
// bazel run --@zml//runtimes:cpu=false --@zml//runtimes:cuda=true --run_under="sudo -E nsys profile -t cuda,syscall,nvtx,cublas,cublas-verbose,cusparse,cusparse-verbose,cudnn,osrt --inherit-environment=true --gpu-metrics-devices='cuda-visible' --cuda-memory-usage true --cuda-event-trace=false --backtrace=dwarf --cuda-graph-trace=node" //examples/loader:test

// Display system memory usage including buffered disk cache
// free -h
//
// Clear the filesystem cache to get accurate disk performance measurements
// sync && sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
//
// Benchmark write performance with 1MB block size and direct I/O, writing 10GB with a specific pattern
// fio --name=write \
// --ioengine=libaio \
// --rw=write \
// --bs=1MB \
// --direct=1 \
// --size=10G \
// --buffer_pattern="0xdeadbeef" \
// --filename=test_file
//
// Benchmark read performance with 100MB block size and direct I/O, reading 10GB
// fio --name=read \
// --ioengine=libaio \
// --rw=read \
// --bs=100MB \
// --direct=1 \
// --size=10G \
// --filename=test_file

const builtin = @import("builtin");
const std = @import("std");

const asynk = @import("async");
const stdx = @import("stdx");
const zml = @import("zml");

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
    .log_scope_levels = &.{
        .{ .scope = .@"zml/async", .level = .info },
    },
};

const log = std.log.scoped(.@"examples/loader");

const DataType = zml.DataType;
const Shape = zml.Shape;

const Dims = stdx.BoundedArray(i64, zml.Shape.MAX_RANK);
const StringBuilder = std.ArrayListUnmanaged(u8);

const Context = zml.Context;
const Platform = zml.Platform;
const Tracer = zml.tools.Tracer;
const pjrtx = zml.pjrt;
const pjrt = pjrtx.pjrt;

const KB = 1024;
const MB = 1024 * KB;

const BUF_1_KB = 1 * KB;
const BUF_4_KB = 4 * KB;
const BUF_8_KB = 8 * KB;
const BUF_16_KB = 16 * KB;
const BUF_32_KB = 32 * KB;
const BUF_64_KB = 64 * KB;

const BUF_1_MB = 1 * MB;
const BUF_4_MB = 4 * MB;
const BUF_8_MB = 8 * MB;
const BUF_16_MB = 16 * MB;
const BUF_32_MB = 32 * MB;
const BUF_64_MB = 64 * MB;
const BUF_128_MB = 128 * MB;
const BUF_256_MB = 256 * MB;

var tracer: Tracer = undefined;

// Utility to create a binary file with a simple byte pattern for testing.
fn createBinFile(tmp_dir: std.testing.TmpDir, filename: []const u8, size: usize, alignment: ?usize) !usize {
    var file = try tmp_dir.dir.createFile(filename, .{});
    defer file.close();

    var writer_buffer: [BUF_64_KB]u8 = undefined;
    var file_writer = file.writer(&writer_buffer);

    var pattern_chunk: [BUF_64_KB]u8 = undefined;
    for (&pattern_chunk, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }

    var remaining_bytes_to_write = if (alignment) |a| std.mem.alignForward(usize, size, a) else size;
    while (remaining_bytes_to_write > 0) {
        const chunk_len = @min(remaining_bytes_to_write, pattern_chunk.len);
        try file_writer.interface.writeAll(pattern_chunk[0..chunk_len]);
        remaining_bytes_to_write -= chunk_len;
    }
    try file_writer.interface.flush();

    return try file_writer.file.getEndPos();
}

// Warm up devices by allocating and freeing a 8MB buffer on each device which allocate the GPU BFC memory.
fn warmupDevices(allocator: std.mem.Allocator, platform: Platform, devices: []const *const pjrt.Device) !void {
    log.warn("Warming up devices (GPU bfc allocation)", .{});
    var timer = try std.time.Timer.start();

    for (devices) |device| {
        const trace = tracer.frameStart("Warmup device");
        defer tracer.frameEnd(trace, "Warmup device");

        const shape = Shape.init(.{BUF_8_MB / @sizeOf(f32)}, .f32);

        const warmup_data = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(warmup_data);

        const shard = Shard{
            .shape = shape,
            .tensor = .{
                .resource_uri = try .parse("file:///warmup_tensor"),
                .name = "tensor",
                .shape = shape,
                .offset = 0,
            },
            .device = device,
        };

        var writer: DeviceWriter = try .init(platform, shard, .device);
        defer writer.deinit();

        try writer.interface.writeAll(warmup_data);
        try writer.interface.flush();
    }

    const elapsed = timer.lap();
    log.warn("Warmed up {d} devices in {d}ms", .{ devices.len, elapsed / std.time.ns_per_ms });
}

pub const ResourceURI = std.Uri;
pub const ResourceIndexMap = std.ArrayHashMapUnmanaged(ResourceURI, std.ArrayList([]const u8), ResourceURIContext, false);
pub const Resources = std.ArrayHashMapUnmanaged(ResourceURI, Resource, ResourceURIContext, false);
pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);
pub const Metadatas = std.StringArrayHashMapUnmanaged(Metadata);

pub const ResourceType = enum {
    index,
    safetensors,
    unknown,
};

const ResourceURIContext = struct {
    // todo: implement full uri
    pub fn hash(_: ResourceURIContext, uri: ResourceURI) u32 {
        return std.array_hash_map.hashString(uri.path.percent_encoded);
    }
    pub fn eql(_: ResourceURIContext, a: ResourceURI, b: ResourceURI, _: usize) bool {
        return std.mem.eql(u8, a.path.percent_encoded, b.path.percent_encoded);
    }
};

const Tensor = struct {
    resource_uri: ResourceURI,
    name: []const u8,
    shape: Shape,
    offset: u64,

    pub fn byteSize(self: Tensor) u64 {
        return self.shape.byteSize();
    }
};

const Shard = struct {
    shape: Shape,
    tensor: Tensor,
    device: *const pjrt.Device,

    pub fn byteSize(self: Shard) u64 {
        return self.shape.byteSize();
    }
};

const ResourceIndex = struct {
    arena: std.heap.ArenaAllocator,

    map: ResourceIndexMap,
    metadata: Metadatas,

    pub fn init(allocator: std.mem.Allocator) ResourceIndex {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .map = .empty,
            .metadata = .empty,
        };
    }

    pub fn deinit(self: *ResourceIndex) void {
        const allocator = self.arena.allocator();
        self.map.deinit(allocator);
        self.metadata.deinit(allocator);
        self.arena.deinit();
    }
};

pub const TensorRegistry = struct {
    arena: std.heap.ArenaAllocator,

    tensors: Tensors,
    metadata: Metadatas,

    pub fn init(allocator: std.mem.Allocator) TensorRegistry {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tensors = .{},
            .metadata = .{},
        };
    }

    pub fn initWithMetadata(
        allocator: std.mem.Allocator,
        metadata: Metadatas,
    ) !TensorRegistry {
        var arena = std.heap.ArenaAllocator.init(allocator);

        return .{
            .arena = arena,
            .tensors = .{},
            .metadata = blk: {
                var arena_allocator = arena.allocator();
                var new_metadata: Metadatas = .{};

                var it = metadata.iterator();
                while (it.next()) |entry| {
                    const key = try arena_allocator.dupe(u8, entry.key_ptr.*);
                    const value = try entry.value_ptr.*.clone(arena_allocator);
                    try new_metadata.put(arena_allocator, key, value);
                }
                break :blk new_metadata;
            },
        };
    }

    pub fn deinit(self: *TensorRegistry) void {
        const allocator = self.arena.allocator();
        self.tensors.deinit(allocator);
        self.metadata.deinit(allocator);
        self.arena.deinit();
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

pub const ModelPathResolver = struct {
    arena: std.heap.ArenaAllocator,

    pub fn init(allocator: std.mem.Allocator) ModelPathResolver {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: *ModelPathResolver) void {
        self.arena.deinit();
    }

    pub fn resolve(self: *ModelPathResolver, path: []const u8) !ResourceURI {
        const arena_allocator = self.arena.allocator();

        const resource_uri: ResourceURI = ResourceURI.parse(path) catch .{
            .scheme = "file",
            .path = .{ .percent_encoded = path },
        };

        if (std.mem.eql(u8, resource_uri.scheme, FileResource.scheme)) {
            const stat = std.fs.cwd().statFile(resource_uri.path.percent_encoded) catch |err| {
                if (err == error.FileNotFound) {
                    log.warn("Path not found: {s}", .{resource_uri.path.percent_encoded});
                    return error.FileNotFound;
                }
                return err;
            };

            if (stat.kind == .file) {
                return resource_uri;
            } else if (stat.kind == .directory) {
                const index_path = try std.fs.path.join(arena_allocator, &[_][]const u8{ resource_uri.path.percent_encoded, "model.safetensors.index.json" });
                const model_path = try std.fs.path.join(arena_allocator, &[_][]const u8{ resource_uri.path.percent_encoded, "model.safetensors" });

                if (std.fs.cwd().statFile(index_path)) |_| {
                    return .{ .scheme = "file", .path = .{ .percent_encoded = index_path } };
                } else |_| {
                    if (std.fs.cwd().statFile(model_path)) |_| {
                        return .{ .scheme = "file", .path = .{ .percent_encoded = model_path } };
                    } else |_| {
                        return error.ModelFileNotFound;
                    }
                }
            } else {
                log.err("Path is neither a file nor a directory: {s}", .{resource_uri.path.percent_encoded});
                return error.InvalidPath;
            }
        } else {
            return resource_uri;
        }
    }

    pub fn resolveFromArgs(self: *ModelPathResolver, args: *std.process.ArgIterator) !ResourceURI {
        _ = args.next().?; // Skip program name

        const arg = args.next() orelse {
            log.err("No model path provided", .{});
            return error.NoModelPathProvided;
        };

        return self.resolve(arg);
    }
};

pub const MemoryResource = struct {
    pub const scheme = "memory";

    uri: ResourceURI,
    data: []const u8,

    pub fn init(data: []const u8) !MemoryResource {
        return .{
            .uri = try .parse("memory://memory"),
            .data = data,
        };
    }

    pub fn deinit(_: *MemoryResource) void {}
};

pub const FileResource = struct {
    pub const scheme = "file";

    uri: ResourceURI,
    file: std.fs.File,

    pub fn init(uri: ResourceURI) !FileResource {
        return .{
            .uri = uri,
            .file = try std.fs.openFileAbsolute(uri.path.percent_encoded, .{ .mode = .read_only }),
        };
    }

    pub fn reader(self: *FileResource, buffer: []u8) std.fs.File.Reader {
        return self.file.reader(buffer);
    }

    pub fn deinit(self: *FileResource) void {
        self.file.close();
    }
};

pub const S3Resource = struct {
    pub const scheme = "s3";

    allocator: std.mem.Allocator,
    uri: ResourceURI, // The original s3:// uri
    headers: std.ArrayList(std.http.Header),
    authenticator: ?AwsAuthenticator,

    request_url: std.Uri,
    request_url_str: []const u8, // Back URI

    pub fn init(allocator: std.mem.Allocator, uri: ResourceURI) !S3Resource {
        const bucket = uri.host orelse return error.InvalidS3Uri;

        const key = if (uri.path.percent_encoded.len > 0 and uri.path.percent_encoded[0] == '/')
            uri.path.percent_encoded[1..]
        else
            uri.path.percent_encoded;

        const url_str = try std.fmt.allocPrint(allocator, "http://127.0.0.1:9000/{s}/{s}", .{
            bucket.percent_encoded, key,
        });
        errdefer allocator.free(url_str);

        var self: S3Resource = .{
            .allocator = allocator,
            .uri = uri,
            .request_url_str = url_str,
            .headers = .{},
            .request_url = try .parse(url_str),
            .authenticator = null,
        };

        try self.headers.append(allocator, .{ .name = "Accept", .value = "*/*" });

        return self;
    }

    pub fn deinit(self: *S3Resource) void {
        self.headers.deinit(self.allocator);
        self.allocator.free(self.request_url_str);
    }
};

pub const Resource = union(enum) {
    memory: MemoryResource,
    file: FileResource,
    s3: S3Resource,

    pub fn init(allocator: std.mem.Allocator, resource_uri: ResourceURI) !Resource {
        if (std.mem.eql(u8, resource_uri.scheme, MemoryResource.scheme)) {
            return .{ .memory = try .init(&[_]u8{}) };
        } else if (std.mem.eql(u8, resource_uri.scheme, FileResource.scheme)) {
            return .{ .file = try .init(resource_uri) };
        } else if (std.mem.eql(u8, resource_uri.scheme, S3Resource.scheme)) {
            return .{ .s3 = try .init(allocator, resource_uri) };
        } else {
            return error.UnsupportedScheme;
        }
    }

    pub fn deinit(self: *Resource) void {
        switch (self.*) {
            .memory => |*m| m.deinit(),
            .file => |*f| f.deinit(),
            .s3 => |*s| s.deinit(),
        }
    }

    pub fn reader(self: *Resource, buffer: []u8, opts: IoReader.IoReaderOpts) !IoReader {
        return .init(self, buffer, opts);
    }

    pub fn uri(self: *Resource) ResourceURI {
        return switch (self.*) {
            inline else => |*r| r.uri,
        };
    }

    pub fn scheme(self: *Resource) []const u8 {
        return switch (self.*) {
            inline else => |r| @TypeOf(r).scheme,
        };
    }
};

pub const IoReader = struct {
    pub const IoReaderOpts = struct {
        offset: u64 = 0,
        use_direct_io: bool = false,
        use_aligned_reader: bool = false,
    };

    reader: union(enum) {
        memory: std.io.Reader,
        file: std.fs.File.Reader,
        s3: S3Reader,
    },
    resource: *Resource,
    opts: IoReaderOpts,

    pub fn init(resource: *Resource, buffer: []u8, opts: IoReaderOpts) !IoReader {
        return .{
            .reader = switch (resource.*) {
                .memory => |*m| .{ .memory = .fixed(m.data[opts.offset..]) },
                .file => |*f| blk: {
                    if (opts.use_direct_io) {
                        _ = try switchToDirectIO(f.file);
                    }
                    var file_reader = f.file.reader(buffer);
                    try file_reader.seekTo(opts.offset);
                    break :blk .{ .file = file_reader };
                },
                .s3 => |*s| .{ .s3 = try .init(s, buffer, opts.offset) },
            },
            .resource = resource,
            .opts = opts,
        };
    }

    pub fn deinit(self: *IoReader) void {
        switch (self.reader) {
            .memory => |_| {},
            .file => |_| {},
            .s3 => |*s| s.deinit(),
        }
    }

    pub fn interface(self: *IoReader) *std.io.Reader {
        return switch (self.reader) {
            .memory => |*m| m,
            .file => |*f| blk: {
                if (self.opts.use_aligned_reader) {
                    // todo: fragile
                    var aligned_reader = AlignedFileReader.init(f.*, .fromByteUnits(BUF_4_KB)) catch unreachable;
                    break :blk &aligned_reader.interface;
                } else {
                    break :blk &f.interface;
                }
            },
            .s3 => |*s| &s.interface,
        };
    }
};

fn resolveResourceType(resource: *Resource, reader_buffer: []u8) !ResourceType {
    var io_reader = try resource.reader(reader_buffer, .{});
    const reader = io_reader.interface();

    // Try to detect file type by examining its content
    var magic_bytes_buffer: [8]u8 = undefined;
    _ = reader.readSliceShort(&magic_bytes_buffer) catch |err| {
        if (err == error.EndOfStream) {
            return .unknown;
        }
        return err;
    };

    // Check if it's a JSON file by looking at the first bytes
    // JSON files typically start with '{' (0x7B) after optional whitespace
    if (magic_bytes_buffer[0] == '{') {
        // Read more content to check if it's an index file
        var json_header_buffer: [100]u8 = undefined;
        _ = reader.readSliceShort(&json_header_buffer) catch |err| {
            if (err == error.EndOfStream) {
                return .unknown;
            }
            return err;
        };

        // If it contains "weight_map", it's likely an index file
        if (std.mem.indexOf(u8, &json_header_buffer, "weight_map") != null) {
            return .index;
        }

        return .unknown;
    }

    // Safetensors files start with a u64 header length
    // Reinterpret first 8 bytes as little endian u64
    const magic_number = std.mem.readInt(u64, &magic_bytes_buffer, .little);

    // If the header is reasonable (< 10MB), it's likely a safetensors file
    if (magic_number > 0 and magic_number < 10 * 1024 * 1024) {
        return .safetensors;
    }

    return .unknown;
}

fn parseSafetensorsIndex(
    allocator: std.mem.Allocator,
    resource: *Resource,
    reader: *std.io.Reader,
) !ResourceIndex {
    const resource_uri = resource.uri();
    const path = resource_uri.path.percent_encoded;
    const basename = path[0 .. std.mem.lastIndexOfScalar(u8, path, '/').? + 1];

    var resource_index: ResourceIndex = .init(allocator);
    errdefer resource_index.deinit();

    const arena_allocator = resource_index.arena.allocator();

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

            const resource_path = try arena_allocator.alloc(u8, basename.len + filename.len);

            @memcpy(resource_path[0..basename.len], basename);
            @memcpy(resource_path[basename.len..], filename);

            const sibling_uri: ResourceURI = .{
                .scheme = resource_uri.scheme,
                .user = resource_uri.user,
                .password = resource_uri.password,
                .host = resource_uri.host,
                .port = resource_uri.port,
                .path = .{ .percent_encoded = resource_path },
            };

            const map_entry = try resource_index.map.getOrPut(arena_allocator, sibling_uri);

            if (!map_entry.found_existing) {
                map_entry.value_ptr.* = .{};
            }

            try map_entry.value_ptr.*.append(arena_allocator, try arena_allocator.dupe(u8, weight_name));
        }
    } else {
        log.warn("No weight_map attribute found in index", .{});
    }

    if (index.object.get("__metadata__")) |metadata_val| {
        resource_index.metadata = try parseMetadata(arena_allocator, metadata_val);
    }

    return resource_index;
}

fn parseSafetensors(
    registry: *TensorRegistry,
    resource_uri: ResourceURI,
    reader: *std.io.Reader,
) !void {
    var arena_allocator = registry.arena.allocator();

    const json_header_length: usize = @intCast(try reader.takeInt(u64, .little));
    const json_data = try arena_allocator.alloc(u8, json_header_length);
    defer arena_allocator.free(json_data);

    try reader.readSliceAll(json_data);

    const data_start_offset = 8 + json_header_length;
    const metadata_val = try std.json.parseFromSliceLeaky(std.json.Value, arena_allocator, json_data, .{});

    var it = metadata_val.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        if (std.mem.eql(u8, key, "__metadata__")) {
            registry.metadata = try parseMetadata(arena_allocator, value);
            continue;
        }

        const shape_field = value.object.get("shape").?.array;

        if (shape_field.items.len > Shape.MAX_RANK) {
            log.warn("Can't load tensor {s}, too many dims: {}", .{ key, shape_field.items.len });
            continue;
        }

        const offset_field = value.object.get("data_offsets").?;
        const start: u64 = @intCast(offset_field.array.items[0].integer);
        const end: u64 = @intCast(offset_field.array.items[1].integer);
        const dtype = try stringToDtype(value.object.get("dtype").?.string);

        var dims: Dims = .{};
        for (shape_field.items) |d| {
            dims.appendAssumeCapacity(d.integer);
        }

        const shape: Shape = .init(dims.constSlice(), dtype);
        const size_in_bytes = end - start;
        std.debug.assert(size_in_bytes == shape.byteSize());

        const tensor_name = try arena_allocator.dupe(u8, key);
        const tensor: Tensor = .{
            .resource_uri = resource_uri,
            .name = tensor_name,
            .shape = shape,
            .offset = data_start_offset + start,
        };

        try registry.tensors.put(arena_allocator, tensor_name, tensor);
    }
}

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
                    if (i > 0) try writer.write(", ");
                    try writer.print("{}", .{v});
                }
                try writer.writeByte(']');
            },
            .array_int => |arr| {
                try writer.writeByte('[');
                for (arr, 0..) |v, i| {
                    if (i > 0) try writer.write(", ");
                    try writer.print("{d}", .{v});
                }
                try writer.writeByte(']');
            },
            .array_float => |arr| {
                try writer.writeByte('[');
                for (arr, 0..) |v, i| {
                    if (i > 0) try writer.write(", ");
                    try writer.print("{d}", .{v});
                }
                try writer.writeByte(']');
            },
            .array_string => |arr| {
                try writer.writeByte('[');
                for (arr, 0..) |v, i| {
                    if (i > 0) try writer.write(", ");
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

fn stringToDtype(safetensor_type: []const u8) !DataType {
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

pub fn parseMetadata(allocator: std.mem.Allocator, val: std.json.Value) !Metadatas {
    var metadatas: Metadatas = .{};
    var prefix_buf: [BUF_1_KB]u8 = undefined;
    var prefix = StringBuilder.initBuffer(&prefix_buf);

    try populateMetadata(allocator, &prefix, val, &metadatas);

    return metadatas;
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

// Switch the given file descriptor to use buffered I/O mode.
fn switchToBufferedIO(file: std.fs.File) !bool {
    const fd = file.handle;

    const flags = try std.posix.fcntl(fd, std.posix.F.GETFL, 0);

    if (builtin.target.os.tag == .linux) {
        if (!@hasField(std.posix.O, "DIRECT")) {
            return true;
        }

        const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });
        if ((flags & direct_flag) == 0) {
            return true;
        }

        const result = try std.posix.fcntl(fd, std.posix.F.SETFL, flags & ~@as(c_uint, @bitCast(@as(u32, @intCast(direct_flag)))));
        return result == 0;
    } else if (builtin.target.os.tag == .macos) {
        if (!@hasField(std.posix.F, "NOCACHE")) {
            return true;
        }

        const nocache_flag: c_int = @bitCast(std.posix.F{ .NOCACHE = true });
        if ((flags & nocache_flag) == 0) {
            return true;
        }

        const result = try std.posix.fcntl(fd, std.posix.F.SETFL, flags & ~@as(c_uint, @bitCast(@as(u32, @intCast(nocache_flag)))));
        return result == 0;
    } else {
        return true;
    }
}

// Switch the given file descriptor to use direct I/O mode.
fn switchToDirectIO(file: std.fs.File) !bool {
    const fd = file.handle;

    const flags = try std.posix.fcntl(fd, std.posix.F.GETFL, 0);

    if (builtin.target.os.tag == .linux) {
        if (!@hasField(std.posix.O, "DIRECT")) {
            return false;
        }

        const direct_flag: c_int = @bitCast(std.posix.O{ .DIRECT = true });
        const result = try std.posix.fcntl(fd, std.posix.F.SETFL, flags | direct_flag);
        return result == 0;
    } else if (builtin.target.os.tag == .macos) {
        if (!@hasField(std.posix.F, "NOCACHE")) {
            return false;
        }

        const nocache_flag: c_int = @bitCast(std.posix.F{ .NOCACHE = true });
        const result = try std.posix.fcntl(fd, std.posix.F.SETFL, flags | nocache_flag);
        return result == 0;
    } else {
        return false;
    }
}

test "switchToDirectIO and switchToBufferedIO" {
    const allocator = std.testing.allocator;
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const filename = "file.bin";
    _ = try createBinFile(tmp_dir, filename, BUF_8_KB, null);

    const file_path = try tmp_dir.dir.realpathAlloc(allocator, filename);
    defer allocator.free(file_path);

    var file = try std.fs.openFileAbsolute(file_path, .{ .mode = .read_write });
    defer file.close();

    // Test buffered I/O
    var unaligned_buf: [10]u8 = undefined;
    try file.seekTo(1);
    const bytes_read_buffered = try file.readAll(&unaligned_buf);
    try std.testing.expectEqual(10, bytes_read_buffered);

    // Switch to Direct I/O and test aligned read
    _ = try switchToDirectIO(file);

    const blk_size = BUF_4_KB;
    const aligned_buf = try allocator.alignedAlloc(u8, .fromByteUnits(blk_size), blk_size * 2);
    defer allocator.free(aligned_buf);

    try file.seekTo(0);
    const bytes_read_aligned = try file.readAll(aligned_buf);
    try std.testing.expectEqual(aligned_buf.len, bytes_read_aligned);

    // Switch back to Buffered I/O and test unaligned read again
    _ = try switchToBufferedIO(file);

    try file.seekTo(1);
    const bytes_read_buffered_again = try file.readAll(&unaligned_buf);
    try std.testing.expectEqual(unaligned_buf.len, bytes_read_buffered_again);
}

const S3Reader = struct {
    resource: *S3Resource,
    http_client: std.http.Client,
    pos: u64,
    total_size: u64,

    interface: std.io.Reader,

    pub fn init(resource: *S3Resource, buffer: []u8, offset: u64) !S3Reader {
        var http_client: std.http.Client = .{ .allocator = resource.allocator };

        var request = try http_client.request(.HEAD, resource.request_url, .{
            .extra_headers = resource.headers.items,
        });

        const total_size = blk: {
            try request.sendBodiless();
            const response = try request.receiveHead(&.{});

            if (response.head.status == .ok) {
                const cl = response.head.content_length orelse {
                    return error.ReadFailed;
                };
                break :blk cl;
            } else {
                log.err("Failed to HEAD S3 resource {any}, status: {d}", .{ request.uri, @intFromEnum(response.head.status) });
                return error.ReadFailed;
            }
        };

        request.deinit();
        http_client.deinit();

        log.warn("S3 resource total size: {d} bytes", .{total_size});

        const self: S3Reader = .{
            .resource = resource,
            .http_client = .{ .allocator = resource.allocator },
            .pos = offset,
            .total_size = total_size,
            .interface = .{
                .vtable = &vtable,
                .buffer = buffer,
                .seek = 0,
                .end = 0,
            },
        };

        return self;
    }

    pub fn deinit(self: *S3Reader) void {
        self.http_client.deinit();
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const trace = tracer.frameStart("S3Reader.stream");
        defer tracer.frameEnd(trace, "S3Reader.stream");

        // This function's job is to refill `r.buffer`. The `std.io` machinery
        // will then copy from `r.buffer` to `w`, respecting `limit`.
        _ = w;
        _ = limit;

        const self = @as(*S3Reader, @alignCast(@fieldParentPtr("interface", r)));
        const resource = self.resource;

        // This function is only called when r.seek == r.end. Reset buffer state.
        r.seek = 0;
        r.end = 0;

        if (self.pos >= self.total_size) {
            return error.EndOfStream;
        }

        const range_start = self.pos;
        // Request a chunk up to our buffer's capacity, but not past the end of the file.
        const bytes_to_read = @min(@as(u64, @intCast(r.buffer.len)), self.total_size - range_start);

        if (bytes_to_read == 0) {
            return error.EndOfStream;
        }

        const range_end = range_start + bytes_to_read - 1;
        var range_header_buf: [128]u8 = undefined;

        const trace_prepareHeaders = tracer.frameStart("S3Reader.stream.prepareHeaders");
        var headers = self.resource.headers.clone(self.resource.allocator) catch |err| {
            log.err("Failed to clone S3 headers: {any}", .{err});
            return error.ReadFailed;
        };
        defer headers.deinit(self.resource.allocator);

        var auth_headers: ?AwsAuthenticatorHeaders = null;
        defer if (auth_headers) |*ah| {
            ah.deinit(self.resource.allocator);
        };

        if (self.resource.authenticator) |authenticator| {
            headers.ensureTotalCapacity(self.resource.allocator, headers.capacity + 3) catch |err| {
                log.err("Failed to ensure S3 headers capacity: {any}", .{err});
                return error.ReadFailed;
            };

            auth_headers = authenticator.generateHeadersValues(self.resource.allocator, .GET, resource.request_url.path.percent_encoded) catch |err| {
                log.err("Failed to sign S3 GET request headers: {any}", .{err});
                return error.ReadFailed;
            };

            headers.appendAssumeCapacity(.{ .name = "Date", .value = auth_headers.?.date });
            headers.appendAssumeCapacity(.{ .name = "Authorization", .value = auth_headers.?.authorization });
        } else {
            headers.ensureTotalCapacity(self.resource.allocator, headers.capacity + 1) catch |err| {
                log.err("Failed to ensure S3 headers capacity: {any}", .{err});
                return error.ReadFailed;
            };

            log.info("No authenticator provided, proceeding without authentication", .{});
        }

        headers.appendAssumeCapacity(.{
            .name = "Range",
            .value = std.fmt.bufPrint(&range_header_buf, "bytes={d}-{d}", .{ range_start, range_end }) catch |err| {
                log.err("Failed to format Range header: {any}", .{err});
                return error.ReadFailed;
            },
        });
        tracer.frameEnd(trace_prepareHeaders, "S3Reader.stream.prepareHeaders");

        const trace_request = tracer.frameStart("S3Reader.stream.request");

        var request = self.http_client.request(.GET, resource.request_url, .{
            .extra_headers = headers.items,
        }) catch |err| {
            log.err("Failed to create S3 GET request: {any}", .{err});
            return error.ReadFailed;
        };
        defer request.deinit();

        tracer.frameEnd(trace_request, "S3Reader.stream.request");

        const trace_sendBodiless = tracer.frameStart("S3Reader.stream.sendBodiless");
        try request.sendBodiless();
        tracer.frameEnd(trace_sendBodiless, "S3Reader.stream.sendBodiless");

        const trace_receiveHead = tracer.frameStart("S3Reader.stream.receiveHead");
        var response = request.receiveHead(&.{}) catch |err| {
            log.err("Failed to receive S3 GET response: {any}", .{err});
            return error.ReadFailed;
        };
        tracer.frameEnd(trace_receiveHead, "S3Reader.stream.receiveHead");

        if (response.head.status != .partial_content and response.head.status != .ok) {
            log.err("S3 GET request to {any} failed with status: {s} (expected 206 or 200)", .{ resource.request_url, @tagName(response.head.status) });
            return error.ReadFailed;
        }

        var body_reader = response.reader(&.{});

        const trace_read = tracer.frameStart("S3Reader.stream.readBody");
        const n = try body_reader.readSliceShort(r.buffer[0..bytes_to_read]);
        tracer.frameEnd(trace_read, "S3Reader.stream.readBody");

        r.end = n;
        self.pos += n;

        log.debug("S3Reader: range: bytes={d}-{d}, status={s}, content length={any}", .{ range_start, range_end, @tagName(response.head.status), response.head.content_length });

        // Return 0 because we did not write to `w`. The caller will now
        // read from our freshly filled buffer.
        return 0;
    }

    const vtable: std.io.Reader.VTable = .{
        .stream = stream,
    };
};

const AwsAuthenticatorHeaders = struct {
    date: []const u8,
    authorization: []const u8,

    pub fn deinit(self: *AwsAuthenticatorHeaders, allocator: std.mem.Allocator) void {
        allocator.free(self.date);
        allocator.free(self.authorization);
    }
};

const AwsAuthenticator = struct {
    allocator: std.mem.Allocator,
    access_key: []const u8,
    secret_key: []const u8,

    pub fn init(allocator: std.mem.Allocator) !AwsAuthenticator {
        const access_key = std.process.getEnvVarOwned(allocator, "AWS_ACCESS_KEY_ID") catch |err| {
            log.warn("AWS_ACCESS_KEY_ID not set, proceeding without authentication: {any}", .{err});
            return error.AwsCredentialsNotFound;
        };

        const secret_key = std.process.getEnvVarOwned(allocator, "AWS_SECRET_ACCESS_KEY") catch |err| {
            log.warn("AWS_SECRET_ACCESS_KEY not set, proceeding without authentication: {any}", .{err});
            allocator.free(access_key);
            return error.AwsCredentialsNotFound;
        };

        return .{
            .allocator = allocator,
            .access_key = access_key,
            .secret_key = secret_key,
        };
    }

    pub fn deinit(self: *AwsAuthenticator) void {
        self.allocator.free(self.access_key);
        self.allocator.free(self.secret_key);
    }

    /// Signs the given headers list for a request.
    /// Appends `Date` and `Authorization` headers.
    pub fn generateHeadersValues(
        self: *const AwsAuthenticator,
        allocator: std.mem.Allocator,
        method: std.http.Method,
        resource_path: []const u8,
    ) !AwsAuthenticatorHeaders {
        // Format according to RFC 1123: "%a, %d %b %Y %H:%M:%S GMT"
        const now = std.time.timestamp();
        const epoch_secs: std.time.epoch.EpochSeconds = .{ .secs = @intCast(now) };
        const year_day = epoch_secs.getEpochDay().calculateYearDay();
        const tm = year_day.calculateMonthDay();

        var date_buffer: [32]u8 = undefined;
        const weekday_names = [_][]const u8{ "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" };
        const month_names = [_][]const u8{ "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };

        const weekday = epoch_secs.getEpochDay();
        const weekday_name = weekday_names[@mod(weekday.day, 7)];
        const month_name = month_names[tm.month.numeric() - 1];

        const hours = @mod(@divTrunc(now, 3600), 24);
        const minutes = @mod(@divTrunc(now, 60), 60);
        const seconds = @mod(now, 60);

        const date = try std.fmt.bufPrint(&date_buffer, "{s}, {d:0>2} {s} {d} {d:0>2}:{d:0>2}:{d:0>2} GMT", .{
            weekday_name,
            tm.day_index + 1,
            month_name,
            year_day.year,
            @as(u32, @intCast(hours)),
            @as(u32, @intCast(minutes)),
            @as(u32, @intCast(seconds)),
        });

        // StringToSign = HTTP-Verb + "\n" +
        //                Content-MD5 + "\n" +
        //                Content-Type + "\n" +
        //                Date + "\n" +
        //                CanonicalizedAmzHeaders +
        //                CanonicalizedResource;
        // For a simple GET, MD5 and Type are empty, and we have no x-amz- headers.
        const string_to_sign = try std.fmt.allocPrint(allocator, "{s}\n\n\n{s}\n{s}", .{
            @tagName(method),
            date,
            resource_path,
        });
        defer allocator.free(string_to_sign);

        var hmac: std.crypto.auth.hmac.HmacSha1 = .init(self.secret_key);
        hmac.update(string_to_sign);
        var signature_bytes: [std.crypto.hash.Sha1.digest_length]u8 = undefined;
        hmac.final(&signature_bytes);

        var b64_buffer: [std.base64.standard.Encoder.calcSize(signature_bytes.len)]u8 = undefined;
        const signature_b64 = std.base64.standard.Encoder.encode(&b64_buffer, &signature_bytes);

        const authorization = try std.fmt.allocPrint(allocator, "AWS {s}:{s}", .{
            self.access_key,
            signature_b64,
        });

        return .{
            .date = try allocator.dupe(u8, date),
            .authorization = authorization,
        };
    }
};

test AwsAuthenticator {
    const allocator = std.testing.allocator;
    var authenticator: AwsAuthenticator = try .init(allocator);
    defer authenticator.deinit();

    var headers = try authenticator.generateHeadersValues(allocator, .GET, "/my-bucket/my-key");
    defer headers.deinit(allocator);

    std.debug.print("Date: {s}\n", .{headers.date});
    std.debug.print("Authorization: {s}\n", .{headers.authorization});
}

test "S3Reader: stream" {
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();
    _ = platform; // autofix

    tracer = Tracer.init("ai.zml.test.S3Reader.stream");

    const offset: u64 = 16;
    const s3_uri: ResourceURI = try .parse("s3://models/meta-llama/Llama-3.1-8B/model-00002-of-00004.safetensors");

    var authenticator: AwsAuthenticator = try .init(allocator);
    defer authenticator.deinit();

    var resource: S3Resource = try .init(allocator, s3_uri);
    resource.authenticator = authenticator;

    const reader_buffer = try allocator.alloc(u8, BUF_128_MB);
    defer allocator.free(reader_buffer);

    var s3_reader: S3Reader = try .init(&resource, reader_buffer, offset);
    defer {
        s3_reader.deinit();
        resource.deinit();
    }

    const expected_bytes_read: u64 = s3_reader.total_size - offset;

    var writer_impl: std.io.Writer.Discarding = .init(&.{});

    // var writer_impl: std.io.Writer.Allocating = try .initCapacity(allocator, expected_bytes_read);
    // defer writer_impl.deinit();

    var limited_reader = std.io.Reader.Limited.init(&s3_reader.interface, .limited(s3_reader.total_size), &.{});

    var timer: std.time.Timer = try .start();
    const bytes_read = try limited_reader.interface.streamRemaining(&writer_impl.writer);

    const elapsed = timer.read();
    const gb_read = @as(f64, @floatFromInt(bytes_read)) / (1.0 * 1024 * 1024 * 1024);
    const read_rate = if (elapsed > 0) gb_read / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0) else 0;
    log.warn("All tensors loaded in {d}ms ({d:.2} GB read at {d:.2} GB/s)", .{ elapsed / std.time.ns_per_ms, gb_read, read_rate });

    try std.testing.expectEqual(expected_bytes_read, bytes_read);
}

const AlignedFileReader = struct {
    reader: std.fs.File.Reader,
    alignment: std.mem.Alignment,
    pos: u64, // Physical offset for the next pread
    file_size: u64,

    buffer: [BUF_8_KB]u8 align(BUF_8_KB), // Hard coded for slow path
    buffer_valid_len: usize, // Total bytes read into buffer
    buffer_consumed: usize, // Bytes consumed/skipped from buffer head

    interface: std.io.Reader,

    pub fn init(reader: std.fs.File.Reader, alignment: std.mem.Alignment) !AlignedFileReader {
        const trace = tracer.frameStart("AlignedFileReader.init");
        defer tracer.frameEnd(trace, "AlignedFileReader.init");

        const file_size = try reader.file.getEndPos();
        const alignment_bytes = alignment.toByteUnits();
        const initial_pos = reader.pos;

        var current_pos = initial_pos;
        var consumed: usize = 0;

        if (initial_pos % alignment_bytes != 0) {
            const unaligned_head = initial_pos % alignment_bytes;
            current_pos = initial_pos - unaligned_head;
            consumed = @intCast(unaligned_head);
        }

        return .{
            .reader = reader,
            .alignment = alignment,
            .pos = current_pos,
            .file_size = file_size,
            .buffer = undefined,
            .buffer_valid_len = 0,
            .buffer_consumed = consumed,
            .interface = .{ .vtable = &vtable, .buffer = &.{}, .seek = 0, .end = 0 },
        };
    }

    fn streamFromInternalBuffer(self: *AlignedFileReader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const available = self.buffer[self.buffer_consumed..self.buffer_valid_len];
        if (available.len == 0) return 0;

        const copy_len = limit.minInt(available.len);
        const written = try w.write(available[0..copy_len]);

        self.buffer_consumed += written;
        return written;
    }

    fn loadAlignedBlockToInternal(self: *AlignedFileReader) std.io.Reader.StreamError!void {
        self.buffer_valid_len = 0;
        const alignment_bytes = self.alignment.toByteUnits();

        if (self.pos >= self.file_size) return error.EndOfStream;

        const bytes_read = self.reader.file.pread(self.buffer[0..alignment_bytes], self.pos) catch |e| {
            log.err("File pread error for aligned block at offset {d}: {}", .{ self.pos, e });
            return error.ReadFailed;
        };

        if (bytes_read == 0) return error.EndOfStream;

        self.buffer_valid_len = bytes_read;
        self.pos += alignment_bytes;
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self = @as(*AlignedFileReader, @alignCast(@fieldParentPtr("interface", r)));
        std.debug.assert(r.seek == 0 and r.end == 0);

        if (limit == .nothing) return error.EndOfStream;

        if (self.buffer_consumed > 0 and self.buffer_valid_len == 0) {
            try self.loadAlignedBlockToInternal();
            return self.streamFromInternalBuffer(w, limit);
        }

        if (self.buffer_consumed < self.buffer_valid_len) {
            return self.streamFromInternalBuffer(w, limit);
        }

        self.buffer_consumed = 0;
        self.buffer_valid_len = 0;

        if (self.pos >= self.file_size) return error.EndOfStream;

        const alignment_bytes = self.alignment.toByteUnits();
        const logical_data_remaining = limit.toInt() orelse (self.file_size - self.pos);

        if (w.writableSliceGreedy(alignment_bytes)) |dest_buffer| {
            if (self.pos % alignment_bytes == 0) {
                const max_read_size = @min(dest_buffer.len, self.file_size - self.pos);
                const max_aligned_read = std.mem.alignBackward(usize, @min(max_read_size, logical_data_remaining), alignment_bytes);

                if (max_aligned_read > 0) {
                    const bytes_read = self.reader.file.pread(dest_buffer[0..max_aligned_read], self.pos) catch |e| {
                        log.err("File pread error (fast path): {}", .{e});
                        return error.ReadFailed;
                    };
                    if (bytes_read == 0) return error.EndOfStream;

                    w.advance(bytes_read);
                    self.pos += bytes_read;
                    return bytes_read;
                }
            }
        } else |err| {
            log.debug("Writer cannot provide slice for fast path: {}. Falling back to slow path.", .{err});
        }

        try self.loadAlignedBlockToInternal();
        return self.streamFromInternalBuffer(w, limit);
    }

    const vtable: std.io.Reader.VTable = .{
        .stream = stream,
    };
};

const TestAlignedFileReader = struct {
    allocator: std.mem.Allocator,
    tmp_dir: std.testing.TmpDir,
    platform: Platform,
    o_direct_file: std.fs.File,
    reference_file: std.fs.File,
    params: Params,

    pub const Params = struct {
        file_size: usize,
        alignment: usize,
        read_offset: u64 = 0,
        read_size: usize,
    };

    pub fn init(allocator: std.mem.Allocator, params: Params) !TestAlignedFileReader {
        const platform: Platform = zml.testing.env();

        var tmp_dir = std.testing.tmpDir(.{});
        errdefer tmp_dir.cleanup();

        const filename = "test.bin";
        const actual_file_size = try createBinFile(tmp_dir, filename, params.file_size, params.alignment);

        const file_path = try tmp_dir.dir.realpathAlloc(allocator, filename);
        defer allocator.free(file_path);

        const o_direct_fd = try std.posix.open(
            file_path,
            .{ .ACCMODE = .RDONLY, .DIRECT = true },
            0,
        );
        const o_direct_file = std.fs.File{ .handle = o_direct_fd };

        const reference_file = try std.fs.openFileAbsolute(file_path, .{ .mode = .read_only });

        var new_params = params;
        new_params.file_size = actual_file_size;

        return .{
            .allocator = allocator,
            .tmp_dir = tmp_dir,
            .platform = platform,
            .o_direct_file = o_direct_file,
            .reference_file = reference_file,
            .params = new_params,
        };
    }

    pub fn deinit(self: *TestAlignedFileReader) void {
        self.o_direct_file.close();
        self.reference_file.close();
        self.tmp_dir.cleanup();
    }

    pub fn runReadExact(self: *TestAlignedFileReader, comptime dest_buffer_alignment: ?usize) !void {
        var reader_init_buf: [1]u8 = undefined;
        var file_reader = self.o_direct_file.reader(&reader_init_buf);
        try file_reader.seekTo(self.params.read_offset);

        var reader: AlignedFileReader = try .init(file_reader, .fromByteUnits(self.params.alignment));

        const dest_buffer = if (dest_buffer_alignment) |alignment|
            try self.allocator.alignedAlloc(u8, .fromByteUnits(alignment), self.params.read_size)
        else
            try self.allocator.alloc(u8, self.params.read_size);
        defer self.allocator.free(dest_buffer);

        var writer: std.io.Writer = .fixed(dest_buffer);

        var total_read: usize = 0;
        while (total_read < self.params.read_size) {
            const n = reader.interface.stream(&writer, .limited(self.params.read_size - total_read)) catch |err| switch (err) {
                error.EndOfStream => break,
                else => |e| return e,
            };

            if (n == 0) break;

            total_read += n;
        }

        const expected_read_size = @min(self.params.read_size, self.params.file_size - @min(self.params.read_offset, self.params.file_size));
        try std.testing.expectEqual(expected_read_size, total_read);

        try self.reference_file.seekTo(self.params.read_offset);

        const expected_content = try self.allocator.alloc(u8, expected_read_size);
        defer self.allocator.free(expected_content);

        if (expected_read_size > 0) {
            _ = try self.reference_file.readAll(expected_content);
        }

        try std.testing.expectEqualSlices(u8, expected_content, dest_buffer[0..total_read]);
    }
};

test "AlignedFileReader: read from aligned offset" {
    var scenario: TestAlignedFileReader = try .init(std.testing.allocator, .{
        .file_size = BUF_1_MB,
        .alignment = BUF_4_KB,
        .read_size = 10 * KB,
    });
    defer scenario.deinit();

    try scenario.runReadExact(null);
}

// todo: implement unaligned read offset
// test "AlignedFileReader: read from unaligned offset" {
//     var scenario: TestAlignedFileReader = try .init(std.testing.allocator, .{
//         .file_size = BUF_1_MB,
//         .alignment = BUF_4_KB,
//         .read_offset = 123,
//         .read_size = 10 * KB,
//     });
//     defer scenario.deinit();

//     try scenario.runReadExact(null);
// }

test "AlignedFileReader: fast path direct read" {
    var scenario: TestAlignedFileReader = try .init(std.testing.allocator, .{
        .file_size = BUF_1_MB,
        .alignment = BUF_4_KB,
        .read_size = 64 * KB, // Must be multiple of alignment
    });
    defer scenario.deinit();

    // Destination buffer must be aligned for fast path
    try scenario.runReadExact(BUF_4_KB);
}

test "AlignedFileReader: mixed path (fast then slow)" {
    var scenario: TestAlignedFileReader = try .init(std.testing.allocator, .{
        .file_size = BUF_1_MB,
        .alignment = BUF_4_KB,
        .read_size = 64 * KB + 123, // Not a multiple of alignment
    });
    defer scenario.deinit();

    // Aligned buffer to trigger at least one fast path read
    try scenario.runReadExact(BUF_4_KB);
}

test "AlignedFileReader: file smaller than alignment" {
    var scenario: TestAlignedFileReader = try .init(std.testing.allocator, .{
        .file_size = 1 * KB,
        .alignment = BUF_4_KB,
        .read_size = 1 * KB,
    });
    defer scenario.deinit();

    try scenario.runReadExact(null);
}

// todo: implement unaligned read offset
// test "AlignedFileReader: read exact file size from unaligned offset" {
//     var scenario: TestAlignedFileReader = try .init(std.testing.allocator, .{
//         .file_size = 10 * KB + 5,
//         .alignment = BUF_4_KB,
//         .read_offset = 1,
//         .read_size = 10 * KB + 4,
//     });
//     defer scenario.deinit();

//     try scenario.runReadExact(null);
// }

test "AlignedFileReader: read until end of stream" {
    const allocator = std.testing.allocator;
    const file_size = 10 * KB + 5;
    const alignment = BUF_4_KB;

    var scenario: TestAlignedFileReader = try .init(allocator, .{
        .file_size = file_size,
        .alignment = alignment,
        .read_size = file_size,
    });
    defer scenario.deinit();

    var reader_init_buf: [1]u8 = undefined;
    const file_reader = scenario.o_direct_file.reader(&reader_init_buf);
    var reader = try AlignedFileReader.init(file_reader, .fromByteUnits(alignment));

    var writer = std.io.Writer.Allocating.init(allocator);
    defer writer.deinit();

    const bytes_read = try reader.interface.streamRemaining(&writer.writer);

    const expected_physical_size = std.mem.alignForward(usize, file_size, alignment);
    try std.testing.expectEqual(expected_physical_size, bytes_read);

    // Verify
    try scenario.reference_file.seekTo(0);

    const expected_content = try scenario.reference_file.readToEndAlloc(allocator, expected_physical_size + 1);
    defer allocator.free(expected_content);

    try std.testing.expectEqualSlices(u8, expected_content, writer.writer.buffered());
}

const DeviceWriterBuffered = struct {
    platform: Platform,
    shard: Shard,
    transfer_manager: *pjrtx.AsyncHostToDeviceTransferManager,
    pjrt_buffer: *pjrtx.Buffer,

    dma_buffer: []u8,
    chunk_size: usize,

    events: [2]?*pjrtx.Event,
    current_buffer_idx: u1,

    bytes_written: u64,
    can_process_last_event: bool,

    interface: std.io.Writer,

    pub fn init(platform: Platform, shard: Shard, buffer: []u8) !DeviceWriterBuffered {
        const trace = tracer.frameStart("DeviceWriterBuffered.init");
        defer tracer.frameEnd(trace, "DeviceWriterBuffered.init");

        std.debug.assert(buffer.len % 2 == 0);
        const chunk_size = buffer.len / 2;

        const memories = shard.device.addressableMemories(platform.pjrt_api);
        var memory = memories[0];

        if (platform.target == .cuda) {
            for (memories) |mem| {
                if (mem.kind(platform.pjrt_api) == .device) {
                    memory = mem;
                    break;
                }
            }
        }

        const shape_spec = pjrt.ShapeSpec.init(shard.shape.dims(), bufferTypeFromDtype(shard.shape.dtype()));

        const transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
            .shape_specs = &.{shape_spec},
            .memory = memory,
        });

        return .{
            .platform = platform,
            .shard = shard,
            .transfer_manager = transfer_manager,
            .pjrt_buffer = try transfer_manager.retrieveBuffer(platform.pjrt_api, 0),
            .dma_buffer = buffer,
            .chunk_size = chunk_size,
            .events = .{ null, null },
            .current_buffer_idx = 0,
            .bytes_written = 0,
            .can_process_last_event = true,
            .interface = .{ .vtable = &vtable, .buffer = buffer[0..chunk_size], .end = 0 },
        };
    }

    pub fn deinit(self: *DeviceWriterBuffered) void {
        self.pjrt_buffer.deinit(self.platform.pjrt_api);
        self.transfer_manager.deinit(self.platform.pjrt_api);
        self.transfer_manager = undefined;
    }

    fn awaitEvent(self: *DeviceWriterBuffered, idx: u1) !void {
        if (self.events[idx]) |event| {
            const trace_await = tracer.frameStart("DeviceWriterBuffered.awaitEvent");
            defer tracer.frameEnd(trace_await, "DeviceWriterBuffered.awaitEvent");
            try event.awaitBlocking(self.platform.pjrt_api);
            self.events[idx] = null;
        }
    }

    fn swap(self: *DeviceWriterBuffered) void {
        log.debug("Swapping buffers: {d} -> {d}", .{ self.current_buffer_idx, 1 - self.current_buffer_idx });
        self.current_buffer_idx = 1 - self.current_buffer_idx;
        const new_offset = self.current_buffer_idx * self.chunk_size;
        self.interface.buffer = self.dma_buffer[new_offset..][0..self.chunk_size];
        self.interface.end = 0;
        log.debug("  Current buffer index: {d}, buffer range: [{}..{}]", .{ self.current_buffer_idx, new_offset, new_offset + self.chunk_size });
    }

    fn awaitTransfer(self: *DeviceWriterBuffered) !void {
        const idx = 1 - self.current_buffer_idx;
        log.debug("Awaiting transfer on buffer index: {d}", .{idx});
        try self.awaitEvent(idx);
    }

    fn transfer(self: *DeviceWriterBuffered, data: []const u8) !*pjrtx.Event {
        defer self.bytes_written += data.len;
        log.debug("Transferring {d} bytes at offset {d}", .{ data.len, self.bytes_written });
        return self.transfer_manager.transferData(self.platform.pjrt_api, 0, data, @intCast(self.bytes_written), false) catch |err| {
            log.err("Error during transferData in drain: {}", .{err});
            return error.WriteFailed;
        };
    }

    fn transferAndSwap(self: *DeviceWriterBuffered) !usize {
        try self.awaitTransfer();
        const len = self.interface.buffered().len;
        self.events[self.current_buffer_idx] = self.transfer(self.interface.buffered()) catch |err| {
            log.err("Error during transferAndSwap: {}", .{err});
            return err;
        };
        self.swap();
        return len;
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const self = @as(*DeviceWriterBuffered, @alignCast(@fieldParentPtr("interface", w)));
        std.debug.assert(splat == 1);

        log.debug("DeviceWriterBuffered.drain with buffer ptr=0x{x} called with {d} slices", .{ @intFromPtr(w.buffer.ptr), data.len });
        log.debug("  Current buffer state: idx={d}, end={d}, capacity={d}, buffered={d}", .{ self.current_buffer_idx, w.end, w.buffer.len, w.buffered().len });
        log.debug("  Bytes written so far: {d}/{d}", .{ self.bytes_written, self.shard.byteSize() });

        for (data, 0..) |slice, i| {
            log.debug("  Slice[{d}]: ptr=0x{x}, len={d}", .{ i, @intFromPtr(slice.ptr), slice.len });
        }

        const trace = tracer.frameStart("DeviceWriterBuffered.drain (flip&wait)");
        defer tracer.frameEnd(trace, "DeviceWriterBuffered.drain (flip&wait)");

        _ = self.transferAndSwap() catch |err| {
            log.err("Error during transferAndSwap in drain: {}", .{err});
            return error.WriteFailed;
        };

        return 0;
    }

    fn flush(w: *std.io.Writer) std.io.Writer.Error!void {
        const trace = tracer.frameStart("DeviceWriterBuffered.flush");
        defer tracer.frameEnd(trace, "DeviceWriterBuffered.flush");
        log.debug("DeviceWriterBuffered.flush called", .{});

        const self = @as(*DeviceWriterBuffered, @alignCast(@fieldParentPtr("interface", w)));

        if (w.end > 0) {
            const current_chunk = w.buffered();
            const event = self.transfer_manager.transferData(self.platform.pjrt_api, 0, current_chunk, @intCast(self.bytes_written), false) catch |err| {
                log.err("Error during flush transfer: {}", .{err});
                return error.WriteFailed;
            };
            self.events[self.current_buffer_idx] = event;
            self.bytes_written += current_chunk.len;
            w.end = 0;
        }

        self.awaitEvent(0) catch |err| {
            log.err("Error awaiting event 0 in flush: {}", .{err});
            return error.WriteFailed;
        };

        self.awaitEvent(1) catch |err| {
            log.err("Error awaiting event 1 in flush: {}", .{err});
            return error.WriteFailed;
        };

        if (self.can_process_last_event) {
            log.debug("Finalizing transfer of {d} bytes (total expected: {d})", .{ self.bytes_written, self.shard.byteSize() });

            std.debug.assert(self.bytes_written == self.shard.byteSize());

            const last_event = self.transfer_manager.transferData(self.platform.pjrt_api, 0, &.{}, @intCast(self.bytes_written), true) catch |err| {
                log.err("Error during final transferData: {}", .{err});
                return error.WriteFailed;
            };
            last_event.awaitBlocking(self.platform.pjrt_api) catch |err| {
                log.err("Error during final awaitBlocking: {}", .{err});
                return error.WriteFailed;
            };

            self.can_process_last_event = false;
        }
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
    };
};

test "DeviceWriterBuffered: write" {
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    tracer = Tracer.init("ai.zml.test.DeviceWriter.write");
    const trace = tracer.frameStart("DeviceWriter.test.write");
    defer tracer.frameEnd(trace, "DeviceWriter.test.write");

    const device = platform.getDevices()[0];

    try warmupDevices(allocator, platform, &.{device});

    const tensor_size = 4 * BUF_256_MB;
    const dma_buffer_size = 64 * MB;
    const write_chunk_size = dma_buffer_size / 4;

    const shape: Shape = .init(.{tensor_size / @sizeOf(f32)}, .f32);
    const shard: Shard = .{
        .shape = shape,
        .tensor = .{
            .resource_uri = try .parse("file:///model.safetensors"),
            .name = "tensor",
            .shape = shape,
            .offset = 0,
        },
        .device = device,
    };

    const writer_buffer = try allocator.alloc(u8, dma_buffer_size);
    defer allocator.free(writer_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, writer_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, writer_buffer) catch {};

    var writer: DeviceWriterBuffered = try .init(platform, shard, writer_buffer);
    defer writer.deinit();

    const data = try allocator.alloc(u8, tensor_size);
    defer allocator.free(data);

    for (data, 0..) |*byte, i| byte.* = @intCast(i % 256);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, data);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, data) catch {};

    var total_bytes_written: u64 = 0;
    while (total_bytes_written < tensor_size) {
        const remaining = tensor_size - total_bytes_written;
        const chunk_size = @min(remaining, write_chunk_size);
        const chunk = data[total_bytes_written..][0..chunk_size];
        log.debug("Writing chunk {*} of size {d}, total_bytes_written: {d}", .{ chunk.ptr, chunk.len, total_bytes_written });

        var index: usize = 0;
        while (index < chunk.len) {
            const trace_write = tracer.frameStart("writer.interface.write");
            defer tracer.frameEnd(trace_write, "writer.interface.write");

            log.debug("Writing chunk {*} of size {d}, total_bytes_written: {d}", .{ chunk.ptr, chunk.len, total_bytes_written });
            index += try writer.interface.write(chunk[index..]);
        }

        total_bytes_written += chunk.len;
    }
    try writer.interface.flush();

    try std.testing.expectEqual(shard.byteSize(), writer.bytes_written);
    try std.testing.expect(!writer.can_process_last_event);
}

test "DeviceWriterBuffered: flush finalizes transfer" {
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    tracer = Tracer.init("ai.zml.test.DeviceWriterBuffered.flush");
    const trace = tracer.frameStart("DeviceWriterBuffered.test.flush");
    defer tracer.frameEnd(trace, "DeviceWriterBuffered.test.flush");

    const device = platform.getDevices()[0];

    try warmupDevices(allocator, platform, &.{device});

    const dma_buffer_size = 64 * MB;

    const shape = Shape.init(.{0}, .u8);
    const shard: Shard = .{
        .shape = shape,
        .tensor = .{ .resource_uri = try .parse("file:///model.safetensors"), .name = "tensor_flush", .shape = shape, .offset = 0 },
        .device = device,
    };

    const writer_buffer = try allocator.alloc(u8, dma_buffer_size);
    defer allocator.free(writer_buffer);

    var writer: DeviceWriterBuffered = try .init(platform, shard, writer_buffer);
    defer writer.deinit();

    try std.testing.expect(writer.can_process_last_event);
    try writer.interface.flush();
    try std.testing.expect(!writer.can_process_last_event);

    try writer.interface.flush();
    try std.testing.expect(!writer.can_process_last_event);
}

const DeviceWriter = struct {
    platform: Platform,
    shard: Shard,
    transfer_manager: *pjrtx.AsyncHostToDeviceTransferManager,

    events: [2]?*pjrtx.Event,
    next_slot_idx: u1,

    bytes_written: u64,
    can_process_last_event: bool,

    interface: std.io.Writer,

    pub fn init(platform: Platform, shard: Shard, memory_kind: pjrt.Memory.Kind) !DeviceWriter {
        const trace = tracer.frameStart("DeviceWriter.init");
        defer tracer.frameEnd(trace, "DeviceWriter.init");

        const trace_memory = tracer.frameStart("DeviceWriter.init.memory");
        const memories = shard.device.addressableMemories(platform.pjrt_api);
        var memory = memories[0];

        if (platform.target == .cuda) {
            for (memories) |mem| {
                if (mem.kind(platform.pjrt_api) == memory_kind) {
                    memory = mem;
                    break;
                }
            }
        }
        tracer.frameEnd(trace_memory, "DeviceWriter.init.memory");

        const shape_spec = pjrt.ShapeSpec.init(shard.shape.dims(), bufferTypeFromDtype(shard.shape.dtype()));
        const trace_transfer_manager = tracer.frameStart("DeviceWriter.init.transfer_manager");
        const transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
            .shape_specs = &.{shape_spec},
            .memory = memory,
        });
        tracer.frameEnd(trace_transfer_manager, "DeviceWriter.init.transfer_manager");

        return .{
            .platform = platform,
            .shard = shard,
            .transfer_manager = transfer_manager,
            .events = .{ null, null },
            .next_slot_idx = 0,
            .bytes_written = 0,
            .can_process_last_event = true,
            .interface = .{ .vtable = &vtable, .buffer = &.{}, .end = 0 },
        };
    }

    pub fn deinit(self: *DeviceWriter) void {
        const buf = self.buffer() catch unreachable;
        buf.deinit(self.platform.pjrt_api);
        self.transfer_manager.deinit(self.platform.pjrt_api);
        self.transfer_manager = undefined;
    }

    pub fn buffer(self: *DeviceWriter) !*pjrtx.Buffer {
        return try self.transfer_manager.retrieveBuffer(self.platform.pjrt_api, 0);
    }

    fn deviceDescription(self: *DeviceWriter) []const u8 {
        return self.shard.device.getDescription(self.platform.pjrt_api).toString(self.platform.pjrt_api);
    }

    fn awaitEvent(self: *DeviceWriter, idx: u1) !void {
        if (self.events[idx]) |event| {
            const trace = tracer.frameStart("DeviceWriter.awaitEvent");
            defer tracer.frameEnd(trace, "DeviceWriter.awaitEvent");

            try event.awaitBlocking(self.platform.pjrt_api);
            self.events[idx] = null;
        }
    }

    fn transfer(self: *DeviceWriter, data: []const u8, is_last: bool) !*pjrtx.Event {
        const trace = tracer.frameStart("DeviceWriter.transfer");
        defer tracer.frameEnd(trace, "DeviceWriter.transfer");

        const offset = self.bytes_written;

        defer {
            if (!is_last) self.bytes_written += data.len;
        }

        log.debug("DeviceWriter({s}).transfer: {d}B, offset: {d}, is_last: {} - progress: {d}/{d}B", .{
            self.deviceDescription(),
            data.len,
            offset,
            is_last,
            offset + data.len,
            self.shard.byteSize(),
        });

        std.debug.assert(offset + data.len <= self.shard.byteSize());

        return self.transfer_manager.transferData(self.platform.pjrt_api, 0, data, @intCast(offset), is_last) catch |err| {
            log.err("PJRT transferData failed: {}", .{err});
            return error.WriteFailed;
        };
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const trace = tracer.frameStart("DeviceWriter.drain");
        defer tracer.frameEnd(trace, "DeviceWriter.drain");

        const self = @as(*DeviceWriter, @alignCast(@fieldParentPtr("interface", w)));

        std.debug.assert(w.end == 0);
        std.debug.assert(splat == 1);
        std.debug.assert(data.len == 1);

        const chunk = data[0];

        log.debug("DeviceWriter({s}).drain: chunk={d}B, progress={d}/{d}, slot={d}, pending=[{s},{s}], last_event_ready={}", .{
            self.deviceDescription(),
            chunk.len,
            self.bytes_written,
            self.shard.byteSize(),
            self.next_slot_idx,
            if (self.events[0] != null) "busy" else "free",
            if (self.events[1] != null) "busy" else "free",
            self.can_process_last_event,
        });

        if (chunk.len == 0) return 0;

        const slot_to_use = self.next_slot_idx;

        self.awaitEvent(slot_to_use) catch |err| {
            log.err("Error awaiting event in drain: {}", .{err});
            return error.WriteFailed;
        };

        self.events[slot_to_use] = try self.transfer(chunk, false);
        self.next_slot_idx = 1 - self.next_slot_idx;

        return chunk.len;
    }

    fn flush(w: *std.io.Writer) std.io.Writer.Error!void {
        const trace = tracer.frameStart("DeviceWriter.flush");
        defer tracer.frameEnd(trace, "DeviceWriter.flush");

        const self = @as(*DeviceWriter, @alignCast(@fieldParentPtr("interface", w)));

        log.debug("DeviceWriter({s}).flush: awaiting pending transfers - progress={d}/{d}B, pending_slots=[{s},{s}], will_finalize={}", .{
            self.deviceDescription(),
            self.bytes_written,
            self.shard.byteSize(),
            if (self.events[0] != null) "busy" else "free",
            if (self.events[1] != null) "busy" else "free",
            self.can_process_last_event,
        });

        self.awaitEvent(0) catch return error.WriteFailed;
        self.awaitEvent(1) catch return error.WriteFailed;

        if (self.can_process_last_event) {
            std.debug.assert(self.bytes_written == self.shard.byteSize());

            const last_event = try self.transfer(&.{}, true);
            last_event.awaitBlocking(self.platform.pjrt_api) catch return error.WriteFailed;

            self.can_process_last_event = false;
        }
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
    };
};

test "DeviceWriter: write" {
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    tracer = Tracer.init("ai.zml.test.DeviceWriter.write");
    const trace = tracer.frameStart("DeviceWriter.test.write");
    defer tracer.frameEnd(trace, "DeviceWriter.test.write");

    const device = platform.getDevices()[0];

    try warmupDevices(allocator, platform, &.{device});

    const chunk_size = BUF_64_MB;
    const tensor_size = 4 * BUF_256_MB;
    const shape: Shape = .init(.{tensor_size / @sizeOf(f32)}, .f32);
    const shard: Shard = .{
        .shape = shape,
        .tensor = .{
            .resource_uri = try .parse("file:///model.safetensors"),
            .name = "tensor",
            .shape = shape,
            .offset = 0,
        },
        .device = device,
    };

    var writer: DeviceWriter = try .init(platform, shard, .device);
    defer writer.deinit();

    const data = try allocator.alloc(u8, tensor_size);
    defer allocator.free(data);

    for (data, 0..) |*byte, i| byte.* = @intCast(i % 256);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, data);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, data) catch {};

    const trace_write = tracer.frameStart("DeviceWriter.test.write.loop");
    defer tracer.frameEnd(trace_write, "DeviceWriter.test.write.loop");

    var total_bytes_written: u64 = 0;
    while (total_bytes_written < tensor_size) {
        const remaining = tensor_size - total_bytes_written;
        const chunk = data[total_bytes_written..][0..@min(remaining, chunk_size)];

        var index: usize = 0;
        while (index < chunk.len) {
            const trace_write_chunk = tracer.frameStart("DeviceWriter.test.write.loop.write");
            defer tracer.frameEnd(trace_write_chunk, "DeviceWriter.test.write.loop.write");

            log.debug("Writing chunk {*} of size {d}, total_bytes_written: {d}", .{ chunk.ptr, chunk.len, total_bytes_written });
            index += try writer.interface.write(chunk[index..]);
        }

        total_bytes_written += chunk.len;
    }
    try writer.interface.flush();

    try std.testing.expectEqual(shard.byteSize(), writer.bytes_written);
    try std.testing.expect(!writer.can_process_last_event);
}

test "DeviceWriter: writeAll" {
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    tracer = Tracer.init("ai.zml.test.DeviceWriter.writeAll");
    const trace = tracer.frameStart("DeviceWriter.test.writeAll");
    defer tracer.frameEnd(trace, "DeviceWriter.test.writeAll");

    const device = platform.getDevices()[0];

    try warmupDevices(allocator, platform, &.{device});

    const tensor_size = BUF_256_MB;
    const shape: Shape = .init(.{tensor_size / @sizeOf(u8)}, .u8);
    const shard: Shard = .{
        .shape = shape,
        .tensor = .{ .resource_uri = try .parse("file:///model.safetensors"), .name = "tensor1", .shape = shape, .offset = 0 },
        .device = device,
    };

    const data = try allocator.alloc(u8, tensor_size);
    defer allocator.free(data);

    for (data, 0..) |*byte, i| byte.* = @intCast(i % 256);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, data);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, data) catch {};

    var writer: DeviceWriter = try .init(platform, shard, .device);
    defer writer.deinit();

    const trace_write_all = tracer.frameStart("DeviceWriter.test.writeAll.writer.writeAll");
    try writer.interface.writeAll(data);
    try writer.interface.flush();
    tracer.frameEnd(trace_write_all, "DeviceWriter.test.writeAll.writer.writeAll");

    try std.testing.expectEqual(shard.byteSize(), writer.bytes_written);
    try std.testing.expect(!writer.can_process_last_event);

    const pjrt_buffer = try writer.buffer();
    defer pjrt_buffer.deinit(platform.pjrt_api);

    const reader_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(BUF_4_KB), BUF_32_MB);
    defer allocator.free(reader_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, reader_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, reader_buffer) catch {};

    var reader: DeviceReader = try .init(platform, pjrt_buffer, reader_buffer);

    const read_back_data = try allocator.alloc(u8, data.len);
    defer allocator.free(read_back_data);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, read_back_data);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, read_back_data) catch {};

    var read_back_writer: std.io.Writer = .fixed(read_back_data);
    const read_bytes = try reader.interface.streamRemaining(&read_back_writer);

    try std.testing.expectEqual(data.len, read_bytes);
    try std.testing.expectEqualSlices(u8, data, read_back_data);
}

test "DeviceWriter: flush finalizes transfer" {
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    tracer = Tracer.init("ai.zml.test.DeviceWriter.flush");
    const trace = tracer.frameStart("DeviceWriter.test.flush");
    defer tracer.frameEnd(trace, "DeviceWriter.test.flush");

    const device = platform.getDevices()[0];

    try warmupDevices(allocator, platform, &.{device});

    const shape = Shape.init(.{0}, .u8);
    const shard: Shard = .{
        .shape = shape,
        .tensor = .{ .resource_uri = try .parse("file:///model.safetensors"), .name = "tensor_flush", .shape = shape, .offset = 0 },
        .device = device,
    };

    var writer: DeviceWriter = try .init(platform, shard, .device);
    defer writer.deinit();

    try std.testing.expect(writer.can_process_last_event);
    try writer.interface.flush();
    try std.testing.expect(!writer.can_process_last_event);

    try writer.interface.flush();
    try std.testing.expect(!writer.can_process_last_event);
}

const TensorWriter = struct {
    device_writers: []DeviceWriter,

    chunk_size: usize,
    current_buffer_idx: u1,
    shard_size: u64,
    is_sharded: bool,
    total_bytes_processed: u64,

    buffer: []u8,
    interface: std.io.Writer,

    pub fn init(device_writers: []DeviceWriter, buffer: []u8) TensorWriter {
        const chunk_size = buffer.len / 2;

        return .{
            .device_writers = device_writers,
            .buffer = buffer,
            .chunk_size = chunk_size,
            .shard_size = if (device_writers.len > 0) device_writers[0].shard.byteSize() else 0,
            .is_sharded = if (device_writers.len > 0)
                device_writers[0].shard.byteSize() < device_writers[0].shard.tensor.byteSize()
            else
                false,
            .current_buffer_idx = 0,
            .total_bytes_processed = 0,
            .interface = .{ .vtable = &vtable, .buffer = buffer[0..chunk_size], .end = 0 },
        };
    }

    fn process(self: *TensorWriter, chunk_to_process: []const u8) !void {
        const trace = tracer.frameStart("TensorWriter.process");
        defer tracer.frameEnd(trace, "TensorWriter.process");

        if (self.device_writers.len == 0 or chunk_to_process.len == 0) return;

        var data_offset: usize = 0;
        while (data_offset < chunk_to_process.len) {
            const current_tensor_offset = self.total_bytes_processed + data_offset;
            const remaining_in_data = chunk_to_process.len - data_offset;

            const current_shard_idx: usize = @intCast(current_tensor_offset / self.shard_size);
            if (current_shard_idx >= self.device_writers.len) break;

            const offset_in_shard = current_tensor_offset % self.shard_size;
            const chunk_limit_by_boundary = self.shard_size - offset_in_shard;
            const chunk_to_write_len = @min(remaining_in_data, chunk_limit_by_boundary);
            if (chunk_to_write_len == 0) break;

            const chunk_to_move = chunk_to_process[data_offset .. data_offset + chunk_to_write_len];

            log.debug("TensorWriter.process: tensor_offset={d}B/{d}B, shard[{d}]@{d}B+{d}B, remaining_data={d}B", .{
                current_tensor_offset,
                self.shard_size * self.device_writers.len,
                current_shard_idx,
                offset_in_shard,
                chunk_to_write_len,
                remaining_in_data - chunk_to_write_len,
            });

            if (self.is_sharded) {
                try self.device_writers[current_shard_idx].interface.writeAll(chunk_to_move);
            } else {
                for (self.device_writers) |*dw| {
                    try dw.interface.writeAll(chunk_to_move);
                }
            }

            data_offset += chunk_to_write_len;
        }

        self.total_bytes_processed += chunk_to_process.len;
    }

    pub fn swap(self: *TensorWriter) void {
        self.current_buffer_idx = 1 - self.current_buffer_idx;
        const new_offset = self.current_buffer_idx * self.chunk_size;
        self.interface.buffer = self.buffer[new_offset .. new_offset + self.chunk_size];
        self.interface.end = 0;
    }

    fn processAndSwap(self: *TensorWriter) !void {
        const trace = tracer.frameStart("TensorWriter.processAndSwap");
        defer tracer.frameEnd(trace, "TensorWriter.processAndSwap");

        const chunk_to_process = self.interface.buffered();

        log.debug("TensorWriter.processAndSwap: switching from buffer #{d} to #{d}, processing {d}B chunk, total processed: {d}B", .{
            self.current_buffer_idx,
            1 - self.current_buffer_idx,
            chunk_to_process.len,
            self.total_bytes_processed,
        });

        if (chunk_to_process.len > 0) {
            try self.process(chunk_to_process);
        }

        self.swap();
    }

    fn rebase(w: *std.io.Writer, preserve: usize, minimum_len: usize) std.io.Writer.Error!void {
        const trace = tracer.frameStart("TensorWriter.rebase");
        defer tracer.frameEnd(trace, "TensorWriter.rebase");

        const self: *TensorWriter = @alignCast(@fieldParentPtr("interface", w));
        std.debug.assert(preserve == 0);

        log.debug("TensorWriter.rebase: preserve={d}, minimum_len={d}, buffered={d}B, processed={d}B, will_flip_buffer={}", .{
            preserve,
            minimum_len,
            w.end,
            self.total_bytes_processed,
            w.end + minimum_len >= w.buffer.len,
        });

        self.processAndSwap() catch |err| {
            log.err("Error processing chunk during rebase: {}", .{err});
            return error.WriteFailed;
        };
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const trace = tracer.frameStart("TensorWriter.drain");
        defer tracer.frameEnd(trace, "TensorWriter.drain");

        const self: *TensorWriter = @alignCast(@fieldParentPtr("interface", w));

        const total_incoming_bytes = if (data.len > 0)
            std.io.Writer.countSplat(data, splat)
        else
            0;

        log.debug("TensorWriter.drain: incoming={d}B (splat={d}), buffered={d}B, processed={d}B, will_flip_buffer={}", .{
            total_incoming_bytes,
            splat,
            w.end,
            self.total_bytes_processed,
            w.end + total_incoming_bytes >= w.buffer.len,
        });

        try self.processAndSwap();

        if (data.len == 0) return 0;

        var consumed: usize = 0;
        for (data[0 .. data.len - 1]) |chunk| {
            try self.process(chunk);
            consumed += chunk.len;
        }
        const last_chunk = data[data.len - 1];
        for (0..splat) |_| {
            try self.process(last_chunk);
            consumed += last_chunk.len;
        }
        return consumed;
    }
    fn flush(w: *std.io.Writer) !void {
        const trace = tracer.frameStart("TensorWriter.flush");
        defer tracer.frameEnd(trace, "TensorWriter.flush");

        const self: *TensorWriter = @alignCast(@fieldParentPtr("interface", w));

        log.debug("TensorWriter.flush: finalizing tensor - buffered={d}B, processed={d}B, remaining_batches={d}, will_double_flush={}", .{
            w.end,
            self.total_bytes_processed,
            if (w.end > 0) @as(u32, 2) else @as(u32, 1),
            w.end > 0,
        });

        try self.processAndSwap();

        if (w.end > 0) {
            try self.processAndSwap();
        }

        for (self.device_writers) |*dw| {
            try dw.interface.flush();
        }
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
        .rebase = rebase,
    };
};

const DeviceReader = struct {
    pub const NUM_SLOTS = 2;

    platform: Platform,
    pjrt_buffer: *const pjrtx.Buffer,
    total_size: u64,
    chunk_size: usize,

    slots: [NUM_SLOTS]?*pjrtx.Event,
    dma_buffer: []u8,
    bytes_requested: u64,
    bytes_activated: u64,
    next_request_slot: u1,
    next_consume_slot: u1,

    is_primed: bool,
    interface: std.io.Reader,

    pub fn init(platform: Platform, pjrt_buffer: *const pjrtx.Buffer, dma_buffer: []u8) !DeviceReader {
        const chunk_size = dma_buffer.len / NUM_SLOTS;
        std.debug.assert(chunk_size > 0);

        return .{
            .platform = platform,
            .pjrt_buffer = pjrt_buffer,
            .total_size = try pjrt_buffer.getOnDeviceSizeInBytes(platform.pjrt_api),
            .chunk_size = chunk_size,
            .slots = .{ null, null },
            .dma_buffer = dma_buffer,
            .bytes_requested = 0,
            .bytes_activated = 0,
            .next_request_slot = 0,
            .next_consume_slot = 0,
            .is_primed = false,
            .interface = .{ .vtable = &vtable, .buffer = &.{}, .seek = 0, .end = 0 },
        };
    }

    fn requestNextChunk(self: *DeviceReader) std.io.Reader.StreamError!void {
        const trace = tracer.frameStart("DeviceReader.requestNextChunk");
        defer tracer.frameEnd(trace, "DeviceReader.requestNextChunk");

        const slot_to_fill = self.next_request_slot;
        std.debug.assert(self.slots[slot_to_fill] == null);

        if (self.bytes_requested >= self.total_size) return;

        const offset_in_dma = slot_to_fill * self.chunk_size;
        const chunk_dma_buf = self.dma_buffer[offset_in_dma .. offset_in_dma + self.chunk_size];
        const remaining_on_device = self.total_size - self.bytes_requested;
        const transfer_size = @min(remaining_on_device, chunk_dma_buf.len);
        const dest_slice = chunk_dma_buf[0..transfer_size];
        const offset: i64 = @intCast(self.bytes_requested);

        log.debug("DeviceReader.requestNextChunk: slot={d}, device_offset={d}B, transfer_size={d}B, remaining={d}B", .{
            slot_to_fill,
            offset,
            transfer_size,
            self.total_size - (self.bytes_requested + transfer_size),
        });

        const event = self.pjrt_buffer.copyRawToHost(self.platform.pjrt_api, dest_slice, offset) catch |err| {
            log.err("PJRT copyRawToHost failed: {}", .{err});
            return error.ReadFailed;
        };

        if (event) |ev| self.slots[slot_to_fill] = ev;
        self.bytes_requested += transfer_size;
        self.next_request_slot = 1 - self.next_request_slot;
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const trace = tracer.frameStart("DeviceReader.stream(vtable)");
        defer tracer.frameEnd(trace, "DeviceReader.stream(vtable)");

        _ = w;
        _ = limit;
        const self = @as(*DeviceReader, @alignCast(@fieldParentPtr("interface", r)));

        std.debug.assert(r.seek == r.end);

        if (!self.is_primed) {
            self.is_primed = true;
            if (self.total_size == 0) return error.EndOfStream;
            log.debug("DeviceReader: Priming pipeline for {d} bytes...", .{self.total_size});
            for (0..NUM_SLOTS) |_| try self.requestNextChunk();
        } else {
            try self.requestNextChunk();
        }

        const slot_to_consume = self.next_consume_slot;

        if (self.slots[slot_to_consume]) |event| {
            const trace_await = tracer.frameStart("DeviceReader.awaitEvent");
            defer tracer.frameEnd(trace_await, "DeviceReader.awaitEvent");

            event.awaitBlocking(self.platform.pjrt_api) catch |err| {
                log.err("Error awaiting event in stream: {}", .{err});
                return error.ReadFailed;
            };
            self.slots[slot_to_consume] = null;
        } else {
            if (self.bytes_activated >= self.total_size) {
                return error.EndOfStream;
            } else {
                log.err("DeviceReader stalled: waiting for slot {d} which has no event.", .{slot_to_consume});
                return error.ReadFailed;
            }
        }

        const offset_in_dma = slot_to_consume * self.chunk_size;
        const remaining_total = self.total_size - self.bytes_activated;
        const actual_chunk_size = @min(self.chunk_size, remaining_total);

        self.bytes_activated += actual_chunk_size;

        r.buffer = self.dma_buffer[offset_in_dma .. offset_in_dma + actual_chunk_size];
        r.seek = 0;
        r.end = r.buffer.len;

        self.next_consume_slot = 1 - self.next_consume_slot;

        return 0;
    }

    const vtable: std.io.Reader.VTable = .{
        .stream = stream,
    };
};

test "DeviceReader: streamRemaining" {
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    tracer = Tracer.init("ai.zml.test.DeviceReader");
    const trace_test = tracer.frameStart("DeviceReader.test.streamRemaining");
    defer tracer.frameEnd(trace_test, "DeviceReader.test.streamRemaining");

    const memory_kind: pjrtx.Memory.Kind = if (platform.target == .cpu) .host_pinned else .device;
    const memories = platform.getDevices()[0].addressableMemories(platform.pjrt_api);
    const memory = for (memories) |m| {
        const kind = m.kind(platform.pjrt_api);
        if (kind == memory_kind) break m;
    } else return error.NotFound;

    const shape: Shape = .init(.{4 * BUF_256_MB / @sizeOf(f32)}, .f32);

    const buffer = try platform.pjrt_client.createUnitializedBuffer(platform.pjrt_api, .{
        .dims = shape.dims(),
        .element_type = bufferTypeFromDtype(shape.dtype()),
        .layout = .{
            .tiled = .{
                .minor_to_major = minor_to_major[Shape.MAX_RANK - shape.rank() ..],
                .tile_dims = &.{},
                .tile_dims_sizes = &.{},
            },
        },
        .dst = .{ .memory = memory },
    });
    defer buffer.deinit(platform.pjrt_api);

    const reader_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(BUF_4_KB), BUF_64_MB);
    defer allocator.free(reader_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, reader_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, reader_buffer) catch {};

    var device_reader: DeviceReader = try .init(platform, buffer, reader_buffer);

    const read_back_buffer = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(read_back_buffer);

    for (read_back_buffer, 0..) |*byte, i| byte.* = @intCast(i % 256);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, read_back_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, read_back_buffer) catch {};

    var read_back_writer: std.io.Writer = .fixed(read_back_buffer);

    const trace_stream = tracer.frameStart("DeviceReader.streamRemaining");
    const bytes_read = try device_reader.interface.streamRemaining(&read_back_writer);
    tracer.frameEnd(trace_stream, "DeviceReader.streamRemaining");

    try std.testing.expectEqual(shape.byteSize(), bytes_read);
}

test "DeviceReader: arange / streamRemaining" {
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    const trace_test = tracer.frameStart("DeviceReader.test.streamRemaining");
    defer tracer.frameEnd(trace_test, "DeviceReader.test.streamRemaining");

    const Local = struct {
        fn forward() zml.Tensor {
            return .arange(.{ .end = 256 * MB }, .u32);
        }
    };

    const x_d = try zml.testing.compileAndCall(platform, Local.forward, .{});
    defer x_d.deinit();

    const dma_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(4 * KB), 1 * MB);
    defer allocator.free(dma_buffer);

    platform.pjrt_client.dmaMap(platform.pjrt_api, dma_buffer) catch {};
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, dma_buffer) catch unreachable;

    var device_reader: DeviceReader = try .init(platform, x_d._shards.get(0), dma_buffer);

    const x_h = try allocator.alignedAlloc(u32, .fromByteUnits(16 * KB), x_d.shape().count());
    defer allocator.free(x_h);
    var x_h_writer: std.io.Writer = .fixed(std.mem.sliceAsBytes(x_h));

    const trace_stream = tracer.frameStart("DeviceReader.streamRemaining");
    const bytes_read = try device_reader.interface.streamRemaining(&x_h_writer);
    tracer.frameEnd(trace_stream, "DeviceReader.streamRemaining");

    std.log.warn("Device: {f}, host: {d}, read: {d}", .{
        x_d,
        256 * MB * @sizeOf(u32),
        bytes_read,
    });
    try std.testing.expectEqual(x_d.shape().byteSize(), bytes_read);
    try std.testing.expectEqual(256 * MB * @sizeOf(u32), bytes_read);
    for (x_h, 0..) |actual, expected| {
        errdefer log.err("Mismatch at offset {d}, expected {x}, got {x}", .{ expected, expected, actual });
        try std.testing.expectEqual(expected, actual);
    }
}

test "DeviceReader: discard writer" {
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();

    tracer = Tracer.init("ai.zml.test.DeviceReader");
    const trace_test = tracer.frameStart("DeviceReader.test.discardWriter");
    defer tracer.frameEnd(trace_test, "DeviceReader.test.discardWriter");

    const memory_kind: pjrtx.Memory.Kind = if (platform.target == .cpu) .host_pinned else .device;
    const memories = platform.getDevices()[0].addressableMemories(platform.pjrt_api);
    const memory = for (memories) |m| {
        const kind = m.kind(platform.pjrt_api);
        if (kind == memory_kind) break m;
    } else return error.NotFound;

    const shape: Shape = .init(.{4 * BUF_256_MB / @sizeOf(f32)}, .f32);

    const buffer = try platform.pjrt_client.createUnitializedBuffer(platform.pjrt_api, .{
        .dims = shape.dims(),
        .element_type = bufferTypeFromDtype(shape.dtype()),
        .layout = .{
            .tiled = .{
                .minor_to_major = minor_to_major[Shape.MAX_RANK - shape.rank() ..],
                .tile_dims = &.{},
                .tile_dims_sizes = &.{},
            },
        },
        .dst = .{ .memory = memory },
    });
    defer buffer.deinit(platform.pjrt_api);

    const reader_buffer_size = BUF_64_MB;
    const reader_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(BUF_4_KB), reader_buffer_size);
    defer allocator.free(reader_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, reader_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, reader_buffer) catch {};

    var device_reader: DeviceReader = try .init(platform, buffer, reader_buffer);

    var total_bytes_consumed: u64 = 0;
    var writer_buffer: [1]u8 = undefined;
    var writer: std.io.Writer = .fixed(&writer_buffer);

    while (true) {
        const chunk = device_reader.interface.buffered();
        if (chunk.len > 0) {
            total_bytes_consumed += chunk.len;
            device_reader.interface.tossBuffered();
        }

        const n = device_reader.interface.stream(&writer, .unlimited) catch |err| switch (err) {
            error.EndOfStream => break,
            else => |e| return e,
        };

        std.debug.assert(n == 0);
    }

    try std.testing.expectEqual(shape.byteSize(), total_bytes_consumed);
}

const TensorReader = struct {
    device_readers: []DeviceReader,
    current_reader_idx: usize,
    interface: std.io.Reader,

    pub fn init(device_readers: []DeviceReader) !TensorReader {
        return .{
            .device_readers = device_readers,
            .current_reader_idx = 0,
            .interface = .{ .vtable = &vtable, .buffer = &.{}, .seek = 0, .end = 0 },
        };
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self = @as(*TensorReader, @alignCast(@fieldParentPtr("interface", r)));

        while (self.current_reader_idx < self.device_readers.len) {
            const current_dr_interface = &self.device_readers[self.current_reader_idx].interface;
            const bytes_read = current_dr_interface.stream(w, limit) catch |err| switch (err) {
                error.EndOfStream => {
                    self.current_reader_idx += 1;
                    continue;
                },
                else => |e| return e,
            };
            return bytes_read;
        }
        return error.EndOfStream;
    }

    const vtable: std.io.Reader.VTable = .{
        .stream = stream,
    };
};

test "Full Pipeline: TensorWriter -> GPU -> TensorReader" {
    const allocator = std.testing.allocator;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();

    const platform = zml.testing.env();

    tracer = Tracer.init("ai.zml.test.FullPipeline");
    const trace_test = tracer.frameStart("DeviceReader.test.streamRemaining");
    defer tracer.frameEnd(trace_test, "DeviceReader.test.streamRemaining");

    const devices = platform.getDevices();
    const memory_kind: pjrtx.Memory.Kind = if (platform.target == .cpu) .host_pinned else .device;

    try warmupDevices(allocator, platform, devices);

    if (devices.len < 2) {
        std.log.warn("Skipping test, requires at least 2 devices, found {d}", .{devices.len});
        return error.SkipZigTest;
    }

    const devices_to_use = devices[0..1];
    const filename = "full_pipeline_test.bin";
    const tensor_size = 1 * 4 * BUF_256_MB;
    const file_buffer_size = BUF_128_MB;
    const writer_buffer_size = BUF_64_MB;
    const device_reader_buffer_size = BUF_128_MB;
    const alignment = BUF_4_KB;

    const shape = Shape.init(.{tensor_size / @sizeOf(f32)}, .f32).withSharding(.{0});
    const tensor: Tensor = .{ .resource_uri = try .parse("file:///model.safetensors"), .name = "full_pipeline_tensor", .shape = shape, .offset = 0 };

    _ = try createBinFile(tmp_dir, filename, shape.byteSize(), null);
    const file_path = try tmp_dir.dir.realpathAlloc(allocator, filename);
    defer allocator.free(file_path);

    const file = try std.fs.openFileAbsolute(file_path, .{});
    defer file.close();

    _ = try switchToDirectIO(file);

    const shards = try computeShards(allocator, tensor, devices_to_use);
    defer allocator.free(shards);

    const file_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(alignment), file_buffer_size);
    defer allocator.free(file_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, file_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, file_buffer) catch {};

    const writer_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(alignment), writer_buffer_size);
    defer allocator.free(writer_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, writer_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, writer_buffer) catch {};

    var device_writers: std.ArrayList(DeviceWriter) = try .initCapacity(allocator, shards.len);
    errdefer for (device_writers.items) |*dw| dw.deinit();
    defer device_writers.deinit(allocator);

    for (shards) |s| device_writers.appendAssumeCapacity(try .init(platform, s, memory_kind));

    var file_reader = file.reader(&.{});
    try file_reader.seekTo(tensor.offset);

    var aligned_file_reader: AlignedFileReader = try .init(file_reader, .fromByteUnits(BUF_4_KB));
    var limited_reader: std.io.Reader.Limited = .init(&aligned_file_reader.interface, .limited64(tensor.shape.byteSize()), &.{});
    var tensor_writer: TensorWriter = .init(device_writers.items, writer_buffer);

    const trace_write = tracer.frameStart("FullPipeline.TensorWriter.streamRemaining");
    const bytes_copied = try limited_reader.interface.streamRemaining(&tensor_writer.interface);
    try tensor_writer.interface.flush();
    tracer.frameEnd(trace_write, "FullPipeline.TensorWriter.streamRemaining");

    try std.testing.expectEqual(tensor.shape.byteSize(), bytes_copied);

    var pjrt_buffers: [devices_to_use.len]*pjrtx.Buffer = undefined;
    for (device_writers.items, 0..) |*dw, i| {
        pjrt_buffers[i] = try dw.buffer();
        dw.deinit();
    }
    defer for (pjrt_buffers) |b| b.deinit(platform.pjrt_api);

    var device_readers_buffers: std.ArrayList([]u8) = .{};
    defer {
        for (device_readers_buffers.items) |b| {
            platform.pjrt_client.dmaUnmap(platform.pjrt_api, b) catch {};
            allocator.free(b);
        }
        device_readers_buffers.deinit(allocator);
    }

    var device_readers: std.ArrayList(DeviceReader) = .{};
    defer device_readers.deinit(allocator);

    for (pjrt_buffers) |buffer| {
        const device_reader_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(alignment), device_reader_buffer_size);
        try device_readers_buffers.append(allocator, device_reader_buffer);
        try platform.pjrt_client.dmaMap(platform.pjrt_api, device_reader_buffer);
        try device_readers.append(allocator, try .init(platform, buffer, device_reader_buffer));
    }

    var tensor_reader: TensorReader = try .init(device_readers.items);

    try file_reader.seekTo(tensor.offset);

    const verification_chunk_size = BUF_32_MB;
    const tensor_reader_buffer = try allocator.alloc(u8, verification_chunk_size);
    defer allocator.free(tensor_reader_buffer);

    const file_reader_buffer = try allocator.alloc(u8, verification_chunk_size);
    defer allocator.free(file_reader_buffer);

    var total_bytes_read: u64 = 0;
    while (total_bytes_read < tensor_size) {
        const limit = @min(verification_chunk_size, tensor_size - total_bytes_read);

        const tensor_bytes_read = try tensor_reader.interface.readSliceShort(tensor_reader_buffer[0..limit]);
        const file_bytes_read = try file_reader.interface.readSliceShort(file_reader_buffer[0..limit]);

        try std.testing.expectEqual(tensor_bytes_read, file_bytes_read);
        if (tensor_bytes_read == 0) break;

        try std.testing.expectEqualSlices(u8, file_reader_buffer[0..file_bytes_read], tensor_reader_buffer[0..tensor_bytes_read]);

        total_bytes_read += tensor_bytes_read;
    }

    try std.testing.expectEqual(tensor_size, total_bytes_read);
}

// This is an example of how sharding metadata might be added to a model registry.
// In a real application, this metadata might come from a config file or be inferred from the
// model architecture. Here, we hardcode some example tensor names and shard them on axis 1.
fn addExampleShardingMetadata(registry: *TensorRegistry) !void {
    const sharded_names = [_][]const u8{
        "model.embed_tokens.weight",
        "lm_head.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.down_proj.weight",
    };

    const sharding_axis_indices = try registry.arena.allocator().dupe(i64, &.{1}); // Shard on axis 1 (columns)
    const sharding_meta_value: Metadata = .{ .array_int = sharding_axis_indices };

    var tensor_it = registry.tensors.iterator();

    for (sharded_names) |name| {
        tensor_it.reset();

        while (tensor_it.next()) |entry| {
            const tensor = entry.value_ptr.*;

            if (std.mem.endsWith(u8, tensor.name, name)) {
                var key_buf: [512]u8 = undefined;
                const sharding_key = try std.fmt.bufPrint(&key_buf, "sharding.{s}", .{tensor.name});

                try registry.metadata.put(
                    registry.arena.allocator(),
                    try registry.arena.allocator().dupe(u8, sharding_key),
                    sharding_meta_value,
                );
            }
        }
    }
}

// Annotate tensor shapes with sharding information from metadata.
fn annotateShapesWithSharding(registry: *TensorRegistry) !void {
    var tensor_it = registry.tensors.iterator();

    while (tensor_it.next()) |entry| {
        const tensor = entry.value_ptr;
        var key_buf: [512]u8 = undefined;
        const sharding_key = std.fmt.bufPrint(&key_buf, "sharding.{s}", .{tensor.name}) catch continue;

        if (registry.metadata.get(sharding_key)) |sharding_axes_meta| {
            for (sharding_axes_meta.array_int) |sharding_axis| {
                tensor.shape = tensor.shape.withSharding(.{sharding_axis});
            }
        }
    }
}

// Compute the shards for a tensor based on its shape and the available devices.
fn computeShards(allocator: std.mem.Allocator, tensor: Tensor, devices: []const *const pjrt.Device) ![]Shard {
    const trace = tracer.frameStart("computeShards");
    defer tracer.frameEnd(trace, "computeShards");

    const sharded_axes_count = std.simd.countTrues(tensor.shape._sharding_info);
    const is_sharded = sharded_axes_count > 0;

    const shards = try allocator.alloc(Shard, devices.len);

    if (!is_sharded) {
        for (devices, 0..) |device, i| {
            shards[i] = .{ .shape = tensor.shape, .tensor = tensor, .device = device };
        }
    } else {
        const sharded_axis = std.simd.firstIndexOfValue(tensor.shape._sharding_info, true) orelse unreachable;
        const original_dim: u64 = @intCast(tensor.shape.dim(sharded_axis));

        const shard_dim = original_dim / @as(u64, @intCast(devices.len));

        var shard_shape = tensor.shape;
        shard_shape._dims.set(sharded_axis, @intCast(shard_dim));

        for (devices, 0..) |device, i| {
            shards[i] = .{ .shape = shard_shape, .tensor = tensor, .device = device };
        }
    }

    return shards;
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    var timer = try std.time.Timer.start();

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    _ = args.next().?;

    const path = args.next() orelse {
        log.err("Usage: bazel run //examples/loader /path/to/model.safetensors...", .{});
        return;
    };

    log.warn("Initializing context and platform...", .{});
    var context: Context = try .init();
    defer context.deinit();

    tracer = Tracer.init("ai.zml.examples.loader");

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);
    const devices = platform.getDevices();
    const memory_kind: pjrtx.Memory.Kind = if (platform.target == .cpu) .host_pinned else .device;

    const elapsed_init = timer.lap();
    log.warn("Initialized context and platform with {d} devices in {d}ms", .{ devices.len, elapsed_init / std.time.ns_per_ms });

    const trace_post_init = tracer.frameStart("Main post context/platform init");
    defer tracer.frameEnd(trace_post_init, "Main post context/platform init");

    log.warn("Discovering model...", .{});
    const trace_discovery = tracer.frameStart("Weights discovery");

    var resolver = ModelPathResolver.init(allocator);
    defer resolver.deinit();

    const uri: ResourceURI = resolver.resolve(path) catch |err| {
        log.err("Error resolving model path: {any}", .{err});
        std.process.exit(1);
    };

    var resource: Resource = try .init(allocator, uri);
    defer resource.deinit();

    const resource_type_buffer = try allocator.alloc(u8, BUF_8_KB);
    defer allocator.free(resource_type_buffer);

    const resource_type: ResourceType = try resolveResourceType(&resource, resource_type_buffer);

    const buffer_reader = try allocator.alloc(u8, BUF_64_MB);
    defer allocator.free(buffer_reader);

    var reader = try resource.reader(buffer_reader, .{});

    var registry: TensorRegistry = undefined;
    defer registry.deinit();

    var resource_index: ?ResourceIndex = null;
    defer if (resource_index) |*ri| ri.deinit();

    switch (resource_type) {
        .safetensors => {
            registry = .init(allocator);

            try parseSafetensors(
                &registry,
                resource.uri(),
                reader.interface(),
            );
        },
        .index => {
            resource_index = try parseSafetensorsIndex(allocator, &resource, reader.interface());

            registry = try .initWithMetadata(allocator, resource_index.?.metadata);

            const buffer_index_reader = try allocator.alloc(u8, BUF_16_MB);
            defer allocator.free(buffer_index_reader);

            var index_entries = resource_index.?.map.iterator();
            while (index_entries.next()) |entry| {
                var subresource: Resource = try .init(allocator, entry.key_ptr.*);
                defer subresource.deinit();

                var subresource_reader = try subresource.reader(buffer_index_reader, .{});

                try parseSafetensors(
                    &registry,
                    subresource.uri(),
                    subresource_reader.interface(),
                );
            }
        },
        .unknown => {
            log.err("Unknown resource type for {s}", .{resource.uri().path.percent_encoded});
            std.process.exit(1);
        },
    }

    const elapsed_discovery = timer.lap();
    tracer.frameEnd(trace_discovery, "Weights discovery");
    log.warn("Discovered {d} tensors in model ({d:.2} GB) in {d}ms", .{ registry.tensors.count(), registry.totalBytes() / (1024 * 1024 * 1024), elapsed_discovery / std.time.ns_per_ms });

    log.warn("Applying sharding information...", .{});
    try addExampleShardingMetadata(&registry);
    try annotateShapesWithSharding(&registry);

    const elapsed_sharding = timer.lap();
    log.warn("Applied sharding information in {d}ms", .{elapsed_sharding / std.time.ns_per_ms});

    log.warn("Sorting tensors by source and offset...", .{});
    const tensors = blk: {
        const ts = registry.tensors.values();

        std.mem.sort(Tensor, ts, {}, struct {
            fn lessThan(_: void, a: Tensor, b: Tensor) bool {
                const name_cmp = std.mem.order(u8, a.resource_uri.path.percent_encoded, b.resource_uri.path.percent_encoded);
                return switch (name_cmp) {
                    .lt => true,
                    .gt => false,
                    .eq => a.offset < b.offset,
                };
            }
        }.lessThan);

        break :blk ts;
    };
    const elapsed_sorting = timer.lap();
    log.warn("Sorted tensors in {d}ms", .{elapsed_sorting / std.time.ns_per_ms});

    log.warn("Allocating DMA buffers", .{});
    const trace_allocation = tracer.frameStart("Buffers allocation and DMA mapping");

    const writer_buffer = try allocator.alignedAlloc(u8, .fromByteUnits(BUF_4_KB), BUF_64_MB);
    defer allocator.free(writer_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, writer_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, writer_buffer) catch unreachable;

    const io_reader_buf = try allocator.alignedAlloc(u8, .fromByteUnits(BUF_4_KB), BUF_64_MB);
    defer allocator.free(io_reader_buf);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, io_reader_buf);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, io_reader_buf) catch unreachable;

    const elapsed_preparation = timer.lap();
    tracer.frameEnd(trace_allocation, "Buffers allocation and DMA mapping");
    log.warn("Prepared for tensor processing in {d}ms", .{elapsed_preparation / std.time.ns_per_ms});

    try warmupDevices(allocator, platform, devices);

    log.warn("Starting tensor processing stream...", .{});
    timer.reset();

    const trace_processing = tracer.frameStart("Tensor Processing Stream");

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    var sum_total_bytes_copied: u64 = 0;
    var sum_total_bytes_read: u64 = 0;

    for (tensors) |tensor| {
        var tensor_timer = try std.time.Timer.start();
        const trace = tracer.frameStart("Tensor Processing");
        defer tracer.frameEnd(trace, "Tensor Processing");

        log.info("Processing {s}/{s} dims={any} size={d} offset={d}", .{
            tensor.resource_uri.path.percent_encoded,
            tensor.name,
            tensor.shape.dims(),
            tensor.byteSize(),
            tensor.offset,
        });

        const trace_arena_reset = tracer.frameStart("Arena reset");
        _ = arena.reset(.retain_capacity);
        tracer.frameEnd(trace_arena_reset, "Arena reset");

        const arena_allocator = arena.allocator();
        const shards = try computeShards(arena_allocator, tensor, devices);

        var tensor_resource: Resource = try .init(allocator, tensor.resource_uri);
        defer tensor_resource.deinit();

        // todo: better handling of different resource types
        if (std.mem.eql(u8, tensor_resource.scheme(), S3Resource.scheme)) {
            tensor_resource.s3.authenticator = try AwsAuthenticator.init(arena_allocator);
        }

        var tensor_reader = try tensor_resource.reader(io_reader_buf, .{
            .offset = tensor.offset,
            .use_aligned_reader = true,
            .use_direct_io = true,
        });
        defer tensor_reader.deinit();

        var tensor_reader_limited: std.io.Reader.Limited = .init(tensor_reader.interface(), .limited64(tensor.byteSize()), &.{});

        var device_writers: std.ArrayList(DeviceWriter) = try .initCapacity(arena_allocator, devices.len);
        errdefer {
            for (device_writers.items) |*device_writer| {
                device_writer.deinit();
            }
            device_writers.deinit(arena_allocator);
        }
        for (0..devices.len) |i| {
            device_writers.appendAssumeCapacity(try .init(platform, shards[i], memory_kind));
        }

        var tensor_writer: TensorWriter = .init(device_writers.items, writer_buffer);
        const bytes_copied = try tensor_reader_limited.interface.streamRemaining(&tensor_writer.interface);
        try tensor_writer.interface.flush();

        const elapsed_tensor = tensor_timer.lap();
        const mb_copied = @as(f64, @floatFromInt(bytes_copied)) / (BUF_1_MB);
        const rate = if (elapsed_tensor > 0) mb_copied / (@as(f64, @floatFromInt(elapsed_tensor)) / 1_000_000_000.0) else 0;
        log.info("Loaded tensor in {d:.2}ms ({d:.2} MB at {d:.2} MB/s)", .{ elapsed_tensor / std.time.ns_per_ms, mb_copied, rate });

        std.debug.assert(bytes_copied == tensor.shape.byteSize());

        for (shards) |shard| {
            sum_total_bytes_copied += shard.byteSize();
        }
        sum_total_bytes_read += tensor.shape.byteSize();
    }
    tracer.frameEnd(trace_processing, "Tensor Processing Stream");

    const elapsed = timer.read();
    const gb_copied = @as(f64, @floatFromInt(sum_total_bytes_copied)) / (1.0 * 1024 * 1024 * 1024);
    const gb_read = @as(f64, @floatFromInt(sum_total_bytes_read)) / (1.0 * 1024 * 1024 * 1024);
    const read_rate = if (elapsed > 0) gb_read / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0) else 0;
    const copy_rate = if (elapsed > 0) gb_copied / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0) else 0;
    log.warn("All tensors loaded in {d}ms ({d:.2} GB read at {d:.2} GB/s, {d:.2} GB copied at {d:.2} GB/s)", .{ elapsed / std.time.ns_per_ms, gb_read, read_rate, gb_copied, copy_rate });
}

// all code below is unmodified (or slightly) / imported strucs / funcs from zml

pub fn bufferTypeFromDtype(dt: DataType) pjrtx.BufferType {
    return switch (dt) {
        inline else => |tag| @field(pjrtx.BufferType, @tagName(tag)),
    };
}

const minor_to_major: [Shape.MAX_RANK]i64 = blk: {
    var min_to_maj: [Shape.MAX_RANK]i64 = undefined;
    for (0..Shape.MAX_RANK) |i| {
        min_to_maj[i] = @intCast(Shape.MAX_RANK - i - 1);
    }
    break :blk min_to_maj;
};
