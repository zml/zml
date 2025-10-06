const std = @import("std");

const asynk = @import("async");
const stdx = @import("stdx");
const zml = @import("zml");

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
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
const BUF_8_MB = 8 * MB;
const BUF_16_MB = 16 * MB;
const BUF_32_MB = 32 * MB;
const BUF_64_MB = 64 * MB;
const BUF_128_MB = 128 * MB;
const BUF_256_MB = 256 * MB;

var tracer: Tracer = undefined;

// Primitives & Safetensors Parser (Unchanged)
const Tensor = struct {
    source_name: []const u8,
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

pub const Registry = struct {
    pub const Tensors = std.StringArrayHashMapUnmanaged(Tensor);
    pub const Metadatas = std.StringArrayHashMapUnmanaged(Metadata);
    pub const Checksums = std.StringArrayHashMapUnmanaged([32]u8);

    arena: std.heap.ArenaAllocator,
    tensors: Tensors,
    metadata: Metadatas,
    checksums: Checksums,

    pub fn deinit(self: *Registry) void {
        const allocator = self.arena.allocator();
        self.checksums.deinit(allocator);
        self.tensors.deinit(allocator);
        self.metadata.deinit(allocator);
        self.arena.deinit();
    }
};

pub fn registerSafetensors(allocator: std.mem.Allocator, source: *Source, path: []const u8) !Registry {
    var registry: Registry = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
        .tensors = .{},
        .metadata = .{},
        .checksums = .{},
    };
    errdefer registry.deinit();

    var processing_arena = std.heap.ArenaAllocator.init(allocator);
    defer processing_arena.deinit();

    const processing_allocator = processing_arena.allocator();

    var io_buffer: [BUF_64_KB]u8 = undefined;
    var source_reader: Source.Reader = undefined;

    try source.initReader(path, &io_buffer, &source_reader);
    const source_iface = sourceInterface(&source_reader);

    if (std.mem.endsWith(u8, path, ".safetensors.index.json")) {
        try parseSafetensorsIndex(
            processing_allocator,
            &registry,
            source,
            &source_reader,
            source_iface,
            &io_buffer,
        );
    } else {
        try parseSafetensors(processing_allocator, &registry, source_iface, path);
    }

    return registry;
}

fn parseSafetensorsIndex(
    allocator: std.mem.Allocator,
    registry: *Registry,
    source: *Source,
    source_reader: *Source.Reader,
    io_reader: *std.io.Reader,
    io_buffer: []u8,
) !void {
    var json_reader: std.json.Reader = .init(allocator, io_reader);
    const index = try std.json.parseFromTokenSourceLeaky(
        std.json.Value,
        allocator,
        &json_reader,
        .{ .allocate = .alloc_if_needed },
    );

    const weight_map = index.object.get("weight_map").?.object;
    var it = weight_map.iterator();

    while (it.next()) |entry| {
        const filename = entry.value_ptr.string;

        try source.initReader(filename, io_buffer, source_reader);
        const chunk_reader = sourceInterface(source_reader);

        try parseSafetensors(allocator, registry, chunk_reader, filename);
    }

    if (index.object.get("__metadata__")) |metadata| {
        var prefix_buf: [BUF_1_KB]u8 = undefined;
        try parseMetadata(registry, StringBuilder.initBuffer(&prefix_buf), metadata);
    }
}

fn parseSafetensors(
    allocator: std.mem.Allocator,
    registry: *Registry,
    reader: *std.io.Reader,
    source_name: []const u8,
) !void {
    const registry_allocator = registry.arena.allocator();
    const json_header_length: usize = @intCast(try reader.takeInt(u64, .little));
    const json_data = try allocator.alloc(u8, json_header_length);
    defer allocator.free(json_data);

    try reader.readSliceAll(json_data);

    const data_start_offset = 8 + json_header_length;
    const metadata = try std.json.parseFromSliceLeaky(std.json.Value, allocator, json_data, .{});

    var it = metadata.object.iterator();

    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        if (std.mem.eql(u8, key, "__metadata__")) {
            var prefix_buf: [BUF_1_KB]u8 = undefined;
            try parseMetadata(registry, StringBuilder.initBuffer(&prefix_buf), value);
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

        const shape = Shape.init(dims.constSlice(), dtype);
        const size_in_bytes = end - start;
        std.debug.assert(size_in_bytes == shape.byteSize());

        const tensor: Tensor = .{
            .source_name = try registry_allocator.dupe(u8, source_name),
            .name = try registry_allocator.dupe(u8, key),
            .shape = shape,
            .offset = data_start_offset + start,
        };

        try registry.tensors.put(registry_allocator, key, tensor);
    }
}

pub const Source = union(enum) {
    fs: *FsSource,

    pub const Reader = union(enum) {
        fs: std.fs.File.Reader,
    };

    pub fn initReader(self: Source, path: []const u8, buffer: []u8, reader_mem: *Reader) !void {
        return switch (self) {
            .fs => |fs_source| try fs_source.initReader(path, buffer, reader_mem),
        };
    }
};

pub fn sourceInterface(reader: *Source.Reader) *std.io.Reader {
    return switch (reader.*) {
        .fs => |*r| &r.interface,
    };
}
pub const FsSource = struct {
    const ManagedFile = struct {
        file: std.fs.File,
    };

    allocator: std.mem.Allocator,
    base_dir: []const u8,
    path_to_file_map: std.StringHashMapUnmanaged(ManagedFile),
    path_to_direct_file_map: std.StringHashMapUnmanaged(ManagedFile),

    pub fn init(allocator: std.mem.Allocator, base_dir: []const u8) FsSource {
        return .{
            .allocator = allocator,
            .base_dir = base_dir,
            .path_to_file_map = .{},
            .path_to_direct_file_map = .{},
        };
    }

    pub fn deinit(self: *FsSource) void {
        var it = self.path_to_file_map.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.file.close();
        }
        self.path_to_file_map.deinit(self.allocator);

        var it2 = self.path_to_direct_file_map.iterator();
        while (it2.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.file.close();
        }
        self.path_to_direct_file_map.deinit(self.allocator);
    }

    pub fn initReader(self: *FsSource, path: []const u8, buffer: []u8, source_reader: *Source.Reader) !void {
        const managed_file = if (self.path_to_file_map.get(path)) |mf| mf else blk: {
            const full_path = try std.fs.path.join(self.allocator, &.{ self.base_dir, path });
            defer self.allocator.free(full_path);

            const file_fd = try std.posix.open(
                full_path,
                .{ .ACCMODE = .RDONLY, .DIRECT = false },
                0,
            );

            const file: std.fs.File = .{ .handle = file_fd };
            errdefer file.close();

            const path_dupe = try self.allocator.dupe(u8, path);
            errdefer self.allocator.free(path_dupe);

            try self.path_to_file_map.put(self.allocator, path_dupe, .{ .file = file });

            break :blk self.path_to_file_map.get(path).?;
        };

        try managed_file.file.seekTo(0);

        source_reader.* = .{ .fs = managed_file.file.reader(buffer) };
    }
};

fn copyHostToDevice(
    api: *const pjrt.Api,
    shard: Shard,
    host_src: []const u8,
    device_dst_offset: u64,
) !void {
    const trace = tracer.frameStart("copyHostToDevice");
    defer tracer.frameEnd(trace, "copyHostToDevice");

    const device_id = shard.device.getLocalHardwareId(api);
    const tensor = shard.tensor;

    log.info("Copying {d} B to device {d} from {s}/{s} dims={any} size={d} offset={d} - tensor dims={any} size={d} offset={d}", .{
        host_src.len,
        device_id,
        tensor.source_name,
        tensor.name,
        shard.shape.dims(),
        shard.byteSize(),
        device_dst_offset,
        tensor.shape.dims(),
        tensor.byteSize(),
        tensor.offset,
    });

    return;
}

const DeviceWriter = struct {
    platform: Platform,
    shard: Shard,

    buffer: []u8,
    bytes_written: u64,
    interface: std.io.Writer,

    pub fn init(platform: Platform, shard: Shard, buffer: []u8) DeviceWriter {
        return .{
            .platform = platform,
            .shard = shard,
            .buffer = buffer,
            .bytes_written = 0,
            .interface = .{ .vtable = &vtable, .buffer = &.{} },
        };
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const trace = tracer.frameStart("DeviceWriter.drain");
        defer tracer.frameEnd(trace, "DeviceWriter.drain");

        const self = @as(*DeviceWriter, @alignCast(@fieldParentPtr("interface", w)));

        var total_written: usize = 0;

        for (data) |d| total_written += try self.write(d);

        if (splat > 1 and data.len > 0) {
            const last_slice = data[data.len - 1];
            for (0..splat - 1) |_| total_written += try self.write(last_slice);
        }

        return total_written;
    }

    fn write(self: *DeviceWriter, data: []const u8) !usize {
        const trace = tracer.frameStart("DeviceWriter.write");
        defer tracer.frameEnd(trace, "DeviceWriter.write");

        var slice = data;
        var written: usize = 0;

        while (slice.len > 0) {
            const chunk_size = @min(slice.len, self.buffer.len);
            const chunk = slice[0..chunk_size];

            const memcpy_trace = tracer.frameStart("memcpy to staging buffer");
            @memcpy(self.buffer[0..chunk_size], chunk);
            tracer.frameEnd(memcpy_trace, "memcpy to staging buffer");

            try copyHostToDevice(
                self.platform.pjrt_api,
                self.shard,
                self.buffer[0..chunk_size],
                self.bytes_written,
            );

            self.bytes_written += chunk_size;
            written += chunk_size;
            slice = slice[chunk_size..];
        }

        return written;
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = std.io.Writer.noopFlush,
        .rebase = std.io.Writer.unreachableRebase,
    };
};

const TensorWriter = struct {
    allocator: std.mem.Allocator,
    shard_size: u64,

    device_writers: []DeviceWriter,
    bytes_written_to_tensor: u64,

    interface: std.io.Writer,

    pub fn init(
        allocator: std.mem.Allocator,
        device_writers: []DeviceWriter,
        buffer: []u8,
    ) TensorWriter {
        return .{
            .allocator = allocator,
            .shard_size = device_writers[0].shard.byteSize(),
            .bytes_written_to_tensor = 0,
            .device_writers = device_writers,
            .interface = .{ .vtable = &vtable, .buffer = buffer, .end = 0 },
        };
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const trace = tracer.frameStart("TensorWriter.drain");
        defer tracer.frameEnd(trace, "TensorWriter.drain");

        const self = @as(*TensorWriter, @alignCast(@fieldParentPtr("interface", w)));

        std.debug.assert(splat == 1);

        const is_sharded = self.device_writers[0].shard.byteSize() < self.device_writers[0].shard.tensor.byteSize();

        var all_data: std.ArrayList([]const u8) = .{};
        defer all_data.deinit(self.allocator);

        if (w.buffered().len > 0) {
            all_data.append(self.allocator, w.buffered()) catch return error.WriteFailed;
        }

        all_data.appendSlice(self.allocator, data) catch return error.WriteFailed;

        var total_written: usize = 0;

        for (all_data.items) |d| {
            if (is_sharded) {
                total_written += self.drainSharded(d) catch return error.WriteFailed;
            } else {
                total_written += self.drainReplicated(d) catch return error.WriteFailed;
            }
        }

        w.end = 0;

        return total_written;
    }

    fn drainReplicated(self: *TensorWriter, data: []const u8) !usize {
        const trace = tracer.frameStart("TensorWriter.drainReplicated");
        defer tracer.frameEnd(trace, "TensorWriter.drainReplicated");

        for (self.device_writers) |*dw| dw.interface.writeAll(data) catch return error.WriteFailed;

        self.bytes_written_to_tensor += data.len;

        return data.len;
    }

    fn drainSharded(self: *TensorWriter, data: []const u8) !usize {
        const trace = tracer.frameStart("TensorWriter.drainSharded");
        defer tracer.frameEnd(trace, "TensorWriter.drainSharded");

        var slice = data;
        var written_this_drain: usize = 0;

        while (slice.len > 0) {
            const total_tensor_size = self.shard_size * self.device_writers.len;

            if (self.bytes_written_to_tensor >= total_tensor_size) return written_this_drain;

            const current_shard_idx: usize = @intCast(self.bytes_written_to_tensor / self.shard_size);
            const offset_in_shard = self.bytes_written_to_tensor % self.shard_size;
            const remaining_in_shard = self.shard_size - offset_in_shard;
            const chunk_size = @min(slice.len, remaining_in_shard);

            try self.device_writers[current_shard_idx].interface.writeAll(slice[0..chunk_size]);

            self.bytes_written_to_tensor += chunk_size;
            written_this_drain += chunk_size;
            slice = slice[chunk_size..];
        }

        return written_this_drain;
    }

    fn flush(w: *std.io.Writer) !void {
        _ = try drain(w, &.{}, 1);
    }

    fn rebase(w: *std.io.Writer, preserve: usize, capacity: usize) !void {
        _ = preserve;
        try w.flush();
        std.debug.assert(w.buffer.len >= capacity);
    }

    const vtable = std.io.Writer.VTable{ .drain = drain, .flush = flush, .rebase = rebase };
};

// This is an example of how sharding metadata might be added to a model registry.
// In a real application, this metadata might come from a config file or be inferred from the
// model architecture. Here, we hardcode some example tensor names and shard them on axis 1.
fn addExampleShardingMetadata(registry: *Registry) !void {
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
fn annotateShapesWithSharding(registry: *Registry) !void {
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
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = std.process.args();
    _ = args.next().?;

    const file_path = args.next() orelse {
        log.err("Usage: bazel run //examples/loader /path/to/model.safetensors...", .{});
        return;
    };

    log.info("--- Initializing context and platform... ---", .{});
    var context = try Context.init();
    defer context.deinit();

    tracer = Tracer.init("ai.zml.examples.loader");

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    const devices = platform.getDevices();

    log.info("--- Loading model metadata... ---", .{});
    var fs_source = FsSource.init(allocator, std.fs.path.dirname(file_path) orelse ".");
    defer fs_source.deinit();

    var source: Source = .{ .fs = &fs_source };

    var registry = try registerSafetensors(allocator, &source, std.fs.path.basename(file_path));
    defer registry.deinit();

    log.info("--- Applying sharding information... ---", .{});
    try addExampleShardingMetadata(&registry);
    try annotateShapesWithSharding(&registry);

    log.info("--- Sorting tensors by source and offset... ---", .{});
    const tensors = blk: {
        const ts = registry.tensors.values();

        std.mem.sort(Tensor, ts, {}, struct {
            fn lessThan(_: void, a: Tensor, b: Tensor) bool {
                const name_cmp = std.mem.order(u8, a.source_name, b.source_name);
                return switch (name_cmp) {
                    .lt => true,
                    .gt => false,
                    .eq => a.offset < b.offset,
                };
            }
        }.lessThan);

        break :blk ts;
    };

    log.info("--- Allocating DMA buffers... ---", .{});
    const DMA_STAGING_BUFFER_SIZE = BUF_8_MB * platform.getDevices().len;

    var dma_buffer = try allocator.alloc(u8, DMA_STAGING_BUFFER_SIZE);
    defer allocator.free(dma_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, dma_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, dma_buffer) catch {};

    log.info("--- Starting tensor processing stream... ---", .{});
    const file_io_buffer = try allocator.alloc(u8, BUF_128_MB);
    defer allocator.free(file_io_buffer);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var current_source_name: ?[]const u8 = null;
    var source_reader_storage: Source.Reader = undefined;
    var source_reader: *std.io.Reader = undefined;
    var file_pos: u64 = 0;
    var timer = try std.time.Timer.start();
    var sum_total_bytes_copied: u64 = 0;

    const tensor_reader_buffer: [0]u8 = undefined;
    const tensor_writer_buffer = try allocator.alloc(u8, BUF_8_MB);
    defer allocator.free(tensor_writer_buffer);

    for (tensors) |tensor| {
        const trace = tracer.frameStart("Tensor Processing");
        defer tracer.frameEnd(trace, "Tensor Processing");

        log.info("Processing {s}/{s} dims={any} size={d} offset={d}", .{
            tensor.source_name,
            tensor.name,
            tensor.shape.dims(),
            tensor.byteSize(),
            tensor.offset,
        });

        _ = arena.reset(.free_all);

        const arena_allocator = arena.allocator();

        if (if (current_source_name) |name| !std.mem.eql(u8, name, tensor.source_name) else true) {
            current_source_name = tensor.source_name;
            try fs_source.initReader(tensor.source_name, file_io_buffer, &source_reader_storage);
            source_reader = sourceInterface(&source_reader_storage);
            file_pos = 0;
        }

        std.debug.assert(tensor.offset >= file_pos);
        try source_reader.discardAll(tensor.offset - file_pos);
        file_pos = tensor.offset;

        var tensor_reader: std.io.Reader.Limited = .init(source_reader, .limited64(tensor.shape.byteSize()), &tensor_reader_buffer);
        var device_writers: std.ArrayList(DeviceWriter) = try .initCapacity(arena_allocator, devices.len);

        const shards = try computeShards(arena_allocator, tensor, devices);
        defer arena_allocator.free(shards);

        for (0..devices.len) |i| {
            try device_writers.append(allocator, .init(
                platform,
                shards[i],
                dma_buffer[i * BUF_8_MB .. (i + 1) * BUF_8_MB],
            ));
        }

        var tensor_writer: TensorWriter = .init(arena_allocator, device_writers.items, tensor_writer_buffer);

        const bytes_copied = try tensor_reader.interface.streamRemaining(&tensor_writer.interface);
        try tensor_writer.interface.flush();

        std.debug.assert(bytes_copied == tensor.shape.byteSize());

        sum_total_bytes_copied += bytes_copied;
        file_pos += bytes_copied;
    }

    const elapsed = timer.read();
    const gb_copied = @as(f64, @floatFromInt(sum_total_bytes_copied)) / (1.0 * 1024 * 1024 * 1024);
    const rate = if (elapsed > 0) gb_copied / (@as(f64, @floatFromInt(elapsed)) / 1_000_000_000.0) else 0;
    log.warn("--- All tensors loaded in {d}ms ({d:.2} GB at {d:.2} GB/s) ---", .{ elapsed / std.time.ns_per_ms, gb_copied, rate });
}

// all code below is unmodified (or slightly) / imported strucs / funcs from zml

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
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        switch (self) {
            .null => _ = try writer.write("null"),
            inline .bool, .array_bool => |b| try writer.print("{any}", .{b}),
            inline else => |v| try writer.print("{d}", .{v}),
        }
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

pub fn parseMetadata(registry: *Registry, prefix: StringBuilder, val: std.json.Value) !void {
    const allocator = registry.arena.allocator();
    const metadata = &registry.metadata;
    const key = prefix.items;
    return switch (val) {
        .null => try metadata.put(allocator, try allocator.dupe(u8, key), .null),
        .bool => |v| try metadata.put(allocator, try allocator.dupe(u8, key), .{ .bool = v }),
        .integer => |v| try metadata.put(allocator, try allocator.dupe(u8, key), .{ .int = v }),
        .float => |v| try metadata.put(allocator, try allocator.dupe(u8, key), .{ .float = v }),
        .number_string, .string => |v| try metadata.put(allocator, try allocator.dupe(u8, key), .{ .string = try allocator.dupe(u8, v) }),
        .array => |v| {
            if (v.items.len == 0) return;
            return if (validSlice(v)) |item_type| {
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
                            values[i] = @field(item, @tagName(tag));
                        }
                        break :blk .{ .array_string = values };
                    },
                    .null, .array, .object => unreachable,
                };
                try metadata.put(allocator, try allocator.dupe(u8, key), data);
            } else {
                for (v.items, 0..) |item, i| {
                    var new_prefix = prefix;
                    if (prefix.items.len > 0)
                        new_prefix.appendAssumeCapacity('.');
                    new_prefix.items.len += std.fmt.printInt(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
                    try parseMetadata(registry, new_prefix, item);
                }
            };
        },
        .object => |v| {
            var obj_iter = v.iterator();
            while (obj_iter.next()) |entry| {
                var new_prefix = prefix;
                if (prefix.items.len > 0)
                    new_prefix.appendAssumeCapacity('.');
                new_prefix.appendSliceAssumeCapacity(entry.key_ptr.*);
                try parseMetadata(registry, new_prefix, entry.value_ptr.*);
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

pub fn bufferTypeFromDtype(dt: DataType) pjrtx.BufferType {
    return switch (dt) {
        inline else => |tag| @field(pjrtx.BufferType, @tagName(tag)),
    };
}
