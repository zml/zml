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
const BUF_4_MB = 4 * MB;
const BUF_8_MB = 8 * MB;
const BUF_16_MB = 16 * MB;
const BUF_32_MB = 32 * MB;
const BUF_64_MB = 64 * MB;
const BUF_128_MB = 128 * MB;
const BUF_256_MB = 256 * MB;

var tracer: Tracer = undefined;

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

    pub fn init(allocator: std.mem.Allocator) Registry {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tensors = .{},
            .metadata = .{},
            .checksums = .{},
        };
    }

    pub fn deinit(self: *Registry) void {
        const allocator = self.arena.allocator();
        self.checksums.deinit(allocator);
        self.tensors.deinit(allocator);
        self.metadata.deinit(allocator);
        self.arena.deinit();
    }

    pub fn totalBytes(self: *Registry) u64 {
        var total: u64 = 0;

        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            total += entry.value_ptr.byteSize();
        }

        return total;
    }
};

pub const SafetensorsSource = []const u8; // rename source to chunk
pub const SafetensorsSources = std.ArrayList(SafetensorsSource);

fn parseSafetensorsIndex(
    allocator: std.mem.Allocator,
    reader: *std.io.Reader,
    registry: *Registry,
) !SafetensorsSources {
    var chunks: SafetensorsSources = .{};

    const registry_allocator = registry.arena.allocator();
    var json_reader: std.json.Reader = .init(registry_allocator, reader);

    const index = try std.json.parseFromTokenSourceLeaky(
        std.json.Value,
        registry_allocator,
        &json_reader,
        .{ .allocate = .alloc_if_needed },
    );

    const weight_map = index.object.get("weight_map").?.object;
    var it = weight_map.iterator();

    while (it.next()) |entry| {
        const filename = entry.value_ptr.string;
        try chunks.append(allocator, try registry_allocator.dupe(u8, filename));
    }

    if (index.object.get("__metadata__")) |metadata| {
        var prefix_buf: [BUF_1_KB]u8 = undefined;
        try parseMetadata(registry, StringBuilder.initBuffer(&prefix_buf), metadata);
    }

    return chunks;
}

fn parseSafetensors(
    allocator: std.mem.Allocator,
    registry: *Registry,
    reader: *std.io.Reader,
    source: SafetensorsSource,
) !void {
    const registry_allocator = registry.arena.allocator();
    const json_header_length: usize = @intCast(try reader.takeInt(u64, .little));
    const json_data = try allocator.alloc(u8, json_header_length);
    defer allocator.free(json_data);

    try reader.readSliceAll(json_data);

    const data_start_offset = 8 + json_header_length;
    const metadata = try std.json.parseFromSliceLeaky(std.json.Value, registry_allocator, json_data, .{});

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
            .source_name = try registry_allocator.dupe(u8, source),
            .name = try registry_allocator.dupe(u8, key),
            .shape = shape,
            .offset = data_start_offset + start,
        };

        try registry.tensors.put(registry_allocator, key, tensor);
    }
}

const DeviceWriter = struct {
    platform: Platform,
    shard: Shard,
    transfer_manager: *pjrtx.AsyncHostToDeviceTransferManager,

    bytes_written: u64,
    interface: std.io.Writer,

    can_process_last_event: bool = true,

    pub fn init(platform: Platform, shard: Shard, buffer: []u8) !DeviceWriter {
        const trace = tracer.frameStart("DeviceWriter.init");
        defer tracer.frameEnd(trace, "DeviceWriter.init");

        const memories = try shard.device.addressableMemories(platform.pjrt_api);
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

        const trace_tm_create = tracer.frameStart("DeviceWriter.init.createBuffersForAsyncHostToDevice");
        const tm = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
            .shape_specs = &.{shape_spec},
            .memory = memory,
        });
        tracer.frameEnd(trace_tm_create, "DeviceWriter.init.createBuffersForAsyncHostToDevice");

        return .{
            .platform = platform,
            .shard = shard,
            .transfer_manager = tm,
            .bytes_written = 0,
            .interface = .{ .vtable = &vtable, .buffer = buffer, .end = 0 },
        };
    }

    fn drain(w: *std.io.Writer, data: []const []const u8, splat: usize) std.io.Writer.Error!usize {
        const trace = tracer.frameStart("DeviceWriter.drain");
        defer tracer.frameEnd(trace, "DeviceWriter.drain");

        const self = @as(*DeviceWriter, @alignCast(@fieldParentPtr("interface", w)));

        if (w.end > 0) {
            const event = self.transfer_manager.transferData(self.platform.pjrt_api, 0, w.buffered(), @intCast(self.bytes_written), false) catch |err| {
                log.err("Error during transferData: {}", .{err});
                return error.WriteFailed;
            };
            event.awaitBlocking(self.platform.pjrt_api) catch |err| {
                log.err("Error during awaitBlocking: {}", .{err});
                return error.WriteFailed;
            };
            self.bytes_written += w.end;
            w.end = 0;
        }

        var total_written: usize = 0;

        for (data) |d| {
            const event = self.transfer_manager.transferData(self.platform.pjrt_api, 0, d, @intCast(self.bytes_written), false) catch |err| {
                log.err("Error during transferData: {}", .{err});
                return error.WriteFailed;
            };
            event.awaitBlocking(self.platform.pjrt_api) catch |err| {
                log.err("Error during awaitBlocking: {}", .{err});
                return error.WriteFailed;
            };
            self.bytes_written += d.len;
            total_written += d.len;
        }

        if (splat > 1 and data.len > 0) {
            const last_slice = data[data.len - 1];
            for (0..splat - 1) |_| {
                const event = self.transfer_manager.transferData(self.platform.pjrt_api, 0, last_slice, @intCast(self.bytes_written), false) catch |err| {
                    log.err("Error during transferData: {}", .{err});
                    return error.WriteFailed;
                };
                event.awaitBlocking(self.platform.pjrt_api) catch |err| {
                    log.err("Error during awaitBlocking: {}", .{err});
                    return error.WriteFailed;
                };
                self.bytes_written += last_slice.len;
                total_written += last_slice.len;
            }
        }

        const is_last_transfer = self.bytes_written == self.shard.byteSize();

        if (is_last_transfer and self.can_process_last_event) {
            const last_event = self.transfer_manager.transferData(self.platform.pjrt_api, 0, &.{}, @intCast(self.bytes_written), true) catch |err| {
                log.err("Error during transferData: {}", .{err});
                return error.WriteFailed;
            };
            last_event.awaitBlocking(self.platform.pjrt_api) catch |err| {
                log.err("Error during awaitBlocking: {}", .{err});
                return error.WriteFailed;
            };
            self.can_process_last_event = false;
            self.transfer_manager.deinit(self.platform.pjrt_api);
            self.transfer_manager = undefined;
        }

        return total_written;
    }

    fn flush(w: *std.io.Writer) !void {
        _ = try drain(w, &.{}, 1);
    }

    const vtable: std.io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
        .rebase = std.io.Writer.defaultRebase,
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

        if (w.end > 0) {
            if (is_sharded) {
                _ = try self.processChunkSharded(w.buffered());
            } else {
                _ = try self.processChunkReplicated(w.buffered());
            }
        }

        w.end = 0;

        var total_written_from_data: usize = 0;

        for (data) |d| {
            const written = if (is_sharded)
                try self.processChunkSharded(d)
            else
                try self.processChunkReplicated(d);

            total_written_from_data += written;
        }

        return total_written_from_data;
    }

    fn processChunkReplicated(self: *TensorWriter, data: []const u8) !usize {
        const trace = tracer.frameStart("TensorWriter.processChunkReplicated");
        defer tracer.frameEnd(trace, "TensorWriter.processChunkReplicated");

        for (self.device_writers) |*dw| {
            try dw.interface.writeAll(data);
        }

        self.bytes_written_to_tensor += data.len;
        return data.len;
    }

    fn processChunkSharded(self: *TensorWriter, data: []const u8) !usize {
        const trace = tracer.frameStart("TensorWriter.processChunkSharded");
        defer tracer.frameEnd(trace, "TensorWriter.processChunkSharded");

        var data_offset: usize = 0;
        while (data_offset < data.len) {
            const total_tensor_size = self.shard_size * self.device_writers.len;
            const current_tensor_offset = self.bytes_written_to_tensor + data_offset;

            if (current_tensor_offset >= total_tensor_size) {
                break;
            }

            const current_shard_idx: usize = @intCast(current_tensor_offset / self.shard_size);
            const offset_in_shard = current_tensor_offset % self.shard_size;
            const remaining_in_shard = self.shard_size - offset_in_shard;

            const remaining_in_data = data.len - data_offset;
            const chunk_size = @min(remaining_in_data, remaining_in_shard);

            if (chunk_size == 0) break;

            const chunk_to_write = data[data_offset .. data_offset + chunk_size];
            try self.device_writers[current_shard_idx].interface.writeAll(chunk_to_write);

            data_offset += chunk_size;
        }

        self.bytes_written_to_tensor += data_offset;
        return data_offset;
    }

    fn flush(w: *std.io.Writer) !void {
        _ = try drain(w, &.{}, 1);

        const self = @as(*TensorWriter, @alignCast(@fieldParentPtr("interface", w)));

        for (self.device_writers) |*dw| {
            try dw.interface.flush();
        }
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
    var timer = try std.time.Timer.start();

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
    tracer.event("Initialized tracer");

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    const devices = platform.getDevices();

    const elapsed_init = timer.lap();
    log.info("--- Initialized context and platform with {d} devices in {d}ms ---", .{ devices.len, elapsed_init / std.time.ns_per_ms });

    const trace_post_init = tracer.frameStart("Main post context/platform init");
    defer tracer.frameEnd(trace_post_init, "Main post context/platform init");

    log.info("--- Discovering model... ---", .{});
    var registry: Registry = .init(allocator);
    defer registry.deinit();

    var files: std.StringHashMapUnmanaged(std.fs.File) = .{};
    defer {
        var it = files.iterator();

        while (it.next()) |entry| {
            entry.value_ptr.close();
            allocator.free(entry.key_ptr.*);
        }

        files.deinit(allocator);
    }

    if (std.mem.endsWith(u8, file_path, ".safetensors.index.json")) {
        const index_file = try std.fs.openFileAbsolute(file_path, .{ .mode = .read_only });

        const index_reader_buffer = try allocator.alloc(u8, BUF_4_MB);
        defer allocator.free(index_reader_buffer);

        var index_reader = index_file.reader(index_reader_buffer);
        var chunks = try parseSafetensorsIndex(allocator, &index_reader.interface, &registry);
        defer chunks.deinit(allocator);

        const file_reader_buffer = try allocator.alloc(u8, BUF_4_MB);
        defer allocator.free(file_reader_buffer);

        for (chunks.items) |chunk| {
            if (files.get(chunk)) |_| {
                // model.safetensors.index.json weight map values may contain duplicates
                continue;
            }

            const chunk_file_path = try std.fs.path.join(allocator, &.{ std.fs.path.dirname(file_path).?, chunk });
            defer allocator.free(chunk_file_path);

            const chunk_file = try std.fs.openFileAbsolute(chunk_file_path, .{ .mode = .read_only });
            try files.put(allocator, try allocator.dupe(u8, std.fs.path.basename(chunk_file_path)), chunk_file);

            var chunk_reader = chunk_file.reader(file_reader_buffer);

            try parseSafetensors(allocator, &registry, &chunk_reader.interface, chunk);
        }
    } else {
        const file_reader_buffer = try allocator.alloc(u8, BUF_4_MB);
        defer allocator.free(file_reader_buffer);

        const file = try std.fs.openFileAbsolute(file_path, .{ .mode = .read_only });
        try files.put(allocator, try allocator.dupe(u8, std.fs.path.basename(file_path)), file);

        var chunk_reader = file.reader(file_reader_buffer);

        try parseSafetensors(allocator, &registry, &chunk_reader.interface, std.fs.path.basename(file_path));
    }

    const elapsed_discovery = timer.lap();
    log.info("--- Discovered {d} tensors in model ({d:.2} GB) in {d}ms ---", .{ registry.tensors.count(), registry.totalBytes() / (1024 * 1024 * 1024), elapsed_discovery / std.time.ns_per_ms });

    log.info("--- Applying sharding information... ---", .{});
    try addExampleShardingMetadata(&registry);
    try annotateShapesWithSharding(&registry);

    const elapsed_sharding = timer.lap();
    log.info("--- Applied sharding information in {d}ms ---", .{elapsed_sharding / std.time.ns_per_ms});

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
    const elapsed_sorting = timer.lap();
    log.info("--- Sorted tensors in {d}ms ---", .{elapsed_sorting / std.time.ns_per_ms});

    const verify_checksums = false;

    if (verify_checksums) {
        log.info("--- Pre-computing checksums from disk... ---", .{});

        const checksum_file_buffer = try allocator.alloc(u8, BUF_4_MB);
        defer allocator.free(checksum_file_buffer);

        const checksum_pump_buffer = try allocator.alloc(u8, BUF_16_MB);
        defer allocator.free(checksum_pump_buffer);

        const registry_allocator = registry.arena.allocator();

        for (tensors) |tensor| {
            if (tensor.shape.byteSize() == 0) continue;

            var hasher: std.crypto.hash.sha2.Sha256 = .init(.{});

            const file = files.get(tensor.source_name).?;
            try file.seekTo(tensor.offset);

            var file_reader = file.reader(checksum_file_buffer);
            var limited_reader: std.io.Reader.Limited = .init(&file_reader.interface, .limited64(tensor.shape.byteSize()), &.{});
            var reader = &limited_reader.interface;

            while (true) {
                const bytes_read = reader.readSliceShort(checksum_pump_buffer) catch |err| {
                    if (err == error.EndOfStream) break;
                    return err;
                };

                if (bytes_read == 0) break;

                hasher.update(checksum_pump_buffer[0..bytes_read]);
            }

            var digest: [32]u8 = undefined;
            hasher.final(&digest);

            try registry.checksums.put(registry_allocator, tensor.name, digest);
        }

        log.warn("--- Pre-computed {d} checksums in {d}ms ---", .{ registry.checksums.count(), timer.lap() / std.time.ns_per_ms });
    }

    log.info("--- Allocating buffers... ---", .{});
    const DMA_STAGING_BUFFER_SIZE = BUF_128_MB * platform.getDevices().len;

    const dma_buffer = try allocator.alloc(u8, DMA_STAGING_BUFFER_SIZE);
    defer allocator.free(dma_buffer);

    try platform.pjrt_client.dmaMap(platform.pjrt_api, dma_buffer);
    defer platform.pjrt_client.dmaUnmap(platform.pjrt_api, dma_buffer) catch unreachable;

    const file_io_buffer = try allocator.alloc(u8, BUF_128_MB);
    defer allocator.free(file_io_buffer);

    const tensor_reader_buffer: [0]u8 = undefined;
    const tensor_writer_buffer = try allocator.alloc(u8, BUF_128_MB);
    defer allocator.free(tensor_writer_buffer);

    const elasped_preparation = timer.lap();
    log.info("--- Prepared for tensor processing in {d}ms ---", .{elasped_preparation / std.time.ns_per_ms});

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var sum_total_bytes_copied: u64 = 0;

    log.info("--- Starting tensor processing stream... ---", .{});
    timer.reset();

    for (tensors) |tensor| {
        var tensor_timer = try std.time.Timer.start();
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

        const file = files.get(tensor.source_name).?;
        try file.seekTo(tensor.offset);

        var file_reader = file.reader(file_io_buffer);

        var tensor_reader: std.io.Reader.Limited = .init(&file_reader.interface, .limited64(tensor.shape.byteSize()), &tensor_reader_buffer);
        var device_writers: std.ArrayList(DeviceWriter) = try .initCapacity(arena_allocator, devices.len);

        const shards = try computeShards(arena_allocator, tensor, devices);
        defer arena_allocator.free(shards);

        for (0..devices.len) |i| {
            try device_writers.append(allocator, try .init(
                platform,
                shards[i],
                dma_buffer[i * BUF_8_MB .. (i + 1) * BUF_8_MB],
            ));
        }

        var tensor_writer: TensorWriter = .init(arena_allocator, device_writers.items, tensor_writer_buffer);

        const bytes_copied = try tensor_reader.interface.streamRemaining(&tensor_writer.interface);
        try tensor_writer.interface.flush();

        const elapsed_tensor = tensor_timer.lap();
        const mb_copied = @as(f64, @floatFromInt(bytes_copied)) / (1.0 * 1024 * 1024);
        const rate = if (elapsed_tensor > 0) mb_copied / (@as(f64, @floatFromInt(elapsed_tensor)) / 1_000_000_000.0) else 0;
        log.info("Loaded tensor in {d:.2}ms ({d:.2} MB at {d:.2} MB/s) ---", .{ elapsed_tensor / std.time.ns_per_ms, mb_copied, rate });

        std.debug.assert(bytes_copied == tensor.shape.byteSize());

        sum_total_bytes_copied += bytes_copied;
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
