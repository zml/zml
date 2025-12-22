const builtin = @import("builtin");
const std = @import("std");

const stdx = @import("stdx");

pub const VFS = @import("io").VFS;

const log = std.log.scoped(.@"zml/io");

const safetensors = @import("safetensors.zig");
const Tensor = @import("tensor.zig").Tensor;
const Shape = @import("shape.zig").Shape;
const Platform = @import("platform.zig").Platform;
const Buffer = @import("buffer.zig").Buffer;
const meta = @import("meta.zig");
const Bufferized = @import("zml.zig").Bufferized;
const pjrt = @import("pjrtx.zig");

const MemoryPool = struct {
    const Slots = std.ArrayList(Slot);
    const Slot = struct {
        buf: []u8,
        capacity: usize,
        in_use: bool,
    };

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex,

    slots: Slots,
    default_capacity: usize,

    pub fn init(allocator: std.mem.Allocator, initial_count: usize, slot_size: usize) !MemoryPool {
        var slots: Slots = .{};

        for (0..initial_count) |_| {
            const buf = try allocator.alloc(u8, slot_size);
            try slots.append(allocator, .{ .buf = buf, .capacity = slot_size, .in_use = false });
        }

        return .{
            .allocator = allocator,
            .slots = slots,
            .default_capacity = slot_size,
            .mutex = .init,
        };
    }

    pub fn deinit(self: *MemoryPool, io: std.Io) void {
        self.mutex.lockUncancelable(io);

        for (self.slots.items) |slot| {
            self.allocator.free(slot.buf);
        }

        self.mutex.unlock(io);
        self.slots.deinit(self.allocator);
    }

    pub fn alloc(self: *MemoryPool, io: std.Io, n: usize) !struct { data: []u8, index: usize } {
        self.mutex.lockUncancelable(io);

        var idx: usize = 0;
        while (idx < self.slots.items.len) : (idx += 1) {
            if (!self.slots.items[idx].in_use and self.slots.items[idx].capacity >= n) {
                self.slots.items[idx].in_use = true;
                const data = self.slots.items[idx].buf[0..n];
                self.mutex.unlock(io);
                return .{ .data = data, .index = idx };
            }
        }

        const capacity = if (n > self.default_capacity) n else self.default_capacity;

        log.warn("MemoryPool: growing pool, allocating new slot of size {d} bytes", .{capacity});

        const buf = try self.allocator.alloc(u8, capacity);
        try self.slots.append(self.allocator, .{ .buf = buf, .capacity = capacity, .in_use = true });

        const new_index = self.slots.items.len - 1;
        const data = buf[0..n];

        self.mutex.unlock(io);

        return .{ .data = data, .index = new_index };
    }

    pub fn free(self: *MemoryPool, io: std.Io, index: usize) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        if (index < self.slots.items.len) {
            self.slots.items[index].in_use = false;
        }
    }
};

pub const TensorStore = struct {
    registry: *safetensors.TensorRegistry,
    id_map: std.AutoHashMapUnmanaged(usize, *safetensors.Tensor),
    allocator: std.mem.Allocator,
    pool: MemoryPool,

    pub fn fromRegistry(allocator: std.mem.Allocator, registry: *safetensors.TensorRegistry, limit: std.Io.Limit) TensorStore {
        return .{
            .registry = registry,
            .id_map = .empty,
            .allocator = allocator,
            .pool = MemoryPool.init(allocator, limit.toInt() orelse 16, 96 * 1024 * 1024) catch unreachable,
        };
    }

    pub fn deinit(self: *TensorStore, io: std.Io) void {
        self.id_map.deinit(self.allocator);
        self.pool.deinit(io);
    }

    fn bindIdToKey(self: *TensorStore, key: []const u8, id: usize) !void {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key).?;

        const gop = try self.id_map.getOrPut(self.allocator, id);
        if (gop.found_existing) {
            stdx.debug.panic("Key {s} already has an associated tensor (id: {})", .{ key, gop.key_ptr.* });
        }
        errdefer self.id_map.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = tensor_desc_ptr;
    }

    fn getPtrFromKey(self: *const TensorStore, key: []const u8) ?*safetensors.Tensor {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key) orelse return null;
        return tensor_desc_ptr;
    }

    fn getPtrFromId(self: *const TensorStore, id: usize) ?*safetensors.Tensor {
        const tensor_desc_ptr = self.id_map.get(id) orelse return null;
        return tensor_desc_ptr;
    }

    pub fn getReader(self: *const TensorStore, key: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        return self.registry.reader(io, key, buffer);
    }

    pub fn getReaderById(self: *const TensorStore, id: usize, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        const tensor_desc = self.id_map.get(id) orelse return error.NotFound;

        return safetensors.TensorReader.init(io, tensor_desc.*, buffer);
    }

    pub fn view(self: *TensorStore) View {
        return .{ .store = self };
    }

    pub const View = struct {
        store: *TensorStore,

        prefix_buffer: [256]u8 = undefined,
        prefix_length: usize = 0,

        pub fn root(self: *const View) View {
            return .{
                .store = self.store,
            };
        }

        pub fn parent(self: *const View) View {
            const slice = self.prefix() orelse unreachable;
            const index = std.mem.lastIndexOfScalar(u8, slice[0 .. slice.len - 1], '.') orelse return self.root();
            var buffer: [256]u8 = undefined;
            @memcpy(buffer[0 .. index + 1], slice[0 .. index + 1]);
            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = index + 1,
            };
        }

        pub fn withPrefix(self: *const View, prefix_: []const u8) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ }) catch unreachable;

            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        pub fn withLayer(self: *const View, index: usize) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{d}.", .{ self.prefix() orelse "", index }) catch unreachable;

            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        fn prefix(self: *const View) ?[]const u8 {
            return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
        }

        pub fn maybeCreateTensor(self: View, subkey: []const u8) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;

            const tensor: Tensor = .fromShape(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensor(self: View, subkey: []const u8) Tensor {
            return self.maybeCreateTensor(subkey).?;
        }

        pub fn maybeCreateTensorWithTags(self: View, subkey: []const u8, tagz: anytype) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;
            ptr.shape = ptr.shape.withTags(tagz);

            const tensor: Tensor = .fromShape(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensorWithTags(self: View, subkey: []const u8, tagz: anytype) Tensor {
            return self.maybeCreateTensorWithTags(subkey, tagz).?;
        }

        pub fn getShape(self: View, subkey: []const u8) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }

        pub fn getShapeOpts(self: View, subkey: []const u8, opts: struct { no_prefix: bool = false }) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = if (opts.no_prefix)
                subkey
            else b: {
                break :b std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            };
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }

        pub fn getReader(self: View, subkey: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
            var key_buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&key_buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            return self.store.getReader(key, io, buffer);
        }
    };
};

fn collectShapes(allocator: std.mem.Allocator, v: anytype) ![]Shape {
    const LocalContext = struct {
        list: *std.array_list.Managed(Shape),
    };
    var list = std.array_list.Managed(Shape).init(allocator);
    errdefer list.deinit();

    var context: LocalContext = .{ .list = &list };
    try meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) !void {
            try ctx_.list.append(tensor.shape());
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

fn collectTensorDesc(allocator: std.mem.Allocator, store: TensorStore.View, v: anytype) ![]safetensors.Tensor {
    const LocalContext = struct {
        list: *std.array_list.Managed(safetensors.Tensor),
        store: TensorStore.View,
    };
    var list = std.array_list.Managed(safetensors.Tensor).init(allocator);
    var context: LocalContext = .{ .list = &list, .store = store };
    meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) void {
            const tensor_desc = ctx_.store.store.getPtrFromId(tensor.id).?.*;
            ctx_.list.append(tensor_desc) catch unreachable;
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

pub fn loadBuffersFromId(allocator: std.mem.Allocator, io: std.Io, model: anytype, store: TensorStore.View, platform: Platform) !Bufferized(@TypeOf(model)) {
    const Model = @TypeOf(model);
    var result: Bufferized(Model) = undefined;
    initBufferizedFrom(model, &result);

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    const shapes = try collectShapes(arena.allocator(), &model);
    const tensor_descs = try collectTensorDesc(arena.allocator(), store, &model);

    const LocalContext = struct {
        tensor_descs: []safetensors.Tensor,
        shapes: []const Shape,
        platform: Platform,
        index: usize = 0,
        pool: *MemoryPool,
        store: TensorStore.View,
        allocator: std.mem.Allocator,
        io: std.Io,
        group: *std.Io.Group,
        errors: *[]?anyerror,
    };
    var group: std.Io.Group = .init;
    defer group.cancel(io);

    const num_buffers = shapes.len;
    var errors = try allocator.alloc(?anyerror, num_buffers);
    defer allocator.free(errors);
    for (errors) |*e| e.* = null;

    var context: LocalContext = .{
        .tensor_descs = tensor_descs,
        .shapes = shapes,
        .platform = platform,
        .pool = &store.store.pool,
        .store = store,
        .allocator = allocator,
        .io = io,
        .group = &group,
        .errors = &errors,
    };

    const AsyncLoadBuffer = struct {
        pub fn call(
            group_ptr: *std.Io.Group,
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: Platform,
            tensor_descs_: []safetensors.Tensor,
            pool: *MemoryPool,
            idx: usize,
            out_buffer: *Buffer,
            out_errors: *[]?anyerror,
        ) void {
            _ = pool; // autofix
            const out_errors_ref = out_errors.*;
            const tensor_desc = tensor_descs_[idx];
            const needed_bytes: usize = tensor_desc.byteSize();

            var timer = stdx.time.Timer.start() catch |err| {
                out_errors_ref[idx] = err;
                group_ptr.cancel(io_);
                return;
            };

            // const buffer_reader_ = pool.alloc(io_, needed_bytes + 4096) catch |err| {
            //     out_errors_ref[idx] = err;
            //     group_ptr.cancel(io_);
            //     return;
            // };
            // defer pool.free(io_, buffer_reader_.index);

            // // allocate writer buffer sized to the tensor
            // const buffer_writer_ = pool.alloc(io_, needed_bytes + 4096) catch |err| {
            //     out_errors_ref[idx] = err;
            //     group_ptr.cancel(io_);
            //     return;
            // };
            // defer pool.free(io_, buffer_writer_.index);

            const buffer_reader_ = allocator_.alloc(u8, @min(needed_bytes, 64 * 1024 * 1024)) catch |err| {
                out_errors_ref[idx] = err;
                group_ptr.cancel(io_);
                return;
            };
            defer allocator_.free(buffer_reader_);

            const buffer_writer_ = allocator_.alloc(u8, @min(needed_bytes, 128 * 1024 * 1024)) catch |err| {
                allocator_.free(buffer_reader_);
                out_errors_ref[idx] = err;
                group_ptr.cancel(io_);
                return;
            };
            defer allocator_.free(buffer_writer_);

            const elapsed_alloc = timer.read().ns;

            var reader = safetensors.TensorReader.init(io_, tensor_desc, buffer_reader_) catch |err| {
                out_errors_ref[idx] = err;
                group_ptr.cancel(io_);
                return;
            };
            defer reader.deinit();

            var device_writers = std.ArrayList(DeviceWriter).initCapacity(allocator_, 1) catch |err| {
                out_errors_ref[idx] = err;
                group_ptr.cancel(io_);
                return;
            };
            errdefer for (device_writers.items) |*dw| dw.deinit();
            defer device_writers.deinit(allocator_);
            const device = platform_.getDevices()[0];
            const device_writer = DeviceWriter.init(platform_, tensor_desc, device, .device) catch |err| {
                out_errors_ref[idx] = err;
                group_ptr.cancel(io_);
                return;
            };
            device_writers.appendAssumeCapacity(device_writer);
            var writer: TensorWriter = .init(device_writers.items, buffer_writer_);

            const elapsed_setup = timer.lap().ns;

            _ = reader.interface.streamRemaining(&writer.interface) catch |err| {
                out_errors_ref[idx] = err;
                group_ptr.cancel(io_);
                return;
            };

            writer.interface.flush() catch |err| {
                out_errors_ref[idx] = err;
                group_ptr.cancel(io_);
                return;
            };

            const elapsed = timer.read().ns;
            const read_mb = @as(f64, @floatFromInt(tensor_desc.byteSize())) / (1024.0 * 1024.0);
            const setup_time_ms = @as(f64, @floatFromInt(elapsed_setup)) / std.time.ns_per_ms;
            const setup_alloc_ms = @as(f64, @floatFromInt(elapsed_alloc)) / std.time.ns_per_ms;
            const setup_init_ms = setup_time_ms - setup_alloc_ms;
            const read_time_ms = @as(f64, @floatFromInt(elapsed)) / std.time.ns_per_ms;
            const read_time_s = read_time_ms / 1000.0;
            const throughput_mb_s = read_mb / read_time_s;
            log.info("Read {s} {d:.2} MB in {d:.2} ms ({d:.2} MB/s) + {d:.2} ms setup (alloc: {d:.2} ms, init: {d:.2} ms)", .{
                tensor_desc.name,
                read_mb,
                read_time_ms,
                throughput_mb_s,
                setup_time_ms,
                setup_alloc_ms,
                setup_init_ms,
            });

            const buf = device_writers.items[0].buffer() catch |err| {
                out_errors_ref[idx] = err;
                group_ptr.cancel(io_);
                return;
            };
            out_buffer.* = buf;
        }
    }.call;

    try meta.visit(struct {
        fn cb(context_: *LocalContext, buffer: *Buffer) !void {
            const idx = context_.index;
            context_.index += 1;

            context_.group.async(context_.io, AsyncLoadBuffer, .{
                context_.group,
                context_.allocator,
                context_.io,
                context_.platform,
                context_.tensor_descs,
                context_.pool,
                idx,
                buffer,
                context_.errors,
            });

            return;
        }
    }.cb, &context, &result);

    group.wait(io);

    for (errors) |e| if (e != null) return e.?;

    return result;
}

pub fn initBufferizedFrom(model: anytype, bufferized_: *Bufferized(@TypeOf(model))) void {
    const Model = @TypeOf(model);
    const type_info = @typeInfo(Bufferized(Model));
    switch (type_info) {
        .@"struct" => |struct_type_info| {
            if (Bufferized(Model) == Buffer) return;
            inline for (struct_type_info.fields) |field| {
                initBufferizedFrom(@field(model, field.name), &@field(bufferized_, field.name));
            }
        },
        .@"union" => {
            switch (model) {
                inline else => |v, tag| {
                    bufferized_.* = @unionInit(Bufferized(Model), @tagName(tag), undefined);
                    initBufferizedFrom(v, @field(bufferized_, @tagName(tag)));
                },
            }
        },
        .optional => {
            if (model == null) {
                bufferized_.* = null;
            } else {
                bufferized_.* = undefined;
                initBufferizedFrom(model.?, &bufferized_.*.?);
            }
        },
        .void, .int, .@"enum", .bool, .enum_literal, .float, .pointer, .vector => {},
        else => unreachable,
    }
}

const DeviceWriter = struct {
    platform: Platform,
    shard: safetensors.Tensor,
    transfer_manager: *pjrt.AsyncHostToDeviceTransferManager,

    events: [2]?*pjrt.Event,
    next_slot_idx: u1,

    bytes_written: u64,
    can_process_last_event: bool,

    interface: std.Io.Writer,

    pub fn init(platform: Platform, shard: safetensors.Tensor, device: *const pjrt.Device, memory_kind: pjrt.Memory.Kind) !DeviceWriter {
        const memories = device.addressableMemories(platform.pjrt_api);
        var memory = memories[0];

        if (platform.target == .cuda) {
            for (memories) |mem| {
                if (mem.kind(platform.pjrt_api) == memory_kind) {
                    memory = mem;
                    break;
                }
            }
        }

        const shape_spec = pjrt.ShapeSpec.init(shard.shape.dims(), pjrt.bufferTypeFromDtype(shard.shape.dtype()));
        const transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
            .shape_specs = &.{shape_spec},
            .memory = memory,
        });

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

    pub fn buffer(self: *DeviceWriter) !Buffer {
        const pjrt_buffer = try self.transfer_manager.retrieveBuffer(self.platform.pjrt_api, 0);
        return .fromPjrtBuffers(self.platform, self.shard.shape, &.{pjrt_buffer});
    }

    fn awaitEvent(self: *DeviceWriter, idx: u1) !void {
        if (self.events[idx]) |event| {
            try event.awaitBlocking(self.platform.pjrt_api);
            self.events[idx] = null;
        }
    }

    fn transfer(self: *DeviceWriter, data: []const u8, is_last: bool) !*pjrt.Event {
        const offset = self.bytes_written;

        defer {
            if (!is_last) self.bytes_written += data.len;
        }

        std.debug.assert(offset + data.len <= self.shard.byteSize());

        return self.transfer_manager.transferData(self.platform.pjrt_api, 0, data, @intCast(offset), is_last) catch |err| {
            log.err("PJRT transferData failed: {}", .{err});
            return error.WriteFailed;
        };
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self = @as(*DeviceWriter, @alignCast(@fieldParentPtr("interface", w)));

        std.debug.assert(w.end == 0);
        std.debug.assert(splat == 1);
        std.debug.assert(data.len == 1);

        const chunk = data[0];

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

    fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self = @as(*DeviceWriter, @alignCast(@fieldParentPtr("interface", w)));

        self.awaitEvent(0) catch return error.WriteFailed;
        self.awaitEvent(1) catch return error.WriteFailed;

        if (self.can_process_last_event) {
            std.debug.assert(self.bytes_written == self.shard.byteSize());

            const last_event = try self.transfer(&.{}, true);
            last_event.awaitBlocking(self.platform.pjrt_api) catch return error.WriteFailed;

            self.can_process_last_event = false;
        }
    }

    const vtable: std.Io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
    };
};

const TensorWriter = struct {
    device_writers: []DeviceWriter,

    chunk_size: usize,
    current_buffer_idx: u1,
    shard_size: u64,
    is_sharded: bool,
    total_bytes_processed: u64,

    buffer: []u8,
    interface: std.Io.Writer,

    pub fn init(device_writers: []DeviceWriter, buffer: []u8) TensorWriter {
        const chunk_size = buffer.len / 2;

        return .{
            .device_writers = device_writers,
            .buffer = buffer,
            .chunk_size = chunk_size,
            .shard_size = if (device_writers.len > 0) device_writers[0].shard.byteSize() else 0,
            .is_sharded = if (device_writers.len > 0)
                device_writers[0].shard.byteSize() < device_writers[0].shard.byteSize()
            else
                false,
            .current_buffer_idx = 0,
            .total_bytes_processed = 0,
            .interface = .{ .vtable = &vtable, .buffer = buffer[0..chunk_size], .end = 0 },
        };
    }

    fn process(self: *TensorWriter, chunk_to_process: []const u8) !void {
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
        const chunk_to_process = self.interface.buffered();

        if (chunk_to_process.len > 0) {
            try self.process(chunk_to_process);
        }

        self.swap();
    }

    fn rebase(w: *std.Io.Writer, preserve: usize, minimum_len: usize) std.Io.Writer.Error!void {
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

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *TensorWriter = @alignCast(@fieldParentPtr("interface", w));

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
    fn flush(w: *std.Io.Writer) !void {
        const self: *TensorWriter = @alignCast(@fieldParentPtr("interface", w));

        try self.processAndSwap();

        if (w.end > 0) {
            try self.processAndSwap();
        }

        for (self.device_writers) |*dw| {
            try dw.interface.flush();
        }
    }

    const vtable: std.Io.Writer.VTable = .{
        .drain = drain,
        .flush = flush,
        .rebase = rebase,
    };
};
