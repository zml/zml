const asynk = @import("async");
const builtin = @import("builtin");
const c = @import("c");
const std = @import("std");
const stdx = @import("stdx");

const zml = @import("zml.zig");
const posix = @import("posix.zig");

pub const gguf = @import("aio/gguf.zig");
pub const nemo = @import("aio/nemo.zig");
pub const safetensors = @import("aio/safetensors.zig");
pub const tinyllama = @import("aio/tinyllama.zig");
pub const torch = @import("aio/torch.zig");
pub const yaml = @import("aio/yaml.zig");

pub const log = std.log.scoped(.@"zml/aio");
const HostBuffer = @import("hostbuffer.zig").HostBuffer;
const pjrt = @import("pjrtx.zig");
const Memory = pjrt.Memory;
const Event = pjrt.Event;
const Shape = @import("shape.zig").Shape;

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(gguf);
    std.testing.refAllDecls(nemo);
    std.testing.refAllDecls(safetensors);
    std.testing.refAllDecls(torch);
    std.testing.refAllDecls(yaml);
}

/// Detects the format of the model file (base on filename) and open it.
pub fn detectFormatAndOpen(allocator: std.mem.Allocator, model_path: []const u8) !BufferStore {
    return if (std.mem.endsWith(u8, model_path, ".safetensors"))
        try safetensors.open(allocator, model_path)
    else if (std.mem.endsWith(u8, model_path, ".safetensors.index.json"))
        try safetensors.open(allocator, model_path)
    else if (std.mem.endsWith(u8, model_path, ".gguf"))
        try gguf.open(allocator, model_path)
    else if (std.mem.endsWith(u8, model_path, ".pt"))
        try torch.open(allocator, model_path)
    else if (std.mem.endsWith(u8, model_path, ".tinyllama"))
        try tinyllama.open(allocator, model_path)
    else {
        std.debug.panic("File extension not recognized: {s}", .{model_path});
    };
}

/// Creates a Model struct with tensor shapes read from the given BufferStore.
/// The result can be used to pass to `compileModel`.
///
/// * The `Tensor` field `Model.a.b` will be populated with a `Tensor`
/// whose shape is read from the "a.b" tensor.
/// * If `Model` contains a list of layers, then the field:
/// `Model.layers[2].a.b` will be populated from the "layers.2.a.b" tensor.
pub fn populateModel(comptime Model: type, allocator: std.mem.Allocator, buffer_store: BufferStore) !Model {
    return populateModelWithPrefix(Model, allocator, buffer_store, "");
}

/// Creates a Model struct with tensor shapes read from the given TensorStore,
/// using a given prefix.
/// The result can be used to pass to `compileWithModel`.
///
/// * The `Tensor` field `Model.a.b` will be populated with a `Tensor`
/// whose shape is read from the "prefix.a.b" tensor.
/// * If `Model` contains a list of layers, then the field:
/// `Model.layers[2].a.b` will be populated from the "prefix.layers.2.a.b" tensor.
pub fn populateModelWithPrefix(comptime Model: type, allocator: std.mem.Allocator, store: BufferStore, prefix: []const u8) !Model {
    var model: Model = undefined;

    var prefix_builder: PrefixBuilder = .{};
    try prefix_builder.push(allocator, prefix);
    defer prefix_builder.deinit(allocator);

    const unique_id = zml.Tensor._reserveIdRange(@intCast(store.buffers.count()));
    const ok = _populateStruct(allocator, &prefix_builder, unique_id, store, &model, true) catch |err| {
        std.debug.panic("Can't populate model of type {s}: {s}", .{ @typeName(type), @errorName(err) });
    };
    if (!ok) return error.TensorNotFound;
    return model;
}

/// A struct containing all the buffers and metadata found in a model file.
pub const BufferStore = struct {
    pub const BufferEntry = struct {
        buffer: ?zml.Buffer = null,
        shape: Shape,
        data: []const u8,
        ready_event: ?*Event = null,
        transfer_manger_buffer_index: usize,
    };

    pub const Buffers = std.StringArrayHashMapUnmanaged(BufferEntry);
    pub const Metadatas = std.StringArrayHashMapUnmanaged(Metadata);
    // pub const TransferManagerSet = std.AutoArrayHashMapUnmanaged(*zml.platform.TransferManager, void);

    arena: std.heap.ArenaAllocator,
    files: []MemoryMappedFile = &.{},
    buffers: Buffers = .{},
    _metadata: Metadatas = .{},
    /// set of transfer managers created in the buffer store:
    /// one per (device, memoryKind)
    // transfer_managers: TransferManagerSet = .{},
    transfer_manager: ?zml.platform.TransferManager = null,

    /// Create an empty BufferStore. Takes owneship of the given files.
    pub fn init(allocator: std.mem.Allocator, files: []const MemoryMappedFile) error{OutOfMemory}!BufferStore {
        var self: zml.aio.BufferStore = .{
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
        self.files = try self.arena.allocator().dupe(MemoryMappedFile, files);
        return self;
    }

    /// when loading shapes (and data) from file, store them here
    /// we rely on the map preserving order of insertion
    pub fn registerBuffer(
        self: *BufferStore,
        alloc: std.mem.Allocator,
        key: []const u8,
        shape: Shape,
        data: []const u8,
    ) !void {
        log.info("Registering buffer {d}: {s}", .{ self.buffers.count(), key });
        try self.buffers.put(alloc, key, .{
            .data = data,
            .shape = shape,
            .transfer_manger_buffer_index = self.buffers.count(),
        });
    }

    pub fn deinit(self: *BufferStore) void {
        for (self.files) |*file| {
            file.deinit();
        }
        if (self.transfer_manager) |*xferman| {
            xferman.deinit();
        }
        self.arena.deinit();
    }

    pub fn getBuffer(self: BufferStore, key: []const u8) ?zml.Buffer {
        const entry = self.buffers.get(key) orelse return null;
        return entry.buffer;
    }

    pub fn getShape(self: BufferStore, key: []const u8) ?zml.Shape {
        const entry = self.buffers.get(key) orelse return null;
        return entry.shape;
    }

    pub fn starTransferToDevice(self: *BufferStore, platform: zml.Platform, memory_kind: Memory.Kind) ![]*Event {
        // TODO: work out how many transfer managers we need
        //       probably one per (memory_kind, device)
        if (self.transfer_manager != null) {
            return error.TransferAlreadyStarted;
        }

        // retrieve the shapes and data slices of all the buffers
        // we rely on the map preserving order of insertion
        var shapes = try std.ArrayList(Shape).initCapacity(self.arena.allocator(), self.buffers.count());
        var data_slices = try std.ArrayList([]const u8).initCapacity(self.arena.allocator(), self.buffers.count());

        var it = self.buffers.iterator();
        var idx: usize = 0;
        while (it.next()) |kv| {
            // for (self.buffers.values(), 0..) |reg_buf, idx| {
            const reg_buf = kv.value_ptr;
            // stdx.debug.assert(idx == reg_buf.transfer_manger_buffer_index, "Internal error: TransferManager indices out of sync: buffer_idx={d}, transfer_manager_buffer_index={d}, registered buffers: {any}", .{ idx, reg_buf.transfer_manger_buffer_index, self.buffers.values() });
            log.info("{d}, {d}", .{ idx, reg_buf.transfer_manger_buffer_index });

            shapes.appendAssumeCapacity(reg_buf.shape);
            data_slices.appendAssumeCapacity(reg_buf.data);
            idx += 1;
        }

        log.info("done", .{});

        self.transfer_manager = try zml.platform.TransferManager.init(
            self.arena.allocator(),
            platform,
            memory_kind,
            try shapes.toOwnedSlice(),
        );

        const slices = try data_slices.toOwnedSlice();
        log.debug("About to transfer {d} slices", .{slices.len});
        const events = try self.transfer_manager.?.transferDataMany(slices, .{});
        log.debug("TransferManager created {d} events", .{events.len});
        for (self.buffers.values(), events) |*reg_buf, event| {
            // TODO: for now, we only use our one and only transfer manager
            reg_buf.buffer = self.transfer_manager.?.buffer(reg_buf.transfer_manger_buffer_index);
            reg_buf.ready_event = event;
        }
        return events;
    }

    /// Count layers starting with the given prefix.
    pub fn countLayers(self: BufferStore, prefix: []const u8) usize {
        // Note: This is kinda inefficient
        const digit_start_index = prefix.len + 1;
        var it = self.buffers.iterator();
        var maybe_max_index: ?usize = null;
        while (it.next()) |entry| {
            if (!std.mem.startsWith(u8, entry.key_ptr.*, prefix)) continue;

            const next_dot_index = std.mem.indexOfScalarPos(u8, entry.key_ptr.*, digit_start_index, '.') orelse entry.key_ptr.len;
            const index = std.fmt.parseInt(usize, entry.key_ptr.*[digit_start_index..next_dot_index], 10) catch continue;
            if (maybe_max_index) |*max_index| {
                max_index.* = @max(max_index.*, index);
            } else {
                maybe_max_index = index;
            }
        }

        return if (maybe_max_index) |index| index + 1 else 0;
    }

    pub fn metadata(self: BufferStore, key: []const u8, comptime tag: std.meta.FieldEnum(Metadata)) ?std.meta.FieldType(Metadata, tag) {
        const wrapped_value = self._metadata.get(key) orelse return null;

        if (wrapped_value != tag) {
            zml.log.err("Tried to interpret metadata '{s}' as {}, but was of type {}", .{ key, tag, wrapped_value });
            @panic("invalid metadata type");
        }
        return @field(wrapped_value, @tagName(tag));
    }

    pub fn metadataSlice(self: BufferStore, key: []const u8, comptime tag: Metadata.ItemType) ?[]const tag.toZigType() {
        const wrapped_value = self._metadata.get(key) orelse return null;
        const true_tag = std.meta.stringToEnum(std.meta.FieldEnum(Metadata), @tagName(tag)).?;
        if (wrapped_value == true_tag) {
            return @field(wrapped_value, "array_" ++ @tagName(tag));
        }

        return null;
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
            else => @panic("Unsupported type for zml.aio.Value: " ++ @typeName(@TypeOf(x))),
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
            else => @panic("Unsupported type for zml.aio.Value: " ++ @typeName(@TypeOf(any_slice))),
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

/// A file containing contiguous/non-contiguous buffers, that can be read with mmap
/// (assumes contiguous if `strides` is `null`).
/// This struct is meant to be wrapped into a format specific struct, like io.gguf.File.
pub const MemoryMappedFile = struct {
    /// underlying file handle
    file: asynk.File,
    data: []align(std.mem.page_size) const u8,
    data_offset: u64 = 0,

    pub fn init(file: asynk.File) !MemoryMappedFile {
        const data_len: usize = (try file.stat()).size;
        const data_ = try asynk.callBlocking(std.posix.mmap, .{
            null,
            data_len,
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle(),
            0,
        });

        try asynk.callBlocking(posix.madvise, .{
            data_.ptr,
            @as(usize, @intCast(data_.len)),
            @as(u32, @intCast(c.MADV_SEQUENTIAL)),
        });

        return .{
            .file = file,
            .data = data_,
        };
    }

    pub fn mappedSlice(self: MemoryMappedFile, start: usize, len: usize) []const u8 {
        return self.data[self.data_offset + start ..][0..len];
    }

    pub fn deinit(self: *MemoryMappedFile) void {
        std.posix.munmap(self.data);
        self.file.close() catch @panic("failed to close file");
        self.* = undefined;
    }
};

/// Helper handling prefix building.
///
/// This allows to easily push/pop prefixes and handles the generation of the string with the correct format.
const PrefixBuilder = struct {
    /// Stores the computed prefix.
    data: std.ArrayListUnmanaged(u8) = .{},
    /// Stack storing the size of the intermediary prefix.
    subprefixes: std.ArrayListUnmanaged(u32) = .{},

    pub fn deinit(self: *PrefixBuilder, allocator: std.mem.Allocator) void {
        self.data.deinit(allocator);
        self.subprefixes.deinit(allocator);
    }

    pub fn push(self: *PrefixBuilder, allocator: std.mem.Allocator, prefix: []const u8) !void {
        const old_len: u32 = @intCast(self.data.items.len);
        try self.subprefixes.append(allocator, old_len);
        errdefer _ = self.subprefixes.pop();

        if (old_len == 0) {
            try self.data.appendSlice(allocator, prefix);
        } else {
            try self.data.ensureUnusedCapacity(allocator, prefix.len + 1);
            self.data.appendAssumeCapacity('.');
            self.data.appendSliceAssumeCapacity(prefix);
        }
    }

    pub fn pushDigit(self: *PrefixBuilder, allocator: std.mem.Allocator, idx: usize) !void {
        const old_len: u32 = @intCast(self.data.items.len);
        try self.subprefixes.append(allocator, old_len);
        errdefer _ = self.subprefixes.pop();

        try self.data.ensureUnusedCapacity(allocator, 16);
        if (old_len > 0) {
            self.data.appendAssumeCapacity('.');
        }
        try self.data.writer(allocator).print("{d}", .{idx});
    }

    pub fn pop(self: *PrefixBuilder) void {
        const last_prefix_len = self.subprefixes.popOrNull() orelse unreachable;
        self.data.shrinkRetainingCapacity(last_prefix_len);
    }
};

fn _populateStruct(
    allocator: std.mem.Allocator,
    prefix_builder: *PrefixBuilder,
    unique_id: u64,
    buffer_store: BufferStore,
    obj: anytype,
    required: bool,
) !bool {
    const err_msg = "_populateStruct must be called with a pointer to type. Received ";
    const type_info, const T = switch (@typeInfo(@TypeOf(obj))) {
        .Pointer => |ptr_info| switch (ptr_info.size) {
            .One => .{ @typeInfo(ptr_info.child), ptr_info.child },
            else => @compileError(err_msg ++ @typeName(@TypeOf(obj))),
        },
        else => @compileError(err_msg ++ @typeName(@TypeOf(obj))),
    };

    const prefix = prefix_builder.data.items;
    if (T == zml.Tensor) {
        return if (buffer_store.buffers.getIndex(prefix)) |entry_idx| {
            log.info("Trying to get buffer `{s}` with index {d}", .{ prefix, entry_idx });
            const shape = buffer_store.getShape(prefix).?;
            obj.* = zml.Tensor{
                ._shape = shape,
                ._id = .{ .buffer_id = unique_id + entry_idx },
                ._donation = .input_buffer,
            };
            return true;
        } else {
            if (required) {
                log.err("Tensor not found: {s} ({d})", .{ prefix, buffer_store.buffers.count() });
            }
            return false;
        };
    }

    log.info("_populateStruct `{s}`, {*}", .{ prefix, obj });
    return switch (type_info) {
        .Pointer => |ptr_info| {
            log.info("     is pointer", {});
            if (ptr_info.size == .Slice) {
                obj.* = &.{};

                const len = buffer_store.countLayers(prefix);
                if (len > 0) {
                    obj.* = try allocator.alloc(ptr_info.child, len);

                    for (obj.*, 0..) |*value, i| {
                        try prefix_builder.pushDigit(allocator, i);
                        defer prefix_builder.pop();
                        const found = try _populateStruct(allocator, prefix_builder, unique_id, buffer_store, value, required);
                        if (!found) {
                            log.err("Not able to load {s} as {s}", .{ prefix_builder.data.items, @typeName(ptr_info.child) });
                            return false;
                        }
                    }
                } else if (required) {
                    log.warn("No layer found at {s}", .{prefix});
                }
                return true;
            } else {
                log.err("{s} - {s}: {s} type not supported", .{ @src().fn_name, prefix, @typeName(T) });
                return false;
            }
        },
        .Array => |arr_info| {
            log.info("     is array", {});
            for (obj, 0..) |*value, i| {
                try prefix_builder.pushDigit(allocator, i);
                defer prefix_builder.pop();
                const found = try _populateStruct(allocator, prefix_builder, unique_id, buffer_store, value, required);
                if (!found) {
                    log.err("Not able to load {s} as {s}", .{ prefix_builder.data.items, @typeName(arr_info.child) });
                    return false;
                }
            }
            return true;
        },
        .Struct => |struct_info| {
            log.info("     is struct", .{});
            var partial_struct = false;
            inline for (struct_info.fields) |field| {
                log.info("     field: `{s}`", .{field.name});
                if (field.is_comptime or @sizeOf(field.type) == 0) continue;
                try prefix_builder.push(allocator, field.name);
                defer prefix_builder.pop();

                var has_default = false;
                if (field.default_value) |_| has_default = true;
                const field_found = try _populateStruct(allocator, prefix_builder, unique_id, buffer_store, &@field(obj, field.name), required and !has_default);

                log.info("     field: `{s}` : found = {}", .{ field.name, field_found });

                partial_struct = partial_struct or field_found;
                if (!field_found) {
                    if (field.default_value) |v| {
                        @field(obj, field.name) = @as(*const field.type, @alignCast(@ptrCast(v))).*;
                    } else {
                        if (partial_struct) {
                            log.warn("Incomplete metadata '{0s}': {1s}. Missing field: '{2s}'. '{0s}' will be ignored.", .{ prefix, @typeName(T), field.name });
                            obj.* = undefined;
                            return false;
                        }

                        return false;
                    }
                }
            }
            return true;
        },
        .Optional => |opt_info| {
            log.info("     is optional", .{});
            obj.* = @as(opt_info.child, undefined);
            const found = try _populateStruct(allocator, prefix_builder, unique_id, buffer_store, &(obj.*.?), false);
            if (!found) obj.* = null;
            return true;
        },
        .Int => {
            log.info("     is int", .{});
            obj.* = undefined;
            return true;
        },
        .Float => {
            log.info("     is float", .{});
            obj.* = undefined;
            return true;
        },
        .Void => {
            log.info("     is void", .{});
            return true;
        },
        .Union => {
            log.info("     is union", .{});
            return true;
        },
        else => if (required) {
            log.err("{s}: {s} type not supported", .{ prefix, @typeName(T) });
            return error.UnsupportedMetadataType;
        } else return false,
    };
}

test populateModel {
    const Model = struct {
        a: zml.Tensor,
        b: struct { a: zml.Tensor, b: u32 },
        c: []zml.Tensor,
        d: []struct { a: zml.Tensor, b: u32 },
        e: struct { zml.Tensor, u32, struct { a: u32, b: zml.Tensor, c: void } },
        f: ?zml.Tensor,
        g: ?zml.Tensor,

        // Create a fake HostBuffer, we use the given integer to identify the created buffer.
        fn _newHostBuffer(n: u32) zml.HostBuffer {
            return .{ ._shape = zml.Shape.init(.{n}, .f16), .data = undefined };
        }
    };

    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    var store: BufferStore = .{ .arena = arena_state };
    try store.buffers.ensureUnusedCapacity(arena_state.allocator(), 16);
    store.buffers.putAssumeCapacity("a", Model._newHostBuffer(10));
    store.buffers.putAssumeCapacity("b.a", Model._newHostBuffer(20));
    store.buffers.putAssumeCapacity("c.0", Model._newHostBuffer(30));
    store.buffers.putAssumeCapacity("c.1", Model._newHostBuffer(31));
    store.buffers.putAssumeCapacity("c.2", Model._newHostBuffer(32));
    store.buffers.putAssumeCapacity("d.0.a", Model._newHostBuffer(40));
    store.buffers.putAssumeCapacity("d.1.a", Model._newHostBuffer(41));
    store.buffers.putAssumeCapacity("d.2.a", Model._newHostBuffer(42));
    store.buffers.putAssumeCapacity("e.0", Model._newHostBuffer(50));
    store.buffers.putAssumeCapacity("e.2.b", Model._newHostBuffer(51));
    store.buffers.putAssumeCapacity("f", Model._newHostBuffer(60));
    // no entry for g.
    store.buffers.putAssumeCapacity("unused_entry", Model._newHostBuffer(1000));

    const model = try populateModel(Model, arena_state.allocator(), store);

    try std.testing.expectEqual(10, model.a.dim(0));
    try std.testing.expectEqual(20, model.b.a.dim(0));
    try std.testing.expectEqual(3, model.c.len);
    try std.testing.expectEqual(30, model.c[0].dim(0));
    try std.testing.expectEqual(31, model.c[1].dim(0));
    try std.testing.expectEqual(32, model.c[2].dim(0));
    try std.testing.expectEqual(3, model.d.len);
    try std.testing.expectEqual(40, model.d[0].a.dim(0));
    try std.testing.expectEqual(41, model.d[1].a.dim(0));
    try std.testing.expectEqual(42, model.d[2].a.dim(0));
    try std.testing.expectEqual(50, model.e[0].dim(0));
    try std.testing.expectEqual(51, model.e[2].b.dim(0));
    try std.testing.expectEqual(60, model.f.?.dim(0));
    try std.testing.expectEqual(null, model.g);
}

/// Creates a bufferized version of a Model from the given BufferStore. For details about
/// bufferization, see the documentation of Bufferized(T).
///
/// This will represent the weights of the model, loaded on a specific platform.
/// It can be used with a `module.Exe` (a compiled version of the same Model), to make a
/// `module.ExeWithWeights` ready to be called.
///
/// The `init_args` are used to initialize the non Buffer fields, using `Model.init` function.
pub fn loadBuffers(
    comptime Model: type,
    init_args: anytype,
    buffer_store: BufferStore,
    allocator: std.mem.Allocator,
    platform: zml.Platform,
) !zml.Bufferized(Model) {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var model: Model = try zml.aio.populateModel(Model, arena, buffer_store);

    // If the Model has an "init" function, call it with the given parameters.
    if (@hasDecl(Model, "init")) {
        @call(.auto, Model.init, .{&model} ++ init_args);
    } else {
        stdx.debug.assertComptime(@TypeOf(init_args) == void or @TypeOf(init_args) == @TypeOf(.{}), "Model of type {} has no init function, so `loadBuffers` should be call with init_args set to {{}} (void)", .{Model});
    }

    return loadModelBuffersWithPrefix(Model, model, buffer_store, allocator, platform, "");
}

/// Creates a bufferized version of a Model from the given BufferStore. For details about
/// bufferization, see the documentation of Bufferized(T).
///
/// This will represent the weights of the model, loaded on a specific platform.
/// It can be used with a `module.Exe` (a compiled version of the same Model), to make a
/// `module.ExeWithWeights` ready to be called.
pub fn loadModelBuffers(
    comptime Model: type,
    model: Model,
    buffer_store: BufferStore,
    allocator: std.mem.Allocator,
    platform: zml.Platform,
) !zml.Bufferized(Model) {
    return try loadModelBuffersWithPrefix(Model, model, buffer_store, allocator, platform, "");
}

/// Creates a bufferized version of a Model from the given BufferStore and the given prefix.
/// For details about bufferization, see the documentation of Bufferized(T).
///
/// This will represent the weights of the model, loaded on a specific platform.
/// It can be used with a `module.Exe` (a compiled version of the same Model), to make a
/// `module.ExeWithWeights` ready to be called.
pub fn loadModelBuffersWithPrefix(
    comptime Model: type,
    model: Model,
    buffer_store: BufferStore,
    allocator: std.mem.Allocator,
    platform: zml.Platform,
    prefix: []const u8,
) !zml.Bufferized(Model) {
    // Allocate the bufferized version.
    // We copy the shape, and let visitStructAndLoadBuffer write the other fields.
    // to write them just afterward.
    var res: zml.Bufferized(Model) = undefined;
    try zml.meta.mapAlloc(struct {
        pub fn initBuffer(_: void, tensor: zml.Tensor) zml.Buffer {
            return .{ ._shape = tensor.shape(), ._api = undefined, ._shards = undefined };
        }
    }.initBuffer, allocator, {}, model, &res);

    var prefix_builder: PrefixBuilder = .{};
    try prefix_builder.push(allocator, prefix);
    defer prefix_builder.deinit(allocator);

    try visitStructAndLoadBuffer(allocator, &prefix_builder, buffer_store, &res, platform);
    return res;
}

/// Takes a bufferized version of a `model`, ie a mirror struct of the `model`, and deinit all the
/// Buffer found.
pub fn unloadBuffers(model: anytype) void {
    zml.meta.visit((struct {
        fn cb(_: void, buffer: *zml.Buffer) void {
            buffer.deinit();
        }
    }).cb, {}, model);
}

/// Assists in debuggigng `BufferNotFound` error
/// This is useful when a buffer key is not found and you want to identify possible alternatives (or typos)
fn findSimilarBufferKeys(original_key: []const u8, store: BufferStore, temp_allocator: std.mem.Allocator) void {
    const suffixes = [_][]const u8{ "", ".weight", ".bias" };
    var shown_keys = std.StringHashMap(void).init(temp_allocator);
    defer shown_keys.deinit();

    // remove suffix .weight and .bias
    var base_key = original_key;
    for (suffixes) |suffix| {
        if (std.mem.endsWith(u8, original_key, suffix)) {
            base_key = original_key[0 .. original_key.len - suffix.len];
            break;
        }
    }

    // first test: look for exact matches
    var matches: usize = 0;
    var it = store.buffers.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.startsWith(u8, key, base_key)) {
            if (matches == 0) log.warn("Similar buffers found:", .{});
            if (!shown_keys.contains(key)) {
                log.warn("  - {s}: {}", .{ key, entry.value_ptr.*.shape() });
                shown_keys.put(key, {}) catch continue;
                matches += 1;
            }
        }
    }

    // second test: progressive partial matches
    if (matches == 0) {
        var components = std.mem.splitScalar(u8, base_key, '.');
        while (components.next()) |component| {
            matches = 0;
            it = store.buffers.iterator();
            while (it.next()) |entry| {
                const key = entry.key_ptr.*;
                if (std.mem.indexOf(u8, key, component) != null and !shown_keys.contains(key)) {
                    if (matches == 0) log.warn("Partial matches for '{s}':", .{component});
                    log.warn("  - {s}: {}", .{ key, entry.value_ptr.*.shape() });
                    shown_keys.put(key, {}) catch continue;
                    matches += 1;
                    if (matches >= 5) break;
                }
            }
            if (matches > 0) break;
        }
    }
}

/// await all buffers in the given struct
pub fn awaitAll(buffers: anytype) !void {
    // TODO: implement once we have async buffers.
    _ = buffers;
}

fn visitStructAndLoadBuffer(allocator: std.mem.Allocator, prefix_builder: *PrefixBuilder, buffer_store: BufferStore, obj: anytype, platform: zml.Platform) !void {
    const err_msg = "visitStructAndLoadBuffer must be called with a pointer to type. Received ";
    const type_info, const T = switch (@typeInfo(@TypeOf(obj))) {
        .Pointer => |ptr_info| switch (ptr_info.size) {
            .One => .{ @typeInfo(ptr_info.child), ptr_info.child },
            else => @compileError(err_msg ++ @typeName(@TypeOf(obj))),
        },
        else => @compileError(err_msg ++ @typeName(@TypeOf(obj))),
    };

    const prefix = prefix_builder.data.items;
    if (T == zml.Buffer) {
        return if (buffer_store.getBuffer(prefix)) |buffer| {
            // obj._shape has been set inside `loadModelBuffersWithPrefix`, before calling us.
            log.debug("Loading buffer {s} ({})", .{ prefix, obj._shape });
            stdx.debug.assert(buffer.shape().eql(obj._shape), "loadModelBuffers expects to find the same shapes in the model and in the buffer store, got {} and {} for tensor {s}", .{ obj._shape, buffer, prefix });
            obj.* = buffer;
        } else {
            log.err("Buffer not found: {s}", .{prefix});

            findSimilarBufferKeys(prefix, buffer_store, allocator);

            return error.BufferNotFound;
        };
    } else if (T == zml.Shape) return;

    switch (type_info) {
        .Pointer => |ptr_info| {
            if (ptr_info.size == .Slice) {
                for (obj.*, 0..) |*value, i| {
                    try prefix_builder.pushDigit(allocator, i);
                    defer prefix_builder.pop();

                    try visitStructAndLoadBuffer(allocator, prefix_builder, buffer_store, value, platform);
                }
            } else stdx.debug.compileError("type not supported by visitStructAndLoadBuffer: {}", .{T});
        },
        .Array => {
            for (obj, 0..) |*value, i| {
                try prefix_builder.pushDigit(allocator, i);
                defer prefix_builder.pop();
                try visitStructAndLoadBuffer(allocator, prefix_builder, buffer_store, value, platform);
            }
        },

        .Struct => |struct_info| {
            inline for (struct_info.fields) |field| {
                if (field.is_comptime or @sizeOf(field.type) == 0) continue;
                try prefix_builder.push(allocator, field.name);
                defer prefix_builder.pop();

                try visitStructAndLoadBuffer(allocator, prefix_builder, buffer_store, &@field(obj, field.name), platform);
            }
        },
        .Optional => {
            if (obj.*) |*obj_val| {
                try visitStructAndLoadBuffer(allocator, prefix_builder, buffer_store, obj_val, platform);
            }
        },
        else => {},
    }
}
