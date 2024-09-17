const builtin = @import("builtin");
const asynk = @import("async");
const std = @import("std");
const zml = @import("zml.zig");
const pjrt = @import("pjrtx.zig");
const c = @import("c");
const posix = @import("posix.zig");

pub const gguf = @import("aio/gguf.zig");
pub const nemo = @import("aio/nemo.zig");
pub const safetensors = @import("aio/safetensors.zig");
pub const sentencepiece = @import("aio/sentencepiece.zig");
pub const tinyllama = @import("aio/tinyllama.zig");
pub const torch = @import("aio/torch.zig");
pub const yaml = @import("aio/yaml.zig");

pub const log = std.log.scoped(.zml_aio);
pub const Value = @import("aio/value.zig").Value;
const HostBuffer = @import("hostbuffer.zig").HostBuffer;

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

pub fn detectFormatAndLoadTokenizer(allocator: std.mem.Allocator, tokenizer_path: []const u8) !zml.tokenizer.Tokenizer {
    return if (std.mem.endsWith(u8, tokenizer_path, ".json"))
        try zml.tokenizer.fromHfJson(allocator, tokenizer_path)
    else if (std.mem.endsWith(u8, tokenizer_path, ".gguf")) {
        const store = try gguf.open(allocator, tokenizer_path);
        return gguf.getGgufTokenizer(store, allocator);
    } else if (std.mem.endsWith(u8, tokenizer_path, ".pb") or std.mem.endsWith(u8, tokenizer_path, ".model"))
        try sentencepiece.loadTokenizerFromPath(allocator, tokenizer_path)
    else if (std.mem.endsWith(u8, tokenizer_path, ".tinyllama"))
        try zml.aio.tinyllama.loadTokenizer(allocator, tokenizer_path, 32000)
    else {
        zml.log.err("Failed to recognized tokenizer format of: {s}", .{tokenizer_path});
        return error.FormatNotRecognized;
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

    var unique_id = zml.Tensor.reserveIdRange(@intCast(store.buffers.count()));
    const ok = _populateStruct(allocator, &prefix_builder, &unique_id, store, &model, true) catch |err| {
        std.debug.panic("Can't populate model of type {s}: {s}", .{ @typeName(type), @errorName(err) });
    };
    if (!ok) return error.TensorNotFound;
    return model;
}

/// A struct containing all the buffers and metadata found in a model file.
pub const BufferStore = struct {
    pub const Buffers = std.StringArrayHashMapUnmanaged(HostBuffer);
    pub const Metadata = std.StringArrayHashMapUnmanaged(Value);

    arena: std.heap.ArenaAllocator,
    files: []MemoryMappedFile = &.{},
    buffers: Buffers = .{},
    _metadata: Metadata = .{},

    pub fn deinit(self: BufferStore) void {
        for (self.files) |*file| file.deinit();
        self.arena.deinit();
    }

    pub fn get(self: BufferStore, key: []const u8) ?HostBuffer {
        return self.buffers.get(key);
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

    pub fn metadata(self: BufferStore, key: []const u8, comptime tag: std.meta.FieldEnum(Value)) ?std.meta.FieldType(Value, tag) {
        const wrapped_value = self._metadata.get(key) orelse return null;

        if (wrapped_value != tag) {
            zml.log.err("Tried to interpret metadata '{s}' as {}, but was of type {}", .{ key, tag, wrapped_value });
            @panic("invalid metadata type");
        }
        return @field(wrapped_value, @tagName(tag));
    }

    pub fn metadataSlice(self: BufferStore, key: []const u8, comptime tag: Value.Slice.ItemType) ?[]const Value.Slice.toZigType(tag) {
        const wrapped_value = self._metadata.get(key) orelse return null;

        if (wrapped_value != .array or wrapped_value.array.item_type != tag) {
            return null;
        }
        const T = Value.Slice.toZigType(tag);
        return @alignCast(std.mem.bytesAsSlice(T, wrapped_value.array.data));
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
        const data_ = try asynk.call(std.posix.mmap, .{
            null,
            data_len,
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.inner.file.fd,
            0,
        });

        try asynk.call(posix.madvise, .{ data_.ptr, @intCast(data_.len), @intCast(c.MADV_SEQUENTIAL) });

        return .{
            .file = file,
            .data = data_,
        };
    }

    pub fn mappedSlice(self: *MemoryMappedFile, start: usize, len: usize) []const u8 {
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
    unique_id: *u64,
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
        return if (buffer_store.get(prefix)) |buffer| {
            obj.* = zml.Tensor{
                ._shape = buffer.shape(),
                ._id = .{ .buffer_id = unique_id.* },
                ._donation = .input_buffer,
            };
            unique_id.* += 1;
            return true;
        } else {
            if (required) {
                std.log.err("Tensor not found: {s} ({d})", .{ prefix, buffer_store.buffers.count() });
            }
            return false;
        };
    }

    switch (type_info) {
        .Pointer => |ptr_info| {
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
                            std.log.err("Not able to load {s} as {s}", .{ prefix, @typeName(ptr_info.child) });
                            return false;
                        }
                    }
                } else if (required) {
                    log.warn("No layer found at {s}", .{prefix});
                }
                return true;
            } else if (ptr_info.size == .One) {
                //if (ptr_info.child != zml.Tensor and ptr_info.child != ?zml.Tensor) {
                //    // Note: should we recurse on all pointers ?
                //    log.warn("Not looking into: {any}", .{prefix});
                //    return false;
                //}
                //obj.* = try allocator.create(ptr_info.child);
                //return try _populateStruct(allocator, buffer_store, unique_id, prefix, obj.*, required);
            } else {
                std.log.err("{s} - {s}: {s} type not supported", .{ @src().fn_name, prefix, @typeName(T) });
                return false;
            }
        },
        .Struct => |struct_info| {
            // TODO(Corentin): See if we keep that
            //if (@hasDecl(T, "_zml_reader_skip_me_")) return false;

            var partial_struct = false;
            inline for (struct_info.fields) |field| {
                try prefix_builder.push(allocator, field.name);
                defer prefix_builder.pop();

                var has_default = false;
                if (field.default_value) |_| has_default = true;

                const field_found = try _populateStruct(allocator, prefix_builder, unique_id, buffer_store, &@field(obj, field.name), required and !has_default);
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
        //.Array => |array_info| {
        //    var new_prefix = prefix;
        //    if (prefix.items.len > 0)
        //        new_prefix.appendAssumeCapacity('.');
        //    const len = new_prefix.items.len;
        //    for (obj, 0..) |*value, i| {
        //        new_prefix.items.len += std.fmt.formatIntBuf(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
        //        const found = try _populateStruct(allocator, buffer_store, unique_id, new_prefix, value, required);
        //        if (!found) return false;
        //        new_prefix.shrinkRetainingCapacity(len);
        //    }
        //    const num_layers = buffer_store.numLayers(prefix.items);
        //    if (num_layers != array_info.len) {
        //        log.warn("Found {d} layers with prefix {s}, but only loaded {d}", .{ num_layers, prefix.items, array_info.len });
        //    }
        //    return true;
        //},
        .Optional => |opt_info| {
            obj.* = @as(opt_info.child, undefined);
            const found = try _populateStruct(allocator, prefix_builder, unique_id, buffer_store, &(obj.*.?), false);
            if (!found) obj.* = null;
            return true;
        },
        //.Union => |union_info| {
        //    // Note: the main issue here is that several fields could match but we only return the first one.
        //    inline for (union_info.fields) |field| {
        //        // interpret obj as a "field", and try to populate that.
        //        obj.* = @unionInit(T, field.name, undefined);
        //        const found = try _populateStruct(allocator, buffer_store, unique_id, prefix, &@field(obj.*, field.name), false);
        //        if (found) {
        //            std.log.info("Interpreted {s} as {s}", .{ prefix.items, @typeName(field.type) });
        //            return true;
        //        }
        //    }
        //    obj.* = undefined;
        //    if (required) {
        //        std.log.err("Not able to intepret {s} as any member of the union: {s}", .{ prefix.items, @typeName(T) });
        //    }
        //    return false;
        //},
        .Int => {
            obj.* = undefined;
            return true;
        },
        .Float => {
            obj.* = undefined;
            return true;
        },
        else => if (required) {
            std.log.err("{s}: {s} type not supported", .{ prefix, @typeName(T) });
            return error.UnsupportedMetadataType;
        } else return false,
    }
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

    // If the Model has a "init" function, call it with the given parameters.
    if (@hasDecl(Model, "init")) {
        @call(.auto, Model.init, .{&model} ++ init_args);
    } else {
        zml.meta.assertComptime(@TypeOf(init_args) == void or @TypeOf(init_args) == @TypeOf(.{}), "Model of type {} has no init function, so `loadBuffers` should be call with init_args set to {{}} (void)", .{Model});
    }

    return loadModelBuffers(Model, model, buffer_store, allocator, platform);
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

pub fn loadModelBuffersWithPrefix(
    comptime Model: type,
    model: Model,
    buffer_store: BufferStore,
    allocator: std.mem.Allocator,
    platform: zml.Platform,
    prefix: []const u8,
) !zml.Bufferized(Model) {
    // Allocate the bufferized version.
    // We set fields to undefined, cause visitStructAndLoadBuffer is responsible
    // to write them just afterward.
    var res: zml.Bufferized(Model) = undefined;
    try zml.meta.mapAlloc(struct {
        pub fn initBuffer(_: void, _: zml.Tensor) zml.Buffer {
            return undefined;
        }
    }.initBuffer, allocator, {}, model, &res);

    var prefix_builder: PrefixBuilder = .{};
    try prefix_builder.push(allocator, prefix);
    defer prefix_builder.deinit(allocator);

    try visitStructAndLoadBuffer(allocator, &prefix_builder, buffer_store, &res, platform);
    return res;
}

/// Creates a bufferized version of a Model from the given BufferStore and the given prefix.
/// For details about bufferization, see the documentation of Bufferized(T).
///
/// This will represent the weights of the model, loaded on a specific platform.
/// It can be used with a `module.Exe` (a compiled version of the same Model), to make a
/// `module.ExeWithWeights` ready to be called.
pub fn loadBuffersFromModelWithPrefix(comptime Model: type, model: Model, buffer_store: BufferStore, allocator: std.mem.Allocator, prefix: []const u8, platform: zml.Platform) !zml.Bufferized(Model) {

    // Allocate the bufferized version.
    // We set fields to undefined, cause visitStructAndLoadBuffer is responsible
    // to write them just afterward.
    var res: zml.Bufferized(Model) = undefined;
    try zml.meta.mapAlloc(struct {
        pub fn initBuffer(_: void, _: zml.Tensor) zml.Buffer {
            return undefined;
        }
    }.initBuffer, allocator, {}, model, &res);

    var prefix_builder: PrefixBuilder = .{};
    defer prefix_builder.deinit(allocator);
    try prefix_builder.push(allocator, prefix);

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
        return if (buffer_store.get(prefix)) |host_buffer| {
            obj.* = try zml.Buffer.from(platform, host_buffer);
        } else {
            return error.BufferNotFound;
        };
    }

    switch (type_info) {
        .Pointer => |ptr_info| {
            if (ptr_info.size == .Slice) {
                for (obj.*, 0..) |*value, i| {
                    var buffer: [100]u8 = undefined;
                    const new_prefix = std.fmt.bufPrint(&buffer, "{d}", .{i}) catch unreachable;

                    try prefix_builder.push(allocator, new_prefix);
                    defer prefix_builder.pop();

                    try visitStructAndLoadBuffer(allocator, prefix_builder, buffer_store, value, platform);
                }
            } else return error.TypeNotSupported;
        },
        .Struct => |struct_info| {
            // TODO(Corentin): See if we keep that
            //if (@hasDecl(T, "_zml_reader_skip_me_")) return false;

            inline for (struct_info.fields) |field| {
                try prefix_builder.push(allocator, field.name);
                defer prefix_builder.pop();

                try visitStructAndLoadBuffer(allocator, prefix_builder, buffer_store, &@field(obj, field.name), platform);
            }
        },
        //.Array => |array_info| {
        //    var new_prefix = prefix;
        //    if (prefix.items.len > 0)
        //        new_prefix.appendAssumeCapacity('.');
        //    const len = new_prefix.items.len;
        //    for (obj, 0..) |*value, i| {
        //        new_prefix.items.len += std.fmt.formatIntBuf(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
        //        const found = try _populateStruct(allocator, buffer_store, unique_id, new_prefix, value, required);
        //        if (!found) return false;
        //        new_prefix.shrinkRetainingCapacity(len);
        //    }
        //    const num_layers = buffer_store.numLayers(prefix.items);
        //    if (num_layers != array_info.len) {
        //        log.warn("Found {d} layers with prefix {s}, but only loaded {d}", .{ num_layers, prefix.items, array_info.len });
        //    }
        //    return true;
        //},
        .Optional => |opt_info| {
            var child = @as(opt_info.child, undefined);
            if (visitStructAndLoadBuffer(allocator, prefix_builder, buffer_store, &child, platform)) {
                obj.* = child;
            } else |err| switch (err) {
                error.BufferNotFound => {},
                else => return err,
            }
        },
        //.Union => |union_info| {
        //    // Note: the main issue here is that several fields could match but we only return the first one.
        //    inline for (union_info.fields) |field| {
        //        // interpret obj as a "field", and try to populate that.
        //        obj.* = @unionInit(T, field.name, undefined);
        //        const found = try _populateStruct(allocator, buffer_store, unique_id, prefix, &@field(obj.*, field.name), false);
        //        if (found) {
        //            std.log.info("Interpreted {s} as {s}", .{ prefix.items, @typeName(field.type) });
        //            return true;
        //        }
        //    }
        //    obj.* = undefined;
        //    if (required) {
        //        std.log.err("Not able to intepret {s} as any member of the union: {s}", .{ prefix.items, @typeName(T) });
        //    }
        //    return false;
        //},
        else => {},
    }
}

test {
    std.testing.refAllDecls(@This());
}
