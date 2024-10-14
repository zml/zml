const asynk = @import("async");
const std = @import("std");
const zml = @import("../zml.zig");

const HostBuffer = @import("../hostbuffer.zig").HostBuffer;

const eval = @import("torch/eval.zig");
const value = @import("torch/value.zig");
const parser = @import("torch/parser.zig");
const PersId = value.PersId;
const Sequence = value.Sequence;
const Value = value.Value;
const ValueType = value.ValueType;

const StringBuilder = std.ArrayListUnmanaged(u8);
const log = std.log.scoped(.zml_io);

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(eval);
    std.testing.refAllDecls(value);
    std.testing.refAllDecls(parser);
}

/// Opens and loads a BufferStore from the torch file at the given path.
pub fn open(allocator: std.mem.Allocator, path: []const u8) !zml.aio.BufferStore {
    const file = asynk.File.open(path, .{}) catch |err| {
        log.err("Failed to open {s}: {}", .{ path, err });
        return err;
    };
    errdefer file.close() catch unreachable;

    // Temporary memory needed to parse the pytorch file.
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const tmp_alloc = arena.allocator();

    const _parser = try parser.Parser.init(tmp_alloc, file);
    const stack, const memo = try eval.evaluate(tmp_alloc, _parser.ops, true);

    // But we create the HostBuffer objects inside the result BufferStore arena.
    var res: zml.aio.BufferStore = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
    };
    res.files = try res.arena.allocator().dupe(zml.aio.MemoryMappedFile, &.{_parser.buffer_file});
    var tmp: PickleData = .{ .data = _parser, .memo = memo, .stack = stack };
    try tmp.parseModel(res.arena.allocator(), &res);
    return res;
}

// TODO: rename me to PytorchFile
pub const PickleData = struct {
    stack: eval.PickleStack,
    memo: eval.PickleMemo,
    data: parser.Parser,

    fn basicTypeCheck(object: *const value.Object, module: []const u8, class: []const u8) bool {
        return switch (object.member) {
            .raw => |raw| return (object.args[0] == .seq and
                std.mem.eql(u8, module, raw.global.module) and
                std.mem.eql(u8, class, raw.global.class)),
            else => false,
        };
    }

    pub fn parseModel(self: *PickleData, allocator: std.mem.Allocator, store: *zml.aio.BufferStore) !void {
        for (self.stack.stack) |item| {
            var prefix_buf: [1024]u8 = undefined;
            try self.parseValue(allocator, store, StringBuilder.initBuffer(&prefix_buf), item);
        }
    }

    pub fn parseValue(self: *PickleData, allocator: std.mem.Allocator, store: *zml.aio.BufferStore, prefix: StringBuilder, v: Value) !void {
        switch (v) {
            .app, .object, .global => |object| {
                if (!(try self.parseTorchGlobal(allocator, store, prefix, v))) {
                    try self.parseValue(allocator, store, prefix, object.member);
                    for (object.args) |item| {
                        // if possible, coerce to `kv_tuple` (only if key val doesn't match root of prefix)
                        if (item == .seq and item.seq.type == .tuple and item.seq.values.len == 2 and item.seq.values[0] == .string) {
                            try self.parseValue(allocator, store, prefix, .{ .seq = .{ .type = .kv_tuple, .values = item.seq.values } });
                        } else try self.parseValue(allocator, store, prefix, item);
                    }
                }
            },
            .build => |build| {
                // `build` contains info about python struct being constructed
                switch (build.member) {
                    .object => |obj| switch (obj.member) {
                        .raw => |raw| switch (raw) {
                            .global => |global| {
                                // in this case, we can capture the name of the python type
                                // which can be used for codegen (e.g. `torch.nn.modules.conv.Conv2d`)
                                var new_prefix = prefix;
                                if (prefix.items.len > 0) {
                                    new_prefix.appendAssumeCapacity('.');
                                }
                                new_prefix.appendSliceAssumeCapacity("_gen_type_helper");
                                const key = try allocator.dupe(u8, new_prefix.items);
                                const d = try store._metadata.getOrPut(allocator, key);
                                if (d.found_existing) {
                                    log.err("Duplicate key: {s}", .{new_prefix.items});
                                    allocator.free(key);
                                } else {
                                    const val = try std.mem.join(allocator, ".", &.{ global.module, global.class });
                                    d.value_ptr.* = .{ .string = val };
                                }
                            },
                            else => try self.parseValue(allocator, store, prefix, build.member), // parse normally
                        },
                        else => try self.parseValue(allocator, store, prefix, build.member), // parse normally
                    },
                    else => try self.parseValue(allocator, store, prefix, build.member), // parse normally
                }
                try self.parseValue(allocator, store, prefix, build.args);
            },
            .pers_id => |pers_id| try self.parseValue(allocator, store, prefix, pers_id.ref),
            .seq => |seq| {
                switch (seq.type) {
                    .list, .tuple, .set, .frozen_set => {
                        if (seq.values.len == 0) return;
                        var valid_slice = true;
                        switch (seq.values[0]) {
                            inline .int64, .float64, .boolval => |val0, tag| {
                                const ItemType = switch (tag) {
                                    .int64 => i64,
                                    .float64 => f64,
                                    .boolval => bool,
                                    else => unreachable,
                                };
                                var values: std.ArrayListUnmanaged(ItemType) = .{};
                                try values.append(allocator, val0);
                                for (seq.values[1..], 1..) |val, i| {
                                    if (std.meta.activeTag(val) != tag) valid_slice = false;
                                    if (valid_slice) {
                                        try values.append(allocator, @field(val, @tagName(tag)));
                                    } else {
                                        var new_prefix = prefix;
                                        if (prefix.items.len > 0) {
                                            new_prefix.appendAssumeCapacity('.');
                                        }
                                        new_prefix.items.len += std.fmt.formatIntBuf(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
                                        try self.parseValue(allocator, store, new_prefix, val);
                                    }
                                }

                                if (valid_slice) {
                                    try store._metadata.put(
                                        allocator,
                                        try allocator.dupe(u8, prefix.items),
                                        try zml.aio.Metadata.copySlice(allocator, values.items),
                                    );
                                } else {
                                    for (values.items, 0..) |val, i| {
                                        var new_prefix = prefix;
                                        if (prefix.items.len > 0) {
                                            new_prefix.appendAssumeCapacity('.');
                                        }
                                        new_prefix.items.len += std.fmt.formatIntBuf(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
                                        const new_tag = switch (tag) {
                                            .int64 => "int",
                                            .float64 => "float",
                                            .boolval => "bool",
                                            else => unreachable, // we are already inside a switch
                                        };
                                        try store._metadata.put(allocator, try allocator.dupe(u8, new_prefix.items), @unionInit(zml.aio.Metadata, new_tag, val));
                                    }
                                }
                            },
                            else => {
                                for (seq.values, 0..) |item, i| {
                                    var new_prefix = prefix;
                                    if (v.isPrimitive()) {
                                        if (prefix.items.len > 0) {
                                            new_prefix.appendAssumeCapacity('.');
                                        }
                                        new_prefix.items.len += std.fmt.formatIntBuf(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
                                    }
                                    try self.parseValue(allocator, store, new_prefix, item);
                                }
                            },
                        }
                    },
                    .dict => for (seq.values) |item| {
                        try self.parseValue(allocator, store, prefix, item);
                    },
                    .kv_tuple => {
                        const key, const val = seq.values[0..2].*;
                        switch (key) {
                            .string => |s| {
                                // Handle Pytorch specific fields
                                if (std.mem.eql(u8, s, "_modules") or std.mem.eql(u8, s, "_parameters") or std.mem.eql(u8, s, "_buffers")) {
                                    try self.parseValue(allocator, store, prefix, val);
                                } else {
                                    var new_prefix = prefix;
                                    if (prefix.items.len > 0) {
                                        new_prefix.appendAssumeCapacity('.');
                                    }
                                    new_prefix.appendSliceAssumeCapacity(s);
                                    try self.parseValue(allocator, store, new_prefix, val);
                                }
                            },
                            .int64 => |int| {
                                var new_prefix = prefix;
                                if (prefix.items.len > 0) {
                                    new_prefix.appendAssumeCapacity('.');
                                }
                                new_prefix.items.len += std.fmt.formatIntBuf(new_prefix.unusedCapacitySlice(), int, 10, .lower, .{});
                                try self.parseValue(allocator, store, new_prefix, val);
                            },
                            inline else => |_, tag| std.debug.panic("Unexpected key type: {s}", .{@tagName(tag)}),
                        }
                    },
                }
            },
            .bytes => |val| {
                const key = try allocator.dupe(u8, prefix.items);
                const d = try store._metadata.getOrPut(allocator, key);
                if (d.found_existing) {
                    log.warn("Duplicate key: {s}", .{prefix.items});
                    allocator.free(key);
                } else d.value_ptr.* = .{ .string = val };
            },
            inline .float64, .int64, .boolval, .bigint, .string => |val| {
                const key = try allocator.dupe(u8, prefix.items);
                const d = try store._metadata.getOrPut(allocator, key);
                if (d.found_existing) {
                    log.warn("Duplicate key: {s}", .{prefix.items});
                    allocator.free(key);
                } else {
                    d.value_ptr.* = zml.aio.Metadata.wrap(val);
                }
            },
            else => {},
        }
    }

    fn parseTorchGlobal(self: *PickleData, allocator: std.mem.Allocator, store: *zml.aio.BufferStore, prefix: StringBuilder, v: Value) !bool {
        return switch (v) {
            .global => |object| {
                if (try self.parseTensor(allocator, object)) |host_buffer| {
                    const key = try allocator.dupe(u8, prefix.items);
                    const entry = try store.buffers.getOrPut(allocator, key);
                    if (entry.found_existing) {
                        log.warn("Duplicate key: {s}", .{prefix.items});
                        allocator.free(key);
                    }
                    entry.value_ptr.* = host_buffer;
                    return true;
                } else if (basicTypeCheck(object, "torch", "Size")) {
                    const size = object.args[0].seq.values[0].seq.values;
                    const key = try allocator.dupe(u8, prefix.items);
                    const entry = try store._metadata.getOrPut(allocator, key);
                    if (entry.found_existing) {
                        log.warn("Duplicate key: {s}", .{prefix.items});
                        allocator.free(key);
                    }
                    const d = try allocator.alloc(i64, size.len);
                    for (d, 0..) |*di, i| di.* = size[i].int64;
                    entry.value_ptr.* = .{ .array_int = d };
                    return true;
                } else if (basicTypeCheck(object, "fractions", "Fraction")) {
                    const fraction_str = object.args[0].seq.values[0].string;
                    if (std.mem.indexOfScalar(u8, fraction_str, '/')) |split_idx| {
                        {
                            var new_prefix = prefix;
                            new_prefix.appendSliceAssumeCapacity(".numerator");
                            try store._metadata.put(allocator, try allocator.dupe(u8, new_prefix.items), .{ .int = try std.fmt.parseInt(i64, fraction_str[0..split_idx], 10) });
                        }
                        {
                            var new_prefix = prefix;
                            new_prefix.appendSliceAssumeCapacity(".denominator");
                            try store._metadata.put(allocator, try allocator.dupe(u8, new_prefix.items), .{ .int = try std.fmt.parseInt(i64, fraction_str[split_idx + 1 ..], 10) });
                        }
                        return true;
                    }
                }
                return false;
            },
            else => false,
        };
    }

    fn parseTensor(self: *PickleData, tmp_allocator: std.mem.Allocator, object: *value.Object) !?zml.HostBuffer {
        if (!basicTypeCheck(object, "torch._utils", "_rebuild_tensor_v2")) {
            return null;
        }

        const args = object.args[0].seq.values;
        if (args.len < 4 or
            args[0] != .pers_id or
            args[1] != .int64 or
            args[2] != .seq or args[2].seq.type != .tuple or
            args[3] != .seq or args[3].seq.type != .tuple)
        {
            log.err("Unexpected value in call to torch._utils._rebuild_tensor_v2", .{});
            return error.InvalidInput;
        }

        const pid: *PersId = args[0].pers_id;
        var offset: u64 = @intCast(args[1].int64);
        const raw_dims: Sequence = args[2].seq;
        const raw_strides: Sequence = args[3].seq;
        const dims = try parseDims(raw_dims.values);
        var strides = try parseDims(raw_strides.values);

        const dtype, const storage_file = try parseStorage(pid.ref);
        // Pytorch store "item" strides, while ZML uses byte strides.
        for (strides.slice()) |*s| s.* *= dtype.sizeOf();
        // Same thing for the offset.
        offset = offset * dtype.sizeOf();

        const filename = try std.mem.join(tmp_allocator, "", &.{ self.data.zip_prefix, "data/", storage_file });
        defer tmp_allocator.free(filename);

        // The offset in the pickle is the offset inside the storage_file.
        // But .pt are made of several files, so we need to append the file offset.
        const storage = try self.getStorage(filename);
        return HostBuffer.fromStridedSlice(
            zml.Shape.init(dims.constSlice(), dtype),
            storage[offset..],
            strides.constSlice(),
        );
    }

    fn parseStorage(val: value.Value) !struct { zml.DataType, []const u8 } {
        if (val != .seq) return error.InvalidInput;
        const sargs = val.seq.values;
        if (val.seq.type == .tuple and
            sargs.len >= 5 and
            sargs[0] == .string and std.mem.eql(u8, sargs[0].string, "storage") and
            sargs[1] == .raw and sargs[1].raw == .global and
            sargs[2] == .string and
            sargs[3] == .string)
        {
            const op = sargs[1].raw.global;
            const storage_file = sargs[2].string;
            // const sdev = sargs[3].string;
            if (!std.mem.eql(u8, "torch", op.module) or
                !std.mem.endsWith(u8, op.class, "Storage"))
                return error.InvalidInput;

            return .{
                try storageToDtype(op.class),
                storage_file,
            };
        } else {
            return error.InvalidInput;
        }
    }

    /// Given the name of one of the files in the .pt tarball,
    /// return the slice of the memory-mapped .pt corresponding to it.
    fn getStorage(self: *PickleData, filename: []const u8) ![]const u8 {
        const maybe_entry = self.data.file_map.get(filename);
        if (maybe_entry == null) {
            std.log.err("Could not find file ending in `{s}` in archive", .{filename});
            return error.TensorNotFound;
        }
        const entry = maybe_entry.?;
        const base_offset: u64 = if (self.data.tar_file) |t| t.start else 0;
        const file_offset: u64 = base_offset + entry.file_offset;
        const file = self.data.buffer_file.file;
        try file.seekTo(entry.file_offset);
        const local_header = try file.reader().readStructEndian(std.zip.LocalFileHeader, .little);

        if (!std.mem.eql(u8, &local_header.signature, &std.zip.local_file_header_sig))
            return error.ZipBadFileOffset;
        if (local_header.compressed_size != 0 and
            local_header.compressed_size != entry.compressed_size)
            return error.ZipMismatchCompLen;
        if (local_header.uncompressed_size != 0 and
            local_header.uncompressed_size != entry.uncompressed_size)
            return error.ZipMismatchUncompLen;
        if (local_header.filename_len != entry.filename_len)
            return error.ZipMismatchFilenameLen;

        const start = file_offset +
            @sizeOf(std.zip.LocalFileHeader) +
            @as(u64, local_header.filename_len) +
            @as(u64, local_header.extra_len);
        return self.data.buffer_file.mappedSlice(start, entry.uncompressed_size);
    }

    fn parseDims(values: []Value) error{InvalidInput}!zml.Shape.DimsArray {
        zml.meta.assert(values.len <= zml.Tensor.MAX_RANK, "Found Pytorch tensor with unsupported rank {}", .{values.len});
        var result: zml.Shape.DimsArray = .{};
        for (values) |val| {
            switch (val) {
                .int64 => |d| result.appendAssumeCapacity(d),
                else => return error.InvalidInput,
            }
        }
        return result;
    }
};

/// Convert from a torch.<type>Storage to a `zml.DataType`.
/// TODO: make this future proof, storage type are going to get replaced with torch.UntypedStorage
/// See https://pytorch.org/docs/stable/storage.html
fn storageToDtype(storage_type: []const u8) !zml.DataType {
    const torch_type = storage_type[0 .. storage_type.len - "Storage".len];
    const map = std.StaticStringMap(zml.DataType).initComptime(.{
        .{ "Double", .f64 },
        .{ "Float", .f32 },
        .{ "Half", .f16 },
        .{ "Long", .i64 },
        .{ "Int", .i32 },
        .{ "Short", .i16 },
        .{ "Char", .i8 },
        .{ "Byte", .u8 },
        .{ "Bool", .bool },
        .{ "BFloat16", .bf16 },
        .{ "ComplexDouble", .c128 },
        .{ "ComplexFloat", .c64 },
        // QUInt8Storage
        // QInt8Storage
        // QInt32Storage
        // QUInt4x2Storage
        // QUInt2x4Storage
    });

    return map.get(torch_type) orelse {
        log.err("Unsupported torch storage type: {s}", .{storage_type});
        return error.UnsupportedDataType;
    };
}
