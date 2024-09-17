const asynk = @import("async");
const std = @import("std");
const zml = @import("../zml.zig");

const HostBuffer = @import("../hostbuffer.zig").HostBuffer;

const toVoidSlice = @import("utils.zig").toVoidSlice;
const eval = @import("torch/eval.zig");
const utils = @import("torch/utils.zig");
const value = @import("torch/value.zig");
const Decoder = @import("torch/parser.zig").Decoder;
const PersId = value.PersId;
const PickleMemo = eval.PickleMemo;
const PickleStack = eval.PickleStack;
const Sequence = value.Sequence;
const Value = value.Value;
const ValueType = value.ValueType;

const StringBuilder = std.ArrayListUnmanaged(u8);
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.zml_io);

const TorchType = enum {
    float64,
    double,
    float32,
    float,
    float16,
    half,
    bfloat16,
    int64,
    long,
    int32,
    int,
    int16,
    short,
    int8,
    char,
    uint8,
    byte,
};

fn dtypeFromStr(str: []const u8) !zml.DataType {
    const case = std.meta.stringToEnum(TorchType, str) orelse return error.UnknownTensorType;
    return switch (case) {
        .float64, .double => .f64,
        .float32, .float => .f32,
        .float16, .half => .f16,
        .bfloat16 => .bf16,
        .int64, .long => .i64,
        .int32, .int => .i32,
        .int16, .short => .i16,
        .int8, .char => .i8,
        .uint8, .byte => .u8,
    };
}

/// Opens and loads a BufferStore from the torch file at the given path.
pub fn open(allocator: Allocator, path: []const u8) !zml.aio.BufferStore {
    const file = asynk.File.open(path, .{}) catch |err| {
        log.err("Failed to open {s}: {}", .{ path, err });
        return err;
    };
    errdefer file.close() catch unreachable;

    var res: zml.aio.BufferStore = .{
        .arena = std.heap.ArenaAllocator.init(allocator),
    };

    const arena = res.arena.allocator();

    var tmp: PickleData = .{
        .data = try Decoder.init(arena, file),
        .memo = undefined,
        .stack = undefined,
    };
    tmp.stack, tmp.memo = try eval.evaluate(arena, tmp.data.ops, true);
    res.files = try arena.dupe(zml.aio.MemoryMappedFile, &.{tmp.data.buffer_file});
    try tmp.parseModel(arena, &res);
    return res;
}

pub const PickleData = struct {
    stack: PickleStack,
    memo: PickleMemo,
    data: Decoder,

    fn basicTypeCheck(v: Value, ns: []const u8, name: []const u8) bool {
        return switch (v) {
            .global => |object| switch (object.member) {
                .raw => |raw| {
                    if (std.mem.eql(u8, ns, raw.global[0]) and std.mem.eql(u8, name, raw.global[1]) and object.args[0] == .seq) {
                        return true;
                    } else return false;
                },
                else => false,
            },
            else => false,
        };
    }

    fn isTensor(v: Value) bool {
        if (basicTypeCheck(v, "torch._utils", "_rebuild_tensor_v2")) {
            const args = v.global.args[0].seq[1];
            if (args.len >= 5 and
                args[0] == .pers_id and
                args[1] == .int and
                args[2] == .seq and args[2].seq[0] == .tuple and
                args[3] == .seq and args[3].seq[0] == .tuple)
            {
                return true;
            } else @panic("Unexpected value in call to torch._utils._rebuild_tensor_v2");
        }
        return false;
    }

    fn dimsFromValues(values: []Value) [zml.Tensor.MAX_RANK]i64 {
        std.debug.assert(values.len <= zml.Tensor.MAX_RANK);
        var result: [zml.Tensor.MAX_RANK]i64 = undefined;
        for (values, result[0..values.len]) |val, *elem| {
            switch (val) {
                .int => |int| elem.* = int,
                else => @panic("Bad value for shape item"),
            }
        }
        return result;
    }

    pub fn parseModel(self: *PickleData, allocator: Allocator, store: *zml.aio.BufferStore) !void {
        for (self.stack.stack) |item| {
            var prefix_buf: [1024]u8 = undefined;
            try self.parseValue(allocator, store, StringBuilder.initBuffer(&prefix_buf), item);
        }
    }

    fn tensorOffset(self: *PickleData, seekable_stream: anytype, sfile: []const u8) !u64 {
        if (self.data.file_map.get(sfile)) |entry| {
            const local_header = blk: {
                try seekable_stream.seekTo(entry.file_offset);
                break :blk try seekable_stream.context.reader().readStructEndian(std.zip.LocalFileHeader, .little);
            };
            if (!std.mem.eql(u8, &local_header.signature, &std.zip.local_file_header_sig))
                return error.ZipBadFileOffset;
            if (local_header.version_needed_to_extract != entry.version_needed_to_extract)
                return error.ZipMismatchVersionNeeded;
            if (local_header.last_modification_time != entry.last_modification_time)
                return error.ZipMismatchModTime;
            if (local_header.last_modification_date != entry.last_modification_date)
                return error.ZipMismatchModDate;

            if (@as(u16, @bitCast(local_header.flags)) != @as(u16, @bitCast(entry.flags)))
                return error.ZipMismatchFlags;
            if (local_header.crc32 != 0 and local_header.crc32 != entry.crc32)
                return error.ZipMismatchCrc32;
            if (local_header.compressed_size != 0 and
                local_header.compressed_size != entry.compressed_size)
                return error.ZipMismatchCompLen;
            if (local_header.uncompressed_size != 0 and
                local_header.uncompressed_size != entry.uncompressed_size)
                return error.ZipMismatchUncompLen;
            if (local_header.filename_len != entry.filename_len)
                return error.ZipMismatchFilenameLen;

            return (try seekable_stream.context.getPos()) +
                @as(u64, local_header.filename_len) +
                @as(u64, local_header.extra_len);
        }

        std.log.err("Could not find file ending in `{s}` in archive", .{sfile});
        return error.TensorNotFound;
    }

    fn parseTorchGlobal(self: *PickleData, allocator: Allocator, store: *zml.aio.BufferStore, prefix: StringBuilder, v: Value) !bool {
        return switch (v) {
            .global => |object| {
                if (isTensor(v)) {
                    const args = object.args[0].seq[1];
                    const pidval: *PersId, var offs: u64, const raw_shape: Sequence, const raw_strides: Sequence = .{ args[0].pers_id, @intCast(args[1].int), args[2].seq, args[3].seq };
                    const rank = raw_shape[1].len;
                    const shape = dimsFromValues(raw_shape[1]);
                    var strides = dimsFromValues(raw_strides[1]);
                    const stype: []const u8, const sfile: []const u8, const sdev: []const u8 = switch (pidval.ref) {
                        .seq => |seq| blk: {
                            const sargs = seq[1];
                            if (seq[0] == .tuple and
                                sargs.len >= 5 and
                                sargs[0] == .string and std.mem.eql(u8, sargs[0].string, "storage") and
                                sargs[1] == .raw and sargs[1].raw == .global and
                                sargs[2] == .string and
                                sargs[3] == .string)
                            {
                                const op = sargs[1].raw.global;
                                const sfile = sargs[2].string;
                                const sdev = sargs[3].string;
                                const styp = op[1];
                                if (std.mem.eql(u8, "torch", op[0]) and std.mem.endsWith(u8, styp, "Storage")) {
                                    break :blk .{ std.ascii.lowerString(styp[0 .. styp.len - 7], styp[0 .. styp.len - 7]), sfile, sdev };
                                } else @panic("Unexpected storage type part of persistant ID");
                            } else @panic("Unexpected value for persistant ID");
                        },
                        else => @panic("Unexpected value for persistant ID"),
                    };
                    _ = sdev;
                    const data_type = try dtypeFromStr(stype);
                    for (strides[0..rank]) |*s| s.* *= data_type.sizeOf();

                    var sfile_buf = std.ArrayList(u8).init(allocator);
                    defer sfile_buf.deinit();
                    try sfile_buf.writer().print("{s}data/{s}", .{ self.data.zip_prefix, sfile });

                    // find offsets for tensor zip file
                    const absolute_offset = blk: {
                        if (self.data.tar_file) |t| {
                            break :blk try self.tensorOffset(t.seekableStream(), sfile_buf.items);
                        } else {
                            break :blk try self.tensorOffset(self.data.buffer_file.file.seekableStream(), sfile_buf.items);
                        }
                    };
                    offs = offs * data_type.sizeOf();
                    const key = try allocator.dupe(u8, prefix.items);
                    const entry = try store.buffers.getOrPut(allocator, key);
                    if (entry.found_existing) {
                        log.warn("Duplicate key: {s}", .{prefix.items});
                        allocator.free(key);
                    }
                    const out_shape = zml.Shape.init(shape[0..rank], data_type);
                    entry.value_ptr.* = HostBuffer.fromStridedSlice(
                        out_shape,
                        self.data.buffer_file.mappedSlice((if (self.data.tar_file) |t| t.start else 0) + absolute_offset + offs, out_shape.byteSize()),
                        strides[0..rank],
                    );
                    return true;
                } else if (basicTypeCheck(v, "torch", "Size")) {
                    const size = object.args[0].seq[1][0].seq[1];
                    const key = try allocator.dupe(u8, prefix.items);
                    const entry = try store._metadata.getOrPut(allocator, key);
                    if (entry.found_existing) {
                        log.warn("Duplicate key: {s}", .{prefix.items});
                        allocator.free(key);
                    }
                    const d = try allocator.alloc(i64, size.len);
                    for (d, 0..) |*di, i| di.* = size[i].int;
                    entry.value_ptr.* = .{ .array = .{ .item_type = .int64, .data = std.mem.sliceAsBytes(d) } };
                    return true;
                } else if (basicTypeCheck(v, "fractions", "Fraction")) {
                    const fraction_str = object.args[0].seq[1][0].string;
                    if (std.mem.indexOfScalar(u8, fraction_str, '/')) |split_idx| {
                        {
                            var new_prefix = prefix;
                            new_prefix.appendSliceAssumeCapacity(".numerator");
                            try store._metadata.put(allocator, try allocator.dupe(u8, new_prefix.items), .{ .int64 = try std.fmt.parseInt(i64, fraction_str[0..split_idx], 10) });
                        }
                        {
                            var new_prefix = prefix;
                            new_prefix.appendSliceAssumeCapacity(".denominator");
                            try store._metadata.put(allocator, try allocator.dupe(u8, new_prefix.items), .{ .int64 = try std.fmt.parseInt(i64, fraction_str[split_idx + 1 ..], 10) });
                        }
                        return true;
                    }
                }
                return false;
            },
            else => false,
        };
    }

    pub fn parseValue(self: *PickleData, allocator: Allocator, store: *zml.aio.BufferStore, prefix: StringBuilder, v: Value) !void {
        switch (v) {
            .app, .object, .global => |object| {
                if (!(try self.parseTorchGlobal(allocator, store, prefix, v))) {
                    try self.parseValue(allocator, store, prefix, object.member);
                    for (object.args) |item| {
                        // if possible, coerce to `kv_tuple` (only if key val doesn't match root of prefix)
                        if (item == .seq and item.seq[0] == .tuple and item.seq[1].len == 2 and item.seq[1][0] == .string) {
                            try self.parseValue(allocator, store, prefix, .{ .seq = .{ .kv_tuple, item.seq[1] } });
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
                                    const val = try allocator.alloc(u8, global[0].len + 1 + global[1].len);
                                    @memcpy(val[0..global[0].len], global[0]);
                                    val[global[0].len] = '.';
                                    @memcpy(val[global[0].len + 1 ..], global[1]);
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
            .seq => |*seq| switch (seq[0]) {
                .list, .tuple, .set, .frozen_set => {
                    const elemCheck = struct {
                        fn call(comptime T: ValueType) fn (v: Value) bool {
                            return struct {
                                fn call(val: Value) bool {
                                    return val == T;
                                }
                            }.call;
                        }
                    }.call;

                    if (seq[1].len > 0 and switch (seq[1][0]) {
                        inline .int, .bool, .float => |_, tag| utils.allTrue(seq[1][1..], elemCheck(tag)),
                        else => false,
                    }) {
                        const out: []u8 = switch (seq[1][0]) {
                            .int => blk: {
                                const d = try allocator.alloc(i64, seq[1].len);
                                for (seq[1], 0..) |item, i| {
                                    d[i] = item.int;
                                }
                                break :blk std.mem.sliceAsBytes(d);
                            },
                            .float => blk: {
                                const d = try allocator.alloc(f64, seq[1].len);
                                for (seq[1], 0..) |item, i| {
                                    d[i] = item.float;
                                }
                                break :blk std.mem.sliceAsBytes(d);
                            },
                            else => blk: {
                                const d = try allocator.alloc(bool, seq[1].len);
                                for (seq[1], 0..) |item, i| {
                                    d[i] = item.bool;
                                }
                                break :blk std.mem.sliceAsBytes(d);
                            },
                        };
                        const key = try allocator.dupe(u8, prefix.items);
                        const d = try store._metadata.getOrPut(allocator, key);
                        if (d.found_existing) {
                            log.warn("Duplicate key: {s}", .{prefix.items});
                            allocator.free(key);
                            allocator.free(out);
                        } else d.value_ptr.* = @unionInit(zml.aio.Value, "array", .{ .item_type = switch (seq[1][0]) {
                            .int => .int64,
                            .float => .float64,
                            .string => .string,
                            else => .boolval,
                        }, .data = out });
                    } else {
                        for (seq[1], 0..) |item, i| {
                            var new_prefix = prefix;
                            if (v.isPrimitive()) {
                                if (prefix.items.len > 0) {
                                    new_prefix.appendAssumeCapacity('.');
                                }
                                new_prefix.items.len += std.fmt.formatIntBuf(new_prefix.unusedCapacitySlice(), i, 10, .lower, .{});
                            }
                            try self.parseValue(allocator, store, new_prefix, item);
                        }
                    }
                },
                .dict => {
                    for (seq[1]) |item| {
                        try self.parseValue(allocator, store, prefix, item);
                    }
                },
                .kv_tuple => {
                    const key = seq[1][0];
                    const val = seq[1][1];
                    switch (key) {
                        .string => |s| {
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
                        .int => |int| {
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
            },
            .bytes => |val| {
                const key = try allocator.dupe(u8, prefix.items);
                const d = try store._metadata.getOrPut(allocator, key);
                if (d.found_existing) {
                    log.warn("Duplicate key: {s}", .{prefix.items});
                    allocator.free(key);
                } else d.value_ptr.* = .{ .array = .{ .item_type = .uint8, .data = @constCast(val) } };
            },
            inline .float, .int, .bool, .bigint, .string => |val, tag| {
                const key = try allocator.dupe(u8, prefix.items);
                const d = try store._metadata.getOrPut(allocator, key);
                if (d.found_existing) {
                    log.warn("Duplicate key: {s}", .{prefix.items});
                    allocator.free(key);
                } else d.value_ptr.* = @unionInit(zml.aio.Value, switch (tag) {
                    .int => "int64",
                    .float => "float64",
                    .bool => "boolval",
                    else => @tagName(tag),
                }, val);
            },
            else => {},
        }
    }
};

test {
    std.testing.refAllDecls(@This());
}
