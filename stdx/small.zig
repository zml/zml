const std = @import("std");
const builtin = @import("builtin");

const u8x16 = @Vector(16, u8);

pub fn Small(capacity_: comptime_int, T: type, sentinel_: T) type {
    if (@sizeOf(@Vector(capacity_, T)) > 4 * 128) {
        @compileError("Small only works up to 512 bits");
    }
    return struct {
        const Smol = @This();
        const Vec = @Vector(capacity_, T);
        const MaskType = @Type(.{ .int = .{ .signedness = .unsigned, .bits = capacity_ } });

        vec: Vec = @splat(sentinel),

        pub const capacity: comptime_int = capacity_;
        pub const sentinel: T = sentinel_;
        const sentinel_v: Vec = @splat(sentinel_);

        pub fn init(values: anytype) Smol {
            comptime {
                if (capacity < values.len) @compileError("Too many items for " ++ @typeName(Smol));
            }
            var res: Smol = undefined;
            inline for (0.., values) |i, v| {
                // stdx.debug.assert(v != sentinel, "{s}.init expects values distinct from sentinel {d}", .{@typeName(Smol), sentinel});
                std.debug.assert(v != sentinel);
                res.vec[i] = v;
            }
            for (values.len..capacity) |i| {
                res.vec[i] = sentinel;
            }
            return res;
        }

        pub fn items(smol_ptr: anytype) Slice(@TypeOf(smol_ptr)) {
            const data: [*]T = @constCast(@ptrCast(smol_ptr));
            return data[0..smol_ptr.len()];
        }

        fn Slice(smol_ptr_type: type) type {
            return switch (smol_ptr_type) {
                *Smol => []T,
                *const Smol => []const T,
                else => @compileError(items),
            };
        }

        pub fn len(x: Smol) usize {
            const sentinel_mask = x.vec == sentinel_v;
            const sentinel_i: MaskType = @bitCast(sentinel_mask);
            return @ctz(sentinel_i);
        }

        pub fn set(self: *Smol, i: usize, item: T) void {
            self.vec[i] = item;
        }

        pub fn append(x: Smol, value: T) Smol {
            std.debug.assert(value != sentinel);
            const n = x.len();
            std.debug.assert(n < capacity);
            var y = x;
            y.vec[n] = value;
            return y;
        }

        pub fn insert(x: Smol, pos: u8, value: T) Smol {
            std.debug.assert(value != sentinel);
            std.debug.assert(x.len() < capacity);
            if (builtin.cpu.arch == .aarch64 and std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon)) {
                const n_blocks = @divExact(@sizeOf(Smol), 16);

                var res: [n_blocks]u8x16 = undefined;
                const pos_v: u8x16 = @splat(@intCast(@as(usize, pos) * @sizeOf(T)));
                inline for (0..n_blocks) |block_id| {
                    const iota_ = std.simd.iota(u8, 16) + @as(u8x16, @splat(block_id * 16));
                    const iota_minus_one = iota_ -% @as(u8x16, @splat(@sizeOf(T)));
                    const index = @select(u8, iota_ < pos_v, iota_, iota_minus_one);
                    res[block_id] = x.lookup_aarch64(index);
                }
                var y: Smol = .{ .vec = @bitCast(res) };
                y.vec[pos] = value;
                if (comptime n_blocks == 4 and sentinel != 0) y.vec[capacity - 1] = sentinel;
                return y;
            }

            var y = x;
            y.vec[pos] = value;
            for (pos..capacity - 1) |i| {
                y.vec[i + 1] = x.vec[i];
            }
            return y;
        }

        pub fn remove(self: Smol, pos: u8) Smol {
            if (builtin.cpu.arch == .aarch64 and std.Target.aarch64.featureSetHas(builtin.cpu.features, .neon)) {
                const n_blocks = @divExact(@sizeOf(Smol), 16);
                var res: [n_blocks]u8x16 = undefined;
                const pos_v: u8x16 = @splat(@intCast(@as(usize, pos) * @sizeOf(T)));
                inline for (0..n_blocks) |block_id| {
                    const iota_ = std.simd.iota(u8, 16) + @as(u8x16, @splat(block_id * 16));
                    const iota_plus_one = iota_ + @as(u8x16, @splat(@sizeOf(T)));
                    const index = @select(u8, iota_ < pos_v, iota_, iota_plus_one);
                    res[block_id] = self.lookup_aarch64(index);
                }
                var y: Smol = .{ .vec = @bitCast(res) };
                if (comptime n_blocks == 4 and sentinel != 0) y.vec[capacity - 1] = sentinel;
                return y;
            }

            var y = self;
            for (pos..capacity - 1) |i| {
                y.vec[i] = self.vec[i + 1];
            }
            y.vec[capacity - 1] = sentinel;
            return y;
        }

        inline fn lookup_aarch64(self: Smol, idx: u8x16) u8x16 {
            const n_blocks = @divExact(@sizeOf(Smol), 16);
            const data: [n_blocks]u8x16 = @bitCast(self.vec);
            return switch (n_blocks) {
                1 => llvm.tbl1(data[0], idx),
                2 => llvm.tbl2(data[0], data[1], idx),
                3 => llvm.tbl3(data[0], data[1], data[2], idx),
                4 => llvm.tbl4(data[0], data[1], data[2], data[3], idx),
                else => @compileError(@typeName(Smol) ++ " is too big"),
            };
        }
    };
}

const llvm = struct {
    extern fn @"llvm.aarch64.neon.tbl1.v16i8"(u8x16, u8x16) u8x16;
    extern fn @"llvm.aarch64.neon.tbl2.v16i8"(u8x16, u8x16, u8x16) u8x16;
    extern fn @"llvm.aarch64.neon.tbl3.v16i8"(u8x16, u8x16, u8x16, u8x16) u8x16;
    extern fn @"llvm.aarch64.neon.tbl4.v16i8"(u8x16, u8x16, u8x16, u8x16, u8x16) u8x16;

    pub const tbl1 = @"llvm.aarch64.neon.tbl1.v16i8";
    pub const tbl2 = @"llvm.aarch64.neon.tbl2.v16i8";
    pub const tbl3 = @"llvm.aarch64.neon.tbl3.v16i8";
    pub const tbl4 = @"llvm.aarch64.neon.tbl4.v16i8";
};

const u32x8 = Small(8, u32, 0);

test "len" {
    try std.testing.expectEqual(0, u32x8.init(.{}).len());
    try std.testing.expectEqual(3, u32x8.init(.{ 1, 2, 3 }).len());
    try std.testing.expectEqual(5, u32x8.init(.{ 1, 2, 3, 4, 5 }).len());
    try std.testing.expectEqual(8, u32x8.init(.{ 1, 2, 3, 4, 5, 6, 7, 8 }).len());
}

test "items" {
    var x = u32x8.init(.{ 1, 2, 3 });
    try std.testing.expectEqual(.{ 1, 2, 3, 0, 0, 0, 0, 0 }, x.vec);

    x.items()[1] = 8;
    try std.testing.expectEqual(.{ 1, 8, 3, 0, 0, 0, 0, 0 }, x.vec);
}

test "append" {
    var x = u32x8.init(.{ 1, 2, 3 });
    x = x.append(4);
    try std.testing.expectEqual(.{ 1, 2, 3, 4, 0, 0, 0, 0 }, x.vec);
}

test "insert" {
    const x = u32x8.init(.{ 1, 2, 3, 4 });
    try std.testing.expectEqual(.{ 1, 2, 10, 3, 4, 0, 0, 0 }, x.insert(2, 10).vec);

    const y = u32x8.init(.{ 1, 2, 3, 4, 5, 6 });
    try std.testing.expectEqual(.{ 10, 1, 2, 3, 4, 5, 6, 0 }, y.insert(0, 10).vec);
}

test "remove" {
    const x = u32x8.init(.{ 1, 2, 3, 4 });
    try std.testing.expectEqual(.{ 1, 2, 4, 0, 0, 0, 0, 0 }, x.remove(2).vec);

    const y = u32x8.init(.{ 1, 2, 3, 4, 5, 6 });
    try std.testing.expectEqual(.{ 1, 2, 4, 5, 6, 0, 0, 0 }, y.remove(2).vec);
}

pub fn main() !void {
    const x = u32x8.init(.{ 1, 2, 3, 4 });
    try std.testing.expectEqual(.{ 1, 2, 4, 0, 0, 0, 0, 0 }, x.remove(2).vec);
}
