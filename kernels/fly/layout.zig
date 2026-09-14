const std = @import("std");

const fly = @import("mlir/dialects/fly");
const mlir = @import("mlir");
const stdx = @import("stdx");

const attributes = fly.attributes;

/// Enough for any tuple or tile a kernel writes by hand.
const max_modes = 32;

/// A dynamic integer leaf: `?`, `?{i64}`, `?{div=8}`, `?{i64 div=8}`.
pub const Dyn = struct {
    width: u8 = 32,
    div: u32 = 1,
};

/// Static, dynamic, or the `*` wildcard.
pub const Leaf = union(enum) {
    s: i64,
    d: Dyn,
    none,

    pub fn print(self: Leaf, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self) {
            .s => |v| try w.print("{d}", .{v}),
            .none => try w.writeAll("*"),
            .d => |dy| {
                try w.writeAll("?");
                if (dy.width != 32 or dy.div != 1) {
                    try w.writeAll("{");
                    if (dy.width != 32) try w.print("i{d}", .{dy.width});
                    if (dy.div != 1) {
                        if (dy.width != 32) try w.writeAll(" ");
                        try w.print("div={d}", .{dy.div});
                    }
                    try w.writeAll("}");
                }
            },
        }
    }
};

fn leafAttr(leaf: Leaf, ctx: *mlir.Context) mlir.Error!*const attributes.IntAttr {
    return switch (leaf) {
        .s => |v| .getStatic(ctx, @intCast(v)),
        .d => |dy| .getDynamic(ctx, dy.width, @intCast(dy.div)),
        .none => .getNone(ctx),
    };
}

/// A scaled basis element: `vE0`, `2E1E2`.
pub const Basis = struct {
    value: Leaf,
    modes: []const i32,
};

pub const IntTuple = union(enum) {
    leaf: Leaf,
    basis: Basis,
    tup: []const IntTuple,

    pub const dyn: IntTuple = .{ .leaf = .{ .d = .{} } };
    pub const star: IntTuple = .{ .leaf = .none };

    pub fn static(v: i64) IntTuple {
        return .{ .leaf = .{ .s = v } };
    }

    pub fn dynamic(width: u8, div: u32) IntTuple {
        return .{ .leaf = .{ .d = .{ .width = width, .div = div } } };
    }

    /// 1 for a leaf.
    pub fn rank(self: IntTuple) usize {
        return switch (self) {
            .tup => |t| t.len,
            else => 1,
        };
    }

    pub fn isLeaf(self: IntTuple) bool {
        return self != .tup;
    }

    pub fn at(self: IntTuple, i: usize) IntTuple {
        return switch (self) {
            .tup => |t| t[i],
            else => if (i == 0) self else @panic("fly.IntTuple.at: index out of range on a leaf"),
        };
    }

    /// Null if any leaf is dynamic or `*`.
    pub fn product(self: IntTuple) ?i64 {
        switch (self) {
            .leaf => |l| return switch (l) {
                .s => |v| v,
                else => null,
            },
            .basis => return null,
            .tup => |t| {
                var p: i64 = 1;
                for (t) |e| p *= e.product() orelse return null;
                return p;
            },
        }
    }

    pub fn isStatic(self: IntTuple) bool {
        return switch (self) {
            .leaf => |l| l == .s,
            .basis => |b| b.value == .s,
            .tup => |t| for (t) |e| {
                if (!e.isStatic()) break false;
            } else true,
        };
    }

    /// At any depth.
    pub fn leafCount(self: IntTuple) usize {
        return switch (self) {
            .tup => |t| blk: {
                var n: usize = 0;
                for (t) |e| n += e.leafCount();
                break :blk n;
            },
            else => 1,
        };
    }

    pub fn print(self: IntTuple, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self) {
            .leaf => |l| try l.print(w),
            .basis => |b| {
                try b.value.print(w);
                for (b.modes) |m| try w.print("E{d}", .{m});
            },
            .tup => |t| {
                try w.writeAll("(");
                for (t, 0..) |e, i| {
                    if (i > 0) try w.writeAll(",");
                    try e.print(w);
                }
                try w.writeAll(")");
            },
        }
    }

    pub fn format(self: IntTuple, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try self.print(w);
    }

    pub fn toAttr(self: IntTuple, ctx: *mlir.Context) mlir.Error!*const attributes.IntTupleAttr {
        return switch (self) {
            .leaf => |l| .get(ctx, .{ .value = (try leafAttr(l, ctx)).attribute() }),
            .basis => |b| .getBasis(ctx, try leafAttr(b.value, ctx), b.modes),
            .tup => |elems| blk: {
                var buf: stdx.BoundedArray(*const attributes.IntTupleAttr, max_modes) = .empty;
                for (elems) |e| buf.appendAssumeCapacity(try e.toAttr(ctx));
                break :blk .getTuple(ctx, buf.constSlice());
            },
        };
    }

    pub fn eql(a: IntTuple, b: IntTuple) bool {
        if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
        return switch (a) {
            .leaf => |l| std.meta.eql(l, b.leaf),
            .basis => |x| std.meta.eql(x.value, b.basis.value) and std.mem.eql(i32, x.modes, b.basis.modes),
            .tup => |t| t.len == b.tup.len and for (t, b.tup) |x, y| {
                if (!x.eql(y)) break false;
            } else true,
        };
    }
};

/// Integers become static leaves, tuples nest, `null` is `*`, an `IntTuple`
/// passes through: `it(.{ .{ 2, 4 }, 8 })` is `((2,4),8)`.
pub fn it(comptime v: anytype) IntTuple {
    return comptime itImpl(v);
}

fn itImpl(comptime v: anytype) IntTuple {
    const T = @TypeOf(v);
    if (T == IntTuple) return v;
    if (T == Leaf) return .{ .leaf = v };
    if (T == @TypeOf(null)) return IntTuple.star;
    switch (@typeInfo(T)) {
        .comptime_int, .int => return IntTuple.static(v),
        .@"struct" => |s| {
            if (!s.is_tuple) @compileError("fly.it: expected an integer, null, IntTuple or tuple, got " ++ @typeName(T));
            var elems: [s.fields.len]IntTuple = undefined;
            for (s.fields, 0..) |f, i| elems[i] = itImpl(@field(v, f.name));
            const frozen = elems;
            return .{ .tup = &frozen };
        },
        else => @compileError("fly.it: expected an integer, null, IntTuple or tuple, got " ++ @typeName(T)),
    }
}

/// `!fly.layout<shape:stride>`.
pub const Layout = struct {
    shape: IntTuple,
    stride: IntTuple,

    pub fn rank(self: Layout) usize {
        return self.shape.rank();
    }

    /// Null if dynamic.
    pub fn size(self: Layout) ?i64 {
        return self.shape.product();
    }

    pub fn print(self: Layout, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try self.shape.print(w);
        try w.writeAll(":");
        try self.stride.print(w);
    }

    pub fn format(self: Layout, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try self.print(w);
    }

    pub fn toAttr(self: Layout, ctx: *mlir.Context) mlir.Error!*const attributes.LayoutAttr {
        return .get(ctx, .{ .shape = try self.shape.toAttr(ctx), .stride = try self.stride.toAttr(ctx) });
    }

    pub fn eql(a: Layout, b: Layout) bool {
        return a.shape.eql(b.shape) and a.stride.eql(b.stride);
    }
};

/// Two congruent tuples: `L(.{ 4, 8 }, .{ 1, 4 })` is `(4,8):(1,4)`.
pub fn L(comptime shape: anytype, comptime stride: anytype) Layout {
    return comptime blk: {
        const s = it(shape);
        const d = it(stride);
        checkCongruent(s, d);
        break :blk .{ .shape = s, .stride = d };
    };
}

fn checkCongruent(comptime a: IntTuple, comptime b: IntTuple) void {
    if (a.isLeaf() != b.isLeaf() or a.rank() != b.rank()) {
        @compileError("fly.L: shape and stride are not congruent");
    }
    if (a == .tup) {
        for (a.tup, b.tup) |x, y| checkCongruent(x, y);
    }
}

/// Last mode contiguous: `rowMajor(.{ 4, 8 })` is `(4,8):(8,1)`.
pub fn rowMajor(comptime shape: anytype) Layout {
    return comptime blk: {
        const s = it(shape);
        const n = s.leafCount();
        var order: [n]i32 = undefined;
        for (0..n) |i| order[i] = @intCast(n - 1 - i);
        break :blk orderedFlat(s, &order);
    };
}

/// First mode contiguous.
pub fn colMajor(comptime shape: anytype) Layout {
    return comptime blk: {
        const s = it(shape);
        const n = s.leafCount();
        var order: [n]i32 = undefined;
        for (0..n) |i| order[i] = @intCast(i);
        break :blk orderedFlat(s, &order);
    };
}

/// `make_ordered_layout` folded at comptime: the smallest order value gets
/// stride 1, the next the product of the previous extents, and so on.
/// `ordered(.{ 8, 16 }, .{ 1, 0 })` is `(8,16):(16,1)`.
pub fn ordered(comptime shape: anytype, comptime order: anytype) Layout {
    return comptime blk: {
        const s = it(shape);
        const o = it(order);
        checkCongruent(s, o);
        const n = s.leafCount();
        var flat_order: [n]i32 = undefined;
        var idx: usize = 0;
        flattenOrder(o, &flat_order, &idx);
        break :blk orderedFlat(s, &flat_order);
    };
}

fn flattenOrder(comptime o: IntTuple, out: []i32, idx: *usize) void {
    switch (o) {
        .leaf => |l| {
            out[idx.*] = @intCast(l.s);
            idx.* += 1;
        },
        .tup => |t| for (t) |e| flattenOrder(e, out, idx),
        .basis => @compileError("fly.ordered: basis elements are not allowed in an order"),
    }
}

fn orderedFlat(comptime shape: IntTuple, comptime order: []const i32) Layout {
    comptime {
        const n = shape.leafCount();
        var extents: [n]i64 = undefined;
        var idx: usize = 0;
        flattenStatic(shape, &extents, &idx);
        var strides: [n]i64 = undefined;
        var assigned: [n]bool = @splat(false);
        var acc: i64 = 1;
        for (0..n) |_| {
            var best: ?usize = null;
            for (0..n) |i| {
                if (assigned[i]) continue;
                if (best == null or order[i] < order[best.?]) best = i;
            }
            strides[best.?] = acc;
            acc *= extents[best.?];
            assigned[best.?] = true;
        }
        var pos: usize = 0;
        return .{ .shape = shape, .stride = unflatten(shape, &strides, &pos) };
    }
}

fn flattenStatic(comptime t: IntTuple, out: []i64, idx: *usize) void {
    switch (t) {
        .leaf => |l| switch (l) {
            .s => |v| {
                out[idx.*] = v;
                idx.* += 1;
            },
            else => @compileError("fly: a folded layout needs a fully static shape"),
        },
        .basis => @compileError("fly: a folded layout needs a plain integer shape"),
        .tup => |elems| for (elems) |e| flattenStatic(e, out, idx),
    }
}

fn unflatten(comptime like: IntTuple, values: []const i64, pos: *usize) IntTuple {
    switch (like) {
        .tup => |elems| {
            var out: [elems.len]IntTuple = undefined;
            for (elems, 0..) |e, i| out[i] = unflatten(e, values, pos);
            const frozen = out;
            return .{ .tup = &frozen };
        },
        else => {
            const v = values[pos.*];
            pos.* += 1;
            return IntTuple.static(v);
        },
    }
}

/// A leaf (`8` or `*`), a layout, or a nested tile.
pub const Tile = union(enum) {
    leaf: Leaf,
    layout: Layout,
    modes: []const Tile,

    pub fn print(self: Tile, w: *std.Io.Writer) std.Io.Writer.Error!void {
        switch (self) {
            .leaf => |l| try l.print(w),
            .layout => |l| try l.print(w),
            .modes => |m| {
                try w.writeAll("[");
                for (m, 0..) |e, i| {
                    if (i > 0) try w.writeAll("|");
                    try e.print(w);
                }
                try w.writeAll("]");
            },
        }
    }

    pub fn format(self: Tile, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try self.print(w);
    }

    /// One mode: an int, a layout, or a nested tile.
    fn modeAttr(self: Tile, ctx: *mlir.Context) mlir.Error!*const mlir.Attribute {
        return switch (self) {
            .leaf => |l| (try leafAttr(l, ctx)).attribute(),
            .layout => |l| (try l.toAttr(ctx)).attribute(),
            .modes => (try self.toAttr(ctx)).attribute(),
        };
    }

    pub fn toAttr(self: Tile, ctx: *mlir.Context) mlir.Error!*const attributes.TileAttr {
        const modes = switch (self) {
            .modes => |m| m,
            else => return .get(ctx, .{ .value = try self.modeAttr(ctx) }),
        };
        var buf: stdx.BoundedArray(*const mlir.Attribute, max_modes) = .empty;
        for (modes) |m| buf.appendAssumeCapacity(try m.modeAttr(ctx));
        return .getModes(ctx, buf.constSlice());
    }
};

/// `tile(.{ 128, 64 })` is `[128|64]`; `tile(.{ null, 8 })` is `[*|8]`.
pub fn tile(comptime v: anytype) Tile {
    return comptime tileImpl(v);
}

fn tileImpl(comptime v: anytype) Tile {
    const T = @TypeOf(v);
    if (T == Tile) return v;
    if (T == Layout) return .{ .layout = v };
    if (T == Leaf) return .{ .leaf = v };
    if (T == @TypeOf(null)) return .{ .leaf = .none };
    switch (@typeInfo(T)) {
        .comptime_int, .int => return .{ .leaf = .{ .s = v } },
        .@"struct" => |s| {
            if (!s.is_tuple) @compileError("fly.tile: expected an integer, null, Layout, Tile or tuple, got " ++ @typeName(T));
            var elems: [s.fields.len]Tile = undefined;
            for (s.fields, 0..) |f, i| elems[i] = tileImpl(@field(v, f.name));
            const frozen = elems;
            return .{ .modes = &frozen };
        },
        else => @compileError("fly.tile: expected an integer, null, Layout, Tile or tuple, got " ++ @typeName(T)),
    }
}

/// `S<mask,base,shift>`.
pub const Swizzle = struct {
    mask: i32,
    base: i32,
    shift: i32,

    pub const trivial: Swizzle = .{ .mask = 0, .base = 0, .shift = 0 };

    pub fn isTrivial(self: Swizzle) bool {
        return self.mask == 0 and self.base == 0 and self.shift == 0;
    }

    pub fn print(self: Swizzle, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("S<{d},{d},{d}>", .{ self.mask, self.base, self.shift });
    }

    pub fn format(self: Swizzle, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try self.print(w);
    }

    pub fn toAttr(self: Swizzle, ctx: *mlir.Context) mlir.Error!*const attributes.SwizzleAttr {
        return .get(ctx, .{ .mask = self.mask, .base = self.base, .shift = self.shift });
    }
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

fn expectPrints(expected: []const u8, v: anytype) !void {
    var buf: [1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{v});
    try std.testing.expectEqualStrings(expected, w.buffered());
}

test "int tuple literals print in fly syntax" {
    try expectPrints("42", it(42));
    try expectPrints("((2,4),8)", it(.{ .{ 2, 4 }, 8 }));
    try expectPrints("(?,8)", it(.{ IntTuple.dyn, 8 }));
    try expectPrints("?{i64 div=8}", IntTuple.dynamic(64, 8));
    try expectPrints("?{div=4}", IntTuple.dynamic(32, 4));
    try expectPrints("?{i64}", IntTuple.dynamic(64, 1));
    try expectPrints("(*,8)", it(.{ null, 8 }));
    try expectPrints("(1,(2,3))", it(.{ 1, .{ 2, 3 } }));
}

test "layout literals" {
    try expectPrints("(4,8):(1,4)", L(.{ 4, 8 }, .{ 1, 4 }));
    try expectPrints("((2,4),8):((1,2),8)", L(.{ .{ 2, 4 }, 8 }, .{ .{ 1, 2 }, 8 }));
    try expectPrints("(4,8):(8,1)", rowMajor(.{ 4, 8 }));
    try expectPrints("(4,8):(1,4)", colMajor(.{ 4, 8 }));
    try expectPrints("((2,4),8):((32,8),1)", rowMajor(.{ .{ 2, 4 }, 8 }));
    try expectPrints("(8,16):(16,1)", ordered(.{ 8, 16 }, .{ 1, 0 }));
    try expectPrints("(1,4):(1,1)", ordered(.{ 1, 4 }, .{ 0, 1 }));
    try expectPrints("(4,2,8):(2,1,8)", ordered(.{ 4, 2, 8 }, .{ 1, 0, 2 }));
}

test "tiles and swizzles" {
    try expectPrints("[128|64]", tile(.{ 128, 64 }));
    try expectPrints("[*|8]", tile(.{ null, 8 }));
    try expectPrints("[(4,8):(1,4)|16]", tile(.{ L(.{ 4, 8 }, .{ 1, 4 }), 16 }));
    try expectPrints("[[2|3]|4]", tile(.{ .{ 2, 3 }, 4 }));
    try expectPrints("S<3,3,3>", Swizzle{ .mask = 3, .base = 3, .shift = 3 });
}

test "static queries" {
    try std.testing.expectEqual(@as(?i64, 64), it(.{ .{ 2, 4 }, 8 }).product());
    try std.testing.expectEqual(@as(?i64, null), it(.{ IntTuple.dyn, 8 }).product());
    try std.testing.expectEqual(@as(usize, 2), it(.{ .{ 2, 4 }, 8 }).rank());
    try std.testing.expectEqual(@as(usize, 3), it(.{ .{ 2, 4 }, 8 }).leafCount());
    try std.testing.expect(it(.{ 2, 4 }).eql(it(.{ 2, 4 })));
    try std.testing.expect(!it(.{ 2, 4 }).eql(it(.{ 4, 2 })));
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    fly.insertDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

/// The type built from the host model must be the type the printed literal
/// parses to: the two paths may never drift.
fn expectBuilt(ctx: *mlir.Context, comptime T: type, comptime fmt: []const u8, value: anytype) !void {
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print(fmt, .{value});
    const parsed = try mlir.Type.parse(ctx, w.buffered());
    const built = try T.get(ctx, .{ .attr = try value.toAttr(ctx) });
    if (!parsed.eql(built.type_())) {
        std.debug.print("parsed `{f}` but built `{f}`\n", .{ parsed, built });
        return error.TestUnexpectedResult;
    }
}

test "the host model builds the types its literals print" {
    const ctx = try testContext();
    defer ctx.deinit();
    const IntTupleType = fly.types.IntTupleType;
    const LayoutType = fly.types.LayoutType;
    const TileType = fly.types.TileType;

    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", it(42));
    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", it(.{ .{ 2, 4 }, 8 }));
    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", it(.{ IntTuple.dyn, 8 }));
    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", IntTuple.dynamic(64, 8));
    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", it(.{ null, 8 }));
    try expectBuilt(ctx, IntTupleType, "!fly.int_tuple<{f}>", IntTuple{ .basis = .{ .value = .{ .s = 2 }, .modes = &.{ 1, 2 } } });

    try expectBuilt(ctx, LayoutType, "!fly.layout<{f}>", L(.{ 4, 8 }, .{ 1, 4 }));
    try expectBuilt(ctx, LayoutType, "!fly.layout<{f}>", rowMajor(.{ .{ 2, 4 }, 8 }));

    try expectBuilt(ctx, TileType, "!fly.tile<{f}>", tile(.{ 128, 64 }));
    try expectBuilt(ctx, TileType, "!fly.tile<{f}>", tile(.{ null, 8 }));
    try expectBuilt(ctx, TileType, "!fly.tile<{f}>", tile(.{ L(.{ 4, 8 }, .{ 1, 4 }), 16 }));
    try expectBuilt(ctx, TileType, "!fly.tile<{f}>", tile(.{ .{ 2, 3 }, 4 }));
}
