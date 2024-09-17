const builtin = @import("builtin");
const std = @import("std");
const testing = std.testing;

const meta = @import("meta.zig");
const DataType = @import("dtype.zig").DataType;

const EnumLiteral = @TypeOf(.enum_literal);
const log = std.log.scoped(.shape);

test {
    std.testing.refAllDecls(@This());
}

/// Represent the shape of a tensor.
pub const Shape = struct {
    pub const MAX_RANK: u8 = 8;

    pub const Tag = [*:0]const u8;
    pub const TagUnknown = "_".ptr;
    const TagLast = "last".ptr;

    // Note: we can't make those public otherwise `refAllDeclsRecursive`
    // will try to compile `std.BoundedArray.Writer` and will produce a compile error.
    const DimsArray = std.BoundedArray(i64, MAX_RANK);
    const TagsArray = std.BoundedArray(Tag, MAX_RANK);
    const AxesArray = std.BoundedArray(u3, MAX_RANK);

    const UnknownTags: TagsArray = .{ .len = 0, .buffer = [_]Tag{TagUnknown} ** MAX_RANK };

    _dtype: DataType,
    _dims: DimsArray = .{},
    _tags: TagsArray = UnknownTags,

    pub fn parseDimensions(v: anytype) struct { DimsArray, TagsArray } {
        const T = @TypeOf(v);

        if (T == Shape) {
            return .{ v._dims, v._tags };
        }

        if (comptime meta.isSliceOfAny(T, meta.isInteger)) {
            var dims_ = DimsArray.init(0) catch unreachable;
            var tags_ = TagsArray.init(0) catch unreachable;
            for (v) |d| {
                dims_.appendAssumeCapacity(@intCast(d));
                tags_.appendAssumeCapacity(TagUnknown);
            }
            return .{ dims_, tags_ };
        }

        if (comptime meta.isStruct(T)) {
            var dims_: DimsArray = .{};
            var tags_: TagsArray = .{};
            inline for (std.meta.fields(T)) |field| {
                const fv = @field(v, field.name);
                if (comptime meta.isInteger(field.type)) {
                    dims_.appendAssumeCapacity(@intCast(fv));
                } else if (comptime isAutoDim(fv)) {
                    dims_.appendAssumeCapacity(-1);
                } else {
                    meta.compileError("Field {s} should be an integer or an auto dimension", .{field.name});
                }
                if (comptime meta.isTuple(T)) {
                    tags_.appendAssumeCapacity(TagUnknown);
                } else {
                    tags_.appendAssumeCapacity(toTag(field));
                }
            }

            return .{ dims_, tags_ };
        }

        meta.compileError("Wrong type, got {}", .{T});
    }

    test parseDimensions {
        const ref = Shape.init(.{ .a = 0, .b = 1 }, .f32);
        const dims_, const tags_ = parseDimensions(.{ .a = 0, .b = 1 });

        try testing.expectEqualSlices(i64, ref.dims(), dims_.constSlice());
        try testing.expectEqualSlices(Tag, ref.tags(), tags_.constSlice());
    }

    pub fn parseAxes(self: Shape, v: anytype) struct { AxesArray, TagsArray } {
        const T = @TypeOf(v);

        if (T == Shape) {
            return self.parseAxes(self.tags());
        }

        var axes_ = AxesArray.init(0) catch unreachable;
        var tags_ = TagsArray.init(0) catch unreachable;

        if (comptime meta.isSliceOfAny(T, isAxisConvertible)) {
            for (v) |d| {
                axes_.appendAssumeCapacity(self.axis(d));
                tags_.appendAssumeCapacity(self.tag(d));
            }
            return .{ axes_, tags_ };
        }

        if (comptime meta.isTupleOfAny(T, isAxisConvertible)) {
            inline for (std.meta.fields(T)) |field| {
                axes_.appendAssumeCapacity(self.axis(@field(v, field.name)));
                tags_.appendAssumeCapacity(self.tag(@field(v, field.name)));
            }
            return .{ axes_, tags_ };
        }

        meta.compileError("Wrong type, got {}. Expected .{{.a, .b}}", .{T});
    }

    pub fn parseTags(v: anytype) TagsArray {
        const T = @TypeOf(v);
        meta.assertComptime(meta.isTupleOf(T, EnumLiteral), "Wrong type, got {}. Expected .{{ .a, .b }}", .{T});
        var tags_ = TagsArray.init(0) catch unreachable;
        inline for (v) |field| {
            tags_.appendAssumeCapacity(toTag(field));
        }
        return tags_;
    }

    /// Create a shape from a struct literal, eg:
    /// Shape.init(.{ .h = 1024, .w = 512, .c = 3 });
    /// Shape.init(.{ 1024, 512, 3 });
    pub fn init(dimz: anytype, dt: DataType) Shape {
        var res: Shape = .{ ._dtype = dt };
        res._dims, res._tags = parseDimensions(dimz);
        return res;
    }

    /// Creates a Shape with dims set to `.{0, 1, 2, ..., rank-1}`.
    pub fn range(rank_: usize, dt: DataType) Shape {
        var res: Shape = .{ ._dtype = dt };
        for (0..rank_) |i| {
            res._dims.append(@intCast(i)) catch {
                meta.panic("Too many dimensions! Max: {d}, passed: {d}", .{ res._dims.capacity(), rank_ });
            };
            res._tags.append(TagUnknown) catch unreachable;
        }
        return res;
    }

    pub fn dtype(self: Shape) DataType {
        return self._dtype;
    }

    pub fn rank(self: Shape) u4 {
        self.ensureDimsAndTagsAreSync();
        return self._dims.len;
    }

    pub fn dim(self: Shape, ax: anytype) i64 {
        self.ensureDimsAndTagsAreSync();
        return self._dims.get(self.axis(ax));
    }

    pub fn dims(self: *const Shape) []const i64 {
        self.ensureDimsAndTagsAreSync();
        return self._dims.constSlice();
    }

    fn isAxisConvertible(comptime T: type) bool {
        return meta.isInteger(T) or isTagConvertible(T);
    }

    fn isTagConvertible(comptime T: type) bool {
        return switch (T) {
            EnumLiteral => true,
            std.builtin.Type.StructField => true,
            Tag => true,
            else => false,
        };
    }

    fn toTag(v: anytype) Tag {
        const T = @TypeOf(v);
        return switch (T) {
            EnumLiteral => @tagName(v).ptr,
            std.builtin.Type.StructField => v.name.ptr,
            Tag => v,
            else => meta.compileError("Value should be an EnumLiteral, a Shape.Tag or a StructField, got {}", .{T}),
        };
    }

    inline fn ensureDimsAndTagsAreSync(self: Shape) void {
        meta.assert(self._dims.len == self._tags.len, "Tags and dims have diverged! dims={d} tags={d}", .{ self._dims.len, self._tags.len });
    }

    pub fn tag(self: Shape, ax: anytype) Tag {
        self.ensureDimsAndTagsAreSync();
        return self._tags.get(self.axis(ax));
    }

    /// Returns a printable name for a given axis.
    /// Either the tag itself, or a digit if it's not tagged.
    pub fn debugTag(self: Shape, ax: usize) []const u8 {
        const t = self.tag(ax);
        if (t != TagUnknown) return std.mem.span(t);

        return "01234567"[ax .. ax + 1];
    }

    pub fn setTag(self: Shape, ax: anytype, tag_: anytype) Shape {
        var res = self;
        res._tags.set(self.axis(ax), toTag(tag_));
        return res;
    }

    pub fn tags(self: *const Shape) []const Tag {
        self.ensureDimsAndTagsAreSync();
        return self._tags.constSlice();
    }

    pub fn hasTag(self: Shape, tag_: anytype) ?u3 {
        return self.axisFromTagMaybe(toTag(tag_));
    }

    pub fn hasTags(self: Shape, tagz: anytype) bool {
        const T = @TypeOf(tagz);

        if (comptime meta.isSliceOf(T, Tag) or meta.isSliceOf(T, EnumLiteral)) {
            for (tagz) |t| {
                if (self.hasTag(t) == null) {
                    return false;
                }
            }
            return true;
        }

        if (comptime meta.isTupleOf(T, Tag) or meta.isTupleOf(T, EnumLiteral)) {
            inline for (tagz) |t| {
                if (self.hasTag(t) == null) {
                    return false;
                }
            }
            return true;
        }

        meta.compileError("Expected tuple of tags, got {any}", .{T});
    }

    pub fn isFullyTagged(self: Shape) bool {
        for (self._tags.constSlice()) |t| {
            if (t == TagUnknown) return false;
        }
        return true;
    }

    pub fn axis(self: Shape, axis_: anytype) u3 {
        self.ensureDimsAndTagsAreSync();

        const T = @TypeOf(axis_);
        if (comptime meta.isInteger(T)) {
            return self.axisFromInt(@intCast(axis_));
        }

        if (comptime isTagConvertible(T)) {
            return self.axisFromTag(toTag(axis_));
        }

        meta.compileError("Wrong axis type, expected .literal, or an integer, got: {any}", .{T});
    }

    pub fn axes(self: Shape, axes_: anytype) AxesArray {
        self.ensureDimsAndTagsAreSync();

        const T = @TypeOf(axes_);

        if (T == Shape) {
            return self.axes(axes_.tags());
        }

        var res = AxesArray.init(0) catch unreachable;

        if (comptime meta.isSliceOfAny(T, meta.isInteger) or meta.isSliceOf(T, Tag)) {
            for (axes_) |ax| {
                res.appendAssumeCapacity(self.axis(ax));
            }
            return res;
        }

        if (comptime meta.isStruct(T)) {
            inline for (std.meta.fields(T)) |field| {
                res.appendAssumeCapacity(self.axis(@field(axes_, field.name)));
            }
            return res;
        }

        meta.compileError("Wrong type, got {}", .{T});
    }

    fn axisFromInt(self: Shape, d: isize) u3 {
        const rank_: i8 = self.rank();
        if (d < 0) {
            return @intCast(d + rank_);
        }
        if (d > rank_) {
            meta.panic("Tensor doesn't have dimension: {d}", .{d});
        }
        return @intCast(d);
    }

    fn axisFromTagMaybe(self: Shape, d: Tag) ?u3 {
        if (d == TagUnknown) {
            return null;
        }
        if (@inComptime()) {
            for (0.., self.tags()) |tagIndex, t| {
                const a: []const u8 = std.mem.span(t);
                const b: []const u8 = std.mem.span(d);
                if (std.mem.eql(u8, a, b)) {
                    return @intCast(tagIndex);
                }
            }
            return null;
        }
        if (std.mem.indexOfScalar(Tag, self.tags(), d)) |d_| {
            return @intCast(d_);
        }
        return null;
    }

    fn axisFromTag(self: Shape, d: Tag) u3 {
        meta.assert(d != TagUnknown, "The unknown tag .{s} can't be used to fetch axis in {}", .{ d, self });
        return self.axisFromTagMaybe(d) orelse {
            meta.panic("Tensor {} doesn't have dimension with tag: {s}", .{ self, d });
        };
    }

    test axis {
        try testing.expectEqual(1, Shape.init(.{ 5, 2 }, .f32).axis(1));
        try testing.expectEqual(1, Shape.init(.{ 5, 2 }, .f32).axis(-1));
        try testing.expectEqual(1, Shape.init(.{ .a = 5, .b = 2 }, .f32).axis(.b));
    }

    /// The number of element inside a Tensor described by this shape.
    pub fn count(self: Shape) usize {
        var res: i64 = 1;
        for (self.dims()) |d| {
            meta.assert(d >= 0, "Can't count elements in shape with negative dimension: {}", .{self});
            res *= d;
        }
        return @intCast(res);
    }

    /// Total size in bytes needed to represent this shape.
    pub fn byteSize(self: Shape) usize {
        return self.dtype().sizeOf() * self.count();
    }

    // Aliases
    pub const numel = count;

    /// Compares the two shapes described, ignoring tagging.
    pub fn eql(self: Shape, other: Shape) bool {
        return std.mem.eql(i64, self.dims(), other.dims()) and self.dtype() == other.dtype();
    }

    /// Compares the two shapes described, ignoring tagging and dtype.
    pub fn eqlDims(self: Shape, other: Shape) bool {
        return std.mem.eql(i64, self.dims(), other.dims());
    }

    /// Compares the two shapes described including tags.
    pub fn eqlWithTags(self: Shape, other: Shape) bool {
        return self.eql(other) and std.mem.eql(Tag, self.tags(), other.tags()) and self.dtype() == other.dtype();
    }

    /// Format the shape.
    /// Default format: "Shape({.a=10, .b=20}, dtype=.f32)"
    /// Bare format {_}: "{.a=10, .b=20}, dtype=.f32"
    pub fn format(
        self: Shape,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        const bare_fmt = fmt.len == 1 and fmt[0] == '_';
        _ = try writer.write(if (bare_fmt) "{" else "Shape({");

        for (self.dims(), 0..) |d, i| {
            const prefix = if (i == 0) "" else ",";
            const t = self.tag(i);
            if (t != TagUnknown) {
                try writer.print("{s}.{s}={d}", .{ prefix, t, d });
            } else {
                try writer.print("{s}{d}", .{ prefix, d });
            }
        }
        _ = try writer.print("}}, dtype=.{s}", .{@tagName(self.dtype())});
        if (!bare_fmt) _ = try writer.write(")");
    }

    pub fn reshape(self: Shape, new_shape_: anytype) Shape {
        var new_shape: Shape = .{ ._dtype = self.dtype() };
        new_shape._dims, new_shape._tags = parseDimensions(new_shape_);
        new_shape.inferMissingAxis(self.count());
        meta.assert(self.count() == new_shape.count(), "Can't reshape {d} to {d}", .{ self.dims(), new_shape.dims() });
        return new_shape;
    }

    fn inferMissingAxis(self: *Shape, n_: usize) void {
        meta.assert(std.mem.count(i64, self.dims(), &.{-1}) < 2, "Cannot infer multiple dimensions when reshaping to: {}", .{self.*});

        const inferred_ax = std.mem.indexOfScalar(i64, self.dims(), -1) orelse return;
        // We can't use `self.count()` yet cause we have negative dims.
        var tmp_count: i64 = 1;
        for (self.dims()) |d| {
            if (d > 0) {
                tmp_count *= d;
            }
        }
        const n: i64 = @intCast(n_);
        // Abort, `reshape` will panic with more context.
        if (@mod(n, tmp_count) != 0) {
            return;
        }
        self._dims.set(inferred_ax, @divExact(n, tmp_count));
    }

    test reshape {
        const x = Shape.init(.{ 2, 5, 3 }, .f32);
        {
            const res = x.reshape(.{ .auto, 3 });
            try testing.expectEqualSlices(i64, &.{ 10, 3 }, res.dims());
        }
        {
            const res = x.reshape(.{ 10, -1 });
            try testing.expectEqualSlices(i64, &.{ 10, 3 }, res.dims());
        }
        {
            const res = x.reshape(.{-1});
            try testing.expectEqualSlices(i64, &.{30}, res.dims());
        }
    }

    pub fn setDim(self: Shape, ax: anytype, d: i64) Shape {
        var res = self;
        res._dims.set(self.axis(ax), d);
        return res;
    }

    pub const set = setDim;

    fn isAutoDim(v: anytype) bool {
        return toTag(v) == toTag(.auto);
    }

    fn isDynDim(v: anytype) bool {
        return toTag(v) == toTag(.dyn);
    }

    /// Inserts one ore more axes with the given dimensions, before the given axis.
    /// Negative axis is interpreted wrt the current shape.
    /// `.last` axis can be used to insert at the end (ie to append).
    /// ```
    /// .{10, 11, 12}.insert(1, 2) -> .{10, 2, 11, 12}
    /// .{10, 11, 12}.insert(-1, 2) -> .{10, 11, 2, 12}
    /// .{10, 11, 12}.insert(.last, 2) -> .{10, 11, 12, 2}
    /// ```
    pub fn insert(self: Shape, axis_: anytype, dimz: anytype) Shape {
        const dims_, const tags_ = parseDimensions(dimz);
        const ax = if (@TypeOf(axis_) == EnumLiteral and axis_ == .last)
            self.rank()
        else
            self.axis(axis_);

        var res = self;
        res._dims.insertSlice(ax, dims_.constSlice()) catch unreachable;
        res._tags.insertSlice(ax, tags_.constSlice()) catch unreachable;
        return res;
    }

    test insert {
        try testing.expectEqualSlices(i64, &.{ 10, 1, 11, 12 }, Shape.init(.{ 10, 11, 12 }, .f32).insert(1, .{1}).dims());
        try testing.expectEqualSlices(i64, &.{ 10, 11, 12, 1, 13 }, Shape.init(.{ 10, 11, 12, 13 }, .f32).insert(-1, .{1}).dims());
        try testing.expectEqualSlices(i64, &.{ 10, 11, 12, 13, 1 }, Shape.init(.{ 10, 11, 12, 13 }, .f32).insert(.last, .{1}).dims());
    }

    pub fn insertTag(self: Shape, axis_: anytype, d: i64, tag_: anytype) Shape {
        meta.assert(self.rank() < MAX_RANK - 1, "Can't insert new axis in {}, it's already at max rank.", .{self});

        const ax = if (@TypeOf(axis_) == EnumLiteral and axis_ == .last)
            self.rank()
        else
            self.axis(axis_);

        var res = self;
        res._dims.insert(ax, d) catch unreachable;
        res._tags.insert(ax, toTag(tag_)) catch unreachable;
        return res;
    }

    pub fn append(self: Shape, v: anytype) Shape {
        var res = self;
        const dims_, const tags_ = parseDimensions(v);
        res._dims.appendSliceAssumeCapacity(dims_.constSlice());
        res._tags.appendSliceAssumeCapacity(tags_.constSlice());
        return res;
    }

    test append {
        try testing.expectEqualSlices(
            i64,
            &.{ 10, 11, 12, 1 },
            Shape.init(.{ 10, 11, 12 }, .f32).append(.{1}).dims(),
        );

        try testing.expect(
            Shape.init(.{ .a = 10, .b = 11, .c = 12 }, .f32).eqlWithTags(
                Shape.init(.{ .a = 10, .b = 11 }, .f32).append(.{ .c = 12 }),
            ),
        );
    }

    pub fn appendDim(self: Shape, d: i64, tag_: ?Tag) Shape {
        var res = self;
        res._dims.appendAssumeCapacity(d);
        res._tags.appendAssumeCapacity(if (tag_) |t| t else TagUnknown);
        return res;
    }

    pub fn remove(self: Shape, d: anytype) Shape {
        var res = self;
        const d_ = self.axis(d);
        _ = res._dims.orderedRemove(d_);
        _ = res._tags.orderedRemove(d_);
        return res;
    }

    pub const drop = remove;

    test remove {
        try std.testing.expectEqualSlices(i64, &.{ 10, 12 }, Shape.init(.{ 10, 11, 12 }, .f32).remove(1).dims());
        try std.testing.expectEqualSlices(i64, &.{ 10, 11, 12 }, Shape.init(.{ 10, 11, 12, 13 }, .f32).remove(-1).dims());
    }

    pub fn transpose(self: Shape, permutations: anytype) Shape {
        std.debug.assert(self.rank() == permutations.len);
        const permutations_ = self.axes(permutations);
        var res = self;
        for (permutations_.constSlice(), 0..) |permutation, i| {
            res._dims.set(i, self.dim(permutation));
            res._tags.set(i, self.tag(permutation));
        }
        return res;
    }

    test transpose {
        try testing.expect(
            Shape.init(.{ 12, 11, 10 }, .f32).eql(
                Shape.init(.{ 10, 11, 12 }, .f32).transpose(.{ 2, 1, 0 }),
            ),
        );

        try testing.expect(
            Shape.init(.{ .a = 10, .c = 12, .b = 11, .d = 13 }, .f32).eqlWithTags(
                Shape.init(.{ .a = 10, .b = 11, .c = 12, .d = 13 }, .f32).transpose(.{ 0, 2, 1, 3 }),
            ),
        );
    }

    /// Tag each ax of this shape with tags from a tuple.
    pub fn withTags(self: Shape, tagz: anytype) Shape {
        const T = @TypeOf(tagz);

        if (T == Shape) {
            return self.withTags(tagz.tags());
        }

        var res = self;

        if (comptime meta.isSliceOf(T, Tag) or meta.isSliceOf(T, EnumLiteral)) {
            meta.assert(tagz.len == self.rank(), "Not enough tags for shape {}, got {any}", .{ self, tagz });
            for (tagz, 0..) |tag_, i| {
                res._tags.set(i, toTag(tag_));
            }
            return res;
        }

        if (comptime meta.isTupleOf(T, Tag) or meta.isTupleOf(T, EnumLiteral)) {
            meta.assert(tagz.len == self.rank(), "Not enough tags for shape {}, got {}", .{ self, tagz });
            inline for (tagz, 0..) |tag_, i| {
                res._tags.set(i, toTag(tag_));
            }
            return res;
        }

        meta.compileError("Expected a tuple of enum literals eg: .{ .a, .b, .c } got: {any}", .{@TypeOf(tagz)});
    }

    test withTags {
        {
            const tagged = Shape.init(.{ 0, 1 }, .f32).withTags(.{ .a, .b });
            try testing.expectEqual(0, tagged.axis(.a));
            try testing.expectEqual(1, tagged.axis(.b));
        }
        {
            const tagged = Shape.init(.{ 0, 1, 2 }, .f32).withTags(.{ ._, .a, .b });
            try testing.expectEqual(1, tagged.axis(.a));
            try testing.expectEqual(2, tagged.axis(.b));
        }
        {
            const tagged = Shape.init(.{ 0, 1, 2, 3 }, .f32).withTags(.{ ._, ._, .a, .b });
            try testing.expectEqual(2, tagged.axis(.a));
            try testing.expectEqual(3, tagged.axis(.b));
        }
    }

    /// Tag the last axes of this shape with tags from a tuple.
    pub fn withPartialTags(self: Shape, tagz: anytype) Shape {
        const T = @TypeOf(tagz);

        if (T == Shape) {
            return self.withPartialTags(tagz.tags());
        }

        var res = self;

        if (comptime meta.isSliceOf(T, Tag) or meta.isSliceOf(T, EnumLiteral)) {
            meta.assert(tagz.len <= self.rank(), "Too many tags for shape {}, got {any}", .{ self, tagz });
            for (tagz, self.rank() - tagz.len..) |tag_, i| {
                res._tags.set(i, toTag(tag_));
            }
            return res;
        }

        if (comptime meta.isTupleOf(T, Tag) or meta.isTupleOf(T, EnumLiteral)) {
            meta.assert(tagz.len <= self.rank(), "Too many tags for shape {}, got {}", .{ self, tagz });
            inline for (tagz, self.rank() - tagz.len..) |tag_, i| {
                res._tags.set(i, toTag(tag_));
            }
            return res;
        }

        meta.compileError("Expected a tuple of enum literals eg: .{ .a, .b, .c } got: {any}", .{@TypeOf(tagz)});
    }

    test withPartialTags {
        {
            const tagged = Shape.init(.{ 0, 1 }, .f32).withPartialTags(.{ .a, .b });
            try testing.expectEqual(0, tagged.axis(.a));
            try testing.expectEqual(1, tagged.axis(.b));
        }
        {
            const tagged = Shape.init(.{ 0, 1, 2 }, .f32).withPartialTags(.{ .a, .b });
            try testing.expectEqual(1, tagged.axis(.a));
            try testing.expectEqual(2, tagged.axis(.b));
        }
        {
            const tagged = Shape.init(.{ 0, 1, 2, 3, 4 }, .f32).withPartialTags(.{ .a, .b });
            try testing.expectEqual(3, tagged.axis(.a));
            try testing.expectEqual(4, tagged.axis(.b));
        }
        {
            const tagged = Shape.init(.{ 0, 1, 2, 3, 4, 5, 6 }, .f32).withPartialTags(.{ .a, .b, .c });
            try testing.expectEqual(4, tagged.axis(.a));
            try testing.expectEqual(5, tagged.axis(.b));
            try testing.expectEqual(6, tagged.axis(.c));
        }
    }

    pub fn withDtype(self: Shape, dt: DataType) Shape {
        var res = self;
        res._dtype = dt;
        return res;
    }

    /// Renames some of the tags in this shape.
    /// Shape.init(.{ .a = 10, .b = 20 }).rename(.{ .b = .batch }); // .{ .a = 10, .batch = 20 };
    pub fn rename(self: Shape, renames: anytype) Shape {
        const T = @TypeOf(renames);
        meta.assertComptime(meta.isStructOfAny(T, isAxisConvertible), "Must pass a struct of enum literals. Passed: {any}", .{T});
        var res = self;
        inline for (std.meta.fields(T)) |field| {
            res._tags.set(self.axis(field), toTag(@field(renames, field.name)));
        }
        return res;
    }

    test rename {
        {
            const tagged = Shape.init(.{ .a = 0, .b = 1 }, .f32).rename(.{ .a = .x, .b = .y });
            try testing.expectEqual(0, tagged.dim(.x));
            try testing.expectEqual(1, tagged.dim(.y));
        }
        {
            const tagged = Shape.init(.{ .a = 0, .b = 1, .c = 2 }, .f32).rename(.{ .a = .x, .c = .z });
            try testing.expectEqual(0, tagged.dim(.x));
            try testing.expectEqual(1, tagged.dim(.b));
            try testing.expectEqual(2, tagged.dim(.z));
        }
    }

    pub fn computeStrides(self: Shape, base_stride: u32) std.BoundedArray(i64, MAX_RANK) {
        const rk = self.rank();
        var strides: std.BoundedArray(i64, MAX_RANK) = .{ .len = @intCast(self.rank()) };
        if (rk == 0) return strides;
        strides.buffer[rk - 1] = base_stride;
        for (1..rk) |i| {
            const j = @as(usize, rk) - 1 - i;
            strides.buffer[j] = self._dims.get(j + 1) * strides.buffer[j + 1];
        }
        return strides;
    }

    /// Returns the permutation needed to transpose this shape
    /// so that the given axes are contiguous.
    pub fn contiguousPerm(self: Shape, axes_: anytype) AxesArray {
        const axes__, _ = self.parseAxes(axes_);
        var perms = AxesArray.init(0) catch unreachable;
        for (0..self.rank()) |i| {
            if (std.mem.indexOfScalar(u3, axes__.constSlice(), @intCast(i))) |_| {
                continue;
            }
            perms.appendAssumeCapacity(@intCast(i));
        }
        perms.appendSliceAssumeCapacity(axes__.constSlice());
        return perms;
    }

    test contiguousPerm {
        const abc = Shape.init(.{ .a = 10, .b = 11, .c = 12 }, .f32);
        try testing.expect(
            Shape.init(.{ .b = 11, .c = 12, .a = 10 }, .f32).eqlWithTags(
                abc.transpose(abc.contiguousPerm(.{.a}).constSlice()),
            ),
        );
        try testing.expect(
            Shape.init(.{ .c = 12, .b = 11, .a = 10 }, .f32).eqlWithTags(
                abc.transpose(abc.contiguousPerm(.{ .b, .a }).constSlice()),
            ),
        );
        const abcd = Shape.init(.{ .a = 10, .b = 11, .c = 12, .d = 13 }, .f32);
        try testing.expect(
            Shape.init(.{ .a = 10, .c = 12, .b = 11, .d = 13 }, .f32).eqlWithTags(
                abcd.transpose(abcd.contiguousPerm(.{ .b, .d }).constSlice()),
            ),
        );

        const abcde = Shape.init(.{ .a = 10, .b = 11, .c = 12, .d = 13, .e = 14 }, .f32);
        try testing.expect(
            Shape.init(.{ .a = 10, .b = 11, .d = 13, .c = 12, .e = 14 }, .f32).eqlWithTags(
                abcde.transpose(abcde.contiguousPerm(.{ .b, .d, .c, .e }).constSlice()),
            ),
        );
    }

    /// Splits the given axis in several axes.
    /// eg: `Shape.init(.{ .a = 10, .b = 3 }).split(.a, .{.a1 = 5, .a2 = 2}); -> .{.a1 = 5, .a2 = 2, .b = 3}`
    /// The number of elements in the split shape must match the number of element
    /// in the target axis.
    pub fn splitAxis(self: Shape, axis_: anytype, split_shape_: anytype) Shape {
        const ax = self.axis(axis_);
        const dims_, const tags_ = parseDimensions(split_shape_);
        var new_shape = self;
        new_shape._dims.replaceRange(ax, 1, dims_.constSlice()) catch unreachable;
        new_shape._tags.replaceRange(ax, 1, tags_.constSlice()) catch unreachable;
        new_shape.inferMissingAxis(self.count());
        return new_shape;
    }

    test splitAxis {
        try testing.expect(
            Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3 }, .f32).eql(
                Shape.init(.{ .a = 10, .b = 3 }, .f32).splitAxis(.a, .{ .a1 = 5, .a2 = 2 }),
            ),
        );
        try testing.expect(
            Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3 }, .f32).eql(
                Shape.init(.{ .a = 10, .b = 3 }, .f32).splitAxis(.a, .{ .a1 = .auto, .a2 = 2 }),
            ),
        );
    }

    pub fn splitAxes(self: Shape, axes_: anytype) Shape {
        const T = @TypeOf(axes_);
        meta.assertComptime(meta.isStruct(T), "Must pass struct of enum literals like .{ .a = .{ .a1, .a2 } }. Passed: {any}", .{T});

        var res = self;
        inline for (std.meta.fields(T)) |field| {
            res = res.splitAxis(field, @field(axes_, field.name));
        }
        return res;
    }

    test splitAxes {
        try testing.expect(
            Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3 }, .f32).eql(
                Shape.init(.{ .a = 10, .b = 3 }, .f32).splitAxes(.{ .a = .{ .a1 = 5, .a2 = .auto } }),
            ),
        );
        try testing.expect(
            Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3 }, .f32).eql(
                Shape.init(.{ .a = 10, .b = 3 }, .f32).splitAxes(.{ .a = .{ .a1 = 5, .a2 = .auto } }),
            ),
        );
    }

    /// Merge the given axes into one axis.
    /// eg: `Shape.init(.{.a1 = 5, .a2 = 2, .b = 3}).merge(.{ .a = .{ .a1, .a2 }); -> .{ .a = 10, .b = 3 }`
    pub fn mergeAxis(self: Shape, axis_: anytype, axes_: anytype) Shape {
        const axes__ = self.axes(axes_);

        const first_axis = axes__.get(0);
        const last_axis = axes__.get(axes__.len - 1);

        var new_dim: i64 = 1;
        for (axes__.constSlice(), first_axis..) |ax, counter| {
            new_dim *= self.dim(ax);
            meta.assert(ax == counter, "Can't merge shape {} along non-contiguous axes {any}", .{ self, axes_ });
        }

        var new_shape = self;
        new_shape._dims.set(first_axis, new_dim);
        new_shape._dims.replaceRange(first_axis + 1, self.dims()[first_axis + 1 ..].len, self.dims()[last_axis + 1 ..]) catch unreachable;
        new_shape._tags.set(first_axis, if (comptime isTagConvertible(@TypeOf(axis_))) toTag(axis_) else TagUnknown);
        new_shape._tags.replaceRange(first_axis + 1, self.dims()[first_axis + 1 ..].len, self.tags()[last_axis + 1 ..]) catch unreachable;
        return new_shape;
    }

    test mergeAxis {
        try testing.expect(
            Shape.init(.{ .a = 10, .b = 3, .c = 4 }, .f32).eqlWithTags(
                Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3, .c = 4 }, .f32).mergeAxis(.a, .{ .a1, .a2 }),
            ),
        );
        try testing.expect(
            Shape.init(.{ .a = 5, .c = 6 }, .f32).eqlWithTags(
                Shape.init(.{ .a = 5, .b1 = 2, .b2 = 3 }, .f32).mergeAxis(.c, .{ .b1, .b2 }),
            ),
        );
        try testing.expect(
            Shape.init(.{ .a = 10, .b = 3 }, .f32).eqlWithTags(
                Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3 }, .f32).mergeAxis(.a, .{ toTag(.a1), toTag(.a2) }),
            ),
        );
        try testing.expect(
            Shape.init(.{ .a = 10, .b = 3 }, .f32).eqlWithTags(
                Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3 }, .f32).mergeAxis(toTag(.a), @as([]const Tag, &.{ toTag(.a1), toTag(.a2) })),
            ),
        );
        try testing.expect(
            Shape.init(.{ .a = 10, .b = 3 }, .f32).eqlWithTags(
                Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3 }, .f32).mergeAxis(.a, @as([]const usize, &.{ 0, 1 })),
            ),
        );
    }

    pub fn mergeAxes(self: Shape, axes_: anytype) Shape {
        const T = @TypeOf(axes_);
        meta.assertComptime(meta.isStruct(T), "Must pass struct of enum literals like .{ .a = .{ .a1, .a2 } }. Passed: {any}", .{T});

        var res = self;
        inline for (std.meta.fields(T)) |field| {
            meta.assertComptime(meta.isTupleOfAny(field.type, isAxisConvertible) or meta.isSliceOfAny(field.type, isAxisConvertible), "Must pass struct of axes. Passed: {any}", .{field.type});
            res = res.mergeAxis(field, @field(axes_, field.name));
        }
        return res;
    }

    test mergeAxes {
        try testing.expect(
            Shape.init(.{ .a = 10, .b = 3 }, .f32).eqlWithTags(
                Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3 }, .f32).mergeAxes(.{ .a = .{ .a1, .a2 } }),
            ),
        );
        try testing.expect(
            Shape.init(.{ .a = 10, .b = 3 }, .f32).eqlWithTags(
                Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3 }, .f32).mergeAxes(.{ .a = .{ toTag(.a1), toTag(.a2) } }),
            ),
        );
        try testing.expect(
            Shape.init(.{ .a = 10, .b = 3 }, .f32).eqlWithTags(
                Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3 }, .f32).mergeAxes(.{ .a = .{ 0, 1 } }),
            ),
        );
        try testing.expect(
            Shape.init(.{ .a = 10, .b = 3 }, .f32).eqlWithTags(
                Shape.init(.{ .a1 = 5, .a2 = 2, .b = 3 }, .f32).mergeAxes(.{ .a = @as([]const usize, &.{ 0, 1 }) }),
            ),
        );
    }

    /// Parses an anytype argument of the form `val` or `.{ .a = val }`.l
    /// Helps offering consistent API through ZML.
    // pub fn parseTaggedValue(
    //     T: type,
    //     default_tag: EnumLiteral,
    //     d: anytype,
    // ) struct { Tag, T } {
    //     const err_msg = "Expected one tagged dimension, received a tuple: " ++ @typeName(@TypeOf(d));
    //     return switch (@typeInfo(@TypeOf(d))) {
    //         .Int, .ComptimeInt => .{ toTag(default_tag), @intCast(d) },
    //         .Struct => |struct_info| {
    //             if (struct_info.fields.len != 1) @compileError(err_msg);
    //             const name = struct_info.fields[0].name;
    //             return .{ name.ptr, @intCast(@field(d, name)) };
    //         },
    //         else => @compileError(err_msg),
    //     };
    // }

    /// Parses a list of tags `.{ .a, .b, .c }` into a `[]Tag`
    // pub inline fn parseTagList(comptime axes_: anytype) []Tag {
    //     switch (@typeInfo(@TypeOf(axes_))) {
    //         .Struct, .Array => {
    //             var _tags: [axes_.len]Tag = undefined;
    //             inline for (axes_, &_tags) |a, *t| t.* = toTag(a);
    //             return &_tags;
    //         },
    //         else => @compileError("Expected a tuple of enum literal, but found " ++ @tagName(@TypeOf(axes))),
    //     }
    // }

    /// Parses a comptime struct into a struct similarly to Shape.init,
    /// but with a custom type in place of the `i64` dimensions.
    /// Helps offering consistent API through ZML.
    // pub fn parseShapedValue(T: type, value: anytype) struct {
    //     std.BoundedArray(Tag, MAX_RANK),
    //     std.BoundedArray(T, MAX_RANK),
    // } {
    //     const too_long_err = std.fmt.comptimePrint("Received too many axes, maximum supported is {d}", .{MAX_RANK});

    //     var _tags: [MAX_RANK]Tag = [_]Tag{TagUnknown} ** MAX_RANK;
    //     const struct_info = switch (@typeInfo(@TypeOf(value))) {
    //         .Struct => |struct_info| struct_info,
    //         else => return .{
    //             .{ .len = 0, .buffer = _tags },
    //             std.BoundedArray(T, MAX_RANK).fromSlice(value) catch @panic(too_long_err),
    //         },
    //     };

    //     meta.assertComptime(struct_info.fields.len <= MAX_RANK, too_long_err, .{});

    //     var values: std.BoundedArray(T, MAX_RANK) = .{};
    //     inline for (struct_info.fields) |field| {
    //         if (T == Tag) {
    //             values.appendAssumeCapacity(toTag(@field(value, field.name)));
    //         } else {
    //             // If you have an error here it means Zig wasn't able to convert between the
    //             // value you passed and the expected `T`.
    //             values.appendAssumeCapacity(@field(value, field.name));
    //         }
    //     }
    //     if (!struct_info.is_tuple) {
    //         inline for (struct_info.fields, 0..) |field, i| {
    //             _tags[i] = toTag(field);
    //         }
    //     }
    //     return .{
    //         .{ .len = struct_info.fields.len, .buffer = _tags },
    //         values,
    //     };
    // }

    fn intersectTags(a: []const Tag, b: []const Tag) TagsArray {
        var res = TagsArray.init(0) catch unreachable;
        for (a) |tag_| {
            if (std.mem.indexOfScalar(Tag, b, tag_)) {
                res.appendAssumeCapacity(tag_);
            }
        }
        return res;
    }

    pub fn parseStruct(T: type, v: anytype) struct { std.BoundedArray(T, MAX_RANK), TagsArray } {
        const V = @TypeOf(v);

        var vals_: std.BoundedArray(T, MAX_RANK) = .{};
        var tags_: TagsArray = .{};

        if (comptime meta.isSliceOf(V, T)) {
            for (v) |d| {
                vals_.appendAssumeCapacity(d);
            }
            return .{ vals_, tags_ };
        }

        if (comptime meta.isStruct(V)) {
            const fields = std.meta.fields(V);
            meta.assertComptime(fields.len <= MAX_RANK, "Too many fields in struct {} ({d}). Max supported is {d}.", .{ V, fields.len, MAX_RANK });
            inline for (fields) |field| {
                const fv = @field(v, field.name);
                vals_.appendAssumeCapacity(fv);

                if (!comptime meta.isTuple(V)) {
                    tags_.appendAssumeCapacity(toTag(field));
                }
            }
            return .{ vals_, tags_ };
        }

        meta.compileError("parseStruct expects struct or tuple, got {}", .{V});
    }

    test parseStruct {
        const vals_, const tags_ = parseStruct(f32, .{ .a = 0.1, .b = 1.2 });

        try testing.expectEqualSlices(f32, &.{ 0.1, 1.2 }, vals_.constSlice());
        try testing.expectEqualSlices(Tag, &.{ "a".ptr, "b".ptr }, tags_.constSlice());
    }

    test "comptimeShape" {
        comptime {
            const s = Shape.init(.{ .a = 5, .b = 6, .c = 7 }, .f32);
            try std.testing.expectEqual(3, s.rank());
            try std.testing.expectEqual(4 * 5 * 6 * 7, s.byteSize());
            try std.testing.expectEqual(1, s.axis(.b));
        }

        // comptime only the shape
        {
            const s = comptime Shape.init(.{ .a = 5, .b = 6, .c = 7 }, .f32);
            try std.testing.expectEqual(3, s.rank());
            try std.testing.expectEqual(4 * 5 * 6 * 7, s.byteSize());
            try std.testing.expectEqual(1, s.axis(.b));
        }
    }
};
