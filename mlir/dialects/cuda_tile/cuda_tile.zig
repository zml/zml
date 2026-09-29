//! Zig bindings for NVIDIA's `cuda_tile` MLIR dialect (CUDA Tile IR): one
//! function per op in `Ops.td` order. Types and enums go through the dialect's
//! C API; what it has no getter for is parsed from text.

const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const dialect_namespace = "cuda_tile";

fn stringRef(s: []const u8) c.MlirStringRef {
    return .{ .data = s.ptr, .length = s.len };
}

fn typeFromC(t: c.MlirType) *const mlir.Type {
    return @ptrCast(t.ptr orelse @panic("cuda_tile: null MlirType"));
}

fn attrFromC(a: c.MlirAttribute) *const mlir.Attribute {
    return @ptrCast(a.ptr orelse @panic("cuda_tile: null MlirAttribute"));
}

const null_attr: c.MlirAttribute = .{ .ptr = null };

/// `!cuda_tile.ptr<T>` — a typed global-memory pointer.
pub const PointerType = opaque {
    const M = mlir.Methods(PointerType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsAPointerType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, pointee_type: *const mlir.Type) *const PointerType {
        return @ptrCast(typeFromC(c.mlirCudaTilePointerTypeGet(ctx.ptr(), pointee_type.ptr())));
    }

    pub fn pointee(self: *const PointerType) *const mlir.Type {
        return typeFromC(c.mlirCudaTilePointerTypeGetPointeeType(self.ptr()));
    }
};

pub fn pointerType(ctx: *mlir.Context, pointee_type: *const mlir.Type) *const mlir.Type {
    return @ptrCast(PointerType.get(ctx, pointee_type));
}

/// `!cuda_tile.tile<SHAPExT>` — every SSA value that carries data. Rank 0 is
/// the scalar. Dimensions must be powers of two.
pub const TileType = opaque {
    const M = mlir.Methods(TileType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsATileType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, shape: []const i64, elem: *const mlir.Type) *const TileType {
        return @ptrCast(typeFromC(c.mlirCudaTileTileTypeGet(ctx.ptr(), @intCast(shape.len), shape.ptr, elem.ptr())));
    }

    pub fn elementType(self: *const TileType) *const mlir.Type {
        return typeFromC(c.mlirCudaTileTileTypeGetElementType(self.ptr()));
    }

    pub fn rank(self: *const TileType) usize {
        return @intCast(c.mlirCudaTileTileTypeGetRank(self.ptr()));
    }

    pub fn dimension(self: *const TileType, i: usize) i64 {
        return c.mlirCudaTileTileTypeGetDimSize(self.ptr(), @intCast(i));
    }
};

pub fn tileType(ctx: *mlir.Context, shape: []const i64, elem: *const mlir.Type) *const mlir.Type {
    return @ptrCast(TileType.get(ctx, shape, elem));
}

/// `!cuda_tile.token` — the ordering token every `*_tko` op produces.
pub const TokenType = opaque {
    const M = mlir.Methods(TokenType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsATokenType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context) *const TokenType {
        return @ptrCast(typeFromC(c.mlirCudaTileTokenTypeGet(ctx.ptr())));
    }
};

pub fn tokenType(ctx: *mlir.Context) *const mlir.Type {
    return @ptrCast(TokenType.get(ctx));
}

/// `!cuda_tile.tensor_view<SHAPExT, strides=[...]>`. `dynamic()` marks a
/// dimension or stride that arrives as an operand of `make_tensor_view`.
pub const TensorViewType = opaque {
    const M = mlir.Methods(TensorViewType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsATensorViewType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn dynamic() i64 {
        return c.mlirCudaTileTensorViewTypeGetDynamicSize();
    }

    pub fn get(ctx: *mlir.Context, elem: *const mlir.Type, shape: []const i64, strides: []const i64) *const TensorViewType {
        return @ptrCast(typeFromC(c.mlirCudaTileTensorViewTypeGet(
            ctx.ptr(),
            elem.ptr(),
            @intCast(shape.len),
            shape.ptr,
            @intCast(strides.len),
            strides.ptr,
        )));
    }

    pub fn elementType(self: *const TensorViewType) *const mlir.Type {
        return typeFromC(c.mlirCudaTileTensorViewTypeGetElementType(self.ptr()));
    }

    pub fn rank(self: *const TensorViewType) usize {
        return @intCast(c.mlirCudaTileTensorViewTypeGetRank(self.ptr()));
    }

    pub fn dimension(self: *const TensorViewType, i: usize) i64 {
        return c.mlirCudaTileTensorViewTypeGetDimSize(self.ptr(), @intCast(i));
    }

    pub fn stride(self: *const TensorViewType, i: usize) i64 {
        return c.mlirCudaTileTensorViewTypeGetStride(self.ptr(), @intCast(i));
    }
};

pub fn tensorViewType(ctx: *mlir.Context, elem: *const mlir.Type, shape: []const i64, strides: []const i64) *const mlir.Type {
    return @ptrCast(TensorViewType.get(ctx, elem, shape, strides));
}

const DimMap = stdx.BoundedArray(i32, mlir.ShapedType.MAX_RANK);

fn identityDimMap(dim_map: []const i32, rank: usize) DimMap {
    var out: DimMap = .empty;
    if (dim_map.len > 0) {
        out.appendSliceAssumeCapacity(dim_map);
    } else {
        for (0..rank) |i| out.appendAssumeCapacity(@intCast(i));
    }
    return out;
}

/// `!cuda_tile.partition_view<tile=(...), padding_value = ..., tensor_view<...>, dim_map=[...]>`.
/// The tile shape, padding and dim_map all live in the type; the op that
/// builds one has no attributes of its own.
pub const PartitionViewType = opaque {
    const M = mlir.Methods(PartitionViewType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsAPartitionViewType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    /// An empty `dim_map` is the identity; the type requires one entry per
    /// tile dimension.
    pub fn get(
        ctx: *mlir.Context,
        tile_shape: []const i32,
        tensor_view: *const mlir.Type,
        dim_map: []const i32,
        padding: ?PaddingValue,
    ) *const PartitionViewType {
        const tile_attr: *const mlir.Attribute = .denseArray(ctx, .i32, tile_shape);
        const map = identityDimMap(dim_map, tile_shape.len);
        return @ptrCast(typeFromC(c.mlirCudaTilePartitionViewTypeGet(
            ctx.ptr(),
            tile_attr.ptr(),
            tensor_view.ptr(),
            @intCast(map.len),
            map.constSlice().ptr,
            if (padding) |p| p.attribute(ctx).ptr() else null_attr,
        )));
    }

    pub fn tensorView(self: *const PartitionViewType) *const mlir.Type {
        return typeFromC(c.mlirCudaTilePartitionViewTypeGetTensorView(self.ptr()));
    }

    pub fn viewTileType(self: *const PartitionViewType) *const mlir.Type {
        return typeFromC(c.mlirCudaTilePartitionViewTypeGetViewTileType(self.ptr()));
    }

    pub fn viewIndexRank(self: *const PartitionViewType) usize {
        return @intCast(c.mlirCudaTilePartitionViewTypeGetViewIndexRank(self.ptr()));
    }
};

/// `!cuda_tile.strided_view<tile=(...), strides=(...), ...>` (13.3).
pub const StridedViewType = opaque {
    const M = mlir.Methods(StridedViewType, c.MlirType);

    pub const isAFn = c.mlirCudaTileTypeIsAStridedViewType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn get(
        ctx: *mlir.Context,
        tile_shape: []const i32,
        traversal_strides: []const i32,
        tensor_view: *const mlir.Type,
        dim_map: []const i32,
        padding: ?PaddingValue,
    ) *const StridedViewType {
        const tile_attr: *const mlir.Attribute = .denseArray(ctx, .i32, tile_shape);
        const strides_attr: *const mlir.Attribute = .denseArray(ctx, .i32, traversal_strides);
        const map = identityDimMap(dim_map, tile_shape.len);
        return @ptrCast(typeFromC(c.mlirCudaTileStridedViewTypeGet(
            ctx.ptr(),
            tile_attr.ptr(),
            strides_attr.ptr(),
            tensor_view.ptr(),
            @intCast(map.len),
            map.constSlice().ptr,
            if (padding) |p| p.attribute(ctx).ptr() else null_attr,
        )));
    }

    pub fn viewTileType(self: *const StridedViewType) *const mlir.Type {
        return typeFromC(c.mlirCudaTileStridedViewTypeGetViewTileType(self.ptr()));
    }

    pub fn viewIndexRank(self: *const StridedViewType) usize {
        return @intCast(c.mlirCudaTileStridedViewTypeGetViewIndexRank(self.ptr()));
    }
};

/// `!cuda_tile.gather_scatter_view<tile=(...), padding_value=..., tensor_view<...>, sparse_dim=N>`
/// (13.3). The C API has no getter for it, so it is parsed from text.
pub fn gatherScatterViewType(
    ctx: *mlir.Context,
    tile_shape: []const i64,
    tensor_view: *const mlir.Type,
    sparse_dim: u32,
    padding: ?PaddingValue,
) *const mlir.Type {
    var buf: [1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    w.writeAll("!cuda_tile.gather_scatter_view<tile=(") catch unreachable;
    for (tile_shape, 0..) |d, i| {
        if (i > 0) w.writeAll("x") catch unreachable;
        w.print("{d}", .{d}) catch unreachable;
    }
    w.writeAll("), ") catch unreachable;
    if (padding) |p| w.print("padding_value={s}, ", .{@tagName(p)}) catch unreachable;
    w.print("{f}, sparse_dim={d}>", .{ tensor_view, sparse_dim }) catch unreachable;
    const text = w.buffered();
    return mlir.Type.parse(ctx, text) catch std.debug.panic("cuda_tile: cannot parse '{s}'", .{text});
}

/// The `tile<...>` type a load through `view` yields, for any view kind —
/// through the dialect's `TileView` type interface (`cuda_tile_capi.cc`), so
/// the gather/scatter view, which has no getters in NVIDIA's C API, works too.
pub fn tileTypeOfView(view: *const mlir.Type) *const mlir.Type {
    if (!c.zmlCudaTileTypeIsATileView(view.ptr())) std.debug.panic("cuda_tile: {f} is not a tiled view", .{view});
    return typeFromC(c.zmlCudaTileTileViewGetViewTileType(view.ptr()));
}

pub fn indexRankOfView(view: *const mlir.Type) usize {
    if (!c.zmlCudaTileTypeIsATileView(view.ptr())) std.debug.panic("cuda_tile: {f} is not a tiled view", .{view});
    return @intCast(c.zmlCudaTileTileViewGetViewIndexRank(view.ptr()));
}

// Enum attributes: built through the C getter from the tag name, so a typo is
// a null attribute rather than a wrong case.

fn enumAttribute(comptime getter: anytype, ctx: *mlir.Context, name: []const u8) *const mlir.Attribute {
    return attrFromC(getter(ctx.ptr(), stringRef(name)));
}

pub const RoundingMode = enum {
    nearest_even,
    zero,
    negative_inf,
    positive_inf,
    approx,
    full,
    nearest_int_to_zero,

    pub fn attribute(self: RoundingMode, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileRoundingModeAttrGet, ctx, @tagName(self));
    }
};

pub const Signedness = enum {
    signed,
    unsigned,

    pub fn attribute(self: Signedness, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileSignednessAttrGet, ctx, @tagName(self));
    }
};

pub const IntegerOverflow = enum {
    none,
    no_signed_wrap,
    no_unsigned_wrap,
    no_wrap,

    pub fn attribute(self: IntegerOverflow, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileIntegerOverflowAttrGet, ctx, @tagName(self));
    }
};

pub const ComparisonOrdering = enum {
    unordered,
    ordered,

    pub fn attribute(self: ComparisonOrdering, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileComparisonOrderingAttrGet, ctx, @tagName(self));
    }
};

pub const ComparisonPredicate = enum {
    equal,
    not_equal,
    less_than,
    less_than_or_equal,
    greater_than,
    greater_than_or_equal,

    pub fn attribute(self: ComparisonPredicate, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileComparisonPredicateAttrGet, ctx, @tagName(self));
    }
};

pub const AtomicRMWMode = enum {
    @"and",
    @"or",
    xor,
    add,
    addf,
    max,
    min,
    umax,
    umin,
    xchg,

    pub fn attribute(self: AtomicRMWMode, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileAtomicRMWModeAttrGet, ctx, @tagName(self));
    }
};

pub const MemoryScope = enum {
    tl_blk,
    device,
    sys,

    pub fn attribute(self: MemoryScope, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileMemoryScopeAttrGet, ctx, @tagName(self));
    }
};

pub const MemoryOrdering = enum {
    weak,
    relaxed,
    acquire,
    release,
    acq_rel,

    pub fn attribute(self: MemoryOrdering, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileMemoryOrderingSemanticsAttrGet, ctx, @tagName(self));
    }
};

pub const PaddingValue = enum {
    zero,
    neg_zero,
    nan,
    pos_inf,
    neg_inf,

    pub fn attribute(self: PaddingValue, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTilePaddingValueAttrGet, ctx, @tagName(self));
    }
};

pub const SymbolVisibility = enum {
    public,
    private,

    pub fn attribute(self: SymbolVisibility, ctx: *mlir.Context) *const mlir.Attribute {
        return enumAttribute(c.mlirCudaTileSymbolVisibilityAttrGet, ctx, @tagName(self));
    }
};

// optimization_hints: per-architecture dictionaries, built from text so
// several architectures can be listed.

pub const Arch = enum { default, sm_80, sm_86, sm_87, sm_88, sm_89, sm_90, sm_100, sm_103, sm_110, sm_120, sm_121 };

/// Hints `entry` accepts: `num_cta_in_cga` (power of two <= 16),
/// `num_worker_warps_per_cta` (4 or 8), `occupancy`.
pub const EntryHint = struct {
    arch: Arch = .default,
    num_cta_in_cga: ?i32 = null,
    num_worker_warps_per_cta: ?i32 = null,
    occupancy: ?i32 = null,
};

/// Hints `load_view_tko` / `store_view_tko` accept.
pub const LoadStoreHint = struct {
    arch: Arch = .default,
    allow_tma: ?bool = null,
    latency: ?i32 = null,
};

/// Hints `load_ptr_tko` / `store_ptr_tko` accept: `latency` only, the
/// pointer arm cannot use TMA. (The dialect drops unsupported keys silently.)
pub const PtrLoadStoreHint = struct {
    arch: Arch = .default,
    latency: ?i32 = null,
};

fn writeHintFields(w: *std.Io.Writer, hint: anytype) !void {
    var first = true;
    inline for (@typeInfo(@TypeOf(hint)).@"struct".fields) |f| {
        if (comptime std.mem.eql(u8, f.name, "arch")) continue;
        if (@field(hint, f.name)) |v| {
            if (!first) try w.writeAll(", ");
            first = false;
            switch (@TypeOf(v)) {
                bool => try w.print("{s} = {}", .{ f.name, v }),
                else => try w.print("{s} = {d}", .{ f.name, v }),
            }
        }
    }
}

/// `#cuda_tile.optimization_hints<sm_120 = {num_cta_in_cga = 2}, ...>` from a
/// slice of `EntryHint` or `LoadStoreHint`.
pub fn optimizationHints(ctx: *mlir.Context, hints: anytype) *const mlir.Attribute {
    var buf: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    w.writeAll("#cuda_tile.optimization_hints<") catch unreachable;
    for (hints, 0..) |h, i| {
        if (i > 0) w.writeAll(", ") catch unreachable;
        w.print("{s} = {{", .{@tagName(h.arch)}) catch unreachable;
        writeHintFields(&w, h) catch unreachable;
        w.writeAll("}") catch unreachable;
    }
    w.writeAll(">") catch unreachable;
    const text = w.buffered();
    return mlir.Attribute.parse(ctx, text) catch std.debug.panic("cuda_tile: cannot parse '{s}'", .{text});
}

// assume predicates: no C getters; parsed from their documented spellings.

fn parseAttr(ctx: *mlir.Context, comptime fmt: []const u8, args: anytype) *const mlir.Attribute {
    var buf: [256]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, fmt, args) catch unreachable;
    return mlir.Attribute.parse(ctx, text) catch std.debug.panic("cuda_tile: cannot parse '{s}'", .{text});
}

/// `#cuda_tile.div_by<N>`, or `<N, every E along A>` for a per-group claim.
pub fn divBy(ctx: *mlir.Context, divisor: u64, every: ?i64, along: ?i64) *const mlir.Attribute {
    if (every) |e| {
        return parseAttr(ctx, "#cuda_tile.div_by<{d}, every {d} along {d}>", .{ divisor, e, along orelse 0 });
    }
    return parseAttr(ctx, "#cuda_tile.div_by<{d}>", .{divisor});
}

pub fn sameElements(ctx: *mlir.Context, values: []const i64) *const mlir.Attribute {
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    w.writeAll("#cuda_tile.same_elements<[") catch unreachable;
    for (values, 0..) |v, i| {
        if (i > 0) w.writeAll(", ") catch unreachable;
        w.print("{d}", .{v}) catch unreachable;
    }
    w.writeAll("]>") catch unreachable;
    const text = w.buffered();
    return mlir.Attribute.parse(ctx, text) catch std.debug.panic("cuda_tile: cannot parse '{s}'", .{text});
}

/// `#cuda_tile.bounded<lb, ub>`, either side `?` when null.
pub fn bounded(ctx: *mlir.Context, lb: ?i64, ub: ?i64) *const mlir.Attribute {
    var buf: [128]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    w.writeAll("#cuda_tile.bounded<") catch unreachable;
    if (lb) |v| w.print("{d}", .{v}) catch unreachable else w.writeAll("?") catch unreachable;
    w.writeAll(", ") catch unreachable;
    if (ub) |v| w.print("{d}", .{v}) catch unreachable else w.writeAll("?") catch unreachable;
    w.writeAll(">") catch unreachable;
    const text = w.buffered();
    return mlir.Attribute.parse(ctx, text) catch std.debug.panic("cuda_tile: cannot parse '{s}'", .{text});
}

// Constant values: DenseElementsAttr over a tile type, which is a ShapedType.

pub fn splatAttr(tile_ty: *const mlir.Type, elem_attr: *const mlir.Attribute) *const mlir.Attribute {
    return attrFromC(c.mlirDenseElementsAttrSplatGet(tile_ty.ptr(), elem_attr.ptr()));
}

pub fn intElem(elem_ty: *const mlir.Type, value: i64) *const mlir.Attribute {
    return attrFromC(c.mlirIntegerAttrGet(elem_ty.ptr(), value));
}

pub fn floatElem(ctx: *mlir.Context, elem_ty: *const mlir.Type, value: f64) *const mlir.Attribute {
    return attrFromC(c.mlirFloatAttrDoubleGet(ctx.ptr(), elem_ty.ptr(), value));
}

/// A dense `DenseElementsAttr` over `tile_ty` from a raw element buffer; the
/// element type's bit width must be a multiple of 8.
pub fn denseAttr(tile_ty: *const mlir.Type, values: anytype) *const mlir.Attribute {
    return .denseElements(@ptrCast(tile_ty), values);
}

const MakeArgs = mlir.Operation.MakeArgs;

fn opName(comptime mnemonic: []const u8) []const u8 {
    return dialect_namespace ++ "." ++ mnemonic;
}

/// Attribute lists are passed by value so a helper's temporary outlives the
/// slice handed to `Operation.make`.
const Attrs = stdx.BoundedArray(mlir.NamedAttribute, 8);

fn attrsOf(list: []const mlir.NamedAttribute) Attrs {
    var attrs: Attrs = .empty;
    attrs.appendSliceAssumeCapacity(list);
    return attrs;
}

fn unary(ctx: *mlir.Context, comptime mnemonic: []const u8, source: *const mlir.Value, attrs: Attrs, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{source.type_()} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

fn binary(ctx: *mlir.Context, comptime mnemonic: []const u8, lhs: *const mlir.Value, rhs: *const mlir.Value, attrs: Attrs, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{lhs.type_()} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

fn convert(ctx: *mlir.Context, comptime mnemonic: []const u8, source: *const mlir.Value, result_type: *const mlir.Type, attrs: Attrs, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = &.{source} },
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

fn roundingAttrs(ctx: *mlir.Context, rounding: RoundingMode, flush_to_zero: bool) Attrs {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "rounding_mode", rounding.attribute(ctx)));
    if (flush_to_zero) attrs.appendAssumeCapacity(.named(ctx, "flush_to_zero", .unit(ctx)));
    return attrs;
}

fn overflowAttrs(ctx: *mlir.Context, overflow: IntegerOverflow) Attrs {
    var attrs: Attrs = .empty;
    if (overflow != .none) attrs.appendAssumeCapacity(.named(ctx, "overflow", overflow.attribute(ctx)));
    return attrs;
}

fn ftzAttrs(ctx: *mlir.Context, flush_to_zero: bool) Attrs {
    var attrs: Attrs = .empty;
    if (flush_to_zero) attrs.appendAssumeCapacity(.named(ctx, "flush_to_zero", .unit(ctx)));
    return attrs;
}

fn i1Like(ctx: *mlir.Context, v: *const mlir.Value) *const mlir.Type {
    const t = v.type_().isA(TileType) orelse std.debug.panic("cuda_tile: {f} is not a tile", .{v.type_()});
    var shape: stdx.BoundedArray(i64, mlir.ShapedType.MAX_RANK) = .empty;
    for (0..t.rank()) |i| shape.appendAssumeCapacity(t.dimension(i));
    return tileType(ctx, shape.constSlice(), .int(ctx, .i1));
}

pub fn absf(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "absf", source, .empty, location);
}

pub fn addf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "addf", lhs, rhs, roundingAttrs(ctx, rounding, flush_to_zero), location);
}

pub fn subf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "subf", lhs, rhs, roundingAttrs(ctx, rounding, flush_to_zero), location);
}

pub fn mulf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "mulf", lhs, rhs, roundingAttrs(ctx, rounding, flush_to_zero), location);
}

pub fn divf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "divf", lhs, rhs, roundingAttrs(ctx, rounding, flush_to_zero), location);
}

pub fn remf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "remf", lhs, rhs, .empty, location);
}

pub fn negf(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "negf", source, .empty, location);
}

pub fn fma(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, acc: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("fma"), .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .results = .{ .flat = &.{acc.type_()} },
        .attributes = roundingAttrs(ctx, rounding, flush_to_zero).constSlice(),
        .location = location,
    });
}

fn minMaxAttrs(ctx: *mlir.Context, propagate_nan: bool, flush_to_zero: bool) Attrs {
    var attrs: Attrs = .empty;
    if (propagate_nan) attrs.appendAssumeCapacity(.named(ctx, "propagate_nan", .unit(ctx)));
    if (flush_to_zero) attrs.appendAssumeCapacity(.named(ctx, "flush_to_zero", .unit(ctx)));
    return attrs;
}

pub fn maxf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, propagate_nan: bool, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "maxf", lhs, rhs, minMaxAttrs(ctx, propagate_nan, flush_to_zero), location);
}

pub fn minf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, propagate_nan: bool, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "minf", lhs, rhs, minMaxAttrs(ctx, propagate_nan, flush_to_zero), location);
}

pub fn pow(ctx: *mlir.Context, source: *const mlir.Value, exponent: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "pow", source, exponent, .empty, location);
}

pub fn atan2(ctx: *mlir.Context, x: *const mlir.Value, y: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "atan2", x, y, .empty, location);
}

/// `rounding` null keeps the op's default (13.3 attribute).
pub fn exp(ctx: *mlir.Context, source: *const mlir.Value, rounding: ?RoundingMode, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    if (rounding) |r| attrs.appendAssumeCapacity(.named(ctx, "rounding_mode", r.attribute(ctx)));
    return unary(ctx, "exp", source, attrs, location);
}

pub fn exp2(ctx: *mlir.Context, source: *const mlir.Value, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "exp2", source, ftzAttrs(ctx, flush_to_zero), location);
}

pub fn log(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "log", source, .empty, location);
}

pub fn log2(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "log2", source, .empty, location);
}

pub fn sqrt(ctx: *mlir.Context, source: *const mlir.Value, rounding: RoundingMode, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "sqrt", source, roundingAttrs(ctx, rounding, flush_to_zero), location);
}

pub fn rsqrt(ctx: *mlir.Context, source: *const mlir.Value, flush_to_zero: bool, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "rsqrt", source, ftzAttrs(ctx, flush_to_zero), location);
}

pub fn sin(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "sin", source, .empty, location);
}

pub fn cos(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "cos", source, .empty, location);
}

pub fn tan(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "tan", source, .empty, location);
}

pub fn sinh(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "sinh", source, .empty, location);
}

pub fn cosh(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "cosh", source, .empty, location);
}

/// `rounding` null keeps the op's default (13.2 attribute). Only `approx`
/// and `full` are legal; the upstream verifier reports the others without
/// failing, so they are refused here.
pub fn tanh(ctx: *mlir.Context, source: *const mlir.Value, rounding: ?RoundingMode, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    if (rounding) |r| {
        switch (r) {
            .approx, .full => {},
            else => std.debug.panic("cuda_tile.tanh: rounding mode {s} is not allowed (only approx or full)", .{@tagName(r)}),
        }
        attrs.appendAssumeCapacity(.named(ctx, "rounding_mode", r.attribute(ctx)));
    }
    return unary(ctx, "tanh", source, attrs, location);
}

pub fn ceil(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "ceil", source, .empty, location);
}

pub fn floor(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "floor", source, .empty, location);
}

pub fn cmpf(ctx: *mlir.Context, predicate: ComparisonPredicate, ordering: ComparisonOrdering, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("cmpf"), .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{i1Like(ctx, lhs)} },
        .attributes = &.{
            .named(ctx, "comparison_predicate", predicate.attribute(ctx)),
            .named(ctx, "comparison_ordering", ordering.attribute(ctx)),
        },
        .location = location,
    });
}

pub fn mmaf(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, acc: *const mlir.Value, fast_acc: bool, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    if (fast_acc) attrs.appendAssumeCapacity(.named(ctx, "fast_acc", .unit(ctx)));
    return mlir.Operation.make(ctx, opName("mmaf"), .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .results = .{ .flat = &.{acc.type_()} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// Block-scaled MMA (13.3): scales are plain tiles in logical M/N/K layout.
pub fn mmaf_scaled(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, acc: *const mlir.Value, lhs_scale: *const mlir.Value, rhs_scale: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("mmaf_scaled"), .{
        .operands = .{ .flat = &.{ lhs, rhs, acc, lhs_scale, rhs_scale } },
        .results = .{ .flat = &.{acc.type_()} },
        .location = location,
    });
}

pub fn absi(ctx: *mlir.Context, source: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "absi", source, .empty, location);
}

pub fn addi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "addi", lhs, rhs, overflowAttrs(ctx, overflow), location);
}

pub fn subi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "subi", lhs, rhs, overflowAttrs(ctx, overflow), location);
}

pub fn muli(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "muli", lhs, rhs, overflowAttrs(ctx, overflow), location);
}

pub fn mulhii(ctx: *mlir.Context, x: *const mlir.Value, y: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "mulhii", x, y, .empty, location);
}

/// `rounding` null keeps the op's default (truncation).
pub fn divi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, rounding: ?RoundingMode, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "signedness", signedness.attribute(ctx)));
    if (rounding) |r| attrs.appendAssumeCapacity(.named(ctx, "rounding", r.attribute(ctx)));
    return binary(ctx, "divi", lhs, rhs, attrs, location);
}

fn signedBinary(ctx: *mlir.Context, comptime mnemonic: []const u8, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, mnemonic, lhs, rhs, attrsOf(&.{.named(ctx, "signedness", signedness.attribute(ctx))}), location);
}

pub fn remi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, location: *const mlir.Location) *mlir.Operation {
    return signedBinary(ctx, "remi", lhs, rhs, signedness, location);
}

pub fn maxi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, location: *const mlir.Location) *mlir.Operation {
    return signedBinary(ctx, "maxi", lhs, rhs, signedness, location);
}

pub fn mini(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, location: *const mlir.Location) *mlir.Operation {
    return signedBinary(ctx, "mini", lhs, rhs, signedness, location);
}

pub fn negi(ctx: *mlir.Context, source: *const mlir.Value, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return unary(ctx, "negi", source, overflowAttrs(ctx, overflow), location);
}

pub fn shli(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "shli", lhs, rhs, overflowAttrs(ctx, overflow), location);
}

pub fn shri(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, signedness: Signedness, location: *const mlir.Location) *mlir.Operation {
    return signedBinary(ctx, "shri", lhs, rhs, signedness, location);
}

pub fn cmpi(ctx: *mlir.Context, predicate: ComparisonPredicate, signedness: Signedness, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("cmpi"), .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{i1Like(ctx, lhs)} },
        .attributes = &.{
            .named(ctx, "comparison_predicate", predicate.attribute(ctx)),
            .named(ctx, "signedness", signedness.attribute(ctx)),
        },
        .location = location,
    });
}

pub fn mmai(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, acc: *const mlir.Value, signedness_lhs: Signedness, signedness_rhs: Signedness, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("mmai"), .{
        .operands = .{ .flat = &.{ lhs, rhs, acc } },
        .results = .{ .flat = &.{acc.type_()} },
        .attributes = &.{
            .named(ctx, "signedness_lhs", signedness_lhs.attribute(ctx)),
            .named(ctx, "signedness_rhs", signedness_rhs.attribute(ctx)),
        },
        .location = location,
    });
}

pub fn andi(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "andi", lhs, rhs, .empty, location);
}

pub fn ori(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "ori", lhs, rhs, .empty, location);
}

pub fn xori(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "xori", lhs, rhs, .empty, location);
}

pub fn bitcast(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "bitcast", source, result_type, .empty, location);
}

pub fn exti(ctx: *mlir.Context, from: *const mlir.Value, signedness: Signedness, to: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "exti", from, to, attrsOf(&.{.named(ctx, "signedness", signedness.attribute(ctx))}), location);
}

pub fn trunci(ctx: *mlir.Context, from: *const mlir.Value, to: *const mlir.Type, overflow: IntegerOverflow, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "trunci", from, to, overflowAttrs(ctx, overflow), location);
}

/// `rounding` null keeps the op's default (nearest_even).
pub fn ftof(ctx: *mlir.Context, from: *const mlir.Value, to: *const mlir.Type, rounding: ?RoundingMode, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    if (rounding) |r| attrs.appendAssumeCapacity(.named(ctx, "rounding_mode", r.attribute(ctx)));
    return convert(ctx, "ftof", from, to, attrs, location);
}

pub fn ftoi(ctx: *mlir.Context, from: *const mlir.Value, signedness: Signedness, rounding: RoundingMode, to: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "ftoi", from, to, attrsOf(&.{
        .named(ctx, "signedness", signedness.attribute(ctx)),
        .named(ctx, "rounding_mode", rounding.attribute(ctx)),
    }), location);
}

pub fn itof(ctx: *mlir.Context, from: *const mlir.Value, signedness: Signedness, rounding: RoundingMode, to: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "itof", from, to, attrsOf(&.{
        .named(ctx, "signedness", signedness.attribute(ctx)),
        .named(ctx, "rounding_mode", rounding.attribute(ctx)),
    }), location);
}

pub fn int_to_ptr(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "int_to_ptr", source, result_type, .empty, location);
}

pub fn ptr_to_int(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "ptr_to_int", source, result_type, .empty, location);
}

pub fn ptr_to_ptr(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "ptr_to_ptr", source, result_type, .empty, location);
}

/// 13.3: sub-byte tile -> i8 tile.
pub fn pack(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "pack", source, result_type, .empty, location);
}

/// 13.3: i8 tile -> sub-byte tile.
pub fn unpack(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "unpack", source, result_type, .empty, location);
}

pub fn constant(ctx: *mlir.Context, value: *const mlir.Attribute, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("constant"), .{
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "value", value)},
        .location = location,
    });
}

pub fn iota(ctx: *mlir.Context, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("iota"), .{
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn reshape(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "reshape", source, result_type, .empty, location);
}

pub fn broadcast(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "broadcast", source, result_type, .empty, location);
}

pub fn cat(ctx: *mlir.Context, lhs: *const mlir.Value, rhs: *const mlir.Value, dim: i64, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("cat"), .{
        .operands = .{ .flat = &.{ lhs, rhs } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "dim", .int(ctx, .i64, dim))},
        .location = location,
    });
}

pub fn extract(ctx: *mlir.Context, source: *const mlir.Value, indices: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, mlir.ShapedType.MAX_RANK + 1) = .empty;
    operands.appendAssumeCapacity(source);
    operands.appendSliceAssumeCapacity(indices);
    return mlir.Operation.make(ctx, opName("extract"), .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

pub fn permute(ctx: *mlir.Context, source: *const mlir.Value, permutation: []const i32, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, "permute", source, result_type, attrsOf(&.{.named(ctx, "permutation", .denseArray(ctx, .i32, permutation))}), location);
}

pub fn select(ctx: *mlir.Context, cond: *const mlir.Value, val_if_true: *const mlir.Value, val_if_false: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("select"), .{
        .operands = .{ .flat = &.{ cond, val_if_true, val_if_false } },
        .results = .{ .flat = &.{val_if_true.type_()} },
        .location = location,
    });
}

pub fn offset(ctx: *mlir.Context, ptr: *const mlir.Value, off: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return binary(ctx, "offset", ptr, off, .empty, location);
}

fn threeI32(ctx: *mlir.Context, comptime mnemonic: []const u8, location: *const mlir.Location) *mlir.Operation {
    const i32_tile = tileType(ctx, &.{}, .int(ctx, .i32));
    return mlir.Operation.make(ctx, opName(mnemonic), .{
        .results = .{ .flat = &.{ i32_tile, i32_tile, i32_tile } },
        .location = location,
    });
}

pub fn get_tile_block_id(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return threeI32(ctx, "get_tile_block_id", location);
}

pub fn get_num_tile_blocks(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return threeI32(ctx, "get_num_tile_blocks", location);
}

pub const GlobalOpts = struct {
    alignment: ?i64 = null,
    constant: bool = false,
    visibility: ?SymbolVisibility = null,
};

pub fn global(ctx: *mlir.Context, name: []const u8, value: *const mlir.Attribute, opts: GlobalOpts, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "sym_name", .string(ctx, name)));
    attrs.appendAssumeCapacity(.named(ctx, "value", value));
    if (opts.alignment) |a| attrs.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, a)));
    if (opts.constant) attrs.appendAssumeCapacity(.named(ctx, "constant", .unit(ctx)));
    if (opts.visibility) |v| attrs.appendAssumeCapacity(.named(ctx, "symbol_visibility", v.attribute(ctx)));
    return mlir.Operation.make(ctx, opName("global"), .{
        .attributes = attrs.constSlice(),
        .verify = false,
        .location = location,
    });
}

pub fn get_global(ctx: *mlir.Context, name: []const u8, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("get_global"), .{
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "name", .flatSymbolRef(ctx, name))},
        // Symbol lookup needs the enclosing module; verified with it.
        .verify = false,
        .location = location,
    });
}

/// `cuda_tile.module @name { ... }`. `body` holds the entries and globals.
pub fn module(ctx: *mlir.Context, name: []const u8, body: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("module"), .{
        .attributes = &.{.named(ctx, "sym_name", .string(ctx, name))},
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

pub const EntryArgs = struct {
    name: []const u8,
    /// The entry block; its arguments are the kernel parameters, every one a
    /// rank-0 tile.
    block: *mlir.Block,
    arg_attrs: ?[]const *const mlir.Attribute = null,
    optimization_hints: ?*const mlir.Attribute = null,
    location: *const mlir.Location,
};

/// `entry @name(...) { ... }` — the kernel. Returns nothing; results reach
/// the host through pointer arguments.
pub fn entry(ctx: *mlir.Context, args: EntryArgs) *mlir.Operation {
    var arg_types: stdx.BoundedArray(*const mlir.Type, 256) = .empty;
    if (args.block.numArguments() > arg_types.capacity()) std.debug.panic("cuda_tile.entry: {d} parameters, at most {d} supported", .{ args.block.numArguments(), arg_types.capacity() });
    for (0..args.block.numArguments()) |i| arg_types.appendAssumeCapacity(args.block.argument(i).type_());

    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "sym_name", .string(ctx, args.name)));
    attrs.appendAssumeCapacity(.named(ctx, "function_type", .typeAttr(.function(ctx, arg_types.constSlice(), &.{}))));
    if (args.arg_attrs) |a| attrs.appendAssumeCapacity(.named(ctx, "arg_attrs", .array(ctx, a)));
    if (args.optimization_hints) |h| attrs.appendAssumeCapacity(.named(ctx, "optimization_hints", h));

    return mlir.Operation.make(ctx, opName("entry"), .{
        .attributes = attrs.constSlice(),
        .blocks = &.{args.block},
        .verify = false,
        .location = args.location,
    });
}

fn terminator(ctx: *mlir.Context, comptime mnemonic: []const u8, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = values },
        .verify = false,
        .location = location,
    });
}

pub fn return_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return terminator(ctx, "return", values, location);
}

pub fn yield(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return terminator(ctx, "yield", values, location);
}

pub fn continue_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return terminator(ctx, "continue", values, location);
}

pub fn break_(ctx: *mlir.Context, values: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return terminator(ctx, "break", values, location);
}

/// `for %iv in (%lb to %ub, step %step) iter_values(...)`. `body` takes
/// `[iv, inits...]` and ends with `continue`.
pub fn for_(ctx: *mlir.Context, lower: *const mlir.Value, upper: *const mlir.Value, step: *const mlir.Value, inits: []const *const mlir.Value, body: *mlir.Block, unsigned_cmp: bool, location: *const mlir.Location) *mlir.Operation {
    // Built on the state directly: any number of carried values, and outer
    // SSA uses only dominate once the op sits in its block, so no per-op
    // verify.
    var state: mlir.OperationState = .init(opName("for"), location);
    state.addOperands(&.{ lower, upper, step });
    state.addOperands(inits);
    for (inits) |v| state.addResults(&.{v.type_()});
    if (unsigned_cmp) state.addAttributes(&.{.named(ctx, "unsignedCmp", .unit(ctx))});
    const region = mlir.Region.init();
    region.appendOwnedBlock(body);
    state.addOwnedRegions(&.{region});
    return mlir.Operation.init(&state) catch @panic("Failed to create cuda_tile.for");
}

/// `if %cond -> (types) { ... } else { ... }`. Branches end with `yield`
/// (or `continue`/`break`/`return` when they leave the enclosing scope).
pub fn if_(ctx: *mlir.Context, cond: *const mlir.Value, result_types: []const *const mlir.Type, then_block: *mlir.Block, else_block: ?*mlir.Block, location: *const mlir.Location) *mlir.Operation {
    _ = ctx;
    var state: mlir.OperationState = .init(opName("if"), location);
    state.addOperands(&.{cond});
    state.addResults(result_types);
    const then_region = mlir.Region.init();
    then_region.appendOwnedBlock(then_block);
    const else_region = mlir.Region.init();
    if (else_block) |b| else_region.appendOwnedBlock(b);
    state.addOwnedRegions(&.{ then_region, else_region });
    return mlir.Operation.init(&state) catch @panic("Failed to create cuda_tile.if");
}

/// An unstructured loop over `inits`; the body takes them as arguments and
/// leaves through `continue` (next iteration) or `break` (the results).
pub fn loop(ctx: *mlir.Context, inits: []const *const mlir.Value, result_types: []const *const mlir.Type, body: *mlir.Block, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("loop"), .{
        .operands = .{ .flat = inits },
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .verify = false,
        .location = location,
    });
}

/// `reduce %operands dim=D identities=[...]` — `body` takes `2N` arguments
/// (element, accumulator per operand) and yields `N` accumulators. Results
/// drop dimension `dim`.
pub fn reduce(ctx: *mlir.Context, operands: []const *const mlir.Value, dim: i32, identities: []const *const mlir.Attribute, body: *mlir.Block, result_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("reduce"), .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .attributes = &.{
            .named(ctx, "dim", .int(ctx, .i32, dim)),
            .named(ctx, "identities", .array(ctx, identities)),
        },
        .verify = false,
        .location = location,
    });
}

/// `scan %operands dim=D reverse=R identities=[...]` — like `reduce`, but the
/// results keep the operand shape.
pub fn scan(ctx: *mlir.Context, operands: []const *const mlir.Value, dim: i32, reverse: bool, identities: []const *const mlir.Attribute, body: *mlir.Block, result_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("scan"), .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = result_types },
        .blocks = &.{body},
        .attributes = &.{
            .named(ctx, "dim", .int(ctx, .i32, dim)),
            .named(ctx, "reverse", .boolean(ctx, reverse)),
            .named(ctx, "identities", .array(ctx, identities)),
        },
        .verify = false,
        .location = location,
    });
}

pub fn assert_(ctx: *mlir.Context, condition: *const mlir.Value, message: []const u8, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("assert"), .{
        .operands = .{ .flat = &.{condition} },
        .attributes = &.{.named(ctx, "message", .string(ctx, message))},
        .location = location,
    });
}

/// `%r = assume #predicate, %value` — a refined copy of `value`.
pub fn assume(ctx: *mlir.Context, value: *const mlir.Value, predicate: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("assume"), .{
        .operands = .{ .flat = &.{value} },
        .results = .{ .flat = &.{value.type_()} },
        .attributes = &.{.named(ctx, "predicate", predicate)},
        .location = location,
    });
}

/// `make_tensor_view %base, shape = [...], strides = [...]`. Static extents
/// live in `result_type`; the operands fill its `?` slots in order.
pub fn make_tensor_view(ctx: *mlir.Context, base: *const mlir.Value, dynamic_shape: []const *const mlir.Value, dynamic_strides: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var segs: Segments = .{};
    segs.add(&.{base});
    segs.add(dynamic_shape);
    segs.add(dynamic_strides);
    return mlir.Operation.make(ctx, opName("make_tensor_view"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{segs.attr(ctx)},
        .location = location,
    });
}

fn viewFromTensorView(ctx: *mlir.Context, comptime mnemonic: []const u8, tensor_view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return convert(ctx, mnemonic, tensor_view, result_type, .empty, location);
}

/// The tile shape, padding and dim_map are all in `result_type`.
pub fn make_partition_view(ctx: *mlir.Context, tensor_view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return viewFromTensorView(ctx, "make_partition_view", tensor_view, result_type, location);
}

pub fn make_strided_view(ctx: *mlir.Context, tensor_view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return viewFromTensorView(ctx, "make_strided_view", tensor_view, result_type, location);
}

pub fn make_gather_scatter_view(ctx: *mlir.Context, tensor_view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return viewFromTensorView(ctx, "make_gather_scatter_view", tensor_view, result_type, location);
}

/// One rank-0 integer tile per view index dimension.
pub fn get_index_space_shape(ctx: *mlir.Context, view: *const mlir.Value, result_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("get_index_space_shape"), .{
        .operands = .{ .flat = &.{view} },
        .results = .{ .flat = result_types },
        .location = location,
    });
}

/// One rank-0 integer tile per tensor dimension.
pub fn get_tensor_shape(ctx: *mlir.Context, tensor_view: *const mlir.Value, result_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("get_tensor_shape"), .{
        .operands = .{ .flat = &.{tensor_view} },
        .results = .{ .flat = result_types },
        .location = location,
    });
}

pub const MemOpts = struct {
    ordering: MemoryOrdering = .weak,
    scope: ?MemoryScope = null,
    token: ?*const mlir.Value = null,
    optimization_hints: ?*const mlir.Attribute = null,
};

fn memAttrs(ctx: *mlir.Context, opts: MemOpts) Attrs {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "memory_ordering_semantics", opts.ordering.attribute(ctx)));
    if (opts.scope) |s| attrs.appendAssumeCapacity(.named(ctx, "memory_scope", s.attribute(ctx)));
    if (opts.optimization_hints) |h| attrs.appendAssumeCapacity(.named(ctx, "optimization_hints", h));
    return attrs;
}

/// An optional operand as a 0- or 1-element segment. Held in a local so the
/// slice handed to `Operation.make` stays valid.
const Seg = struct {
    buf: [1]*const mlir.Value = undefined,
    len: usize = 0,

    fn slice(self: *const Seg) []const *const mlir.Value {
        return self.buf[0..self.len];
    }
};

fn optional(v: ?*const mlir.Value) Seg {
    var s: Seg = .{};
    if (v) |x| {
        s.buf[0] = x;
        s.len = 1;
    }
    return s;
}

/// Operands of an `AttrSizedOperandSegments` op, flat plus the
/// `operandSegmentSizes` the op's properties expect (a DenseI32ArrayAttr).
const Segments = struct {
    values: stdx.BoundedArray(*const mlir.Value, 128) = .empty,
    sizes: stdx.BoundedArray(i32, 8) = .empty,

    fn add(self: *Segments, segment: []const *const mlir.Value) void {
        self.values.appendSliceAssumeCapacity(segment);
        self.sizes.appendAssumeCapacity(@intCast(segment.len));
    }

    fn attr(self: *const Segments, ctx: *mlir.Context) mlir.NamedAttribute {
        return .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, self.sizes.constSlice()));
    }
};

/// Two results: the tile (the view's tile type) and the token.
pub fn load_view_tko(ctx: *mlir.Context, view: *const mlir.Value, index: []const *const mlir.Value, opts: MemOpts, location: *const mlir.Location) *mlir.Operation {
    const tok = optional(opts.token);
    var segs: Segments = .{};
    segs.add(&.{view});
    segs.add(index);
    segs.add(tok.slice());
    var attrs = memAttrs(ctx, opts);
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("load_view_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{ tileTypeOfView(view.type_()), tokenType(ctx) } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// One result: the token.
pub fn store_view_tko(ctx: *mlir.Context, tile: *const mlir.Value, view: *const mlir.Value, index: []const *const mlir.Value, opts: MemOpts, location: *const mlir.Location) *mlir.Operation {
    const tok = optional(opts.token);
    var segs: Segments = .{};
    segs.add(&.{tile});
    segs.add(&.{view});
    segs.add(index);
    segs.add(tok.slice());
    var attrs = memAttrs(ctx, opts);
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("store_view_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{tokenType(ctx)} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub const LoadPtrOpts = struct {
    mask: ?*const mlir.Value = null,
    padding: ?*const mlir.Value = null,
    mem: MemOpts = .{},
};

/// Gather through a tile of pointers. Two results: the loaded tile and the
/// token. `result_type` is the pointee tile.
pub fn load_ptr_tko(ctx: *mlir.Context, source: *const mlir.Value, result_type: *const mlir.Type, opts: LoadPtrOpts, location: *const mlir.Location) *mlir.Operation {
    const mask = optional(opts.mask);
    const padding = optional(opts.padding);
    const tok = optional(opts.mem.token);
    var segs: Segments = .{};
    segs.add(&.{source});
    segs.add(mask.slice());
    segs.add(padding.slice());
    segs.add(tok.slice());
    var attrs = memAttrs(ctx, opts.mem);
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("load_ptr_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{ result_type, tokenType(ctx) } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub const StorePtrOpts = struct {
    mask: ?*const mlir.Value = null,
    mem: MemOpts = .{},
};

/// Scatter through a tile of pointers. One result: the token.
pub fn store_ptr_tko(ctx: *mlir.Context, destination: *const mlir.Value, value: *const mlir.Value, opts: StorePtrOpts, location: *const mlir.Location) *mlir.Operation {
    const mask = optional(opts.mask);
    const tok = optional(opts.mem.token);
    var segs: Segments = .{};
    segs.add(&.{destination});
    segs.add(&.{value});
    segs.add(mask.slice());
    segs.add(tok.slice());
    var attrs = memAttrs(ctx, opts.mem);
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("store_ptr_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{tokenType(ctx)} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub const AtomicOpts = struct {
    ordering: MemoryOrdering = .acq_rel,
    scope: MemoryScope = .device,
    mask: ?*const mlir.Value = null,
    token: ?*const mlir.Value = null,
};

fn atomicAttrs(ctx: *mlir.Context, opts: AtomicOpts) Attrs {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "memory_ordering_semantics", opts.ordering.attribute(ctx)));
    attrs.appendAssumeCapacity(.named(ctx, "memory_scope", opts.scope.attribute(ctx)));
    return attrs;
}

/// Two results: the old values and the token.
pub fn atomic_cas_tko(ctx: *mlir.Context, pointers: *const mlir.Value, cmp: *const mlir.Value, val: *const mlir.Value, opts: AtomicOpts, location: *const mlir.Location) *mlir.Operation {
    const mask = optional(opts.mask);
    const tok = optional(opts.token);
    var segs: Segments = .{};
    segs.add(&.{pointers});
    segs.add(&.{cmp});
    segs.add(&.{val});
    segs.add(mask.slice());
    segs.add(tok.slice());
    var attrs = atomicAttrs(ctx, opts);
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("atomic_cas_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{ val.type_(), tokenType(ctx) } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// Two results: the old values and the token.
pub fn atomic_rmw_tko(ctx: *mlir.Context, pointers: *const mlir.Value, mode: AtomicRMWMode, arg: *const mlir.Value, opts: AtomicOpts, location: *const mlir.Location) *mlir.Operation {
    const mask = optional(opts.mask);
    const tok = optional(opts.token);
    var segs: Segments = .{};
    segs.add(&.{pointers});
    segs.add(&.{arg});
    segs.add(mask.slice());
    segs.add(tok.slice());
    var attrs = atomicAttrs(ctx, opts);
    attrs.appendAssumeCapacity(.named(ctx, "mode", mode.attribute(ctx)));
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("atomic_rmw_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{ arg.type_(), tokenType(ctx) } },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// `atomic_red_view_tko` is relaxed-only, block- or device-scoped, and has
/// no mask operand.
pub const RedViewOpts = struct {
    scope: MemoryScope = .device,
    token: ?*const mlir.Value = null,
};

/// 13.3: an atomic reduction through an unpadded tiled view. One result:
/// the token. `xchg` is not a reduction.
pub fn atomic_red_view_tko(ctx: *mlir.Context, view: *const mlir.Value, index: []const *const mlir.Value, mode: AtomicRMWMode, value: *const mlir.Value, opts: RedViewOpts, location: *const mlir.Location) *mlir.Operation {
    if (mode == .xchg) @panic("cuda_tile.atomic_red_view_tko: xchg is not a reduction");
    if (opts.scope == .sys) @panic("cuda_tile.atomic_red_view_tko: only tl_blk and device scopes are allowed");
    const tok = optional(opts.token);
    var segs: Segments = .{};
    segs.add(&.{view});
    segs.add(index);
    segs.add(&.{value});
    segs.add(tok.slice());
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "memory_ordering_semantics", MemoryOrdering.relaxed.attribute(ctx)));
    attrs.appendAssumeCapacity(.named(ctx, "memory_scope", opts.scope.attribute(ctx)));
    attrs.appendAssumeCapacity(.named(ctx, "mode", mode.attribute(ctx)));
    attrs.appendAssumeCapacity(segs.attr(ctx));
    return mlir.Operation.make(ctx, opName("atomic_red_view_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{tokenType(ctx)} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

/// 13.3: `num_elem` elements of the pointee, as a rank-0 pointer tile.
pub fn alloca(ctx: *mlir.Context, num_elem: i64, alignment: i64, is_global: bool, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var attrs: Attrs = .empty;
    attrs.appendAssumeCapacity(.named(ctx, "num_elem", .int(ctx, .i64, num_elem)));
    attrs.appendAssumeCapacity(.named(ctx, "alignment", .int(ctx, .i64, alignment)));
    if (is_global) attrs.appendAssumeCapacity(.named(ctx, "global", .unit(ctx)));
    return mlir.Operation.make(ctx, opName("alloca"), .{
        .results = .{ .flat = &.{result_type} },
        .attributes = attrs.constSlice(),
        .location = location,
    });
}

pub fn make_token(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("make_token"), .{
        .results = .{ .flat = &.{tokenType(ctx)} },
        .location = location,
    });
}

pub fn join_tokens(ctx: *mlir.Context, tokens: []const *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, opName("join_tokens"), .{
        .operands = .{ .flat = tokens },
        .results = .{ .flat = &.{tokenType(ctx)} },
        .location = location,
    });
}

/// Device-side printf. One result: the token.
pub fn print_tko(ctx: *mlir.Context, str: []const u8, args: []const *const mlir.Value, token: ?*const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    const tok = optional(token);
    var segs: Segments = .{};
    segs.add(args);
    segs.add(tok.slice());
    return mlir.Operation.make(ctx, opName("print_tko"), .{
        .operands = .{ .flat = segs.values.constSlice() },
        .results = .{ .flat = &.{tokenType(ctx)} },
        .attributes = &.{ .named(ctx, "str", .string(ctx, str)), segs.attr(ctx) },
        .location = location,
    });
}

// =============================================================================
// Tests
// =============================================================================

test {
    std.testing.refAllDecls(@This());
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    mlir.DialectHandle.fromString(dialect_namespace).insertDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

test "dialect registers and types round-trip" {
    const ctx = try testContext();
    defer ctx.deinit();

    try std.testing.expect(ctx.isRegisteredOperation("cuda_tile.entry"));
    try std.testing.expect(ctx.isRegisteredOperation("cuda_tile.load_view_tko"));
    try std.testing.expect(ctx.isRegisteredOperation("cuda_tile.mmaf"));

    const f32_ty = mlir.Type.float(ctx, .f32);
    const tile = tileType(ctx, &.{ 64, 64 }, f32_ty);
    try std.testing.expect(tile.eql(try mlir.Type.parse(ctx, "!cuda_tile.tile<64x64xf32>")));
    try std.testing.expectEqual(@as(usize, 2), tile.isA(TileType).?.rank());
    try std.testing.expectEqual(@as(i64, 64), tile.isA(TileType).?.dimension(1));

    const ptr = pointerType(ctx, f32_ty);
    try std.testing.expect(ptr.eql(try mlir.Type.parse(ctx, "!cuda_tile.ptr<f32>")));
    try std.testing.expect(ptr.isA(PointerType).?.pointee().eql(f32_ty));

    const dyn = TensorViewType.dynamic();
    const tv = tensorViewType(ctx, f32_ty, &.{ dyn, 32 }, &.{ dyn, 1 });
    try std.testing.expect(tv.eql(try mlir.Type.parse(ctx, "!cuda_tile.tensor_view<?x32xf32, strides=[?,1]>")));

    const static_tv = tensorViewType(ctx, f32_ty, &.{16}, &.{1});
    const pv: *const mlir.Type = @ptrCast(PartitionViewType.get(ctx, &.{2}, static_tv, &.{}, .zero));
    try std.testing.expect(pv.eql(try mlir.Type.parse(ctx, "!cuda_tile.partition_view<tile=(2), padding_value = zero, tensor_view<16xf32, strides=[1]>>")));
    try std.testing.expect(tileTypeOfView(pv).eql(tileType(ctx, &.{2}, f32_ty)));
    try std.testing.expectEqual(@as(usize, 1), indexRankOfView(pv));

    try std.testing.expect(tokenType(ctx).eql(try mlir.Type.parse(ctx, "!cuda_tile.token")));
}

test "enum and hint attributes" {
    const ctx = try testContext();
    defer ctx.deinit();

    try std.testing.expect(RoundingMode.nearest_even.attribute(ctx).eql(try mlir.Attribute.parse(ctx, "#cuda_tile.rounding<nearest_even>")));
    const padding = PaddingValue.neg_inf.attribute(ctx);
    const padding_text = c.mlirCudaTilePaddingValueAttrGetValue(padding.ptr());
    try std.testing.expectEqualStrings("neg_inf", padding_text.data[0..padding_text.length]);

    const hints = optimizationHints(ctx, &[_]EntryHint{
        .{ .arch = .sm_100, .num_cta_in_cga = 2 },
        .{ .arch = .sm_120, .num_cta_in_cga = 2, .occupancy = 2 },
    });
    try std.testing.expect(hints.eql(try mlir.Attribute.parse(ctx, "#cuda_tile.optimization_hints<sm_100 = {num_cta_in_cga = 2}, sm_120 = {num_cta_in_cga = 2, occupancy = 2}>")));

    _ = divBy(ctx, 16, null, null);
    _ = sameElements(ctx, &.{ 2, 4 });
    _ = bounded(ctx, 0, null);
}

test "entry round-trip through print and parse" {
    const ctx = try testContext();
    defer ctx.deinit();

    const loc: *const mlir.Location = .unknown(ctx);
    const module_op: *mlir.Module = .init(loc);
    defer module_op.deinit();

    const f32_ty = mlir.Type.float(ctx, .f32);
    const ptr_tile = tileType(ctx, &.{}, pointerType(ctx, f32_ty));
    const block = mlir.Block.init(&.{ ptr_tile, ptr_tile }, &.{ loc, loc });

    // out[i] = in[i] + 1 over 128 elements, the shape of cuda-tile's README example.
    const idx_ty = tileType(ctx, &.{128}, .int(ctx, .i32));
    const offsets = iota(ctx, idx_ty, loc).appendTo(block);
    const ptr1 = tileType(ctx, &.{1}, pointerType(ctx, f32_ty));
    const ptr128 = tileType(ctx, &.{128}, pointerType(ctx, f32_ty));
    const in_r = reshape(ctx, block.argument(0), ptr1, loc).appendTo(block);
    const in_b = broadcast(ctx, in_r.result(0), ptr128, loc).appendTo(block);
    const in_p = offset(ctx, in_b.result(0), offsets.result(0), loc).appendTo(block);
    const f32_128 = tileType(ctx, &.{128}, f32_ty);
    const loaded = load_ptr_tko(ctx, in_p.result(0), f32_128, .{}, loc).appendTo(block);
    const one = constant(ctx, splatAttr(f32_128, floatElem(ctx, f32_ty, 1.0)), f32_128, loc).appendTo(block);
    const sum = addf(ctx, loaded.result(0), one.result(0), .nearest_even, false, loc).appendTo(block);
    const out_r = reshape(ctx, block.argument(1), ptr1, loc).appendTo(block);
    const out_b = broadcast(ctx, out_r.result(0), ptr128, loc).appendTo(block);
    const out_p = offset(ctx, out_b.result(0), offsets.result(0), loc).appendTo(block);
    _ = store_ptr_tko(ctx, out_p.result(0), sum.result(0), .{ .mem = .{ .token = loaded.result(1) } }, loc).appendTo(block);
    _ = return_(ctx, &.{}, loc).appendTo(block);

    const ct_body = mlir.Block.init(&.{}, &.{});
    _ = entry(ctx, .{ .name = "add_one", .block = block, .location = loc }).appendTo(ct_body);
    _ = module(ctx, "m", ct_body, loc).appendTo(module_op.body());

    try std.testing.expect(module_op.operation().verify());

    var al: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer al.deinit();
    try al.writer.print("{f}", .{module_op.operation()});
    const text = al.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "cuda_tile.module @m") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "entry @add_one(") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "load_ptr_tko weak") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "addf") != null);

    const parsed = try mlir.Module.parse(ctx, text);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "views, control flow and mma verify" {
    const ctx = try testContext();
    defer ctx.deinit();

    const loc: *const mlir.Location = .unknown(ctx);
    const module_op: *mlir.Module = .init(loc);
    defer module_op.deinit();

    const bf16_ty = mlir.Type.float(ctx, .bf16);
    const f32_ty = mlir.Type.float(ctx, .f32);
    const i32_ty = mlir.Type.int(ctx, .i32);
    const bf16_ptr = tileType(ctx, &.{}, pointerType(ctx, bf16_ty));
    const f32_ptr = tileType(ctx, &.{}, pointerType(ctx, f32_ty));
    const block = mlir.Block.init(&.{ bf16_ptr, bf16_ptr, f32_ptr }, &.{ loc, loc, loc });

    // C[128x128] = A[128x64] @ B[64x128] in 64-wide K tiles.
    const tv_a = make_tensor_view(ctx, block.argument(0), &.{}, &.{}, tensorViewType(ctx, bf16_ty, &.{ 128, 64 }, &.{ 64, 1 }), loc).appendTo(block);
    const tv_b = make_tensor_view(ctx, block.argument(1), &.{}, &.{}, tensorViewType(ctx, bf16_ty, &.{ 64, 128 }, &.{ 128, 1 }), loc).appendTo(block);
    const tv_c = make_tensor_view(ctx, block.argument(2), &.{}, &.{}, tensorViewType(ctx, f32_ty, &.{ 128, 128 }, &.{ 128, 1 }), loc).appendTo(block);
    const pv_a: *const mlir.Type = @ptrCast(PartitionViewType.get(ctx, &.{ 64, 64 }, tv_a.result(0).type_(), &.{}, .zero));
    const pv_b: *const mlir.Type = @ptrCast(PartitionViewType.get(ctx, &.{ 64, 64 }, tv_b.result(0).type_(), &.{}, .zero));
    const pv_c: *const mlir.Type = @ptrCast(PartitionViewType.get(ctx, &.{ 64, 64 }, tv_c.result(0).type_(), &.{}, null));
    const va = make_partition_view(ctx, tv_a.result(0), pv_a, loc).appendTo(block);
    const vb = make_partition_view(ctx, tv_b.result(0), pv_b, loc).appendTo(block);
    const vc = make_partition_view(ctx, tv_c.result(0), pv_c, loc).appendTo(block);

    const bid = get_tile_block_id(ctx, loc).appendTo(block);
    const i32_tile = tileType(ctx, &.{}, i32_ty);
    const zero = constant(ctx, splatAttr(i32_tile, intElem(i32_ty, 0)), i32_tile, loc).appendTo(block);
    const one = constant(ctx, splatAttr(i32_tile, intElem(i32_ty, 1)), i32_tile, loc).appendTo(block);
    const acc_ty = tileType(ctx, &.{ 64, 64 }, f32_ty);
    const acc0 = constant(ctx, splatAttr(acc_ty, floatElem(ctx, f32_ty, 0.0)), acc_ty, loc).appendTo(block);

    const body = mlir.Block.init(&.{ i32_tile, acc_ty }, &.{ loc, loc });
    const at = load_view_tko(ctx, va.result(0), &.{ bid.result(0), body.argument(0) }, .{}, loc).appendTo(body);
    const bt = load_view_tko(ctx, vb.result(0), &.{ body.argument(0), bid.result(1) }, .{}, loc).appendTo(body);
    const acc = mmaf(ctx, at.result(0), bt.result(0), body.argument(1), false, loc).appendTo(body);
    _ = continue_(ctx, &.{acc.result(0)}, loc).appendTo(body);
    const loop_op = for_(ctx, zero.result(0), one.result(0), one.result(0), &.{acc0.result(0)}, body, false, loc).appendTo(block);

    // if %cond -> (tile<f32>) { yield } else { yield }
    const cond = cmpi(ctx, .equal, .signed, bid.result(0), zero.result(0), loc).appendTo(block);
    const then_block = mlir.Block.init(&.{}, &.{});
    _ = yield(ctx, &.{loop_op.result(0)}, loc).appendTo(then_block);
    const else_block = mlir.Block.init(&.{}, &.{});
    _ = yield(ctx, &.{acc0.result(0)}, loc).appendTo(else_block);
    const if_op = if_(ctx, cond.result(0), &.{acc_ty}, then_block, else_block, loc).appendTo(block);

    _ = store_view_tko(ctx, if_op.result(0), vc.result(0), &.{ bid.result(0), bid.result(1) }, .{}, loc).appendTo(block);
    _ = return_(ctx, &.{}, loc).appendTo(block);

    const ct_body = mlir.Block.init(&.{}, &.{});
    _ = entry(ctx, .{
        .name = "gemm",
        .block = block,
        .optimization_hints = optimizationHints(ctx, &[_]EntryHint{.{ .arch = .sm_120, .num_worker_warps_per_cta = 4 }}),
        .location = loc,
    }).appendTo(ct_body);
    _ = module(ctx, "m", ct_body, loc).appendTo(module_op.body());

    try std.testing.expect(module_op.operation().verify());

    var al: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer al.deinit();
    try al.writer.print("{f}", .{module_op.operation()});
    const text = al.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "optimization_hints=<sm_120 = {num_worker_warps_per_cta = 4}>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "load_view_tko weak") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "continue ") != null);

    const parsed = try mlir.Module.parse(ctx, text);
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "the verifier rejects what the dialect forbids" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc: *const mlir.Location = .unknown(ctx);
    const f32_ty = mlir.Type.float(ctx, .f32);

    // An entry with a rank-1 argument. (Per-op negatives go through
    // `Operation.try_make`, whose logged error the test runner counts as a
    // failure, so the module-level verifier is the oracle here.)
    const module_op: *mlir.Module = .init(loc);
    defer module_op.deinit();
    const block = mlir.Block.init(&.{tileType(ctx, &.{4}, f32_ty)}, &.{loc});
    _ = return_(ctx, &.{}, loc).appendTo(block);
    const ct_body = mlir.Block.init(&.{}, &.{});
    _ = entry(ctx, .{ .name = "bad", .block = block, .location = loc }).appendTo(ct_body);
    _ = module(ctx, "m", ct_body, loc).appendTo(module_op.body());
    try std.testing.expect(!module_op.operation().verify());
}
