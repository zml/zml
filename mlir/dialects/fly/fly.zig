const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

pub const attributes = @import("attributes.zig");
pub const ops = @import("ops.zig");
pub const types = @import("types.zig");

pub const AddressSpace = attributes.AddressSpace;
pub const GemmTraversalOrder = attributes.GemmTraversalOrder;
pub const MmaOperand = attributes.MmaOperand;

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("fly");
}

/// `Operation.make` that names the op and its operand types on failure: a
/// rejected `inferReturnTypes` otherwise panics with nothing to go on.
pub fn make(ctx: *mlir.Context, name: []const u8, args: mlir.Operation.MakeArgs) *mlir.Operation {
    return mlir.Operation.try_make(ctx, name, args) catch |err| {
        var buf: [4096]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        if (args.operands) |operands| switch (operands) {
            .flat => |vs| for (vs, 0..) |v, i| {
                w.print("\n  operand {d}: {f}", .{ i, v.type_() }) catch break;
            },
            .variadic => |segments| for (segments) |seg| for (seg) |v| {
                w.print("\n  operand: {f}", .{v.type_()}) catch break;
            },
        };
        if (args.results) |results| switch (results) {
            .flat => |ts| for (ts, 0..) |t, i| {
                w.print("\n  result {d}: {f}", .{ i, t }) catch break;
            },
            .variadic => {},
        };
        std.debug.panic("{s}: cannot create operation ({}){s}", .{ name, err, w.buffered() });
    };
}

pub const TypeKind = enum {
    memref,
    coord_tensor,
    ptr,
    int_tuple,
    layout,
    composed_layout,
    tile,
    swizzle,
    copy_atom,
    mma_atom,
    tiled_copy,
    tiled_mma,
    other,
};

pub fn KindType(comptime kind: TypeKind) type {
    return switch (kind) {
        .memref => types.MemRefType,
        .coord_tensor => types.CoordTensorType,
        .ptr => types.PointerType,
        .int_tuple => types.IntTupleType,
        .layout => types.LayoutType,
        .composed_layout => types.ComposedLayoutType,
        .tile => types.TileType,
        .swizzle => types.SwizzleType,
        .copy_atom => types.CopyAtomType,
        .mma_atom => types.MmaAtomType,
        .tiled_copy => types.TiledCopyType,
        .tiled_mma => types.TiledMmaType,
        .other => @compileError("fly.KindType: `.other` is not a fly type"),
    };
}

pub fn typeKind(ty: *const mlir.Type) TypeKind {
    inline for (comptime std.meta.tags(TypeKind)) |kind| {
        if (kind != .other and ty.isA(KindType(kind)) != null) return kind;
    }
    return .other;
}

/// Panics if `ty` is not a `!fly.<kind>`.
pub fn expect(ty: *const mlir.Type, comptime kind: TypeKind) *const KindType(kind) {
    return ty.isA(KindType(kind)) orelse
        std.debug.panic("fly: `{f}` is not a !fly.{s}", .{ ty, @tagName(kind) });
}

/// The shape of the outermost plain layout of any layout-carrying type.
pub fn layoutLikeShape(ty: *const mlir.Type) *const types.IntTupleType {
    if (ty.isA(types.LayoutType) == null and ty.isA(types.ComposedLayoutType) == null and
        ty.isA(types.MemRefType) == null and ty.isA(types.CoordTensorType) == null)
    {
        std.debug.panic("fly: `{f}` carries no layout", .{ty});
    }
    return @ptrCast(c.mlirFlyLayoutLikeTypeGetShape(ty.ptr()).ptr.?);
}

pub fn elemType(ty: *const mlir.Type) *const mlir.Type {
    if (ty.isA(types.MemRefType)) |m| return m.elemTy();
    if (ty.isA(types.PointerType)) |p| return p.elemTy();
    std.debug.panic("fly: `{f}` has no element type", .{ty});
}

pub fn addressSpace(ty: *const mlir.Type) *const mlir.Attribute {
    if (ty.isA(types.MemRefType)) |m| return m.addressSpace();
    if (ty.isA(types.PointerType)) |p| return p.addressSpace();
    std.debug.panic("fly: `{f}` has no address space", .{ty});
}

/// Keeps the address space, alignment and swizzle.
pub fn ptrWithElem(ctx: *mlir.Context, ptr: *const mlir.Type, elem: *const mlir.Type) mlir.Error!*const types.PointerType {
    const p = expect(ptr, .ptr);
    return types.PointerType.get(ctx, .{
        .elemTy = elem,
        .addressSpace = p.addressSpace(),
        .alignment = p.alignment(),
        .swizzle = p.swizzle(),
    });
}
