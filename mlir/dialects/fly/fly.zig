const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const attributes = @import("attributes.zig");
pub const rocdl = @import("rocdl.zig");
pub const types = @import("types.zig");

pub const AddressSpace = attributes.AddressSpace;
pub const GemmTraversalOrder = attributes.GemmTraversalOrder;
pub const MmaOperand = attributes.MmaOperand;

/// `gpu`/`rocdl` for `gpu.func` kernels and the ROCDL escape hatches; `ub`
/// because fly ops may fold to `ub.poison`.
pub const dialects_needed = [_][]const u8{ "func", "arith", "scf", "math", "vector", "memref", "llvm", "gpu", "rocdl", "ub", "fly", "fly_rocdl" };

pub fn insertDialects(registry: *mlir.DialectRegistry) void {
    inline for (dialects_needed) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }
}

pub fn opName(comptime mnemonic: []const u8) []const u8 {
    return "fly." ++ mnemonic;
}

/// By value, so a helper's temporary outlives the slice `Operation.make` gets.
pub const Attrs = stdx.BoundedArray(mlir.NamedAttribute, 8);

pub fn attrs(list: []const mlir.NamedAttribute) Attrs {
    var out: Attrs = .empty;
    out.appendSliceAssumeCapacity(list);
    return out;
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

/// Result type inferred by FlyDSL.
pub fn inferred(ctx: *mlir.Context, comptime mnemonic: []const u8, operands: []const *const mlir.Value, attributes_: Attrs, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = operands },
        .result_type_inference = true,
        .attributes = attributes_.constSlice(),
        .location = location,
    });
}

/// Explicit result types.
pub fn typed(ctx: *mlir.Context, comptime mnemonic: []const u8, operands: []const *const mlir.Value, results: []const *const mlir.Type, attributes_: Attrs, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = results },
        .attributes = attributes_.constSlice(),
        .location = location,
    });
}

/// No results.
pub fn effect(ctx: *mlir.Context, comptime mnemonic: []const u8, operands: []const *const mlir.Value, attributes_: Attrs, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = operands },
        .attributes = attributes_.constSlice(),
        .location = location,
    });
}

/// Panics with the text on failure.
pub fn parseType(ctx: *mlir.Context, text: []const u8) *const mlir.Type {
    return mlir.Type.parse(ctx, text) catch std.debug.panic("fly: cannot parse type `{s}`", .{text});
}

pub fn parseAttr(ctx: *mlir.Context, text: []const u8) *const mlir.Attribute {
    return mlir.Attribute.parse(ctx, text) catch std.debug.panic("fly: cannot parse attribute `{s}`", .{text});
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
    if (ty.isA(types.MemRefType)) |m| return m.getElemTy();
    if (ty.isA(types.PointerType)) |p| return p.getElemTy();
    std.debug.panic("fly: `{f}` has no element type", .{ty});
}

pub fn addressSpace(ty: *const mlir.Type) *const mlir.Attribute {
    if (ty.isA(types.MemRefType)) |m| return m.getAddressSpace();
    if (ty.isA(types.PointerType)) |p| return p.getAddressSpace();
    std.debug.panic("fly: `{f}` has no address space", .{ty});
}

/// Keeps the address space, alignment and swizzle.
pub fn ptrWithElem(ctx: *mlir.Context, ptr: *const mlir.Type, elem: *const mlir.Type) mlir.Error!*const types.PointerType {
    const p = expect(ptr, .ptr);
    return types.PointerType.get(ctx, .{
        .elemTy = elem,
        .addressSpace = p.getAddressSpace(),
        .alignment = p.getAlignment(),
        .swizzle = p.getSwizzle(),
    });
}

/// A value whose whole content lives in its type.
pub fn static(ctx: *mlir.Context, ty: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return typed(ctx, "static", &.{}, &.{ty}, .empty, location);
}

/// `ty` names the tuple with a `?` per dynamic operand, in order.
pub fn makeIntTuple(ctx: *mlir.Context, dynamic: []const *const mlir.Value, ty: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return typed(ctx, "make_int_tuple", dynamic, &.{ty}, .empty, location);
}

pub fn makeCopyAtom(ctx: *mlir.Context, ty: *const mlir.Type, val_bits: i32, location: *const mlir.Location) *mlir.Operation {
    return typed(ctx, "make_copy_atom", &.{}, &.{ty}, attrs(&.{.named(ctx, "valBits", .int(ctx, .i32, val_bits))}), location);
}

pub fn makeMmaAtom(ctx: *mlir.Context, ty: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return typed(ctx, "make_mma_atom", &.{}, &.{ty}, .empty, location);
}

pub fn makeFragmentLike(ctx: *mlir.Context, src: *const mlir.Value, dtype: ?*const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    var a: Attrs = .empty;
    if (dtype) |t| a.appendAssumeCapacity(.named(ctx, "dtype", .typeAttr(t)));
    return inferred(ctx, "make_fragment_like", &.{src}, a, location);
}

pub const CopyOpts = struct {
    pred: ?*const mlir.Value = null,
};

pub fn copy(ctx: *mlir.Context, atom: *const mlir.Value, src: *const mlir.Value, dst: *const mlir.Value, opts: CopyOpts, location: *const mlir.Location) *mlir.Operation {
    if (opts.pred) |p| return effect(ctx, "copy", &.{ atom, src, dst, p }, .empty, location);
    return effect(ctx, "copy", &.{ atom, src, dst }, .empty, location);
}

pub const GemmOpts = struct {
    traversal_order: ?GemmTraversalOrder = null,
    traversal_layout: ?*const mlir.Value = null,
};

pub fn gemm(ctx: *mlir.Context, atom: *const mlir.Value, d: *const mlir.Value, a: *const mlir.Value, b: *const mlir.Value, cc: *const mlir.Value, opts: GemmOpts, location: *const mlir.Location) *mlir.Operation {
    var at: Attrs = .empty;
    if (opts.traversal_order) |o| {
        const attr = attributes.gemmTraversalOrderAttr(ctx, o) catch @panic("fly.gemm: invalid traversal order");
        at.appendAssumeCapacity(.named(ctx, "traversalOrder", attr));
    }
    if (opts.traversal_layout) |l| return effect(ctx, "gemm", &.{ atom, d, a, b, cc, l }, at, location);
    return effect(ctx, "gemm", &.{ atom, d, a, b, cc }, at, location);
}

pub fn get(ctx: *mlir.Context, input: *const mlir.Value, mode: []const i32, location: *const mlir.Location) *mlir.Operation {
    return inferred(ctx, "get", &.{input}, attrs(&.{.named(ctx, "mode", .denseArray(ctx, .i32, mode))}), location);
}

pub fn select(ctx: *mlir.Context, input: *const mlir.Value, indices: []const i32, location: *const mlir.Location) *mlir.Operation {
    return inferred(ctx, "select", &.{input}, attrs(&.{.named(ctx, "indices", .denseArray(ctx, .i32, indices))}), location);
}

pub fn takeOrGroup(ctx: *mlir.Context, comptime mnemonic: []const u8, input: *const mlir.Value, begin: i32, end: i32, location: *const mlir.Location) *mlir.Operation {
    return inferred(ctx, mnemonic, &.{input}, attrs(&.{
        .named(ctx, "begin", .int(ctx, .i32, begin)),
        .named(ctx, "end", .int(ctx, .i32, end)),
    }), location);
}

/// Explicit pointer type; `fly.make_ptr` infers nothing.
pub fn makePtr(ctx: *mlir.Context, operands: []const *const mlir.Value, ty: *const mlir.Type, dict_attrs: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var a: Attrs = .empty;
    if (dict_attrs) |d| a.appendAssumeCapacity(.named(ctx, "dictAttrs", d));
    return typed(ctx, "make_ptr", operands, &.{ty}, a, location);
}

test {
    std.testing.refAllDecls(@This());
    inline for (.{ types, attributes, rocdl }) |module| {
        std.testing.refAllDecls(module);
        inline for (comptime std.meta.declarations(module)) |decl| {
            const value = @field(module, decl.name);
            if (@TypeOf(value) == type and @typeInfo(value) == .@"opaque") std.testing.refAllDecls(value);
        }
    }
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    insertDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectPrints(expected: []const u8, value: anytype) !void {
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{value});
    try std.testing.expectEqualStrings(expected, w.buffered());
}

test "fly dialects register" {
    const ctx = try testContext();
    defer ctx.deinit();

    try std.testing.expect(ctx.isRegisteredOperation("fly.make_int_tuple"));
    try std.testing.expect(ctx.isRegisteredOperation("fly.make_layout"));
    try std.testing.expect(ctx.isRegisteredOperation("fly.ptr.load"));
    try std.testing.expect(ctx.isRegisteredOperation("fly.copy_atom_call"));
    try std.testing.expect(ctx.isRegisteredOperation("gpu.func"));
    try std.testing.expect(ctx.isRegisteredOperation("rocdl.sched.barrier"));
    try std.testing.expect(ctx.isRegisteredOperation("ub.poison"));
}

test "fly types parse and print round-trip" {
    const ctx = try testContext();
    defer ctx.deinit();

    const cases = [_][]const u8{
        "!fly.int_tuple<((2, 4), 8)>",
        "!fly.int_tuple<((1, 4), (32, 64))>",
        "!fly.layout<((2, 3), 3) : ((57, 69), 19)>",
        "!fly.memref<f16, global, (128, 64) : (64, 1)>",
        "!fly.memref<bf16, register, 16 : 1>",
        "!fly.ptr<f16, global>",
        "!fly.ptr<bf16, shared>",
        "!fly.ptr<f16, #fly_rocdl.buffer_desc>",
        "!fly.memref<f16, #fly_rocdl.buffer_desc, 4 : 1>",
        "!fly_rocdl.cdna3.buffer_copy<128>",
        "!fly_rocdl.cdna3.buffer_copy_lds<32>",
        "!fly_rocdl.cdna3.mfma<16x16x16, (f16, f16) -> f32>",
        "!fly.copy_atom<!fly.universal_copy<128>, 32>",
        "!fly.tile<[8|64]>",
        "!fly.coord_tensor<(0,0), (100,1000):(1E0,1E1)>",
    };
    for (cases) |src| {
        const ty = mlir.Type.parse(ctx, src) catch |err| {
            std.debug.print("failed to parse `{s}`: {}\n", .{ src, err });
            return err;
        };
        var buf: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        try w.print("{f}", .{ty});
        const ty2 = try mlir.Type.parse(ctx, w.buffered());
        try std.testing.expect(ty.eql(ty2));
    }
}

test "types and attributes are built through the C API" {
    const ctx = try testContext();
    defer ctx.deinit();

    const two = try attributes.IntTupleAttr.getStatic(ctx, 2);
    const four = try attributes.IntTupleAttr.getStatic(ctx, 4);
    const one = try attributes.IntTupleAttr.getStatic(ctx, 1);
    const shape = try attributes.IntTupleAttr.getTuple(ctx, &.{ two, four });
    const stride = try attributes.IntTupleAttr.getTuple(ctx, &.{ four, one });
    const layout = try types.LayoutType.get(ctx, .{ .attr = try .get(ctx, .{ .shape = shape, .stride = stride }) });
    try expectPrints("!fly.layout<(2,4):(4,1)>", layout);
    try expectPrints("!fly.int_tuple<(2,4)>", layout.getShape());
    try std.testing.expectEqual(@as(usize, 2), layout.getShape().getNumElements());
    try std.testing.expectEqual(types.IntTupleType.Leaf{ .static = 4 }, layout.getShape().getElement(1).getLeaf());

    const global = try attributes.AddressSpaceAttr.get(ctx, .{ .addressSpace = .global });
    const memref = try types.MemRefType.get(ctx, .{
        .elemTy = .float(ctx, .f16),
        .addressSpace = global.attribute(),
        .layout = layout.getAttr().attribute(),
    });
    try expectPrints("!fly.memref<f16, global, (2,4):(4,1)>", memref);
    try expectPrints("!fly.int_tuple<(2,4)>", layoutLikeShape(memref.type_()));

    const shared = try attributes.AddressSpaceAttr.get(ctx, .{ .addressSpace = .shared });
    const ptr = try types.PointerType.get(ctx, .{
        .elemTy = .int(ctx, .i8),
        .addressSpace = shared.attribute(),
        .alignment = try .get(ctx, .{ .alignment = 16 }),
    });
    try expectPrints("!fly.ptr<i8, shared, align<16>>", ptr);
    try expectPrints("!fly.ptr<f32, shared, align<16>>", try ptrWithElem(ctx, ptr.type_(), .float(ctx, .f32)));

    const tile = try types.TileType.get(ctx, .{ .attr = try .getModes(ctx, &.{
        (try attributes.IntAttr.getStatic(ctx, 8)).attribute(),
        (try attributes.IntAttr.getStatic(ctx, 64)).attribute(),
    }) });
    try expectPrints("!fly.tile<[8|64]>", tile);

    const copy_atom = try types.CopyAtomType.get(ctx, .{
        .copyOp = (try types.CopyOpUniversalCopyType.get(ctx, .{ .bitSize = 128 })).type_(),
        .valBits = 32,
    });
    try expectPrints("!fly.copy_atom<!fly.universal_copy<128>, 32>", copy_atom);

    const buffer_desc = try rocdl.BufferDescAddressAttr.get(ctx);
    try expectPrints("!fly.ptr<f16, #fly_rocdl.buffer_desc>", try types.PointerType.get(ctx, .{
        .elemTy = .float(ctx, .f16),
        .addressSpace = buffer_desc.attribute(),
    }));
}

test "fly enum and gpu attributes" {
    const ctx = try testContext();
    defer ctx.deinit();
    _ = try attributes.mmaOperandAttr(ctx, .a);
    _ = try attributes.gemmTraversalOrderAttr(ctx, .kmn);
    _ = parseAttr(ctx, "#gpu<dim x>");
}

test "inferred fly ops compute layout algebra" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc: *const mlir.Location = .unknown(ctx);
    const module: *mlir.Module = .init(loc);
    defer module.deinit();
    const block = module.body();

    // raked_product((8,16):(16,1), (1,4):(1,1)) — the vectorAdd TV layout.
    const thr = static(ctx, parseType(ctx, "!fly.layout<(8,16):(16,1)>"), loc).appendTo(block);
    const val = static(ctx, parseType(ctx, "!fly.layout<(1,4):(1,1)>"), loc).appendTo(block);
    const mn = inferred(ctx, "raked_product", &.{ thr.result(0), val.result(0) }, .empty, loc).appendTo(block);
    const shape = inferred(ctx, "get_shape", &.{mn.result(0)}, .empty, loc).appendTo(block);
    const tiler = inferred(ctx, "int_tuple_product_each", &.{shape.result(0)}, .empty, loc).appendTo(block);

    try expectPrints("!fly.int_tuple<(8,64)>", tiler.result(0).type_());
    try std.testing.expect(module.operation().verify());
}

test "atom traits" {
    const ctx = try testContext();
    defer ctx.deinit();

    // MFMA 16x16x4 f32 tiled over a (2,2,1) atom layout.
    const tm = expect(parseType(ctx, "!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>, <(2,2,1):(1,2,0)>>"), .tiled_mma);
    try expectPrints("!fly.int_tuple<(32,32,4)>", tm.getTileSizeMNK());
    _ = tm.getTiledThrValLayoutA();
    _ = tm.getThrLayoutVMNK();

    const atom = try types.MmaAtomType.get(ctx, .{ .mmaOp = (try rocdl.MmaOpCDNA3MFMAType.get(ctx, .{
        .m = 16,
        .n = 16,
        .k = 4,
        .elemTyA = .float(ctx, .f32),
        .elemTyB = .float(ctx, .f32),
        .elemTyAcc = .float(ctx, .f32),
    })).type_() });
    try expectPrints("!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>", atom);
    try expectPrints("!fly.int_tuple<(16,16,4)>", atom.getShapeMNK());
    try expectPrints("!fly.layout<64:1>", atom.getThrLayout());

    const wmma = try types.MmaAtomType.get(ctx, .{ .mmaOp = (try rocdl.MmaOpGFX11WMMAType.get(ctx, .{
        .m = 16,
        .n = 16,
        .k = 16,
        .elemTyA = .float(ctx, .f16),
        .elemTyB = .float(ctx, .f16),
        .elemTyAcc = .float(ctx, .f32),
    })).type_() });
    try expectPrints("!fly.layout<32:1>", wmma.getThrLayout());

    const ca = expect(parseType(ctx, "!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>"), .copy_atom);
    _ = ca.getThrValLayoutSrc();
}

test "type kinds" {
    const ctx = try testContext();
    defer ctx.deinit();
    const cases = .{
        .{ "!fly.memref<f32, global, (8,4):(4,1)>", TypeKind.memref },
        .{ "!fly.ptr<f32, global>", TypeKind.ptr },
        .{ "!fly.int_tuple<(2,4)>", TypeKind.int_tuple },
        .{ "!fly.layout<(8,4):(4,1)>", TypeKind.layout },
        .{ "!fly.tile<[8|4]>", TypeKind.tile },
        .{ "!fly.copy_atom<!fly.universal_copy<128>, 32>", TypeKind.copy_atom },
        .{ "!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>", TypeKind.mma_atom },
    };
    inline for (cases) |case| {
        try std.testing.expectEqual(case[1], typeKind(parseType(ctx, case[0])));
    }
    try std.testing.expectEqual(TypeKind.other, typeKind(.float(ctx, .f32)));
}

test "structural readers" {
    const ctx = try testContext();
    defer ctx.deinit();
    const it = expect(parseType(ctx, "!fly.int_tuple<((2,4),?)>"), .int_tuple);
    try std.testing.expectEqual(@as(usize, 2), it.getNumElements());
    try std.testing.expect(!it.isLeaf());
    try std.testing.expect(!it.isStatic());
    const inner = it.getElement(0);
    try std.testing.expectEqual(@as(usize, 2), inner.getNumElements());
    try std.testing.expectEqual(types.IntTupleType.Leaf{ .static = 4 }, inner.getElement(1).getLeaf());
    try std.testing.expectEqual(types.IntTupleType.Leaf.dynamic, it.getElement(1).getLeaf());

    const m = parseType(ctx, "!fly.memref<bf16, shared, S<3,3,3> o 0 o (64,32):(32,1), align<16>>");
    try expectPrints("!fly.int_tuple<(64,32)>", layoutLikeShape(m));
    try expectPrints("bf16", elemType(m));
}
