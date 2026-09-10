const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

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
pub fn inferred(ctx: *mlir.Context, comptime mnemonic: []const u8, operands: []const *const mlir.Value, attributes: Attrs, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = operands },
        .result_type_inference = true,
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// Explicit result types.
pub fn typed(ctx: *mlir.Context, comptime mnemonic: []const u8, operands: []const *const mlir.Value, results: []const *const mlir.Type, attributes: Attrs, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = operands },
        .results = .{ .flat = results },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// No results.
pub fn effect(ctx: *mlir.Context, comptime mnemonic: []const u8, operands: []const *const mlir.Value, attributes: Attrs, location: *const mlir.Location) *mlir.Operation {
    return make(ctx, opName(mnemonic), .{
        .operands = .{ .flat = operands },
        .attributes = attributes.constSlice(),
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

pub const MmaOperand = enum { a, b, c, d };

pub fn mmaOperandAttr(ctx: *mlir.Context, operand: MmaOperand) *const mlir.Attribute {
    return switch (operand) {
        inline else => |o| parseAttr(ctx, "#fly<mma_operand " ++ @tagName(o) ++ ">"),
    };
}

pub const GemmTraversalOrder = enum { kmn, knm, mkn, mnk, nkm, nmk, kmn_serpentine, knm_serpentine, mkn_serpentine, mnk_serpentine, nkm_serpentine, nmk_serpentine };

pub fn gemmTraversalOrderAttr(ctx: *mlir.Context, order: GemmTraversalOrder) *const mlir.Attribute {
    return switch (order) {
        inline else => |o| parseAttr(ctx, "#fly<gemm_traversal_order " ++ @tagName(o) ++ ">"),
    };
}

/// Layouts FlyDSL computes from an atom's traits; each yields a
/// `!fly.layout` or `!fly.int_tuple` type.
pub const Derived = enum {
    copy_atom_thr_layout,
    copy_atom_tv_layout_src,
    copy_atom_tv_layout_dst,
    copy_atom_tv_layout_ref,
    mma_atom_thr_layout,
    mma_atom_shape_mnk,
    mma_atom_tv_layout_a,
    mma_atom_tv_layout_b,
    mma_atom_tv_layout_c,
    tiled_copy_tiled_tv_layout_src,
    tiled_copy_tiled_tv_layout_dst,
    tiled_mma_tile_size_mnk,
    tiled_mma_thr_layout_vmnk,
    tiled_mma_tiled_tv_layout_a,
    tiled_mma_tiled_tv_layout_b,
    tiled_mma_tiled_tv_layout_c,
};

/// Panics if `ty` is not the kind the accessor expects.
pub fn derived(which: Derived, ty: *const mlir.Type) *const mlir.Type {
    const raw = switch (which) {
        .copy_atom_thr_layout => c.zmlFlyCopyAtomThrLayout(ty.ptr()),
        .copy_atom_tv_layout_src => c.zmlFlyCopyAtomTvLayoutSrc(ty.ptr()),
        .copy_atom_tv_layout_dst => c.zmlFlyCopyAtomTvLayoutDst(ty.ptr()),
        .copy_atom_tv_layout_ref => c.zmlFlyCopyAtomTvLayoutRef(ty.ptr()),
        .mma_atom_thr_layout => c.zmlFlyMmaAtomThrLayout(ty.ptr()),
        .mma_atom_shape_mnk => c.zmlFlyMmaAtomShapeMNK(ty.ptr()),
        .mma_atom_tv_layout_a => c.zmlFlyMmaAtomTvLayoutA(ty.ptr()),
        .mma_atom_tv_layout_b => c.zmlFlyMmaAtomTvLayoutB(ty.ptr()),
        .mma_atom_tv_layout_c => c.zmlFlyMmaAtomTvLayoutC(ty.ptr()),
        .tiled_copy_tiled_tv_layout_src => c.zmlFlyTiledCopyTiledTvLayoutSrc(ty.ptr()),
        .tiled_copy_tiled_tv_layout_dst => c.zmlFlyTiledCopyTiledTvLayoutDst(ty.ptr()),
        .tiled_mma_tile_size_mnk => c.zmlFlyTiledMmaTileSizeMNK(ty.ptr()),
        .tiled_mma_thr_layout_vmnk => c.zmlFlyTiledMmaThrLayoutVMNK(ty.ptr()),
        .tiled_mma_tiled_tv_layout_a => c.zmlFlyTiledMmaTiledTvLayoutA(ty.ptr()),
        .tiled_mma_tiled_tv_layout_b => c.zmlFlyTiledMmaTiledTvLayoutB(ty.ptr()),
        .tiled_mma_tiled_tv_layout_c => c.zmlFlyTiledMmaTiledTvLayoutC(ty.ptr()),
    };
    const out: ?*const mlir.Type = @ptrCast(raw.ptr);
    return out orelse std.debug.panic("fly.derived({s}): `{f}` is not the expected atom/tiled type", .{ @tagName(which), ty });
}

// -----------------------------------------------------------------------------
// Structural readers: the accessors behind Python's `IntTuple.unpack()`.
// -----------------------------------------------------------------------------

fn typeFromC(raw: c.MlirType) ?*const mlir.Type {
    return @ptrCast(raw.ptr);
}

fn attrFromC(raw: c.MlirAttribute) ?*const mlir.Attribute {
    return @ptrCast(raw.ptr);
}

/// Which `fly` type a value has. The tags mirror `zmlFlyTypeKind`, which is a
/// `TypeSwitch` over the dialect: a type renamed at a pin bump fails to
/// compile there instead of silently reading as `.other` here.
pub const TypeKind = enum(i32) {
    memref = 0,
    coord_tensor = 1,
    ptr = 2,
    int_tuple = 3,
    layout = 4,
    composed_layout = 5,
    tile = 6,
    swizzle = 7,
    copy_atom = 8,
    mma_atom = 9,
    tiled_copy = 10,
    tiled_mma = 11,
    other = -1,
};

pub fn typeKind(ty: *const mlir.Type) TypeKind {
    return std.enums.fromInt(TypeKind, c.zmlFlyTypeKind(ty.ptr())) orelse .other;
}

/// 1 for a leaf.
pub fn intTupleRank(ty: *const mlir.Type) usize {
    const r = c.zmlFlyIntTupleRank(ty.ptr());
    if (r < 0) std.debug.panic("fly: `{f}` is not a !fly.int_tuple", .{ty});
    return @intCast(r);
}

pub fn intTupleIsLeaf(ty: *const mlir.Type) bool {
    return c.zmlFlyIntTupleIsLeaf(ty.ptr());
}

pub fn intTupleIsStatic(ty: *const mlir.Type) bool {
    return c.zmlFlyIntTupleIsStatic(ty.ptr());
}

pub fn intTupleAt(ty: *const mlir.Type, i: usize) *const mlir.Type {
    return typeFromC(c.zmlFlyIntTupleAt(ty.ptr(), @intCast(i))) orelse std.debug.panic("fly: no mode {d} in `{f}`", .{ i, ty });
}

pub const Leaf = union(enum) { static: i64, dynamic, none, basis };

pub fn intTupleLeaf(ty: *const mlir.Type) Leaf {
    var v: i64 = 0;
    return switch (c.zmlFlyIntTupleLeafKind(ty.ptr(), &v)) {
        0 => .{ .static = v },
        1 => .dynamic,
        2 => .none,
        3 => .basis,
        else => std.debug.panic("fly: `{f}` is not a leaf !fly.int_tuple", .{ty}),
    };
}

/// The shape of the outermost plain layout of any layout-like type.
pub fn layoutLikeShape(ty: *const mlir.Type) *const mlir.Type {
    return typeFromC(c.zmlFlyLayoutLikeShape(ty.ptr())) orelse std.debug.panic("fly: `{f}` carries no layout", .{ty});
}

pub fn layoutShape(ty: *const mlir.Type) *const mlir.Type {
    return typeFromC(c.zmlFlyLayoutShape(ty.ptr())) orelse std.debug.panic("fly: `{f}` is not a !fly.layout", .{ty});
}

pub fn layoutStride(ty: *const mlir.Type) *const mlir.Type {
    return typeFromC(c.zmlFlyLayoutStride(ty.ptr())) orelse std.debug.panic("fly: `{f}` is not a !fly.layout", .{ty});
}

pub fn elemType(ty: *const mlir.Type) *const mlir.Type {
    if (typeFromC(c.zmlFlyMemRefElemType(ty.ptr()))) |t| return t;
    if (typeFromC(c.zmlFlyPtrElemType(ty.ptr()))) |t| return t;
    std.debug.panic("fly: `{f}` has no element type", .{ty});
}

pub fn addressSpace(ty: *const mlir.Type) *const mlir.Attribute {
    if (attrFromC(c.zmlFlyMemRefAddressSpace(ty.ptr()))) |a| return a;
    if (attrFromC(c.zmlFlyPtrAddressSpace(ty.ptr()))) |a| return a;
    std.debug.panic("fly: `{f}` has no address space", .{ty});
}

pub fn ptrType(elem: *const mlir.Type, space: *const mlir.Attribute, alignment: i32, swizzle: ?*const mlir.Attribute) *const mlir.Type {
    return typeFromC(c.zmlFlyPtrTypeGet(elem.ptr(), space.ptr(), alignment, if (swizzle) |sw| sw.ptr() else .{ .ptr = null })) orelse @panic("fly: cannot build pointer type");
}

/// Keeps the address space, alignment and swizzle.
pub fn ptrWithElem(ptr: *const mlir.Type, elem: *const mlir.Type) *const mlir.Type {
    const raw_sw = c.zmlFlyPtrSwizzle(ptr.ptr());
    if (raw_sw.ptr == null) std.debug.panic("fly: `{f}` is not a !fly.ptr", .{ptr});
    return ptrType(elem, addressSpace(ptr), c.zmlFlyPtrAlignment(ptr.ptr()), attrFromC(raw_sw));
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
    if (opts.traversal_order) |o| at.appendAssumeCapacity(.named(ctx, "traversalOrder", gemmTraversalOrderAttr(ctx, o)));
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
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    insertDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
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
        const printed = w.buffered();
        const ty2 = try mlir.Type.parse(ctx, printed);
        try std.testing.expect(ty.eql(ty2));
    }
}

test "fly enum and gpu attributes parse" {
    const ctx = try testContext();
    defer ctx.deinit();
    _ = mmaOperandAttr(ctx, .a);
    _ = gemmTraversalOrderAttr(ctx, .kmn);
    _ = parseAttr(ctx, "#gpu<dim x>");
    _ = parseAttr(ctx, "#fly_rocdl.buffer_desc");
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
    const val = static(ctx, parseType(
        ctx,
        "!fly.layout<(1,4):(1,1)>",
    ), loc).appendTo(block);
    const mn = inferred(ctx, "raked_product", &.{ thr.result(0), val.result(0) }, .empty, loc).appendTo(block);
    const shape = inferred(ctx, "get_shape", &.{mn.result(0)}, .empty, loc).appendTo(block);
    const tiler = inferred(ctx, "int_tuple_product_each", &.{shape.result(0)}, .empty, loc).appendTo(block);

    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{tiler.result(0).type_()});
    try std.testing.expectEqualStrings("!fly.int_tuple<(8,64)>", w.buffered());
    try std.testing.expect(module.operation().verify());
}

test "derived atom layouts" {
    const ctx = try testContext();
    defer ctx.deinit();

    // MFMA 16x16x4 f32 tiled over a (2,2,1) atom layout.
    const tm = parseType(ctx, "!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>, <(2,2,1):(1,2,0)>>");
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{derived(.tiled_mma_tile_size_mnk, tm)});
    try std.testing.expectEqualStrings("!fly.int_tuple<(32,32,4)>", w.buffered());

    const atom = parseType(ctx, "!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>");
    w = .fixed(&buf);
    try w.print("{f}", .{derived(.mma_atom_shape_mnk, atom)});
    try std.testing.expectEqualStrings("!fly.int_tuple<(16,16,4)>", w.buffered());

    _ = derived(.tiled_mma_tiled_tv_layout_a, tm);
    _ = derived(.tiled_mma_thr_layout_vmnk, tm);
    const ca = parseType(ctx, "!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>");
    _ = derived(.copy_atom_tv_layout_src, ca);
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
    const it = parseType(ctx, "!fly.int_tuple<((2,4),?)>");
    try std.testing.expectEqual(@as(usize, 2), intTupleRank(it));
    try std.testing.expect(!intTupleIsLeaf(it));
    try std.testing.expect(!intTupleIsStatic(it));
    const inner = intTupleAt(it, 0);
    try std.testing.expectEqual(@as(usize, 2), intTupleRank(inner));
    try std.testing.expectEqual(Leaf{ .static = 4 }, intTupleLeaf(intTupleAt(inner, 1)));
    try std.testing.expectEqual(Leaf.dynamic, intTupleLeaf(intTupleAt(it, 1)));

    const m = parseType(ctx, "!fly.memref<bf16, shared, S<3,3,3> o 0 o (64,32):(32,1), align<16>>");
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{layoutLikeShape(m)});
    try std.testing.expectEqualStrings("!fly.int_tuple<(64,32)>", w.buffered());
    w = .fixed(&buf);
    try w.print("{f}", .{elemType(m)});
    try std.testing.expectEqualStrings("bf16", w.buffered());

    const p = parseType(ctx, "!fly.ptr<i8, shared, align<16>>");
    w = .fixed(&buf);
    try w.print("{f}", .{ptrWithElem(p, .float(ctx, .f32))});
    try std.testing.expectEqualStrings("!fly.ptr<f32, shared, align<16>>", w.buffered());
}
