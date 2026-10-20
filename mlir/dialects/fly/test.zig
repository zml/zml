const std = @import("std");
const mlir = @import("mlir");
const fly = @import("mlir/dialects/fly");
const rocdl = @import("mlir/dialects/fly_rocdl");
const attributes = fly.attributes;
const ops = fly.ops;
const types = fly.types;
const TypeKind = fly.TypeKind;
const typeKind = fly.typeKind;
const expect = fly.expect;
const layoutLikeShape = fly.layoutLikeShape;
const elemType = fly.elemType;
const addressSpace = fly.addressSpace;
const ptrWithElem = fly.ptrWithElem;
pub const dialects_needed = [_][]const u8{ "func", "arith", "scf", "math", "vector", "memref", "llvm", "gpu", "rocdl", "ub", "fly", "fly_rocdl" };
fn insertDialects(registry: *mlir.DialectRegistry) void {
    inline for (dialects_needed) |name| registry.registerDialect(name);
}

test {
    std.testing.refAllDecls(fly);
    inline for (.{ types, attributes, ops, rocdl, rocdl.ops }) |module| {
        std.testing.refAllDecls(module);
        inline for (comptime std.meta.declarations(module)) |decl| {
            const value = @field(module, decl);
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

test "every generated operation builder names a registered operation" {
    const ctx = try testContext();
    defer ctx.deinit();

    for (ops.names ++ rocdl.ops.names) |name| {
        if (!ctx.isRegisteredOperation(name)) {
            std.debug.print("`{s}` is not registered\n", .{name});
            return error.TestUnexpectedResult;
        }
    }
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

test "fly enum attributes" {
    const ctx = try testContext();
    defer ctx.deinit();
    _ = try attributes.mmaOperandAttr(ctx, .a);
    _ = try attributes.gemmTraversalOrderAttr(ctx, .kmn);
}

test "inferred fly ops compute layout algebra" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc: *const mlir.Location = .unknown(ctx);
    const module: *mlir.Module = .init(loc);
    defer module.deinit();
    const block = module.body();

    // raked_product((8,16):(16,1), (1,4):(1,1)) — the vectorAdd TV layout.
    const thr = ops.static(ctx, try mlir.Type.parse(ctx, "!fly.layout<(8,16):(16,1)>"), loc).appendTo(block);
    const val = ops.static(ctx, try mlir.Type.parse(ctx, "!fly.layout<(1,4):(1,1)>"), loc).appendTo(block);
    const mn = ops.raked_product(ctx, thr.result(0), val.result(0), loc).appendTo(block);
    const shape = ops.get_shape(ctx, mn.result(0), loc).appendTo(block);
    const tiler = ops.int_tuple_product_each(ctx, shape.result(0), loc).appendTo(block);

    try expectPrints("!fly.int_tuple<(8,64)>", tiler.result(0).type_());
    try std.testing.expect(module.operation().verify());
}

test "atom traits" {
    const ctx = try testContext();
    defer ctx.deinit();

    // MFMA 16x16x4 f32 tiled over a (2,2,1) atom layout.
    const tm = expect(try mlir.Type.parse(ctx, "!fly.tiled_mma<!fly.mma_atom<!fly_rocdl.cdna3.mfma<16x16x4, (f32, f32) -> f32>>, <(2,2,1):(1,2,0)>>"), .tiled_mma);
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

    const ca = expect(try mlir.Type.parse(ctx, "!fly.copy_atom<!fly_rocdl.cdna3.buffer_copy<32>, 32>"), .copy_atom);
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
        try std.testing.expectEqual(case[1], typeKind(try mlir.Type.parse(ctx, case[0])));
    }
    try std.testing.expectEqual(TypeKind.other, typeKind(.float(ctx, .f32)));
}

test "structural readers" {
    const ctx = try testContext();
    defer ctx.deinit();
    const it = expect(try mlir.Type.parse(ctx, "!fly.int_tuple<((2,4),?)>"), .int_tuple);
    try std.testing.expectEqual(@as(usize, 2), it.getNumElements());
    try std.testing.expect(!it.isLeaf());
    try std.testing.expect(!it.isStatic());
    const inner = it.getElement(0);
    try std.testing.expectEqual(@as(usize, 2), inner.getNumElements());
    try std.testing.expectEqual(types.IntTupleType.Leaf{ .static = 4 }, inner.getElement(1).getLeaf());
    try std.testing.expectEqual(types.IntTupleType.Leaf.dynamic, it.getElement(1).getLeaf());

    const m = try mlir.Type.parse(ctx, "!fly.memref<bf16, shared, S<3,3,3> o 0 o (64,32):(32,1), align<16>>");
    try expectPrints("!fly.int_tuple<(64,32)>", layoutLikeShape(m));
    try expectPrints("bf16", elemType(m));
}
