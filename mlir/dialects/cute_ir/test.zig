const std = @import("std");

const cute_ir = @import("mlir/dialects/cute_ir");
const cute_nvgpu = @import("mlir/dialects/cute_nvgpu");
const mlir = @import("mlir");

const ArithTupleIteratorType = cute_ir.ArithTupleIteratorType;
const ComposedLayoutType = cute_ir.ComposedLayoutType;
const ConstrainedIntType = cute_ir.ConstrainedIntType;
const CoordTensorType = cute_ir.CoordTensorType;
const CoordType = cute_ir.CoordType;
const FastDivmodDivisorType = cute_ir.FastDivmodDivisorType;
const IntTupleType = cute_ir.IntTupleType;
const LayoutType = cute_ir.LayoutType;
const MemRefType = cute_ir.MemRefType;
const PtrType = cute_ir.PtrType;
const ShapeType = cute_ir.ShapeType;
const SparseElemType = cute_ir.SparseElemType;
const StrideType = cute_ir.StrideType;
const SwizzleType = cute_ir.SwizzleType;
const TileType = cute_ir.TileType;
const TiledCopyType = cute_ir.TiledCopyType;
const TiledCopyV2Type = cute_ir.TiledCopyV2Type;
const TiledMmaType = cute_ir.TiledMmaType;
const TupleType = cute_ir.TupleType;
const BitLayoutAttr = cute_ir.BitLayoutAttr;
const CopyAtomAttr = cute_ir.CopyAtomAttr;
const CopyAtomV2Attr = cute_ir.CopyAtomV2Attr;
const MmaAtomAttr = cute_ir.MmaAtomAttr;
const MxMmaAtomAttr = cute_ir.MxMmaAtomAttr;
const SparseMmaAtomAttr = cute_ir.SparseMmaAtomAttr;
const SparseMxMmaAtomAttr = cute_ir.SparseMxMmaAtomAttr;

const operation_names: []const []const u8 = &.{
    "cute.add_offset",
    "cute.append_to_rank",
    "cute.apply_swizzle",
    "cute.assume",
    "cute.blocked_product",
    "cute.ceil_div",
    "cute.coalesce",
    "cute.complement",
    "cute.composed_get_inner",
    "cute.composed_get_offset",
    "cute.composed_get_outer",
    "cute.composition",
    "cute.copy",
    "cute.copy.make_fragment",
    "cute.copy_atom_call",
    "cute.cosize",
    "cute.crd2idx",
    "cute.deref_arith_tuple_iter",
    "cute.deref_desc_iter",
    "cute.derefine",
    "cute.dice",
    "cute.elem_less",
    "cute.equal",
    "cute.fast_divmod.compute",
    "cute.fast_divmod.create_divisor",
    "cute.fast_divmod.divide",
    "cute.fast_divmod.get_aux",
    "cute.fast_divmod.get_divisor",
    "cute.fast_divmod.make_divisor",
    "cute.filter",
    "cute.filter_zeros",
    "cute.flat_divide",
    "cute.flat_product",
    "cute.gemm",
    "cute.get",
    "cute.get_flat_coord",
    "cute.get_hier_coord",
    "cute.get_integral_coord",
    "cute.get_iter",
    "cute.get_layout",
    "cute.get_layouts_from_tile",
    "cute.get_leaves",
    "cute.get_scalars",
    "cute.get_shape",
    "cute.get_stride",
    "cute.group_modes",
    "cute.idx2crd",
    "cute.increment_coord",
    "cute.inttoptr",
    "cute.kernel_smem_size",
    "cute.left_inverse",
    "cute.load_scaled_index",
    "cute.local_partition",
    "cute.local_tile",
    "cute.logical_divide",
    "cute.logical_product",
    "cute.make_arith_tuple_iter",
    "cute.make_atom",
    "cute.make_composed_layout",
    "cute.make_coord",
    "cute.make_desc_iter",
    "cute.make_fragment_like",
    "cute.make_identity_layout",
    "cute.make_identity_tensor",
    "cute.make_int_tuple",
    "cute.make_layout",
    "cute.make_layout_like",
    "cute.make_ordered_layout",
    "cute.make_shape",
    "cute.make_sparse_elem",
    "cute.make_stride",
    "cute.make_tile",
    "cute.make_tiled_copy",
    "cute.make_tiled_copy_v2",
    "cute.make_tiled_mma",
    "cute.make_tuple",
    "cute.make_view",
    "cute.memref.alloc_smem",
    "cute.memref.alloca",
    "cute.memref.load",
    "cute.memref.load_vec",
    "cute.memref.store",
    "cute.memref.store_vec",
    "cute.mma.make_fragment",
    "cute.mma_atom_call",
    "cute.prefetch",
    "cute.prefetch_atom_call",
    "cute.prepend_to_rank",
    "cute.print",
    "cute.print_tma_desc_im2col",
    "cute.print_tma_desc_tiled",
    "cute.print_view",
    "cute.ptr.load",
    "cute.ptr.store",
    "cute.ptrtoint",
    "cute.raked_product",
    "cute.random_delay",
    "cute.recast_iter",
    "cute.recast_layout",
    "cute.reduce",
    "cute.right_inverse",
    "cute.select",
    "cute.shape_div",
    "cute.size",
    "cute.slice",
    "cute.smem_partition_size",
    "cute.static",
    "cute.stencil_divide",
    "cute.symbolic",
    "cute.tile_to_shape",
    "cute.tiled.copy.partition_D",
    "cute.tiled.copy.partition_S",
    "cute.tiled.copy.retile",
    "cute.tiled.mma.partition",
    "cute.tiled.mma.partition_shape",
    "cute.tiled_divide",
    "cute.tiled_product",
    "cute.to_int_tuple",
    "cute.tuple.product",
    "cute.tuple.product_each",
    "cute.tuple_add",
    "cute.tuple_div",
    "cute.tuple_mod",
    "cute.tuple_mul",
    "cute.tuple_sub",
    "cute.unpack_tuple",
    "cute.zipped_divide",
    "cute.zipped_product",
};

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    cute_ir.registerDialect(registry);
    cute_nvgpu.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(cute_ir);
    inline for (comptime std.meta.declarations(cute_ir)) |decl| {
        const T = @field(cute_ir, decl);
        if (@TypeOf(T) == type) std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const ctx = try testContext();
    defer ctx.deinit();
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}

test "generated types and attributes rebuild from their getters" {
    @setEvalBranchQuota(100_000);
    const ctx = try testContext();
    defer ctx.deinit();
    inline for (.{
        .{ ArithTupleIteratorType, "!cute.arith_tuple_iter<\"(?,?{div=16})\">" },
        .{ ComposedLayoutType, "!cute.composed_layout<\"(4,5):(1,4) o 2 o (2,3):(1,2)\">" },
        .{ ConstrainedIntType, "!cute.i16<pow2, divby 8>" },
        .{ CoordTensorType, "!cute.coord_tensor<\"(?,?,?)\", \"((512,1),1):((1/2@0,0),0)\">" },
        .{ CoordType, "!cute.coord<\"1\">" },
        .{ FastDivmodDivisorType, "!cute.fast_divmod_divisor<32>" },
        .{ IntTupleType, "!cute.int_tuple<\"1\">" },
        .{ LayoutType, "!cute.layout<\"(2,3):(1,2)\">" },
        .{ MemRefType, "!cute.memref<bf16, smem, align<1024>, S<2,4,3>, \"((8,16),(32,1),(1,3)):((32,256),(1,0),(0,4096))\">" },
        .{ PtrType, "!cute.ptr<bf16, smem, align<1024>, S<2,4,3>>" },
        .{ ShapeType, "!cute.shape<\"(2,3)\">" },
        .{ SparseElemType, "!cute.sparse_elem<2, vector<1xf16>>" },
        .{ StrideType, "!cute.stride<\"1\">" },
        .{ SwizzleType, "!cute.swizzle<\"S<0,4,3>\">" },
        .{ TileType, "!cute.tile<\"[(2,3):(1,2)]\">" },
        .{ TiledCopyType, "!cute.tiled_copy<!cute_nvgpu.atom.universal_copy<f32>, layout_copy_tv = <\"(32,4):(4,1)\">, tiler_mn = <\"[4:1;32:1]\">>" },
        .{ TiledCopyV2Type, "!cute.tiled_copy_v2<!cute_nvgpu.atom.sm100_s2t_copy_v2<f8E4M3FN, num_dp = 32, num_bit = 128, num_cta = 1, smem_major = k, broadcast = x4>, layout_copy_tv = <\"(1,8):(0,1)\">, tiler_mn = <\"[8:1]\">>" },
        .{ TiledMmaType, "!cute.tiled_mma<!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >, atom_layout_MNK = <\"(16,16,1):(16,1,0)\">, permutation_MNK = <\"[(16,4):(4,1);(16,4):(4,1);_]\">>" },
        .{ TupleType, "!cute.tuple<i32, f32>" },
        .{ BitLayoutAttr, "#cute.bitlayout<<\"(8,4):(1,8)\">>" },
        .{ CopyAtomAttr, "#cute.copy_atom<thr_id = <\"1:0\">, layout_tv = (<\"(1,8):(0,1)\">, <\"(1,8):(0,1)\">, <\"(1,8):(0,1)\">)>" },
        .{ CopyAtomV2Attr, "#cute.copy_atom_v2<thr_id = <\"1:0\">, layout_tv = (<\"(1,8):(0,1)\">, <\"(1,8):(0,1)\">), frg = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>" },
        .{ MmaAtomAttr, "#cute.mma_atom<<\"(16,8,16)\">, <\"32:1\">, layout_TV = (<\"1:0\">, <\"1:0\">, <\"1:0\">), shapes = (<\"(16,16)\">, <\"(8,16)\">, <\"(16,8)\">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>" },
        .{ MxMmaAtomAttr, "#cute.mx_mma_atom<<\"(16,8,16)\">, <\"32:1\">, layout_TV = (<\"1:0\">, <\"1:0\">, <\"1:0\">, <\"1:0\">, <\"1:0\">), shapes = (<\"(16,16)\">, <\"(8,16)\">, <\"(16,8)\">, <\"1\">, <\"2\">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>" },
        .{ SparseMmaAtomAttr, "#cute.sparse_mma_atom<<\"(16,8,16)\">, <\"32:1\">, layout_TV = (<\"1:0\">, <\"1:0\">, <\"1:0\">, <\"1:0\">), shapes = (<\"(16,16)\">, <\"(8,16)\">, <\"(16,8)\">, <\"1\">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>" },
        .{ SparseMxMmaAtomAttr, "#cute.sparse_mx_mma_atom<<\"(16,8,16)\">, <\"32:1\">, layout_TV = (<\"1:0\">, <\"2:0\">, <\"3:0\">, <\"4:0\">, <\"5:0\">, <\"6:0\">), shapes = (<\"1\">, <\"2\">, <\"3\">, <\"4\">, <\"5\">, <\"6\">) frgs = (#cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>, #cute_nvgpu.not_implemented_frg<>)>" },
    }) |example| {
        const T = example[0];
        const is_type = @hasDecl(T, "type_");
        const parsed = if (is_type) try mlir.Type.parse(ctx, example[1]) else try mlir.Attribute.parse(ctx, example[1]);
        const value = parsed.isA(T) orelse return error.TestUnexpectedResult;
        try std.testing.expect((try rebuild(T, ctx, value)).eql(value));
        var text: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer text.deinit();
        try text.writer.print("{f}", .{value});
        try std.testing.expect(parsed.eql(if (is_type) try mlir.Type.parse(ctx, text.written()) else try mlir.Attribute.parse(ctx, text.written())));
    }
    // Every case of every enum attribute.
    inline for (.{cute_ir}) |namespace| {
        inline for (comptime std.meta.declarations(namespace)) |decl| {
            const T = @field(namespace, decl);
            if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque" and @hasDecl(T, "InitArgs")) {
                const fields = @typeInfo(T.InitArgs).@"struct".field_names;
                if (fields.len == 1 and comptime std.mem.eql(u8, fields[0], "value")) {
                    inline for (@typeInfo(@FieldType(T.InitArgs, fields[0])).@"enum".field_names) |case| {
                        const case_value = @field(@FieldType(T.InitArgs, fields[0]), case);
                        const attr = try T.get(ctx, .{ .value = case_value });
                        try std.testing.expectEqual(@backingInt(case_value), @backingInt(attr.value()));
                        try std.testing.expect((try rebuild(T, ctx, attr)).eql(attr));
                    }
                }
            }
        }
    }
}

test "algebra attributes build from their leaves and tuples" {
    const ctx = try testContext();
    defer ctx.deinit();

    const int_tuple = try cute_ir.IntTupleAttr.getTuple(ctx, &.{
        try .getStatic(ctx, 0),
        try .getDynamic(ctx, 32, 1),
        try .getDynamic(ctx, 32, 16),
        try .getTuple(ctx, &.{ try .getStatic(ctx, 4), try .getDynamic(ctx, 64, 1) }),
    });
    try expectType(ctx, "!cute.int_tuple<\"(0,?,?{div=16},(4,?{i64}))\">", (try IntTupleType.get(ctx, .{ .attr = int_tuple.attribute() })).type_());
    try expectType(ctx, "!cute.int_tuple<\"()\">", (try IntTupleType.get(ctx, .{ .attr = (try cute_ir.IntTupleAttr.getTuple(ctx, &.{})).attribute() })).type_());

    const coord = try cute_ir.CoordAttr.getTuple(ctx, &.{
        try .getUnderscore(ctx),
        try .getStatic(ctx, 1),
        try .getTuple(ctx, &.{ try .getDynamic(ctx, 32, 1), try .getUnderscore(ctx) }),
    });
    try expectType(ctx, "!cute.coord<\"(_,1,(?,_))\">", (try CoordType.get(ctx, .{ .attr = coord.attribute() })).type_());

    const shape = try cute_ir.ShapeAttr.getTuple(ctx, &.{
        try .getTuple(ctx, &.{ try .getStatic(ctx, 512), try .getStatic(ctx, 1) }),
        try .getStatic(ctx, 1),
    });
    const stride = try cute_ir.StrideAttr.getTuple(ctx, &.{
        try .getTuple(ctx, &.{ try .getBasis(ctx, 1, 2, &.{0}), try .getStatic(ctx, 0) }),
        try .getBasis(ctx, 3, 1, &.{ 0, 1 }),
    });
    const layout = try cute_ir.LayoutAttr.get(ctx, .{ .shape = shape, .stride = stride });
    try expectType(ctx, "!cute.layout<\"((512,1),1):((1/2@0,0),3@1@0)\">", (try LayoutType.get(ctx, .{ .attr = layout.attribute() })).type_());
    try std.testing.expectError(error.InvalidMlir, cute_ir.LayoutAttr.get(ctx, .{ .shape = shape, .stride = try .getStatic(ctx, 1) }));

    const row = try cute_ir.LayoutAttr.get(ctx, .{
        .shape = try .getTuple(ctx, &.{ try .getStatic(ctx, 8), try .getStatic(ctx, 64) }),
        .stride = try .getTuple(ctx, &.{ try .getStatic(ctx, 64), try .getDynamic(ctx, 32, 8) }),
    });
    try expectType(ctx, "!cute.layout<\"(8,64):(64,?{div=8})\">", (try LayoutType.get(ctx, .{ .attr = row.attribute() })).type_());

    const tile = try cute_ir.TileAttr.getTuple(ctx, &.{ try .getLayout(ctx, row), try .getUnderscore(ctx) });
    try expectType(ctx, "!cute.tile<\"[(8,64):(64,?{div=8});_]\">", (try TileType.get(ctx, .{ .attr = tile.attribute() })).type_());

    const swizzle = try cute_ir.SwizzleAttr.get(ctx, .{ .bits = 3, .base = 4, .shift = 3 });
    try expectType(ctx, "!cute.swizzle<\"S<3,4,3>\">", (try SwizzleType.get(ctx, .{ .attr = swizzle.attribute() })).type_());

    const outer = try cute_ir.LayoutAttr.get(ctx, .{
        .shape = try .getTuple(ctx, &.{ try .getStatic(ctx, 8), try .getStatic(ctx, 64) }),
        .stride = try .getTuple(ctx, &.{ try .getStatic(ctx, 64), try .getStatic(ctx, 1) }),
    });
    const composed = try cute_ir.ComposedLayoutAttr.get(ctx, .{ .inner = .{ .swizzle = swizzle }, .offset = try .getStatic(ctx, 0), .outer = outer });
    try expectType(ctx, "!cute.composed_layout<\"S<3,4,3> o 0 o (8,64):(64,1)\">", (try ComposedLayoutType.get(ctx, .{ .attr = composed.attribute() })).type_());

    try std.testing.expectError(error.InvalidMlir, cute_ir.IntTupleAttr.getDynamic(ctx, 7, 1));
    try std.testing.expectError(error.InvalidMlir, cute_ir.StrideAttr.getBasis(ctx, 1, 1, &.{}));
    try std.testing.expectError(error.InvalidMlir, cute_ir.SwizzleAttr.get(ctx, .{ .bits = 5, .base = 4, .shift = 3 }));
}

fn expectType(ctx: *mlir.Context, expected: []const u8, actual: *const mlir.Type) !void {
    const parsed = try mlir.Type.parse(ctx, expected);
    if (!parsed.eql(actual)) {
        std.debug.print("expected {s}, built {f}\n", .{ expected, actual });
        return error.TestUnexpectedResult;
    }
}

fn rebuild(comptime T: type, ctx: *mlir.Context, value: *const T) !*const T {
    var args: T.InitArgs = undefined;
    var elements: [8]*const mlir.Type = undefined;
    inline for (@typeInfo(T.InitArgs).@"struct".field_names) |field| {
        const name = [_]u8{comptime std.ascii.toUpper(field[0])} ++ field[1..];
        const noun = [_]u8{comptime std.ascii.toLower(field[0])} ++ field[1..];
        const accessor = if (@hasDecl(T, noun ++ "_")) noun ++ "_" else noun;
        if (@FieldType(T.InitArgs, field) == []const *const mlir.Type) {
            const count = @field(T, "num" ++ name)(value);
            for (elements[0..count], 0..) |*element, i| element.* = @field(T, noun[0 .. noun.len - 1])(value, i);
            @field(args, field) = elements[0..count];
        } else {
            @field(args, field) = @field(T, accessor)(value);
        }
    }
    return T.get(ctx, args);
}
