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
    for (cute_ir.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
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
