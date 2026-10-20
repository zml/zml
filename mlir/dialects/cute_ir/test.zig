const std = @import("std");
const mlir = @import("mlir");
const cute = @import("mlir/dialects/cute_ir");
const nvgpu = @import("mlir/dialects/cute_nvgpu");
const nvvm = @import("mlir/dialects/nvvm").ops;
const cuda = @import("mlir/dialects/cuda").ops;
const ArithTupleIteratorType = cute.ArithTupleIteratorType;
const ComposedLayoutType = cute.ComposedLayoutType;
const ConstrainedIntType = cute.ConstrainedIntType;
const CoordTensorType = cute.CoordTensorType;
const CoordType = cute.CoordType;
const FastDivmodDivisorType = cute.FastDivmodDivisorType;
const IntTupleType = cute.IntTupleType;
const LayoutType = cute.LayoutType;
const MemRefType = cute.MemRefType;
const PtrType = cute.PtrType;
const ShapeType = cute.ShapeType;
const SparseElemType = cute.SparseElemType;
const StrideType = cute.StrideType;
const SwizzleType = cute.SwizzleType;
const TileType = cute.TileType;
const TiledCopyType = cute.TiledCopyType;
const TiledCopyV2Type = cute.TiledCopyV2Type;
const TiledMmaType = cute.TiledMmaType;
const TupleType = cute.TupleType;
const BitLayoutAttr = cute.BitLayoutAttr;
const CopyAtomAttr = cute.CopyAtomAttr;
const CopyAtomV2Attr = cute.CopyAtomV2Attr;
const MmaAtomAttr = cute.MmaAtomAttr;
const MxMmaAtomAttr = cute.MxMmaAtomAttr;
const SparseMmaAtomAttr = cute.SparseMmaAtomAttr;
const SparseMxMmaAtomAttr = cute.SparseMxMmaAtomAttr;
const get = cute.get;
const print = cute.print;
const size = cute.size;
const slice = cute.slice;
const operation_names = cute.operation_names;
fn registerDialects(registry: *mlir.DialectRegistry) void {
    inline for (.{ "cute", "cute_nvgpu", nvvm.dialect_handle, cuda.dialect_handle }) |name| registry.registerDialect(name);
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    registerDialects(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(cute);
    std.testing.refAllDecls(nvgpu);
    std.testing.refAllDecls(nvvm);
    std.testing.refAllDecls(cuda);
    inline for (.{ nvvm, cuda }) |namespace| {
        inline for (comptime std.meta.declarations(namespace)) |decl| {
            const T = @field(namespace, decl);
            if (@TypeOf(T) == type) std.testing.refAllDecls(T);
        }
    }
    inline for (comptime std.meta.declarations(cute)) |decl| {
        const T = @field(cute, decl);
        if (@TypeOf(T) == type) std.testing.refAllDecls(T);
    }
    inline for (comptime std.meta.declarations(nvgpu)) |decl| {
        const T = @field(nvgpu, decl);
        if (@TypeOf(T) == type) std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const ctx = try testContext();
    defer ctx.deinit();
    for (cute.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    for (nvgpu.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    for (nvvm.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    for (cuda.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
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
        .{ nvgpu.CopyAtomBulkCopyG2SType, "!cute_nvgpu.atom.bulk_copy_g2s<f32 copy_bits = 128 mcast = true>" },
        .{ nvgpu.CopyAtomBulkCopyS2GType, "!cute_nvgpu.atom.bulk_copy_s2g<f32 copy_bits = 128 mask = true>" },
        .{ nvgpu.CopyAtomBulkCopyS2SType, "!cute_nvgpu.atom.bulk_copy_s2s<f32 copy_bits = 128>" },
        .{ nvgpu.CopyAtomDsmemStoreType, "!cute_nvgpu.atom.dsmem_store<f32, 64 b>" },
        .{ nvgpu.CopyAtomG2RType, "!cute_nvgpu.atom.g2r<f32 copy_bits = 64 mem_order = relaxed mem_scope = gpu l2_prefetch_size = size_128b l1_cache_evict_priority = evict_last shared_space = cluster>" },
        .{ nvgpu.CopyAtomIm2ColTmaLoadType, "!cute_nvgpu.atom.im2col_tma_load<f16, copy_bits = 131072, num_cta = 1, g_stride = <\"()\"> mcast = true tma_gbasis = <\"(64,128):(1@1,1@0)\">>" },
        .{ nvgpu.CopyAtomIm2ColTmaStoreType, "!cute_nvgpu.atom.im2col_tma_store<f16, copy_bits = 65536, g_stride = <\"()\"> tma_gbasis = <\"(32,128):(1@1,1@0)\">>" },
        .{ nvgpu.CopyAtomLdsmType, "!cute_nvgpu.atom.ldsm<val_type = f16, mode = <\"(8,8)\">, sz_pattern = u16, num_matrices = 4, t>" },
        .{ nvgpu.CopyAtomNonExec2DGather4TmaLoadType, "!cute_nvgpu.atom.non_exec_2d_gather4_tma_load<sm_100, f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ nvgpu.CopyAtomNonExec2DScatter4TmaStoreType, "!cute_nvgpu.atom.non_exec_2d_scatter4_tma_store<f16, copy_bits = 1024, tma_gbasis = <\"(4,64):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ nvgpu.CopyAtomNonExecIm2ColTmaLoadType, "!cute_nvgpu.atom.non_exec_im2col_tma_load<sm_90, f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ nvgpu.CopyAtomNonExecIm2ColTmaStoreType, "!cute_nvgpu.atom.non_exec_im2col_tma_store<f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ nvgpu.CopyAtomNonExecTiledTmaLoadType, "!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = BF16_RN>" },
        .{ nvgpu.CopyAtomNonExecTiledTmaReduceType, "!cute_nvgpu.atom.non_exec_tiled_tma_reduce<ADD, f32, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F32_RN>" },
        .{ nvgpu.CopyAtomNonExecTiledTmaStoreType, "!cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = BF16_RN>" },
        .{ nvgpu.CopyAtomR2GType, "!cute_nvgpu.atom.r2g<f16 copy_bits = 128 l1_cache_evict_priority = no_allocate>" },
        .{ nvgpu.CopyAtomR2SType, "!cute_nvgpu.atom.r2s<f32 copy_bits = 64 mem_order = relaxed mem_scope = gpu shared_space = cluster>" },
        .{ nvgpu.CopyAtomS2RType, "!cute_nvgpu.atom.s2r<f32 copy_bits = 64 mem_order = relaxed mem_scope = gpu shared_space = cluster>" },
        .{ nvgpu.CopyAtomSIMTAsyncCopyType, "!cute_nvgpu.atom.simt_async_copy<bf16, cache = global, 128 b>" },
        .{ nvgpu.CopyAtomSIMTMultimemLdReduceType, "!cute_nvgpu.atom.simt_multimem_ld_reduce<bf16, 64 b, MIN, <F32>, acquire, gpu>" },
        .{ nvgpu.CopyAtomSIMTMultimemRedType, "!cute_nvgpu.atom.simt_multimem_red<f16, 64 b, MAX, release, gpu>" },
        .{ nvgpu.CopyAtomSIMTMultimemStType, "!cute_nvgpu.atom.simt_multimem_st<f16, 64 b, release, gpu>" },
        .{ nvgpu.CopyAtomSIMTSyncCopyType, "!cute_nvgpu.atom.universal_copy<f32, 128 b, src_space = gmem, dst_space = smem>" },
        .{ nvgpu.CopyAtomSM100CopyS2TType, "!cute_nvgpu.atom.s2t_copy<f8E4M3FN, 32 DP, 128 bit, 1 cta, x4>" },
        .{ nvgpu.CopyAtomSM100S2TCopyV2Type, "!cute_nvgpu.atom.sm100_s2t_copy_v2<f8E4M3FN, num_dp = 32, num_bit = 128, num_cta = 1, smem_major = k, broadcast = x4>" },
        .{ nvgpu.CopyAtomSM100TmemLoadType, "!cute_nvgpu.atom.tmem_load<f32, 32 DP, 32 bit, x4, pack16b>" },
        .{ nvgpu.CopyAtomSM100TmemStoreType, "!cute_nvgpu.atom.tmem_store<f32, 32 DP, 32 bit, x4, expand16b>" },
        .{ nvgpu.CopyAtomSM107TmemLoadSPCompressType, "!cute_nvgpu.atom.tmem_load_spcompress<f32, 32 DP, 32 bit, x4, maxabs, red, nan>" },
        .{ nvgpu.CopyAtomSM10xTmemLoadRedType, "!cute_nvgpu.atom.tmem_load_red<f32, 16 DP, 32 bit, x4, maxabs, nan, half_split_off=3 : i64>" },
        .{ nvgpu.CopyAtomStsmType, "!cute_nvgpu.atom.stsm<f16, mode = <\"(8,8)\">, num_matrices = 4, n>" },
        .{ nvgpu.CopyAtomTmaLoadType, "!cute_nvgpu.atom.tma_load<bf16, copy_bits = 1024, mode = tiled, num_cta = 2, g_stride = <\"()\"> mcast = true tma_gbasis = <\"(64,1):(1@0,1@1)\">>" },
        .{ nvgpu.CopyAtomTmaReduceType, "!cute_nvgpu.atom.tma_reduce<f32, copy_bits = 131072, mode = tiled, kind = ADD, g_stride = <\"()\"> tma_gbasis = <\"(32,128):(1@1,1@0)\">>" },
        .{ nvgpu.CopyAtomTmaStoreType, "!cute_nvgpu.atom.tma_store<bf16, copy_bits = 16384, mode = tiled, g_stride = <\"()\"> tma_gbasis = <\"(64,16):(1@0,1@1)\">>" },
        .{ nvgpu.MmaAtomSM100UMMABlockScaledSparseType, "!cute_nvgpu.sm100.mma_bs_sp<128x128x64, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, sparse_metadata_format = tid, frag_kind = ss, vec_size = 32>" },
        .{ nvgpu.MmaAtomSM100UMMABlockScaledType, "!cute_nvgpu.sm100.mma_bs<128x128x96, num_cta = 1, ab_major = (k, k), elem_type = (f4E2M1FN, f4E2M1FN, f32), sf_type = f8E8M0FNU, frag_kind = ss, vec_size = 16, arch_promote = sm_103>" },
        .{ nvgpu.MmaAtomSM100UMMASparseType, "!cute_nvgpu.sm100.mma_sp<128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f16, f16, f32), e_type = i8, sparse_metadata_format = tid, frag_kind = ss, c_scale_exp = 0>" },
        .{ nvgpu.MmaAtomSM100UMMAType, "!cute_nvgpu.sm100.mma<128x128x16, num_cta = 1, ab_major = (k, mn), elem_type = (f16, f16, f32), frag_kind = ss, c_scale_exp = 0>" },
        .{ nvgpu.MmaAtomSM107UMMABlockScaledSparseType, "!cute_nvgpu.sm107.mma_bs_sp<128x128x64, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, sparse_metadata_format = tid, frag_kind = ss, vec_size = 32, ab_collector_op = (discard, discard)>" },
        .{ nvgpu.MmaAtomSM107UMMABlockScaledType, "!cute_nvgpu.sm107.mma_bs<128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, frag_kind = ss, vec_size = 32, ab_collector_op = (discard, use), int_overflow = <satfinite>>" },
        .{ nvgpu.MmaAtomSM107UMMASparseType, "!cute_nvgpu.sm107.mma_sp<128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f16, f16, f32), e_type = i8, sparse_metadata_format = tid, frag_kind = ss, c_scale_exp = 0, ab_collector_op = (discard, discard)>" },
        .{ nvgpu.MmaAtomSM107UMMAType, "!cute_nvgpu.sm107.mma<128x128x16, num_cta = 1, ab_major = (k, mn), elem_type = (f16, f16, f32), frag_kind = ss, c_scale_exp = 0, ab_collector_op = (lastuse, fill)>" },
        .{ nvgpu.MmaAtomSM120BlockScaledType, "!cute_nvgpu.SM120.mma_bs<16x8x64, vec_size = 32, elem_type = (f4E2M1FN, f4E2M1FN, f32), sf_type = f8E8M0FNU, use_sf_layout_TV = true>" },
        .{ nvgpu.MmaAtomSM80SparseType, "!cute_nvgpu.sm80.sparse_mma<16x8x64, elem_type = (i8, i8, i32), md_format = tid, int_overflow = <satfinite> >" },
        .{ nvgpu.MmaAtomSM80Type, "!cute_nvgpu.sm80.mma<16x8x32, elem_type = (i8, i8, i32), int_overflow = <wrapped>, binary_op = <none> >" },
        .{ nvgpu.MmaAtomSM89Type, "!cute_nvgpu.sm89.mma<16x8x32, elem_type = (f8E4M3FN, f8E4M3FN, f32) >" },
        .{ nvgpu.MmaAtomSM90Type, "!cute_nvgpu.sm90.mma<64x128x32, ab_major = (k, k), elem_type = (i8, i8, i32), frag_kind = ss, int_overflow = <satfinite>, a_neg, b_neg>" },
        .{ nvgpu.SmemDescCircularSM103Type, "!cute_nvgpu.sm103.smem_desc_circular<<\"(8,128,2):(128,1,1024)\">>" },
        .{ nvgpu.SmemDescCircularSM107Type, "!cute_nvgpu.sm107.smem_desc_circular<<\"(8,128,2):(128,1,1024)\">>" },
        .{ nvgpu.SmemDescSM107Type, "!cute_nvgpu.sm107.smem_desc" },
        .{ nvgpu.SmemDescType, "!cute_nvgpu.smem_desc" },
        .{ nvgpu.SmemDescViewType, "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(2,3):(1,2)\">" },
        .{ nvgpu.TmaDescriptorIm2ColType, "!cute_nvgpu.tma_descriptor_im2col" },
        .{ nvgpu.TmaDescriptorTiledType, "!cute_nvgpu.tma_descriptor_tiled" },
        .{ nvgpu.UniversalFmaAtomType, "!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >" },
        .{ nvgpu.WorkIdResponseType, "!cute_nvgpu.workid_response" },
        .{ nvgpu.NotImplementedFrgAttr, "#cute_nvgpu.not_implemented_frg<>" },
        .{ nvgpu.RmemFrgAttr, "#cute_nvgpu.rmem_frg<value_type = f32, operand = A, derive_elem_type = true>" },
        .{ nvgpu.SM100CircularSmemFrgAttr, "#cute_nvgpu.arch.sm100.circular_smem_frg<major = k>" },
        .{ nvgpu.SM100SmemFrgAttr, "#cute_nvgpu.arch.sm100.smem_frg<major = k>" },
        .{ nvgpu.SM100TmemEFrgAttr, "#cute_nvgpu.arch.sm100.tmem_e_frg<a_type = f16, e_type = i8>" },
        .{ nvgpu.SM100TmemFrgAttr, "#cute_nvgpu.arch.sm100.tmem_frg<data_type = f32, storage_type = f32, cta_group = 1, tmem_alloc_mode = Interleaved>" },
        .{ nvgpu.SM100TmemSfFrgAttr, "#cute_nvgpu.arch.sm100.tmem_sf_frg<sf_type = f8E8M0FNU, sf_vec_size = 32, cta_group = 1, is_sfa = true, tmem_alloc_mode = Interleaved>" },
        .{ nvgpu.SM107SmemFrgAttr, "#cute_nvgpu.arch.sm107.smem_frg<major = k>" },
        .{ nvgpu.SM90SmemFrgAttr, "#cute_nvgpu.arch.sm90.smem_frg<major = k>" },
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
    inline for (.{ cute, nvgpu }) |namespace| {
        inline for (comptime std.meta.declarations(namespace)) |decl| {
            const T = @field(namespace, decl);
            if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque" and @hasDecl(T, "InitArgs")) {
                const fields = @typeInfo(T.InitArgs).@"struct".field_names;
                if (fields.len == 1 and comptime std.mem.eql(u8, fields[0], "value")) {
                    inline for (@typeInfo(@FieldType(T.InitArgs, fields[0])).@"enum".field_names) |case| {
                        const case_value = @field(@FieldType(T.InitArgs, fields[0]), case);
                        const attr = try T.get(ctx, .{ .value = case_value });
                        try std.testing.expectEqual(@backingInt(case_value), @backingInt(attr.getValue()));
                        try std.testing.expect((try rebuild(T, ctx, attr)).eql(attr));
                    }
                }
            }
        }
    }
    try std.testing.expect((try nvgpu.SmemDescType.get(ctx, .{})).type_().isA(nvgpu.NVGPUType) != null);
}

test "nvvm and cuda types and attributes print as the compiler does" {
    @setEvalBranchQuota(100_000);
    const ctx = try testContext();
    defer ctx.deinit();
    // The compiler's printed forms, which these must print byte for byte.
    inline for (.{
        .{ nvvm.MMAShapeAttr, "#nvvm.shape<m = 16, n = 8, k = 16>" },
        .{ nvvm.LdStMatrixShapeAttr, "#nvvm.ld_st_matrix_shape<m = 8, n = 8>" },
        .{ nvvm.TargetAttr, "#nvvm.target" },
        .{ nvvm.TargetAttr, "#nvvm.target<verifyTarget = true>" },
        .{ nvvm.TargetAttr, "#nvvm.target<O = 3, triple = \"t\", chip = \"sm_90\", features = \"f\", flags = {x}, link = [\"a\"]>" },
        .{ cuda.AssumeKernelAttr, "#cuda.assume_kernel_attr<true>" },
        .{ cuda.ComputeTargetAttr, "#cuda.compute_target<sass, portable, [sm_90, sm_100]>" },
        .{ cuda.DevMaxSharedMemoryOptinAttr, "#cuda.dev_max_shared_memory_optin" },
        .{ cuda.ExecutableAttr, "#cuda.executable<elf>" },
        .{ cuda.LaunchConfigType, "!cuda.launch_cfg<max_attrs = 3>" },
        .{ cuda.StreamType, "!cuda.stream" },
        .{ cuda.ResultType, "!cuda.result" },
        .{ cuda.IntType, "!cuda.int" },
        .{ cuda.RuntimeDim3Type, "!cuda.runtime.dim_3" },
        .{ cuda.RuntimeLaunchAttributeValueSyncPolicyType, "!cuda.runtime.launch_attribute_value.sync_policy" },
        .{ cuda.RuntimeLaunchAttributeValueClusterSchedulingPolicyPreferenceType, "!cuda.runtime.launch_attribute_value.cluster_scheduling_policy_preference" },
        .{ cuda.RuntimeLaunchAttributeValueNvlinkUtilCentricSchedulingType, "!cuda.runtime.launch_attribute_value.nvlink_util_centric_scheduling" },
        .{ cuda.RuntimeLaunchAttributeValueProgrammaticStreamSerializationAllowedType, "!cuda.runtime.launch_attribute_value.programmatic_stream_serialization_allowed" },
    }) |example| {
        const T = example[0];
        const is_type = @hasDecl(T, "type_");
        const parsed = if (is_type) try mlir.Type.parse(ctx, example[1]) else try mlir.Attribute.parse(ctx, example[1]);
        const value = parsed.isA(T) orelse return error.TestUnexpectedResult;
        try std.testing.expect((try rebuildEnums(T, ctx, value)).eql(value));
        var buf: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buf);
        try w.print("{f}", .{value});
        try std.testing.expectEqualStrings(example[1], w.buffered());
    }
    // The compiler prints these without the space before the dictionary; it
    // parses both.
    inline for (.{
        .{ cuda.DeviceAttributesAttr, "#cuda<device_attributes{}>" },
        .{ cuda.FuncAttributesAttr, "#cuda<func_attributes{}>" },
    }) |example| {
        const parsed = try mlir.Attribute.parse(ctx, example[1]);
        const value = parsed.isA(example[0]) orelse return error.TestUnexpectedResult;
        try std.testing.expect((try rebuildEnums(example[0], ctx, value)).eql(value));
    }
}

test "nvvm target rejects what the compiler rejects" {
    const ctx = try testContext();
    defer ctx.deinit();
    try std.testing.expectError(error.InvalidMlir, nvvm.TargetAttr.get(ctx, .{ .O = 4 }));
    try std.testing.expectError(error.InvalidMlir, nvvm.TargetAttr.get(ctx, .{ .chip = "" }));
    const not_strings = try mlir.Attribute.parse(ctx, "[\"a\", 2]");
    try std.testing.expectError(error.InvalidMlir, nvvm.TargetAttr.get(ctx, .{ .link = not_strings }));
}

/// `rebuild` for parameters that are slices of enums.
fn rebuildEnums(comptime T: type, ctx: *mlir.Context, value: *const T) !*const T {
    var args: T.InitArgs = undefined;
    var buffers: [4][16]u32 = undefined;
    inline for (@typeInfo(T.InitArgs).@"struct".field_names, 0..) |field, i| {
        const name = [_]u8{comptime std.ascii.toUpper(field[0])} ++ field[1..];
        const info = @typeInfo(@FieldType(T.InitArgs, field));
        if (info == .pointer and info.pointer.size == .slice and @typeInfo(info.pointer.child) == .@"enum") {
            const E = info.pointer.child;
            const count = @field(T, "getNum" ++ name)(value);
            const elements: []E = @ptrCast(buffers[i][0..count]);
            for (elements, 0..) |*element, pos| element.* = @field(T, "get" ++ name[0 .. name.len - 1])(value, pos);
            @field(args, field) = elements;
        } else {
            @field(args, field) = @field(T, "get" ++ name)(value);
        }
    }
    return T.get(ctx, args);
}

fn rebuild(comptime T: type, ctx: *mlir.Context, value: *const T) !*const T {
    var args: T.InitArgs = undefined;
    var elements: [8]*const mlir.Type = undefined;
    inline for (@typeInfo(T.InitArgs).@"struct".field_names) |field| {
        const name = [_]u8{comptime std.ascii.toUpper(field[0])} ++ field[1..];
        if (@FieldType(T.InitArgs, field) == []const *const mlir.Type) {
            const count = @field(T, "getNum" ++ name)(value);
            for (elements[0..count], 0..) |*element, i| element.* = @field(T, "get" ++ name[0 .. name.len - 1])(value, i);
            @field(args, field) = elements[0..count];
        } else {
            @field(args, field) = @field(T, "get" ++ name)(value);
        }
    }
    return T.get(ctx, args);
}
