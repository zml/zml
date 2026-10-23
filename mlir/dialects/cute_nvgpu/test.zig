const std = @import("std");

const cute_ir = @import("mlir/dialects/cute_ir");
const cute_nvgpu = @import("mlir/dialects/cute_nvgpu");
const mlir = @import("mlir");

const operation_names: []const []const u8 = &.{
    "cute_nvgpu.arch.alloc_rmem",
    "cute_nvgpu.arch.alloc_smem",
    "cute_nvgpu.arch.copy.SM100.copy_s2t",
    "cute_nvgpu.arch.copy.SM100.tma_load",
    "cute_nvgpu.arch.copy.SM100.tma_reduce",
    "cute_nvgpu.arch.copy.SM100.tma_store",
    "cute_nvgpu.arch.copy.SM100.tmem_load",
    "cute_nvgpu.arch.copy.SM100.tmem_store",
    "cute_nvgpu.arch.copy.SM107.tma_load_override",
    "cute_nvgpu.arch.copy.SM107.tma_store_override",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_g2s",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_s2g",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_s2s",
    "cute_nvgpu.arch.get_dyn_smem",
    "cute_nvgpu.arch.get_dyn_smem_size",
    "cute_nvgpu.arch.make_warp_uniform",
    "cute_nvgpu.arch.mma.SM107.umma",
    "cute_nvgpu.arch.mma.SM107.umma_block_scaled",
    "cute_nvgpu.arch.mma.SM107.umma_block_scaled_sparse",
    "cute_nvgpu.arch.mma.SM107.umma_sparse",
    "cute_nvgpu.arch.mma.SM120.block_scaled",
    "cute_nvgpu.arch.prefetch_tma_desc",
    "cute_nvgpu.arch.sm100.alloc_tmem",
    "cute_nvgpu.arch.sm100.dealloc_tmem",
    "cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit",
    "cute_nvgpu.arch.sm100.retrieve_tmem_ptr",
    "cute_nvgpu.atom.get_coord_tensor",
    "cute_nvgpu.atom.get_copy_s2t_smem_desc_view",
    "cute_nvgpu.atom.get_value",
    "cute_nvgpu.atom.make_exec_tma",
    "cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load",
    "cute_nvgpu.atom.make_non_exec_2d_scatter4_tma_store",
    "cute_nvgpu.atom.make_non_exec_im2col_tma_load",
    "cute_nvgpu.atom.make_non_exec_im2col_tma_store",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_load",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_reduce",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_store",
    "cute_nvgpu.atom.make_s2t_copy",
    "cute_nvgpu.atom.make_tma_residue_tensor",
    "cute_nvgpu.atom.make_tmem_copy",
    "cute_nvgpu.atom.set_value",
    "cute_nvgpu.atom.sm107.get_copy_s2t_smem_desc_view",
    "cute_nvgpu.atom.tma_partition",
    "cute_nvgpu.cast_tma_desc_to_integer",
    "cute_nvgpu.copy_tma_desc",
    "cute_nvgpu.get_grid_constant_pointer",
    "cute_nvgpu.get_tma_desc_addr",
    "cute_nvgpu.make_gmma_smem_desc",
    "cute_nvgpu.make_tma_desc_im2col",
    "cute_nvgpu.make_tma_desc_im2col_at",
    "cute_nvgpu.make_tma_desc_tiled",
    "cute_nvgpu.make_tma_desc_tiled_at",
    "cute_nvgpu.make_umma_smem_desc",
    "cute_nvgpu.prefetch_tma_desc",
    "cute_nvgpu.sm103.make_umma_smem_desc_circular",
    "cute_nvgpu.sm107.make_umma_smem_desc",
    "cute_nvgpu.sm107.make_umma_smem_desc_circular",
    "cute_nvgpu.update_tma_desc",
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
    std.testing.refAllDecls(cute_nvgpu);
    inline for (comptime std.meta.declarations(cute_nvgpu)) |decl| {
        const T = @field(cute_nvgpu, decl);
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
        .{ cute_nvgpu.CopyAtomBulkCopyG2SType, "!cute_nvgpu.atom.bulk_copy_g2s<f32 copy_bits = 128 mcast = true>" },
        .{ cute_nvgpu.CopyAtomBulkCopyS2GType, "!cute_nvgpu.atom.bulk_copy_s2g<f32 copy_bits = 128 mask = true>" },
        .{ cute_nvgpu.CopyAtomBulkCopyS2SType, "!cute_nvgpu.atom.bulk_copy_s2s<f32 copy_bits = 128>" },
        .{ cute_nvgpu.CopyAtomDsmemStoreType, "!cute_nvgpu.atom.dsmem_store<f32, 64 b>" },
        .{ cute_nvgpu.CopyAtomG2RType, "!cute_nvgpu.atom.g2r<f32 copy_bits = 64 mem_order = relaxed mem_scope = gpu l2_prefetch_size = size_128b l1_cache_evict_priority = evict_last shared_space = cluster>" },
        .{ cute_nvgpu.CopyAtomIm2ColTmaLoadType, "!cute_nvgpu.atom.im2col_tma_load<f16, copy_bits = 131072, num_cta = 1, g_stride = <\"()\"> mcast = true tma_gbasis = <\"(64,128):(1@1,1@0)\">>" },
        .{ cute_nvgpu.CopyAtomIm2ColTmaStoreType, "!cute_nvgpu.atom.im2col_tma_store<f16, copy_bits = 65536, g_stride = <\"()\"> tma_gbasis = <\"(32,128):(1@1,1@0)\">>" },
        .{ cute_nvgpu.CopyAtomLdsmType, "!cute_nvgpu.atom.ldsm<val_type = f16, mode = <\"(8,8)\">, sz_pattern = u16, num_matrices = 4, t>" },
        .{ cute_nvgpu.CopyAtomNonExec2DGather4TmaLoadType, "!cute_nvgpu.atom.non_exec_2d_gather4_tma_load<sm_100, f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ cute_nvgpu.CopyAtomNonExec2DScatter4TmaStoreType, "!cute_nvgpu.atom.non_exec_2d_scatter4_tma_store<f16, copy_bits = 1024, tma_gbasis = <\"(4,64):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ cute_nvgpu.CopyAtomNonExecIm2ColTmaLoadType, "!cute_nvgpu.atom.non_exec_im2col_tma_load<sm_90, f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ cute_nvgpu.CopyAtomNonExecIm2ColTmaStoreType, "!cute_nvgpu.atom.non_exec_im2col_tma_store<f16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F16_RN>" },
        .{ cute_nvgpu.CopyAtomNonExecTiledTmaLoadType, "!cute_nvgpu.atom.non_exec_tiled_tma_load<sm_90, bf16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = BF16_RN>" },
        .{ cute_nvgpu.CopyAtomNonExecTiledTmaReduceType, "!cute_nvgpu.atom.non_exec_tiled_tma_reduce<ADD, f32, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = F32_RN>" },
        .{ cute_nvgpu.CopyAtomNonExecTiledTmaStoreType, "!cute_nvgpu.atom.non_exec_tiled_tma_store<bf16, copy_bits = 1024, tma_gbasis = <\"(64,1):(1@0,1@1)\">, tma_format = BF16_RN>" },
        .{ cute_nvgpu.CopyAtomR2GType, "!cute_nvgpu.atom.r2g<f16 copy_bits = 128 l1_cache_evict_priority = no_allocate>" },
        .{ cute_nvgpu.CopyAtomR2SType, "!cute_nvgpu.atom.r2s<f32 copy_bits = 64 mem_order = relaxed mem_scope = gpu shared_space = cluster>" },
        .{ cute_nvgpu.CopyAtomS2RType, "!cute_nvgpu.atom.s2r<f32 copy_bits = 64 mem_order = relaxed mem_scope = gpu shared_space = cluster>" },
        .{ cute_nvgpu.CopyAtomSIMTAsyncCopyType, "!cute_nvgpu.atom.simt_async_copy<bf16, cache = global, 128 b>" },
        .{ cute_nvgpu.CopyAtomSIMTMultimemLdReduceType, "!cute_nvgpu.atom.simt_multimem_ld_reduce<bf16, 64 b, MIN, <F32>, acquire, gpu>" },
        .{ cute_nvgpu.CopyAtomSIMTMultimemRedType, "!cute_nvgpu.atom.simt_multimem_red<f16, 64 b, MAX, release, gpu>" },
        .{ cute_nvgpu.CopyAtomSIMTMultimemStType, "!cute_nvgpu.atom.simt_multimem_st<f16, 64 b, release, gpu>" },
        .{ cute_nvgpu.CopyAtomSIMTSyncCopyType, "!cute_nvgpu.atom.universal_copy<f32, 128 b, src_space = gmem, dst_space = smem>" },
        .{ cute_nvgpu.CopyAtomSM100CopyS2TType, "!cute_nvgpu.atom.s2t_copy<f8E4M3FN, 32 DP, 128 bit, 1 cta, x4>" },
        .{ cute_nvgpu.CopyAtomSM100S2TCopyV2Type, "!cute_nvgpu.atom.sm100_s2t_copy_v2<f8E4M3FN, num_dp = 32, num_bit = 128, num_cta = 1, smem_major = k, broadcast = x4>" },
        .{ cute_nvgpu.CopyAtomSM100TmemLoadType, "!cute_nvgpu.atom.tmem_load<f32, 32 DP, 32 bit, x4, pack16b>" },
        .{ cute_nvgpu.CopyAtomSM100TmemStoreType, "!cute_nvgpu.atom.tmem_store<f32, 32 DP, 32 bit, x4, expand16b>" },
        .{ cute_nvgpu.CopyAtomSM107TmemLoadSPCompressType, "!cute_nvgpu.atom.tmem_load_spcompress<f32, 32 DP, 32 bit, x4, maxabs, red, nan>" },
        .{ cute_nvgpu.CopyAtomSM10xTmemLoadRedType, "!cute_nvgpu.atom.tmem_load_red<f32, 16 DP, 32 bit, x4, maxabs, nan, half_split_off=3 : i64>" },
        .{ cute_nvgpu.CopyAtomStsmType, "!cute_nvgpu.atom.stsm<f16, mode = <\"(8,8)\">, num_matrices = 4, n>" },
        .{ cute_nvgpu.CopyAtomTmaLoadType, "!cute_nvgpu.atom.tma_load<bf16, copy_bits = 1024, mode = tiled, num_cta = 2, g_stride = <\"()\"> mcast = true tma_gbasis = <\"(64,1):(1@0,1@1)\">>" },
        .{ cute_nvgpu.CopyAtomTmaReduceType, "!cute_nvgpu.atom.tma_reduce<f32, copy_bits = 131072, mode = tiled, kind = ADD, g_stride = <\"()\"> tma_gbasis = <\"(32,128):(1@1,1@0)\">>" },
        .{ cute_nvgpu.CopyAtomTmaStoreType, "!cute_nvgpu.atom.tma_store<bf16, copy_bits = 16384, mode = tiled, g_stride = <\"()\"> tma_gbasis = <\"(64,16):(1@0,1@1)\">>" },
        .{ cute_nvgpu.MmaAtomSM100UMMABlockScaledSparseType, "!cute_nvgpu.sm100.mma_bs_sp<128x128x64, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, sparse_metadata_format = tid, frag_kind = ss, vec_size = 32>" },
        .{ cute_nvgpu.MmaAtomSM100UMMABlockScaledType, "!cute_nvgpu.sm100.mma_bs<128x128x96, num_cta = 1, ab_major = (k, k), elem_type = (f4E2M1FN, f4E2M1FN, f32), sf_type = f8E8M0FNU, frag_kind = ss, vec_size = 16, arch_promote = sm_103>" },
        .{ cute_nvgpu.MmaAtomSM100UMMASparseType, "!cute_nvgpu.sm100.mma_sp<128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f16, f16, f32), e_type = i8, sparse_metadata_format = tid, frag_kind = ss, c_scale_exp = 0>" },
        .{ cute_nvgpu.MmaAtomSM100UMMAType, "!cute_nvgpu.sm100.mma<128x128x16, num_cta = 1, ab_major = (k, mn), elem_type = (f16, f16, f32), frag_kind = ss, c_scale_exp = 0>" },
        .{ cute_nvgpu.MmaAtomSM107UMMABlockScaledSparseType, "!cute_nvgpu.sm107.mma_bs_sp<128x128x64, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, sparse_metadata_format = tid, frag_kind = ss, vec_size = 32, ab_collector_op = (discard, discard)>" },
        .{ cute_nvgpu.MmaAtomSM107UMMABlockScaledType, "!cute_nvgpu.sm107.mma_bs<128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f8E4M3FN, f8E4M3FN, f32), sf_type = f8E8M0FNU, frag_kind = ss, vec_size = 32, ab_collector_op = (discard, use), int_overflow = <satfinite>>" },
        .{ cute_nvgpu.MmaAtomSM107UMMASparseType, "!cute_nvgpu.sm107.mma_sp<128x128x32, num_cta = 1, ab_major = (k, k), elem_type = (f16, f16, f32), e_type = i8, sparse_metadata_format = tid, frag_kind = ss, c_scale_exp = 0, ab_collector_op = (discard, discard)>" },
        .{ cute_nvgpu.MmaAtomSM107UMMAType, "!cute_nvgpu.sm107.mma<128x128x16, num_cta = 1, ab_major = (k, mn), elem_type = (f16, f16, f32), frag_kind = ss, c_scale_exp = 0, ab_collector_op = (lastuse, fill)>" },
        .{ cute_nvgpu.MmaAtomSM120BlockScaledType, "!cute_nvgpu.SM120.mma_bs<16x8x64, vec_size = 32, elem_type = (f4E2M1FN, f4E2M1FN, f32), sf_type = f8E8M0FNU, use_sf_layout_TV = true>" },
        .{ cute_nvgpu.MmaAtomSM80SparseType, "!cute_nvgpu.sm80.sparse_mma<16x8x64, elem_type = (i8, i8, i32), md_format = tid, int_overflow = <satfinite> >" },
        .{ cute_nvgpu.MmaAtomSM80Type, "!cute_nvgpu.sm80.mma<16x8x32, elem_type = (i8, i8, i32), int_overflow = <wrapped>, binary_op = <none> >" },
        .{ cute_nvgpu.MmaAtomSM89Type, "!cute_nvgpu.sm89.mma<16x8x32, elem_type = (f8E4M3FN, f8E4M3FN, f32) >" },
        .{ cute_nvgpu.MmaAtomSM90Type, "!cute_nvgpu.sm90.mma<64x128x32, ab_major = (k, k), elem_type = (i8, i8, i32), frag_kind = ss, int_overflow = <satfinite>, a_neg, b_neg>" },
        .{ cute_nvgpu.SmemDescCircularSM103Type, "!cute_nvgpu.sm103.smem_desc_circular<<\"(8,128,2):(128,1,1024)\">>" },
        .{ cute_nvgpu.SmemDescCircularSM107Type, "!cute_nvgpu.sm107.smem_desc_circular<<\"(8,128,2):(128,1,1024)\">>" },
        .{ cute_nvgpu.SmemDescSM107Type, "!cute_nvgpu.sm107.smem_desc" },
        .{ cute_nvgpu.SmemDescType, "!cute_nvgpu.smem_desc" },
        .{ cute_nvgpu.SmemDescViewType, "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(2,3):(1,2)\">" },
        .{ cute_nvgpu.TmaDescriptorIm2ColType, "!cute_nvgpu.tma_descriptor_im2col" },
        .{ cute_nvgpu.TmaDescriptorTiledType, "!cute_nvgpu.tma_descriptor_tiled" },
        .{ cute_nvgpu.UniversalFmaAtomType, "!cute_nvgpu.atom.universal_fma<1x1x1, (f32, f32) -> f32 >" },
        .{ cute_nvgpu.WorkIdResponseType, "!cute_nvgpu.workid_response" },
        .{ cute_nvgpu.NotImplementedFrgAttr, "#cute_nvgpu.not_implemented_frg<>" },
        .{ cute_nvgpu.RmemFrgAttr, "#cute_nvgpu.rmem_frg<value_type = f32, operand = A, derive_elem_type = true>" },
        .{ cute_nvgpu.SM100CircularSmemFrgAttr, "#cute_nvgpu.arch.sm100.circular_smem_frg<major = k>" },
        .{ cute_nvgpu.SM100SmemFrgAttr, "#cute_nvgpu.arch.sm100.smem_frg<major = k>" },
        .{ cute_nvgpu.SM100TmemEFrgAttr, "#cute_nvgpu.arch.sm100.tmem_e_frg<a_type = f16, e_type = i8>" },
        .{ cute_nvgpu.SM100TmemFrgAttr, "#cute_nvgpu.arch.sm100.tmem_frg<data_type = f32, storage_type = f32, cta_group = 1, tmem_alloc_mode = Interleaved>" },
        .{ cute_nvgpu.SM100TmemSfFrgAttr, "#cute_nvgpu.arch.sm100.tmem_sf_frg<sf_type = f8E8M0FNU, sf_vec_size = 32, cta_group = 1, is_sfa = true, tmem_alloc_mode = Interleaved>" },
        .{ cute_nvgpu.SM107SmemFrgAttr, "#cute_nvgpu.arch.sm107.smem_frg<major = k>" },
        .{ cute_nvgpu.SM90SmemFrgAttr, "#cute_nvgpu.arch.sm90.smem_frg<major = k>" },
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
    inline for (.{cute_nvgpu}) |namespace| {
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
    try std.testing.expect((try cute_nvgpu.SmemDescType.get(ctx, .{})).type_().isA(cute_nvgpu.NVGPUType) != null);
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
