const std = @import("std");

const mlir = @import("mlir");
const nvvm = @import("mlir/dialects/nvvm");

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(nvvm);
    std.testing.refAllDecls(nvvm.ops);
    inline for (comptime std.meta.declarations(nvvm.ops)) |decl| {
        const T = @field(nvvm.ops, decl);
        if (@TypeOf(T) == type) std.testing.refAllDecls(T);
    }
}

test "every enum attribute prints as the compiler does" {
    @setEvalBranchQuota(1_000_000);
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    nvvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    inline for (.{
        .{ nvvm.ops.AtomicOpKindAttr, "#nvvm<atomic_op {s}>" },
        .{ nvvm.ops.BarrierReductionAttr, "#nvvm.reduction<{s}>" },
        .{ nvvm.ops.BarrierReduxKindAttr, "#nvvm.barrier_redux_kind<{s}>" },
        .{ nvvm.ops.BlockScaleFormatAttr, "#nvvm.block_scale_format<{s}>" },
        .{ nvvm.ops.CTAGroupKindAttr, "#nvvm.cta_group<{s}>" },
        .{ nvvm.ops.CVTPackFloatKindAttr, "#nvvm.packfloat_type<{s}>" },
        .{ nvvm.ops.CacheEvictionPriorityAttr, "#nvvm<cache_eviction_priority {s}>" },
        .{ nvvm.ops.ClusterLaunchControlQueryTypeAttr, "#nvvm<cluster_launch_control_query_type {s}>" },
        .{ nvvm.ops.CompareOpKindAttr, "#nvvm.op<{s}>" },
        .{ nvvm.ops.ConvertFP4TypeAttr, "#nvvm.convert_fp4_type<{s}>" },
        .{ nvvm.ops.ConvertFP8TypeAttr, "#nvvm.convert_fp8_type<{s}>" },
        .{ nvvm.ops.ConvertScaleKindAttr, "#nvvm.convert_scale_kind<{s}>" },
        .{ nvvm.ops.DotAccumulateTypeAttr, "#nvvm.dot_accumulate_type<{s}>" },
        .{ nvvm.ops.EvictKindAttr, "#nvvm.evict_kind<{s}>" },
        .{ nvvm.ops.FPRoundingModeAttr, "#nvvm.fp_rnd_mode<{s}>" },
        .{ nvvm.ops.GridDepActionKindAttr, "#nvvm<grid_dep_action {s}>" },
        .{ nvvm.ops.IntegerRoundingModeAttr, "#nvvm.int_rnd_mode<{s}>" },
        .{ nvvm.ops.L2PrefetchSizeAttr, "#nvvm.l2_prefetch<{s}>" },
        .{ nvvm.ops.LdStMatrixEltTypeAttr, "#nvvm.ld_st_matrix_elt_type<{s}>" },
        .{ nvvm.ops.LoadCacheModifierExtKindAttr, "#nvvm.load_cache_modifier_ext<{s}>" },
        .{ nvvm.ops.LoadCacheModifierKindAttr, "#nvvm<load_cache_modifier {s}>" },
        .{ nvvm.ops.LoadShapeAttr, "#nvvm.load_shape<{s}>" },
        .{ nvvm.ops.LoadSrcFormatAttr, "#nvvm.load_src_format<{s}>" },
        .{ nvvm.ops.MBarrierLayoutAttr, "#nvvm.mbarrier_layout<{s}>" },
        .{ nvvm.ops.MBarrierPhaseAttr, "#nvvm.mbarrier_phase<{s}>" },
        .{ nvvm.ops.MBarrierScopeKindAttr, "#nvvm.mbar_scope<{s}>" },
        .{ nvvm.ops.MBarrierSpaceKindAttr, "#nvvm.mbar_space<{s}>" },
        .{ nvvm.ops.MBarrierTxnKindAttr, "#nvvm.mbar_txn_kind<{s}>" },
        .{ nvvm.ops.MBarrierWaitKindAttr, "#nvvm.mbar_wait<{s}>" },
        .{ nvvm.ops.MMAB1OpAttr, "#nvvm.mma_b1op<{s}>" },
        .{ nvvm.ops.MMABlockScaleKindAttr, "#nvvm.block_scale_kind<{s}>" },
        .{ nvvm.ops.MMACtaCountAttr, "#nvvm.mma_cta_count<{s}>" },
        .{ nvvm.ops.MMAFragAttr, "#nvvm.mma_frag<{s}>" },
        .{ nvvm.ops.MMAIntOverflowAttr, "#nvvm.mma_int_overflow<{s}>" },
        .{ nvvm.ops.MMAKindAttr, "#nvvm.mma_kind<{s}>" },
        .{ nvvm.ops.MMALayoutAttr, "#nvvm.mma_layout<{s}>" },
        .{ nvvm.ops.MMATypesAttr, "#nvvm.mma_type<{s}>" },
        .{ nvvm.ops.MatchSyncKindAttr, "#nvvm<match_sync_kind {s}>" },
        .{ nvvm.ops.MemOrderKindAttr, "#nvvm.mem_order<{s}>" },
        .{ nvvm.ops.MemScopeKindAttr, "#nvvm.mem_scope<{s}>" },
        .{ nvvm.ops.MulModeAttr, "#nvvm<mul_mode {s}>" },
        .{ nvvm.ops.NVVMMemorySpaceAttr, "#nvvm.memory_space<{s}>" },
        .{ nvvm.ops.PermuteModeAttr, "#nvvm.permute_mode<{s}>" },
        .{ nvvm.ops.PrefetchCacheLevelAttr, "#nvvm<prefetch_cache_level {s}>" },
        .{ nvvm.ops.ProxyKindAttr, "#nvvm.proxy_kind<{s}>" },
        .{ nvvm.ops.ReductionKindAttr, "#nvvm<reduction_kind {s}>" },
        .{ nvvm.ops.ReductionOpAttr, "#nvvm.red_op<{s}>" },
        .{ nvvm.ops.ReductionTypeAttr, "#nvvm.red_type<{s}>" },
        .{ nvvm.ops.SPCompressElemSizeAttr, "#nvvm.spcomp_elem_size<{s}>" },
        .{ nvvm.ops.SPCompressFactorTypeAttr, "#nvvm.spcomp_factor<{s}>" },
        .{ nvvm.ops.SPCompressIndexSizeAttr, "#nvvm.spcomp_index_size<{s}>" },
        .{ nvvm.ops.SPCompressOpKindAttr, "#nvvm.spcomp_op_kind<{s}>" },
        .{ nvvm.ops.SPCompressRepFactorAttr, "#nvvm.spcomp_rep_factor<{s}>" },
        .{ nvvm.ops.SPDecompressElemSizeAttr, "#nvvm.spdecomp_elem_size<{s}>" },
        .{ nvvm.ops.SPDecompressFactorTypeAttr, "#nvvm.spdecomp_factor<{s}>" },
        .{ nvvm.ops.SPDecompressIndexSizeAttr, "#nvvm.spdecomp_index_size<{s}>" },
        .{ nvvm.ops.SPDecompressRepFactorAttr, "#nvvm.spdecomp_rep_factor<{s}>" },
        .{ nvvm.ops.SaturationModeAttr, "#nvvm.sat_mode<{s}>" },
        .{ nvvm.ops.SaturationModeKindAttr, "#nvvm.sat<{s}>" },
        .{ nvvm.ops.ScaleVecSizeAttr, "#nvvm.scale_vec_size<{s}>" },
        .{ nvvm.ops.SetMaxRegisterActionAttr, "#nvvm<action {s}>" },
        .{ nvvm.ops.SharedSpaceAttr, "#nvvm.shared_space<{s}>" },
        .{ nvvm.ops.ShflKindAttr, "#nvvm<shfl_kind {s}>" },
        .{ nvvm.ops.SparsityFormatAttr, "#nvvm.sparsity_format<{s}>" },
        .{ nvvm.ops.StateSpaceAttr, "#nvvm.state_space<{s}>" },
        .{ nvvm.ops.StoreCacheModifierKindAttr, "#nvvm.store_cache_modifier<{s}>" },
        .{ nvvm.ops.StoreShapeAttr, "#nvvm.store_shape<{s}>" },
        .{ nvvm.ops.TCBarParamAttr, "#nvvm.TCBarParam<{s}>" },
        .{ nvvm.ops.TMALoadModeAttr, "#nvvm.tma_load_mode<{s}>" },
        .{ nvvm.ops.TMAReduxKindAttr, "#nvvm.tma_redux_kind<{s}>" },
        .{ nvvm.ops.TMAStoreModeAttr, "#nvvm.tma_store_mode<{s}>" },
        .{ nvvm.ops.Tcgen05CpMulticastAttr, "#nvvm.tcgen05_cp_multicast<{s}>" },
        .{ nvvm.ops.Tcgen05CpShapeAttr, "#nvvm.tcgen05_cp_shape<{s}>" },
        .{ nvvm.ops.Tcgen05CpSrcFormatAttr, "#nvvm.tcgen05_cp_src_fmt<{s}>" },
        .{ nvvm.ops.Tcgen05FenceKindAttr, "#nvvm.tcgen05_fence<{s}>" },
        .{ nvvm.ops.Tcgen05LdStShapeAttr, "#nvvm.tcgen05_ldst_shape<{s}>" },
        .{ nvvm.ops.Tcgen05MMABlockScaleAttr, "#nvvm.tcgen05_mma_block_scale<{s}>" },
        .{ nvvm.ops.Tcgen05MMACollectorBBufferAttr, "#nvvm.tcgen05_mma_collectorb<{s}>" },
        .{ nvvm.ops.Tcgen05MMACollectorOpAttr, "#nvvm.tcgen05_mma_collectorop<{s}>" },
        .{ nvvm.ops.Tcgen05MMAKindAttr, "#nvvm.tcgen05_mma_kind<{s}>" },
        .{ nvvm.ops.Tcgen05WaitKindAttr, "#nvvm.tcgen05_wait<{s}>" },
        .{ nvvm.ops.TensormapElemtypeAttr, "#nvvm.tensormap_elemtype<{s}>" },
        .{ nvvm.ops.TensormapFieldAttr, "#nvvm<tensormap_field {s}>" },
        .{ nvvm.ops.TensormapFillModeAttr, "#nvvm.tensormap_fill_mode<{s}>" },
        .{ nvvm.ops.TensormapInterleaveLayoutAttr, "#nvvm.tensormap_interleave_layout<{s}>" },
        .{ nvvm.ops.TensormapSwizzleAtomicityAttr, "#nvvm.tensormap_swizzle_atomicity<{s}>" },
        .{ nvvm.ops.TensormapSwizzleModeAttr, "#nvvm.tensormap_swizzle_mode<{s}>" },
        .{ nvvm.ops.TmemLayoutAttr, "#nvvm.TmemLayout<{s}>" },
        .{ nvvm.ops.ValidatePatternAttr, "#nvvm.validate_pattern<{s}>" },
        .{ nvvm.ops.VoteSyncKindAttr, "#nvvm<vote_sync_kind {s}>" },
        .{ nvvm.ops.WGMMAScaleInAttr, "#nvvm.wgmma_scale_in<{s}>" },
        .{ nvvm.ops.WGMMAScaleOutAttr, "#nvvm.wgmma_scale_out<{s}>" },
        .{ nvvm.ops.WGMMATypesAttr, "#nvvm.wgmma_type<{s}>" },
    }) |example| {
        const T = example[0];
        const E = @FieldType(T.InitArgs, "value");
        inline for (@typeInfo(E).@"enum".field_names) |name| {
            const attr = try T.get(ctx, .{ .value = @field(E, name) });
            try std.testing.expectEqual(@field(E, name), attr.value());
            var buf: [256]u8 = undefined;
            var w: std.Io.Writer = .fixed(&buf);
            try w.print("{f}", .{attr});
            try std.testing.expectEqualStrings(std.fmt.comptimePrint(example[1], .{name}), w.buffered());
        }
    }
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    nvvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

test "every bound operation is registered" {
    const ctx = try testContext();
    defer ctx.deinit();
    for (nvvm.operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}

test "nvvm types and attributes print as the compiler does" {
    @setEvalBranchQuota(100_000);
    const ctx = try testContext();
    defer ctx.deinit();
    // The compiler's printed forms, which these must print byte for byte.
    inline for (.{
        .{ nvvm.ops.MMAShapeAttr, "#nvvm.shape<m = 16, n = 8, k = 16>" },
        .{ nvvm.ops.LdStMatrixShapeAttr, "#nvvm.ld_st_matrix_shape<m = 8, n = 8>" },
        .{ nvvm.ops.TargetAttr, "#nvvm.target" },
        .{ nvvm.ops.TargetAttr, "#nvvm.target<verifyTarget = true>" },
        .{ nvvm.ops.TargetAttr, "#nvvm.target<O = 3, triple = \"t\", chip = \"sm_90\", features = \"f\", flags = {x}, link = [\"a\"]>" },
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
}

test "nvvm target rejects what the compiler rejects" {
    const ctx = try testContext();
    defer ctx.deinit();
    try std.testing.expectError(error.InvalidMlir, nvvm.ops.TargetAttr.get(ctx, .{ .O = 4 }));
    try std.testing.expectError(error.InvalidMlir, nvvm.ops.TargetAttr.get(ctx, .{ .chip = "" }));
    const not_strings = try mlir.Attribute.parse(ctx, "[\"a\", 2]");
    try std.testing.expectError(error.InvalidMlir, nvvm.ops.TargetAttr.get(ctx, .{ .link = not_strings }));
}

/// `rebuild` for parameters that are slices of enums.
fn rebuildEnums(comptime T: type, ctx: *mlir.Context, value: *const T) !*const T {
    var args: T.InitArgs = undefined;
    var buffers: [4][16]u32 = undefined;
    inline for (@typeInfo(T.InitArgs).@"struct".field_names, 0..) |field, i| {
        const name = [_]u8{comptime std.ascii.toUpper(field[0])} ++ field[1..];
        const noun = [_]u8{comptime std.ascii.toLower(field[0])} ++ field[1..];
        const accessor = if (@hasDecl(T, noun ++ "_")) noun ++ "_" else noun;
        const info = @typeInfo(@FieldType(T.InitArgs, field));
        if (info == .pointer and info.pointer.size == .slice and @typeInfo(info.pointer.child) == .@"enum") {
            const E = info.pointer.child;
            const count = @field(T, "num" ++ name)(value);
            const elements: []E = @ptrCast(buffers[i][0..count]);
            for (elements, 0..) |*element, pos| element.* = @field(T, noun[0 .. noun.len - 1])(value, pos);
            @field(args, field) = elements;
        } else {
            @field(args, field) = @field(T, accessor)(value);
        }
    }
    return T.get(ctx, args);
}
