const std = @import("std");

const mlir = @import("mlir");
const nvvm = @import("mlir/dialects/nvvm");

const operation_names: []const []const u8 = &.{
    "nvvm.add.packed.bf16x2.f32x2.f32x2",
    "nvvm.add.packed.f16x2.f32x2.f32x2",
    "nvvm.add.packed.f32x2",
    "nvvm.add.packed.f32x2.bf16x2.f32x2",
    "nvvm.add.packed.f32x2.f16x2.f32x2",
    "nvvm.addf",
    "nvvm.applypriority.async.bulk",
    "nvvm.applypriority.async.bulk.tensor",
    "nvvm.applypriority.async.bulk.tensor.override",
    "nvvm.atomicrmw",
    "nvvm.bar.warp.sync",
    "nvvm.barrier",
    "nvvm.barrier.arrive",
    "nvvm.barrier.cta.arrive",
    "nvvm.barrier.cta.red",
    "nvvm.barrier.cta.sync",
    "nvvm.barrier0",
    "nvvm.breakpoint",
    "nvvm.clmad",
    "nvvm.cluster.arrive",
    "nvvm.cluster.arrive.relaxed",
    "nvvm.cluster.wait",
    "nvvm.clusterlaunchcontrol.query.cancel",
    "nvvm.clusterlaunchcontrol.try.cancel",
    "nvvm.compare.and.set",
    "nvvm.convert.and.pack.integer",
    "nvvm.convert.bf16x2.to.f4x2",
    "nvvm.convert.bf16x2.to.f6x2",
    "nvvm.convert.bf16x2.to.f8x2",
    "nvvm.convert.bf16x2.to.s2f6x2",
    "nvvm.convert.f16x2.to.f4x2",
    "nvvm.convert.f16x2.to.f6x2",
    "nvvm.convert.f16x2.to.f8x2",
    "nvvm.convert.f32x2.to.bf16x2",
    "nvvm.convert.f32x2.to.f16x2",
    "nvvm.convert.f32x2.to.f4x2",
    "nvvm.convert.f32x2.to.f6x2",
    "nvvm.convert.f32x2.to.f8x2",
    "nvvm.convert.f32x2.to.s2f6x2",
    "nvvm.convert.f32x4.to.f4x4",
    "nvvm.convert.f32x4.to.f6x4",
    "nvvm.convert.f32x4.to.f8x4",
    "nvvm.convert.f4x2.to.f16x2",
    "nvvm.convert.f6x2.to.f16x2",
    "nvvm.convert.f8x2.to.bf16x2",
    "nvvm.convert.f8x2.to.f16x2",
    "nvvm.convert.float.to.integer",
    "nvvm.convert.float.to.tf32",
    "nvvm.convert.s2f6x2.to.bf16x2",
    "nvvm.cos",
    "nvvm.cp.async.bulk.commit.group",
    "nvvm.cp.async.bulk.global.shared.cta",
    "nvvm.cp.async.bulk.prefetch",
    "nvvm.cp.async.bulk.shared.cluster.global",
    "nvvm.cp.async.bulk.shared.cluster.shared.cta",
    "nvvm.cp.async.bulk.tensor.global.shared.cta",
    "nvvm.cp.async.bulk.tensor.global.shared.cta.override",
    "nvvm.cp.async.bulk.tensor.prefetch",
    "nvvm.cp.async.bulk.tensor.prefetch.override",
    "nvvm.cp.async.bulk.tensor.reduce",
    "nvvm.cp.async.bulk.tensor.reduce.override",
    "nvvm.cp.async.bulk.tensor.shared.cluster.global",
    "nvvm.cp.async.bulk.tensor.shared.cluster.global.override",
    "nvvm.cp.async.bulk.wait_group",
    "nvvm.cp.async.commit.group",
    "nvvm.cp.async.mbarrier.arrive",
    "nvvm.cp.async.shared.global",
    "nvvm.cp.async.wait.group",
    "nvvm.cvt.packfloat",
    "nvvm.cvt.packfloat.f32",
    "nvvm.cvt.to.f4x8.packed",
    "nvvm.dot.accumulate.2way",
    "nvvm.dot.accumulate.4way",
    "nvvm.elect.sync",
    "nvvm.ex2",
    "nvvm.exit",
    "nvvm.fabs",
    "nvvm.fence.mbarrier.init",
    "nvvm.fence.proxy",
    "nvvm.fence.proxy.acquire",
    "nvvm.fence.proxy.release",
    "nvvm.fence.proxy.sync_restrict",
    "nvvm.fence.sc.cluster",
    "nvvm.fence.sync_restrict",
    "nvvm.fma",
    "nvvm.fma.packed.f32x2",
    "nvvm.fma.packed.f32x2.bf16x2.f32x2.f32x2",
    "nvvm.fma.packed.f32x2.f16x2.f32x2.f32x2",
    "nvvm.fmax",
    "nvvm.fmin",
    "nvvm.griddepcontrol",
    "nvvm.inline_ptx",
    "nvvm.ldmatrix",
    "nvvm.load.ext",
    "nvvm.log2",
    "nvvm.mapa",
    "nvvm.match.sync",
    "nvvm.mbarrier.arrive",
    "nvvm.mbarrier.arrive.expect_tx",
    "nvvm.mbarrier.arrive.nocomplete",
    "nvvm.mbarrier.arrive_drop",
    "nvvm.mbarrier.arrive_drop.expect_tx",
    "nvvm.mbarrier.arrive_drop.nocomplete",
    "nvvm.mbarrier.check.layout",
    "nvvm.mbarrier.complete_tx",
    "nvvm.mbarrier.expect_tx",
    "nvvm.mbarrier.init",
    "nvvm.mbarrier.inval",
    "nvvm.mbarrier.test.wait",
    "nvvm.mbarrier.try_wait",
    "nvvm.mbarrier.try_wait.parity",
    "nvvm.mbarrier.try_wait.parity.timelimit",
    "nvvm.mbarrier.try_wait.timelimit",
    "nvvm.mbarrier.txn",
    "nvvm.mbarrier.txn.cta",
    "nvvm.mbarrier.wait",
    "nvvm.mbarrier.wait.parity",
    "nvvm.memory.barrier",
    "nvvm.mma.block_scale",
    "nvvm.mma.block_scale.internal",
    "nvvm.mma.sp.block_scale",
    "nvvm.mma.sp.block_scale.internal",
    "nvvm.mma.sp.sync",
    "nvvm.mma.sync",
    "nvvm.mma_smem_desc",
    "nvvm.movmatrix",
    "nvvm.mul",
    "nvvm.mul.packed.bf16x2.bf16x2.f16x2",
    "nvvm.mul.packed.bf16x2.f32x2.f32x2",
    "nvvm.mul.packed.f16x2.f16x2.bf16x2",
    "nvvm.mul.packed.f16x2.f32x2.f32x2",
    "nvvm.mul.packed.f32x2",
    "nvvm.multimem.ld.reduce",
    "nvvm.multimem.red",
    "nvvm.multimem.st",
    "nvvm.nanosleep",
    "nvvm.pmevent",
    "nvvm.prefetch",
    "nvvm.prmt",
    "nvvm.rcp.approx.ftz.f",
    "nvvm.read.ptx.sreg.aggr.smem.size",
    "nvvm.read.ptx.sreg.clock",
    "nvvm.read.ptx.sreg.clock64",
    "nvvm.read.ptx.sreg.cluster.ctaid.x",
    "nvvm.read.ptx.sreg.cluster.ctaid.y",
    "nvvm.read.ptx.sreg.cluster.ctaid.z",
    "nvvm.read.ptx.sreg.cluster.ctarank",
    "nvvm.read.ptx.sreg.cluster.nctaid.x",
    "nvvm.read.ptx.sreg.cluster.nctaid.y",
    "nvvm.read.ptx.sreg.cluster.nctaid.z",
    "nvvm.read.ptx.sreg.cluster.nctarank",
    "nvvm.read.ptx.sreg.clusterid.x",
    "nvvm.read.ptx.sreg.clusterid.y",
    "nvvm.read.ptx.sreg.clusterid.z",
    "nvvm.read.ptx.sreg.ctaid.x",
    "nvvm.read.ptx.sreg.ctaid.y",
    "nvvm.read.ptx.sreg.ctaid.z",
    "nvvm.read.ptx.sreg.dynamic.smem.size",
    "nvvm.read.ptx.sreg.envreg0",
    "nvvm.read.ptx.sreg.envreg1",
    "nvvm.read.ptx.sreg.envreg10",
    "nvvm.read.ptx.sreg.envreg11",
    "nvvm.read.ptx.sreg.envreg12",
    "nvvm.read.ptx.sreg.envreg13",
    "nvvm.read.ptx.sreg.envreg14",
    "nvvm.read.ptx.sreg.envreg15",
    "nvvm.read.ptx.sreg.envreg16",
    "nvvm.read.ptx.sreg.envreg17",
    "nvvm.read.ptx.sreg.envreg18",
    "nvvm.read.ptx.sreg.envreg19",
    "nvvm.read.ptx.sreg.envreg2",
    "nvvm.read.ptx.sreg.envreg20",
    "nvvm.read.ptx.sreg.envreg21",
    "nvvm.read.ptx.sreg.envreg22",
    "nvvm.read.ptx.sreg.envreg23",
    "nvvm.read.ptx.sreg.envreg24",
    "nvvm.read.ptx.sreg.envreg25",
    "nvvm.read.ptx.sreg.envreg26",
    "nvvm.read.ptx.sreg.envreg27",
    "nvvm.read.ptx.sreg.envreg28",
    "nvvm.read.ptx.sreg.envreg29",
    "nvvm.read.ptx.sreg.envreg3",
    "nvvm.read.ptx.sreg.envreg30",
    "nvvm.read.ptx.sreg.envreg31",
    "nvvm.read.ptx.sreg.envreg4",
    "nvvm.read.ptx.sreg.envreg5",
    "nvvm.read.ptx.sreg.envreg6",
    "nvvm.read.ptx.sreg.envreg7",
    "nvvm.read.ptx.sreg.envreg8",
    "nvvm.read.ptx.sreg.envreg9",
    "nvvm.read.ptx.sreg.globaltimer",
    "nvvm.read.ptx.sreg.globaltimer.lo",
    "nvvm.read.ptx.sreg.gridid",
    "nvvm.read.ptx.sreg.laneid",
    "nvvm.read.ptx.sreg.lanemask.eq",
    "nvvm.read.ptx.sreg.lanemask.ge",
    "nvvm.read.ptx.sreg.lanemask.gt",
    "nvvm.read.ptx.sreg.lanemask.le",
    "nvvm.read.ptx.sreg.lanemask.lt",
    "nvvm.read.ptx.sreg.nclusterid.x",
    "nvvm.read.ptx.sreg.nclusterid.y",
    "nvvm.read.ptx.sreg.nclusterid.z",
    "nvvm.read.ptx.sreg.nctaid.x",
    "nvvm.read.ptx.sreg.nctaid.y",
    "nvvm.read.ptx.sreg.nctaid.z",
    "nvvm.read.ptx.sreg.nsmid",
    "nvvm.read.ptx.sreg.ntid.x",
    "nvvm.read.ptx.sreg.ntid.y",
    "nvvm.read.ptx.sreg.ntid.z",
    "nvvm.read.ptx.sreg.nwarpid",
    "nvvm.read.ptx.sreg.smid",
    "nvvm.read.ptx.sreg.tid.x",
    "nvvm.read.ptx.sreg.tid.y",
    "nvvm.read.ptx.sreg.tid.z",
    "nvvm.read.ptx.sreg.total.smem.size",
    "nvvm.read.ptx.sreg.warpid",
    "nvvm.read.ptx.sreg.warpsize",
    "nvvm.red",
    "nvvm.redux.sync",
    "nvvm.rsqrt",
    "nvvm.setmaxregister",
    "nvvm.shfl.sync",
    "nvvm.sin",
    "nvvm.spcompress",
    "nvvm.spdecompress",
    "nvvm.st.bulk",
    "nvvm.stmatrix",
    "nvvm.store.async",
    "nvvm.store.ext",
    "nvvm.sub.packed.bf16x2.f32x2.f32x2",
    "nvvm.sub.packed.f16x2.f32x2.f32x2",
    "nvvm.sub.packed.f32x2",
    "nvvm.sub.packed.f32x2.bf16x2.f32x2",
    "nvvm.sub.packed.f32x2.f16x2.f32x2",
    "nvvm.subf",
    "nvvm.tcgen05.alloc",
    "nvvm.tcgen05.commit",
    "nvvm.tcgen05.cp",
    "nvvm.tcgen05.dealloc",
    "nvvm.tcgen05.fence",
    "nvvm.tcgen05.ld",
    "nvvm.tcgen05.ld.red",
    "nvvm.tcgen05.ld.red.spcompress",
    "nvvm.tcgen05.ld.spcompress",
    "nvvm.tcgen05.mma",
    "nvvm.tcgen05.mma.block_scale",
    "nvvm.tcgen05.mma.block_scale.decompress_b",
    "nvvm.tcgen05.mma.decompress_b",
    "nvvm.tcgen05.mma.sp",
    "nvvm.tcgen05.mma.sp.block_scale",
    "nvvm.tcgen05.mma.ws",
    "nvvm.tcgen05.mma.ws.sp",
    "nvvm.tcgen05.mma_smem_desc",
    "nvvm.tcgen05.relinquish_alloc_permit",
    "nvvm.tcgen05.shift",
    "nvvm.tcgen05.st",
    "nvvm.tcgen05.wait",
    "nvvm.tcgen05_mma_smem_desc_v2",
    "nvvm.tensormap.cp_fenceproxy",
    "nvvm.tensormap.replace",
    "nvvm.trace.mark",
    "nvvm.vote.sync",
    "nvvm.wgmma.commit.group.sync.aligned",
    "nvvm.wgmma.fence.aligned",
    "nvvm.wgmma.mma_async",
    "nvvm.wgmma.wait.group.sync.aligned",
    "nvvm.wmma.load",
    "nvvm.wmma.mma",
    "nvvm.wmma.store",
};

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
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
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

test "reqntid and minctasm are kernel attributes of the nvvm dialect" {
    const ctx = try testContext();
    defer ctx.deinit();
    var buf: [128]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{mlir.Attribute.dict(ctx, &.{ nvvm.reqntid(ctx, .{ 128, 1, 1 }), nvvm.minctasm(ctx, 1) })});
    try std.testing.expectEqualStrings("{nvvm.minctasm = 1 : i32, nvvm.reqntid = array<i32: 128, 1, 1>}", w.buffered());
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
