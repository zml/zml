const std = @import("std");

const mlir = @import("mlir");

// =============================================================================
// Enum attributes — parsed from their textual `#nvvm.<mnemonic><value>` form.
// =============================================================================

fn enumAttribute(ctx: *mlir.Context, comptime mnemonic: []const u8, value: []const u8) *const mlir.Attribute {
    var buf: [96]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "#nvvm." ++ mnemonic ++ "<{s}>", .{value}) catch unreachable;
    return mlir.Attribute.parse(ctx, text) catch std.debug.panic("failed to parse NVVM attribute '{s}'", .{text});
}

fn EnumAttribute(comptime mnemonic: []const u8, comptime Tag: type) type {
    return struct {
        pub fn attribute(self: Tag, ctx: *mlir.Context) *const mlir.Attribute {
            return enumAttribute(ctx, mnemonic, @tagName(self));
        }
    };
}

/// `#nvvm.cta_group<...>`: CTAs cooperating on a tcgen05 operation.
pub const CtaGroup = enum {
    cta_1,
    cta_2,
    pub const attribute = EnumAttribute("cta_group", CtaGroup).attribute;
};

/// `#nvvm.tcgen05_fence<...>`: `tcgen05.fence::{before,after}_thread_sync`.
pub const Tcgen05FenceKind = enum {
    before,
    after,
    pub const attribute = EnumAttribute("tcgen05_fence", Tcgen05FenceKind).attribute;
};

/// `#nvvm.tcgen05_wait<...>`: `tcgen05.wait::{ld,st}`.
pub const Tcgen05WaitKind = enum {
    load,
    store,
    pub const attribute = EnumAttribute("tcgen05_wait", Tcgen05WaitKind).attribute;
};

/// `#nvvm.tcgen05_mma_kind<...>`: input types of `tcgen05.mma`.
pub const Tcgen05MmaKind = enum {
    f16,
    tf32,
    f8f6f4,
    i8,
    pub const attribute = EnumAttribute("tcgen05_mma_kind", Tcgen05MmaKind).attribute;
};

/// `#nvvm.tcgen05_ldst_shape<...>`: lane/bit shape of `tcgen05.ld` / `tcgen05.st`.
pub const Tcgen05LdStShape = enum {
    shape_16x64b,
    shape_16x128b,
    shape_16x256b,
    shape_32x32b,
    shape_16x32bx2,
    pub const attribute = EnumAttribute("tcgen05_ldst_shape", Tcgen05LdStShape).attribute;
};

/// `#nvvm.mem_scope<...>`: scope of a memory operation.
pub const MemScope = enum {
    cta,
    cluster,
    gpu,
    sys,
    pub const attribute = EnumAttribute("mem_scope", MemScope).attribute;
};

/// `#nvvm.mbar_wait<...>`: `test` (non-blocking) or `try` (may suspend) wait.
pub const MBarrierWaitKind = enum {
    @"test",
    @"try",
    pub const attribute = EnumAttribute("mbar_wait", MBarrierWaitKind).attribute;
};

/// `#nvvm.mbar_scope<...>`: scope of an mbarrier wait.
pub const MBarrierScope = enum {
    cta,
    cluster,
    pub const attribute = EnumAttribute("mbar_scope", MBarrierScope).attribute;
};

/// `#nvvm.proxy_kind<...>`: memory proxy of `fence.proxy`.
pub const ProxyKind = enum {
    alias,
    @"async",
    @"async.global",
    @"async.shared",
    pub const attribute = EnumAttribute("proxy_kind", ProxyKind).attribute;
};

/// `#nvvm.shared_space<...>`: `shared::cta` or `shared::cluster`.
pub const SharedSpace = enum {
    cta,
    cluster,
    pub const attribute = EnumAttribute("shared_space", SharedSpace).attribute;
};

// =============================================================================
// Special registers and thread synchronization
// =============================================================================

/// nvvm.read.ptx.sreg.<name> — an i32 special register, e.g. `tid.x`, `ctaid.y`,
/// `cluster.ctarank`.
pub fn read_sreg(ctx: *mlir.Context, comptime name: []const u8, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.read.ptx.sreg." ++ name, .{
        .results = .{ .flat = &.{.int(ctx, .i32)} },
        .location = location,
    });
}

/// nvvm.barrier — `bar.sync 0` over the whole CTA.
pub fn barrier(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.barrier", .{
        .attributes = &.{.named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{ 0, 0, 0 }))},
        .location = location,
    });
}

/// nvvm.bar.warp.sync — `bar.warp.sync mask` (i32 lane mask, -1 for the full warp).
pub fn bar_warp_sync(ctx: *mlir.Context, mask: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.bar.warp.sync", .{
        .operands = .{ .flat = &.{mask} },
        .location = location,
    });
}

/// nvvm.elect.sync — i1, true in one elected lane of the warp.
pub fn elect_sync(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.elect.sync", .{
        .results = .{ .flat = &.{.int(ctx, .i1)} },
        .location = location,
    });
}

// =============================================================================
// mbarrier
// =============================================================================

/// nvvm.mbarrier.init — initialize the shared mbarrier at `ptr` (`!llvm.ptr<3>`)
/// expecting `count` (i32) arrivals per phase.
pub fn mbarrier_init(ctx: *mlir.Context, ptr: *const mlir.Value, count: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mbarrier.init", .{
        .operands = .{ .flat = &.{ ptr, count } },
        .attributes = &.{.named(ctx, "layout", enumAttribute(ctx, "mbarrier_layout", "v0"))},
        .location = location,
    });
}

/// nvvm.mbarrier.arrive — arrive `count` (i32) times (release semantics).
pub fn mbarrier_arrive(ctx: *mlir.Context, ptr: *const mlir.Value, count: *const mlir.Value, scope: MemScope, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive", .{
        .operands = .{ .flat = &.{ ptr, count } },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{ 1, 1, 0 })),
            .named(ctx, "relaxed", .boolean(ctx, false)),
            .named(ctx, "scope", scope.attribute(ctx)),
        },
        .location = location,
    });
}

/// nvvm.mbarrier.arrive.expect_tx — arrive and expect `bytes` (i32) of asynchronous
/// transactions (SM90+).
pub fn mbarrier_arrive_expect_tx(ctx: *mlir.Context, ptr: *const mlir.Value, bytes: *const mlir.Value, scope: MemScope, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mbarrier.arrive.expect_tx", .{
        .operands = .{ .flat = &.{ ptr, bytes } },
        .attributes = &.{
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{ 1, 1, 0, 0 })),
            .named(ctx, "relaxed", .boolean(ctx, false)),
            .named(ctx, "scope", scope.attribute(ctx)),
        },
        .location = location,
    });
}

/// nvvm.mbarrier.wait.parity — i1, whether phase `parity` (i32) has completed
/// (`test`: never suspends; `try`: may suspend for a system-dependent time).
pub fn mbarrier_wait_parity(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    parity: *const mlir.Value,
    kind: MBarrierWaitKind,
    scope: MBarrierScope,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mbarrier.wait.parity", .{
        .operands = .{ .flat = &.{ ptr, parity } },
        .results = .{ .flat = &.{.int(ctx, .i1)} },
        .attributes = &.{
            .named(ctx, "kind", kind.attribute(ctx)),
            .named(ctx, "scope", scope.attribute(ctx)),
        },
        .location = location,
    });
}

/// nvvm.mbarrier.try_wait.parity — loop until phase `parity` (i32) has completed,
/// suspending up to `suspend_time` (i32, cycles) per attempt.
pub fn mbarrier_try_wait_parity(
    ctx: *mlir.Context,
    ptr: *const mlir.Value,
    parity: *const mlir.Value,
    suspend_time: *const mlir.Value,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mbarrier.try_wait.parity", .{
        .operands = .{ .flat = &.{ ptr, parity, suspend_time } },
        .attributes = &.{.named(ctx, "useIntrinsic", .boolean(ctx, false))},
        .location = location,
    });
}

/// nvvm.fence.mbarrier.init — make mbarrier initialization visible to the async proxy.
pub fn fence_mbarrier_init(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.fence.mbarrier.init", .{ .location = location });
}

/// nvvm.fence.proxy — order memory accesses across proxies (e.g. generic writes
/// before asynchronous TMA / tcgen05 reads of shared memory).
pub fn fence_proxy(ctx: *mlir.Context, kind: ProxyKind, space: SharedSpace, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.fence.proxy", .{
        .attributes = &.{
            .named(ctx, "kind", kind.attribute(ctx)),
            .named(ctx, "space", space.attribute(ctx)),
        },
        .location = location,
    });
}

// =============================================================================
// Asynchronous copies
// =============================================================================

/// nvvm.cp.async.commit.group — close the current group of cp.async copies.
pub fn cp_async_commit_group(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cp.async.commit.group", .{ .location = location });
}

/// nvvm.cp.async.wait.group — wait until at most `pending` cp.async groups are in flight.
pub fn cp_async_wait_group(ctx: *mlir.Context, pending: u32, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cp.async.wait.group", .{
        .attributes = &.{.named(ctx, "n", .int(ctx, .i32, pending))},
        .location = location,
    });
}

/// nvvm.cp.async.bulk.commit.group — close the current group of bulk copies (SM90+).
pub fn cp_async_bulk_commit_group(ctx: *mlir.Context, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.commit.group", .{ .location = location });
}

/// nvvm.cp.async.bulk.wait_group — wait until at most `pending` bulk groups are in
/// flight; with `read`, only until their sources have been read.
pub fn cp_async_bulk_wait_group(ctx: *mlir.Context, pending: u32, read: bool, location: *const mlir.Location) *mlir.Operation {
    var attrs: [2]mlir.NamedAttribute = undefined;
    attrs[0] = .named(ctx, "group", .int(ctx, .i32, pending));
    var len: usize = 1;
    if (read) {
        attrs[len] = .named(ctx, "read", .unit(ctx));
        len += 1;
    }
    return mlir.Operation.make(ctx, "nvvm.cp.async.bulk.wait_group", .{
        .attributes = attrs[0..len],
        .location = location,
    });
}

// =============================================================================
// tcgen05 (SM100 tensor cores and tensor memory)
// =============================================================================

/// nvvm.tcgen05.alloc — allocate `columns` (i32) TMEM columns for the warp; the base
/// address is written to `holder` (`!llvm.ptr<3>`).
pub fn tcgen05_alloc(ctx: *mlir.Context, holder: *const mlir.Value, columns: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05.alloc", .{
        .operands = .{ .flat = &.{ holder, columns } },
        .location = location,
    });
}

/// nvvm.tcgen05.dealloc — free `columns` (i32) TMEM columns at `taddr` (`!llvm.ptr<6>`).
pub fn tcgen05_dealloc(ctx: *mlir.Context, taddr: *const mlir.Value, columns: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05.dealloc", .{
        .operands = .{ .flat = &.{ taddr, columns } },
        .location = location,
    });
}

/// nvvm.tcgen05.fence — `tcgen05.fence::{before,after}_thread_sync`.
pub fn tcgen05_fence(ctx: *mlir.Context, kind: Tcgen05FenceKind, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05.fence", .{
        .attributes = &.{.named(ctx, "kind", kind.attribute(ctx))},
        .location = location,
    });
}

/// nvvm.tcgen05.wait — wait for this thread's tcgen05 loads or stores.
pub fn tcgen05_wait(ctx: *mlir.Context, kind: Tcgen05WaitKind, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05.wait", .{
        .attributes = &.{.named(ctx, "kind", kind.attribute(ctx))},
        .location = location,
    });
}

/// nvvm.tcgen05.commit — arrive on the mbarrier at `barrier` (`!llvm.ptr<3>`) once the
/// preceding tcgen05 MMAs complete.
pub fn tcgen05_commit(ctx: *mlir.Context, barrier_ptr: *const mlir.Value, group: CtaGroup, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05.commit", .{
        .operands = .{ .flat = &.{barrier_ptr} },
        .attributes = &.{.named(ctx, "group", group.attribute(ctx))},
        .location = location,
    });
}

/// nvvm.tcgen05.mma — D (TMEM, `!llvm.ptr<6>`) = A * B (+ D when `accumulate`, i1),
/// A and B given by shared-memory descriptors (i64), `idesc` the instruction descriptor (i32).
pub fn tcgen05_mma(
    ctx: *mlir.Context,
    d: *const mlir.Value,
    a_desc: *const mlir.Value,
    b_desc: *const mlir.Value,
    idesc: *const mlir.Value,
    accumulate: *const mlir.Value,
    kind: Tcgen05MmaKind,
    cta_group: CtaGroup,
    location: *const mlir.Location,
) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05.mma", .{
        .operands = .{ .flat = &.{ d, a_desc, b_desc, idesc, accumulate } },
        .attributes = &.{
            .named(ctx, "kind", kind.attribute(ctx)),
            .named(ctx, "ctaGroup", cta_group.attribute(ctx)),
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{ 1, 1, 1, 1, 1, 0, 0 })),
        },
        .location = location,
    });
}

/// nvvm.tcgen05.ld — load from TMEM at `taddr` (`!llvm.ptr<6>`) into `result_type`
/// (i32 or `vector<n x i32>`).
pub fn tcgen05_ld(ctx: *mlir.Context, taddr: *const mlir.Value, result_type: *const mlir.Type, shape: Tcgen05LdStShape, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05.ld", .{
        .operands = .{ .flat = &.{taddr} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{.named(ctx, "shape", shape.attribute(ctx))},
        .location = location,
    });
}

/// nvvm.tcgen05.st — store `value` (i32 or `vector<n x i32>`) to TMEM at `taddr`.
pub fn tcgen05_st(ctx: *mlir.Context, taddr: *const mlir.Value, value: *const mlir.Value, shape: Tcgen05LdStShape, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.tcgen05.st", .{
        .operands = .{ .flat = &.{ taddr, value } },
        .attributes = &.{.named(ctx, "shape", shape.attribute(ctx))},
        .location = location,
    });
}

// =============================================================================
// Clusters
// =============================================================================

/// nvvm.mapa — the shared-memory location `ptr` (`!llvm.ptr<3>`) in CTA `rank` (i32)
/// of the cluster, as a `result_type` (`!llvm.ptr<7>`) pointer.
pub fn mapa(ctx: *mlir.Context, ptr: *const mlir.Value, rank: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.mapa", .{
        .operands = .{ .flat = &.{ ptr, rank } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// nvvm.cluster.arrive — `barrier.cluster.arrive` (release), `.aligned` when every
/// thread of the warp executes it.
pub fn cluster_arrive(ctx: *mlir.Context, aligned: bool, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cluster.arrive", .{
        .attributes = if (aligned) &.{.named(ctx, "aligned", .unit(ctx))} else &.{},
        .location = location,
    });
}

/// nvvm.cluster.wait — `barrier.cluster.wait` (acquire).
pub fn cluster_wait(ctx: *mlir.Context, aligned: bool, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.cluster.wait", .{
        .attributes = if (aligned) &.{.named(ctx, "aligned", .unit(ctx))} else &.{},
        .location = location,
    });
}

// =============================================================================
// Math
// =============================================================================

/// nvvm.ex2 — approximate 2^x (f32), flushing denormals when `ftz`.
pub fn ex2(ctx: *mlir.Context, value: *const mlir.Value, ftz: bool, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.ex2", .{
        .operands = .{ .flat = &.{value} },
        .results = .{ .flat = &.{mlir.Type.float(ctx, .f32)} },
        .attributes = &.{.named(ctx, "ftz", .boolean(ctx, ftz))},
        .location = location,
    });
}

/// nvvm.rcp.approx.ftz.f — approximate 1/x (f32), flushing denormals.
pub fn rcp_approx_ftz(ctx: *mlir.Context, value: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "nvvm.rcp.approx.ftz.f", .{
        .operands = .{ .flat = &.{value} },
        .results = .{ .flat = &.{mlir.Type.float(ctx, .f32)} },
        .location = location,
    });
}

test {
    std.testing.refAllDecls(@This());
}
