//! Source-level SM100 NVFP4 grouped GEMM.
//!
//! The NVFP4 counterpart of `persistent_mxfp4.zig`, following the CuTe DSL
//! `Sm100BlockScaledPersistentDenseGemmKernel` with FP4 x FP4 operands, E4M3
//! scale factors every 16 elements and the `kind::mxf4nvf4.block16` MMA. Both
//! operands are packed FP4 in global and shared memory (TMA format `U4`); one
//! pipeline stage holds a 256-deep K slice, consumed by four K = 64 MMAs.
//!
//! One CTA computes a 128xN output tile of one expert group: warp 5 drives
//! the TMA ring, warp 4 issues the S2T scale copies and UMMA, and warps 0-3
//! drain the accumulator from tensor memory into one of the two epilogues.
//! The pipeline, scheduler and warp roles are shared with the MXFP4 kernel.
const std = @import("std");
const mlir = @import("mlir");

const zml = @import("../../../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;
const common = @import("common.zig");
const nvfp4 = @import("nvfp4.zig");

const accumulator_stages = 2;
/// K elements of one pipeline stage: four K = 64 FP4 MMAs.
pub const k_tile = 256;
/// Largest dynamic shared-memory allocation of an SM100 CTA.
const smem_capacity = 232448;
const mma_kind: cute.BlockScaledMmaConfig.Kind = .nvfp4_nvfp4;

/// Shared-memory plan for one tile width, laid out like the Python
/// `SharedStorage`: barriers, epilogue scratch, then the A, B, SFA and SFB
/// rings. The ring is as deep as the capacity allows.
pub const Tile = struct {
    n: i64,
    stages: usize,
    smem_bytes: i64,
    tmem_columns: u16,
    empty_barriers: i64,
    accumulator_full: i64,
    accumulator_empty: i64,
    tmem_dealloc: i64,
    tmem_holding: i64,
    c: i64,
    a: i64,
    b: i64,
    sfa: i64,
    sfb: i64,

    /// 128x256 packed FP4.
    const a_stage_bytes = 128 * k_tile / 2;
    /// One 128x16 E4M3 scale tile: four 512-byte 128x4 atoms.
    const sf_stage_bytes = 128 * k_tile / 16;
    const c_bytes = 512;
    const header_bytes = 1024;

    pub fn of(comptime n: i64) Tile {
        comptime {
            std.debug.assert(n >= 8 and n <= 128 and @rem(n, 8) == 0);
            const per_stage = a_stage_bytes + @divExact(n * k_tile, 2) + 2 * sf_stage_bytes;
            const stages: usize = @intCast(@divTrunc(smem_capacity - header_bytes - c_bytes, per_stage));
            const s: i64 = @intCast(stages);
            const a = header_bytes + 1024;
            const b = a + s * a_stage_bytes;
            const sfa = b + @divExact(s * n * k_tile, 2);
            const sfb = sfa + s * sf_stage_bytes;
            // Accumulators, then 16 SFA and 16 SFB columns.
            const columns = std.math.ceilPowerOfTwo(u16, @max(32, accumulator_stages * n + 32)) catch unreachable;
            return .{
                .n = n,
                .stages = stages,
                .smem_bytes = sfb + s * sf_stage_bytes,
                .tmem_columns = columns,
                .empty_barriers = 8 * s,
                .accumulator_full = 16 * s,
                .accumulator_empty = 16 * s + 16,
                .tmem_dealloc = 16 * s + 32,
                .tmem_holding = 16 * s + 40,
                .c = header_bytes,
                .a = a,
                .b = b,
                .sfa = sfa,
                .sfb = sfb,
            };
        }
    }

    /// Routed rows an epilogue pass handles: one per warp lane.
    pub fn chunkRows(self: Tile) i64 {
        return @min(self.n, 32);
    }

    /// Epilogue passes over the tile width.
    pub fn chunks(self: Tile) usize {
        return @intCast(@divExact(self.n, self.chunkRows()));
    }

    fn bStageBytes(self: Tile) i64 {
        return @divExact(self.n * k_tile, 2);
    }

    /// TMA bytes landing in one stage: packed A and B, then both scale tiles.
    fn transactionBytes(self: Tile) u32 {
        return @intCast(a_stage_bytes + self.bStageBytes() + 2 * sf_stage_bytes);
    }

    fn fields(self: Tile, b: *B) []const []const u8 {
        const arena = b.arena.allocator();
        const s: i64 = @intCast(self.stages);
        const out = arena.alloc([]const u8, 11) catch @panic("OOM");
        const Field = struct { []const u8, i64, i64 };
        const list = [_]Field{
            .{ "ab_full_mbar_ptr", 8 * s, 0 },
            .{ "ab_empty_mbar_ptr", 8 * s, self.empty_barriers },
            .{ "acc_full_mbar_ptr", 16, self.accumulator_full },
            .{ "acc_empty_mbar_ptr", 16, self.accumulator_empty },
            .{ "tmem_dealloc_mbar", 8, self.tmem_dealloc },
            .{ "tmem_holding_buf", 4, self.tmem_holding },
            .{ "sC", c_bytes, self.c },
            .{ "sA", s * a_stage_bytes, self.a },
            .{ "sB", s * self.bStageBytes(), self.b },
            .{ "sSFA", s * sf_stage_bytes, self.sfa },
            .{ "sSFB", s * sf_stage_bytes, self.sfb },
        };
        for (list, out) |field, *text| {
            text.* = std.fmt.allocPrint(arena, "{s}:{d}:{d}", .{ field[0], field[1], field[2] }) catch @panic("OOM");
        }
        return out;
    }
};

const StageStorage = struct {
    a: V,
    b: V,
    sfa: V,
    sfb: V,
    c: V,
    load_barriers: V,
    mma_barriers: V,
    accumulator_full_barriers: V,
    accumulator_empty_barriers: V,
};

const StageBuffers = struct {
    a_destination: cute.View,
    b_destination: cute.View,
    sfa_destination: cute.View,
    sfb_destination: cute.View,
    load_barrier: V,
    mma_barrier: V,
};

const StagedOperands = struct {
    a: cute.View,
    b: cute.View,
    sfa: cute.View,
    sfb: cute.View,
};

/// What the epilogue warps do with each finished 128xN accumulator tile.
pub const Epilogue = enum {
    /// Up projection: per-expert dequantization scale, BF16 rounding,
    /// SwiGLU of the interleaved gate/up rows (optionally clamped), the
    /// routing weight when it precedes the down projection, BF16 rounding,
    /// then NVFP4 quantization (E4M3 scale per 16 activations, scaled by the
    /// per-expert activation global scale) straight into the down GEMM's
    /// packed input and 128x4 scale layouts.
    swiglu_nvfp4,
    /// Down projection: per-expert dequantization scale and, when it follows
    /// the down projection, the routing weight, then BF16 rows stored in
    /// route order `[routes, m]`, skipping padding rows.
    route_rows,
};

pub const Config = struct {
    experts: i64,
    /// GEMM output rows (2 x intermediate for gate/up, hidden for down).
    m: i64,
    /// Tile N: rows in one routed expert group.
    n: i64 = 16,
    /// Reduction dimension, a multiple of `k_tile`.
    k: i64,
    groups: i64,
    /// Resident CTAs of the static persistent tile scheduler: one per SM.
    persistent_ctas: i64 = 148,
    epilogue: Epilogue,
    /// Token/top-k routes; sizes the route-indexed epilogue operands.
    routes: i64,
    /// Clamp of both halves of the fused SwiGLU, if any.
    swiglu_limit: ?f32 = null,
    /// Multiply rows by their routing weight in this projection's epilogue.
    routing_weights: bool = false,
    /// Ungrouped routing: group `g` is route `g` and holds one live row.
    direct: bool = false,

    /// Rows stored per group in the routed operands: one whole N tile.
    pub fn groupRows(self: Config) i64 {
        return self.n;
    }

    pub fn routedRows(self: Config) i64 {
        return self.groups * self.groupRows();
    }
};

/// `schedule` is the routing schedule `[group experts | group sizes | active
/// group count]`; `alpha` the per-expert FP32 dequantization scale of the
/// GEMM output.
const inputs = [_][:0]const u8{
    "weight",
    "schedule",
    "input_quant",
    "weight_scale",
    "input_scale",
    "alpha",
};

/// `route_inverse` maps each physical routed row to its token/top-k route;
/// `act_scale` is the per-expert global scale of the down GEMM's NVFP4 input.
const UpQuantized = cute.Program(Config, .{
    .name = "zml_nvfp4_up_swiglu_zig",
    .inputs = &(inputs ++ [_][:0]const u8{ "route_inverse", "routing_weights", "act_scale" }),
    .outputs = &.{ "q", "s" },
    .run = buildProgram,
});

const DownRows = cute.Program(Config, .{
    .name = "zml_nvfp4_down_rows_zig",
    .inputs = &(inputs ++ [_][:0]const u8{ "route_inverse", "routing_weights" }),
    .outputs = &.{"output"},
    .run = buildProgram,
});

pub const Inputs = struct {
    /// `[experts, m, k / 2]` packed FP4.
    weight: zml.Tensor,
    schedule: zml.Tensor,
    /// `[groups * n, k / 2]` packed FP4.
    input_quant: zml.Tensor,
    /// `[experts, m, k / 16]` E4M3 bytes in the 128x4 layout.
    weight_scale: zml.Tensor,
    /// `[groups, k / 16 * 128]` E4M3 bytes in the 128x4 layout.
    input_scale: zml.Tensor,
    /// `[experts]` FP32.
    alpha: zml.Tensor,
};

/// Up projection with the SwiGLU + NVFP4 epilogue: the routed packed FP4
/// rows `[groups * n, m / 4]` and their 128x4 E4M3 scales
/// `[groups, m / 32 * 128]`, ready for the down GEMM. `routing_weights` is
/// `[routes]` FP32 and `act_scale` `[experts]` FP32.
pub fn upQuantized(cfg: Config, a: Inputs, route_inverse: zml.Tensor, routing_weights: zml.Tensor, act_scale: zml.Tensor) struct { q: zml.Tensor, s: zml.Tensor } {
    std.debug.assert(cfg.epilogue == .swiglu_nvfp4);
    const activations = @divExact(cfg.m, 2);
    const result = UpQuantized.call(.{
        .weight = a.weight,
        .schedule = a.schedule,
        .input_quant = a.input_quant,
        .weight_scale = a.weight_scale,
        .input_scale = a.input_scale,
        .alpha = a.alpha,
        .route_inverse = route_inverse,
        .routing_weights = routing_weights,
        .act_scale = act_scale,
    }, .{
        .q = .init(.{ cfg.routedRows(), @divExact(activations, 2) }, .u8),
        .s = .init(.{ cfg.groups, @divExact(activations, 16) * 128 }, .u8),
    }, .{ .cfg = cfg, .scalars = &.{ cfg.m, cfg.n, cfg.k, cfg.groups, cfg.experts } });
    return .{ .q = result.q, .s = result.s };
}

/// Down projection storing BF16 `[routes, m]` in token/top-k order.
pub fn downRows(cfg: Config, a: Inputs, route_inverse: zml.Tensor, routing_weights: zml.Tensor) zml.Tensor {
    std.debug.assert(cfg.epilogue == .route_rows);
    return DownRows.call(.{
        .weight = a.weight,
        .schedule = a.schedule,
        .input_quant = a.input_quant,
        .weight_scale = a.weight_scale,
        .input_scale = a.input_scale,
        .alpha = a.alpha,
        .route_inverse = route_inverse,
        .routing_weights = routing_weights,
    }, .{ .output = .init(.{ @max(cfg.routes, 1), cfg.m }, .bf16) }, .{ .cfg = cfg, .scalars = &.{ cfg.m, cfg.n, cfg.k, cfg.groups, cfg.experts } }).output;
}

const typed = common.typed;
const gridTyped = common.gridTyped;
const fmt = common.fmt;
const view = common.view;
const barrierAt = common.barrierAt;
const PipelinePosition = common.PipelinePosition;
const DirectRoutes = common.DirectRoutes;
const persistentWorkAt = common.persistentWorkAt;
const advancePersistentWork = common.advancePersistentWork;
const advancePipeline = common.advancePipeline;
const peekBarrierIfRemaining = common.peekBarrierIfRemaining;
const sharedPointerAt = common.sharedPointerAt;
const offsetAlignment = common.offsetAlignment;
const typesOf = common.typesOf;
const tupleOf = common.tupleOf;

fn deviceName(comptime epilogue: Epilogue) [:0]const u8 {
    return switch (epilogue) {
        .swiglu_nvfp4 => "nvfp4_sm100_persistent_grouped_gemm_up_swiglu",
        .route_rows => "nvfp4_sm100_persistent_grouped_gemm_down_rows",
    };
}

pub const tile_widths = [_]i64{ 8, 16, 32, 64, 128 };

fn buildProgram(b: *B, cfg: Config) cute.FinishError!void {
    return switch (cfg.n) {
        inline 8, 16, 32, 64, 128 => |n| switch (cfg.epilogue) {
            inline else => |epilogue| buildTiledProgram(b, cfg, comptime Tile.of(n), epilogue),
        },
        else => std.debug.panic("unsupported NVFP4 tile N {d}", .{cfg.n}),
    };
}

/// Types and layouts shared by the device ABI and the host launch, as the
/// CuTe DSL derives them for FP4 x FP4 with 16-element E4M3 scale vectors.
fn Layouts(comptime tile: Tile) type {
    return struct {
        const n = tile.n;
        const stages = tile.stages;
        const ab_swizzle: B.Swizzle = .{ .bits = 3, .base = 4, .shift = 3 };

        fn aCoord(b: *B, cfg: Config) cute.LayoutSpec {
            return b.layoutSpec(.{ cfg.m, cfg.k, cfg.experts }, .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), b.basis(1, 1, .{2}) });
        }
        fn bCoord(b: *B, cfg: Config) cute.LayoutSpec {
            return b.layoutSpec(.{ cfg.groupRows(), cfg.k, cfg.groups }, .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), b.basis(1, 1, .{2}) });
        }
        /// TMA coordinates of the 128x4 scale layout: byte in the atom, K
        /// atom, M block, L.
        fn sfCoord(b: *B) cute.LayoutSpec {
            return b.layoutSpec(
                .{ .{ .{ 32, 4 }, cute.AlgebraToken.dynamic }, .{ .{ 16, 4 }, cute.AlgebraToken.dynamic }, .{ 1, cute.AlgebraToken.dynamic } },
                .{
                    .{ .{ b.basis(8, 1, .{0}), b.basis(2, 1, .{0}) }, b.basis(1, 1, .{2}) },
                    .{ .{ 0, b.basis(1, 2, .{0}) }, b.basis(1, 1, .{1}) },
                    .{ 0, b.basis(1, 1, .{3}) },
                },
            );
        }
        fn aBasis(b: *B) cute.LayoutSpec {
            return b.layoutSpec(.{ k_tile, 128, 1 }, .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), b.basis(1, 1, .{2}) });
        }
        fn bBasis(b: *B) cute.LayoutSpec {
            return b.layoutSpec(.{ k_tile, n, 1 }, .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), b.basis(1, 1, .{2}) });
        }
        fn sfBasis(b: *B) cute.LayoutSpec {
            return b.layoutSpec(
                .{ .{ 2, 4, 32 }, 4, 1, 1 },
                .{ .{ b.basis(1, 1, .{ 1, 0, 1 }), b.basis(1, 1, .{ 1, 0, 0 }), b.basis(1, 1, .{ 0, 0, 0 }) }, b.basis(1, 1, .{ 1, 1 }), b.basis(1, 1, .{ 1, 0 }), b.basis(1, 1, .{ 1, 2 }) },
            );
        }
        fn tmaA(b: *B, cfg: Config) cute.TmaConfig {
            return .{ .dtype = .f4e2m1fn, .copy_bits = 128 * k_tile * 4, .global_basis = aBasis(b), .coordinate_layout = aCoord(b, cfg), .coordinate_rank = 3, .format = .u4 };
        }
        fn tmaB(b: *B, cfg: Config) cute.TmaConfig {
            return .{ .dtype = .f4e2m1fn, .copy_bits = @intCast(n * k_tile * 4), .global_basis = bBasis(b), .coordinate_layout = bCoord(b, cfg), .coordinate_rank = 3, .format = .u4 };
        }
        fn tmaSf(b: *B) cute.TmaConfig {
            return .{ .dtype = .f8e4m3fn, .copy_bits = Tile.sf_stage_bytes * 8, .global_basis = sfBasis(b), .coordinate_layout = sfCoord(b), .coordinate_rank = 4, .format = .u16 };
        }
        // Shared-memory layouts in FP4 elements for A/B, bytes for scales.
        fn aSmem(b: *B, comptime staged: bool) cute.LayoutSpec {
            return if (staged)
                b.layoutSpec(.{ .{ 128, 64 }, 1, 4, stages }, .{ .{ k_tile, 1 }, 0, 64, 2 * Tile.a_stage_bytes })
            else
                b.layoutSpec(.{ .{ 128, 64 }, 1, 4 }, .{ .{ k_tile, 1 }, 0, 64 });
        }
        fn bSmem(b: *B, comptime staged: bool) cute.LayoutSpec {
            return if (staged)
                b.layoutSpec(.{ .{ n, 64 }, 1, 4, stages }, .{ .{ k_tile, 1 }, 0, 64, 2 * tile.bStageBytes() })
            else
                b.layoutSpec(.{ .{ n, 64 }, 1, 4 }, .{ .{ k_tile, 1 }, 0, 64 });
        }
        fn sfSmem(b: *B, comptime staged: bool) cute.LayoutSpec {
            return if (staged)
                b.layoutSpec(.{ .{ .{ .{ 32, 4 }, 1 }, .{ 16, 4 } }, 1, 4, stages }, .{ .{ .{ .{ 16, 4 }, 0 }, .{ 0, 1 } }, 0, 512, Tile.sf_stage_bytes })
            else
                b.layoutSpec(.{ .{ .{ .{ 32, 4 }, 1 }, .{ 16, 4 } }, 1, 4 }, .{ .{ .{ .{ 16, 4 }, 0 }, .{ 0, 1 } }, 0, 512 });
        }
        fn fp4SmemPtrTy(b: *B) *const mlir.Type {
            return b.swizzledPtrTy(.f4e2m1fn, .smem, 1024, ab_swizzle) catch @panic("bad FP4 shared pointer");
        }
    };
}

/// Device memrefs of the epilogue operands.
const EpilogueTypes = struct {
    sizes: *const mlir.Type,
    inverse: *const mlir.Type,
    weights: *const mlir.Type,
    per_expert: *const mlir.Type,
    q: *const mlir.Type,
    s: *const mlir.Type,
    rows: *const mlir.Type,

    fn of(b: *B, cfg: Config) EpilogueTypes {
        const activations = @divExact(cfg.m, 2);
        return .{
            .sizes = b.memrefType(.i32, .gmem, 4, b.layoutType(b.layoutSpec(cfg.groups, 1))),
            .inverse = b.memrefType(.i32, .gmem, 16, b.layoutType(b.layoutSpec(cfg.routedRows(), 1))),
            .weights = b.memrefType(.f32, .gmem, 16, b.layoutType(b.layoutSpec(@max(cfg.routes, 1), 1))),
            .per_expert = b.memrefType(.f32, .gmem, 16, b.layoutType(b.layoutSpec(cfg.experts, 1))),
            // Eight packed FP4 activations per word.
            .q = b.memrefType(.i32, .gmem, 16, b.layoutType(b.layoutSpec(.{ cfg.routedRows(), @divExact(activations, 8) }, .{ @divExact(activations, 8), 1 }))),
            .s = b.memrefType(.i8, .gmem, 16, b.layoutType(b.layoutSpec(.{ cfg.groups, @divExact(activations, 16) * 128 }, .{ @divExact(activations, 16) * 128, 1 }))),
            .rows = b.memrefType(.bf16, .gmem, 16, b.layoutType(b.layoutSpec(.{ @max(cfg.routes, 1), cfg.m }, .{ cfg.m, 1 }))),
        };
    }
};

fn buildTiledProgram(b: *B, cfg: Config, comptime tile: Tile, comptime epilogue: Epilogue) cute.FinishError!void {
    @setEvalBranchQuota(20_000);
    std.debug.assert(@rem(cfg.m, 128) == 0);
    std.debug.assert(@rem(cfg.k, k_tile) == 0);
    const L = Layouts(tile);
    const public_name = b.name;

    const ids_memref = b.memrefType(.i32, .gmem, 16, b.layoutType(b.layoutSpec(cfg.groups, 1)));
    const active_memref = b.memrefType(.i32, .gmem, 4, b.layoutType(b.layoutSpec(1, 1)));
    const et = EpilogueTypes.of(b, cfg);
    const i32_type = cute.DType.i32.toMlir(b.ctx);
    const mma_type = b.blockScaledMmaType(.{ .kind = mma_kind, .n = @intCast(tile.n) });
    const mma_sfb_type = b.blockScaledMmaType(.{ .kind = mma_kind, .n = 128 });

    b.beginFunction(deviceName(epilogue), .cuda_kernel);
    const d = try b.declareArgs(switch (epilogue) {
        .swiglu_nvfp4 => .{
            .expert_ids = typed(ids_memref),
            .active_count = typed(active_memref),
            .mma = typed(mma_type),
            .mma_sfb = typed(mma_sfb_type),
            .tma_a = gridTyped(b.tmaLoadAtomType(L.tmaA(b, cfg))),
            .gA = typed(b.coordTensorType(3, L.aCoord(b, cfg))),
            .tma_b = gridTyped(b.tmaLoadAtomType(L.tmaB(b, cfg))),
            .gB = typed(b.coordTensorType(3, L.bCoord(b, cfg))),
            .tma_sfa = gridTyped(b.tmaLoadAtomType(L.tmaSf(b))),
            .gSFA = typed(b.coordTensorType(4, L.sfCoord(b))),
            .tma_sfb = gridTyped(b.tmaLoadAtomType(L.tmaSf(b))),
            .gSFB = typed(b.coordTensorType(4, L.sfCoord(b))),
            .alpha = typed(et.per_expert),
            .group_sizes = typed(et.sizes),
            .route_inverse = typed(et.inverse),
            .routing_weights = typed(et.weights),
            .act_scale = typed(et.per_expert),
            .q = typed(et.q),
            .s = typed(et.s),
            .tiles_m = typed(i32_type),
        },
        .route_rows => .{
            .expert_ids = typed(ids_memref),
            .active_count = typed(active_memref),
            .mma = typed(mma_type),
            .mma_sfb = typed(mma_sfb_type),
            .tma_a = gridTyped(b.tmaLoadAtomType(L.tmaA(b, cfg))),
            .gA = typed(b.coordTensorType(3, L.aCoord(b, cfg))),
            .tma_b = gridTyped(b.tmaLoadAtomType(L.tmaB(b, cfg))),
            .gB = typed(b.coordTensorType(3, L.bCoord(b, cfg))),
            .tma_sfa = gridTyped(b.tmaLoadAtomType(L.tmaSf(b))),
            .gSFA = typed(b.coordTensorType(4, L.sfCoord(b))),
            .tma_sfb = gridTyped(b.tmaLoadAtomType(L.tmaSf(b))),
            .gSFB = typed(b.coordTensorType(4, L.sfCoord(b))),
            .alpha = typed(et.per_expert),
            .group_sizes = typed(et.sizes),
            .route_inverse = typed(et.inverse),
            .routing_weights = typed(et.weights),
            .rows = typed(et.rows),
            .tiles_m = typed(i32_type),
        },
    });
    b.setFunctionAttribute("nvvm.minctasm", .int(b.ctx, .i32, 1));
    b.setFunctionAttribute("smem.partition_num", .int(b.ctx, .i32, 2));
    b.setFunctionAttribute("smem.hint_smem_base_uniform", .unit(b.ctx));
    b.setFunctionAttribute("cu_attrs", b.parseAttribute("{max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}"));

    // Prefetch every tensor-map descriptor while the producing grid drains.
    const warp = b.makeWarpUniform(b.makeWarpUniform(b.threadIdx().x.div(32)));
    var tma_warp = b.openIf(warp.eq(5));
    inline for (.{ d.tma_a, d.tma_b, d.tma_sfa, d.tma_sfb }) |value| {
        const descriptor: cute.Atom = .{ .inner = value.inner, .kernel = b };
        b.prefetchTmaDesc(descriptor);
    }
    tma_warp.yieldThen(.{});
    b.waitForDependency();
    buildUmmaDevice(b, cfg, tile, epilogue, d);
    b.launchDependents();
    b.endFunction(.{ 192, 1, 1 });

    // Public host ABI: the XLA operands, the outputs, then the problem shape.
    b.beginFunction(public_name, .host);
    const operands = .{
        .weight = .{ .ptr = cute.DType.i8 },
        .schedule = .{ .ptr = cute.DType.i32 },
        .input_quant = .{ .ptr = cute.DType.i8 },
        .weight_scale = .{ .ptr = cute.DType.i8 },
        .input_scale = .{ .ptr = cute.DType.i8 },
        .alpha = .{ .ptr = cute.DType.f32 },
        .route_inverse = .{ .ptr = cute.DType.i32 },
        .routing_weights = .{ .ptr = cute.DType.f32 },
    };
    const shape = .{
        .problem_m = typed(i32_type),
        .problem_n = typed(i32_type),
        .problem_k = typed(i32_type),
        .problem_l = typed(i32_type),
        .expert_count = typed(i32_type),
    };
    _ = try b.declareArgs(switch (epilogue) {
        .swiglu_nvfp4 => .{
            .weight = operands.weight,
            .schedule = operands.schedule,
            .input_quant = operands.input_quant,
            .weight_scale = operands.weight_scale,
            .input_scale = operands.input_scale,
            .alpha = operands.alpha,
            .route_inverse = operands.route_inverse,
            .routing_weights = operands.routing_weights,
            .act_scale = .{ .ptr = cute.DType.f32 },
            .q = .{ .ptr = cute.DType.i8 },
            .s = .{ .ptr = cute.DType.i8 },
            .problem_m = shape.problem_m,
            .problem_n = shape.problem_n,
            .problem_k = shape.problem_k,
            .problem_l = shape.problem_l,
            .expert_count = shape.expert_count,
        },
        .route_rows => .{
            .weight = operands.weight,
            .schedule = operands.schedule,
            .input_quant = operands.input_quant,
            .weight_scale = operands.weight_scale,
            .input_scale = operands.input_scale,
            .alpha = operands.alpha,
            .route_inverse = operands.route_inverse,
            .routing_weights = operands.routing_weights,
            .output = .{ .ptr = cute.DType.bf16 },
            .problem_m = shape.problem_m,
            .problem_n = shape.problem_n,
            .problem_k = shape.problem_k,
            .problem_l = shape.problem_l,
            .expert_count = shape.expert_count,
        },
    });
    try buildHostLaunch(b, cfg, tile, epilogue);
    b.endFunction(null);
}

/// Host argument positions: the six mainloop operands, the route operands,
/// the epilogue's extra inputs and outputs, then the five shape scalars.
fn HostArgs(comptime epilogue: Epilogue) type {
    return struct {
        const weight = 0;
        const schedule = 1;
        const input_quant = 2;
        const weight_scale = 3;
        const input_scale = 4;
        const alpha = 5;
        const route_inverse = 6;
        const routing_weights = 7;
        const act_scale = 8;
        const output = switch (epilogue) {
            .swiglu_nvfp4 => 9,
            .route_rows => 8,
        };
        const scales = 10;
        const problem_m = switch (epilogue) {
            .swiglu_nvfp4 => 11,
            .route_rows => 9,
        };
        const problem_n = problem_m + 1;
        const problem_k = problem_m + 2;
        const problem_l = problem_m + 3;
        const expert_count = problem_m + 4;
    };
}

/// Complete staged shared-memory tensors. Per-stage views are slices.
fn stagedOperands(b: *B, comptime tile: Tile, storage: StageStorage) StagedOperands {
    const L = Layouts(tile);
    const a_full = b.makeViewTyped(
        b.recastSwizzledPointer(storage.a, .f4e2m1fn, .smem, 1024, L.ab_swizzle),
        b.staticLayout(L.aSmem(b, true)),
        b.memrefTypeFromPointer(L.fp4SmemPtrTy(b), b.layoutType(L.aSmem(b, true))),
    );
    const b_full = b.makeViewTyped(
        b.recastSwizzledPointer(storage.b, .f4e2m1fn, .smem, 1024, L.ab_swizzle),
        b.staticLayout(L.bSmem(b, true)),
        b.memrefTypeFromPointer(L.fp4SmemPtrTy(b), b.layoutType(L.bSmem(b, true))),
    );
    const sfa_full = b.makeViewTyped(storage.sfa, b.staticLayout(L.sfSmem(b, true)), b.memrefType(.f8e4m3fn, .smem, 1024, b.layoutType(L.sfSmem(b, true))));
    const sfb_full = b.makeViewTyped(storage.sfb, b.staticLayout(L.sfSmem(b, true)), b.memrefType(.f8e4m3fn, .smem, 1024, b.layoutType(L.sfSmem(b, true))));
    return .{ .a = a_full, .b = b_full, .sfa = sfa_full, .sfb = sfb_full };
}

fn stageBuffersAt(b: *B, comptime tile: Tile, storage: StageStorage, stage_index: V) StageBuffers {
    const L = Layouts(tile);
    const full = stagedOperands(b, tile, storage);
    const all = cute.AlgebraToken.all;
    const a = b.sliceTyped(full.a, .{ all, all, all, stage_index }, b.memrefTypeFromPointer(L.fp4SmemPtrTy(b), b.layoutType(L.aSmem(b, false))));
    const rhs = b.sliceTyped(full.b, .{ all, all, all, stage_index }, b.memrefTypeFromPointer(L.fp4SmemPtrTy(b), b.layoutType(L.bSmem(b, false))));
    const sf_stage_align = 1024;
    const sfa = b.sliceTyped(full.sfa, .{ all, all, all, stage_index }, b.memrefType(.f8e4m3fn, .smem, sf_stage_align, b.layoutType(L.sfSmem(b, false))));
    const sfb = b.sliceTyped(full.sfb, .{ all, all, all, stage_index }, b.memrefType(.f8e4m3fn, .smem, sf_stage_align, b.layoutType(L.sfSmem(b, false))));

    // `tma_partition` keeps a singleton CTA mode; the copy views keep it too.
    const a_partition_layout = b.staticLayout(b.layoutSpec(.{ .{ 2 * Tile.a_stage_bytes, 1 }, .{1} }, .{ .{ 1, 0 }, .{0} }));
    const b_partition_layout = b.staticLayout(b.layoutSpec(.{ .{ 2 * tile.bStageBytes(), 1 }, .{1} }, .{ .{ 1, 0 }, .{0} }));
    const sf_partition_layout = b.staticLayout(b.layoutSpec(.{ .{ Tile.sf_stage_bytes, 1 }, .{1} }, .{ .{ 1, 0 }, .{0} }));

    return .{
        .a_destination = b.withLayout(a, a_partition_layout),
        .b_destination = b.withLayout(rhs, b_partition_layout),
        .sfa_destination = b.withLayout(sfa, sf_partition_layout),
        .sfb_destination = b.withLayout(sfb, sf_partition_layout),
        .load_barrier = barrierAt(b, storage.load_barriers, stage_index),
        .mma_barrier = barrierAt(b, storage.mma_barriers, stage_index),
    };
}

/// The TMA coordinates of one stage: slice the TMA-partitioned operand at
/// its output tile and L index, then at the K stage, and rebuild the
/// iterator as the Python front end does.
fn tmaSource(b: *B, partitioned: cute.View, tile_coord: anytype, k: V, comptime rows: i64, reduction_tiles: usize, copy_layout: cute.View) cute.View {
    const all = cute.AlgebraToken.all;
    const tile_k = b.sliceTyped(
        partitioned,
        tile_coord,
        b.coordTensorTypePayload(fmt(b, "(0,?{{div={d}}},?)", .{rows}), b.layoutSpec(
            .{ .{ .{ k_tile, rows }, 1 }, reduction_tiles },
            .{ .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }, b.basis(k_tile, 1, .{0}) },
        )),
    );
    const payload = fmt(b, "(?{{div={d}}},?{{div={d}}},?)", .{ k_tile, rows });
    const stage = b.sliceTyped(
        tile_k,
        .{ all, k },
        b.coordTensorTypePayload(payload, b.layoutSpec(
            .{.{ .{ k_tile, rows }, 1 }},
            .{.{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }},
        )),
    );
    return b.rebuildCoordTensorIterator(3, stage, payload, .{ k_tile, rows, 1 }, copy_layout);
}

/// Issue the four transactions that fill one mainloop stage.
fn issueTmaStage(
    b: *B,
    comptime tile: Tile,
    reduction_tiles: usize,
    partitioned_a_target: cute.View,
    partitioned_b_target: cute.View,
    stage: StageBuffers,
    k: V,
    tile_m: V,
    group: V,
    expert: V,
    exec_a: cute.Atom,
    exec_b: cute.Atom,
    exec_sfa: cute.Atom,
    exec_sfb: cute.Atom,
) void {
    const n = tile.n;
    const all = cute.AlgebraToken.all;
    const a_copy_layout = b.staticLayout(b.layoutSpec(
        .{ .{ .{ k_tile, 128 }, 1 }, .{1} },
        .{ .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }, .{0} },
    ));
    const b_copy_layout = b.staticLayout(b.layoutSpec(
        .{ .{ .{ k_tile, n }, 1 }, .{1} },
        .{ .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }, .{0} },
    ));
    // One stage moves four 512-byte atoms, one per K = 64 slice.
    const sf_copy_spec = b.layoutSpec(
        .{ .{ .{ 512, 4 }, 1 }, .{1} },
        .{ .{ .{ b.basis(1, 2, .{0}), b.basis(1, 1, .{1}) }, 0 }, .{0} },
    );
    // B's tile spans all of a group's rows, so its N-tile index is zero. Pass
    // it as a value: a static zero leaf collapses out of the coordinate tuple
    // and shifts the remaining tensor-map coordinates.
    const b_tile_n = b.cst(.i32, 0);
    const a_source = tmaSource(b, partitioned_a_target, .{ all, tile_m, all, expert }, k, 128, reduction_tiles, a_copy_layout);
    const b_source = tmaSource(b, partitioned_b_target, .{ all, b_tile_n, all, group }, k, n, reduction_tiles, b_copy_layout);
    // Scale coordinates: (byte, K atom, M block, L); a stage covers 4 K atoms.
    const k_atom = k.mul(4);
    const sfa_source = b.makeCoordTensor(.{ 0, k_atom, tile_m, expert }, b.staticCoordinateLayout(4, sf_copy_spec));
    const sfb_source = b.makeCoordTensor(.{ 0, k_atom, b.cst(.i32, 0), group }, b.staticCoordinateLayout(4, sf_copy_spec));

    var elected = b.openIf(b.electSync());
    b.mbarrierArriveExpectTx(stage.load_barrier, tile.transactionBytes());
    elected.yieldThen(.{});
    b.copy(b.setTmaBarrier(exec_a, stage.load_barrier), .{a_source}, .{stage.a_destination}, null);
    b.copy(b.setTmaBarrier(exec_b, stage.load_barrier), .{b_source}, .{stage.b_destination}, null);
    b.copy(b.setTmaBarrier(exec_sfa, stage.load_barrier), .{sfa_source}, .{stage.sfa_destination}, null);
    b.copy(b.setTmaBarrier(exec_sfb, stage.load_barrier), .{sfb_source}, .{stage.sfb_destination}, null);
}

fn accumulatorTmemAt(b: *B, comptime tile: Tile, tmem: V, stage: V) V {
    return b.addOffsetTyped(
        tmem,
        b.makeIntTupleDivBy(stage.mul(tile.n), @intCast(tile.n)),
        b.ptrTy(.f32, .tmem, 16) catch @panic("bad accumulator TMEM pointer"),
    ).value();
}

/// Warp 5 runs independently over every persistent output tile.
fn runTmaProducer(
    b: *B,
    comptime tile: Tile,
    k_tiles: usize,
    storage: StageStorage,
    active: V,
    tiles_m: V,
    expert_ids: cute.View,
    direct: ?DirectRoutes,
    partitioned_a_target: cute.View,
    partitioned_b_target: cute.View,
    exec_a: cute.Atom,
    exec_b: cute.Atom,
    exec_sfa: cute.Atom,
    exec_sfb: cute.Atom,
) void {
    const initial_index = b.cst(.i32, 0);
    const initial_phase = b.cst(.i32, 1);
    const initial_count = b.cst(.i32, 0);
    const persistent_clusters = b.gridDim().z;
    const cta_m = b.blockIdx().x;
    const cta_n = b.blockIdx().y;
    const cta_l = b.cst(.i32, 0);
    const div_m = b.fastDivmodCreate(tiles_m);
    const div_n = b.fastDivmodCreate(1);
    const initial_work = persistentWorkAt(b, b.blockIdx().z, b.cst(.i32, 0), active, tiles_m, div_m, div_n, direct);
    const carried = .{
        initial_work.m,        initial_work.n, initial_work.group, initial_work.valid,
        initial_count,         initial_index,  initial_phase,      persistent_clusters,
        initial_work.linear,   cta_m,          cta_n,              cta_l,
        initial_work.executed, tiles_m,        active,             div_m,
        div_n,
    };
    var tiles = b.openWhile(carried, typesOf(carried));
    tiles.yieldBefore(tiles.before_carried[3], tupleOf(tiles.before_carried));
    const tile_m = tiles.after_carried[0];
    const group = tiles.after_carried[2];
    const expert = expert_ids.get(.{group}, .i32);
    const first_state: PipelinePosition = .{
        .index = tiles.after_carried[5],
        .phase = tiles.after_carried[6],
    };
    const first_ready = b.mbarrierWaitParity(
        barrierAt(b, storage.mma_barriers, first_state.index),
        first_state.phase,
    );
    var stages = b.openFor(
        0,
        @as(i32, @intCast(k_tiles)),
        1,
        .{ first_ready, b.cst(.i32, 0), first_state.index, first_state.phase },
    );
    const k = stages.carried[1];
    const state: PipelinePosition = .{
        .index = stages.carried[2],
        .phase = stages.carried[3],
    };
    const buffers = stageBuffersAt(b, tile, storage, state.index);
    var pending = b.openIf(stages.carried[0].eq(false));
    b.mbarrierTryWaitParity(buffers.mma_barrier, state.phase, 10_000_000);
    pending.yieldThen(.{});
    issueTmaStage(b, tile, k_tiles, partitioned_a_target, partitioned_b_target, buffers, k, tile_m, group, expert, exec_a, exec_b, exec_sfa, exec_sfb);
    const next_count = stages.carried[1].add(1);
    const next_state = advancePipeline(b, state.index, state.phase, tile.stages);
    const next_ready = peekBarrierIfRemaining(b, storage.mma_barriers, next_count, k_tiles, next_state);
    stages.yield(.{ next_ready, next_count, next_state.index, next_state.phase });
    const next_work = advancePersistentWork(
        b,
        tiles.after_carried[8],
        tiles.after_carried[12],
        tiles.after_carried[7],
        tiles.after_carried[14],
        tiles.after_carried[13],
        tiles.after_carried[15],
        tiles.after_carried[16],
        direct,
    );
    tiles.yieldAfter(.{
        next_work.m,
        next_work.n,
        next_work.group,
        next_work.valid,
        stages.results[1],
        stages.results[2],
        stages.results[3],
        tiles.after_carried[7],
        next_work.linear,
        tiles.after_carried[9],
        tiles.after_carried[10],
        tiles.after_carried[11],
        next_work.executed,
        tiles.after_carried[13],
        tiles.after_carried[14],
        tiles.after_carried[15],
        tiles.after_carried[16],
    });

    // PipelineTmaUmma.producer_tail: drain every empty barrier before exit.
    var tail_state: PipelinePosition = .{
        .index = tiles.results[5],
        .phase = tiles.results[6],
    };
    inline for (0..tile.stages) |stage| {
        b.mbarrierTryWaitParity(
            barrierAt(b, storage.mma_barriers, tail_state.index),
            tail_state.phase,
            10_000_000,
        );
        if (stage + 1 < tile.stages) {
            tail_state = advancePipeline(b, tail_state.index, tail_state.phase, tile.stages);
        }
    }
}

/// NVFP4 mainloop objects of one pipeline stage.
const Mainloop = struct {
    a: V,
    b: V,
    accumulator: V,
    sfa: V,
    sfb: V,
    sfa_copy: cute.Atom,
    sfb_copy: cute.Atom,
    sfa_source: V,
    sfb_source: V,
    sfa_target: V,
    sfb_target: V,

    pub fn copyScales(self: Mainloop, builder: *B) void {
        builder.copy(self.sfa_copy, .{self.sfa_source}, .{self.sfa_target}, null);
        builder.copy(self.sfb_copy, .{self.sfb_source}, .{self.sfb_target}, null);
    }
};

/// Pipeline-wide UMMA and S2T descriptors derived from the complete staged
/// tensors once; `atStage` slices them at the current stage index.
const StagedMainloop = struct {
    a: cute.View,
    b: cute.View,
    sfa_source: cute.View,
    sfb_source: cute.View,
    sfa: V,
    sfb: V,
    sfa_copy: cute.Atom,
    sfb_copy: cute.Atom,
    sfa_target: V,
    sfb_target: V,
    n: i64,

    const desc_stage = "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(1,1,4):(0,0,2)\">";
    const sf_desc_stage = "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"((((32,1,1),4),1),1,1,4):((((1,1,1),0),0),0,0,32)\">";
    const sf_desc_grouped_layout = "((((32,1,1),4),1),(1,1,4)):((((1,1,1),0),0),(0,0,32))";

    pub fn atStage(self: StagedMainloop, b: *B, index: V, accumulator_tmem: V) Mainloop {
        const all = cute.AlgebraToken.all;
        const a = b.sliceTyped(self.a, .{ all, all, all, index }, b.parseType(desc_stage));
        const rhs = b.sliceTyped(self.b, .{ all, all, all, index }, b.parseType(desc_stage));
        const sfa_stage = b.sliceTyped(self.sfa_source, .{ all, all, all, all, index }, b.parseType(sf_desc_stage));
        const sfb_stage = b.sliceTyped(self.sfb_source, .{ all, all, all, all, index }, b.parseType(sf_desc_stage));
        const source_grouped_layout = b.staticValue(fmt(b, "!cute.layout<\"{s}\">", .{sf_desc_grouped_layout}));
        const source_grouped_ty = fmt(b, "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"{s}\">", .{sf_desc_grouped_layout});
        const sfa_source = b.makeView(b.getIter(sfa_stage, "!cute_nvgpu.smem_desc"), source_grouped_layout, source_grouped_ty);
        const sfb_source = b.makeView(b.getIter(sfb_stage, "!cute_nvgpu.smem_desc"), source_grouped_layout, source_grouped_ty);
        const acc_payload = fmt(b, "((128,{d}),1,1):((65536,1),0,0)", .{self.n});
        const acc_layout = b.staticValue(fmt(b, "!cute.layout<\"{s}\">", .{acc_payload}));
        const accumulator = b.makeView(
            accumulator_tmem,
            acc_layout,
            fmt(b, "!cute.memref<f32, tmem, align<16>, \"{s}\">", .{acc_payload}),
        );
        return .{
            .a = a.value(),
            .b = rhs.value(),
            .accumulator = accumulator.value(),
            .sfa = self.sfa,
            .sfb = self.sfb,
            .sfa_copy = self.sfa_copy,
            .sfb_copy = self.sfb_copy,
            .sfa_source = sfa_source.value(),
            .sfb_source = sfb_source.value(),
            .sfa_target = self.sfa_target,
            .sfb_target = self.sfb_target,
        };
    }
};

fn makeStagedMainloop(
    b: *B,
    tiled_mma: cute.Atom,
    comptime n: i64,
    comptime stages: usize,
    smem_a: V,
    smem_b: V,
    smem_sfa: V,
    smem_sfb: V,
    scale_tmem: V,
) StagedMainloop {
    // Descriptor strides are in 16-byte units: one 64-deep K slice of packed
    // FP4 is 32 bytes, a 128x256 A stage 16 KiB and an Nx256 B stage N*128 bytes.
    const frag_a = b.mmaMakeFragment(
        tiled_mma,
        smem_a,
        0,
        fmt(b, "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(1,1,4,{d}):(0,0,2,{d})\">", .{ stages, Tile.a_stage_bytes / 16 }),
    );
    const frag_b = b.mmaMakeFragment(
        tiled_mma,
        smem_b,
        1,
        fmt(b, "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(1,1,4,{d}):(0,0,2,{d})\">", .{ stages, n * 8 }),
    );

    // TMEM columns: two N-wide accumulator stages, then 16 SFA and 16 SFB
    // columns: four K = 64 slices of four E4M3 scales per lane each.
    const sf_payload = "((((32,4),4),(16,4)),1,4):((((262144,4),8388608),(0,1)),0,16)";
    const sf_layout = b.staticValue("!cute.layout<\"" ++ sf_payload ++ "\">");
    const sfa_col = b.addOffset(scale_tmem, b.staticValue(fmt(b, "!cute.int_tuple<\"{d}\">", .{2 * n})), "!cute.ptr<f32, tmem, align<16>>");
    const sfb_col = b.addOffset(scale_tmem, b.staticValue(fmt(b, "!cute.int_tuple<\"{d}\">", .{2 * n + 16})), "!cute.ptr<f32, tmem, align<16>>");
    const sfa_ptr = b.recastIter(sfa_col, "!cute.ptr<f8E4M3FN, tmem, align<16>>");
    const sfb_ptr = b.recastIter(sfb_col, "!cute.ptr<f8E4M3FN, tmem, align<16>>");
    const sf_memref = "!cute.memref<f8E4M3FN, tmem, align<16>, \"" ++ sf_payload ++ "\">";
    const sfa = b.makeView(sfa_ptr, sf_layout, sf_memref);
    const sfb = b.makeView(sfb_ptr, sf_layout, sf_memref);

    const smem_filtered_ty = fmt(b, "!cute.memref<f8E4M3FN, smem, align<1024>, \"((((32,4),1),(1,4)),1,4,{d}):((((16,4),0),(0,1)),0,512,2048)\">", .{stages});
    const tmem_filtered_ty = "!cute.memref<f8E4M3FN, tmem, align<16>, \"((((32,4),4),(1,4)),1,4):((((262144,4),8388608),(0,1)),0,16)\">";
    const sfa_smem_filtered = b.filterZeros(smem_sfa, null, smem_filtered_ty);
    const sfb_smem_filtered = b.filterZeros(smem_sfb, null, smem_filtered_ty);
    const sfa_tmem_filtered = b.filterZeros(sfa, null, tmem_filtered_ty);
    const sfb_tmem_filtered = b.filterZeros(sfb, null, tmem_filtered_ty);

    const copy_atom_ty = "!cute_nvgpu.atom.s2t_copy<f8E4M3FN, 32 DP, 128 bit, 1 cta, x4>";
    const copy_atom = b.makeAtom(copy_atom_ty, .{});
    const copy_ty = "!cute.tiled_copy<" ++ copy_atom_ty ++ ", layout_copy_tv = <\"(1,(32,(4,4),4)):(0,(1,(32,128),512))\">, tiler_mn = <\"[(32,4,16):(1,512,32);1:0;1:0]\">>";
    const sfa_copy = b.makeS2tCopy(copy_atom, sfa_tmem_filtered, copy_ty);
    const sfb_copy = b.makeS2tCopy(copy_atom, sfb_tmem_filtered, copy_ty);

    const source_partition_ty = fmt(b, "!cute.memref<f8E4M3FN, smem, align<1024>, \"((((32,4,4),4),1),1,1,4,{d}):((((16,4,4),0),0),0,0,512,2048)\">", .{stages});
    const source_desc_ty = fmt(b, "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"((((32,1,1),4),1),1,1,4,{d}):((((1,1,1),0),0),0,0,32,128)\">", .{stages});
    const target_partition_ty = "!cute.memref<f8E4M3FN, tmem, align<16>, \"(((32,16,4),1),1,1,4):(((262144,1,8388608),0),0,0,16)\">";
    const target_grouped_payload = "(((32,16,4),1),(1,1,4)):(((262144,1,8388608),0),(0,0,16))";
    const target_grouped_ty = "!cute.memref<f8E4M3FN, tmem, align<16>, \"" ++ target_grouped_payload ++ "\">";
    const sfa_src_partition = b.tiledCopyPartition(sfa_copy, sfa_smem_filtered, .{0}, false, source_partition_ty);
    const sfb_src_partition = b.tiledCopyPartition(sfb_copy, sfb_smem_filtered, .{0}, false, source_partition_ty);
    const sfa_desc = b.s2tSmemDescriptor(sfa_copy, sfa_src_partition, source_desc_ty);
    const sfb_desc = b.s2tSmemDescriptor(sfb_copy, sfb_src_partition, source_desc_ty);
    const sfa_dst_partition = b.tiledCopyPartition(sfa_copy, sfa_tmem_filtered, .{0}, true, target_partition_ty);
    const sfb_dst_partition = b.tiledCopyPartition(sfb_copy, sfb_tmem_filtered, .{0}, true, target_partition_ty);
    const target_grouped_layout = b.staticValue("!cute.layout<\"" ++ target_grouped_payload ++ "\">");
    const sfa_dst = b.makeView(b.getIter(sfa_dst_partition, "!cute.ptr<f8E4M3FN, tmem, align<16>>"), target_grouped_layout, target_grouped_ty);
    const sfb_dst = b.makeView(b.getIter(sfb_dst_partition, "!cute.ptr<f8E4M3FN, tmem, align<16>>"), target_grouped_layout, target_grouped_ty);

    return .{
        .a = frag_a,
        .b = frag_b,
        .sfa_source = sfa_desc,
        .sfb_source = sfb_desc,
        .sfa = sfa.value(),
        .sfb = sfb.value(),
        .sfa_copy = sfa_copy,
        .sfb_copy = sfb_copy,
        .sfa_target = sfa_dst.value(),
        .sfb_target = sfb_dst.value(),
        .n = n,
    };
}

/// Warp 4 drains the A/B ring into one of two independent accumulator stages.
fn runMmaConsumer(
    b: *B,
    comptime tile: Tile,
    k_tiles: usize,
    storage: StageStorage,
    tiled_mma: cute.Atom,
    tmem_holding: V,
    active: V,
    tiles_m: V,
) void {
    const tmem = b.retrieveTmemPtr(tmem_holding, .f32, 16);
    const staged = stagedOperands(b, tile, storage);
    const staged_mainloop = makeStagedMainloop(
        b,
        tiled_mma,
        tile.n,
        tile.stages,
        staged.a.value(),
        staged.b.value(),
        staged.sfa.value(),
        staged.sfb.value(),
        tmem,
    );
    const persistent_clusters = b.gridDim().z;
    const div_m = b.fastDivmodCreate(tiles_m);
    const div_n = b.fastDivmodCreate(1);
    const initial_work = persistentWorkAt(b, b.blockIdx().z, b.cst(.i32, 0), active, tiles_m, div_m, div_n, null);
    const carried = .{
        initial_work.m,        initial_work.n, initial_work.group, initial_work.valid,
        b.cst(.i32, 0),        b.cst(.i32, 0), b.cst(.i32, 0),     tiled_mma.value(),
        b.cst(.i32, 0),        b.cst(.i32, 0), b.cst(.i32, 1),     persistent_clusters,
        initial_work.linear,   b.blockIdx().x, b.blockIdx().y,     b.cst(.i32, 0),
        initial_work.executed, tiles_m,        active,             div_m,
        div_n,
    };
    var tiles = b.openWhile(carried, typesOf(carried));
    tiles.yieldBefore(tiles.before_carried[3], tupleOf(tiles.before_carried));
    const first_state: PipelinePosition = .{
        .index = tiles.after_carried[5],
        .phase = tiles.after_carried[6],
    };
    const first_ready = b.mbarrierWaitParity(
        barrierAt(b, storage.load_barriers, first_state.index),
        first_state.phase,
    );
    const accumulator_state: PipelinePosition = .{
        .index = tiles.after_carried[9],
        .phase = tiles.after_carried[10],
    };
    const accumulator_stage = accumulator_state.index;
    const accumulator_empty = barrierAt(b, storage.accumulator_empty_barriers, accumulator_stage);
    b.mbarrierTryWaitParity(accumulator_empty, accumulator_state.phase, 10_000_000);

    const persistent_mma: cute.Atom = .{ .inner = tiles.after_carried[7].inner, .kernel = b };
    const initial_mma = b.setMmaAccumulate(persistent_mma, b.cst(.i1, false));
    var stages = b.openFor(
        0,
        @as(i32, @intCast(k_tiles)),
        1,
        .{ first_ready, initial_mma.value(), b.cst(.i32, 0), first_state.index, first_state.phase },
    );
    const k = stages.carried[2];
    const state: PipelinePosition = .{ .index = stages.carried[3], .phase = stages.carried[4] };
    var pending = b.openIf(stages.carried[0].eq(false));
    b.mbarrierTryWaitParity(barrierAt(b, storage.load_barriers, state.index), state.phase, 10_000_000);
    pending.yieldThen(.{});
    const accumulator_tmem = accumulatorTmemAt(b, tile, tmem, accumulator_stage);
    const mainloop = staged_mainloop.atStage(b, state.index, accumulator_tmem);
    mainloop.copyScales(b);
    const loop_mma: cute.Atom = .{ .inner = stages.carried[1].inner, .kernel = b };
    const mma = b.setMmaAccumulate(loop_mma, k.ne(0));
    b.gemm(
        mma,
        mainloop.accumulator,
        .{ mainloop.a, mainloop.sfa },
        .{ mainloop.b, mainloop.sfb },
        mainloop.accumulator,
    );
    var commit_lane = b.openIf(b.electSync());
    b.tcgen05Commit(barrierAt(b, storage.mma_barriers, state.index));
    commit_lane.yieldThen(.{});
    const next_count = stages.carried[2].add(1);
    const next_state = advancePipeline(b, state.index, state.phase, tile.stages);
    const next_ready = peekBarrierIfRemaining(b, storage.load_barriers, next_count, k_tiles, next_state);
    stages.yield(.{ next_ready, mma.value(), next_count, next_state.index, next_state.phase });

    var full_lane = b.openIf(b.electSync());
    b.tcgen05Commit(barrierAt(b, storage.accumulator_full_barriers, accumulator_stage));
    full_lane.yieldThen(.{});
    const next_accumulator_state = advancePipeline(b, accumulator_state.index, accumulator_state.phase, accumulator_stages);
    const next_acc_count = tiles.after_carried[8].add(1);
    const next_work = advancePersistentWork(
        b,
        tiles.after_carried[12],
        tiles.after_carried[16],
        tiles.after_carried[11],
        tiles.after_carried[18],
        tiles.after_carried[17],
        tiles.after_carried[19],
        tiles.after_carried[20],
        null,
    );
    tiles.yieldAfter(.{
        next_work.m,
        next_work.n,
        next_work.group,
        next_work.valid,
        stages.results[2],
        stages.results[3],
        stages.results[4],
        stages.results[1],
        next_acc_count,
        next_accumulator_state.index,
        next_accumulator_state.phase,
        tiles.after_carried[11],
        next_work.linear,
        tiles.after_carried[13],
        tiles.after_carried[14],
        tiles.after_carried[15],
        next_work.executed,
        tiles.after_carried[17],
        tiles.after_carried[18],
        tiles.after_carried[19],
        tiles.after_carried[20],
    });

    // PipelineUmmaAsync.producer_tail.
    var accumulator_tail: PipelinePosition = .{
        .index = tiles.results[9],
        .phase = tiles.results[10],
    };
    inline for (0..accumulator_stages - 1) |_| {
        accumulator_tail = advancePipeline(b, accumulator_tail.index, accumulator_tail.phase, accumulator_stages);
    }
    b.mbarrierTryWaitParity(
        barrierAt(b, storage.accumulator_empty_barriers, accumulator_tail.index),
        accumulator_tail.phase,
        10_000_000,
    );
}

/// Epilogue operands; which fields are set depends on the `Epilogue`.
const EpilogueOperands = struct {
    expert_ids: cute.View,
    alpha: cute.View,
    group_sizes: cute.View,
    route_inverse: cute.View,
    routing_weights: ?cute.View = null,
    act_scale: ?cute.View = null,
    q: ?cute.View = null,
    s: ?cute.View = null,
    rows: ?cute.View = null,
    direct: bool,
    group_rows: i64 = 0,
    swiglu_limit: ?f32 = null,
};

/// Warps 0-3 drain each TMEM accumulator stage into registers, release it to
/// the MMA warp, then write the tile out according to `epilogue`.
fn runEpilogue(
    b: *B,
    comptime tile: Tile,
    comptime epilogue: Epilogue,
    storage: StageStorage,
    tmem_holding: V,
    active: V,
    tiles_m: V,
    direct: ?DirectRoutes,
    operands: EpilogueOperands,
) void {
    const n = tile.n;
    const tid = b.threadIdx().x;
    const warp = b.makeWarpUniform(b.makeWarpUniform(tid.div(32)));
    const tmem = b.retrieveTmemPtr(tmem_holding, .f32, 16);

    const div_m = b.fastDivmodCreate(tiles_m);
    const div_n = b.fastDivmodCreate(1);
    const initial_work = persistentWorkAt(b, b.blockIdx().z, b.cst(.i32, 0), active, tiles_m, div_m, div_n, direct);
    const carried = .{
        initial_work.m,        initial_work.n, initial_work.group, initial_work.valid,
        b.cst(.i32, 0),        b.cst(.i32, 0), b.cst(.i32, 0),     initial_work.linear,
        initial_work.executed,
    };
    var tiles = b.openWhile(carried, .{
        initial_work.m.type_(),
        initial_work.n.type_(),
        initial_work.group.type_(),
        cute.DType.i32.toMlir(b.ctx),
        cute.DType.i32.toMlir(b.ctx),
        cute.DType.i32.toMlir(b.ctx),
        initial_work.linear.type_(),
        initial_work.executed.type_(),
    });
    tiles.yieldBefore(tiles.before_carried[3], .{
        tiles.before_carried[0],
        tiles.before_carried[1],
        tiles.before_carried[2],
        tiles.before_carried[4],
        tiles.before_carried[5],
        tiles.before_carried[6],
        tiles.before_carried[7],
        tiles.before_carried[8],
    });
    const tile_m = tiles.after_carried[0];
    const group = tiles.after_carried[2];
    const accumulator_state: PipelinePosition = .{
        .index = tiles.after_carried[4],
        .phase = tiles.after_carried[5],
    };
    const stage = accumulator_state.index;
    // Per-expert scalars do not depend on the accumulator.
    const expert = operands.expert_ids.get(.{group}, .i32);
    const alpha = operands.alpha.get(.{expert}, .f32);
    const act_scale = if (operands.act_scale) |scales| scales.get(.{expert}, .f32) else null;
    b.mbarrierTryWaitParity(barrierAt(b, storage.accumulator_full_barriers, stage), accumulator_state.phase, 10_000_000);

    const accumulator_tmem = accumulatorTmemAt(b, tile, tmem, stage);
    const warp_tmem = b.addOffsetTyped(
        accumulator_tmem,
        b.makeIntTupleDivBy(warp.mul(32 * 65536), 4),
        b.ptrTy(.f32, .tmem, 16) catch @panic("bad epilogue TMEM pointer"),
    ).value();
    const chunk_rows: usize = comptime @intCast(tile.chunkRows());
    const tile_chunks = comptime tile.chunks();
    inline for (0..tile_chunks) |chunk| {
        const chunk_tmem = if (chunk == 0) warp_tmem else b.addOffsetTyped(
            warp_tmem,
            b.makeIntTuple(.{@as(i32, @intCast(chunk * chunk_rows))}),
            b.ptrTy(.f32, .tmem, offsetAlignment(16, @intCast(chunk * chunk_rows), 4)) catch @panic("bad epilogue TMEM pointer"),
        ).value();
        const routed = routedLanes(b, tile, operands, group, chunk);
        const values = b.tmemLoadVector(chunk_tmem, @intCast(chunk_rows), .{ .num_dp = 32, .num_b = 32, .num_rep = @intCast(chunk_rows) });
        b.fenceTmemLoad();
        if (chunk + 1 == tile_chunks) {
            b.mbarrierArrive(barrierAt(b, storage.accumulator_empty_barriers, stage), 1);
        }
        var columns: [@intCast(@min(n, 32))]V = undefined;
        inline for (&columns, 0..) |*column, i| column.* = b.vectorExtract(values, i, .i32).bitCast(.f32).mul(alpha);
        switch (epilogue) {
            .swiglu_nvfp4 => storeTileSwiglu(b, tile, operands, routed, tile_m, group, act_scale.?, &columns, chunk),
            .route_rows => storeTileRows(b, tile, operands, routed, tile_m, &columns, chunk),
        }
    }
    const next_accumulator_state = advancePipeline(b, accumulator_state.index, accumulator_state.phase, accumulator_stages);
    const next_work = advancePersistentWork(b, tiles.after_carried[6], tiles.after_carried[7], b.gridDim().z, active, tiles_m, div_m, div_n, direct);
    tiles.yieldAfter(.{
        next_work.m,
        next_work.n,
        next_work.group,
        next_work.valid,
        tiles.after_carried[3].add(1),
        next_accumulator_state.index,
        next_accumulator_state.phase,
        next_work.linear,
        next_work.executed,
    });

    var release_warp = b.openIf(warp.eq(0));
    b.relinquishTmemAllocPermit(.{});
    release_warp.yieldThen(.{});
    b.namedBarrier(1, 128);
    var dealloc_warp = b.openIf(warp.eq(0));
    b.deallocTmem(tmem, tile.tmem_columns, .{});
    dealloc_warp.yieldThen(.{});
}

/// Per-row routing data of one expert group, spread across a warp: lane `l`
/// holds routed row `l` of the current chunk.
const RoutedLanes = struct {
    count: V,
    /// Token/top-k route of the lane's row; 0 for padding rows.
    route: V,
    /// Routing weight of the lane's row, when this projection applies it.
    weight: ?V,
};

fn routedLanes(b: *B, comptime tile: Tile, operands: EpilogueOperands, group: V, comptime chunk: usize) RoutedLanes {
    const weights = operands.routing_weights;
    if (operands.direct) return .{
        .count = b.cst(.i32, 1),
        .route = group,
        .weight = if (weights) |w| w.get(.{group}, .f32) else null,
    };
    const lane = b.threadIdx().x.rem(32);
    const count = operands.group_sizes.get(.{group}, .i32);
    // Lanes past the tile width alias row 0, which is always live.
    const chunk_rows = comptime tile.chunkRows();
    const in_chunk = b.select(lane.lt(@as(i32, @intCast(chunk_rows))), lane, b.cst(.i32, 0));
    const row_in_group = if (chunk == 0) in_chunk else in_chunk.add(@as(i32, @intCast(@as(i64, @intCast(chunk)) * chunk_rows)));
    const live = row_in_group.lt(count);
    const inverse = operands.route_inverse.get(.{group.mul(tile.n).add(row_in_group)}, .i32);
    const route = b.select(live, inverse, b.cst(.i32, 0));
    const weight = if (weights) |w| w.get(.{route}, .f32) else null;
    return .{ .count = count, .route = route, .weight = weight };
}

const swiglu_chunk = 8;

/// Thread `t` owns output row `128 * tile_m + t` of the interleaved
/// projection: even threads hold a gate and their odd neighbour the matching
/// up value, so each even thread computes one SwiGLU activation per routed
/// row. A warp's 16 activations are exactly one NVFP4 block, whose maximum is
/// a single warp reduction. Activation `a` of row `r` lands in nibble `a` of
/// row `r` of `q`, and its scale at the 128x4 position of `(r, a / 16)`.
fn storeTileSwiglu(b: *B, comptime tile: Tile, operands: EpilogueOperands, routed: RoutedLanes, tile_m: V, group: V, act_scale: V, columns: []const V, comptime chunk: usize) void {
    if (operands.direct)
        swigluRows(b, tile, operands, routed, tile_m, group, act_scale, columns, 1, false, chunk)
    else
        swigluRows(b, tile, operands, routed, tile_m, group, act_scale, columns, comptime @intCast(tile.chunkRows()), true, chunk);
}

fn swigluRows(b: *B, comptime tile: Tile, operands: EpilogueOperands, routed: RoutedLanes, tile_m: V, group: V, act_scale: V, columns: []const V, comptime live: usize, comptime skip_padding: bool, comptime chunk: usize) void {
    const n = comptime tile.chunkRows();
    const chunk_first: i32 = comptime @intCast(chunk * @as(usize, @intCast(tile.chunkRows())));
    const tid = b.threadIdx().x;
    const lane = tid.rem(32);
    const warp = b.makeWarpUniform(b.makeWarpUniform(tid.div(32)));
    const is_gate = tid.rem(2).eq(0);
    const first_row = group.mul(operands.group_rows);
    const f32_type = cute.DType.f32.toMlir(b.ctx);
    const row_chunks = std.math.divCeil(usize, live, swiglu_chunk) catch unreachable;

    var activations: [@intCast(n)]V = undefined;
    // Lane `c` collects the block maximum of column `c`.
    var lane_maximum = b.cst(.f32, 0);
    for (0..row_chunks) |row_chunk| {
        const first = row_chunk * swiglu_chunk;
        const last = @min(first + swiglu_chunk, live);
        const guarded = skip_padding and (row_chunk > 0 or chunk > 0);
        const types: [swiglu_chunk + 1]*const mlir.Type = @splat(f32_type);
        var guard: ?@TypeOf(b.openIfElse(routed.count, tupleOf(types))) = if (guarded) b.openIfElse(routed.count.gt(chunk_first + @as(i32, @intCast(first))), tupleOf(types)) else null;
        var chunk_maximum = lane_maximum;
        for (columns[first..last], activations[first..last], first..) |value, *activation, column| {
            const c: i32 = @intCast(column);
            // The reference rounds the projection to BF16.
            var gate = value.to(.bf16).to(.f32);
            var up_value = b.shuffleXor(value, 1).to(.bf16).to(.f32);
            if (operands.swiglu_limit) |limit| {
                gate = gate.minimum(limit);
                up_value = up_value.maximum(-limit).minimum(limit);
            }
            // up * SiLU(gate), exp(-g) = 2^(-g * log2 e).
            const exp_neg = b.unaryF32("ex2.approx.ftz.f32", gate.mul(-std.math.log2e));
            var result = up_value.mul(gate).mul(b.unaryF32("rcp.approx.ftz.f32", exp_neg.add(1.0)));
            if (routed.weight) |weight| result = result.mul(b.shuffleIdx(weight, c));
            // The NVFP4 quantizer reads BF16 activations.
            result = result.to(.bf16).to(.f32);
            // Odd threads hold no activation and must not raise the maxima.
            activation.* = b.select(is_gate, result, b.cst(.f32, 0));
            const maximum = b.warpMaxNonNegative(activation.*.abs());
            chunk_maximum = b.select(lane.eq(c), maximum, chunk_maximum);
        }
        if (guard) |*g| {
            var then_values: [swiglu_chunk + 1]V = @splat(chunk_maximum);
            for (activations[first..last], then_values[0 .. last - first]) |activation, *t| t.* = activation;
            g.yieldThen(tupleOf(then_values));
            var else_values: [swiglu_chunk + 1]V = @splat(b.cst(.f32, 0));
            else_values[swiglu_chunk] = lane_maximum;
            g.yieldElse(tupleOf(else_values));
            for (activations[first..last], g.results[0 .. last - first]) |*activation, result| activation.* = result;
            lane_maximum = g.results[swiglu_chunk];
        } else {
            lane_maximum = chunk_maximum;
        }
    }
    const scale = nvfp4.blockScale(b, lane_maximum, act_scale);

    // Lane `16 h` writes the word of activations `16 * warp + 8 h ..`.
    const word = tile_m.mul(8).add(warp.mul(2)).add(lane.div(16));
    const word_lane = lane.rem(16).eq(0);
    for (0..row_chunks) |row_chunk| {
        const first = row_chunk * swiglu_chunk;
        const last = @min(first + swiglu_chunk, live);
        var guard: ?@TypeOf(b.openIf(routed.count)) = if (skip_padding and (chunk > 0 or first > 0)) b.openIf(routed.count.gt(chunk_first + @as(i32, @intCast(first)))) else null;
        for (activations[first..last], first..) |activation, column| {
            const c: i32 = @intCast(column);
            const quantized = activation.mul(b.shuffleIdx(scale.multiplier, c));
            // Gate lane `4i` pairs with gate lane `4i + 2`, then pairs of
            // bytes and pairs of halves meet at lanes `8i` and `16i`.
            const pair = b.packFp4x2(quantized, b.shuffleXor(quantized, 2));
            const half = pair.bitOr(b.shuffleXor(pair, 4).shl(8));
            const packed_word = half.bitOr(b.shuffleXor(half, 8).shl(16));
            var writer = b.openIf(word_lane);
            operands.q.?.set(.{ first_row.add(chunk_first + c), word }, packed_word);
            writer.yieldThen(.{});
        }
        if (guard) |*g| g.yieldThen(.{});
    }

    // Lane `c` stores the scale of row `c`, block `4 * tile_m + warp`.
    var scale_lane = b.openIf(lane.lt(@as(i32, @intCast(n))).bitAnd(lane.add(chunk_first).lt(routed.count)));
    const swizzled = tile_m.mul(512).add(lane.mul(16)).add(chunk * 4).add(warp);
    operands.s.?.set(.{ group, swizzled }, scale.bits.to(.i8));
    scale_lane.yieldThen(.{});
}

/// Store each live routed row at its token/top-k route, so the final
/// reduction reads rows in route order.
fn storeTileRows(b: *B, comptime tile: Tile, operands: EpilogueOperands, routed: RoutedLanes, tile_m: V, columns: []const V, comptime chunk: usize) void {
    const chunk_first: i32 = comptime @intCast(chunk * @as(usize, @intCast(tile.chunkRows())));
    const output_column = tile_m.mul(128).add(b.threadIdx().x);
    if (operands.direct) {
        const value = if (routed.weight) |weight| columns[0].mul(weight) else columns[0];
        operands.rows.?.set(.{ routed.route, output_column }, value.to(.bf16));
        return;
    }
    var routes: [32]V = undefined;
    var weights: [32]?V = @splat(null);
    for (routes[0..columns.len], weights[0..columns.len], 0..) |*route, *weight, column| {
        route.* = b.shuffleIdx(routed.route, @as(i32, @intCast(column)));
        if (routed.weight) |w| weight.* = b.shuffleIdx(w, @as(i32, @intCast(column)));
    }
    for (columns, routes[0..columns.len], weights[0..columns.len], 0..) |value, route, weight, column| {
        var live: ?@TypeOf(b.openIf(routed.count)) = if (column == 0 and chunk == 0) null else b.openIf(routed.count.gt(chunk_first + @as(i32, @intCast(column))));
        const scaled = if (weight) |w| value.mul(w) else value;
        operands.rows.?.set(.{ route, output_column }, scaled.to(.bf16));
        if (live) |*l| l.yieldThen(.{});
    }
}

fn buildUmmaDevice(b: *B, cfg: Config, comptime tile: Tile, comptime epilogue: Epilogue, d: anytype) void {
    const L = Layouts(tile);
    const n = tile.n;
    const stages = tile.stages;
    const shared = b.allocSmemStorage(
        .i8,
        1024,
        b.layoutType(b.layoutSpec(.{tile.smem_bytes}, .{1})),
        1,
        tile.fields(b),
    );
    const shared_base = b.getIterTyped(
        shared,
        b.ptrTy(.i8, .smem, 1024) catch @panic("bad shared base pointer"),
    ).value();

    const tmem_holding = sharedPointerAt(b, shared_base, tile.tmem_holding, .i32, 8);
    const storage: StageStorage = .{
        .c = sharedPointerAt(b, shared_base, tile.c, .f32, 1024),
        .a = sharedPointerAt(b, shared_base, tile.a, .i8, 1024),
        .b = sharedPointerAt(b, shared_base, tile.b, .i8, 1024),
        .sfa = sharedPointerAt(b, shared_base, tile.sfa, .f8e4m3fn, 1024),
        .sfb = sharedPointerAt(b, shared_base, tile.sfb, .f8e4m3fn, 1024),
        .load_barriers = sharedPointerAt(b, shared_base, 0, .i64, 1024),
        .mma_barriers = sharedPointerAt(b, shared_base, tile.empty_barriers, .i64, 8),
        .accumulator_full_barriers = sharedPointerAt(b, shared_base, tile.accumulator_full, .i64, 16),
        .accumulator_empty_barriers = sharedPointerAt(b, shared_base, tile.accumulator_empty, .i64, 16),
    };

    const tid = b.threadIdx().x;
    const warp = b.makeWarpUniform(b.makeWarpUniform(tid.div(32)));
    const tiled_mma: cute.Atom = .{ .inner = d.mma.inner, .kernel = b };

    // coordinate tensor -> local_tile -> tiled_mma.partition -> group modes
    // -> tma_partition, with the result types of the CuTe DSL.
    const m_tiles = @divExact(cfg.m, 128);
    const k_tiles = @divExact(cfg.k, k_tile);
    const m_step = b.basis(128, 1, .{1});
    const k_step = b.basis(k_tile, 1, .{0});
    const a_local_spec = b.layoutSpec(
        .{ 128, k_tile, m_tiles, k_tiles, cfg.experts },
        .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), m_step, k_step, b.basis(1, 1, .{2}) },
    );
    const b_local_spec = b.layoutSpec(
        .{ n, k_tile, 1, k_tiles, cfg.groups },
        .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), 0, k_step, b.basis(1, 1, .{2}) },
    );
    const all_tiles = .{ cute.AlgebraToken.all, cute.AlgebraToken.all, cute.AlgebraToken.all };
    const tiled_a = b.localTileTyped(d.gA, b.makeTile(&.{ 128, k_tile }), all_tiles, b.coordTensorType(3, a_local_spec), null);
    const tiled_b = b.localTileTyped(d.gB, b.makeTile(&.{ n, k_tile }), all_tiles, b.coordTensorType(3, b_local_spec), null);

    const a_mma_spec = b.layoutSpec(
        .{ .{ 128, 64 }, 1, 4, m_tiles, k_tiles, cfg.experts },
        .{ .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}) }, 0, b.basis(64, 1, .{0}), m_step, k_step, b.basis(1, 1, .{2}) },
    );
    const b_mma_spec = b.layoutSpec(
        .{ .{ n, 64 }, 1, 4, 1, k_tiles, cfg.groups },
        .{ .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}) }, 0, b.basis(64, 1, .{0}), 0, k_step, b.basis(1, 1, .{2}) },
    );
    const mma_a = b.tiledMmaPartitionTyped(tiled_mma, tiled_a, .{0}, 0, b.coordTensorType(3, a_mma_spec));
    const mma_b = b.tiledMmaPartitionTyped(tiled_mma, tiled_b, .{0}, 1, b.coordTensorType(3, b_mma_spec));

    const a_grouped_layout = b.staticLayout(b.layoutSpec(
        .{ .{ .{ 128, 64 }, 1, 4 }, m_tiles, k_tiles, cfg.experts },
        .{ .{ .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}) }, 0, b.basis(64, 1, .{0}) }, m_step, k_step, b.basis(1, 1, .{2}) },
    ));
    const b_grouped_layout = b.staticLayout(b.layoutSpec(
        .{ .{ .{ n, 64 }, 1, 4 }, 1, k_tiles, cfg.groups },
        .{ .{ .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}) }, 0, b.basis(64, 1, .{0}) }, b.basis(n, 1, .{1}), k_step, b.basis(1, 1, .{2}) },
    ));
    const staged = stagedOperands(b, tile, storage);
    const smem_a_grouped = b.withLayout(staged.a, b.staticLayout(b.layoutSpec(
        .{ .{ .{ 128, 64 }, 1, 4 }, stages },
        .{ .{ .{ k_tile, 1 }, 0, 64 }, 2 * Tile.a_stage_bytes },
    )));
    const smem_b_grouped = b.withLayout(staged.b, b.staticLayout(b.layoutSpec(
        .{ .{ .{ n, 64 }, 1, 4 }, stages },
        .{ .{ .{ k_tile, 1 }, 0, 64 }, 2 * tile.bStageBytes() },
    )));
    const one_cta = b.staticLayout(b.layoutSpec(.{1}, .{0}));
    const a_partition_layout = b.layoutSpec(.{ .{ 2 * Tile.a_stage_bytes, 1 }, stages }, .{ .{ 1, 0 }, 2 * Tile.a_stage_bytes });
    const b_partition_layout = b.layoutSpec(.{ .{ 2 * tile.bStageBytes(), 1 }, stages }, .{ .{ 1, 0 }, 2 * tile.bStageBytes() });
    const a_target_layout = b.layoutSpec(
        .{ .{ .{ k_tile, 128 }, 1 }, m_tiles, k_tiles, cfg.experts },
        .{ .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }, m_step, k_step, b.basis(1, 1, .{2}) },
    );
    const b_target_layout = b.layoutSpec(
        .{ .{ .{ k_tile, n }, 1 }, 1, k_tiles, cfg.groups },
        .{ .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }, b.basis(n, 1, .{1}), k_step, b.basis(1, 1, .{2}) },
    );
    const partitioned_a = b.tmaPartitionTyped(
        .{ .inner = d.tma_a.inner, .kernel = b },
        .{0},
        one_cta,
        smem_a_grouped,
        .{b.withLayout(mma_a, a_grouped_layout)},
        b.memrefTypeFromPointer(L.fp4SmemPtrTy(b), b.layoutType(a_partition_layout)),
        &.{b.coordTensorType(3, a_target_layout)},
    );
    const partitioned_b = b.tmaPartitionTyped(
        .{ .inner = d.tma_b.inner, .kernel = b },
        .{0},
        one_cta,
        smem_b_grouped,
        .{b.withLayout(mma_b, b_grouped_layout)},
        b.memrefTypeFromPointer(L.fp4SmemPtrTy(b), b.layoutType(b_partition_layout)),
        &.{b.coordTensorType(3, b_target_layout)},
    );

    const exec_a = b.makeExecTmaTyped(.{ .inner = d.tma_a.inner, .kernel = b }, b.tmaLoadExecAtomType(L.tmaA(b, cfg)));
    const exec_b = b.makeExecTmaTyped(.{ .inner = d.tma_b.inner, .kernel = b }, b.tmaLoadExecAtomType(L.tmaB(b, cfg)));
    const exec_sfa = b.makeExecTmaTyped(.{ .inner = d.tma_sfa.inner, .kernel = b }, b.tmaLoadExecAtomType(L.tmaSf(b)));
    const exec_sfb = b.makeExecTmaTyped(.{ .inner = d.tma_sfb.inner, .kernel = b }, b.tmaLoadExecAtomType(L.tmaSf(b)));

    const expert_ids: cute.View = .{ .inner = d.expert_ids.inner, .kernel = b };
    const active_count: cute.View = .{ .inner = d.active_count.inner, .kernel = b };
    const direct: ?DirectRoutes = if (cfg.direct) .{ .expert_ids = expert_ids, .routes = @intCast(cfg.groups) } else null;
    const active = if (direct) |routes| routes.count(b) else active_count.get(.{0}, .i32);

    // Warp 0 initializes the barriers; one elected lane per barrier array.
    var init_warp = b.openIf(warp.eq(0));
    const barrier_arrays = .{
        .{ storage.load_barriers, stages, 1 },
        .{ storage.mma_barriers, stages, 1 },
        .{ storage.accumulator_full_barriers, accumulator_stages, 1 },
        .{ storage.accumulator_empty_barriers, accumulator_stages, 128 },
    };
    inline for (barrier_arrays) |array| {
        var lane = b.openIf(b.electSync());
        inline for (0..array[1]) |stage| {
            b.mbarrierInit(barrierAt(b, array[0], b.cst(.i32, stage)), array[2]);
        }
        lane.yieldThen(.{});
    }
    b.fenceMbarrierInit();
    init_warp.yieldThen(.{});
    b.syncThreads();

    const reduction_tiles: usize = @intCast(k_tiles);
    var tma_warp = b.openIf(warp.eq(5));
    runTmaProducer(b, tile, reduction_tiles, storage, active, d.tiles_m, expert_ids, direct, partitioned_a.targets[0], partitioned_b.targets[0], exec_a, exec_b, exec_sfa, exec_sfb);
    tma_warp.yieldThen(.{});

    var mma_warp = b.openIf(warp.eq(4));
    b.namedBarrier(2, 160);
    runMmaConsumer(b, tile, reduction_tiles, storage, tiled_mma, tmem_holding, active, d.tiles_m);
    mma_warp.yieldThen(.{});

    var epilogue_warps = b.openIf(warp.lt(4));
    var alloc_warp = b.openIf(warp.eq(0));
    b.allocTmem(tile.tmem_columns, tmem_holding, .{});
    alloc_warp.yieldThen(.{});
    b.namedBarrier(2, 160);
    const shared_operands: EpilogueOperands = .{
        .expert_ids = expert_ids,
        .alpha = view(b, d.alpha),
        .group_sizes = view(b, d.group_sizes),
        .route_inverse = view(b, d.route_inverse),
        .routing_weights = if (cfg.routing_weights) view(b, d.routing_weights) else null,
        .direct = cfg.direct,
        .group_rows = cfg.groupRows(),
        .swiglu_limit = cfg.swiglu_limit,
    };
    var operands = shared_operands;
    switch (epilogue) {
        .swiglu_nvfp4 => {
            operands.act_scale = view(b, d.act_scale);
            operands.q = view(b, d.q);
            operands.s = view(b, d.s);
        },
        .route_rows => operands.rows = view(b, d.rows),
    }
    runEpilogue(b, tile, epilogue, storage, tmem_holding, active, d.tiles_m, direct, operands);
    epilogue_warps.yieldThen(.{});
}

fn buildHostLaunch(b: *B, cfg: Config, comptime tile: Tile, comptime epilogue: Epilogue) cute.FinishError!void {
    const L = Layouts(tile);
    const host_arg = HostArgs(epilogue);
    const n = tile.n;
    const a_layout = b.layoutSpec(.{ cfg.m, cfg.k, cfg.experts }, .{ cfg.k, 1, cfg.m * cfg.k });
    const b_layout = b.layoutSpec(.{ cfg.groupRows(), cfg.k, cfg.groups }, .{ cfg.k, 1, cfg.k * cfg.groupRows() });
    // `cute.tile_to_shape` derives that the dynamic strides step whole
    // 512-byte scale atoms, and the result type has to carry it.
    const sf_stride: cute.ConstrainedDynamic = .{ .divisible_by = 512 };
    const dynamic_sf_layout = b.layoutSpec(
        .{ .{ .{ 32, 4 }, cute.AlgebraToken.dynamic }, .{ .{ 16, 4 }, cute.AlgebraToken.dynamic }, .{ 1, cute.AlgebraToken.dynamic } },
        .{ .{ .{ 16, 4 }, sf_stride }, .{ .{ 0, 1 }, 512 }, .{ 0, sf_stride } },
    );

    // XLA owns packed storage as signless bytes; recast to FP4 here.
    const gA = b.makeTensorView(b.recastPointer(b.arg(host_arg.weight), .f4e2m1fn, .gmem, 16), b.staticLayout(a_layout));
    const schedule = b.arg(host_arg.schedule);
    const ids = b.makeTensorView(schedule, b.staticLayout(b.layoutSpec(cfg.groups, 1)));
    const active_ptr = b.addOffsetTyped(
        schedule,
        b.makeIntTuple(.{2 * cfg.groups}),
        b.ptrTy(.i32, .gmem, offsetAlignment(16, 2 * cfg.groups, 4)) catch @panic("bad active count pointer"),
    );
    const active = b.makeTensorView(active_ptr, b.staticLayout(b.layoutSpec(1, 1)));
    const gB = b.makeTensorView(b.recastPointer(b.arg(host_arg.input_quant), .f4e2m1fn, .gmem, 16), b.staticLayout(b_layout));
    const sf_atom_layout = b.staticLayout(b.layoutSpec(
        .{ .{ 32, 4 }, .{ 16, 4 } },
        .{ .{ 16, 4 }, .{ 0, 1 } },
    ));
    // Zero-based form of Python CuTe's `(2, 1, 3)` tile-to-shape order.
    const sf_order = b.makeIntTuple(.{ 1, 0, 2 });
    const sfa_shape = b.makeShapeValues(.{ b.arg(host_arg.problem_m), b.arg(host_arg.problem_k), b.arg(host_arg.expert_count) }, b.parseType("!cute.shape<\"(?,?,?)\">"));
    const sfb_shape = b.makeShapeValues(.{ b.arg(host_arg.problem_n), b.arg(host_arg.problem_k), b.arg(host_arg.problem_l) }, b.parseType("!cute.shape<\"(?,?,?)\">"));
    const dynamic_sfa_layout = b.tileToShapeTyped(sf_atom_layout, sfa_shape, sf_order, b.layoutType(dynamic_sf_layout));
    const dynamic_sfb_layout = b.tileToShapeTyped(sf_atom_layout, sfb_shape, sf_order, b.layoutType(dynamic_sf_layout));
    const gSFA = b.makeTensorView(b.recastPointer(b.arg(host_arg.weight_scale), .f8e4m3fn, .gmem, 16), dynamic_sfa_layout);
    const gSFB = b.makeTensorView(b.recastPointer(b.arg(host_arg.input_scale), .f8e4m3fn, .gmem, 16), dynamic_sfb_layout);

    const zero_tmem = b.intToPtr(0, .f8e4m3fn, .tmem, 1);
    const mma = b.makeBlockScaledMma(.{ .kind = mma_kind, .n = @intCast(n) }, zero_tmem);
    const mma_sfb = b.makeBlockScaledMma(.{ .kind = mma_kind, .n = 128 }, zero_tmem);

    const a_smem = b.makeComposedLayout(L.ab_swizzle, 0, L.aSmem(b, false));
    const b_smem = b.makeComposedLayout(L.ab_swizzle, 0, L.bSmem(b, false));
    const sf_smem = b.staticLayout(L.sfSmem(b, false));
    const a_map = b.staticLayout(b.layoutSpec(.{ .{ 128, 64 }, 1, 4 }, .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0, b.basis(64, 1, .{1}) }));
    const b_map = b.staticLayout(b.layoutSpec(.{ .{ n, 64 }, 1, 4 }, .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0, b.basis(64, 1, .{1}) }));
    const sf_map = b.staticLayout(b.layoutSpec(
        .{ .{ .{ 32, 4 }, .{ 16, 4 } }, 1, 4 },
        .{ .{ .{ b.basis(1, 1, .{ 0, 0, 0 }), b.basis(1, 1, .{ 1, 0, 0 }) }, .{ b.basis(1, 1, .{ 0, 0, 1 }), b.basis(1, 1, .{ 1, 0, 1 }) } }, 0, b.basis(1, 1, .{ 1, 1 }) },
    ));

    const tma_a = b.makeTiledTmaLoadAtomTyped(gA.value(), a_smem.value(), a_map.value(), L.tmaA(b, cfg));
    const tma_b = b.makeTiledTmaLoadAtomTyped(gB.value(), b_smem.value(), b_map.value(), L.tmaB(b, cfg));
    const tma_sfa = b.makeTiledTmaLoadAtomTyped(gSFA.value(), sf_smem.value(), sf_map.value(), L.tmaSf(b));
    const tma_sfb = b.makeTiledTmaLoadAtomTyped(gSFB.value(), sf_smem.value(), sf_map.value(), L.tmaSf(b));

    const one = b.cst(.i32, 1);
    const tiles_m = b.cst(.i32, @divExact(cfg.m, 128));
    const grid_z = b.cst(.i32, @min(@divExact(cfg.m, 128) * cfg.groups, cfg.persistent_ctas));
    const config = b.makeLaunchConfig(.{
        .grid = .{ one, one, grid_z },
        .block = .{ b.cst(.i32, 192), one, one },
        .dynamic_smem = b.kernelSmemSize(deviceName(epilogue)),
        .stream = b.cudaStream(),
        .cluster = .{ one, one, one },
        .use_pdl = true,
    });
    const group_sizes = b.makeTensorView(
        b.addOffsetTyped(schedule, b.makeIntTuple(.{cfg.groups}), b.ptrTy(.i32, .gmem, offsetAlignment(16, cfg.groups, 4)) catch @panic("bad group size pointer")),
        b.staticLayout(b.layoutSpec(cfg.groups, 1)),
    );
    const per_expert = b.staticLayout(b.layoutSpec(cfg.experts, 1));
    const mainloop = .{
        ids,          active,         mma.tiled,    mma_sfb.tiled,  tma_a.atom,                                          tma_a.tensor, tma_b.atom,                                                                                         tma_b.tensor,
        tma_sfa.atom, tma_sfa.tensor, tma_sfb.atom, tma_sfb.tensor, b.makeTensorView(b.arg(host_arg.alpha), per_expert), group_sizes,  b.makeTensorView(b.arg(host_arg.route_inverse), b.staticLayout(b.layoutSpec(cfg.routedRows(), 1))), b.makeTensorView(b.arg(host_arg.routing_weights), b.staticLayout(b.layoutSpec(@max(cfg.routes, 1), 1))),
    };
    const activations = @divExact(cfg.m, 2);
    const launch = switch (epilogue) {
        .swiglu_nvfp4 => b.launchEx(deviceName(epilogue), config, mainloop ++ .{
            b.makeTensorView(b.arg(host_arg.act_scale), per_expert),
            b.makeTensorView(b.recastPointer(b.arg(host_arg.output), .i32, .gmem, 16), b.staticLayout(b.layoutSpec(.{ cfg.routedRows(), @divExact(activations, 8) }, .{ @divExact(activations, 8), 1 }))),
            b.makeTensorView(b.arg(host_arg.scales), b.staticLayout(b.layoutSpec(.{ cfg.groups, @divExact(activations, 16) * 128 }, .{ @divExact(activations, 16) * 128, 1 }))),
            tiles_m,
        }),
        .route_rows => b.launchEx(deviceName(epilogue), config, mainloop ++ .{
            b.makeTensorView(b.arg(host_arg.output), b.staticLayout(b.layoutSpec(.{ @max(cfg.routes, 1), cfg.m }, .{ cfg.m, 1 }))),
            tiles_m,
        }),
    };
    b.returnHostStatus(b.cudaResultStatus(launch));
}

test "NVFP4 GEMM programs emit their host launch for every tile width" {
    inline for (tile_widths) |n| {
        inline for (.{ .{ UpQuantized, Epilogue.swiglu_nvfp4, 4608, 5120 }, .{ DownRows, Epilogue.route_rows, 5120, 2304 } }) |program| {
            const cfg: Config = .{
                .experts = 192,
                .m = program[2],
                .n = n,
                .k = program[3],
                .groups = 227,
                .epilogue = program[1],
                .routes = 576,
                .direct = n == 8,
                .routing_weights = program[1] == .route_rows,
            };
            const ir = try program[0].emit(std.testing.allocator, cfg);
            defer std.testing.allocator.free(ir);
            try std.testing.expect(std.mem.indexOf(u8, ir, deviceName(program[1])) != null);
            try std.testing.expect(std.mem.indexOf(u8, ir, "cuda.launch_ex") != null);
            try std.testing.expect(std.mem.indexOf(u8, ir, "tma_format = U4") != null);
            try std.testing.expect(std.mem.indexOf(u8, ir, std.fmt.comptimePrint("sm100.mma_bs<128x{d}x64", .{n})) != null);
        }
    }
}
