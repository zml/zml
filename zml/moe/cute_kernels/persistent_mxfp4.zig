//! Source-level SM100 MXFP4 grouped GEMM.
//!
//! This file is the Zig CuTe-DSL counterpart of the CuTe DSL
//! `Sm100BlockScaledPersistentDenseGemmKernel` with its fused MoE epilogues.
//! It emits a complete CuTe program: the nested `cuda.kernel` consumes the
//! objects constructed by the public host `func.func`, and the host function
//! creates the block-scaled MMA objects, four TMA tensor maps, persistent
//! scheduler parameters, and the PDL-enabled launch.
//!
//! One CTA computes a 128xN output tile of one expert group: warp 5 drives
//! the TMA ring, warp 4 issues the S2T scale copies and UMMA, and warps 0-3
//! drain the accumulator from tensor memory into one of the two epilogues.
const std = @import("std");
const mlir = @import("mlir");

const zml = @import("../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;

const accumulator_stages = 2;

/// Shared-memory plan of the Python kernel for one tile width, copied from
/// its generated `SharedStorage` (`smem.struct_fields`).  The A/B ring depth
/// is whatever fits next to the two C epilogue stages; the fused epilogues
/// only use the start of `sC` as scratch.
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
    /// Bytes reserved at `c`. Only the epilogue scratch is used; the wide
    /// prefill tiles keep just that instead of two full C stages.
    c_bytes: ?i64 = null,

    pub fn of(comptime n: i64) Tile {
        return switch (n) {
            8 => .{
                .n = 8,
                .stages = 12,
                .smem_bytes = 230400,
                .tmem_columns = 64,
                .empty_barriers = 96,
                .accumulator_full = 192,
                .accumulator_empty = 208,
                .tmem_dealloc = 224,
                .tmem_holding = 232,
                .c = 1024,
                .a = 9216,
                .b = 205824,
                .sfa = 218112,
                .sfb = 224256,
            },
            16 => .{
                .n = 16,
                .stages = 11,
                .smem_bytes = 232448,
                .tmem_columns = 64,
                .empty_barriers = 88,
                .accumulator_full = 176,
                .accumulator_empty = 192,
                .tmem_dealloc = 208,
                .tmem_holding = 216,
                .c = 1024,
                .a = 17408,
                .b = 197632,
                .sfa = 220160,
                .sfb = 226304,
            },
            32 => .{
                .n = 32,
                .stages = 9,
                .smem_bytes = 228352,
                .tmem_columns = 128,
                .empty_barriers = 72,
                .accumulator_full = 144,
                .accumulator_empty = 160,
                .tmem_dealloc = 176,
                .tmem_holding = 184,
                .c = 1024,
                .a = 33792,
                .b = 181248,
                .sfa = 218112,
                .sfb = 223232,
            },
            // Prefill tiles. One expert weight tile feeds 64 or 128 routed
            // rows instead of 32, which is what keeps large batches from
            // re-reading the expert weights once per group. They hold only
            // the epilogue scratch at `c`, so the A/B ring stays deep.
            64 => .{
                .n = 64,
                .stages = 9,
                .smem_bytes = 232448,
                .tmem_columns = 256,
                .empty_barriers = 72,
                .accumulator_full = 144,
                .accumulator_empty = 160,
                .tmem_dealloc = 176,
                .tmem_holding = 184,
                .c = 1024,
                .c_bytes = 512,
                .a = 2048,
                .b = 149504,
                .sfa = 223232,
                .sfb = 227840,
            },
            128 => .{
                .n = 128,
                .stages = 6,
                .smem_bytes = 204800,
                .tmem_columns = 512,
                .empty_barriers = 48,
                .accumulator_full = 96,
                .accumulator_empty = 112,
                .tmem_dealloc = 128,
                .tmem_holding = 136,
                .c = 1024,
                .c_bytes = 512,
                .a = 2048,
                .b = 100352,
                .sfa = 198656,
                .sfb = 201728,
            },
            else => @compileError("unsupported MXFP4 tile N"),
        };
    }

    /// Routed rows an epilogue pass handles: one per warp lane.
    pub fn chunkRows(self: Tile) i64 {
        return @min(self.n, 32);
    }

    /// Epilogue passes over the tile width.
    pub fn chunks(self: Tile) usize {
        return @intCast(@divExact(self.n, self.chunkRows()));
    }

    const a_stage_bytes = 128 * 128;
    const sf_stage_bytes = 512;

    fn bStageBytes(self: Tile) i64 {
        return self.n * 128;
    }

    fn cStageElements(self: Tile) i64 {
        return 128 * self.n;
    }

    /// TMA bytes landing in one A/B stage: unpacked FP4 A is counted as
    /// 128x128 nibbles, then the FP8 B tile and both 512-byte scale atoms.
    fn transactionBytes(self: Tile) u32 {
        return @intCast(8192 + self.bStageBytes() + 2 * sf_stage_bytes);
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
            .{ "sC", self.c_bytes orelse accumulator_stages * self.cStageElements() * 4, self.c },
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
    /// Up projection: clamped SwiGLU of the interleaved gate/up rows, BF16
    /// rounding and the routing weight as in the reference model, then MXFP8
    /// quantization straight into the down GEMM's input and 128x4 scale
    /// layouts. Python `fused=1`.
    swiglu_mxfp8,
    /// Down projection: BF16 rows stored in route order `[routes, m]`,
    /// skipping padding rows. Python `fused=2`. BF16 halves the traffic of
    /// this store and of the top-k reduction that reads it back; the routed
    /// partial sums are rounded exactly like the reference finalize.
    route_rows,
};

pub const Config = struct {
    experts: i64,
    /// GEMM output rows (4608 for gate/up, 5120 for down).
    m: i64,
    /// Tile N: rows in one routed expert group.
    n: i64 = 16,
    /// Reduction dimension (5120 for gate/up, 2304 for down).
    k: i64,
    groups: i64,
    /// Resident CTAs of the static persistent tile scheduler: one per SM.
    persistent_ctas: i64 = 148,
    epilogue: Epilogue,
    /// Token/top-k routes; sizes the route-indexed epilogue operands.
    routes: i64,
    /// Ungrouped routing: group `g` is route `g` and holds one live row.
    /// `schedule` is then just the `[routes]` expert ids; its size and
    /// active-count sections and `route_inverse` are not read.
    direct: bool = false,

    /// Rows stored per group in the routed operands. Direct routing stores
    /// one; the B tensor map zero-fills the rest of each N tile, like
    /// Python's ungrouped `(1, k, routes)` activations.
    pub fn groupRows(self: Config) i64 {
        return if (self.direct) 1 else self.n;
    }

    pub fn routedRows(self: Config) i64 {
        return self.groups * self.groupRows();
    }
};

/// `schedule` is the routing schedule `[group experts | group sizes | active
/// group count]`; the kernel reads the first and last sections in place.
const inputs = [_][:0]const u8{
    "weight",
    "schedule",
    "input_quant",
    "weight_scale",
    "input_scale",
};

/// `route_inverse` maps each physical routed row to its token/top-k route;
/// entries of padding rows are never read.
const UpQuantized = cute.Program(Config, .{
    .name = "zml_mxfp4_up_swiglu_zig",
    .inputs = &(inputs ++ [_][:0]const u8{ "route_inverse", "routing_weights" }),
    .outputs = &.{ "q", "s" },
    .run = buildProgram,
});

const DownRows = cute.Program(Config, .{
    .name = "zml_mxfp4_down_rows_zig",
    .inputs = &(inputs ++ [_][:0]const u8{"route_inverse"}),
    .outputs = &.{"output"},
    .run = buildProgram,
});

pub const Inputs = struct {
    weight: zml.Tensor,
    schedule: zml.Tensor,
    input_quant: zml.Tensor,
    weight_scale: zml.Tensor,
    input_scale: zml.Tensor,
};

/// Up projection with the SwiGLU + MXFP8 epilogue: the routed FP8 rows
/// `[groups * n, m / 2]` and their 128x4 scales `[groups, m / 64 * 128]`,
/// ready for the down GEMM. `routing_weights` is `[routes]` FP32.
pub fn upQuantized(cfg: Config, a: Inputs, route_inverse: zml.Tensor, routing_weights: zml.Tensor) struct { q: zml.Tensor, s: zml.Tensor } {
    std.debug.assert(cfg.epilogue == .swiglu_mxfp8);
    const activations = @divExact(cfg.m, 2);
    const result = UpQuantized.call(.{
        .weight = a.weight,
        .schedule = a.schedule,
        .input_quant = a.input_quant,
        .weight_scale = a.weight_scale,
        .input_scale = a.input_scale,
        .route_inverse = route_inverse,
        .routing_weights = routing_weights,
    }, .{
        .q = .init(.{ cfg.routedRows(), activations }, .f8e4m3fn),
        .s = .init(.{ cfg.groups, @divExact(activations, 32) * 128 }, .u8),
    }, .{ .cfg = cfg, .scalars = &.{ cfg.m, cfg.n, cfg.k, cfg.groups, cfg.experts } });
    return .{ .q = result.q, .s = result.s };
}

/// Down projection storing FP32 `[routes, m]` in token/top-k order.
pub fn downRows(cfg: Config, a: Inputs, route_inverse: zml.Tensor) zml.Tensor {
    std.debug.assert(cfg.epilogue == .route_rows);
    return DownRows.call(.{
        .weight = a.weight,
        .schedule = a.schedule,
        .input_quant = a.input_quant,
        .weight_scale = a.weight_scale,
        .input_scale = a.input_scale,
        .route_inverse = route_inverse,
    }, .{ .output = .init(.{ cfg.routes, cfg.m }, .bf16) }, .{ .cfg = cfg, .scalars = &.{ cfg.m, cfg.n, cfg.k, cfg.groups, cfg.experts } }).output;
}

fn typed(type_: *const mlir.Type) struct { mlir_type: cute.ArgSpec.MlirTypeSpec } {
    return .{ .mlir_type = .{ .type_ = type_ } };
}

fn gridTyped(type_: *const mlir.Type) struct { mlir_type: cute.ArgSpec.MlirTypeSpec } {
    return .{ .mlir_type = .{ .type_ = type_, .grid_constant = true } };
}

const device_name = "mxfp4_sm100_persistent_grouped_gemm";

fn buildProgram(b: *B, cfg: Config) cute.FinishError!void {
    return switch (cfg.n) {
        inline 8, 16, 32, 64, 128 => |n| switch (cfg.epilogue) {
            inline else => |epilogue| buildTiledProgram(b, cfg, comptime Tile.of(n), epilogue),
        },
        else => std.debug.panic("unsupported MXFP4 tile N {d}", .{cfg.n}),
    };
}

/// Types and layouts shared by the device ABI and the host launch.
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
        fn sfCoord(b: *B) cute.LayoutSpec {
            return b.layoutSpec(
                .{ .{ .{ 32, 4 }, cute.AlgebraToken.dynamic }, .{ .{ 32, 4 }, cute.AlgebraToken.dynamic }, .{ 1, cute.AlgebraToken.dynamic } },
                .{
                    .{ .{ b.basis(8, 1, .{0}), b.basis(2, 1, .{0}) }, b.basis(1, 1, .{1}) },
                    .{ .{ 0, b.basis(1, 2, .{0}) }, b.basis(1, 1, .{2}) },
                    .{ 0, b.basis(1, 1, .{3}) },
                },
            );
        }
        fn aBasis(b: *B) cute.LayoutSpec {
            return b.layoutSpec(.{ 128, 128, 1 }, .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), b.basis(1, 1, .{2}) });
        }
        fn bBasis(b: *B) cute.LayoutSpec {
            return b.layoutSpec(.{ 128, n, 1 }, .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), b.basis(1, 1, .{2}) });
        }
        fn sfBasis(b: *B) cute.LayoutSpec {
            return b.layoutSpec(
                .{ .{ 2, 4, 32 }, 1, 1, 1 },
                .{ .{ b.basis(1, 1, .{ 1, 0, 1 }), b.basis(1, 1, .{ 1, 0, 0 }), b.basis(1, 1, .{ 0, 0, 0 }) }, b.basis(1, 1, .{ 1, 0 }), b.basis(1, 1, .{ 1, 1 }), b.basis(1, 1, .{ 1, 2 }) },
            );
        }
        fn tmaA(b: *B, cfg: Config) cute.TmaConfig {
            return .{ .dtype = .f4e2m1fn, .copy_bits = 65536, .global_basis = aBasis(b), .coordinate_layout = aCoord(b, cfg), .coordinate_rank = 3, .format = .u4_unpack_u8 };
        }
        fn tmaB(b: *B, cfg: Config) cute.TmaConfig {
            return .{ .dtype = .f8e4m3fn, .copy_bits = @intCast(tile.bStageBytes() * 8), .global_basis = bBasis(b), .coordinate_layout = bCoord(b, cfg), .coordinate_rank = 3 };
        }
        fn tmaSf(b: *B) cute.TmaConfig {
            return .{ .dtype = .f8e8m0fnu, .copy_bits = 4096, .global_basis = sfBasis(b), .coordinate_layout = sfCoord(b), .coordinate_rank = 4, .format = .u16 };
        }
        // Shared-memory layouts, in the same element units as Python.
        fn aSmem(b: *B, comptime staged: bool) cute.LayoutSpec {
            return if (staged)
                b.layoutSpec(.{ .{ 128, 32 }, 1, 4, stages }, .{ .{ 128, 1 }, 0, 32, Tile.a_stage_bytes })
            else
                b.layoutSpec(.{ .{ 128, 32 }, 1, 4 }, .{ .{ 128, 1 }, 0, 32 });
        }
        fn bSmem(b: *B, comptime staged: bool) cute.LayoutSpec {
            return if (staged)
                b.layoutSpec(.{ .{ n, 32 }, 1, 4, stages }, .{ .{ 128, 1 }, 0, 32, tile.bStageBytes() })
            else
                b.layoutSpec(.{ .{ n, 32 }, 1, 4 }, .{ .{ 128, 1 }, 0, 32 });
        }
        fn sfSmem(b: *B, comptime staged: bool) cute.LayoutSpec {
            return if (staged)
                b.layoutSpec(.{ .{ .{ .{ 32, 4 }, 1 }, .{ 32, 1 } }, 1, 4, stages }, .{ .{ .{ .{ 16, 4 }, 0 }, .{ 0, 0 } }, 0, 1, Tile.sf_stage_bytes })
            else
                b.layoutSpec(.{ .{ .{ .{ 32, 4 }, 1 }, .{ 32, 1 } }, 1, 4 }, .{ .{ .{ .{ 16, 4 }, 0 }, .{ 0, 0 } }, 0, 1 });
        }
    };
}

/// Device memrefs of the route-indexed epilogue operands.
const EpilogueTypes = struct {
    sizes: *const mlir.Type,
    inverse: *const mlir.Type,
    weights: *const mlir.Type,
    q: *const mlir.Type,
    s: *const mlir.Type,
    rows: *const mlir.Type,

    fn of(b: *B, cfg: Config) EpilogueTypes {
        const activations = @divExact(cfg.m, 2);
        return .{
            // Group sizes follow the group experts inside `schedule`.
            .sizes = b.memrefType(.i32, .gmem, 4, b.layoutType(b.layoutSpec(cfg.groups, 1))),
            .inverse = b.memrefType(.i32, .gmem, 16, b.layoutType(b.layoutSpec(cfg.routedRows(), 1))),
            .weights = b.memrefType(.f32, .gmem, 16, b.layoutType(b.layoutSpec(@max(cfg.routes, 1), 1))),
            .q = b.memrefType(.f8e4m3fn, .gmem, 16, b.layoutType(b.layoutSpec(.{ cfg.routedRows(), activations }, .{ activations, 1 }))),
            .s = b.memrefType(.i8, .gmem, 16, b.layoutType(b.layoutSpec(.{ cfg.groups, @divExact(activations, 32) * 128 }, .{ @divExact(activations, 32) * 128, 1 }))),
            .rows = b.memrefType(.bf16, .gmem, 16, b.layoutType(b.layoutSpec(.{ @max(cfg.routes, 1), cfg.m }, .{ cfg.m, 1 }))),
        };
    }
};

fn buildTiledProgram(b: *B, cfg: Config, comptime tile: Tile, comptime epilogue: Epilogue) cute.FinishError!void {
    @setEvalBranchQuota(20_000);
    std.debug.assert(@rem(cfg.m, 128) == 0);
    std.debug.assert(@rem(cfg.k, 128) == 0);
    const L = Layouts(tile);
    const public_name = b.name;

    const ids_memref = b.memrefType(.i32, .gmem, 16, b.layoutType(b.layoutSpec(cfg.groups, 1)));
    // The active group count is the last schedule entry, 4-byte aligned.
    const active_memref = b.memrefType(.i32, .gmem, 4, b.layoutType(b.layoutSpec(1, 1)));
    const et = EpilogueTypes.of(b, cfg);
    const i32_type = cute.DType.i32.toMlir(b.ctx);
    const mma_type = b.blockScaledMmaType(.{ .n = @intCast(tile.n) });
    const mma_sfb_type = b.blockScaledMmaType(.{ .n = 128 });

    // Device ABI. The host function below constructs every non-buffer value.
    b.beginFunction(device_name, .cuda_kernel);
    const d = try b.declareArgs(switch (epilogue) {
        .swiglu_mxfp8 => .{
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
            .group_sizes = typed(et.sizes),
            .route_inverse = typed(et.inverse),
            .routing_weights = typed(et.weights),
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
            .group_sizes = typed(et.sizes),
            .route_inverse = typed(et.inverse),
            .rows = typed(et.rows),
            .tiles_m = typed(i32_type),
        },
    });
    b.setFunctionAttribute("nvvm.minctasm", .int(b.ctx, .i32, 1));
    b.setFunctionAttribute("smem.partition_num", .int(b.ctx, .i32, 2));
    b.setFunctionAttribute("smem.hint_smem_base_uniform", .unit(b.ctx));
    b.setFunctionAttribute("cu_attrs", b.parseAttribute("{max_dynamic_shared_size_bytes = #cuda.dev_max_shared_memory_optin, non_portable_cluster_size_allowed = 1 : i32}"));

    // Prefetch every tensor-map descriptor before the specialized warps begin
    // the persistent pipeline.
    b.launchDependents();
    const warp = b.makeWarpUniform(b.makeWarpUniform(b.threadIdx().x.div(32)));
    var tma_warp = b.openIf(warp.eq(5));
    inline for (.{ d.tma_a, d.tma_b, d.tma_sfa, d.tma_sfb }) |value| {
        const descriptor: cute.Atom = .{ .inner = value.inner, .kernel = b };
        b.prefetchTmaDesc(descriptor);
    }
    tma_warp.yieldThen(.{});
    buildUmmaDevice(b, cfg, tile, epilogue, d);
    b.endFunction(.{ 192, 1, 1 });

    // Public host ABI: the XLA operands, the outputs, then the problem shape.
    // XLA owns packed storage as signless bytes. Recast only inside the host
    // program when constructing typed tensor maps.
    b.beginFunction(public_name, .host);
    const operands = .{
        .weight = .{ .ptr = cute.DType.i8 },
        .schedule = .{ .ptr = cute.DType.i32 },
        .input_quant = .{ .ptr = cute.DType.f8e4m3fn },
        .weight_scale = .{ .ptr = cute.DType.i8 },
        .input_scale = .{ .ptr = cute.DType.i8 },
    };
    const shape = .{
        .problem_m = typed(i32_type),
        .problem_n = typed(i32_type),
        .problem_k = typed(i32_type),
        .problem_l = typed(i32_type),
        .expert_count = typed(i32_type),
    };
    _ = try b.declareArgs(switch (epilogue) {
        .swiglu_mxfp8 => .{
            .weight = operands.weight,
            .schedule = operands.schedule,
            .input_quant = operands.input_quant,
            .weight_scale = operands.weight_scale,
            .input_scale = operands.input_scale,
            .route_inverse = .{ .ptr = cute.DType.i32 },
            .routing_weights = .{ .ptr = cute.DType.f32 },
            .q = .{ .ptr = cute.DType.f8e4m3fn },
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
            .route_inverse = .{ .ptr = cute.DType.i32 },
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

/// Host argument positions: the five mainloop operands, the epilogue's
/// extra inputs and outputs, then the five problem-shape scalars.
fn HostArgs(comptime epilogue: Epilogue) type {
    return struct {
        const weight = 0;
        const schedule = 1;
        const input_quant = 2;
        const weight_scale = 3;
        const input_scale = 4;
        const route_inverse = 5;
        const routing_weights = 6;
        const output = switch (epilogue) {
            .swiglu_mxfp8 => 7,
            .route_rows => 6,
        };
        const scales = 8;
        const problem_m = switch (epilogue) {
            .swiglu_mxfp8 => 9,
            .route_rows => 7,
        };
        const problem_n = problem_m + 1;
        const problem_k = problem_m + 2;
        const problem_l = problem_m + 3;
        const expert_count = problem_m + 4;
    };
}

/// Complete staged shared-memory tensors, as Python's `PipelineTmaUmma`
/// consumers see them.  Per-stage views are slices of these.
fn stagedOperands(b: *B, comptime tile: Tile, storage: StageStorage) StagedOperands {
    const L = Layouts(tile);
    const a_full = b.makeViewTyped(
        b.recastSwizzledPointer(storage.a, .i8, .smem, 1024, L.ab_swizzle),
        b.staticLayout(L.aSmem(b, true)),
        b.memrefTypeFromPointer(b.swizzledPtrTy(.i8, .smem, 1024, L.ab_swizzle) catch @panic("bad A shared pointer"), b.layoutType(L.aSmem(b, true))),
    );
    const b_full = b.makeViewTyped(
        b.recastSwizzledPointer(storage.b, .f8e4m3fn, .smem, 1024, L.ab_swizzle),
        b.staticLayout(L.bSmem(b, true)),
        b.memrefTypeFromPointer(b.swizzledPtrTy(.f8e4m3fn, .smem, 1024, L.ab_swizzle) catch @panic("bad B shared pointer"), b.layoutType(L.bSmem(b, true))),
    );
    const sfa_full = b.makeViewTyped(storage.sfa, b.staticLayout(L.sfSmem(b, true)), b.memrefType(.f8e8m0fnu, .smem, 1024, b.layoutType(L.sfSmem(b, true))));
    const sfb_full = b.makeViewTyped(storage.sfb, b.staticLayout(L.sfSmem(b, true)), b.memrefType(.f8e8m0fnu, .smem, 1024, b.layoutType(L.sfSmem(b, true))));
    return .{ .a = a_full, .b = b_full, .sfa = sfa_full, .sfb = sfb_full };
}

fn stageBuffersAt(b: *B, comptime tile: Tile, storage: StageStorage, stage_index: V) StageBuffers {
    const L = Layouts(tile);
    // Keep the shared operands as staged tensors and select the current stage
    // with CuTe slicing. This is the representation used by the Python
    // PipelineTmaUmma implementation; rebuilding a memref from a runtime
    // pointer loses the staged descriptor semantics at ring wrap.
    const full = stagedOperands(b, tile, storage);
    const all = cute.AlgebraToken.all;
    const a = b.sliceTyped(full.a, .{ all, all, all, stage_index }, b.memrefTypeFromPointer(b.swizzledPtrTy(.i8, .smem, 1024, L.ab_swizzle) catch @panic("bad A stage pointer"), b.layoutType(L.aSmem(b, false))));
    const rhs = b.sliceTyped(full.b, .{ all, all, all, stage_index }, b.memrefTypeFromPointer(b.swizzledPtrTy(.f8e4m3fn, .smem, 1024, L.ab_swizzle) catch @panic("bad B stage pointer"), b.layoutType(L.bSmem(b, false))));
    const sfa = b.sliceTyped(full.sfa, .{ all, all, all, stage_index }, b.memrefType(.f8e8m0fnu, .smem, 1024, b.layoutType(L.sfSmem(b, false))));
    const sfb = b.sliceTyped(full.sfb, .{ all, all, all, stage_index }, b.memrefType(.f8e8m0fnu, .smem, 1024, b.layoutType(L.sfSmem(b, false))));

    // `tma_partition` preserves a singleton CTA mode. Keep it in the copy
    // views: it is part of the executable TMA atom's expected algebra even
    // for a one-CTA cluster.
    const a_partition_layout = b.staticLayout(b.layoutSpec(.{ .{ Tile.a_stage_bytes, 1 }, .{1} }, .{ .{ 1, 0 }, .{0} }));
    const b_partition_layout = b.staticLayout(b.layoutSpec(.{ .{ tile.bStageBytes(), 1 }, .{1} }, .{ .{ 1, 0 }, .{0} }));
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

/// Issue the four transactions that fill one mainloop stage. The caller is
/// already restricted to the TMA warp; one elected lane owns the mbarrier
/// arrival.
fn issueTmaStage(
    b: *B,
    comptime tile: Tile,
    comptime reduction_tiles: usize,
    partitioned_a_target: cute.View,
    partitioned_b_target: cute.View,
    stage: StageBuffers,
    k_tile: V,
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
        .{ .{ .{ 128, 128 }, 1 }, .{1} },
        .{ .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }, .{0} },
    ));
    const b_copy_layout = b.staticLayout(b.layoutSpec(
        .{ .{ .{ 128, n }, 1 }, .{1} },
        .{ .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }, .{0} },
    ));
    const sf_copy_spec = b.layoutSpec(
        .{ .{ 512, 1 }, .{1} },
        .{ .{ b.basis(1, 2, .{0}), 0 }, .{0} },
    );
    // Follow the same coordinate algebra as the Python kernel. TMA
    // partitioning produces a tensor over (tile, M, K, expert/group); first
    // select the output tile and expert/group, then select the K stage. The
    // remaining iterator carries the tensor-map coordinates needed by copy.
    const a_tile_k = b.sliceTyped(
        partitioned_a_target,
        .{ all, tile_m, all, expert },
        b.coordTensorTypePayload("(0,?{div=128},?)", b.layoutSpec(
            .{ .{ .{ 128, 128 }, 1 }, reduction_tiles },
            .{ .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }, b.basis(128, 1, .{0}) },
        )),
    );
    const a_tile = b.sliceTyped(
        a_tile_k,
        .{ all, k_tile },
        b.coordTensorTypePayload("(?{div=128},?{div=128},?)", b.layoutSpec(
            .{.{ .{ 128, 128 }, 1 }},
            .{.{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }},
        )),
    );
    const a_source = b.rebuildCoordTensorIterator(3, a_tile, "(?{div=128},?{div=128},?)", .{ 128, 128, 1 }, a_copy_layout);
    const b_tile_k = b.sliceTyped(
        partitioned_b_target,
        .{ all, 0, all, group },
        b.coordTensorTypePayload(fmt(b, "(0,?{{div={d}}},?)", .{n}), b.layoutSpec(
            .{ .{ .{ 128, n }, 1 }, reduction_tiles },
            .{ .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }, b.basis(128, 1, .{0}) },
        )),
    );
    const b_tile = b.sliceTyped(
        b_tile_k,
        .{ all, k_tile },
        b.coordTensorTypePayload(fmt(b, "(?{{div=128}},?{{div={d}}},?)", .{n}), b.layoutSpec(
            .{.{ .{ 128, n }, 1 }},
            .{.{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }},
        )),
    );
    const b_source = b.rebuildCoordTensorIterator(3, b_tile, fmt(b, "(?{{div=128}},?{{div={d}}},?)", .{n}), .{ 128, n, 1 }, b_copy_layout);
    const sfa_source = b.makeCoordTensor(.{ 0, tile_m, k_tile, expert }, b.staticCoordinateLayout(4, sf_copy_spec));
    const sfb_source = b.makeCoordTensor(.{ 0, b.cst(.i32, 0), k_tile, group }, b.staticCoordinateLayout(4, sf_copy_spec));

    var elected = b.openIf(b.electSync());
    b.mbarrierArriveExpectTx(stage.load_barrier, tile.transactionBytes());
    elected.yieldThen(.{});
    b.copy(b.setTmaBarrier(exec_a, stage.load_barrier), .{a_source}, .{stage.a_destination}, null);
    b.copy(b.setTmaBarrier(exec_b, stage.load_barrier), .{b_source}, .{stage.b_destination}, null);
    b.copy(b.setTmaBarrier(exec_sfa, stage.load_barrier), .{sfa_source}, .{stage.sfa_destination}, null);
    b.copy(b.setTmaBarrier(exec_sfb, stage.load_barrier), .{sfb_source}, .{stage.sfb_destination}, null);
}

fn barrierAt(b: *B, base: V, stage: V) V {
    return b.addOffsetTyped(
        base,
        b.makeIntTuple(.{stage}),
        b.ptrTy(.i64, .smem, 8) catch @panic("bad pipeline barrier pointer"),
    ).value();
}

const PipelinePosition = struct {
    index: V,
    phase: V,
};

/// Scalar form of CUTLASS's StaticPersistentTileScheduler state for the
/// one-CTA DeepSeek specialization.  Python carries the decoded M/N/L
/// coordinate, validity predicate, linear work index, and executed-tile
/// count through every role's `scf.while`.  Keeping the same state shape is
/// important to the CuTe software-pipeline passes, even though N is always
/// zero for one tile per group.
const PersistentWork = struct {
    m: V,
    n: V,
    group: V,
    valid: V,
    linear: V,
    executed: V,
};

/// Direct routing (one token) runs one single-row group per route. Routes
/// owned by other expert-parallel ranks carry expert -1: the persistent tiles
/// cover the valid routes only, in route order, so they do not occupy a wave.
const DirectRoutes = struct {
    expert_ids: cute.View,
    routes: usize,

    fn count(self: DirectRoutes, b: *B) V {
        var valid = b.cst(.i32, 0);
        for (0..self.routes) |r| valid = valid.add(self.expertAt(b, r).ge(0).to(.i32));
        return valid;
    }

    /// Route of the `k`-th valid route.
    fn nth(self: DirectRoutes, b: *B, k: V) V {
        var seen = b.cst(.i32, 0);
        var route = b.cst(.i32, 0);
        for (0..self.routes) |r| {
            const valid = self.expertAt(b, r).ge(0);
            route = b.select(valid.bitAnd(seen.eq(k)), b.cst(.i32, @as(i32, @intCast(r))), route);
            seen = seen.add(valid.to(.i32));
        }
        return route;
    }

    fn expertAt(self: DirectRoutes, b: *B, r: usize) V {
        return self.expert_ids.get(.{b.cst(.i32, @as(i32, @intCast(r)))}, .i32);
    }
};

fn persistentWorkAt(
    b: *B,
    linear: V,
    executed: V,
    active: V,
    tiles_m: V,
    div_m: V,
    div_n: V,
    direct: ?DirectRoutes,
) PersistentWork {
    // StaticPersistentTileScheduler uses two first-class FastDivmod objects
    // to decode the column-major (M, N=1, L) problem layout.  Preserve those
    // operations even though ordinary rem/div is mathematically equivalent;
    // CuTe's scheduler reconstruction and loop canonicalization key off these
    // values and carry them through every persistent loop.
    const group_and_m = b.fastDivmod(linear, div_m);
    const group_and_n = b.fastDivmod(group_and_m[0], div_n);
    const group = if (direct) |routes| routes.nth(b, group_and_n[0]) else group_and_n[0];
    return .{
        .m = b.makeWarpUniform(group_and_m[1]),
        .n = b.makeWarpUniform(group_and_n[1]),
        .group = b.makeWarpUniform(group),
        .valid = b.makeWarpUniform(linear.lt(active.mul(tiles_m))),
        .linear = linear,
        .executed = executed,
    };
}

fn advancePersistentWork(
    b: *B,
    linear: V,
    executed: V,
    persistent_clusters: V,
    active: V,
    tiles_m: V,
    div_m: V,
    div_n: V,
    direct: ?DirectRoutes,
) PersistentWork {
    return persistentWorkAt(b, linear.add(persistent_clusters), executed.add(1), active, tiles_m, div_m, div_n, direct);
}

/// Equivalent to CUTLASS PipelineState.advance().  Keep the index and phase
/// as explicit loop-carried SSA values instead of recovering them with
/// division/remainder; the generated CuTe program then has the same circular
/// state transition as the Python DSL.
fn advancePipeline(b: *B, index: V, phase: V, comptime stages: usize) PipelinePosition {
    const advanced = index.add(1);
    const wraps = advanced.eq(stages);
    const next_index = b.select(wraps, 0, advanced);
    var next_phase = b.openIfElse(wraps, .{phase.type_()});
    next_phase.yieldThen(.{phase.bitXor(1)});
    next_phase.yieldElse(.{phase});
    return .{ .index = next_index, .phase = next_phase.results[0] };
}

fn peekBarrierIfRemaining(b: *B, base: V, count: V, limit: usize, state: PipelinePosition) V {
    var peek = b.openIfElse(count.lt(limit), .{cute.DType.i1.toMlir(b.ctx)});
    peek.yieldThen(.{b.mbarrierWaitParity(barrierAt(b, base, state.index), state.phase)});
    peek.yieldElse(.{b.cst(.i1, true)});
    return peek.results[0];
}

fn accumulatorTmemAt(b: *B, comptime tile: Tile, tmem: V, stage: V) V {
    return b.addOffsetTyped(
        tmem,
        b.makeIntTuple(.{stage.mul(tile.n)}),
        b.ptrTy(.f32, .tmem, 16) catch @panic("bad accumulator TMEM pointer"),
    ).value();
}

fn sharedPointerAt(
    b: *B,
    base: V,
    comptime byte_offset: i64,
    comptime dtype: cute.DType,
    comptime alignment: u64,
) V {
    const byte_ptr = b.addOffsetTyped(
        base,
        b.makeIntTuple(.{byte_offset}),
        b.ptrTy(.i8, .smem, alignment) catch @panic("bad shared byte pointer"),
    );
    return b.recastPointer(byte_ptr, dtype, .smem, alignment);
}

/// Warp 5 runs independently over every persistent output tile.  Keep the
/// producer index and phase as loop-carried values across tiles, exactly like
/// CUTLASS PipelineState.  Only the per-tile count is reset to zero.
fn runTmaProducer(
    b: *B,
    comptime tile: Tile,
    comptime k_tiles: usize,
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
        k_tiles,
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

    // PipelineTmaUmma.producer_tail: drain every empty barrier before the TMA
    // warp exits so no outstanding tcgen05 arrival outlives the CTA.
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

fn Tuple(comptime T: type, comptime len: usize) type {
    return std.meta.Tuple(&([_]type{T} ** len));
}

fn arity(comptime T: type) usize {
    return @typeInfo(T).@"struct".fields.len;
}

/// The MLIR types of a tuple of values, as `openWhile` expects them.
fn typesOf(values: anytype) Tuple(*const mlir.Type, arity(@TypeOf(values))) {
    const len = comptime arity(@TypeOf(values));
    var out: Tuple(*const mlir.Type, len) = undefined;
    inline for (0..len) |i| out[i] = values[i].type_();
    return out;
}

/// Forward a whole loop-carried array, as `yieldBefore` expects a tuple.
fn tupleOf(array: anytype) Tuple(@TypeOf(array[0]), array.len) {
    var out: Tuple(@TypeOf(array[0]), array.len) = undefined;
    inline for (0..array.len) |i| out[i] = array[i];
    return out;
}

/// SM100 MXFP4 mainloop objects of one pipeline stage, produced by
/// `StagedMxfp4Mainloop.atStage`. The dependent CuTe result types are
/// centralized here; callers only provide tensors and the allocated
/// tensor-memory pointer.
const Mxfp4Mainloop = struct {
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

    pub fn copyScales(self: Mxfp4Mainloop, builder: *B) void {
        builder.copy(self.sfa_copy, .{self.sfa_source}, .{self.sfa_target}, null);
        builder.copy(self.sfb_copy, .{self.sfb_source}, .{self.sfb_target}, null);
    }
};

/// Pipeline-wide SM100 MXFP4 mainloop objects.  CUTLASS derives the UMMA and
/// S2T descriptors from the complete staged tensors once, then slices those
/// descriptors at the current PipelineState index.  Keeping the stage mode in
/// the descriptor is required when the circular buffer wraps back to stage 0.
const StagedMxfp4Mainloop = struct {
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
    /// MMA tile N: accumulator columns per stage.
    n: i64,

    pub fn atStage(self: StagedMxfp4Mainloop, b: *B, index: V, accumulator_tmem: V) Mxfp4Mainloop {
        const all = cute.AlgebraToken.all;
        const a = b.sliceTyped(
            self.a,
            .{ all, all, all, index },
            b.parseType("!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(1,1,4):(0,0,2)\">"),
        );
        const rhs = b.sliceTyped(
            self.b,
            .{ all, all, all, index },
            b.parseType("!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(1,1,4):(0,0,2)\">"),
        );
        const sfa_stage = b.sliceTyped(
            self.sfa_source,
            .{ all, all, all, all, index },
            b.parseType("!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(((32,1,1,4),1),1,1,1):(((1,1,1,0),0),0,0,0)\">"),
        );
        const sfb_stage = b.sliceTyped(
            self.sfb_source,
            .{ all, all, all, all, index },
            b.parseType("!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(((32,1,1,4),1),1,1,1):(((1,1,1,0),0),0,0,0)\">"),
        );
        const source_grouped_layout = b.staticValue("!cute.layout<\"(((32,1,1,4),1),(1,1,1)):(((1,1,1,0),0),(0,0,0))\">");
        const source_grouped_ty = "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(((32,1,1,4),1),(1,1,1)):(((1,1,1,0),0),(0,0,0))\">";
        const sfa_source = b.makeView(
            b.getIter(sfa_stage, "!cute_nvgpu.smem_desc"),
            source_grouped_layout,
            source_grouped_ty,
        );
        const sfb_source = b.makeView(
            b.getIter(sfb_stage, "!cute_nvgpu.smem_desc"),
            source_grouped_layout,
            source_grouped_ty,
        );
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

/// Build the same pipeline-wide descriptors as Python's
/// `Sm100BlockScaledPersistentDenseGemmKernel`: `make_fragment_A/B` and the
/// S2T partitions see the complete staged tensors.  `atStage` performs the
/// per-iteration slicing afterwards.
fn makeMxfp4StagedMainloop(
    b: *B,
    tiled_mma: cute.Atom,
    comptime n: i64,
    comptime stages: usize,
    smem_a: V,
    smem_b: V,
    smem_sfa: V,
    smem_sfb: V,
    scale_tmem: V,
) StagedMxfp4Mainloop {
    // Descriptor strides are in 16-byte units: a 128x128 unpacked-FP4 A
    // stage is 16 KiB and an Nx128 FP8 B stage is N*128 bytes.
    const frag_a = b.mmaMakeFragment(
        tiled_mma,
        smem_a,
        0,
        fmt(b, "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(1,1,4,{d}):(0,0,2,1024)\">", .{stages}),
    );
    const frag_b = b.mmaMakeFragment(
        tiled_mma,
        smem_b,
        1,
        fmt(b, "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(1,1,4,{d}):(0,0,2,{d})\">", .{ stages, n * 8 }),
    );

    // TMEM columns: two N-wide accumulator stages, then 16 SFA and 16 SFB
    // columns, matching num_accumulator_tmem_cols/num_sfa_tmem_cols in Python.
    const sf_layout = b.staticValue("!cute.layout<\"((((32,4),4),(32,1)),1,4):((((262144,4),8388608),(0,0)),0,1)\">");
    const sfa_col = b.addOffset(scale_tmem, b.staticValue(fmt(b, "!cute.int_tuple<\"{d}\">", .{2 * n})), "!cute.ptr<f32, tmem, align<16>>");
    const sfb_col = b.addOffset(scale_tmem, b.staticValue(fmt(b, "!cute.int_tuple<\"{d}\">", .{2 * n + 16})), "!cute.ptr<f32, tmem, align<16>>");
    const sfa_ptr = b.recastIter(sfa_col, "!cute.ptr<f8E8M0FNU, tmem, align<16>>");
    const sfb_ptr = b.recastIter(sfb_col, "!cute.ptr<f8E8M0FNU, tmem, align<16>>");
    const sf_memref = "!cute.memref<f8E8M0FNU, tmem, align<16>, \"((((32,4),4),(32,1)),1,4):((((262144,4),8388608),(0,0)),0,1)\">";
    const sfa = b.makeView(sfa_ptr, sf_layout, sf_memref);
    const sfb = b.makeView(sfb_ptr, sf_layout, sf_memref);

    const smem_filtered_ty = fmt(b, "!cute.memref<f8E8M0FNU, smem, align<1024>, \"((((32,4),1),(1,1)),1,4,{d}):((((16,4),0),(0,0)),0,1,512)\">", .{stages});
    const tmem_filtered_ty = "!cute.memref<f8E8M0FNU, tmem, align<16>, \"((((32,4),4),(1,1)),1,4):((((262144,4),8388608),(0,0)),0,1)\">";
    const sfa_smem_filtered = b.filterZeros(smem_sfa, null, smem_filtered_ty);
    const sfb_smem_filtered = b.filterZeros(smem_sfb, null, smem_filtered_ty);
    const sfa_tmem_filtered = b.filterZeros(sfa, null, tmem_filtered_ty);
    const sfb_tmem_filtered = b.filterZeros(sfb, null, tmem_filtered_ty);

    const copy_atom = b.makeAtom("!cute_nvgpu.atom.s2t_copy<f8E8M0FNU, 32 DP, 128 bit, 1 cta, x4>", .{});
    const copy_ty = "!cute.tiled_copy<!cute_nvgpu.atom.s2t_copy<f8E8M0FNU, 32 DP, 128 bit, 1 cta, x4>, layout_copy_tv = <\"(1,(32,(4,4),4)):(0,(1,(512,32),128))\">, tiler_mn = <\"[512:1;1:0;4:1]\">>";
    const sfa_copy = b.makeS2tCopy(copy_atom, sfa_tmem_filtered, copy_ty);
    const sfb_copy = b.makeS2tCopy(copy_atom, sfb_tmem_filtered, copy_ty);

    const source_partition_ty = fmt(b, "!cute.memref<f8E8M0FNU, smem, align<1024>, \"(((32,4,4,4),1),1,1,1,{d}):(((16,1,4,0),0),0,0,0,512)\">", .{stages});
    const source_desc_ty = fmt(b, "!cute_nvgpu.smem_desc_view<!cute_nvgpu.smem_desc, \"(((32,1,1,4),1),1,1,1,{d}):(((1,1,1,0),0),0,0,0,32)\">", .{stages});
    const target_partition_ty = "!cute.memref<f8E8M0FNU, tmem, align<16>, \"(((32,4,(4,4)),1),1,1,1):(((262144,1,(4,8388608)),0),0,0,0)\">";
    const target_grouped_ty = "!cute.memref<f8E8M0FNU, tmem, align<16>, \"(((32,4,(4,4)),1),(1,1,1)):(((262144,1,(4,8388608)),0),(0,0,0))\">";
    const sfa_src_partition = b.tiledCopyPartition(sfa_copy, sfa_smem_filtered, .{0}, false, source_partition_ty);
    const sfb_src_partition = b.tiledCopyPartition(sfb_copy, sfb_smem_filtered, .{0}, false, source_partition_ty);
    const sfa_desc = b.s2tSmemDescriptor(sfa_copy, sfa_src_partition, source_desc_ty);
    const sfb_desc = b.s2tSmemDescriptor(sfb_copy, sfb_src_partition, source_desc_ty);
    const sfa_dst_partition = b.tiledCopyPartition(sfa_copy, sfa_tmem_filtered, .{0}, true, target_partition_ty);
    const sfb_dst_partition = b.tiledCopyPartition(sfb_copy, sfb_tmem_filtered, .{0}, true, target_partition_ty);
    const target_grouped_layout = b.staticValue("!cute.layout<\"(((32,4,(4,4)),1),(1,1,1)):(((262144,1,(4,8388608)),0),(0,0,0))\">");
    const sfa_dst = b.makeView(b.getIter(sfa_dst_partition, "!cute.ptr<f8E8M0FNU, tmem, align<16>>"), target_grouped_layout, target_grouped_ty);
    const sfb_dst = b.makeView(b.getIter(sfb_dst_partition, "!cute.ptr<f8E8M0FNU, tmem, align<16>>"), target_grouped_layout, target_grouped_ty);

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

fn fmt(b: *B, comptime format: []const u8, args: anytype) []const u8 {
    return std.fmt.allocPrint(b.arena.allocator(), format, args) catch @panic("OOM");
}

/// Warp 4 drains the A/B ring into one of two independent accumulator stages.
fn runMmaConsumer(
    b: *B,
    comptime tile: Tile,
    comptime k_tiles: usize,
    storage: StageStorage,
    tiled_mma: cute.Atom,
    tmem_holding: V,
    active: V,
    tiles_m: V,
) void {
    const tmem = b.retrieveTmemPtr(tmem_holding, .f32, 16);
    // Match Python's construction order: derive A/B UMMA fragments and S2T
    // scale descriptors from the complete staged tensors once.  The K loop
    // slices these compiler-derived descriptors by PipelineState.index.
    const staged = stagedOperands(b, tile, storage);
    const staged_mainloop = makeMxfp4StagedMainloop(
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
    // The MMA warp never reads the group, only the tile count.
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
    // Python peeks the first AB-full token before it blocks on an available
    // accumulator stage. Preserve that ordering so the two independent
    // pipelines enter the K loop with the same outstanding operations.
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
        k_tiles,
        1,
        // Match PipelineTmaUmma's Python loop-carried ABI exactly.  The
        // software-pipeline pass recognizes the leading try-wait token and
        // keeps the MMA atom immediately after it.
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

    // Match PipelineUmmaAsync.producer_tail.  The state points at the next
    // accumulator slot; advance to the last slot used by this producer and
    // wait until the epilogue has released it before the MMA warp exits.
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

fn view(b: *B, value: V) cute.View {
    return .{ .inner = value.inner, .kernel = b };
}

/// Epilogue operands; which fields are set depends on the `Epilogue`.
const EpilogueOperands = struct {
    group_sizes: cute.View,
    route_inverse: cute.View,
    routing_weights: ?cute.View = null,
    q: ?cute.View = null,
    s: ?cute.View = null,
    rows: ?cute.View = null,
    direct: bool,
    /// `Config.groupRows`.
    group_rows: i64 = 0,
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
    b.mbarrierTryWaitParity(barrierAt(b, storage.accumulator_full_barriers, stage), accumulator_state.phase, 10_000_000);

    const accumulator_tmem = accumulatorTmemAt(b, tile, tmem, stage);
    const warp_tmem = b.addOffsetTyped(
        accumulator_tmem,
        b.makeIntTuple(.{warp.mul(32 * 65536)}),
        b.ptrTy(.f32, .tmem, 0) catch @panic("bad epilogue TMEM pointer"),
    ).value();
    // A warp lane holds one routed row, so tiles wider than a warp are drained
    // 32 rows at a time. The accumulator stage is released once its last chunk
    // is in registers.
    const chunk_rows: usize = comptime @intCast(tile.chunkRows());
    const tile_chunks = comptime tile.chunks();
    inline for (0..tile_chunks) |chunk| {
        const chunk_tmem = if (chunk == 0) warp_tmem else b.addOffsetTyped(
            warp_tmem,
            b.makeIntTuple(.{@as(i32, @intCast(chunk * chunk_rows))}),
            b.ptrTy(.f32, .tmem, 0) catch @panic("bad epilogue TMEM pointer"),
        ).value();
        // Route metadata does not depend on the accumulator.
        const routed = routedLanes(b, tile, operands, group, chunk);
        const values = b.tmemLoadVector(chunk_tmem, @intCast(chunk_rows), .{ .num_dp = 32, .num_b = 32, .num_rep = @intCast(chunk_rows) });
        b.fenceTmemLoad();
        if (chunk + 1 == tile_chunks) {
            // Registers now own the accumulator values, so all 128 epilogue
            // threads release this TMEM stage before the stores.
            b.mbarrierArrive(barrierAt(b, storage.accumulator_empty_barriers, stage), 1);
        }
        var columns: [@intCast(@min(n, 32))]V = undefined;
        inline for (&columns, 0..) |*column, i| column.* = b.vectorExtract(values, i, .i32).bitCast(.f32);
        switch (epilogue) {
            .swiglu_mxfp8 => storeTileSwiglu(b, tile, storage, operands, routed, tile_m, group, &columns, chunk),
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
    // Permit release and deallocation are warp collective as well.
    b.relinquishTmemAllocPermit(.{});
    release_warp.yieldThen(.{});
    b.namedBarrier(1, 128);
    var dealloc_warp = b.openIf(warp.eq(0));
    b.deallocTmem(tmem, tile.tmem_columns, .{});
    dealloc_warp.yieldThen(.{});
}

/// E8M0 exponent of the power-of-two MX scale for a block maximum, rounded
/// up so that `maximum / scale <= 448`, and the matching inverse scale.
fn mxScale(b: *B, maximum: V) struct { exponent: V, inverse: V } {
    const raw_bits = maximum.maximum(1e-4).mul(1.0 / 448.0).bitCast(.i32);
    const exponent = raw_bits.shrLogical(23).add(raw_bits.bitAnd(0x7fffff).ne(0).to(.i32));
    return .{ .exponent = exponent, .inverse = b.cst(.i32, 254).sub(exponent).shl(23).bitCast(.f32) };
}

/// Per-row routing data of one expert group, spread across a warp: lane `l`
/// holds routed row `l` of the group. Columns index it with `shuffleIdx`.
const RoutedLanes = struct {
    count: V,
    /// Token/top-k route of the lane's row; 0 for padding rows.
    route: V,
    /// Routing weight of the lane's row (up projection only).
    weight: ?V,
};

fn routedLanes(b: *B, comptime tile: Tile, operands: EpilogueOperands, group: V, comptime chunk: usize) RoutedLanes {
    // Only the up projection applies routing weights.
    const weights = operands.routing_weights;
    if (operands.direct) return .{
        .count = b.cst(.i32, 1),
        .route = group,
        .weight = if (weights) |w| w.get(.{group}, .f32) else null,
    };
    const lane = b.threadIdx().x.rem(32);
    const count = operands.group_sizes.get(.{group}, .i32);
    // Lanes past the tile width alias row 0, which is always live. Wide tiles
    // run one 32-row chunk at a time, so lane `l` holds row `32 * chunk + l`.
    const chunk_rows = comptime tile.chunkRows();
    const in_chunk = b.select(lane.lt(@as(i32, @intCast(chunk_rows))), lane, b.cst(.i32, 0));
    const row_in_group = if (chunk == 0) in_chunk else in_chunk.add(@as(i32, @intCast(@as(i64, @intCast(chunk)) * chunk_rows)));
    const live = row_in_group.lt(count);
    const inverse = operands.route_inverse.get(.{group.mul(tile.n).add(row_in_group)}, .i32);
    const route = b.select(live, inverse, b.cst(.i32, 0));
    const weight = if (weights) |w| w.get(.{route}, .f32) else null;
    return .{ .count = count, .route = route, .weight = weight };
}

/// Python `fused_epilogue`: thread `t` owns output row `128 * tile_m + t`
/// of the interleaved projection, so even threads hold a gate and their odd
/// neighbour the matching up value. Each even thread computes one SwiGLU
/// activation per routed row; 32 consecutive activations span two warps,
/// whose partial maxima meet in shared memory.
///
/// At decode sizes most rows of a group are padding, whose outputs are never
/// read. `count` is uniform across the CTA, so whole 8-row chunks of padding
/// are skipped; per-row branches would serialize the shuffle chains of
/// independent rows. Do not duplicate this body under a full-group branch:
/// ptxas then loses track of convergence at the tile loop's warp-uniform
/// shuffles, and lanes still outside the gate-lane stores read garbage.
fn storeTileSwiglu(b: *B, comptime tile: Tile, storage: StageStorage, operands: EpilogueOperands, routed: RoutedLanes, tile_m: V, group: V, columns: []const V, comptime chunk: usize) void {
    // Direct routing has exactly one live row per group.
    if (operands.direct)
        swigluRows(b, tile, storage, operands, routed, tile_m, group, columns, 1, false, chunk)
    else
        swigluRows(b, tile, storage, operands, routed, tile_m, group, columns, comptime @intCast(tile.chunkRows()), true, chunk);
}

const swiglu_chunk = 8;

/// The first `live` rows of `storeTileSwiglu`; with `skip_padding`, chunks
/// of rows past `count` are skipped.
fn swigluRows(b: *B, comptime tile: Tile, storage: StageStorage, operands: EpilogueOperands, routed: RoutedLanes, tile_m: V, group: V, columns: []const V, comptime live: usize, comptime skip_padding: bool, comptime chunk: usize) void {
    const n = comptime tile.chunkRows();
    // First routed row of this epilogue pass.
    const chunk_first: i32 = comptime @intCast(chunk * @as(usize, @intCast(tile.chunkRows())));
    const tid = b.threadIdx().x;
    const lane = tid.rem(32);
    const warp = b.makeWarpUniform(b.makeWarpUniform(tid.div(32)));
    const is_gate = tid.rem(2).eq(0);
    const first_row = group.mul(operands.group_rows);
    const f32_type = cute.DType.f32.toMlir(b.ctx);
    // The TMA C staging buffer is unused by this epilogue.
    const scratch_layout = b.layoutSpec(.{ 4, n }, .{ n, 1 });
    const scratch = b.makeViewTyped(storage.c, b.staticLayout(scratch_layout), b.memrefType(.f32, .smem, 1024, b.layoutType(scratch_layout)));
    const row_chunks = std.math.divCeil(usize, live, swiglu_chunk) catch unreachable;

    var activations: [@intCast(n)]V = undefined;
    // Lane `c` collects the warp maximum of column `c`.
    var lane_maximum = b.cst(.f32, 0);
    for (0..row_chunks) |row_chunk| {
        const first = row_chunk * swiglu_chunk;
        const last = @min(first + swiglu_chunk, live);
        // Row 0 of an active group is always live.
        const guarded = skip_padding and (row_chunk > 0 or chunk > 0);
        const types: [swiglu_chunk + 1]*const mlir.Type = @splat(f32_type);
        // `@TypeOf` does not evaluate its operand.
        var guard: ?@TypeOf(b.openIfElse(routed.count, tupleOf(types))) = if (guarded) b.openIfElse(routed.count.gt(chunk_first + @as(i32, @intCast(first))), tupleOf(types)) else null;
        var chunk_maximum = lane_maximum;
        for (columns[first..last], activations[first..last], first..) |value, *activation, column| {
            const c: i32 = @intCast(column);
            // The reference model rounds the projection to BF16.
            const gate = value.to(.bf16).to(.f32).minimum(10.0);
            const up_value = b.shuffleXor(value, 1).to(.bf16).to(.f32).maximum(-10.0).minimum(10.0);
            // up * SiLU(gate), with the hardware approximations Python's
            // fastmath selects: exp(-g) = 2^(-g * log2 e).
            const exp_neg = b.unaryF32("ex2.approx.ftz.f32", gate.mul(-std.math.log2e));
            var result = up_value.mul(gate).mul(b.unaryF32("rcp.approx.ftz.f32", exp_neg.add(1.0)));
            result = result.mul(b.shuffleIdx(routed.weight.?, c)).to(.bf16).to(.f32);
            // Odd threads hold no activation and must not raise the maxima.
            activation.* = b.select(is_gate, result, b.cst(.f32, 0));
            const maximum = b.warpMaxNonNegative(activation.*.abs());
            chunk_maximum = b.select(lane.eq(c), maximum, chunk_maximum);
        }
        if (guard) |*g| {
            var then_values: [swiglu_chunk + 1]V = @splat(chunk_maximum);
            for (activations[first..last], then_values[0 .. last - first]) |activation, *t| t.* = activation;
            g.yieldThen(tupleOf(then_values));
            // Padding chunk: zero activations, maxima unchanged. The zero is
            // emitted in the else block, where it is used.
            var else_values: [swiglu_chunk + 1]V = @splat(b.cst(.f32, 0));
            else_values[swiglu_chunk] = lane_maximum;
            g.yieldElse(tupleOf(else_values));
            for (activations[first..last], g.results[0 .. last - first]) |*activation, result| activation.* = result;
            lane_maximum = g.results[swiglu_chunk];
        } else {
            lane_maximum = chunk_maximum;
        }
    }
    var owner = b.openIf(lane.lt(@as(i32, @intCast(n))));
    scratch.set(.{ warp, lane }, lane_maximum);
    owner.yieldThen(.{});
    b.namedBarrier(1, 128);

    // Lane `c` turns the two half-block maxima of column `c` into its scale.
    const pair = warp.bitAnd(~@as(i32, 1));
    const lane_column = b.select(lane.lt(@as(i32, @intCast(n))), lane, b.cst(.i32, 0));
    const scale = mxScale(b, scratch.get(.{ pair, lane_column }, .f32).maximum(scratch.get(.{ pair.add(1), lane_column }, .f32)));
    // The next tile reuses the scratch maxima.
    b.namedBarrier(1, 128);

    const activation_index = tile_m.mul(64).add(tid.div(2));
    for (0..row_chunks) |row_chunk| {
        const first = row_chunk * swiglu_chunk;
        const last = @min(first + swiglu_chunk, live);
        var guard: ?@TypeOf(b.openIf(routed.count)) = if (skip_padding and (chunk > 0 or first > 0)) b.openIf(routed.count.gt(chunk_first + @as(i32, @intCast(first)))) else null;
        var inverses: [swiglu_chunk]V = undefined;
        for (inverses[0 .. last - first], first..) |*inverse, column| inverse.* = b.shuffleIdx(scale.inverse, @as(i32, @intCast(column)));
        var gate_lane = b.openIf(is_gate);
        for (activations[first..last], inverses[0 .. last - first], first..) |activation, inverse, column| {
            const quantized = activation.mul(inverse).maximum(-448.0).minimum(448.0).to(.f8e4m3fn);
            operands.q.?.set(.{ first_row.add(chunk_first + @as(i32, @intCast(column))), activation_index }, quantized);
        }
        gate_lane.yieldThen(.{});
        if (guard) |*g| g.yieldThen(.{});
    }

    // Warps 0 and 2 each own one 32-activation block of this tile.
    var scale_lane = b.openIf(warp.rem(2).eq(0).bitAnd(lane.add(chunk_first).lt(routed.count)));
    const block = tile_m.mul(2).add(warp.div(2));
    const swizzled = block.div(4).mul(512).add(lane.mul(16)).add(chunk * 4).add(block.rem(4));
    operands.s.?.set(.{ group, swizzled }, scale.exponent.to(.i8));
    scale_lane.yieldThen(.{});
}

/// Python `down_epilogue`: store each live routed row at its token/top-k
/// route, so the final reduction reads rows in route order.
fn storeTileRows(b: *B, comptime tile: Tile, operands: EpilogueOperands, routed: RoutedLanes, tile_m: V, columns: []const V, comptime chunk: usize) void {
    const chunk_first: i32 = comptime @intCast(chunk * @as(usize, @intCast(tile.chunkRows())));
    const output_column = tile_m.mul(128).add(b.threadIdx().x);
    if (operands.direct) {
        operands.rows.?.set(.{ routed.route, output_column }, columns[0].to(.bf16));
        return;
    }
    var routes: [32]V = undefined;
    for (routes[0..columns.len], 0..) |*route, column| route.* = b.shuffleIdx(routed.route, @as(i32, @intCast(column)));
    for (columns, routes[0..columns.len], 0..) |value, route, column| {
        var live: ?@TypeOf(b.openIf(routed.count)) = if (column == 0 and chunk == 0) null else b.openIf(routed.count.gt(chunk_first + @as(i32, @intCast(column))));
        operands.rows.?.set(.{ route, output_column }, value.to(.bf16));
        if (live) |*l| l.yieldThen(.{});
    }
}

fn buildUmmaDevice(b: *B, cfg: Config, comptime tile: Tile, comptime epilogue: Epilogue, d: anytype) void {
    const L = Layouts(tile);
    const n = tile.n;
    const stages = tile.stages;
    // Allocate the complete Python SharedStorage object as one compiler-visible
    // memref. `cute.kernel_smem_size` only accounts for memref allocations;
    // treating every field as an architecture pointer left the launch at zero
    // dynamic shared bytes and made the first TMA instruction fault.
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
        .b = sharedPointerAt(b, shared_base, tile.b, .f8e4m3fn, 1024),
        .sfa = sharedPointerAt(b, shared_base, tile.sfa, .f8e8m0fnu, 1024),
        .sfb = sharedPointerAt(b, shared_base, tile.sfb, .f8e8m0fnu, 1024),
        .load_barriers = sharedPointerAt(b, shared_base, 0, .i64, 1024),
        .mma_barriers = sharedPointerAt(b, shared_base, tile.empty_barriers, .i64, 8),
        .accumulator_full_barriers = sharedPointerAt(b, shared_base, tile.accumulator_full, .i64, 16),
        .accumulator_empty_barriers = sharedPointerAt(b, shared_base, tile.accumulator_empty, .i64, 16),
    };

    const tid = b.threadIdx().x;
    const warp = b.makeWarpUniform(b.makeWarpUniform(tid.div(32)));
    const tiled_mma: cute.Atom = .{ .inner = d.mma.inner, .kernel = b };

    // Reproduce the Python CuTe path for the two matrix operands:
    // coordinate tensor -> local_tile -> tiled_mma.partition -> group modes
    // -> tma_partition. All dependent result types come from Zig layout
    // objects; this kernel does not embed textual MLIR.
    const m_tiles = @divExact(cfg.m, 128);
    const k_tiles = @divExact(cfg.k, 128);
    const a_local_spec = b.layoutSpec(
        .{ 128, 128, m_tiles, k_tiles, cfg.experts },
        .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), b.basis(128, 1, .{1}), b.basis(128, 1, .{0}), b.basis(1, 1, .{2}) },
    );
    const b_local_spec = b.layoutSpec(
        .{ n, 128, 1, k_tiles, cfg.groups },
        .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}), b.basis(n, 1, .{1}), b.basis(128, 1, .{0}), b.basis(1, 1, .{2}) },
    );
    const all_tiles = .{ cute.AlgebraToken.all, cute.AlgebraToken.all, cute.AlgebraToken.all };
    const tiled_a = b.localTileTyped(d.gA, b.makeTile(&.{ 128, 128 }), all_tiles, b.coordTensorType(3, a_local_spec), null);
    const tiled_b = b.localTileTyped(d.gB, b.makeTile(&.{ n, 128 }), all_tiles, b.coordTensorType(3, b_local_spec), null);

    const a_mma_spec = b.layoutSpec(
        .{ .{ 128, 32 }, 1, 4, m_tiles, k_tiles, cfg.experts },
        .{ .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}) }, 0, b.basis(32, 1, .{0}), b.basis(128, 1, .{1}), b.basis(128, 1, .{0}), b.basis(1, 1, .{2}) },
    );
    const b_mma_spec = b.layoutSpec(
        .{ .{ n, 32 }, 1, 4, 1, k_tiles, cfg.groups },
        .{ .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}) }, 0, b.basis(32, 1, .{0}), b.basis(n, 1, .{1}), b.basis(128, 1, .{0}), b.basis(1, 1, .{2}) },
    );
    const mma_a = b.tiledMmaPartitionTyped(tiled_mma, tiled_a, .{0}, 0, b.coordTensorType(3, a_mma_spec));
    const mma_b = b.tiledMmaPartitionTyped(tiled_mma, tiled_b, .{0}, 1, b.coordTensorType(3, b_mma_spec));

    const a_grouped_layout = b.staticLayout(b.layoutSpec(
        .{ .{ .{ 128, 32 }, 1, 4 }, m_tiles, k_tiles, cfg.experts },
        .{ .{ .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}) }, 0, b.basis(32, 1, .{0}) }, b.basis(128, 1, .{1}), b.basis(128, 1, .{0}), b.basis(1, 1, .{2}) },
    ));
    const b_grouped_layout = b.staticLayout(b.layoutSpec(
        .{ .{ .{ n, 32 }, 1, 4 }, 1, k_tiles, cfg.groups },
        .{ .{ .{ b.basis(1, 1, .{1}), b.basis(1, 1, .{0}) }, 0, b.basis(32, 1, .{0}) }, b.basis(n, 1, .{1}), b.basis(128, 1, .{0}), b.basis(1, 1, .{2}) },
    ));
    const staged = stagedOperands(b, tile, storage);
    const smem_a_grouped = b.withLayout(staged.a, b.staticLayout(b.layoutSpec(
        .{ .{ .{ 128, 32 }, 1, 4 }, stages },
        .{ .{ .{ 128, 1 }, 0, 32 }, Tile.a_stage_bytes },
    )));
    const smem_b_grouped = b.withLayout(staged.b, b.staticLayout(b.layoutSpec(
        .{ .{ .{ n, 32 }, 1, 4 }, stages },
        .{ .{ .{ 128, 1 }, 0, 32 }, tile.bStageBytes() },
    )));
    const one_cta = b.staticLayout(b.layoutSpec(.{1}, .{0}));
    const a_partition_layout = b.layoutSpec(.{ .{ Tile.a_stage_bytes, 1 }, stages }, .{ .{ 1, 0 }, Tile.a_stage_bytes });
    const b_partition_layout = b.layoutSpec(.{ .{ tile.bStageBytes(), 1 }, stages }, .{ .{ 1, 0 }, tile.bStageBytes() });
    const a_target_layout = b.layoutSpec(
        .{ .{ .{ 128, 128 }, 1 }, m_tiles, k_tiles, cfg.experts },
        .{ .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }, b.basis(128, 1, .{1}), b.basis(128, 1, .{0}), b.basis(1, 1, .{2}) },
    );
    const b_target_layout = b.layoutSpec(
        .{ .{ .{ 128, n }, 1 }, 1, k_tiles, cfg.groups },
        .{ .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0 }, b.basis(n, 1, .{1}), b.basis(128, 1, .{0}), b.basis(1, 1, .{2}) },
    );
    const partitioned_a = b.tmaPartitionTyped(
        .{ .inner = d.tma_a.inner, .kernel = b },
        .{0},
        one_cta,
        smem_a_grouped,
        .{b.withLayout(mma_a, a_grouped_layout)},
        b.memrefTypeFromPointer(b.swizzledPtrTy(.i8, .smem, 1024, L.ab_swizzle) catch @panic("bad smem pointer"), b.layoutType(a_partition_layout)),
        &.{b.coordTensorType(3, a_target_layout)},
    );
    const partitioned_b = b.tmaPartitionTyped(
        .{ .inner = d.tma_b.inner, .kernel = b },
        .{0},
        one_cta,
        smem_b_grouped,
        .{b.withLayout(mma_b, b_grouped_layout)},
        b.memrefTypeFromPointer(b.swizzledPtrTy(.f8e4m3fn, .smem, 1024, L.ab_swizzle) catch @panic("bad smem pointer"), b.layoutType(b_partition_layout)),
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

    // Match CUTLASS PipelineTmaUmma/PipelineUmmaAsync initialization.  Warp 0
    // owns initialization, and one elected lane initializes each complete
    // barrier array.  mbarrier.init is not warp collective: issuing it from
    // every lane races on the same shared state and breaks the first phase
    // transition when the A/B ring wraps.
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

    // Each role walks the same persistent tile sequence independently. The
    // AB and accumulator mbarriers carry all cross-warp dependencies.
    var tma_warp = b.openIf(warp.eq(5));
    // The TMA producer consumes buffers produced by the preceding quantizer.
    // PDL allows this grid to launch early, so wait before the first TMA read.
    b.waitForDependency();
    switch (k_tiles) {
        inline 18, 40 => |kt| runTmaProducer(b, tile, kt, storage, active, d.tiles_m, expert_ids, direct, partitioned_a.targets[0], partitioned_b.targets[0], exec_a, exec_b, exec_sfa, exec_sfb),
        else => std.debug.panic("unsupported MXFP4 reduction tile count {d}", .{k_tiles}),
    }
    tma_warp.yieldThen(.{});

    var mma_warp = b.openIf(warp.eq(4));
    // Python's MMA role reaches its own TmemAllocator.wait_for_alloc site.
    b.namedBarrier(2, 160);
    switch (k_tiles) {
        inline 18, 40 => |kt| runMmaConsumer(b, tile, kt, storage, tiled_mma, tmem_holding, active, d.tiles_m),
        else => unreachable,
    }
    mma_warp.yieldThen(.{});

    var epilogue_warps = b.openIf(warp.lt(4));
    // TmemAllocator.allocate is issued collectively by warp 0.  The four
    // epilogue warps then meet the MMA warp at named barrier 2.
    var alloc_warp = b.openIf(warp.eq(0));
    b.allocTmem(tile.tmem_columns, tmem_holding, .{});
    alloc_warp.yieldThen(.{});
    b.namedBarrier(2, 160);
    switch (epilogue) {
        .swiglu_mxfp8 => runEpilogue(b, tile, epilogue, storage, tmem_holding, active, d.tiles_m, direct, .{
            .group_sizes = view(b, d.group_sizes),
            .route_inverse = view(b, d.route_inverse),
            .routing_weights = view(b, d.routing_weights),
            .q = view(b, d.q),
            .s = view(b, d.s),
            .direct = cfg.direct,
            .group_rows = cfg.groupRows(),
        }),
        .route_rows => runEpilogue(b, tile, epilogue, storage, tmem_holding, active, d.tiles_m, direct, .{
            .group_sizes = view(b, d.group_sizes),
            .route_inverse = view(b, d.route_inverse),
            .rows = view(b, d.rows),
            .direct = cfg.direct,
        }),
    }
    epilogue_warps.yieldThen(.{});
}

fn buildHostLaunch(b: *B, cfg: Config, comptime tile: Tile, comptime epilogue: Epilogue) cute.FinishError!void {
    const L = Layouts(tile);
    const host_arg = HostArgs(epilogue);
    const n = tile.n;
    const a_layout = b.layoutSpec(.{ cfg.m, cfg.k, cfg.experts }, .{ cfg.k, 1, cfg.m * cfg.k });
    const b_layout = b.layoutSpec(.{ cfg.groupRows(), cfg.k, cfg.groups }, .{ cfg.k, 1, cfg.k * cfg.groupRows() });
    const dynamic_sf_layout = b.layoutSpec(
        .{ .{ .{ 32, 4 }, cute.AlgebraToken.dynamic }, .{ .{ 32, 4 }, cute.AlgebraToken.dynamic }, .{ 1, cute.AlgebraToken.dynamic } },
        .{ .{ .{ 16, 4 }, cute.AlgebraToken.dynamic }, .{ .{ 0, 1 }, 512 }, .{ 0, cute.AlgebraToken.dynamic } },
    );

    const gA = b.makeTensorView(b.recastPointer(b.arg(host_arg.weight), .f4e2m1fn, .gmem, 16), b.staticLayout(a_layout));
    // `schedule` is `[group experts | group sizes | active count]`.
    const schedule = b.arg(host_arg.schedule);
    const ids = b.makeTensorView(schedule, b.staticLayout(b.layoutSpec(cfg.groups, 1)));
    const active_ptr = b.addOffsetTyped(
        schedule,
        b.makeIntTuple(.{2 * cfg.groups}),
        b.ptrTy(.i32, .gmem, 4) catch @panic("bad active count pointer"),
    );
    const active = b.makeTensorView(active_ptr, b.staticLayout(b.layoutSpec(1, 1)));
    const gB = b.makeTensorView(b.arg(host_arg.input_quant), b.staticLayout(b_layout));
    // Match Python's host construction instead of materializing these two
    // layouts with `cute.static`. In particular, SFB's M extent is one; CuTe
    // must still retain that mode while encoding the tensor map.
    const sf_atom_layout = b.staticLayout(b.layoutSpec(
        .{ .{ 32, 4 }, .{ 32, 4 } },
        .{ .{ 16, 4 }, .{ 0, 1 } },
    ));
    // The Zig dialect binding exposes the zero-based form of Python CuTe's
    // `(2, 1, 3)` tile-to-shape order.
    const sf_order = b.makeIntTuple(.{ 1, 0, 2 });
    const sfa_shape = b.makeShapeValues(.{ b.arg(host_arg.problem_m), b.arg(host_arg.problem_k), b.arg(host_arg.expert_count) }, b.parseType("!cute.shape<\"(?,?,?)\">"));
    const sfb_shape = b.makeShapeValues(.{ b.arg(host_arg.problem_n), b.arg(host_arg.problem_k), b.arg(host_arg.problem_l) }, b.parseType("!cute.shape<\"(?,?,?)\">"));
    const dynamic_sfa_layout = b.tileToShapeTyped(sf_atom_layout, sfa_shape, sf_order, b.layoutType(dynamic_sf_layout));
    const dynamic_sfb_layout = b.tileToShapeTyped(sf_atom_layout, sfb_shape, sf_order, b.layoutType(dynamic_sf_layout));
    const gSFA = b.makeTensorView(b.recastPointer(b.arg(host_arg.weight_scale), .f8e8m0fnu, .gmem, 16), dynamic_sfa_layout);
    const gSFB = b.makeTensorView(b.recastPointer(b.arg(host_arg.input_scale), .f8e8m0fnu, .gmem, 16), dynamic_sfb_layout);

    const zero_tmem = b.intToPtr(0, .f8e8m0fnu, .tmem, 1);
    const mma = b.makeBlockScaledMma(.{ .n = @intCast(n) }, zero_tmem);
    const mma_sfb = b.makeBlockScaledMma(.{ .n = 128 }, zero_tmem);

    const a_smem = b.makeComposedLayout(L.ab_swizzle, 0, L.aSmem(b, false));
    const b_smem = b.makeComposedLayout(L.ab_swizzle, 0, L.bSmem(b, false));
    const sf_smem = b.staticLayout(L.sfSmem(b, false));
    const a_map = b.staticLayout(b.layoutSpec(.{ .{ 128, 32 }, 1, 4 }, .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0, b.basis(32, 1, .{1}) }));
    const b_map = b.staticLayout(b.layoutSpec(.{ .{ n, 32 }, 1, 4 }, .{ .{ b.basis(1, 1, .{0}), b.basis(1, 1, .{1}) }, 0, b.basis(32, 1, .{1}) }));
    const sf_map = b.staticLayout(b.layoutSpec(
        .{ .{ .{ 32, 4 }, 32 }, 1, 4 },
        .{ .{ .{ b.basis(1, 1, .{ 0, 0, 0 }), b.basis(1, 1, .{ 1, 0, 0 }) }, b.basis(1, 1, .{ 0, 0, 1 }) }, 0, b.basis(1, 1, .{ 1, 0, 1 }) },
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
        .dynamic_smem = b.kernelSmemSize(device_name),
        .stream = b.cudaStream(),
        .cluster = .{ one, one, one },
        .use_pdl = true,
    });
    const mainloop = .{
        ids,          active,         mma.tiled,    mma_sfb.tiled,  tma_a.atom, tma_a.tensor, tma_b.atom, tma_b.tensor,
        tma_sfa.atom, tma_sfa.tensor, tma_sfb.atom, tma_sfb.tensor,
    };
    const activations = @divExact(cfg.m, 2);
    const group_sizes = b.makeTensorView(
        b.addOffsetTyped(schedule, b.makeIntTuple(.{cfg.groups}), b.ptrTy(.i32, .gmem, 4) catch @panic("bad group size pointer")),
        b.staticLayout(b.layoutSpec(cfg.groups, 1)),
    );
    const launch = switch (epilogue) {
        .swiglu_mxfp8 => b.launchEx(device_name, config, mainloop ++ .{
            group_sizes,
            b.makeTensorView(b.arg(host_arg.route_inverse), b.staticLayout(b.layoutSpec(cfg.routedRows(), 1))),
            b.makeTensorView(b.arg(host_arg.routing_weights), b.staticLayout(b.layoutSpec(@max(cfg.routes, 1), 1))),
            b.makeTensorView(b.arg(host_arg.output), b.staticLayout(b.layoutSpec(.{ cfg.routedRows(), activations }, .{ activations, 1 }))),
            b.makeTensorView(b.arg(host_arg.scales), b.staticLayout(b.layoutSpec(.{ cfg.groups, @divExact(activations, 32) * 128 }, .{ @divExact(activations, 32) * 128, 1 }))),
            tiles_m,
        }),
        .route_rows => b.launchEx(device_name, config, mainloop ++ .{
            group_sizes,
            b.makeTensorView(b.arg(host_arg.route_inverse), b.staticLayout(b.layoutSpec(cfg.routedRows(), 1))),
            b.makeTensorView(b.arg(host_arg.output), b.staticLayout(b.layoutSpec(.{ @max(cfg.routes, 1), cfg.m }, .{ cfg.m, 1 }))),
            tiles_m,
        }),
    };
    b.returnHostStatus(b.cudaResultStatus(launch));
}

test "MXFP4 GEMM programs emit their host launch for every tile width" {
    inline for (.{ 8, 16, 32 }) |n| {
        inline for (.{ .{ UpQuantized, Epilogue.swiglu_mxfp8, 4608, 5120 }, .{ DownRows, Epilogue.route_rows, 5120, 2304 } }) |program| {
            const cfg: Config = .{
                .experts = 192,
                .m = program[2],
                .n = n,
                .k = program[3],
                .groups = 227,
                .epilogue = program[1],
                .routes = 576,
                .direct = n == 8,
            };
            const ir = try program[0].emit(std.testing.allocator, cfg);
            defer std.testing.allocator.free(ir);
            try std.testing.expect(std.mem.indexOf(u8, ir, device_name) != null);
            try std.testing.expect(std.mem.indexOf(u8, ir, "cuda.launch_ex") != null);
            try std.testing.expect(std.mem.indexOf(u8, ir, "make_non_exec_tiled_tma_load") != null);
            try std.testing.expect(std.mem.indexOf(u8, ir, std.fmt.comptimePrint("sm100.mma_bs<128x{d}x32", .{n})) != null);
        }
    }
}
