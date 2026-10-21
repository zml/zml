//! Machinery shared by the MXFP4 and NVFP4 CuTe MoE kernels: pipeline
//! barriers and state, the static persistent tile scheduler, direct routing,
//! shared-memory pointers, boundary kernel launches and small IR helpers.
const std = @import("std");
const mlir = @import("mlir");

const zml = @import("../../../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const V = cute.Value;

pub fn typed(type_: *const mlir.Type) struct { mlir_type: cute.ArgSpec.MlirTypeSpec } {
    return .{ .mlir_type = .{ .type_ = type_ } };
}

pub fn gridTyped(type_: *const mlir.Type) struct { mlir_type: cute.ArgSpec.MlirTypeSpec } {
    return .{ .mlir_type = .{ .type_ = type_, .grid_constant = true } };
}

pub fn barrierAt(b: *B, base: V, stage: V) V {
    return b.addOffsetTyped(
        base,
        b.makeIntTuple(.{stage}),
        b.ptrTy(.i64, .smem, 8) catch @panic("bad pipeline barrier pointer"),
    ).value();
}

pub const PipelinePosition = struct {
    index: V,
    phase: V,
};

/// Scalar form of CUTLASS's StaticPersistentTileScheduler state for the
/// one-CTA DeepSeek specialization.  Python carries the decoded M/N/L
/// coordinate, validity predicate, linear work index, and executed-tile
/// count through every role's `scf.while`.  Keeping the same state shape is
/// important to the CuTe software-pipeline passes, even though N is always
/// zero for one tile per group.
pub const PersistentWork = struct {
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
pub const DirectRoutes = struct {
    expert_ids: cute.View,
    routes: usize,

    pub fn count(self: DirectRoutes, b: *B) V {
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

pub fn persistentWorkAt(
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

pub fn advancePersistentWork(
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
pub fn advancePipeline(b: *B, index: V, phase: V, comptime stages: usize) PipelinePosition {
    const advanced = index.add(1);
    const wraps = advanced.eq(stages);
    const next_index = b.select(wraps, 0, advanced);
    var next_phase = b.openIfElse(wraps, .{phase.type_()});
    next_phase.yieldThen(.{phase.bitXor(1)});
    next_phase.yieldElse(.{phase});
    return .{ .index = next_index, .phase = next_phase.results[0] };
}

pub fn peekBarrierIfRemaining(b: *B, base: V, count: V, limit: usize, state: PipelinePosition) V {
    var peek = b.openIfElse(count.lt(limit), .{cute.DType.i1.toMlir(b.ctx)});
    peek.yieldThen(.{b.mbarrierWaitParity(barrierAt(b, base, state.index), state.phase)});
    peek.yieldElse(.{b.cst(.i1, true)});
    return peek.results[0];
}

/// `cute.add_offset` infers the alignment a pointer keeps as the gcd of the
/// base alignment and the offset, and `cute.recast_iter` may not change it, so
/// the field's own alignment has to be what the offset leaves.
pub fn sharedPointerAt(
    b: *B,
    base: V,
    comptime byte_offset: i64,
    comptime dtype: cute.DType,
    comptime alignment: u64,
) V {
    const moved = comptime movedAlignment(byte_offset);
    comptime std.debug.assert(moved >= alignment);
    const byte_ptr = b.addOffsetTyped(
        base,
        b.makeIntTuple(.{byte_offset}),
        b.ptrTy(.i8, .smem, moved) catch @panic("bad shared byte pointer"),
    );
    return b.recastPointer(byte_ptr, dtype, .smem, moved);
}

/// Alignment of `shared_base + byte_offset`, which the shared storage is
/// allocated with.
fn movedAlignment(comptime byte_offset: i64) u64 {
    return offsetAlignment(1024, byte_offset, 1);
}

/// `cute.add_offset` keeps the gcd of the operand alignment and the offset,
/// and the op's result type has to say so.
pub fn offsetAlignment(base_bytes: u64, offset_elements: i64, element_bytes: u64) u64 {
    if (offset_elements == 0) return base_bytes;
    const moved = @as(u64, @intCast(@abs(offset_elements))) * element_bytes;
    return @max(std.math.gcd(base_bytes, moved), element_bytes);
}

pub fn Tuple(comptime T: type, comptime len: usize) type {
    return @Tuple(&@as([len]type, @splat(T)));
}

pub fn arity(comptime T: type) usize {
    return @typeInfo(T).@"struct".field_names.len;
}

/// The MLIR types of a tuple of values, as `openWhile` expects them.
pub fn typesOf(values: anytype) Tuple(*const mlir.Type, arity(@TypeOf(values))) {
    const len = comptime arity(@TypeOf(values));
    var out: Tuple(*const mlir.Type, len) = undefined;
    inline for (0..len) |i| out[i] = values[i].type_();
    return out;
}

/// Forward a whole loop-carried array, as `yieldBefore` expects a tuple.
pub fn tupleOf(array: anytype) Tuple(@TypeOf(array[0]), array.len) {
    var out: Tuple(@TypeOf(array[0]), array.len) = undefined;
    inline for (0..array.len) |i| out[i] = array[i];
    return out;
}

pub fn fmt(b: *B, comptime format: []const u8, args: anytype) []const u8 {
    return std.fmt.allocPrint(b.arena.allocator(), format, args) catch @panic("OOM");
}

pub fn view(b: *B, value: V) cute.View {
    return .{ .inner = value.inner, .kernel = b };
}

/// Host launch of a boundary kernel: forward the buffer pointers, with PDL.
pub fn launchBoundary(b: *B, kernel_name: [:0]const u8, grid: [3]i32, args: anytype) cute.FinishError!void {
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.cst(.i32, grid[0]), b.cst(.i32, grid[1]), b.cst(.i32, grid[2]) },
        .block = .{ b.cst(.i32, 128), one, one },
        .dynamic_smem = b.kernelSmemSize(kernel_name),
        .stream = b.cudaStream(),
        .cluster = .{ one, one, one },
        .use_pdl = true,
    });
    const launch = b.launchEx(kernel_name, config, args);
    b.returnHostStatus(b.cudaResultStatus(launch));
}
