//! Host-side construction of DeepGEMM's MegaMoE tensor maps with the CuTe builder: each
//! `layout.TensorMapSpec` (the parameters DeepGEMM passes to `cuTensorMapEncodeTiled`) becomes
//! a `Builder.RawTmaMap` over the tensor's base pointer, so the descriptor has DeepGEMM's
//! rank, extents, strides, box, swizzle and data type. The kernel issues DeepGEMM's
//! `cp.async.bulk.tensor` PTX on `Builder.rawTmaDescAddr`.
const std = @import("std");

const cute = @import("kernels/cute/builder");

const layout = @import("layout.zig");

const B = cute.Builder;

/// A `RawTmaMap` with its own storage.
pub const Map = struct {
    rank: usize,
    dims: [5]i64,
    strides: [4]i64,
    box: [5]i64,
    dtype: cute.DType,
    format: cute.TmaFormat,
    swizzle_bytes: u32,
    cta_group: cute.CtaGroup,
    store: bool,

    /// DeepGEMM's descriptor parameters as a CuTe raw map. FP4 (`16U4_ALIGN16B`) extents
    /// and strides count 4-bit elements; DeepGEMM's byte strides are converted.
    pub fn init(spec: layout.TensorMapSpec) Map {
        var m: Map = undefined;
        m.rank = spec.rank;
        m.dtype, m.format = switch (spec.data_type) {
            .uint8 => .{ .f8e4m3fn, .default },
            .int32 => .{ .i32, .s32 },
            .uint32 => .{ .i32, .u32 },
            .@"16u4_align16b" => .{ .f4e2m1fn, .u4_unpack_u8 },
            else => std.debug.panic("unsupported tensor-map data type {s}", .{@tagName(spec.data_type)}),
        };
        const element_bits: u64 = switch (spec.data_type) {
            .uint8 => 8,
            .int32, .uint32 => 32,
            .@"16u4_align16b" => 4,
            else => unreachable,
        };
        for (0..spec.rank) |i| {
            m.dims[i] = @intCast(spec.global_dims[i]);
            m.box[i] = spec.box_dims[i];
            std.debug.assert(spec.element_strides[i] == 1);
        }
        for (0..spec.rank - 1) |i| m.strides[i] = @intCast(spec.global_strides[i] * 8 / element_bits);
        m.swizzle_bytes = switch (spec.swizzle) {
            .none => 0,
            .@"32b" => 32,
            .@"64b" => 64,
            .@"128b" => 128,
            else => unreachable,
        };
        m.cta_group = if (spec.two_cta) .cta_2 else .cta_1;
        m.store = !spec.two_cta;
        std.debug.assert(spec.interleave == .none);
        return m;
    }

    pub fn raw(self: *const Map) B.RawTmaMap {
        return .{
            .dtype = self.dtype,
            .format = self.format,
            .dims = self.dims[0..self.rank],
            .strides = self.strides[0 .. self.rank - 1],
            .box = self.box[0..self.rank],
            .swizzle_bytes = self.swizzle_bytes,
            .cta_group = self.cta_group,
            .store = self.store,
        };
    }
};

// ==================== DeepGEMM's exact descriptors: `cuTensorMapEncodeTiled` ====================
// CuTe's inline encoder (`Builder.makeRawTmaAtom`) writes L2 promotion 128B (DeepGEMM: 256B) and
// sets qword 1 bit 1. `cuda.tensor_map_encode_tiled` (the CuTe DSL's `create_tensor_map_tiled`)
// calls the driver's `cuTensorMapEncodeTiled` from the host function (XLA binds the DSL runtime's
// `_cuTensorMapEncodeTiled` to it), so the kernel gets DeepGEMM's descriptor bit for bit. The
// kernel declares the map as a `!cuda.tensor_map` argument marked `cuda.grid_constant` (DeepGEMM's
// `const __grid_constant__ CUtensorMap`) and issues its PTX on the parameter's generic address.

const mlir = @import("mlir");
const dialects = @import("mlir/dialects");

/// `!cuda.tensor_map`: the kernel argument type of an explicitly encoded map.
pub fn tensorMapType(b: *B) *const mlir.Type {
    return (dialects.cute.cuda.TensorMapType.get(b.ctx, .{}) catch unreachable).type_();
}

/// A host stack array of `values` (u64 -> i64 slots, u32 -> i32 slots): a `cuuint64_t*` /
/// `cuuint32_t*` argument of `cuTensorMapEncodeTiled`.
fn hostArray(b: *B, comptime T: type, values: []const T) cute.Value {
    const ctx = b.ctx;
    const elem: *const mlir.Type = if (T == u64) .int(ctx, .i64) else .int(ctx, .i32);
    const ptr_type = dialects.llvm.pointerType(ctx, dialects.llvm.NVPTXAddressSpace.generic);
    const array = b.emit(mlir.Operation.make(ctx, "llvm.alloca", .{
        .operands = .{ .flat = &.{b.cst(.i32, @as(i32, @intCast(@max(values.len, 1)))).inner} },
        .results = .{ .flat = &.{ptr_type} },
        .attributes = &.{ .named(ctx, "elem_type", .typeAttr(elem)), .named(ctx, "alignment", .int(ctx, .i64, 8)) },
        .location = b.loc(),
    }));
    for (values, 0..) |v, i| {
        const slot = b.emit(mlir.Operation.make(ctx, "llvm.getelementptr", .{
            .operands = .{ .flat = &.{array.inner} },
            .results = .{ .flat = &.{ptr_type} },
            .attributes = &.{ .named(ctx, "elem_type", .typeAttr(elem)), .named(ctx, "rawConstantIndices", .denseArray(ctx, .i32, &[_]i32{@intCast(i)})) },
            .location = b.loc(),
        }));
        const value = if (T == u64) b.cst(.i64, @as(i64, @intCast(v))) else b.cst(.i32, @as(i32, @intCast(v)));
        b.emitVoid(dialects.llvm.store(ctx, value.inner, slot.inner, .{ .alignment = @sizeOf(T) }, b.loc()));
    }
    return array;
}

/// Host side: `cuTensorMapEncodeTiled` with DeepGEMM's parameters `spec` over the global
/// address `address` (i64, the tensor's base plus `spec.base_offset`). Returns the
/// `!cuda.tensor_map` value to pass to the launch, and the call's `!cuda.result`.
pub fn encodeTiled(b: *B, spec: layout.TensorMapSpec, address: cute.Value) struct { map: cute.Value, result: cute.Value } {
    const ctx = b.ctx;
    // The `cuda` dialect's address operand is a `ptr` dialect pointer, which zml does not register.
    ctx.setAllowUnregisteredDialects(true);
    const addr = b.emit(dialects.cuda.cast(ctx, b.parseType("!ptr.ptr<#ptr.generic_space>"), address.inner, b.loc()));
    const results = b.emitResults(2, dialects.cute.cuda.tensor_map_encode_tiled(
        ctx,
        b.cst(.i32, @as(i32, @intCast(@backingInt(spec.data_type)))).inner,
        b.cst(.i32, @as(i32, @intCast(spec.rank))).inner,
        addr.inner,
        hostArray(b, u64, spec.global_dims[0..spec.rank]).inner,
        hostArray(b, u64, spec.global_strides[0 .. spec.rank - 1]).inner,
        hostArray(b, u32, spec.box_dims[0..spec.rank]).inner,
        hostArray(b, u32, spec.element_strides[0..spec.rank]).inner,
        b.cst(.i32, @as(i32, @intCast(@backingInt(spec.interleave)))).inner,
        b.cst(.i32, @as(i32, @intCast(@backingInt(spec.swizzle)))).inner,
        b.cst(.i32, @as(i32, @intCast(@backingInt(spec.l2_promotion)))).inner,
        b.cst(.i32, @as(i32, @intCast(@backingInt(spec.oob_fill)))).inner,
        dialects.cuda.resultType(ctx),
        tensorMapType(b),
        b.loc(),
    ));
    return .{ .map = results[1], .result = results[0] };
}

/// Device side: the generic address (i64) of the `!cuda.tensor_map` kernel argument `arg`
/// (DeepGEMM's `&tensor_map` of a `__grid_constant__` parameter).
pub fn tensorMapAddr(b: *B, arg: cute.Value) cute.Value {
    const ctx = b.ctx;
    const llvm_ptr = b.emit(dialects.cuda.cast(ctx, dialects.llvm.pointerType(ctx, dialects.llvm.NVPTXAddressSpace.generic), arg.inner, b.loc()));
    return b.emit(mlir.Operation.make(ctx, "llvm.ptrtoint", .{
        .operands = .{ .flat = &.{llvm_ptr.inner} },
        .results = .{ .flat = &.{cute.DType.i64.toMlir(ctx)} },
        .location = b.loc(),
    }));
}
