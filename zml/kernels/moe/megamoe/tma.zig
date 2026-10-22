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
