//! Native matrix instruction descriptions. No runtime or DSL dependencies.
const std = @import("std");

pub const Format = enum {
    f32,
    f16,
    bf16,
    tf32,
    i8,
    u8,
    i4,
    u4,
    f8e4m3fn,
    f8e5m2,
    f8e4m3fnuz,
    f8e5m2fnuz,
    f6e2m3,
    f6e3m2,
    f4e2m1,
    mxfp8_e4m3,
    mxfp8_e5m2,
    mxfp6_e2m3,
    mxfp6_e3m2,
    mxfp4,
    nvfp4,

    pub const Set = std.EnumSet(Format);

    pub fn bits(self: Format) u8 {
        return switch (self) {
            .f16, .bf16 => 16,
            .f32, .tf32 => 32,
            .i8, .u8, .f8e4m3fn, .f8e5m2, .f8e4m3fnuz, .f8e5m2fnuz, .mxfp8_e4m3, .mxfp8_e5m2 => 8,
            .f6e2m3, .f6e3m2, .mxfp6_e2m3, .mxfp6_e3m2 => 6,
            .i4, .u4, .f4e2m1, .mxfp4, .nvfp4 => 4,
        };
    }

    pub fn scaling(self: Format) ?Scaling {
        return switch (self) {
            .mxfp8_e4m3, .mxfp8_e5m2, .mxfp6_e2m3, .mxfp6_e3m2, .mxfp4 => .{ .encoding = .ue8m0, .block_k = 32 },
            .nvfp4 => .{ .encoding = .ue4m3, .block_k = 16 },
            .f32, .f16, .bf16, .tf32, .i8, .u8, .i4, .u4, .f8e4m3fn, .f8e5m2, .f8e4m3fnuz, .f8e5m2fnuz, .f6e2m3, .f6e3m2, .f4e2m1 => null,
        };
    }
};

pub const Scaling = struct {
    encoding: enum { ue8m0, ue4m3 },
    /// One scale per this many logical elements along K, per row of A / column of B.
    block_k: u16,
};

pub const Tile = struct { m: u16, n: u16, k: u16 };
pub const Accumulator = enum { f16, f32, i32 };

/// Subgroup width is distinct from the instruction's issue scope. For DPAS it
/// denotes SIMD channels, not a count of independently issuing GPU threads.
pub const Execution = struct {
    subgroup_size: u16,
    issue_scope: enum { subgroup, warpgroup, single_thread, simd },
    cta_group: u8,

    pub fn issuingThreads(self: Execution) u16 {
        return switch (self.issue_scope) {
            .subgroup => self.subgroup_size,
            .warpgroup => 4 * self.subgroup_size,
            .single_thread, .simd => 1,
        };
    }
};

/// WGMMA and tcgen05 accept N in steps of 8 up to 256 for a fixed M and K.
fn nSweep(comptime m: u16, comptime k: u16) [32]Tile {
    var shapes: [32]Tile = undefined;
    for (&shapes, 1..) |*shape, i| shape.* = .{ .m = m, .n = @intCast(8 * i), .k = k };
    return shapes;
}
const wgmma_k8 = nSweep(64, 8);
const wgmma_k16 = nSweep(64, 16);
const wgmma_k32 = nSweep(64, 32);
// tcgen05 M=64 and the cta_group::2 M=256 shapes are not modeled yet.
const tcgen05_k32 = nSweep(128, 32);
const tcgen05_k64 = nSweep(128, 64);
const tcgen05_k96 = nSweep(128, 96);

/// A form identifies an encoding/layout family, with legal types and shapes
/// in description(). Different RDNA operand ABIs deliberately have distinct IDs.
/// This is a catalog of modeled forms, not the entire matrix ISA.
// Sources: NVIDIA PTX dense MMA, WGMMA and tcgen05 instruction tables;
// LLVM AMDGPU VOP3PInstructions.td and RDNA3/4 ISA manuals; Intel vISA DPAS.
// https://docs.nvidia.com/cuda/parallel-thread-execution/
// https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/VOP3PInstructions.td
// https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md
// mfma_scale with unscaled formats requires neutral scale exponents.
pub const Instruction = enum {
    wmma_f16_f16,
    wmma_f16_f32,
    mma_f16_f16,
    mma_f16_f32,
    mma_i8_i32,
    mma_i4_i32,
    mma_bf16_f32,
    mma_tf32_f32,
    mma_fp8_f32,
    mma_f8f6f4_f32,
    mma_mxf8f6f4_f32,
    mma_nvfp4_f32,
    tcgen05_f8f6f4_f32,
    tcgen05_mxf8f6f4_f32,
    tcgen05_nvfp4_f32,
    tcgen05_nvfp4_k96_f32,
    wgmma_f16_f32,
    wgmma_bf16_f32,
    wgmma_tf32_f32,
    wgmma_fp8_f32,
    mfma_f32_16x16x4f32,
    mfma_f32_f16,
    /// gfx908/gfx90a two-element BF16 forms; gfx940+ keep only the 1k forms.
    mfma_f32_bf16,
    /// gfx908/gfx90a INT8 forms; gfx940+ replace them with gfx940_mfma_i32_i8.
    mfma_i32_i8,
    mfma_f32_16x16x16bf16_1k,
    gfx940_mfma_i32_i8,
    mfma_f32_fp8_fnuz,
    mfma_f32_fp8,
    mfma_scale_f32_f8f6f4,
    mfma_scale_f32_mxf8f6f4,
    gfx11_wmma_f16_f32,
    gfx11_wmma_bf16_f32,
    gfx11_wmma_i8_i32,
    gfx11_wmma_i4_i32,
    gfx12_wmma_f16_f32,
    gfx12_wmma_bf16_f32,
    gfx12_wmma_i8_i32,
    gfx12_wmma_i4_i32,
    gfx12_wmma_fp8_f32,
    dpas_simd8_f16_f16,
    dpas_simd8_f16_f32,
    dpas_simd8_bf16_f32,
    dpas_simd8_i8_i32,
    dpas_simd16_f16_f16,
    dpas_simd16_f16_f32,
    dpas_simd16_bf16_f32,
    dpas_simd16_i8_i32,

    pub fn description(self: Instruction) Capability {
        return switch (self) {
            .wmma_f16_f16 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f16,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .wmma_f16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mma_f16_f16 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f16,
                .native_shapes = &.{.{ .m = 16, .n = 8, .k = 8 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mma_f16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 8, .k = 8 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mma_i8_i32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .i8, .u8 }),
                .accumulator = .i32,
                .native_shapes = &.{.{ .m = 8, .n = 8, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mma_i4_i32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .i4, .u4 }),
                .accumulator = .i32,
                .native_shapes = &.{.{ .m = 8, .n = 8, .k = 32 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mma_bf16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.bf16}),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 16, .n = 8, .k = 8 },
                    .{ .m = 16, .n = 8, .k = 16 },
                },
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mma_tf32_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.tf32}),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 16, .n = 8, .k = 4 },
                    .{ .m = 16, .n = 8, .k = 8 },
                },
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mma_fp8_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .f8e4m3fn, .f8e5m2 }),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 16, .n = 8, .k = 16 },
                    .{ .m = 16, .n = 8, .k = 32 },
                },
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mma_f8f6f4_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .f8e4m3fn, .f8e5m2, .f6e2m3, .f6e3m2, .f4e2m1 }),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 8, .k = 32 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mma_mxf8f6f4_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .mxfp8_e4m3, .mxfp8_e5m2, .mxfp6_e2m3, .mxfp6_e3m2, .mxfp4 }),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 8, .k = 32 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mma_nvfp4_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.nvfp4}),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 8, .k = 64 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .tcgen05_f8f6f4_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .f8e4m3fn, .f8e5m2, .f6e2m3, .f6e3m2, .f4e2m1 }),
                .accumulator = .f32,
                .native_shapes = &tcgen05_k32,
                .execution = .{ .subgroup_size = 32, .issue_scope = .single_thread, .cta_group = 1 },
            },
            .tcgen05_mxf8f6f4_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .mxfp8_e4m3, .mxfp8_e5m2, .mxfp6_e2m3, .mxfp6_e3m2, .mxfp4 }),
                .accumulator = .f32,
                .native_shapes = &tcgen05_k32,
                .execution = .{ .subgroup_size = 32, .issue_scope = .single_thread, .cta_group = 1 },
            },
            .tcgen05_nvfp4_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.nvfp4}),
                .accumulator = .f32,
                .native_shapes = &tcgen05_k64,
                .execution = .{ .subgroup_size = 32, .issue_scope = .single_thread, .cta_group = 1 },
            },
            .tcgen05_nvfp4_k96_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.nvfp4}),
                .accumulator = .f32,
                .native_shapes = &tcgen05_k96,
                .execution = .{ .subgroup_size = 32, .issue_scope = .single_thread, .cta_group = 1 },
            },
            .wgmma_f16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f32,
                .native_shapes = &wgmma_k16,
                .execution = .{ .subgroup_size = 32, .issue_scope = .warpgroup, .cta_group = 1 },
            },
            .wgmma_bf16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.bf16}),
                .accumulator = .f32,
                .native_shapes = &wgmma_k16,
                .execution = .{ .subgroup_size = 32, .issue_scope = .warpgroup, .cta_group = 1 },
            },
            .wgmma_tf32_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.tf32}),
                .accumulator = .f32,
                .native_shapes = &wgmma_k8,
                .execution = .{ .subgroup_size = 32, .issue_scope = .warpgroup, .cta_group = 1 },
            },
            .wgmma_fp8_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .f8e4m3fn, .f8e5m2 }),
                .accumulator = .f32,
                .native_shapes = &wgmma_k32,
                .execution = .{ .subgroup_size = 32, .issue_scope = .warpgroup, .cta_group = 1 },
            },
            .mfma_f32_16x16x4f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.f32}),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 4 }},
                .execution = .{ .subgroup_size = 64, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mfma_f32_f16 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 16, .n = 16, .k = 16 },
                    .{ .m = 32, .n = 32, .k = 8 },
                },
                .execution = .{ .subgroup_size = 64, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mfma_f32_bf16 => .{
                .instruction = self,
                .formats = .initMany(&.{.bf16}),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 16, .n = 16, .k = 8 },
                    .{ .m = 32, .n = 32, .k = 4 },
                },
                .execution = .{ .subgroup_size = 64, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mfma_i32_i8 => .{
                .instruction = self,
                .formats = .initMany(&.{.i8}),
                .accumulator = .i32,
                .native_shapes = &.{
                    .{ .m = 16, .n = 16, .k = 16 },
                    .{ .m = 32, .n = 32, .k = 8 },
                },
                .execution = .{ .subgroup_size = 64, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .gfx940_mfma_i32_i8 => .{
                .instruction = self,
                .formats = .initMany(&.{.i8}),
                .accumulator = .i32,
                .native_shapes = &.{
                    .{ .m = 16, .n = 16, .k = 32 },
                    .{ .m = 32, .n = 32, .k = 16 },
                },
                .execution = .{ .subgroup_size = 64, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mfma_f32_16x16x16bf16_1k => .{
                .instruction = self,
                .formats = .initMany(&.{.bf16}),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 64, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mfma_f32_fp8_fnuz => .{
                .instruction = self,
                .formats = .initMany(&.{ .f8e4m3fnuz, .f8e5m2fnuz }),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 16, .n = 16, .k = 32 },
                    .{ .m = 32, .n = 32, .k = 16 },
                },
                .execution = .{ .subgroup_size = 64, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mfma_f32_fp8 => .{
                .instruction = self,
                .formats = .initMany(&.{ .f8e4m3fn, .f8e5m2 }),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 16, .n = 16, .k = 32 },
                    .{ .m = 32, .n = 32, .k = 16 },
                },
                .execution = .{ .subgroup_size = 64, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mfma_scale_f32_f8f6f4 => .{
                .instruction = self,
                .formats = .initMany(&.{ .f8e4m3fn, .f8e5m2, .f6e2m3, .f6e3m2, .f4e2m1 }),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 16, .n = 16, .k = 128 },
                    .{ .m = 32, .n = 32, .k = 64 },
                },
                .execution = .{ .subgroup_size = 64, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .mfma_scale_f32_mxf8f6f4 => .{
                .instruction = self,
                .formats = .initMany(&.{ .mxfp8_e4m3, .mxfp8_e5m2, .mxfp6_e2m3, .mxfp6_e3m2, .mxfp4 }),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 16, .n = 16, .k = 128 },
                    .{ .m = 32, .n = 32, .k = 64 },
                },
                .execution = .{ .subgroup_size = 64, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .gfx11_wmma_f16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .gfx11_wmma_bf16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.bf16}),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .gfx11_wmma_i8_i32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .i8, .u8 }),
                .accumulator = .i32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .gfx11_wmma_i4_i32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .i4, .u4 }),
                .accumulator = .i32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .gfx12_wmma_f16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .gfx12_wmma_bf16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.bf16}),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .gfx12_wmma_i8_i32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .i8, .u8 }),
                .accumulator = .i32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .gfx12_wmma_i4_i32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .i4, .u4 }),
                .accumulator = .i32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .gfx12_wmma_fp8_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .f8e4m3fn, .f8e5m2 }),
                .accumulator = .f32,
                .native_shapes = &.{.{ .m = 16, .n = 16, .k = 16 }},
                .execution = .{ .subgroup_size = 32, .issue_scope = .subgroup, .cta_group = 1 },
            },
            .dpas_simd8_f16_f16 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f16,
                .native_shapes = &.{
                    .{ .m = 1, .n = 8, .k = 16 },
                    .{ .m = 2, .n = 8, .k = 16 },
                    .{ .m = 3, .n = 8, .k = 16 },
                    .{ .m = 4, .n = 8, .k = 16 },
                    .{ .m = 5, .n = 8, .k = 16 },
                    .{ .m = 6, .n = 8, .k = 16 },
                    .{ .m = 7, .n = 8, .k = 16 },
                    .{ .m = 8, .n = 8, .k = 16 },
                },
                .execution = .{ .subgroup_size = 8, .issue_scope = .simd, .cta_group = 1 },
            },
            .dpas_simd8_f16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 1, .n = 8, .k = 16 },
                    .{ .m = 2, .n = 8, .k = 16 },
                    .{ .m = 3, .n = 8, .k = 16 },
                    .{ .m = 4, .n = 8, .k = 16 },
                    .{ .m = 5, .n = 8, .k = 16 },
                    .{ .m = 6, .n = 8, .k = 16 },
                    .{ .m = 7, .n = 8, .k = 16 },
                    .{ .m = 8, .n = 8, .k = 16 },
                },
                .execution = .{ .subgroup_size = 8, .issue_scope = .simd, .cta_group = 1 },
            },
            .dpas_simd8_bf16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.bf16}),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 1, .n = 8, .k = 16 },
                    .{ .m = 2, .n = 8, .k = 16 },
                    .{ .m = 3, .n = 8, .k = 16 },
                    .{ .m = 4, .n = 8, .k = 16 },
                    .{ .m = 5, .n = 8, .k = 16 },
                    .{ .m = 6, .n = 8, .k = 16 },
                    .{ .m = 7, .n = 8, .k = 16 },
                    .{ .m = 8, .n = 8, .k = 16 },
                },
                .execution = .{ .subgroup_size = 8, .issue_scope = .simd, .cta_group = 1 },
            },
            .dpas_simd8_i8_i32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .i8, .u8 }),
                .accumulator = .i32,
                .native_shapes = &.{
                    .{ .m = 1, .n = 8, .k = 32 },
                    .{ .m = 2, .n = 8, .k = 32 },
                    .{ .m = 3, .n = 8, .k = 32 },
                    .{ .m = 4, .n = 8, .k = 32 },
                    .{ .m = 5, .n = 8, .k = 32 },
                    .{ .m = 6, .n = 8, .k = 32 },
                    .{ .m = 7, .n = 8, .k = 32 },
                    .{ .m = 8, .n = 8, .k = 32 },
                },
                .execution = .{ .subgroup_size = 8, .issue_scope = .simd, .cta_group = 1 },
            },
            .dpas_simd16_f16_f16 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f16,
                .native_shapes = &.{
                    .{ .m = 1, .n = 16, .k = 16 },
                    .{ .m = 2, .n = 16, .k = 16 },
                    .{ .m = 3, .n = 16, .k = 16 },
                    .{ .m = 4, .n = 16, .k = 16 },
                    .{ .m = 5, .n = 16, .k = 16 },
                    .{ .m = 6, .n = 16, .k = 16 },
                    .{ .m = 7, .n = 16, .k = 16 },
                    .{ .m = 8, .n = 16, .k = 16 },
                },
                .execution = .{ .subgroup_size = 16, .issue_scope = .simd, .cta_group = 1 },
            },
            .dpas_simd16_f16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.f16}),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 1, .n = 16, .k = 16 },
                    .{ .m = 2, .n = 16, .k = 16 },
                    .{ .m = 3, .n = 16, .k = 16 },
                    .{ .m = 4, .n = 16, .k = 16 },
                    .{ .m = 5, .n = 16, .k = 16 },
                    .{ .m = 6, .n = 16, .k = 16 },
                    .{ .m = 7, .n = 16, .k = 16 },
                    .{ .m = 8, .n = 16, .k = 16 },
                },
                .execution = .{ .subgroup_size = 16, .issue_scope = .simd, .cta_group = 1 },
            },
            .dpas_simd16_bf16_f32 => .{
                .instruction = self,
                .formats = .initMany(&.{.bf16}),
                .accumulator = .f32,
                .native_shapes = &.{
                    .{ .m = 1, .n = 16, .k = 16 },
                    .{ .m = 2, .n = 16, .k = 16 },
                    .{ .m = 3, .n = 16, .k = 16 },
                    .{ .m = 4, .n = 16, .k = 16 },
                    .{ .m = 5, .n = 16, .k = 16 },
                    .{ .m = 6, .n = 16, .k = 16 },
                    .{ .m = 7, .n = 16, .k = 16 },
                    .{ .m = 8, .n = 16, .k = 16 },
                },
                .execution = .{ .subgroup_size = 16, .issue_scope = .simd, .cta_group = 1 },
            },
            .dpas_simd16_i8_i32 => .{
                .instruction = self,
                .formats = .initMany(&.{ .i8, .u8 }),
                .accumulator = .i32,
                .native_shapes = &.{
                    .{ .m = 1, .n = 16, .k = 32 },
                    .{ .m = 2, .n = 16, .k = 32 },
                    .{ .m = 3, .n = 16, .k = 32 },
                    .{ .m = 4, .n = 16, .k = 32 },
                    .{ .m = 5, .n = 16, .k = 32 },
                    .{ .m = 6, .n = 16, .k = 32 },
                    .{ .m = 7, .n = 16, .k = 32 },
                    .{ .m = 8, .n = 16, .k = 32 },
                },
                .execution = .{ .subgroup_size = 16, .issue_scope = .simd, .cta_group = 1 },
            },
        };
    }
};

pub const Capability = struct {
    instruction: Instruction,
    /// Every A/B pairing within this set is legal. Incompatible pairs use
    /// separate forms, including scaled versus unscaled operand encodings.
    formats: Format.Set,
    accumulator: Accumulator,
    native_shapes: []const Tile,
    execution: Execution,

    pub fn supportsNative(self: Capability, a: Format, b: Format, acc: Accumulator, shape: Tile) bool {
        if (!self.formats.contains(a) or !self.formats.contains(b) or self.accumulator != acc) return false;
        for (self.native_shapes) |native| {
            if (std.meta.eql(native, shape)) return true;
        }
        return false;
    }

    /// Arithmetic only: a tile can be composed from a modeled native shape.
    /// Does not establish compiler lowering, operand layout or resource support.
    pub fn supports(self: Capability, a: Format, b: Format, tile: Tile) bool {
        if (!self.formats.contains(a) or !self.formats.contains(b)) return false;
        for (self.native_shapes) |native| {
            if (tile.m != 0 and tile.n != 0 and tile.k != 0 and
                tile.m % native.m == 0 and tile.n % native.n == 0 and tile.k % native.k == 0) return true;
        }
        return false;
    }
};
