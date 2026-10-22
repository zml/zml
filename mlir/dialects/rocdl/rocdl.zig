//! LLVM ROCDL operations, AMDGPU attributes, and GPU target configuration.
//! All operation builders from the pinned LLVM revision are available in `ops`.

const mlir = @import("mlir");

pub const types = @import("types.zig");
pub const annotations = @import("annotations.zig");
pub const AddressSpace = types.AddressSpace;
pub const pointerType = types.pointerType;

pub const ops = @import("ops.zig");
pub const enums = @import("enums.zig");
pub const attributes = @import("attributes.zig");

pub const BufferOOBModeAttr = attributes.BufferOOBModeAttr;
pub const BufferOOBModeModuleFlagAttr = attributes.BufferOOBModeModuleFlagAttr;
pub const Gfx12AtomicCachePolicyAttr = attributes.Gfx12AtomicCachePolicyAttr;
pub const Gfx12CachePolicyAttr = attributes.Gfx12CachePolicyAttr;
pub const Gfx942CachePolicyAttr = attributes.Gfx942CachePolicyAttr;
pub const MFMANegModifierAttr = attributes.MFMANegModifierAttr;
pub const MFMAPermBAttr = attributes.MFMAPermBAttr;
pub const MatrixFormatAttr = attributes.MatrixFormatAttr;
pub const PreGfx12CachePolicyAttr = attributes.PreGfx12CachePolicyAttr;
pub const SchedGroupMaskAttr = attributes.SchedGroupMaskAttr;
pub const TBufferOOBModeModuleFlagAttr = attributes.TBufferOOBModeModuleFlagAttr;
pub const TargetAttr = attributes.TargetAttr;
pub const WMMACModifierAttr = attributes.WMMACModifierAttr;
pub const WMMAMatrixScaleAttr = attributes.WMMAMatrixScaleAttr;
pub const WMMAMatrixScaleFormatAttr = attributes.WMMAMatrixScaleFormatAttr;

pub const BufferOOBMode = enums.BufferOOBMode;
pub const Gfx12AtomicCachePolicy = enums.Gfx12AtomicCachePolicy;
pub const Gfx12CachePolicy = enums.Gfx12CachePolicy;
pub const Gfx942CachePolicy = enums.Gfx942CachePolicy;
pub const MFMANegModifier = enums.MFMANegModifier;
pub const MFMAPermB = enums.MFMAPermB;
pub const MatrixFormat = enums.MatrixFormat;
pub const PreGfx12CachePolicy = enums.PreGfx12CachePolicy;
pub const SchedGroupMask = enums.SchedGroupMask;
pub const WMMACModifier = enums.WMMACModifier;
pub const WMMAMatrixScale = enums.WMMAMatrixScale;
pub const WMMAMatrixScaleFormat = enums.WMMAMatrixScaleFormat;

pub const dialect_namespace = "rocdl";

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("rocdl");
}

pub const operation_names = ops.operation_names;

/// Converts two packed FP8 values from the selected word to vector<2xf32>.
pub fn cvtPkF32Fp8(
    ctx: *mlir.Context,
    bits: *const mlir.Value,
    high: bool,
    location: *const mlir.Location,
) *mlir.Operation {
    return ops.cvt_pk_f32_fp8(ctx, bits, .vector(&.{2}, .float(ctx, .f32)), .boolean(ctx, high), location);
}
