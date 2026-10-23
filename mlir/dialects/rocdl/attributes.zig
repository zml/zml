// Generated from LLVM ROCDL TableGen records; see REGENERATING.md.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

const c = @import("c");
const mlir = @import("mlir");
const enums = @import("enums.zig");

pub const TargetAttr = @import("target.zig").TargetAttr;

/// `#rocdl.buffer_oob_mode`.
pub const BufferOOBModeAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.BufferOOBMode;
    pub const isAFn = c.mlirAttributeIsAROCDLBufferOOBMode;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLBufferOOBModeAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLBufferOOBModeAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.buffer_oob_mode_flag`.
pub const BufferOOBModeModuleFlagAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.BufferOOBMode;
    pub const isAFn = c.mlirAttributeIsAROCDLBufferOOBModeModuleFlag;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLBufferOOBModeModuleFlagAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLBufferOOBModeModuleFlagAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.gfx12_atomic_cache_policy`.
pub const Gfx12AtomicCachePolicyAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.Gfx12AtomicCachePolicy;
    pub const isAFn = c.mlirAttributeIsAROCDLGfx12AtomicCachePolicy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLGfx12AtomicCachePolicyAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLGfx12AtomicCachePolicyAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.gfx12_cache_policy`.
pub const Gfx12CachePolicyAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.Gfx12CachePolicy;
    pub const isAFn = c.mlirAttributeIsAROCDLGfx12CachePolicy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLGfx12CachePolicyAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLGfx12CachePolicyAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.gfx942_cache_policy`.
pub const Gfx942CachePolicyAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.Gfx942CachePolicy;
    pub const isAFn = c.mlirAttributeIsAROCDLGfx942CachePolicy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLGfx942CachePolicyAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLGfx942CachePolicyAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.mfma_neg_modifier`.
pub const MFMANegModifierAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.MFMANegModifier;
    pub const isAFn = c.mlirAttributeIsAROCDLMFMANegModifier;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLMFMANegModifierAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLMFMANegModifierAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.mfma_perm_b`.
pub const MFMAPermBAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.MFMAPermB;
    pub const isAFn = c.mlirAttributeIsAROCDLMFMAPermB;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLMFMAPermBAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLMFMAPermBAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.matrix_format`.
pub const MatrixFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.MatrixFormat;
    pub const isAFn = c.mlirAttributeIsAROCDLMatrixFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLMatrixFormatAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLMatrixFormatAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.pre_gfx12_cache_policy`.
pub const PreGfx12CachePolicyAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.PreGfx12CachePolicy;
    pub const isAFn = c.mlirAttributeIsAROCDLPreGfx12CachePolicy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLPreGfx12CachePolicyAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLPreGfx12CachePolicyAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.sched_group_mask`.
pub const SchedGroupMaskAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.SchedGroupMask;
    pub const isAFn = c.mlirAttributeIsAROCDLSchedGroupMask;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLSchedGroupMaskAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLSchedGroupMaskAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.tbuffer_oob_mode_flag`.
pub const TBufferOOBModeModuleFlagAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.BufferOOBMode;
    pub const isAFn = c.mlirAttributeIsAROCDLTBufferOOBModeModuleFlag;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLTBufferOOBModeModuleFlagAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLTBufferOOBModeModuleFlagAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.wmma_c_modifier`.
pub const WMMACModifierAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.WMMACModifier;
    pub const isAFn = c.mlirAttributeIsAROCDLWMMACModifier;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLWMMACModifierAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLWMMACModifierAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.wmma_matrix_scale`.
pub const WMMAMatrixScaleAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.WMMAMatrixScale;
    pub const isAFn = c.mlirAttributeIsAROCDLWMMAMatrixScale;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLWMMAMatrixScaleAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLWMMAMatrixScaleAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

/// `#rocdl.wmma_matrix_scale_format`.
pub const WMMAMatrixScaleFormatAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const Value = enums.WMMAMatrixScaleFormat;
    pub const isAFn = c.mlirAttributeIsAROCDLWMMAMatrixScaleFormat;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        value: Value,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLWMMAMatrixScaleFormatAttrGet(ctx.ptr(), @backingInt(args.value)).ptr orelse return error.InvalidMlir);
    }

    pub fn value(self: *const Self) Value {
        return @fromBackingInt(@intCast(c.mlirROCDLWMMAMatrixScaleFormatAttrGetValue(self.ptr())));
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
};

pub const enum_attributes = .{
    BufferOOBModeAttr,
    BufferOOBModeModuleFlagAttr,
    Gfx12AtomicCachePolicyAttr,
    Gfx12CachePolicyAttr,
    Gfx942CachePolicyAttr,
    MFMANegModifierAttr,
    MFMAPermBAttr,
    MatrixFormatAttr,
    PreGfx12CachePolicyAttr,
    SchedGroupMaskAttr,
    TBufferOOBModeModuleFlagAttr,
    WMMACModifierAttr,
    WMMAMatrixScaleAttr,
    WMMAMatrixScaleFormatAttr,
};
