//! Native CuteNVGPU bindings backed by the typed C API.
//! Payload-backed types take a StringAttr containing the complete payload,
//! including angle brackets (for example <f32>).

const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");
const stdx = @import("stdx");

pub const dialect_namespace = "cute_nvgpu";

pub const NVGPUType = opaque {
    const M = mlir.Methods(NVGPUType, c.MlirType);

    pub const isAFn = c.mlirTypeIsACuteNVGPUType;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;

    pub fn format(self: *const NVGPUType, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const t: *const mlir.Type = @ptrCast(self);
        return t.format(writer);
    }
};

/// `!cute_nvgpu.atom.bulk_copy_g2s`.
pub const CopyAtomBulkCopyG2SType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomBulkCopyG2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomBulkCopyG2STypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomBulkCopyG2STypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.bulk_copy_s2g`.
pub const CopyAtomBulkCopyS2GType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2G;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomBulkCopyS2GTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.bulk_copy_s2s`.
pub const CopyAtomBulkCopyS2SType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomBulkCopyS2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomBulkCopyS2STypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomBulkCopyS2STypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.dsmem_store`.
pub const CopyAtomDsmemStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomDsmemStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomDsmemStoreTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomDsmemStoreTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.g2r`.
pub const CopyAtomG2RType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomG2R;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomG2RTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomG2RTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.im2col_tma_load`.
pub const CopyAtomIm2ColTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaLoadTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.im2col_tma_store`.
pub const CopyAtomIm2ColTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomIm2ColTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomIm2ColTmaStoreTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.ldsm`.
pub const CopyAtomLdsmType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomLdsm;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomLdsmTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomLdsmTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.non_exec_2d_gather4_tma_load`.
pub const CopyAtomNonExec2DGather4TmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExec2DGather4TmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExec2DGather4TmaLoadTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.non_exec_im2col_tma_load`.
pub const CopyAtomNonExecIm2ColTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaLoadTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.non_exec_im2col_tma_store`.
pub const CopyAtomNonExecIm2ColTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecIm2ColTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecIm2ColTmaStoreTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.non_exec_tiled_tma_load`.
pub const CopyAtomNonExecTiledTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaLoadTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.non_exec_tiled_tma_reduce`.
pub const CopyAtomNonExecTiledTmaReduceType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaReduceTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.non_exec_tiled_tma_store`.
pub const CopyAtomNonExecTiledTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomNonExecTiledTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomNonExecTiledTmaStoreTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.r2g`.
pub const CopyAtomR2GType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomR2G;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomR2GTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomR2GTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.r2s`.
pub const CopyAtomR2SType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomR2S;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomR2STypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomR2STypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.s2r`.
pub const CopyAtomS2RType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomS2R;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomS2RTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomS2RTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.simt_async_copy`.
pub const CopyAtomSIMTAsyncCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTAsyncCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTAsyncCopyTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.simt_multimem_ld_reduce`.
pub const CopyAtomSIMTMultimemLdReduceType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemLdReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTMultimemLdReduceTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.simt_multimem_red`.
pub const CopyAtomSIMTMultimemRedType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemRed;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTMultimemRedTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.simt_multimem_st`.
pub const CopyAtomSIMTMultimemStType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTMultimemSt;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTMultimemStTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.universal_copy`.
pub const CopyAtomSIMTSyncCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSIMTSyncCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSIMTSyncCopyTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.tmem_load_red`.
pub const CopyAtomSM10xTmemLoadRedType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM10xTmemLoadRed;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM10xTmemLoadRedTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.s2t_copy`.
pub const CopyAtomSM100CopyS2TType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100CopyS2T;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100CopyS2TTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.sm100_s2t_copy_v2`.
pub const CopyAtomSM100S2TCopyV2Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100S2TCopyV2;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100S2TCopyV2TypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.tmem_load`.
pub const CopyAtomSM100TmemLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100TmemLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100TmemLoadTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.tmem_store`.
pub const CopyAtomSM100TmemStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomSM100TmemStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomSM100TmemStoreTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.stsm`.
pub const CopyAtomStsmType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomStsm;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomStsmTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomStsmTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.tma_load`.
pub const CopyAtomTmaLoadType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomTmaLoad;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomTmaLoadTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaLoadTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.tma_reduce`.
pub const CopyAtomTmaReduceType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomTmaReduce;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomTmaReduceTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaReduceTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.atom.tma_store`.
pub const CopyAtomTmaStoreType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUCopyAtomTmaStore;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUCopyAtomTmaStoreTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUCopyAtomTmaStoreTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm80.sparse_mma`.
pub const MmaAtomSM80SparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM80Sparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM80SparseTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80SparseTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm80.mma`.
pub const MmaAtomSM80Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM80;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM80TypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM80TypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm89.mma`.
pub const MmaAtomSM89Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM89;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM89TypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM89TypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm90.mma`.
pub const MmaAtomSM90Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM90;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM90TypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM90TypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm100.mma_bs_sp`.
pub const MmaAtomSM100UMMABlockScaledSparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaledSparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledSparseTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm100.mma_bs`.
pub const MmaAtomSM100UMMABlockScaledType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMABlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMABlockScaledTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm100.mma_sp`.
pub const MmaAtomSM100UMMASparseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMASparse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMASparseTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm100.mma`.
pub const MmaAtomSM100UMMAType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM100UMMA;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM100UMMATypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM100UMMATypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.SM120.mma_bs`.
pub const MmaAtomSM120BlockScaledType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUMmaAtomSM120BlockScaled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUMmaAtomSM120BlockScaledTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.sm103.smem_desc_circular`.
pub const SmemDescCircularSM103Type = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDescCircularSM103;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSmemDescCircularSM103TypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUSmemDescCircularSM103TypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.smem_desc`.
pub const SmemDescType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDesc;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUSmemDescTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cute_nvgpu.smem_desc_view`.
pub const SmemDescViewType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUSmemDescView;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUSmemDescViewTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUSmemDescViewTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.tiled_copy`.
pub const TiledCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUTiledCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTiledCopyTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUTiledCopyTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.tiled_mma`.
pub const TiledMmaType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUTiledMma;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUTiledMmaTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUTiledMmaTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.tma_descriptor_im2col`.
pub const TmaDescriptorIm2ColType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUTmaDescriptorIm2Col;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUTmaDescriptorIm2ColTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cute_nvgpu.tma_descriptor_tiled`.
pub const TmaDescriptorTiledType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUTmaDescriptorTiled;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUTmaDescriptorTiledTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!cute_nvgpu.atom.universal_fma`.
pub const UniversalFmaAtomType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUUniversalFmaAtom;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {
        payload: *const mlir.Attribute,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirCuteNVGPUUniversalFmaAtomTypeGet(ctx.ptr(), args.payload.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
    pub fn getPayload(self: *const Self) *const mlir.Attribute {
        return @ptrCast(c.mlirCuteNVGPUUniversalFmaAtomTypeGetPayload(self.ptr()).ptr.?);
    }
};

/// `!cute_nvgpu.workid_response`.
pub const WorkIdResponseType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsACuteNVGPUWorkIdResponse;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub const InitArgs = struct {};

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        _ = args;
        const result = c.mlirCuteNVGPUWorkIdResponseTypeGet(ctx.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `cute_nvgpu.arch.alloc_rmem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_alloc_rmem(ctx: *mlir.Context, input: *const mlir.Value, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.alloc_rmem", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{ptr_type} },
        .location = location,
    });
}

/// `cute_nvgpu.arch.alloc_smem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_alloc_smem(ctx: *mlir.Context, ptr_type: *const mlir.Type, input: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.alloc_smem", .{
        .results = .{ .flat = &.{ptr_type} },
        .attributes = &.{
            .named(ctx, "input", input),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM90.bulk_copy_g2s`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM90_bulk_copy_g2s(ctx: *mlir.Context, gmem_data_addr: *const mlir.Value, dsmem_data_addr: *const mlir.Value, dsmem_bar_addr: *const mlir.Value, multicast_mask: ?*const mlir.Value, cache_policy: ?*const mlir.Value, size: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 5) = .empty;
    operands.appendSliceAssumeCapacity(&.{gmem_data_addr});
    operands.appendSliceAssumeCapacity(&.{dsmem_data_addr});
    operands.appendSliceAssumeCapacity(&.{dsmem_bar_addr});
    if (multicast_mask) |value| operands.appendSliceAssumeCapacity(&.{value});
    if (cache_policy) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM90.bulk_copy_g2s", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = &.{
            .named(ctx, "size", size),
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{
                1,
                1,
                1,
                if (multicast_mask != null) 1 else 0,
                if (cache_policy != null) 1 else 0,
            })),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM90.bulk_copy_s2g`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM90_bulk_copy_s2g(ctx: *mlir.Context, smem_data_addr: *const mlir.Value, gmem_data_addr: *const mlir.Value, byte_mask: ?*const mlir.Value, cache_policy: ?*const mlir.Value, size: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 4) = .empty;
    operands.appendSliceAssumeCapacity(&.{smem_data_addr});
    operands.appendSliceAssumeCapacity(&.{gmem_data_addr});
    if (byte_mask) |value| operands.appendSliceAssumeCapacity(&.{value});
    if (cache_policy) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM90.bulk_copy_s2g", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = &.{
            .named(ctx, "size", size),
            .named(ctx, "operandSegmentSizes", .denseArray(ctx, .i32, &.{
                1,
                1,
                if (byte_mask != null) 1 else 0,
                if (cache_policy != null) 1 else 0,
            })),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM90.bulk_copy_s2s`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM90_bulk_copy_s2s(ctx: *mlir.Context, smem_data_addr: *const mlir.Value, dsmem_data_addr: *const mlir.Value, dsmem_bar_addr: *const mlir.Value, cta_rank: *const mlir.Value, size: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM90.bulk_copy_s2s", .{
        .operands = .{ .flat = &.{ smem_data_addr, dsmem_data_addr, dsmem_bar_addr, cta_rank } },
        .attributes = &.{
            .named(ctx, "size", size),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.copy_s2t`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_copy_s2t(ctx: *mlir.Context, src: *const mlir.Value, dst: *const mlir.Value, dp: *const mlir.Attribute, bits: *const mlir.Attribute, cta: *const mlir.Attribute, broadcast: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.copy_s2t", .{
        .operands = .{ .flat = &.{ src, dst } },
        .attributes = &.{
            .named(ctx, "dp", dp),
            .named(ctx, "bits", bits),
            .named(ctx, "cta", cta),
            .named(ctx, "broadcast", broadcast),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tma_load(ctx: *mlir.Context, src_desc: *const mlir.Value, dsmem_data_addr: *const mlir.Value, dsmem_bar_addr: *const mlir.Value, coord: []const *const mlir.Value, multicast_mask: ?*const mlir.Value, offsets: []const *const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, num_cta: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tma_load", .{
        .operands = .{ .variadic = &.{
            &.{src_desc},
            &.{dsmem_data_addr},
            &.{dsmem_bar_addr},
            coord,
            if (multicast_mask) |value| &.{value} else &.{},
            offsets,
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
            .named(ctx, "num_cta", num_cta),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.tma_reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tma_reduce(ctx: *mlir.Context, dst_desc: *const mlir.Value, src_smem_data_addr: *const mlir.Value, coord: []const *const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, kind: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tma_reduce", .{
        .operands = .{ .variadic = &.{
            &.{dst_desc},
            &.{src_smem_data_addr},
            coord,
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
            .named(ctx, "kind", kind),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.tma_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tma_store(ctx: *mlir.Context, dst_desc: *const mlir.Value, src_smem_data_addr: *const mlir.Value, coord: []const *const mlir.Value, cache_policy: ?*const mlir.Value, mode: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tma_store", .{
        .operands = .{ .variadic = &.{
            &.{dst_desc},
            &.{src_smem_data_addr},
            coord,
            if (cache_policy) |value| &.{value} else &.{},
        } },
        .attributes = &.{
            .named(ctx, "mode", mode),
        },
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.tmem_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tmem_load(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, num_dp: *const mlir.Attribute, num_b: *const mlir.Attribute, num_rep: *const mlir.Attribute, pack_16: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "num_dp", num_dp));
    attributes.appendAssumeCapacity(.named(ctx, "num_b", num_b));
    attributes.appendAssumeCapacity(.named(ctx, "num_rep", num_rep));
    if (pack_16) |value| attributes.appendAssumeCapacity(.named(ctx, "pack_16", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tmem_load", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.copy.SM100.tmem_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_copy_SM100_tmem_store(ctx: *mlir.Context, dst: *const mlir.Value, src: *const mlir.Value, num_dp: *const mlir.Attribute, num_b: *const mlir.Attribute, num_rep: *const mlir.Attribute, expand_16: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 4) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "num_dp", num_dp));
    attributes.appendAssumeCapacity(.named(ctx, "num_b", num_b));
    attributes.appendAssumeCapacity(.named(ctx, "num_rep", num_rep));
    if (expand_16) |value| attributes.appendAssumeCapacity(.named(ctx, "expand_16", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.copy.SM100.tmem_store", .{
        .operands = .{ .flat = &.{ dst, src } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.get_dyn_smem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_get_dyn_smem(ctx: *mlir.Context, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.get_dyn_smem", .{
        .results = .{ .flat = &.{ptr_type} },
        .location = location,
    });
}

/// `cute_nvgpu.arch.get_dyn_smem_size`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_get_dyn_smem_size(ctx: *mlir.Context, size_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.get_dyn_smem_size", .{
        .results = .{ .flat = &.{size_type} },
        .location = location,
    });
}

/// `cute_nvgpu.arch.make_warp_uniform`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_make_warp_uniform(ctx: *mlir.Context, val: *const mlir.Value, res_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.make_warp_uniform", .{
        .operands = .{ .flat = &.{val} },
        .results = .{ .flat = &.{res_type} },
        .location = location,
    });
}

/// `cute_nvgpu.arch.mma.SM120.block_scaled`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_mma_SM120_block_scaled(ctx: *mlir.Context, opA: []const *const mlir.Value, opB: []const *const mlir.Value, opC: []const *const mlir.Value, sfA: *const mlir.Value, sfB: *const mlir.Value, byte_id_a: ?*const mlir.Value, byte_id_b: ?*const mlir.Value, res_types: []const *const mlir.Type, shape_MNK: *const mlir.Attribute, vec_size: *const mlir.Attribute, thread_id_a: ?*const mlir.Attribute, thread_id_b: ?*const mlir.Attribute, a_type: *const mlir.Attribute, b_type: *const mlir.Attribute, sf_type: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 7) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "shape_MNK", shape_MNK));
    attributes.appendAssumeCapacity(.named(ctx, "vec_size", vec_size));
    if (thread_id_a) |value| attributes.appendAssumeCapacity(.named(ctx, "thread_id_a", value));
    if (thread_id_b) |value| attributes.appendAssumeCapacity(.named(ctx, "thread_id_b", value));
    attributes.appendAssumeCapacity(.named(ctx, "a_type", a_type));
    attributes.appendAssumeCapacity(.named(ctx, "b_type", b_type));
    attributes.appendAssumeCapacity(.named(ctx, "sf_type", sf_type));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.mma.SM120.block_scaled", .{
        .operands = .{ .variadic = &.{
            opA,
            opB,
            opC,
            &.{sfA},
            &.{sfB},
            if (byte_id_a) |value| &.{value} else &.{},
            if (byte_id_b) |value| &.{value} else &.{},
        } },
        .results = .{ .variadic = &.{res_types} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.prefetch_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_prefetch_tma_desc(ctx: *mlir.Context, tma_descriptor_ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.prefetch_tma_desc", .{
        .operands = .{ .flat = &.{tma_descriptor_ptr} },
        .location = location,
    });
}

/// `cute_nvgpu.arch.sm100.alloc_tmem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_alloc_tmem(ctx: *mlir.Context, n_cols: *const mlir.Value, dst_ptr: *const mlir.Value, is_two_cta: ?*const mlir.Attribute, exclusive: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (is_two_cta) |value| attributes.appendAssumeCapacity(.named(ctx, "is_two_cta", value));
    if (exclusive) |value| attributes.appendAssumeCapacity(.named(ctx, "exclusive", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.alloc_tmem", .{
        .operands = .{ .flat = &.{ n_cols, dst_ptr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.sm100.dealloc_tmem`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_dealloc_tmem(ctx: *mlir.Context, tmem_ptr: *const mlir.Value, n_cols: *const mlir.Value, is_two_cta: ?*const mlir.Attribute, exclusive: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (is_two_cta) |value| attributes.appendAssumeCapacity(.named(ctx, "is_two_cta", value));
    if (exclusive) |value| attributes.appendAssumeCapacity(.named(ctx, "exclusive", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.dealloc_tmem", .{
        .operands = .{ .flat = &.{ tmem_ptr, n_cols } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_relinquish_tmem_alloc_permit(ctx: *mlir.Context, is_two_cta: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (is_two_cta) |value| attributes.appendAssumeCapacity(.named(ctx, "is_two_cta", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit", .{
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.arch.sm100.retrieve_tmem_ptr`. Result types are explicit; attributes use mlir.Attribute.
pub fn arch_sm100_retrieve_tmem_ptr(ctx: *mlir.Context, smem_ptr: *const mlir.Value, tmem_ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.arch.sm100.retrieve_tmem_ptr", .{
        .operands = .{ .flat = &.{smem_ptr} },
        .results = .{ .flat = &.{tmem_ptr_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.get_coord_tensor`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_get_coord_tensor(ctx: *mlir.Context, atom: *const mlir.Value, shape: *const mlir.Value, offset_b: []const *const mlir.Value, layout_b: []const *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.get_coord_tensor", .{
        .operands = .{ .variadic = &.{
            &.{atom},
            &.{shape},
            offset_b,
            layout_b,
        } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.get_copy_s2t_smem_desc_view`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_get_copy_s2t_smem_desc_view(ctx: *mlir.Context, atom: *const mlir.Value, view: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.get_copy_s2t_smem_desc_view", .{
        .operands = .{ .flat = &.{ atom, view } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.get_value`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_get_value(ctx: *mlir.Context, atom: *const mlir.Value, result_type: *const mlir.Type, field: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.get_value", .{
        .operands = .{ .flat = &.{atom} },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "field", field),
        },
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_exec_tma`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_exec_tma(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_exec_tma", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_2d_gather4_tma_load(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, gmem_coord_layout: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load", .{
        .operands = .{ .flat = &.{ gmem_tensor, gmem_coord_layout, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_im2col_tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_im2col_tma_load(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, lower_corner_whd: *const mlir.Value, upper_corner_whd: *const mlir.Value, lower_padding_whd: *const mlir.Value, upper_padding_whd: *const mlir.Value, stride_whd: *const mlir.Value, lower_srt: *const mlir.Value, stride_srt: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_im2col_tma_load", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map, lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_im2col_tma_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_im2col_tma_store(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_im2col_tma_store", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_tiled_tma_load`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_tiled_tma_load(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_tiled_tma_load", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_tiled_tma_reduce`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_tiled_tma_reduce(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, kind: *const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "kind", kind));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_tiled_tma_reduce", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_non_exec_tiled_tma_store`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_non_exec_tiled_tma_store(ctx: *mlir.Context, gmem_tensor: *const mlir.Value, smem_layout: *const mlir.Value, cta_v_map: *const mlir.Value, non_exec_atom_type: *const mlir.Type, tma_tensor_type: *const mlir.Type, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 1) = .empty;
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_non_exec_tiled_tma_store", .{
        .operands = .{ .flat = &.{ gmem_tensor, smem_layout, cta_v_map } },
        .results = .{ .flat = &.{ non_exec_atom_type, tma_tensor_type } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_s2t_copy`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_s2t_copy(ctx: *mlir.Context, copy_s2t_atom: *const mlir.Value, tmem_memref: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_s2t_copy", .{
        .operands = .{ .flat = &.{ copy_s2t_atom, tmem_memref } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.make_tmem_copy`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_make_tmem_copy(ctx: *mlir.Context, atom: *const mlir.Value, tmem_memref: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.make_tmem_copy", .{
        .operands = .{ .flat = &.{ atom, tmem_memref } },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.atom.set_value`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_set_value(ctx: *mlir.Context, atom: *const mlir.Value, value_: *const mlir.Value, result_type: *const mlir.Type, field: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.set_value", .{
        .operands = .{ .flat = &.{ atom, value_ } },
        .results = .{ .flat = &.{result_type} },
        .attributes = &.{
            .named(ctx, "field", field),
        },
        .location = location,
    });
}

/// `cute_nvgpu.atom.tma_partition`. Result types are explicit; attributes use mlir.Attribute.
pub fn atom_tma_partition(ctx: *mlir.Context, tma_atom: *const mlir.Value, cta_coord: *const mlir.Value, cta_layout: *const mlir.Value, smem_tensor: *const mlir.Value, target_tensors: []const *const mlir.Value, res_smem_tensor_type: *const mlir.Type, res_target_tensors_types: []const *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.atom.tma_partition", .{
        .operands = .{ .variadic = &.{
            &.{tma_atom},
            &.{cta_coord},
            &.{cta_layout},
            &.{smem_tensor},
            target_tensors,
        } },
        .results = .{ .variadic = &.{
            &.{res_smem_tensor_type},
            res_target_tensors_types,
        } },
        .location = location,
    });
}

/// `cute_nvgpu.cast_tma_desc_to_integer`. Result types are explicit; attributes use mlir.Attribute.
pub fn cast_tma_desc_to_integer(ctx: *mlir.Context, src: *const mlir.Value, dst_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.cast_tma_desc_to_integer", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{dst_type} },
        .location = location,
    });
}

/// `cute_nvgpu.copy_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn copy_tma_desc(ctx: *mlir.Context, atom: *const mlir.Value, tma_desc_ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.copy_tma_desc", .{
        .operands = .{ .flat = &.{ atom, tma_desc_ptr } },
        .location = location,
    });
}

/// `cute_nvgpu.get_grid_constant_pointer`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_grid_constant_pointer(ctx: *mlir.Context, input: *const mlir.Value, result_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.get_grid_constant_pointer", .{
        .operands = .{ .flat = &.{input} },
        .results = .{ .flat = &.{result_type} },
        .location = location,
    });
}

/// `cute_nvgpu.get_tma_desc_addr`. Result types are explicit; attributes use mlir.Attribute.
pub fn get_tma_desc_addr(ctx: *mlir.Context, src: *const mlir.Value, ptr_type: *const mlir.Type, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.get_tma_desc_addr", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{ptr_type} },
        .location = location,
    });
}

/// `cute_nvgpu.make_gmma_smem_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_gmma_smem_desc(ctx: *mlir.Context, src: *const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.make_gmma_smem_desc", .{
        .operands = .{ .flat = &.{src} },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

/// `cute_nvgpu.make_tma_desc_im2col_at`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_im2col_at(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, lower_corner_whd: *const mlir.Value, upper_corner_whd: *const mlir.Value, lower_padding_whd: *const mlir.Value, upper_padding_whd: *const mlir.Value, stride_whd: *const mlir.Value, lower_srt: *const mlir.Value, stride_srt: *const mlir.Value, tma_desc_addr: *const mlir.Value, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_im2col_at", .{
        .operands = .{ .flat = &.{ gmem_view, smem_layout, cta_value_tile, lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt, tma_desc_addr } },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.make_tma_desc_im2col`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_im2col(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, lower_corner_whd: *const mlir.Value, upper_corner_whd: *const mlir.Value, lower_padding_whd: *const mlir.Value, upper_padding_whd: *const mlir.Value, stride_whd: *const mlir.Value, lower_srt: *const mlir.Value, stride_srt: *const mlir.Value, result_type: *const mlir.Type, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 2) = .empty;
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_im2col", .{
        .operands = .{ .flat = &.{ gmem_view, smem_layout, cta_value_tile, lower_corner_whd, upper_corner_whd, lower_padding_whd, upper_padding_whd, stride_whd, lower_srt, stride_srt } },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.make_tma_desc_tiled_at`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_tiled_at(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, tma_desc_addr: *const mlir.Value, traversal_stride: ?*const mlir.Value, mode: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 5) = .empty;
    operands.appendSliceAssumeCapacity(&.{gmem_view});
    operands.appendSliceAssumeCapacity(&.{smem_layout});
    operands.appendSliceAssumeCapacity(&.{cta_value_tile});
    operands.appendSliceAssumeCapacity(&.{tma_desc_addr});
    if (traversal_stride) |value| operands.appendSliceAssumeCapacity(&.{value});
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "mode", mode));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_tiled_at", .{
        .operands = .{ .flat = operands.constSlice() },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.make_tma_desc_tiled`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_tma_desc_tiled(ctx: *mlir.Context, gmem_view: *const mlir.Value, smem_layout: *const mlir.Value, cta_value_tile: *const mlir.Value, traversal_stride: ?*const mlir.Value, result_type: *const mlir.Type, mode: *const mlir.Attribute, num_multicast: ?*const mlir.Attribute, tma_format: ?*const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 4) = .empty;
    operands.appendSliceAssumeCapacity(&.{gmem_view});
    operands.appendSliceAssumeCapacity(&.{smem_layout});
    operands.appendSliceAssumeCapacity(&.{cta_value_tile});
    if (traversal_stride) |value| operands.appendSliceAssumeCapacity(&.{value});
    var attributes: stdx.BoundedArray(mlir.NamedAttribute, 3) = .empty;
    attributes.appendAssumeCapacity(.named(ctx, "mode", mode));
    if (num_multicast) |value| attributes.appendAssumeCapacity(.named(ctx, "num_multicast", value));
    if (tma_format) |value| attributes.appendAssumeCapacity(.named(ctx, "tma_format", value));
    return mlir.Operation.make(ctx, "cute_nvgpu.make_tma_desc_tiled", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{result_type} },
        .attributes = attributes.constSlice(),
        .location = location,
    });
}

/// `cute_nvgpu.make_umma_smem_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn make_umma_smem_desc(ctx: *mlir.Context, src: *const mlir.Value, nextSrc: ?*const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands.appendSliceAssumeCapacity(&.{src});
    if (nextSrc) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute_nvgpu.make_umma_smem_desc", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

/// `cute_nvgpu.prefetch_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn prefetch_tma_desc(ctx: *mlir.Context, atom: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.prefetch_tma_desc", .{
        .operands = .{ .flat = &.{atom} },
        .location = location,
    });
}

/// `cute_nvgpu.sm103.make_umma_smem_desc_circular`. Result types are explicit; attributes use mlir.Attribute.
pub fn sm103_make_umma_smem_desc_circular(ctx: *mlir.Context, src: *const mlir.Value, nextSrc: ?*const mlir.Value, res_type: *const mlir.Type, layout: *const mlir.Attribute, major: *const mlir.Attribute, location: *const mlir.Location) *mlir.Operation {
    var operands: stdx.BoundedArray(*const mlir.Value, 2) = .empty;
    operands.appendSliceAssumeCapacity(&.{src});
    if (nextSrc) |value| operands.appendSliceAssumeCapacity(&.{value});
    return mlir.Operation.make(ctx, "cute_nvgpu.sm103.make_umma_smem_desc_circular", .{
        .operands = .{ .flat = operands.constSlice() },
        .results = .{ .flat = &.{res_type} },
        .attributes = &.{
            .named(ctx, "layout", layout),
            .named(ctx, "major", major),
        },
        .location = location,
    });
}

/// `cute_nvgpu.update_tma_desc`. Result types are explicit; attributes use mlir.Attribute.
pub fn update_tma_desc(ctx: *mlir.Context, tma_atom: *const mlir.Value, gmem_tensor: *const mlir.Value, tma_desc_ptr: *const mlir.Value, location: *const mlir.Location) *mlir.Operation {
    return mlir.Operation.make(ctx, "cute_nvgpu.update_tma_desc", .{
        .operands = .{ .flat = &.{ tma_atom, gmem_tensor, tma_desc_ptr } },
        .location = location,
    });
}

pub const operation_names: []const []const u8 = &.{
    "cute_nvgpu.arch.alloc_rmem",
    "cute_nvgpu.arch.alloc_smem",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_g2s",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_s2g",
    "cute_nvgpu.arch.copy.SM90.bulk_copy_s2s",
    "cute_nvgpu.arch.copy.SM100.copy_s2t",
    "cute_nvgpu.arch.copy.SM100.tma_load",
    "cute_nvgpu.arch.copy.SM100.tma_reduce",
    "cute_nvgpu.arch.copy.SM100.tma_store",
    "cute_nvgpu.arch.copy.SM100.tmem_load",
    "cute_nvgpu.arch.copy.SM100.tmem_store",
    "cute_nvgpu.arch.get_dyn_smem",
    "cute_nvgpu.arch.get_dyn_smem_size",
    "cute_nvgpu.arch.make_warp_uniform",
    "cute_nvgpu.arch.mma.SM120.block_scaled",
    "cute_nvgpu.arch.prefetch_tma_desc",
    "cute_nvgpu.arch.sm100.alloc_tmem",
    "cute_nvgpu.arch.sm100.dealloc_tmem",
    "cute_nvgpu.arch.sm100.relinquish_tmem_alloc_permit",
    "cute_nvgpu.arch.sm100.retrieve_tmem_ptr",
    "cute_nvgpu.atom.get_coord_tensor",
    "cute_nvgpu.atom.get_copy_s2t_smem_desc_view",
    "cute_nvgpu.atom.get_value",
    "cute_nvgpu.atom.make_exec_tma",
    "cute_nvgpu.atom.make_non_exec_2d_gather4_tma_load",
    "cute_nvgpu.atom.make_non_exec_im2col_tma_load",
    "cute_nvgpu.atom.make_non_exec_im2col_tma_store",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_load",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_reduce",
    "cute_nvgpu.atom.make_non_exec_tiled_tma_store",
    "cute_nvgpu.atom.make_s2t_copy",
    "cute_nvgpu.atom.make_tmem_copy",
    "cute_nvgpu.atom.set_value",
    "cute_nvgpu.atom.tma_partition",
    "cute_nvgpu.cast_tma_desc_to_integer",
    "cute_nvgpu.copy_tma_desc",
    "cute_nvgpu.get_grid_constant_pointer",
    "cute_nvgpu.get_tma_desc_addr",
    "cute_nvgpu.make_gmma_smem_desc",
    "cute_nvgpu.make_tma_desc_im2col_at",
    "cute_nvgpu.make_tma_desc_im2col",
    "cute_nvgpu.make_tma_desc_tiled_at",
    "cute_nvgpu.make_tma_desc_tiled",
    "cute_nvgpu.make_umma_smem_desc",
    "cute_nvgpu.prefetch_tma_desc",
    "cute_nvgpu.sm103.make_umma_smem_desc_circular",
    "cute_nvgpu.update_tma_desc",
};
