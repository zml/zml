const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

/// `#fly_rocdl.buffer_desc`, an address space for buffer-descriptor pointers.
pub const BufferDescAddressAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.attribute().format(writer);
    }

    pub fn get(ctx: *mlir.Context) mlir.Error!*const Self {
        return @ptrCast(c.mlirFlyROCDLBufferDescAddressAttrGet(ctx.ptr()).ptr orelse return error.InvalidMlir);
    }
};

/// `!fly_rocdl.cdna3.buffer_copy`.
pub const CopyOpCDNA3BufferCopyType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyROCDLCopyOpCDNA3BufferCopy;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        bitSize: i32,
        /// 0 = cached, 2 = non-temporal.
        cacheModifier: i32 = 0,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyROCDLCopyOpCDNA3BufferCopyTypeGet(ctx.ptr(), args.bitSize, args.cacheModifier);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!fly_rocdl.cdna3.buffer_copy_lds`.
pub const CopyOpCDNA3BufferCopyLDSType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyROCDLCopyOpCDNA3BufferCopyLDS;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        bitSize: i32,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyROCDLCopyOpCDNA3BufferCopyLDSTypeGet(ctx.ptr(), args.bitSize);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!fly_rocdl.cdna3.mfma`.
pub const MmaOpCDNA3MFMAType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyROCDLMmaOpCDNA3MFMA;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = struct {
        m: i32,
        n: i32,
        k: i32,
        elemTyA: *const mlir.Type,
        elemTyB: *const mlir.Type,
        elemTyAcc: *const mlir.Type,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyROCDLMmaOpCDNA3MFMATypeGet(ctx.ptr(), args.m, args.n, args.k, args.elemTyA.ptr(), args.elemTyB.ptr(), args.elemTyAcc.ptr());
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!fly_rocdl.gfx11.wmma`.
pub const MmaOpGFX11WMMAType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyROCDLMmaOpGFX11WMMA;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = WmmaInitArgs;

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyROCDLMmaOpGFX11WMMATypeGet(ctx.ptr(), args.m, args.n, args.k, args.elemTyA.ptr(), args.elemTyB.ptr(), args.elemTyAcc.ptr(), args.signA, args.signB, args.clamp);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `!fly_rocdl.gfx120x.wmma`.
pub const MmaOpGFX120XWMMAType = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirType);
    pub const isAFn = c.mlirTypeIsAFlyROCDLMmaOpGFX120XWMMA;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const isA = M.isA;
    pub fn type_(self: *const Self) *const mlir.Type {
        return @ptrCast(self);
    }
    pub fn format(self: *const Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.type_().format(writer);
    }

    pub const InitArgs = WmmaInitArgs;

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        const result = c.mlirFlyROCDLMmaOpGFX120XWMMATypeGet(ctx.ptr(), args.m, args.n, args.k, args.elemTyA.ptr(), args.elemTyB.ptr(), args.elemTyAcc.ptr(), args.signA, args.signB, args.clamp);
        return @ptrCast(result.ptr orelse return error.InvalidMlir);
    }
};

/// `signA`/`signB`/`clamp` drive the integer WMMA intrinsics and must stay
/// false on the float paths.
pub const WmmaInitArgs = struct {
    m: i32,
    n: i32,
    k: i32,
    elemTyA: *const mlir.Type,
    elemTyB: *const mlir.Type,
    elemTyAcc: *const mlir.Type,
    signA: bool = false,
    signB: bool = false,
    clamp: bool = false,
};
