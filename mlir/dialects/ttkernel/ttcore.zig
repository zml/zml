//! tt-lang's `ttcore` dialect: tiles, data types and architectures shared by
//! `ttkernel` and `ttl`.

const std = @import("std");

const c = @import("c");
const mlir = @import("mlir");

pub const ops = @import("ttcore_ops.zig");

/// `ttcore::DataType`.
pub const DataType = enum(u32) {
    f32 = 0,
    f16 = 1,
    bf16 = 2,
    bfp_f8 = 3,
    bfp_bf8 = 4,
    bfp_f4 = 5,
    bfp_bf4 = 6,
    bfp_f2 = 7,
    bfp_bf2 = 8,
    u32 = 9,
    u16 = 10,
    u8 = 11,
    si32 = 12,
    i1 = 13,
};

/// `ttcore::Arch`.
pub const Arch = enum(u32) {
    wormhole_b0 = 1,
    blackhole = 2,
    quasar = 3,
};

/// `#ttcore.arch<...>`.
pub fn archAttr(ctx: *mlir.Context, arch: Arch) *const mlir.Attribute {
    return @ptrCast(c.mlirTTCoreArchAttrGet(ctx.ptr(), @intFromEnum(arch)).ptr);
}

/// `!ttcore.tile<height x width, data_type>`.
pub const TileType = opaque {
    const M = mlir.Methods(TileType, c.MlirType);

    pub const isAFn = c.mlirTypeIsATTCoreTile;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirTypeEqual);
    pub const format = M.format(c.mlirTypePrint);
    pub const isA = M.isA;

    pub fn get(ctx: *mlir.Context, height: i64, width: i64, data_type: DataType) *const TileType {
        return @ptrCast(c.mlirTTCoreTileTypeGet(ctx.ptr(), height, width, @intFromEnum(data_type)).ptr);
    }

    pub fn type_(self: *const TileType) *const mlir.Type {
        return @ptrCast(self);
    }

    pub fn dataType(self: *const TileType) DataType {
        return @enumFromInt(c.mlirTTCoreTileTypeGetDataType(self.ptr()));
    }

    /// Bytes of one tile, a circular buffer page.
    pub fn sizeBytes(self: *const TileType) i64 {
        return c.mlirTTCoreTileTypeGetSizeBytes(self.ptr());
    }
};

test {
    std.testing.refAllDecls(@This());
}
