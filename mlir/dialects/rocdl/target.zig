const c = @import("c");
const mlir = @import("mlir");

/// GPU compilation target, `#rocdl.target`.
pub const TargetAttr = opaque {
    const Self = @This();
    const M = mlir.Methods(Self, c.MlirAttribute);
    pub const isAFn = c.mlirAttributeIsAROCDLTarget;
    pub const ptr = M.ptr;
    pub const eql = M.eql(c.mlirAttributeEqual);
    pub const format = M.format(c.mlirAttributePrint);
    pub const isA = M.isA;

    pub const InitArgs = struct {
        optLevel: i32 = 2,
        triple: []const u8 = "amdgcn-amd-amdhsa",
        chip: []const u8 = "gfx900",
        features: []const u8 = "",
        abi: []const u8 = "600",
        flags: ?*const mlir.Attribute = null,
        link: ?*const mlir.Attribute = null,
    };

    pub fn get(ctx: *mlir.Context, args: InitArgs) mlir.Error!*const Self {
        return @ptrCast(c.mlirROCDLTargetAttrGet(
            ctx.ptr(),
            args.optLevel,
            mlir.stringRef(args.triple),
            mlir.stringRef(args.chip),
            mlir.stringRef(args.features),
            mlir.stringRef(args.abi),
            .{ .ptr = if (args.flags) |a| @constCast(a.ptr().ptr) else null },
            .{ .ptr = if (args.link) |a| @constCast(a.ptr().ptr) else null },
        ).ptr orelse return error.InvalidMlir);
    }

    pub fn attribute(self: *const Self) *const mlir.Attribute {
        return @ptrCast(self);
    }

    pub fn optLevel(self: *const Self) i32 {
        return c.mlirROCDLTargetAttrGetOptLevel(self.ptr());
    }

    pub fn triple(self: *const Self) []const u8 {
        return string(c.mlirROCDLTargetAttrGetTriple(self.ptr()));
    }

    pub fn chip(self: *const Self) []const u8 {
        return string(c.mlirROCDLTargetAttrGetChip(self.ptr()));
    }

    pub fn features(self: *const Self) []const u8 {
        return string(c.mlirROCDLTargetAttrGetFeatures(self.ptr()));
    }

    pub fn abi(self: *const Self) []const u8 {
        return string(c.mlirROCDLTargetAttrGetAbi(self.ptr()));
    }

    pub fn flags(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirROCDLTargetAttrGetFlags(self.ptr()).ptr);
    }

    pub fn link(self: *const Self) ?*const mlir.Attribute {
        return @ptrCast(c.mlirROCDLTargetAttrGetLink(self.ptr()).ptr);
    }

    pub fn hasFlag(self: *const Self, flag: []const u8) bool {
        return c.mlirROCDLTargetAttrHasFlag(self.ptr(), mlir.stringRef(flag));
    }

    pub fn hasWave64(self: *const Self) bool {
        return c.mlirROCDLTargetAttrHasWave64(self.ptr());
    }

    pub fn hasFastMath(self: *const Self) bool {
        return c.mlirROCDLTargetAttrHasFastMath(self.ptr());
    }

    pub fn hasDaz(self: *const Self) bool {
        return c.mlirROCDLTargetAttrHasDaz(self.ptr());
    }

    pub fn hasFiniteOnly(self: *const Self) bool {
        return c.mlirROCDLTargetAttrHasFiniteOnly(self.ptr());
    }

    pub fn hasUnsafeMath(self: *const Self) bool {
        return c.mlirROCDLTargetAttrHasUnsafeMath(self.ptr());
    }

    pub fn hasCorrectSqrt(self: *const Self) bool {
        return c.mlirROCDLTargetAttrHasCorrectSqrt(self.ptr());
    }
};

fn string(ref: c.MlirStringRef) []const u8 {
    return if (ref.length == 0) "" else mlir.string(ref);
}
