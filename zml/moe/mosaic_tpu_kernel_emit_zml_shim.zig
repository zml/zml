const std = @import("std");

const mlir = @import("mlir");
const mosaic_tpu_builder = @import("kernels/mosaic_tpu/builder");

pub const kernel = struct {
    pub const mosaic_tpu = struct {
        pub fn newContext() std.mem.Allocator.Error!*mlir.Context {
            return makeKernelContext(&mosaic_tpu_builder.dialects_needed);
        }
    };
};

fn makeKernelContext(comptime dialects_needed: []const []const u8) std.mem.Allocator.Error!*mlir.Context {
    mlir.registerPasses("Transforms");

    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();

    inline for (dialects_needed) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }

    mlir.registerFuncExtensions(registry);

    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();

    return ctx;
}
