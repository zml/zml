const mlir = @import("mlir");

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("ub");
}
