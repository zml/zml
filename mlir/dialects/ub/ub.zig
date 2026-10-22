const mlir = @import("mlir");

pub const dialect_namespace = "ub";

pub fn registerDialect(registry: *mlir.DialectRegistry) void {
    registry.registerDialect("ub");
}
