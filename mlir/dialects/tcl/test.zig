const std = @import("std");

const mlir = @import("mlir");
const tcl = @import("mlir/dialects/tcl");

const operation_names: []const []const u8 = &.{
    "tcl.dpe",
    "tcl.graph.all_gather",
    "tcl.graph.arange",
    "tcl.graph.as_dram",
    "tcl.graph.as_logical",
    "tcl.graph.concat",
    "tcl.graph.copy",
    "tcl.graph.for",
    "tcl.graph.full",
    "tcl.graph.gather",
    "tcl.graph.index_read",
    "tcl.graph.index_write",
    "tcl.graph.reduce_max_i32",
    "tcl.graph.reshape",
    "tcl.graph.scatter",
    "tcl.graph.scratchpad",
    "tcl.graph.slice",
    "tcl.graph.sym_expr",
    "tcl.graph.transmute",
    "tcl.graph.vector",
    "tcl.graph.yield",
    "tcl.kernel",
    "tcl.read",
    "tcl.ve",
    "tcl.ve_reduce",
    "tcl.ve_select",
    "tcl.write",
};

test {
    std.testing.refAllDecls(tcl);
    std.testing.refAllDecls(tcl.LogicalType);
}

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    tcl.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

test "every bound operation is registered" {
    const ctx = try testContext();
    defer ctx.deinit();
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}

test "types and attributes verify" {
    const ctx = try testContext();
    defer ctx.deinit();

    const a = try tcl.symbolAttr(ctx, "A");
    const t = try tcl.logicalType(ctx, .float(ctx, .bf16), &.{a});
    try std.testing.expect(t.isA(tcl.LogicalType) != null);
    try std.testing.expectError(error.InvalidMlir, tcl.symbolAttr(ctx, "not an axis"));
    try std.testing.expectError(error.InvalidMlir, tcl.logicalType(ctx, .float(ctx, .bf16), &.{ a, a }));
    try std.testing.expectError(error.InvalidMlir, tcl.veOpcodeAttr(ctx, "unknown"));
    _ = try tcl.veOpcodeAttr(ctx, "to_f16");
}

test "attributes and types print and reparse" {
    const ctx = try testContext();
    defer ctx.deinit();
    const a = try tcl.symbolAttr(ctx, "A");
    const t = try tcl.logicalType(ctx, .float(ctx, .bf16), &.{a});
    var type_text: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer type_text.deinit();
    try type_text.writer.print("{f}", .{t});
    try std.testing.expect((try mlir.Type.parse(ctx, type_text.written())).eql(t));
    var attr_text: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer attr_text.deinit();
    try attr_text.writer.print("{f}", .{a});
    try std.testing.expect((try mlir.Attribute.parse(ctx, attr_text.written())).eql(a));
}
