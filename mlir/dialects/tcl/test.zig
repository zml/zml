const std = @import("std");

const mlir = @import("mlir");
const func = @import("mlir/dialects/func");
const tcl = @import("mlir/dialects/tcl");

const operation_names: []const []const u8 = &.{
    "tcl.dpe",
    "tcl.filter_compaction",
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
    inline for (.{ func, tcl }) |dialect| dialect.registerDialect(registry);
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

/// `tcl.kernel(input) { read; filter_compaction along `axis`; write }`.
fn compactionKernel(ctx: *mlir.Context, entry: *mlir.Block, axis: *const mlir.Attribute) !*mlir.Operation {
    const loc: *const mlir.Location = .unknown(ctx);
    const n = try tcl.symbolAttr(ctx, "N");
    const k = try tcl.symbolAttr(ctx, "K");
    const input = try tcl.logicalType(ctx, .int(ctx, .i32), &.{n});
    const output = try tcl.logicalType(ctx, .int(ctx, .i32), &.{k});
    const body = mlir.Block.init(&.{}, &.{});
    const x = tcl.read(ctx, &.{entry.argument(0)}, input, .{}, loc).appendTo(body);
    const compacted = tcl.filter_compaction(ctx, x.result(0), x.result(0), output, .ge, .int(ctx, .i32, 0), &.{axis}, k, loc).appendTo(body);
    _ = tcl.write(ctx, compacted.result(0), loc).appendTo(body);
    return tcl.kernel(ctx, &.{entry.argument(0)}, output, .filter_compaction, body, .{}, loc).appendTo(entry);
}

test "filter_compaction verifies inside a kernel" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc: *const mlir.Location = .unknown(ctx);
    const n = try tcl.symbolAttr(ctx, "N");
    const entry = mlir.Block.init(&.{try tcl.logicalType(ctx, .int(ctx, .i32), &.{n})}, &.{loc});
    // The kernel verifier walks to its inputs' regions.
    const function = func.func(ctx, .{ .name = "compact", .block = entry, .results = &.{}, .location = loc, .verify = false });
    defer function.deinit();
    try std.testing.expect((try compactionKernel(ctx, entry, n)).verify());
    try std.testing.expect(!(try compactionKernel(ctx, entry, try tcl.symbolAttr(ctx, "Missing"))).verify());
}

test "function attributes carry their tcl names" {
    const ctx = try testContext();
    defer ctx.deinit();
    const loc: *const mlir.Location = .unknown(ctx);
    const entry = mlir.Block.init(&.{}, &.{});
    _ = func.returns(ctx, &.{}, loc).appendTo(entry);
    const function = func.func(ctx, .{ .name = "f", .block = entry, .location = loc });
    defer function.deinit();
    const config = try tcl.configAttr(ctx, &.{.named(ctx, "lowering_mode", .string(ctx, "Heuristic"))});
    tcl.annotateFunction(ctx, function, &.{.named(ctx, "A", .int(ctx, .i64, 4))}, .{
        .updated_inputs = &.{-1},
        .compiler_config = config,
        .auto_config = config,
        .fusible = true,
    });
    try std.testing.expect(function.verify());
    var text: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer text.deinit();
    try text.writer.print("{f}", .{function});
    try std.testing.expect(std.mem.indexOf(u8, text.written(), "attributes {tcl.auto_config = #tcl.config<{lowering_mode = \"Heuristic\"}>, tcl.axes = {A = 4 : i64}, tcl.compiler_config = #tcl.config<{lowering_mode = \"Heuristic\"}>, tcl.fusible, tcl.updated_inputs = array<i64: -1>}") != null);
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
