const std = @import("std");

const arith = @import("mlir/dialects/arith");
const func = @import("mlir/dialects/func");
const math = @import("mlir/dialects/math");
const mlir = @import("mlir");
const scf = @import("mlir/dialects/scf");
const ttir = @import("mlir/dialects/ttir");

const operation_names: []const []const u8 = &.{
    "tt.addptr",
    "tt.assert",
    "tt.atomic_cas",
    "tt.atomic_rmw",
    "tt.bitcast",
    "tt.broadcast",
    "tt.call",
    "tt.clampf",
    "tt.descriptor_gather",
    "tt.descriptor_load",
    "tt.descriptor_reduce",
    "tt.descriptor_scatter",
    "tt.descriptor_store",
    "tt.dot",
    "tt.dot_scaled",
    "tt.elementwise_inline_asm",
    "tt.expand_dims",
    "tt.extern_elementwise",
    "tt.fp_to_fp",
    "tt.func",
    "tt.gather",
    "tt.get_num_programs",
    "tt.get_program_id",
    "tt.histogram",
    "tt.int_to_ptr",
    "tt.join",
    "tt.load",
    "tt.make_range",
    "tt.make_tensor_descriptor",
    "tt.map_elementwise",
    "tt.map_elementwise.return",
    "tt.mulhiui",
    "tt.precise_divf",
    "tt.precise_sqrt",
    "tt.print",
    "tt.ptr_to_int",
    "tt.reduce",
    "tt.reduce.return",
    "tt.reshape",
    "tt.return",
    "tt.scan",
    "tt.scan.return",
    "tt.splat",
    "tt.split",
    "tt.store",
    "tt.trans",
    "tt.unsplat",
};

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(ttir);
    inline for (comptime std.meta.declarations(ttir)) |decl| {
        const T = @field(ttir, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "tt.func round-trip" {
    // Build: module { tt.func public @id(%arg0: !tt.ptr<f32>) { tt.return } }
    // by constructing it with the Zig wrappers, printing to string, and re-parsing.

    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    func.registerDialect(registry);
    arith.registerDialect(registry);
    scf.registerDialect(registry);
    math.registerDialect(registry);
    ttir.registerDialect(registry);

    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    try std.testing.expect(ctx.isRegisteredOperation("tt.func"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.return"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.load"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.store"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.get_program_id"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.splat"));
    try std.testing.expect(ctx.isRegisteredOperation("tt.addptr"));
    try std.testing.expect(ctx.isRegisteredOperation("scf.for"));
    try std.testing.expect(ctx.isRegisteredOperation("arith.constant"));

    const loc: *const mlir.Location = .unknown(ctx);
    const module: *mlir.Module = .init(loc);
    defer module.deinit();

    const f32_ty = mlir.Type.float(ctx, .f32);
    const ptr_ty = ttir.pointerType(f32_ty, 1);

    const entry = mlir.Block.init(&.{ ptr_ty, ptr_ty }, &.{ loc, loc });

    const arg0 = entry.argument(0);
    const arg1 = entry.argument(1);

    // %0 = tt.load %arg0 : !tt.ptr<f32>
    const loaded = ttir.load(ctx, arg0, f32_ty, .none, .normal, false, .{}, loc);
    _ = loaded.appendTo(entry);

    // %1 = arith.constant 1.0 : f32 — built with Operation.make to avoid cross-dialect import
    const one_attr: *const mlir.Attribute = .float(ctx, .f32, 1.0);
    const one = mlir.Operation.make(ctx, "arith.constant", .{
        .results = .{ .flat = &.{f32_ty} },
        .attributes = &.{.named(ctx, "value", one_attr)},
        .location = loc,
    });
    _ = one.appendTo(entry);

    // %2 = arith.addf %0, %1 : f32
    const summed = mlir.Operation.make(ctx, "arith.addf", .{
        .operands = .{ .flat = &.{ loaded.result(0), one.result(0) } },
        .result_type_inference = true,
        .location = loc,
    });
    _ = summed.appendTo(entry);

    // tt.store %arg1, %2
    _ = ttir.store(ctx, arg1, summed.result(0), .none, .normal, .{}, loc).appendTo(entry);

    // tt.return
    _ = ttir.return_(ctx, &.{}, loc).appendTo(entry);

    const f = ttir.func(ctx, .{
        .name = "add_one",
        .block = entry,
        .location = loc,
    });
    _ = f.appendTo(module.body());

    // Verify the operation.
    try std.testing.expect(module.operation().verify());

    // Print IR and re-parse it to confirm it's valid text.
    var al: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer al.deinit();
    try al.writer.print("{f}", .{module.operation()});

    const parsed = try mlir.Module.parse(ctx, al.written());
    defer parsed.deinit();
    try std.testing.expect(parsed.operation().verify());
}

test "divisibility is an i32 attribute of the tt dialect" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();
    var buf: [64]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{mlir.Attribute.dict(ctx, &.{ttir.divisibility(ctx, 16)})});
    try std.testing.expectEqualStrings("{tt.divisibility = 16 : i32}", w.buffered());
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    ttir.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}
