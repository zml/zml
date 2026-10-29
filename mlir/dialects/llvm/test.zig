const std = @import("std");

const llvm = @import("mlir/dialects/llvm");
const mlir = @import("mlir");

const operation_names: []const []const u8 = &.{
    "llvm.alloca",
    "llvm.atomicrmw",
    "llvm.bitcast",
    "llvm.br",
    "llvm.call_intrinsic",
    "llvm.extractelement",
    "llvm.extractvalue",
    "llvm.fmul",
    "llvm.freeze",
    "llvm.getelementptr",
    "llvm.inline_asm",
    "llvm.insertelement",
    "llvm.intr.assume",
    "llvm.intr.lifetime.end",
    "llvm.intr.lifetime.start",
    "llvm.inttoptr",
    "llvm.load",
    "llvm.mlir.poison",
    "llvm.mlir.undef",
    "llvm.shufflevector",
    "llvm.store",
    "llvm.unreachable",
};

test {
    @setEvalBranchQuota(100_000);
    std.testing.refAllDecls(llvm);
    inline for (comptime std.meta.declarations(llvm)) |decl| {
        const T = @field(llvm, decl);
        if (@TypeOf(T) == type and @typeInfo(T) == .@"opaque") std.testing.refAllDecls(T);
    }
}

test "every bound operation is registered" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    llvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    for (operation_names) |name| try std.testing.expect(ctx.isRegisteredOperation(name));
    try std.testing.expect(!ctx.allowUnregisteredDialects());
}

test "literal struct types build from their fields" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    llvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    const struct_type = llvm.structType(ctx, &.{ .int(ctx, .i32), .float(ctx, .f32) });
    try std.testing.expect(struct_type.eql(try mlir.Type.parse(ctx, "!llvm.struct<(i32, f32)>")));
}

test "identified struct types build from their name and fields" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    llvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    const fields: []const *const mlir.Type = &.{ .int(ctx, .i32), .float(ctx, .f32) };
    const struct_type = llvm.identifiedStructType(ctx, "struct.args", fields).?;
    try std.testing.expect(struct_type.eql(try mlir.Type.parse(ctx, "!llvm.struct<\"struct.args\", (i32, f32)>")));
    try std.testing.expect(llvm.identifiedStructType(ctx, "struct.args", fields).?.eql(struct_type));
    try std.testing.expect(llvm.identifiedStructType(ctx, "struct.args", fields[0..1]) == null);
}

test "getelementptr and alloca build verified operations" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    llvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    const loc: *const mlir.Location = .unknown(ctx);
    const ptr = llvm.pointerType(ctx, llvm.NVPTXAddressSpace.global);
    const block = mlir.Block.init(&.{ ptr, .int(ctx, .i64) }, &.{ loc, loc });
    defer block.deinit();
    const array = llvm.arrayType(.float(ctx, .f32), 8);
    const gep = llvm.getelementptr(ctx, block.argument(0), &.{ .{ .constant = 0 }, .{ .value = block.argument(1) } }, array, loc).appendTo(block);
    try std.testing.expect(gep.verify());
    try std.testing.expect(gep.result(0).type_().eql(ptr));
    const alloca = llvm.alloca(ctx, block.argument(1), llvm.pointerType(ctx, llvm.NVPTXAddressSpace.generic), array, .{ .alignment = 4 }, loc).appendTo(block);
    try std.testing.expect(alloca.verify());
}

test "extractvalue, atomicrmw and intr.assume build verified operations" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    llvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    const loc: *const mlir.Location = .unknown(ctx);
    const pair = llvm.structType(ctx, &.{ .int(ctx, .i32), .float(ctx, .f32) });
    const block = mlir.Block.init(&.{ pair, llvm.pointerType(ctx, llvm.NVPTXAddressSpace.global), .int(ctx, .i1) }, &.{ loc, loc, loc });
    defer block.deinit();
    const first = llvm.extractvalue(ctx, block.argument(0), .int(ctx, .i32), &.{0}, loc).appendTo(block);
    try std.testing.expect(first.verify());
    const previous = llvm.atomicrmw(ctx, block.argument(1), first.result(0), .add, .monotonic, .{ .syncscope = "agent", .alignment = 4 }, loc).appendTo(block);
    try std.testing.expect(previous.verify());
    try std.testing.expect(previous.result(0).type_().eql(.int(ctx, .i32)));
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer writer.deinit();
    try writer.writer.print("{f}", .{previous});
    try std.testing.expect(std.mem.indexOf(u8, writer.written(), "llvm.atomicrmw add ") != null);
    try std.testing.expect(std.mem.indexOf(u8, writer.written(), " monotonic ") != null);
    try std.testing.expect(llvm.intr_assume(ctx, block.argument(2), loc).appendTo(block).verify());
}

test "vector, freeze and lifetime operations build verified operations" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    llvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    const loc: *const mlir.Location = .unknown(ctx);
    const i64x2: *const mlir.Type = .vector(&.{2}, .int(ctx, .i64));
    const block = mlir.Block.init(&.{ i64x2, .int(ctx, .i64), .int(ctx, .i32), llvm.pointerType(ctx, llvm.NVPTXAddressSpace.generic) }, &.{ loc, loc, loc, loc });
    defer block.deinit();
    const v = block.argument(0);
    const inserted = llvm.insertelement(ctx, v, block.argument(1), block.argument(2), loc).appendTo(block);
    try std.testing.expect(inserted.verify());
    try std.testing.expect(inserted.result(0).type_().eql(i64x2));
    try std.testing.expect(llvm.extractelement(ctx, v, block.argument(2), .int(ctx, .i64), loc).appendTo(block).verify());
    const poison = llvm.mlir_poison(ctx, i64x2, loc).appendTo(block);
    try std.testing.expect(poison.verify());
    try std.testing.expect(llvm.mlir_undef(ctx, i64x2, loc).appendTo(block).verify());
    try std.testing.expect(llvm.shufflevector(ctx, v, poison.result(0), i64x2, &.{ 1, 0 }, loc).appendTo(block).verify());
    try std.testing.expect(llvm.freeze(ctx, v, loc).appendTo(block).verify());
    try std.testing.expect(llvm.intr_lifetime_start(ctx, block.argument(3), loc).appendTo(block).verify());
    try std.testing.expect(llvm.intr_lifetime_end(ctx, block.argument(3), loc).appendTo(block).verify());
}

test "br carries its loop annotation" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    llvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    const loc: *const mlir.Location = .unknown(ctx);
    const dest = mlir.Block.init(&.{}, &.{});
    defer dest.deinit();
    const annotation = llvm.LoopAnnotationAttr.get(ctx, .{ .mustProgress = true });
    const op = llvm.br(ctx, dest, &.{}, .{ .loop_annotation = annotation }, loc);
    defer op.deinit();
    try std.testing.expect(op.attributeByName("loop_annotation").?.eql(annotation.attribute()));
    const end = llvm.unreachable_(ctx, loc);
    defer end.deinit();
}

test "argument and function attributes are named by the LLVM dialect" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    llvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try w.print("{f}", .{mlir.Attribute.dict(ctx, &.{
        llvm.noundef(ctx),            llvm.zeroext(ctx),             llvm.signext(ctx),
        llvm.alignment(ctx, 16),      llvm.dereferenceable(ctx, 64), llvm.elementtype(ctx, .float(ctx, .f32)),
        llvm.linkage(ctx, .internal), llvm.inlineHint(ctx),
    })});
    try std.testing.expectEqualStrings("{llvm.align = 16 : i64, llvm.dereferenceable = 64 : i64, llvm.elementtype = f32, llvm.inline_hint, llvm.linkage = #llvm.linkage<internal>, llvm.noundef, llvm.signext, llvm.zeroext}", w.buffered());
}

test "emit_c_interface is a unit attribute of the LLVM dialect" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer writer.deinit();
    try writer.writer.print("{f}", .{mlir.Attribute.dict(ctx, &.{llvm.emitCInterface(ctx)})});
    try std.testing.expectEqualStrings("{llvm.emit_c_interface}", writer.written());
}

test "loop annotations are built through the C API" {
    const ctx = try mlir.Context.init(.{});
    defer ctx.deinit();
    const unroll = llvm.LoopUnrollAttr.get(ctx, .{ .full = true });
    try std.testing.expect(unroll.attribute().isA(llvm.LoopUnrollAttr) != null);
    const annotation = llvm.LoopAnnotationAttr.get(ctx, .{ .unroll = unroll, .mustProgress = true });
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer writer.deinit();
    try writer.writer.print("{f}", .{annotation});
    try std.testing.expectEqualStrings("#llvm.loop_annotation<unroll = <full = true>, mustProgress = true>", writer.written());
    try std.testing.expect((try mlir.Attribute.parse(ctx, writer.written())).eql(annotation.attribute()));
}

test "linkage attributes are built through the C API" {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    llvm.registerDialect(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    var writer: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer writer.deinit();
    try writer.writer.print("{f}", .{llvm.linkageAttr(ctx, .weak_odr)});
    try std.testing.expectEqualStrings("#llvm.linkage<weak_odr>", writer.written());
}
