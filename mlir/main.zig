const std = @import("std");

const dialects = @import("mlir/dialects");
const mlir = @import("mlir2");

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();

    // const handle = mlir.DialectHandle.fromString("func");

    inline for (.{ "memref", "func", "stablehlo" }) |d| {
        registry.registerDialect(d);
    }

    // mlir.registerPasses("AllStablehlo");

    const ctx = try mlir.Context.init(.{
        .registry = registry,
        .threading = true,
    });
    defer ctx.deinit();

    std.debug.print(">>> {f}\n", .{ctx});
    ctx.loadAllAvailableDialects();
    std.debug.print(">>> {f}\n", .{ctx});

    const unk = mlir.Location.unknown(ctx);
    _ = unk; // autofix

    var serialized: std.Io.Writer.Allocating = .init(allocator);
    defer serialized.deinit();

    // const stablehlo_version = blk: {
    //     if (dialects.stablehlo.currentVersion()) |requested_version| {
    //         break :blk dialects.stablehlo.stablehloGetSmallerVersion(requested_version, dialects.stablehlo.getCurrentVersion());
    //     }
    //     break :blk dialects.stablehlo.minimumVersion();
    // };

    // const attr = mlir.integerAttribute(ctx, @as(i32, 5));
    // _ = attr; // autofix

    // const one: i32 = 1;
    // const two: i32 = 2;

    // const dict = mlir.DictionaryAttribute.init(ctx, &.{
    //     .named(ctx, "pouet", mlir.integerAttribute(ctx, one)),
    //     .named(ctx, "pouet2", mlir.integerAttribute(ctx, two)),
    // });
    // _ = dict; // autofix

    const module = mlir.Module.init(.unknown(ctx));
    defer module.deinit();

    const tensor_i32 = mlir.rankedTensorType(&.{ 4096, 4096 }, mlir.integerType(ctx, .i32));

    _ = dialects.func.func(ctx, .{
        .name = "pouet",
        .visibility = .private,
        .block = blk: {
            const body = mlir.Block.init(&.{ tensor_i32, tensor_i32 }, &.{ .unknown(ctx), .unknown(ctx) });
            const add_op = dialects.stablehlo.add(ctx, body.argument(0), body.argument(1), .unknown(ctx)).appendTo(body);
            const mul_op = dialects.stablehlo.multiply(ctx, add_op.result(0), body.argument(1), .unknown(ctx)).appendTo(body);
            _ = dialects.func.return_(ctx, mul_op.result(0), .unknown(ctx)).appendTo(body);
            break :blk body;
        },
        .location = .unknown(ctx),
    }).appendTo(module.body());

    std.debug.print(">>>>>>\n{f}\n", .{module.operation().fmt(.{ .print_generic_op_form = false })});
    std.debug.print(">>>>>>\n{f}\n", .{module.operation().fmt(.{ .print_generic_op_form = true })});

    // dialects.stablehlo.serializePortableArtifact2(module, dialects.stablehlo.minimumVersion(), &serialized.writer) catch |err| {
    //     std.debug.print("failed to serialize to portable artifact: {}\n", .{err});
    //     return err;
    // };

    // const body = mod.body();
    // _ = body.addArgument(mlir.integerType(ctx, 64, .signless), .unknown(ctx));
    // _ = body.addArgument(mlir.integerType(ctx, 64, .signless), .unknown(ctx));
    // _ = body.addArgument(mlir.integerType(ctx, 64, .signless), .unknown(ctx));

    // body.appendOwnedOperation(op);

    // op.deinit();
    // body.detach();
    // body.deinit();

    // std.debug.print(">>>>>> {any}\n", .{op.block().?.parentOperation()});

    // std.debug.print(">>>>>> {any}\n", .{op.block().?.parentOperation()});

    // std.debug.print(">>> {f}\n", .{body});
    // const attr = mlir.integerAttribute(ctx, @as(i32, 5));
    // const attr2: *const mlir.Attribute = @ptrCast(attr);

    // std.debug.print("coucou 2 {any} {f}\n", .{
    //     ctx.isRegisteredOperation("func.func"),
    //     attr2,
    // });

    // var buffer: [256]u8 = undefined;
    // const vv = "coucou\n";
    // var writer = std.fs.File.stdout().writer(&buffer);
    // try writer.interface.writeAll(vv);
    // try writer.interface.flush();
}
