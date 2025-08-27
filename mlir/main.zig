const std = @import("std");

const dialects = @import("mlir/dialects");
const mlir = @import("mlir2");

pub fn main() !void {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();

    // const handle = mlir.DialectHandle.fromString("func");

    inline for (.{ "arith", "func" }) |d| {
        registry.registerDialect(d);
    }

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

    // const attr = mlir.integerAttribute(ctx, @as(i32, 5));
    // _ = attr; // autofix

    // const one: i32 = 1;
    // const two: i32 = 2;

    // const dict = mlir.DictionaryAttribute.init(ctx, &.{
    //     .named(ctx, "pouet", mlir.integerAttribute(ctx, one)),
    //     .named(ctx, "pouet2", mlir.integerAttribute(ctx, two)),
    // });
    // _ = dict; // autofix

    const int64 = mlir.integerType(ctx, 64, .signless);

    const block = mlir.Block.init(&.{ int64, int64 }, &.{ .unknown(ctx), .unknown(ctx) });

    block.appendOwnedOperation(dialects.func.return_(ctx, &.{block.argument(0)}, .unknown(ctx)));

    const op = dialects.func.func(ctx, .{
        .name = "pouet",
        .args = &.{ int64, int64 },
        .results = &.{int64},
        .block = block,
        .location = .unknown(ctx),
    });

    // const op = try mlir.Operation.parse(ctx, "func.func private @pouet()", "pouet");

    // // // const mod = mlir.Module.init(mlir.Location.unknown(ctx));

    // const op = mlir.Operation.make(ctx, "arith.addi", .{
    //     .operands = .{ .flat = &.{ block.argument(0), block.argument(1) } },
    //     .results = .{ .variadic = &.{ &.{ int64, int64 }, &.{int64} } },
    // });
    // defer op.deinit();

    std.debug.print(">>>>>> {f}\n", .{op.fmt(.{ .print_generic_op_form = true })});

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
