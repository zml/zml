//! Build a tiny Mosaic TPU kernel with `kernels/mosaic_tpu` and print the
//! generated MLIR to stdout. Standalone: no platform / no execution.
//!
//! What the kernel does
//! --------------------
//!   func.func @add_one(%x : memref<128x128xbf16, #tpu.memory_space<vmem>>,
//!                      %y : memref<128x128xbf16, #tpu.memory_space<vmem>>) {
//!     %tile = vector_load %x[0, 0] : memref<...> -> vector<128x128xbf16>
//!     %one  = vector.broadcast 1.0 : vector<128x128xbf16>
//!     %sum  = arith.addf %tile, %one : vector<128x128xbf16>
//!     vector_store %sum, %y[0, 0] : vector<128x128xbf16> -> memref<...>
//!     func.return
//!   }
//!
//! Note: this example sits behind `//kernels/mosaic_tpu` which is `manual`-tagged
//! (the `tpu` dialect comes from `@jax`, not yet in `MODULE.bazel`).
//! Build / run it once `@jax` is wired up:
//!   bazel run //examples/mosaic_print_ir:mosaic_print_ir

const std = @import("std");

const mlir = @import("mlir");
const mtt = @import("kernels/mosaic_tpu/builder");

const BLOCK_M: i64 = 128;
const BLOCK_N: i64 = 128;

fn buildAddOneIr(allocator: std.mem.Allocator, ctx: *mlir.Context) ![:0]const u8 {
    var spec = try mtt.Builder.build(allocator, ctx, "add_one_kernel", .{
        .x_ref = .{ .ref = .{ .shape = &.{ BLOCK_M, BLOCK_N }, .dtype = .bf16, .memory_space = .vmem } },
        .y_ref = .{ .ref = .{ .shape = &.{ BLOCK_M, BLOCK_N }, .dtype = .bf16, .memory_space = .vmem } },
    }, &.{});
    defer spec.deinit();
    const k = &spec.kernel;
    const a = spec.args;

    // Hoist the splat above the load so the emitted op order matches
    // what Pallas/Mosaic produces (`%cst` before `%c0`). Each builder
    // call appends one op, so source order *is* IR order — there's no
    // reordering inside the DSL.
    const one = k.splat(@as(f32, 1.0), &.{ BLOCK_M, BLOCK_N }, .bf16);
    const tile = k.refLoad(a.x_ref);
    const sum = tile.add(one);
    k.refStore(a.y_ref, sum);

    return k.finish(&.{});
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    // Stand up an MLIR context with the dialects the Mosaic DSL emits.
    // `tpu` is the Mosaic TPU dialect; the rest are upstream MLIR.
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (.{ "func", "arith", "scf", "math", "memref", "vector", "tpu" }) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }

    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    const ir = try buildAddOneIr(allocator, ctx);
    defer allocator.free(ir);

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    defer stdout.interface.flush() catch {};
    try stdout.interface.print("{s}\n", .{ir});
}
