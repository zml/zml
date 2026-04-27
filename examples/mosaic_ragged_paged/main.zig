//! Demo driver for the ragged_paged_attention kernel — emits the textual
//! MLIR for the same demo shapes as `kernel.py`. The kernel itself lives
//! in `platforms/tpu/ragged_paged.zig` so the production attention path
//! and this offline tool share one source of truth.
//!
//! Usage:
//!     bazel run //examples/mosaic_ragged_paged:ragged_paged_attention -- [HEAD_DIM] [KV_DTYPE]
//!
//! HEAD_DIM defaults to 128. KV_DTYPE defaults to `bf16`. Supported dtypes
//! mirror Pallas's `get_dtype_packing` table (1, 2, 4):
//!   - packing 1: f32, i32
//!   - packing 2: bf16, f16, i16
//!   - packing 4: i8, f8e4m3fn, f8e5m2
//!
//! Compare with the canonicalized Pallas IR via the sibling tool:
//!     bazel run //examples/mosaic_ragged_paged:canonicalize -- /tmp/rpa_compare/pallas_mosaic.mlir

const std = @import("std");

const mlir = @import("mlir");
const ragged_paged = @import("platforms/tpu/ragged_paged");

const DType = @FieldType(ragged_paged.Cfg, "kv_dtype");

fn parseDType(s: []const u8) !DType {
    inline for (std.meta.fields(DType)) |f| {
        if (std.mem.eql(u8, f.name, s)) return @field(DType, f.name);
    }
    return error.UnknownDType;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var head_dim: i64 = 128;
    var kv_dtype: DType = .bf16;
    {
        var it = init.minimal.args.iterate();
        _ = it.next(); // exe path
        if (it.next()) |a| head_dim = try std.fmt.parseInt(i64, a, 10);
        if (it.next()) |a| kv_dtype = try parseDType(a);
    }

    const cfg: ragged_paged.Cfg = .{
        .num_q_tokens = 8,
        .num_q_heads = 8,
        .num_kv_heads = 2,
        .head_dim = head_dim,
        .total_num_pages = 8,
        .page_size = 16,
        .max_num_seqs = 2,
        .pages_per_seq = 2,
        .num_kv_pages_per_block = 1,
        .num_queries_per_block = 8,
        .sm_scale = 0.0883883461,
        .kv_dtype = kv_dtype,
    };

    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (.{ "func", "arith", "scf", "math", "memref", "vector", "tpu" }) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }
    mlir.registerPasses("Transforms");

    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    const ir = try ragged_paged.buildIr(allocator, ctx, cfg);
    defer allocator.free(ir);

    var stdout_buf: [128 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    defer stdout.interface.flush() catch {};
    try stdout.interface.print("{s}\n", .{ir});
}
