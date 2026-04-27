//! Declarative Triton kernel wrapper for ZML models.
//!
//! `zml.Kernel(decl, Impl)` returns a kernel type:
//!   - `.emit(allocator, ctx, cfg)` — returns TTIR string (offline tooling).
//!   - `.call(inputs, outputs, opts)` — emits TTIR and inserts a custom_call into forward.
//!
//!     pub const VectorAdd = zml.Kernel(.{
//!         .name = "triton_add_kernel",
//!         .config = struct { BLOCK_SIZE: i32 = 1024 },
//!     }, struct {
//!         pub fn run(b: *tri.Builder, cfg: anytype) !void {
//!             const a = try b.declareArgs(.{
//!                 .x_ptr      = .{ .ptr = .f32 },
//!                 .y_ptr      = .{ .ptr = .f32 },
//!                 .output_ptr = .{ .ptr = .f32 },
//!                 .n_elements = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
//!             });
//!             const offs = b.programId(.x).mul(cfg.BLOCK_SIZE)
//!                 .add(b.arange(0, cfg.BLOCK_SIZE, .i32));
//!             const mask = offs.lt(a.n_elements);
//!             const x = b.loadOpts(a.x_ptr.addPtr(offs), .{ .mask = mask });
//!             const y = b.loadOpts(a.y_ptr.addPtr(offs), .{ .mask = mask });
//!             b.storeOpts(a.output_ptr.addPtr(offs), x.add(y), .{ .mask = mask });
//!         }
//!     });
//!
//!     const out = VectorAdd.call(
//!         .{ x, y, x.shape() },
//!         .{ x.shape() },
//!         .{ .cfg = .{ .BLOCK_SIZE = 1024 }, .grid = .{ ceilDiv(x.dim(0), 1024), 1, 1 },
//!            .num_warps = 4, .num_stages = 2 },
//!     );

const std = @import("std");
const mlir = @import("mlir");

const tri = @import("zml/triton");
const ops = @import("ops.zig");
const Tensor = @import("tensor.zig").Tensor;
const CompilationContext = @import("module.zig").CompilationContext;

/// See module-level doc for usage.
pub fn Kernel(comptime decl: anytype, comptime Impl: type) type {
    const ConfigT = if (@hasField(@TypeOf(decl), "config")) decl.config else struct {};

    return struct {
        pub const name: [:0]const u8 = decl.name;
        pub const Config = ConfigT;

        pub const CallOpts = struct {
            cfg: ConfigT,
            grid: [3]i32,
            num_stages: i32,
            num_warps: i32,
            output_operand_aliases: @FieldType(ops.TritonOps, "output_operand_aliases") = &.{},
            debug: bool = false,
        };

        /// Emit TTIR string. Caller owns it; use `.call(...)` in model code.
        pub fn emit(
            allocator: std.mem.Allocator,
            ctx: *mlir.Context,
            cfg: ConfigT,
        ) ![:0]const u8 {
            var b = try tri.Builder.open(allocator, ctx, name);
            defer b.deinit();
            try Impl.run(&b, cfg);
            return b.finish(&.{});
        }

        /// Emit TTIR and insert a custom_call into the current forward. Returns `[outputs.len]Tensor`.
        pub fn call(inputs: anytype, outputs: anytype, opts: CallOpts) [outputs.len]Tensor {
            const cur = CompilationContext.current();
            const ttir = emit(cur.allocator, cur.mlir_ctx, opts.cfg) catch |err| {
                std.debug.panic("zml.Kernel({s}).call: emit failed: {}", .{ name, err });
            };
            defer cur.allocator.free(ttir);

            return ops.triton(inputs, outputs, .{
                .debug = opts.debug,
                .name = name,
                .ir = ttir,
                .grid = opts.grid,
                .num_stages = opts.num_stages,
                .num_warps = opts.num_warps,
                .output_operand_aliases = opts.output_operand_aliases,
            });
        }
    };
}
