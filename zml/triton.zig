//! High-layer declarative form for Triton kernels in ZML models.
//!
//! `zml.Kernel(decl, Impl)` returns a kernel type with two methods:
//!
//!   - `.emit(allocator, ctx, cfg) ![:0]const u8` — emit the TTIR string for
//!     this kernel under `cfg`. Useful for offline tooling (dump-to-disk
//!     diff harnesses, logging).
//!
//!   - `.call(args) [N]Tensor` — emit TTIR *and* drop a corresponding
//!     `stablehlo.custom_call @ "__gpu$xla.gpu.triton"` into the model's
//!     current `CompilationContext`. The preferred form for production
//!     model code; consumes Tensor inputs and produces Tensor outputs that
//!     interleave naturally with regular ZML ops.
//!
//! Skeleton:
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
//! Inside `forward(...)`:
//!
//!     const out = VectorAdd.call(
//!         .{ x, y, x.shape() },           // tuple of input Tensors
//!         .{ x.shape() },                 // tuple of output Shapes
//!         .{
//!             .cfg = .{ .BLOCK_SIZE = 1024 },
//!             .grid = .{ ceilDiv(x.dim(0), 1024), 1, 1 },
//!             .num_warps = 4,
//!             .num_stages = 2,
//!         },
//!     );
//!
//! `cfg` can be a comptime literal or a runtime value built from
//! `Tensor.dim()` / `Options` fields — same API either way.

const std = @import("std");
const mlir = @import("mlir");

const tri = @import("zml/triton");
const ops = @import("ops.zig");
const Tensor = @import("tensor.zig").Tensor;
const CompilationContext = @import("module.zig").CompilationContext;

/// Declarative kernel form. See module-level doc.
pub fn Kernel(comptime decl: anytype, comptime Impl: type) type {
    const ConfigT = if (@hasField(@TypeOf(decl), "config")) decl.config else struct {};

    return struct {
        pub const name: [:0]const u8 = decl.name;
        pub const Config = ConfigT;

        /// Typed launch options for `.call(...)`. Splitting the launch
        /// parameters into a typed struct (separate from `inputs`/`outputs`,
        /// which must stay `anytype` to carry tuple arity) means call-site
        /// literals like `.grid = .{ @intCast(g), 1, 1 }` and
        /// `.cfg = .{ .NUMEL = @intCast(n) }` get a known result type — the
        /// previous unified `args: anytype` form forced every `@intCast`
        /// inside `.cfg` / `.grid` / `.output_operand_aliases` to be wrapped
        /// in `@as(T, @intCast(x))` or a typed local.
        pub const CallOpts = struct {
            cfg: ConfigT,
            grid: [3]i32,
            num_stages: i32,
            num_warps: i32,
            output_operand_aliases: @FieldType(ops.TritonOps, "output_operand_aliases") = &.{},
            debug: bool = false,
        };

        /// Emit TTIR for this kernel. Caller owns the returned string.
        /// Useful for offline tooling; production code calls `.call(...)`.
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

        /// Emit TTIR *and* insert a `stablehlo.custom_call` op into the
        /// current `CompilationContext` (i.e. inside a model's `forward`).
        ///
        ///   - `inputs`: tuple of `Tensor` — operands fed to the kernel.
        ///   - `outputs`: tuple of `Shape` — declared output shapes.
        ///   - `opts`: typed `CallOpts` carrying `cfg`, `grid`,
        ///     `num_warps`, `num_stages`, optional `output_operand_aliases`.
        ///
        /// Returns `[outputs.len]Tensor`. For single-output kernels, index
        /// with `[0]` at the call site.
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
