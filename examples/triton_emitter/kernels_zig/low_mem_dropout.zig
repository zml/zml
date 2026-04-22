//! Zig port of `kernels_py/low_mem_dropout.py:_triton_dropout`.

const arith = @import("mlir/dialects").arith;

const tri = @import("zml/triton");
const zml = @import("zml");

pub const LowMemDropout = zml.Kernel(.{
    .name = "_triton_dropout",
    .config = struct { BLOCK_SIZE: i32 = 1024 },
}, struct {
    pub fn run(b: *tri.Builder, cfg: anytype) !void {
        const a = try b.declareArgs(.{
            .x_ptr = .{ .ptr = .f32 },
            .x_keep_ptr = .{ .ptr = .f32 },
            .output_ptr = .{ .ptr = .f32 },
            .n_elements = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
            .p = .{ .scalar = .f32 },
        });

        const pid = b.programId(.x);
        const offsets = pid.mul(cfg.BLOCK_SIZE).add(b.arange(0, cfg.BLOCK_SIZE, .i32));
        const mask = offsets.lt(a.n_elements);

        const x = b.loadOpts(a.x_ptr.addPtr(offsets), .{ .mask = mask });
        const x_keep = b.loadOpts(a.x_keep_ptr.addPtr(offsets), .{ .mask = mask });

        const one_minus_p = b.liftAs(@as(f32, 1.0), .f32).sub(a.p);
        const scaled = x.div(one_minus_p.splatTo(&.{cfg.BLOCK_SIZE}));

        // `tl.where(f32_value, ...)` lowers to `arith.cmpf une` (unordered) +
        // `arith.select`. `Value.ne` defaults to ordered, so use cmpf directly.
        const zero_tensor = b.zeros(&.{cfg.BLOCK_SIZE}, .f32);
        const keep_bool = b.cmpf(arith.CmpFPredicate.une, x_keep, zero_tensor);
        const out = b.where(keep_bool, scaled, zero_tensor);

        b.storeOpts(a.output_ptr.addPtr(offsets), out, .{ .mask = mask });
    }
});
