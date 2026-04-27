//! Zig port of `kernels_py/vector_add.py:triton_add_kernel`.

const tri = @import("zml").kernel.triton;
const zml = @import("zml");

pub const VectorAdd = struct {
    pub const Cfg = struct { BLOCK_SIZE: i32 = 1024 };
    pub const Kernel = tri.Kernel(Cfg, .{
        .name = "triton_add_kernel",
        .inputs = &.{ "x_ptr", "y_ptr", "n_elements" },
        .outputs = &.{"output"},
        .run = run,
    });
    fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
        const a = try b.declareArgs(.{
            .x_ptr = .{ .ptr = .f32 },
            .y_ptr = .{ .ptr = .f32 },
            .output_ptr = .{ .ptr = .f32 },
            .n_elements = .{ .scalar_opts = .{ .dtype = .i32, .divisibility = 16 } },
        });

        const pid = b.programId(.x);
        const offsets = pid.mul(cfg.BLOCK_SIZE).add(b.arange(0, cfg.BLOCK_SIZE, .i32));
        const mask = offsets.lt(a.n_elements);

        const x = b.loadOpts(a.x_ptr.addPtr(offsets), .{ .mask = mask });
        const y = b.loadOpts(a.y_ptr.addPtr(offsets), .{ .mask = mask });
        // Bind sum before computing output ptrs — matches Python's source order
        // (`out = x + y` before `tl.store(output_ptr + offsets, out, ...)`).
        const out = x.add(y);
        b.storeOpts(a.output_ptr.addPtr(offsets), out, .{ .mask = mask });
    }
};
