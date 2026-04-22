//! Zig port of `kernels_py/vector_add.py:triton_add_kernel`.

const tri = @import("zml/triton");
const zml = @import("zml");

pub const VectorAdd = zml.Kernel(.{
    .name = "triton_add_kernel",
    .config = struct { BLOCK_SIZE: i32 = 1024 },
}, struct {
    pub fn run(b: *tri.Builder, cfg: anytype) !void {
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
});
