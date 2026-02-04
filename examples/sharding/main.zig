const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const log = std.log.scoped(.sharding);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Model = struct {
    pos: zml.Tensor, // 2D: {s, d}
    scale: zml.Tensor, // 1D: {d}
    proj_w: zml.Tensor, // 2D: {d, h}
    bias: zml.Tensor, // 1D: {h}

    pub fn init(pos: zml.Tensor, scale: zml.Tensor, proj_w: zml.Tensor, bias: zml.Tensor) Model {
        return .{
            .pos = pos,
            .scale = scale,
            .proj_w = proj_w,
            .bias = bias,
        };
    }

    pub fn forward(self: Model, input: zml.Tensor) zml.Tensor {
        // 3D input
        var x = input.withPartialTags(.{ .b, .s, .d }).convert(.bf16);

        // 2D positional embedding broadcast to 3D
        const pos = self.pos.withTags(.{ .s, .d }).broad(x.shape());
        x = x.add(pos);

        // 1D scale broadcast to 3D
        const scale = self.scale.withTags(.{.d}).broad(x.shape());
        x = x.mul(scale);

        // Reduce 3D -> 2D
        var pooled = x.mean(.s).squeeze(.s);

        // 2D projection (b,d) · (d,h) -> (b,h)
        const w = self.proj_w.withTags(.{ .d, .h });
        var y = pooled.dot(w, .d);

        // 1D bias broadcast to 2D
        const bias = self.bias.withTags(.{.h}).broad(y.shape());
        y = y.add(bias).relu();

        // Expand back to 3D (b,s,h)
        const out_shape = zml.Shape.init(.{
            .b = y.dim(.b),
            .s = x.dim(.s),
            .h = y.dim(.h),
        }, y.dtype());

        var out = y.broad(out_shape);
        return out.convert(.u8);
    }
};

pub fn main() !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer std.debug.assert(debug_allocator.deinit() == .ok);

    const allocator = debug_allocator.allocator();
    // const allocator = std.heap.smp_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    var platform: *zml.Platform = try .auto(allocator, io, .{ .cpu = .{ .device_count = 8 } });
    defer platform.deinit(allocator);

    log.info("{f}\n\n", .{platform.fmtVerbose()});

    const dtype: zml.DataType = .bf16;

    // var physical_mesh: zml.sharding.PhysicalMesh = try .init(allocator, .cpu, .{ .bus = 4 }, .tree);
    // defer physical_mesh.deinit();

    // var physical_mesh: zml.sharding.PhysicalMesh = try .init(allocator, .cuda, .{ .link = 8 }, .tree);
    // defer physical_mesh.deinit();

    var physical_mesh: zml.sharding.PhysicalMesh = try .init(allocator, .tpu, .{ .link_x = 2, .link_y = 2, .link_z = 2 }, .{ .mesh = .torus });
    defer physical_mesh.deinit();

    // var physical_mesh: zml.sharding.PhysicalMesh = try .init(allocator, .neuron, .{ .bus = 2, .link = 12 }, .{ .ring = .closed_ring });
    // defer physical_mesh.deinit();
    log.info("{f}", .{physical_mesh});

    // 2D logical mesh (data)
    const mesh_data: zml.sharding.LogicalMesh = .init("data_mesh", .{
        .batch = .low_bandwidth,
        .context = .balanced,
    });
    log.info("{f}", .{mesh_data});

    // 3D logical mesh (model)
    const mesh_model: zml.sharding.LogicalMesh = .init("model_mesh_3d", .{
        .model = .high_bandwidth,
        .head = .balanced,
        .expert = .low_bandwidth,
    });
    log.info("{f}", .{mesh_model});

    var strategy_data = try zml.sharding.suggestStrategy(allocator, mesh_data, physical_mesh);
    defer strategy_data.deinit(allocator);

    var strategy_model = try zml.sharding.suggestStrategy(allocator, mesh_model, physical_mesh);
    defer strategy_model.deinit(allocator);

    var sharding_data = try zml.sharding.resolveStrategyConstraints(
        allocator,
        mesh_data,
        physical_mesh,
        strategy_data,
    );
    defer sharding_data.deinit();
    log.info("{f}", .{sharding_data});

    var sharding_model = try zml.sharding.resolveStrategyConstraints(
        allocator,
        mesh_model,
        physical_mesh,
        strategy_model,
    );
    defer sharding_model.deinit();
    log.info("{f}", .{sharding_model});

    const input_shape = zml.Shape.init(.{ .b = 8, .s = 512, .d = 4096 }, dtype)
        .withPartitioning(.{ .b = .batch, .s = .context, .d = .replicated });

    const pos_shape = zml.Shape.init(.{ .s = 512, .d = 4096 }, dtype)
        .withPartitioning(.{ .s = .context, .d = .replicated });

    const scale_shape = zml.Shape.init(.{ .d = 4096 }, dtype)
        .withPartitioning(.{ .d = .replicated });

    const proj_w_shape = zml.Shape.init(.{ .d = 4096, .h = 4096 }, dtype)
        .withPartitioning(.{ .d = .model, .h = .head });

    const bias_shape = zml.Shape.init(.{ .h = 4096 }, dtype)
        .withPartitioning(.{ .h = .replicated });

    const input: zml.Tensor = .fromShape(input_shape);
    const pos: zml.Tensor = .fromShape(pos_shape);
    const scale: zml.Tensor = .fromShape(scale_shape);
    const proj_w: zml.Tensor = .fromShape(proj_w_shape);
    const bias: zml.Tensor = .fromShape(bias_shape);

    var assign_input = try zml.sharding.assignTensor(sharding_data, input_shape);
    defer assign_input.deinit();

    var assign_expert = try zml.sharding.assignTensor(sharding_model, bias_shape);
    defer assign_expert.deinit();

    var buf = std.Io.Writer.Allocating.init(allocator);
    defer buf.deinit();

    try assign_input.format(input_shape, &buf.writer);
    std.debug.print("{s}", .{buf.written()});

    buf.clearRetainingCapacity();

    try assign_expert.format(proj_w_shape, &buf.writer);
    std.debug.print("{s}", .{buf.written()});

    const model: Model = .init(pos, scale, proj_w, bias);

    var exe = try platform.compile(allocator, io, model, .forward, .{input}, .{
        .input_sharding = sharding_data,
        .output_sharding = sharding_data,
        .program_sharding = &.{sharding_model},
    });
    defer exe.deinit();

    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();

    var pos_buffer = try createRandomBuffer(allocator, io, platform, pos.shape(), sharding_data, random);
    defer pos_buffer.deinit();

    var scale_buffer = try createRandomBuffer(allocator, io, platform, scale.shape(), sharding_data, random);
    defer scale_buffer.deinit();

    var proj_w_buffer = try createRandomBuffer(allocator, io, platform, proj_w.shape(), sharding_model, random);
    defer proj_w_buffer.deinit();

    var bias_buffer = try createRandomBuffer(allocator, io, platform, bias.shape(), sharding_model, random);
    defer bias_buffer.deinit();

    var input_buffer = try createRandomBuffer(allocator, io, platform, input.shape(), sharding_data, random);
    defer input_buffer.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ pos_buffer, scale_buffer, proj_w_buffer, bias_buffer, input_buffer });

    log.info("Running model...", .{});

    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    _ = try result.await(io);
    defer result.deinit();

    log.info("Exiting", .{});
}

fn createRandomBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, sharding: zml.sharding.Sharding, random: std.Random) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .bool => unreachable,
                .integer => {
                    for (slice.items(ZigType)) |*e| e.* = random.int(ZigType);
                },
                .float => {
                    const value = random.float(f32);
                    for (slice.items(ZigType)) |*e| e.* = switch (ZigType) {
                        f64, f32 => value,
                        f16 => @floatCast(value),
                        inline else => |T| if (@hasDecl(T, "fromF32")) T.fromF32(value) else unreachable,
                    };
                },
                .complex => unreachable,
            }
        },
    }

    return .fromSlice(io, platform, slice, sharding);
}
