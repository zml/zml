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

const AddModel = struct {
    pub fn init() AddModel {
        return .{};
    }

    pub fn forward(_: AddModel, a: zml.Tensor, b: zml.Tensor, c: zml.Tensor, d: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        const sum_ac = a.add(c);
        const sum_bd = b.add(d);
        return .{ sum_ac, sum_bd };
    }
};

fn runComplexModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding_data: zml.sharding.Sharding,
    sharding_model: zml.sharding.Sharding,
) !void {
    const dtype: zml.DataType = .bf16;

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

fn runAdditionExample(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding_data: zml.sharding.Sharding,
    sharding_model: zml.sharding.Sharding,
) !void {
    const add_shape_data = zml.Shape.init(.{ .x = 16, .y = 24 }, .f32)
        .withPartitioning(.{ .x = .batch, .y = .context });

    const add_shape_model = zml.Shape.init(.{ .x = 8, .y = 16, .z = 8 }, .f32)
        .withPartitioning(.{ .x = .model, .y = .replicated, .z = .expert });

    const a: zml.Tensor = .fromShape(add_shape_data);
    const b: zml.Tensor = .fromShape(add_shape_model);
    const c: zml.Tensor = .fromShape(add_shape_data);
    const d: zml.Tensor = .fromShape(add_shape_model);

    const model: AddModel = .init();

    var shardings = [_]zml.sharding.Sharding{ sharding_data, sharding_model };

    var exe = try platform.compile(allocator, io, model, .forward, .{ a, b, c, d }, try .init(.shardy, shardings[0..]));
    defer exe.deinit();

    var a_buf = try createSequenceBuffer(allocator, io, platform, a.shape(), sharding_data, 0.0);
    defer a_buf.deinit();
    var b_buf = try createSequenceBuffer(allocator, io, platform, b.shape(), sharding_model, 1.0);
    defer b_buf.deinit();

    var c_buf = try createSequenceBuffer(allocator, io, platform, c.shape(), sharding_data, 2.0);
    defer c_buf.deinit();
    var d_buf = try createSequenceBuffer(allocator, io, platform, d.shape(), sharding_model, 3.0);
    defer d_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(.{ a_buf, b_buf, c_buf, d_buf });

    log.info("Buffer a: {f}", .{a_buf.placement()});
    log.info("Buffer b: {f}", .{b_buf.placement()});
    log.info("Buffer c: {f}", .{c_buf.placement()});
    log.info("Buffer d: {f}", .{d_buf.placement()});

    log.info("Running add example (a+c, b+d)...", .{});
    exe.call(exe_args, &exe_results);

    const Out = struct { sum_ac: zml.Buffer, sum_bd: zml.Buffer };
    var out = exe_results.get(Out);
    defer out.sum_ac.deinit();
    defer out.sum_bd.deinit();

    const sum_ac_slice = try out.sum_ac.toSliceAlloc(allocator, io);
    defer sum_ac_slice.free(allocator);

    const sum_bd_slice = try out.sum_bd.toSliceAlloc(allocator, io);
    defer sum_bd_slice.free(allocator);

    const sum_ac = sum_ac_slice.items(f32);
    const sum_bd = sum_bd_slice.items(f32);

    log.info(" sum_ac[0]={any} sum_bd[0]={any}", .{ sum_ac[0..32], sum_bd[0..32] });

    // var stdout = std.Io.File.stdout().writer(io, &.{});
    // try sum_bd_slice.prettyPrint(&stdout.interface, .{});

    log.info("\n\n{f}", .{out.sum_bd});
}

pub fn main() !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer std.debug.assert(debug_allocator.deinit() == .ok);

    const allocator = debug_allocator.allocator();
    // const allocator = std.heap.smp_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    var env_map = try std.process.getEnvMap(allocator);
    defer env_map.deinit();

    const device_count = blk: {
        if (env_map.get("DEVICE_COUNT")) |v| {
            break :blk std.fmt.parseInt(usize, v, 10) catch unreachable;
        }

        break :blk 8;
    };

    var platform: *zml.Platform = try .auto(allocator, io, .{ .cpu = .{ .device_count = @intCast(device_count) } });
    defer platform.deinit(allocator);

    log.info("{f}\n\n", .{platform.fmtVerbose()});

    // var physical_mesh: zml.sharding.PhysicalMesh = try .auto(allocator, platform);
    // defer physical_mesh.deinit();

    // var physical_mesh: zml.sharding.PhysicalMesh = try .init(allocator, .cuda, .{ .link = 8 }, .tree);
    // defer physical_mesh.deinit();

    // var physical_mesh: zml.sharding.PhysicalMesh = try .init(allocator, .neuron, .{ .bus = 2, .link = 12 }, .{ .ring = .closed_ring });
    // defer physical_mesh.deinit();

    var physical_mesh: zml.sharding.PhysicalMesh = blk: {
        if (device_count == 9) {
            const topology: zml.sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
                .axis(.link_y, .{ .mesh = .torus }, &.{
                    .axis(.link_z, .{ .mesh = .torus }, &.{
                        .device(platform.devices[3]), .device(platform.devices[1]),
                    }),
                    .axis(.link_z, .{ .mesh = .torus }, &.{
                        .device(platform.devices[2]), .device(platform.devices[0]),
                    }),
                }),
                .axis(.link_y, .{ .mesh = .torus }, &.{
                    .axis(.link_z, .{ .mesh = .torus }, &.{
                        .device(platform.devices[4]), .device(platform.devices[5]),
                    }),
                    .axis(.link_z, .{ .mesh = .torus }, &.{
                        .device(platform.devices[6]), .device(platform.devices[7]),
                    }),
                }),
            });

            break :blk try .fromTree(allocator, platform.target, topology);
        } else {
            break :blk try .auto(allocator, platform);
        }
    };
    defer physical_mesh.deinit();

    log.info("{f}", .{physical_mesh});

    // 2D logical mesh (data)
    const mesh_data: zml.sharding.LogicalMesh = try .init("data_mesh", .{
        .batch = .low_bandwidth,
        .context = .balanced,
    });
    log.info("{f}", .{mesh_data});

    // 3D logical mesh (model)
    const mesh_model: zml.sharding.LogicalMesh = try .init("model_mesh_3d", .{
        .model = .high_bandwidth,
        .head = .balanced,
        .expert = .low_bandwidth,
    });
    log.info("{f}", .{mesh_model});

    const strategy_data: zml.sharding.Strategy = try .suggest(mesh_data, physical_mesh);
    const strategy_model: zml.sharding.Strategy = try .suggest(mesh_model, physical_mesh);

    const sharding_data: zml.sharding.Sharding = try .initFromStrategy(mesh_data, physical_mesh, strategy_data);
    log.info("{f}", .{sharding_data});

    const sharding_model: zml.sharding.Sharding = try .initFromStrategy(mesh_model, physical_mesh, strategy_model);
    log.info("{f}", .{sharding_model});

    try runAdditionExample(allocator, io, platform, sharding_data, sharding_model);
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

fn createSequenceBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.sharding.Sharding,
    start: f32,
) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    for (slice.items(f32), 0..) |*e, i| {
        e.* = start + @as(f32, @floatFromInt(i));
    }

    return zml.Buffer.fromSlice(io, platform, slice, sharding);
}

fn createConstantBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.sharding.Sharding,
    value: f32,
) !zml.Buffer {
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);

    for (slice.items(f32)) |*e| {
        e.* = value;
    }

    return zml.Buffer.fromSlice(io, platform, slice, sharding);
}
