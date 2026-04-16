const std = @import("std");

const zml = @import("zml");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    activations: []const u8 = "/workspace/ref_nvfp4_quant_small.safetensors",

    pub const help =
        \\Use nvfp4_quant_compare --activations=<path>
        \\
        \\Compare two ZML NVFP4 quantization implementations against the
        \\reference outputs stored in a safetensors fixture.
        \\
        \\Options:
        \\   --activations=<path>      Path to the raw ref_nvfp4_quant safetensors
        \\
    ;
};

const QuantizeNVFP4Result = struct {
    block: zml.Tensor,
    scale: zml.Tensor,
};

const Nvfp4FromScratch = struct {
    pub fn forward(_: @This(), x_: zml.Tensor, global_scale_: zml.Tensor) QuantizeNVFP4Result {
        const x = x_.withTags(.{ .b, .d });
        const global_scale = global_scale_.reshape(.{}).convert(.f32);
        const one = zml.Tensor.scalar(@as(f32, 1.0), .f32);
        const global_scale_inv = one.div(global_scale);

        const result = quantizeNVFP4FromScratch(x, global_scale_inv);
        return .{
            .block = result.block.convert(.f32),
            .scale = result.scale.convert(.f32),
        };
    }
};

const Nvfp4CustomCall = struct {
    pub fn forward(_: @This(), x_: zml.Tensor, global_scale_: zml.Tensor) QuantizeNVFP4Result {
        const x = x_.withTags(.{ .b, .d }).convert(.f32);
        const global_scale = global_scale_.reshape(.{}).convert(.f32);
        const x_scaled = x.mul(global_scale.broad(x.shape()));
        const result = quantizeNVFP4CustomCall(x_scaled);
        return .{
            .block = result.block.convert(.f32),
            .scale = result.scale.convert(.f32),
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    var raw_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.activations);
    defer raw_registry.deinit();

    var fixture_registry = try buildFixtureRegistry(allocator, &raw_registry);
    defer fixture_registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &fixture_registry);
    defer activation_store.deinit();

    const replicated = try zml.sharding.replicatedSharding(platform);

    try testLayer(
        allocator,
        io,
        platform,
        activation_store.view(),
        "nvfp4_quant.from_scratch",
        Nvfp4FromScratch{},
        {},
        replicated,
        .{ .absolute_tolerance = 1e-6, .relative_tolerance = 1e-6 },
    );

    // try testLayer(
    //     allocator,
    //     io,
    //     platform,
    //     activation_store.view(),
    //     "nvfp4_quant.custom_call",
    //     Nvfp4CustomCall{},
    //     {},
    //     replicated,
    //     .{ .absolute_tolerance = 1e-6, .relative_tolerance = 1e-6 },
    // );
}

fn testLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    activation_store: zml.io.TensorStore.View,
    name: []const u8,
    layer: anytype,
    layer_weights: zml.Bufferized(@TypeOf(layer)),
    sharding: zml.sharding.Sharding,
    opts: zml.testing.CompareOpts,
) !void {
    const x_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name});
    defer allocator.free(x_key);
    const x_shape = activation_store.getShape(x_key) orelse return error.NotFound;
    var x_buffer = try loadBufferFromStore(allocator, io, platform, activation_store, x_key, sharding);
    defer x_buffer.deinit();
    const x_tensor = zml.Tensor.fromShape(x_shape);

    const scale_key = try std.fmt.allocPrint(allocator, "{s}.in.1", .{name});
    defer allocator.free(scale_key);
    const scale_shape = activation_store.getShape(scale_key) orelse return error.NotFound;
    var scale_buffer = try loadBufferFromStore(allocator, io, platform, activation_store, scale_key, sharding);
    defer scale_buffer.deinit();
    const scale_tensor = zml.Tensor.fromShape(scale_shape);

    const block_key = try std.fmt.allocPrint(allocator, "{s}.out.0", .{name});
    defer allocator.free(block_key);
    var block_expected = try loadBufferFromStore(allocator, io, platform, activation_store, block_key, sharding);
    defer block_expected.deinit();

    const out_scale_key = try std.fmt.allocPrint(allocator, "{s}.out.1", .{name});
    defer allocator.free(out_scale_key);
    var out_scale_expected = try loadBufferFromStore(allocator, io, platform, activation_store, out_scale_key, sharding);
    defer out_scale_expected.deinit();

    const exe = try platform.compileFn(allocator, io, @TypeOf(layer).forward, .{ layer, x_tensor, scale_tensor }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ layer_weights, x_buffer, scale_buffer });

    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    exe.call(args, &results);

    var block_result, var scale_result = results.get(struct { zml.Buffer, zml.Buffer });
    defer block_result.deinit();
    defer scale_result.deinit();

    try printFirst20(io, "block.expected", block_expected);
    try printFirst20(io, "block.actual", block_result);
    try printFirst20(io, "scale.expected", out_scale_expected);
    try printFirst20(io, "scale.actual", scale_result);

    try zml.testing.expectClose(io, block_result, block_expected, opts);
    try zml.testing.expectClose(io, scale_result, out_scale_expected, opts);
}

fn printFirst20(io: std.Io, label: []const u8, buffer: zml.Buffer) !void {
    const allocator = std.heap.smp_allocator;
    const slice = try buffer.toSliceAlloc(allocator, io);
    defer slice.free(allocator);

    switch (buffer.shape().dtype()) {
        inline else => |dt| {
            const T = dt.toZigType();
            const values = slice.constItems(T);
            const n = @min(values.len, 20);

            std.log.info("{s} ({s}) first {d} values:", .{ label, dt.str(), n });
            for (values[0..n], 0..) |value, i| {
                switch (comptime dt.class()) {
                    .float => std.log.info("  [{d}] {d}", .{ i, zml.floats.floatCast(f32, value) }),
                    .integer => std.log.info("  [{d}] {d}", .{ i, value }),
                    .bool => std.log.info("  [{d}] {}", .{ i, value }),
                    .complex => std.log.info("  [{d}] {any}", .{ i, value }),
                }
            }
        },
    }
}

fn loadBufferFromStore(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: zml.io.TensorStore.View,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    const shape = store.getShape(key) orelse return error.NotFound;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.getReader(key, io, &io_buffer);
    defer reader.deinit();

    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}

fn buildFixtureRegistry(
    allocator: std.mem.Allocator,
    raw_registry: *zml.safetensors.TensorRegistry,
) !zml.safetensors.TensorRegistry {
    var registry = try zml.safetensors.TensorRegistry.initWithMetadata(allocator, raw_registry.metadata);

    try registerAlias(&registry, raw_registry, "x", "nvfp4_quant.from_scratch.in.0");
    try registerAlias(&registry, raw_registry, "global_scale", "nvfp4_quant.from_scratch.in.1");
    try registerAlias(&registry, raw_registry, "q_fp4", "nvfp4_quant.from_scratch.out.0");
    try registerAlias(&registry, raw_registry, "scale", "nvfp4_quant.from_scratch.out.1");

    try registerAlias(&registry, raw_registry, "x", "nvfp4_quant.custom_call.in.0");
    try registerAlias(&registry, raw_registry, "global_scale", "nvfp4_quant.custom_call.in.1");
    try registerAlias(&registry, raw_registry, "q_fp4", "nvfp4_quant.custom_call.out.0");
    try registerAlias(&registry, raw_registry, "scale", "nvfp4_quant.custom_call.out.1");

    return registry;
}

fn registerAlias(
    registry: *zml.safetensors.TensorRegistry,
    raw_registry: *zml.safetensors.TensorRegistry,
    source_name: []const u8,
    alias_name: []const u8,
) !void {
    const source = raw_registry.tensors.get(source_name) orelse return error.TensorNotFound;
    try registry.registerTensor(.{
        .file_uri = source.file_uri,
        .name = alias_name,
        .shape = source.shape,
        .offset = source.offset,
    });
}

fn quantizeNVFP4CustomCall(x: zml.Tensor) QuantizeNVFP4Result {
    const block_size: i64 = 16;
    const d_group = @divExact(x.dim(.d), block_size);

    const quant_shape = zml.Shape.init(.{ .b = x.dim(.b), .d = x.dim(.d) }, .f4e2m1);
    const scale_shape = zml.Shape.init(.{ .b = x.dim(.b), .d_group = d_group }, .f8e4m3fn);

    const outputs: [2]zml.Tensor = zml.ops.customCall(
        "__op$quantize",
        .{x},
        .{ quant_shape, scale_shape },
        .{},
        .{ .has_side_effect = false },
    );

    return .{
        .block = outputs[0],
        .scale = outputs[1],
    };
}

fn quantizeNVFP4FromScratch(x: zml.Tensor, input_global_scale_inv: zml.Tensor) QuantizeNVFP4Result {
    x.print("input in custom quantization");
    input_global_scale_inv.print("input global scale inv in custom quantization");
    const x_f32 = x.convert(.f32);
    const block_size: i64 = 16;
    const d_group = @divExact(x_f32.dim(.d), block_size);

    const x_blocks = x_f32.reshape(.{ .b = x_f32.dim(.b), .d_group = d_group, .d_block = block_size });

    var amax = x_blocks.abs().max(.d_block);
    amax = amax.reshape(.{ .b = x_f32.dim(.b), .d_group = d_group });

    const one = zml.Tensor.scalar(@as(f32, 1.0), .f32);
    const fp4_max = zml.Tensor.scalar(@as(f32, 6.0), .f32);
    const fp8_max = zml.Tensor.scalar(@as(f32, 448.0), .f32);
    const fp8_min = zml.Tensor.scalar(@as(f32, -448.0), .f32);
    const zero = zml.Tensor.scalar(@as(f32, 0.0), .f32);

    const input_global_scale = one.div(input_global_scale_inv);

    const scale = blk: {
        const amax_norm = amax.div(fp4_max.broad(amax.shape()));
        var s = input_global_scale.broad(amax.shape()).mul(amax_norm);
        s = s.maximum(fp8_min.broad(s.shape())).minimum(fp8_max.broad(s.shape()));
        break :blk quantizeE4M3FnLikeTorch(s);
    };

    const input_scale_recip = input_global_scale_inv.broad(scale.shape());
    const denom = scale.convert(.f32).mul(input_scale_recip);
    const output_scale = denom.cmp(.EQ, zero).select(
        zero.broad(denom.shape()),
        one.broad(denom.shape()).div(denom),
    );

    var scaled_x = x_blocks.mul(output_scale.broad(x_blocks.shape()))
        .reshape(.{ .b = x_f32.dim(.b), .d = x_f32.dim(.d) });

    scaled_x = scaled_x
        .maximum(zml.Tensor.scalar(@as(f32, -6.0), .f32).broad(scaled_x.shape()))
        .minimum(fp4_max.broad(scaled_x.shape()));

    return .{
        .block = quantizeFp4LikeVllm(scaled_x).convert(.f4e2m1),
        .scale = scale.convert(.f32).convert(.f8e4m3fn),
    };
}

fn quantizeFp4LikeVllm(x: zml.Tensor) zml.Tensor {
    const x_abs = x.abs();
    var q = zml.Tensor.scalar(@as(f32, 0.0), .f32).broad(x.shape());

    q = x_abs.cmp(.GT, zml.Tensor.scalar(@as(f32, 0.25), .f32)).select(zml.Tensor.scalar(@as(f32, 0.5), .f32), q);
    q = x_abs.cmp(.GE, zml.Tensor.scalar(@as(f32, 0.75), .f32)).select(zml.Tensor.scalar(@as(f32, 1.0), .f32), q);
    q = x_abs.cmp(.GT, zml.Tensor.scalar(@as(f32, 1.25), .f32)).select(zml.Tensor.scalar(@as(f32, 1.5), .f32), q);
    q = x_abs.cmp(.GE, zml.Tensor.scalar(@as(f32, 1.75), .f32)).select(zml.Tensor.scalar(@as(f32, 2.0), .f32), q);
    q = x_abs.cmp(.GT, zml.Tensor.scalar(@as(f32, 2.5), .f32)).select(zml.Tensor.scalar(@as(f32, 3.0), .f32), q);
    q = x_abs.cmp(.GE, zml.Tensor.scalar(@as(f32, 3.5), .f32)).select(zml.Tensor.scalar(@as(f32, 4.0), .f32), q);
    q = x_abs.cmp(.GT, zml.Tensor.scalar(@as(f32, 5.0), .f32)).select(zml.Tensor.scalar(@as(f32, 6.0), .f32), q);

    const sign = x.cmp(.LT, zml.Tensor.scalar(@as(f32, 0.0), .f32)).select(
        zml.Tensor.scalar(@as(f32, -1.0), .f32),
        zml.Tensor.scalar(@as(f32, 1.0), .f32),
    );
    return q.mul(sign);
}

fn quantizeE4M3FnLikeTorch(x: zml.Tensor) zml.Tensor {
    const abs_x = x.abs();
    var q_abs = zml.Tensor.scalar(@as(f32, 0.0), .f32).broad(x.shape());

    inline for (fp8e4m3fn_boundaries, fp8e4m3fn_positive_values[0..]) |boundary, value| {
        q_abs = abs_x.cmp(.GT, zml.Tensor.scalar(boundary, .f32)).select(
            zml.Tensor.scalar(value, .f32),
            q_abs,
        );
    }

    const sign = x.cmp(.LT, zml.Tensor.scalar(@as(f32, 0.0), .f32)).select(
        zml.Tensor.scalar(@as(f32, -1.0), .f32),
        zml.Tensor.scalar(@as(f32, 1.0), .f32),
    );
    return q_abs.mul(sign);
}

const fp8e4m3fn_positive_values = buildPositiveFiniteE4M3FnValues();
const fp8e4m3fn_boundaries = buildPositiveFiniteE4M3FnBoundaries(fp8e4m3fn_positive_values);

fn buildPositiveFiniteE4M3FnValues() [126]f32 {
    var values: [126]f32 = undefined;
    inline for (1..127) |bits_usize| {
        const bits: u8 = @intCast(bits_usize);
        values[bits_usize - 1] = zml.floats.Float8E4M3FN.toF32(@bitCast(bits));
    }
    return values;
}

fn buildPositiveFiniteE4M3FnBoundaries(comptime values: [126]f32) [126]f32 {
    var boundaries: [126]f32 = undefined;
    boundaries[0] = values[0] / 2.0;

    inline for (1..values.len) |i| {
        boundaries[i] = (values[i - 1] + values[i]) / 2.0;
    }

    return boundaries;
}
