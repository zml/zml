const std = @import("std");
const async = @import("async");
const zml = @import("zml");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const ShapeOf = zml.ShapeOf;
const floats = zml.floats;

const log = std.log.scoped(.gather_mlp);
pub const std_options: std.Options = .{
    .log_level = .info,
};

pub const Gather = struct {
    split_number: i64 = 32,

    pub fn forward(self: Gather, input: zml.Tensor, indices: zml.Tensor) zml.Tensor {
        const slice_size = @divExact(input.dim(.d), self.split_number);
        var gathered = input.gatherSlices(Shape.init(.{ .b = 1, .d = slice_size }, .f32), indices.scale(slice_size), .{});
        log.info("Gathered shape: {f}", .{gathered.shape()});
        gathered = gathered.transpose(.{ .b, .n, .d }).reshape(Shape.init(.{ .b = input.dim(.b), .d = input.dim(.d) }, .f32));
        return gathered;
    }
};

pub const Scatter = struct {
    split_number: i64 = 32,

    pub fn forward(self: Scatter, input: zml.Tensor, indices: zml.Tensor, empty: zml.Tensor) zml.Tensor {
        _ = self;
        //var scattered = Tensor.constant(.{ .f32 = 0 }).broad(Shape.init(.{ .b = input.dim(.b), .d = input.dim(.d) }, .f32));
        log.info("Input shape: {f}", .{input.shape()});
        log.info("Indices shape: {f}", .{indices.shape()});
        log.info("Empty shape: {f}", .{empty.shape()});
        const scattered = empty.scatterSlices(.{ .d = indices }, input, .{});
        log.info("Scattered shape: {f}", .{scattered.shape()});
        // scattered = scattered.reshape(Shape.init(.{
        //     .b = 256,
        //     .d = 8192,
        // }, .f32));
        return scattered;
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    const Config = struct {
        up_proj_shape: Shape,
        gate_proj_shape: Shape,
        down_proj_shape: Shape,
    };

    pub fn init(config: Config) !Mlp {
        return Mlp{
            .up_proj = zml.nn.Linear.init(config.up_proj_shape, false),
            .gate_proj = zml.nn.Linear.init(config.gate_proj_shape, false),
            .down_proj = zml.nn.Linear.init(config.down_proj_shape, false),
        };
    }

    pub fn deinit(self: *Mlp) void {
        _ = self;
    }

    pub fn loadBuffers(model: Mlp, allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform) !zml.Bufferized(Mlp) {
        var rng_state = std.Random.DefaultPrng.init(42);
        const random = rng_state.random();

        const up_proj_weight_slice: zml.Slice = try .alloc(allocator, model.up_proj.weight.shape());
        defer up_proj_weight_slice.free(allocator);
        for (up_proj_weight_slice.items(f32)) |*val| {
            val.* = @floatCast((random.float(f32) * 2.0) - 1.0);
        }

        for (up_proj_weight_slice.items(f32)) |*val| {
            val.* = @floatCast((random.float(f32) * 2.0) - 1.0);
        }
        std.log.info("Up proj weights: {any}", .{up_proj_weight_slice.items(f32)[0..10]});

        const gate_proj_weight_slice: zml.Slice = try .alloc(allocator, model.gate_proj.weight.shape());
        defer gate_proj_weight_slice.free(allocator);
        for (gate_proj_weight_slice.items(f32)) |*val| {
            val.* = @floatCast((random.float(f32) * 2.0) - 1.0);
        }
        std.log.info("Gate proj weights: {any}", .{gate_proj_weight_slice.items(f32)[0..10]});

        const down_proj_weight_slice: zml.Slice = try .alloc(allocator, model.down_proj.weight.shape());
        defer down_proj_weight_slice.free(allocator);
        for (down_proj_weight_slice.items(f32)) |*val| {
            val.* = @floatCast((random.float(f32) * 2.0) - 1.0);
        }
        std.log.info("Down proj weights: {any}", .{down_proj_weight_slice.items(f32)[0..10]});

        const up_proj_weight_buffer: zml.Buffer = try .fromBytes(io, platform, up_proj_weight_slice.shape, up_proj_weight_slice.data); // COOL LE SLICE.DATA
        errdefer up_proj_weight_buffer.deinit();

        const gate_proj_weight_buffer: zml.Buffer = try .fromBytes(io, platform, gate_proj_weight_slice.shape, gate_proj_weight_slice.data);
        errdefer gate_proj_weight_buffer.deinit();

        const down_proj_weight_buffer: zml.Buffer = try .fromBytes(io, platform, down_proj_weight_slice.shape, down_proj_weight_slice.data);
        errdefer down_proj_weight_buffer.deinit();

        return .{ .up_proj = .{ .weight = up_proj_weight_buffer }, .gate_proj = .{ .weight = gate_proj_weight_buffer }, .down_proj = .{ .weight = down_proj_weight_buffer } };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        self.up_proj.weight.deinit();
        self.gate_proj.weight.deinit();
        self.down_proj.weight.deinit();
    }

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const proj = self.up_proj.forward(x);
        var output = self.gate_proj.forward(x);
        output = output.silu().mul(proj);
        const result = self.down_proj.forward(output);
        return result;
    }
};

pub const ScatterMlp = struct {
    scatter: Scatter,
    mlp: Mlp,

    pub fn init(config: Mlp.Config) !ScatterMlp {
        return ScatterMlp{
            .scatter = Scatter{},
            .mlp = try Mlp.init(config),
        };
    }

    pub fn deinit(self: *ScatterMlp) void {
        _ = self;
    }

    pub fn loadBuffers(self: ScatterMlp, allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform) !zml.Bufferized(ScatterMlp) {
        const mlp_buffers = try self.mlp.loadBuffers(allocator, io, platform);
        return .{
            //.scatter = .{}, // I wanted to put a void struct here
            .mlp = mlp_buffers,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ScatterMlp)) void {
        Mlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(self: ScatterMlp, input: zml.Tensor, indices: zml.Tensor, empty: zml.Tensor) zml.Tensor {
        const scattered = self.scatter.forward(input, indices, empty);
        const output = self.mlp.forward(scattered);
        return output.reuseBuffer(empty);
    }
};

pub const GatherMlp = struct {
    gather: Gather,
    mlp: Mlp,

    pub fn init(config: Mlp.Config) !GatherMlp {
        return GatherMlp{
            .gather = Gather{},
            .mlp = try Mlp.init(config),
        };
    }

    pub fn deinit(self: *GatherMlp) void {
        _ = self;
    }

    pub fn loadBuffers(self: GatherMlp, allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform) !zml.Bufferized(GatherMlp) {
        const mlp_buffers = try self.mlp.loadBuffers(allocator, io, platform);
        return .{
            .mlp = mlp_buffers,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GatherMlp)) void {
        Mlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(self: GatherMlp, input: zml.Tensor, indices: zml.Tensor) zml.Tensor {
        const gathered = self.gather.forward(input, indices);
        const output = self.mlp.forward(gathered);
        return output;
    }
};

const Result = struct {
    batch_size: i64,
    avg_time_ms: f64,
    min_time_ms: f64,
    max_time_ms: f64,
    times_ms: []f64,
};

// const params = clap.parseParamsComptime(
//     \\--help                      print this help
//     \\--hf-model-path  <STRING>   path to the directory containing model weights, config and tokenizer
//     \\--seq-len        <UINT>     sequence length (default: 512)
//     \\--create-options <STRING>   platform creation options in ZON format, defaults to {}
// );

// fn bool_parser(in: []const u8) error{}!bool {
//     return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
// }

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator);
    defer threaded.deinit();

    const io = threaded.io();

    zml.init();
    defer zml.deinit();

    var platform = try zml.Platform.init(.cuda, io, .{});
    defer platform.deinit();

    const config = Mlp.Config{
        .up_proj_shape = zml.Shape.init(.{ 28672, 8192 }, .f32),
        .gate_proj_shape = zml.Shape.init(.{ 28672, 8192 }, .f32),
        .down_proj_shape = zml.Shape.init(.{ 8192, 28672 }, .f32),
    };

    const batch_size = 256;
    const hidden_size = config.up_proj_shape.dim(1);
    // const input: zml.Tensor = .init(zml.Shape.init(.{ .b = batch_size, .d = hidden_size }, .f32), .f32);
    const input: zml.Tensor = .init(zml.Shape.init(.{ .b = batch_size, .d = hidden_size }, .f32), .f32);

    const split_number = 32;
    const sub_slice_size = @as(usize, @intCast(@divExact(hidden_size, split_number)));

    const indices_shape = zml.Shape.init(.{ .n = split_number * batch_size, .dim = 2 }, .u32);
    const indices: zml.Tensor = .fromShape(indices_shape);
    _ = indices; // autofix
    const indices_slice: zml.Slice = try .alloc(allocator, indices_shape);
    defer indices_slice.free(allocator);

    const slice: zml.Slice = try .alloc(allocator, input.shape());
    defer slice.free(allocator);

    const f32_items = slice.items(f32);
    const indices_slice_size = batch_size * split_number * 2; // 2 coord per index
    const indices_slice_data = try allocator.alloc(u32, indices_slice_size);
    defer allocator.free(indices_slice_data);

    for (0..split_number * batch_size) |i| {
        const b = @divFloor(i, split_number);
        const i_as_f32: f32 = @as(f32, @floatFromInt(i));
        const value_to_assign: f32 = try std.math.mod(f32, (32.0 - 1.0 - i_as_f32), split_number);

        const sub_slice = f32_items[i * sub_slice_size .. (i + 1) * sub_slice_size];

        for (sub_slice) |*val| {
            val.* = value_to_assign;
        }

        const index_value: u32 = 32 - 1 - @as(u32, @intCast(try std.math.mod(u32, @as(u32, @intCast(i)), split_number))); // Convert to u32

        indices_slice.items(u32)[i * 2 + 0] = @as(u32, @intCast(b)); // batch offset

        indices_slice.items(u32)[i * 2 + 1] = index_value;
    }

    // {
    //     var gather_mlp = try GatherMlp.init(config);
    //     defer gather_mlp.deinit();

    //     var model_buffers = try gather_mlp.loadBuffers(allocator, io, platform);
    //     defer GatherMlp.unloadBuffers(&model_buffers);

    //     log.info("Input : {f}", .{input.shape()});
    //     var exe_gather_mlp = try platform.compileModel(allocator, io, GatherMlp.forward, gather_mlp, .{ input, indices });
    //     defer exe_gather_mlp.deinit();
    //     var gather_mlp_args = try exe_gather_mlp.args(allocator);
    //     defer gather_mlp_args.deinit(allocator);
    //     var gather_mlp_results = try exe_gather_mlp.results(allocator);
    //     defer gather_mlp_results.deinit(allocator);

    //     const input_buffer: zml.Buffer = try .from(io, platform, slice.shape, slice.data, .{ .wait = true, .memory = .host_pinned });
    //     defer input_buffer.deinit();

    //     const indices_buffer: zml.Buffer = try .from(io, platform, indices_shape, indices_slice.data, .{ .wait = true, .memory = .host_pinned });
    //     defer indices_buffer.deinit();

    //     gather_mlp_args.set(.{ model_buffers, input_buffer, indices_buffer });
    //     //warmup
    //     exe_gather_mlp.call(gather_mlp_args, &gather_mlp_results, io);
    //     //Real call
    //     exe_gather_mlp.call(gather_mlp_args, &gather_mlp_results, io);
    //     const output = gather_mlp_results.get(zml.Buffer);
    //     defer output.deinit();
    //     const output_slice = try output.toSliceAlloc(allocator, io);
    //     defer output_slice.free(allocator);
    //     std.log.info("Output gather mlp: {any}", .{output_slice.items(f32)[0..10]});
    // }

    // {
    //     const gather = Gather{};
    //     var exe_gather = try platform.compileModel(allocator, io, Gather.forward, gather, .{ input, indices });
    //     defer exe_gather.deinit();
    //     var gather_args = try exe_gather.args(allocator);
    //     defer gather_args.deinit(allocator);
    //     var gather_results = try exe_gather.results(allocator);
    //     defer gather_results.deinit(allocator);

    //     const input_buffer: zml.Buffer = try .from(io, platform, slice.shape, slice.data, .{ .wait = true, .memory = .host_pinned });
    //     defer input_buffer.deinit();

    //     const indices_buffer: zml.Buffer = try .from(io, platform, indices_shape, indices_slice.data, .{ .wait = true, .memory = .host_pinned });
    //     defer indices_buffer.deinit();

    //     gather_args.set(.{ input_buffer, indices_buffer });
    //     //warmup
    //     exe_gather.call(gather_args, &gather_results, io);
    //     //Real call
    //     exe_gather.call(gather_args, &gather_results, io);
    //     const output = gather_results.get(zml.Buffer);
    //     defer output.deinit();
    //     const output_slice = try output.toSliceAlloc(allocator, io);
    //     defer output_slice.free(allocator);
    //     std.log.info("Output gather: {any}", .{output_slice.items(f32)[8192..16384]});
    // }

    const indices_scatter_shape = zml.Shape.init(.{ .b = batch_size, .d = hidden_size }, .u32);
    const indices_scatter: zml.Tensor = .init(indices_scatter_shape, .u32);
    const indices_scatter_slice: zml.Slice = try .alloc(allocator, indices_scatter_shape);
    defer indices_scatter_slice.free(allocator);

    log.info("Indices scatter shape: {f}", .{indices_scatter.shape()});

    for (0..split_number * batch_size) |i| {
        const shuffle_i_u32: u32 = 32 - 1 - (try std.math.mod(u32, @as(u32, @intCast(i)), split_number));
        for (0..sub_slice_size) |j| {
            const scatter_index = i * sub_slice_size + j;
            const scatter_value = shuffle_i_u32 * @as(u32, @intCast(sub_slice_size)) + @as(u32, @intCast(j));

            indices_scatter_slice.items(u32)[scatter_index] = scatter_value;
        }
    }

    log.info("Indices scatter slice: {any}", .{indices_scatter_slice.items(u32)[0..8192]});

    // std.log.info("Input: {any}", .{slice.items(f32)[0..8192]});
    // std.log.info("Indices: {any}", .{indices_slice.items(u32)[0..]});

    // const indices_buffer: zml.Buffer = try .fromSlice(io, platform, indices_slice);
    // defer indices_buffer.deinit();

    // const indices_scatter_buffer: zml.Buffer = try .fromSlice(io, platform, indices_scatter_slice);
    // defer indices_scatter_buffer.deinit();

    const empty: zml.Tensor = .init(zml.Shape.init(.{ .b = batch_size, .d = hidden_size }, .f32), .f32);
    const empty_slice: zml.Slice = try .alloc(allocator, empty.shape());
    defer empty_slice.free(allocator);
    for (empty_slice.items(f32)) |*val| {
        val.* = 0.0;
    }

    // const empty_buffer: zml.Buffer = try .fromSlice(io, platform, empty_slice);
    // defer empty_buffer.deinit();

    {
        var scatter_mlp = try ScatterMlp.init(config);
        defer scatter_mlp.deinit();

        var model_buffers = try scatter_mlp.loadBuffers(allocator, io, platform);
        defer ScatterMlp.unloadBuffers(&model_buffers);

        log.info("Input : {f}", .{input.shape()});
        var exe_scatter_mlp = try platform.compileModel(allocator, io, ScatterMlp.forward, scatter_mlp, .{ input, indices_scatter, empty });
        defer exe_scatter_mlp.deinit();
        var scatter_mlp_args = try exe_scatter_mlp.args(allocator);
        defer scatter_mlp_args.deinit(allocator);
        var scatter_mlp_results = try exe_scatter_mlp.results(allocator);
        defer scatter_mlp_results.deinit(allocator);

        const indices_scatter_buffer: zml.Buffer = try .from(io, platform, indices_scatter_shape, indices_scatter_slice.data, .{ .wait = true, .memory = .host_pinned });
        defer indices_scatter_buffer.deinit();

        const input_buffer: zml.Buffer = try .from(io, platform, slice.shape, slice.data, .{ .wait = true, .memory = .host_pinned });
        defer input_buffer.deinit();

        var empty_buffer: zml.Buffer = try .fromSlice(io, platform, empty_slice);
        defer empty_buffer.deinit();
        

        //warmup
        scatter_mlp_args.set(.{ model_buffers, input_buffer, indices_scatter_buffer, empty_buffer });
        exe_scatter_mlp.call(scatter_mlp_args, &scatter_mlp_results, io);
        empty_buffer = scatter_mlp_results.get(zml.Buffer);

        //Real call
        scatter_mlp_args.set(.{ model_buffers, input_buffer, indices_scatter_buffer, empty_buffer });
        exe_scatter_mlp.call(scatter_mlp_args, &scatter_mlp_results, io);
        empty_buffer = scatter_mlp_results.get(zml.Buffer);

        const output_slice = try empty_buffer.toSliceAlloc(allocator, io);
        defer output_slice.free(allocator);
        std.log.info("Output scatter mlp: {any}", .{output_slice.items(f32)[0..10]});
    }

    // {
    //     var mlp = try Mlp.init(config);
    //     defer mlp.deinit();
    //     var model_buffers = try mlp.loadBuffers(allocator, io, platform);
    //     defer Mlp.unloadBuffers(&model_buffers);
    //     var exe_mlp = try platform.compileModel(allocator, io, Mlp.forward, mlp, .{input});
    //     defer exe_mlp.deinit();
    //     var mlp_args = try exe_mlp.args(allocator);
    //     defer mlp_args.deinit(allocator);
    //     var mlp_results = try exe_mlp.results(allocator);
    //     defer mlp_results.deinit(allocator);
    //     mlp_args.set(.{ model_buffers, input_buffer });
    //     //warmup
    //     exe_mlp.call(mlp_args, &mlp_results, io);
    //     //Real call
    //     exe_mlp.call(mlp_args, &mlp_results, io);
    //     const output = mlp_results.get(zml.Buffer);
    //     defer output.deinit();
    //     const output_slice = try output.toSliceAlloc(allocator, io);
    //     defer output_slice.free(allocator);
    //     std.log.info("Output mlp: {any}", .{output_slice.items(f32)[0..8192]});
    // }

    // log.info("After scatter_mlp", .{});

    // {
    //     var model: Mlp = try .init(config);
    //     defer model.deinit();

    //     var model_buffers = try model.loadBuffers(allocator, io, platform);

    //     defer Mlp.unloadBuffers(&model_buffers);

    //     var exe = try platform.compileModel(allocator, io, Mlp.forward, model, .{input});
    //     defer exe.deinit();

    //     var args = try exe.args(allocator);
    //     defer args.deinit(allocator);

    //     var results = try exe.results(allocator);
    //     defer results.deinit(allocator);

    //     args.set(.{ model_buffers, input_buffer });
    //     exe.call(args, &results, io);

    //     const output = results.get(zml.Buffer);
    //     defer output.deinit();

    //     const output_slice = try output.toSliceAlloc(allocator, io);
    //     defer output_slice.free(allocator);
    //     std.log.info("Output: {any}", .{output_slice.items(f32)[0..10]});
    // }

    //Simple scatter

    // {
    //     const scatter = Scatter{};
    //     var exe_scatter = try platform.compileModel(allocator, io, Scatter.forward, scatter, .{ input, indices_scatter, empty });
    //     defer exe_scatter.deinit();
    //     var scatter_args = try exe_scatter.args(allocator);
    //     defer scatter_args.deinit(allocator);
    //     var scatter_results = try exe_scatter.results(allocator);
    //     defer scatter_results.deinit(allocator);
    //     scatter_args.set(.{ input_buffer, indices_scatter_buffer, empty_buffer });
    //     //warmup
    //     exe_scatter.call(scatter_args, &scatter_results, io);

    //     //Real call
    //     exe_scatter.call(scatter_args, &scatter_results, io);
    //     const output = scatter_results.get(zml.Buffer);
    //     defer output.deinit();
    //     const output_slice = try output.toSliceAlloc(allocator, io);
    //     defer output_slice.free(allocator);
    //     std.log.info("Output scatter: {any}", .{output_slice.items(f32)[0..]});
    // }
    // log.info("After scatter", .{});

    // Simple Gather

}

// pub fn convertBufferType(
//     store: *zml.aio.BufferStore,
//     allocator: std.mem.Allocator,
//     key: []const u8,
//     target_dtype: zml.DataType,
// ) !void {
//     _ = allocator; // autofix
//     const source_buffer = store.get(key) orelse {
//         std.log.err("Buffer not found: {s}", .{key});
//         return error.BufferNotFound;
//     };

//     const source_dtype = source_buffer.dtype();
//     const shape = source_buffer.shape();

//     const target_byte_size = shape.withDtype(target_dtype).byteSize();
//     const target_data = try store.arena.allocator().alloc(u8, target_byte_size);

//     switch (source_dtype) {
//         .f32 => {
//             const source_slice = source_buffer.items(f32);
//             switch (target_dtype) {
//                 .f8e4m3fn => {
//                     const target_slice = std.mem.bytesAsSlice(floats.Float8E4M3FN, target_data);
//                     for (source_slice, 0..) |val, i| {
//                         target_slice[i] = floats.Float8E4M3FN.fromF32(val);
//                     }
//                 },
//                 .f8e4m3 => {
//                     const target_slice = std.mem.bytesAsSlice(floats.Float8E4M3, target_data);
//                     for (source_slice, 0..) |val, i| {
//                         target_slice[i] = floats.Float8E4M3.fromF32(val);
//                     }
//                 },
//                 .f8e4m3fnuz => {
//                     const target_slice = std.mem.bytesAsSlice(floats.Float8E4M3FNUZ, target_data);
//                     for (source_slice, 0..) |val, i| {
//                         target_slice[i] = floats.Float8E4M3FNUZ.fromF32(val);
//                     }
//                 },
//                 .f8e5m2 => {
//                     const target_slice = std.mem.bytesAsSlice(floats.Float8E5M2, target_data);
//                     for (source_slice, 0..) |val, i| {
//                         target_slice[i] = floats.Float8E5M2.fromF32(val);
//                     }
//                 },
//                 .f8e5m2fnuz => {
//                     const target_slice = std.mem.bytesAsSlice(floats.Float8E5M2FNUZ, target_data);
//                     for (source_slice, 0..) |val, i| {
//                         target_slice[i] = floats.Float8E5M2FNUZ.fromF32(val);
//                     }
//                 },
//                 else => return error.UnsupportedTargetType,
//             }
//         },
//         .bf16 => {
//             const source_slice = source_buffer.items(floats.BFloat16);
//             switch (target_dtype) {
//                 .f8e4m3fn => {
//                     const target_slice = std.mem.bytesAsSlice(floats.Float8E4M3FN, target_data);
//                     for (source_slice, 0..) |val, i| {
//                         const f32_val = val.toF32();
//                         target_slice[i] = floats.Float8E4M3FN.fromF32(f32_val);
//                     }
//                 },
//                 .f8e4m3 => {
//                     const target_slice = std.mem.bytesAsSlice(floats.Float8E4M3, target_data);
//                     for (source_slice, 0..) |val, i| {
//                         const f32_val = val.toF32();
//                         target_slice[i] = floats.Float8E4M3.fromF32(f32_val);
//                     }
//                 },
//                 .f8e4m3fnuz => {
//                     const target_slice = std.mem.bytesAsSlice(floats.Float8E4M3FNUZ, target_data);
//                     for (source_slice, 0..) |val, i| {
//                         const f32_val = val.toF32();
//                         target_slice[i] = floats.Float8E4M3FNUZ.fromF32(f32_val);
//                     }
//                 },
//                 .f8e5m2 => {
//                     const target_slice = std.mem.bytesAsSlice(floats.Float8E5M2, target_data);
//                     for (source_slice, 0..) |val, i| {
//                         const f32_val = val.toF32();
//                         target_slice[i] = floats.Float8E5M2.fromF32(f32_val);
//                     }
//                 },
//                 .f8e5m2fnuz => {
//                     const target_slice = std.mem.bytesAsSlice(floats.Float8E5M2FNUZ, target_data);
//                     for (source_slice, 0..) |val, i| {
//                         const f32_val = val.toF32();
//                         target_slice[i] = floats.Float8E5M2FNUZ.fromF32(f32_val);
//                     }
//                 },
//                 else => return error.UnsupportedTargetType,
//             }
//         },
//         else => return error.UnsupportedSourceType,
//     }

//     const target_shape = shape.withDtype(target_dtype);
//     const target_host_buffer = zml.HostBuffer.fromBytes(target_shape, target_data);

//     const key_dup = try store.arena.allocator().dupe(u8, key);

//     try store.buffers.put(store.arena.allocator(), key_dup, target_host_buffer);
// }
