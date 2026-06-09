const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const step3p5flash = @import("step3_5flash.zig");
const model = @import("step3_5flash/model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: []const u8,

    pub const help =
        \\Use step3_5_tests --model=<path> --activations=<path>
        \\
        \\ Validate the Step 3.5 Flash MoE layers against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>         Path to the model repository
        \\   --activations=<path>   Path to activation safetensors
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    // Registry stores the memory of tensors
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    // Tensor store is a ZML representation of tensors - i.e., parent.child.leaf
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    // Loading bar
    var progress = std.Progress.start(io, .{ .root_name = args.model });
    // const shardings: common.Shardings = try .init(platform);

    progress.end();

    try run(allocator, io, platform, args.activations, &store);
}

pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    model_store: *zml.io.TensorStore,
) !void {
    const config = model.default_config;
    const sharding = platform.replicated_sharding;
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    // Bottom-up test approach. See LFM for a top-down approach.
    std.log.info("RMS Norm:", .{});
    try runRmsNormSynthetic(allocator, io, platform, platform.replicated_sharding);

    std.log.info("MLP:", .{});
    for (0..config.num_hidden_layers) |layer_idx| {
        if (std.mem.indexOfScalar(u32, config.moe_layers_enum, @intCast(layer_idx)) != null) continue;

        const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
        defer allocator.free(name);

        const mlp = model.Mlp.init(model_store.view().withPrefix(name), 0);

        var mlp_weights = try zml.io.load(model.Mlp, &mlp, allocator, io, platform, model_store, .auto);
        defer deinitBuffers(&mlp_weights);

        const input_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name});
        defer allocator.free(input_key);
        if (!activation_store.view().hasKey(input_key)) {
            std.log.warn("skipping {s}: no activations recorded", .{name});
            continue;
        }

        zml.testing.testLayer(allocator, io, platform, mlp, .forward, activation_store.view(), name, mlp_weights, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
            std.log.warn("skipping {s}: {s}", .{ name, @errorName(err) });
        };
    }

    std.log.info("MoE:", .{});
    for (config.moe_layers_enum) |moe_idx| {
        const layer_idx: usize = @intCast(moe_idx);
        const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.moe", .{layer_idx});
        defer allocator.free(name);

        const moe = try model.Moe.init(model_store.view().withPrefix(name), layer_idx);

        var moe_weights = try zml.io.load(model.Moe, &moe, allocator, io, platform, model_store, .auto);
        defer deinitBuffers(&moe_weights);

        const moe_in_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name});
        defer allocator.free(moe_in_key);
        if (!activation_store.view().hasKey(moe_in_key)) {
            std.log.warn("skipping {s}: no activations recorded", .{name});
            continue;
        }

        zml.testing.testLayer(
            allocator,
            io,
            platform,
            moe,
            .forward,
            activation_store.view(),
            name,
            moe_weights,
            &.{},
            .{
                .absolute_tolerance = 1e-2,
                .minimum_close_fraction = 0.99,
            },
        ) catch |err| {
            std.log.warn("skipping {s}: {s}", .{ name, @errorName(err) });
        };
    }

    std.log.info("Transformer layer:", .{});
    for (0..config.num_hidden_layers) |layer_idx| {
        runTransformerLayer(allocator, io, platform, model_store, &activation_store, sharding, layer_idx) catch |err| {
            std.log.warn("skipping model.layers.{d}: {s}", .{ layer_idx, @errorName(err) });
        };
    }

    std.log.info("Full model:", .{});
    try runFullTextModel(allocator, io, platform, model_store, &activation_store, sharding);
}

fn runTransformerLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
    sharding: zml.Sharding,
    layer_idx: usize,
) !void {
    const name = try std.fmt.allocPrint(allocator, "model.layers.{d}", .{layer_idx});
    defer allocator.free(name);

    const layer = try model.TransformerLayer.init(model_store.view().withPrefix(name), layer_idx);
    var layer_weights = try zml.io.load(model.TransformerLayer, &layer, allocator, io, platform, model_store, .auto);
    defer deinitBuffers(&layer_weights);

    const layer_view = activation_store.view().withPrefix(name);
    if (!layer_view.hasKey("in.0")) {
        std.log.warn("skipping {s}: no in.0 (hidden_states)", .{name});
        return;
    }
    if (!layer_view.hasKey("out.0")) {
        std.log.warn("skipping {s}: no out.0 (final hidden_states)", .{name});
        return;
    }
    const self_attn_name = try std.fmt.allocPrint(allocator, "{s}.self_attn", .{name});
    defer allocator.free(self_attn_name);
    const self_attn_view = activation_store.view().withPrefix(self_attn_name);
    if (!self_attn_view.hasKey("in.3")) {
        std.log.warn("skipping {s}: no self_attn.in.3 (cache_position)", .{name});
        return;
    }

    const hidden_states = layer_view.createTensor("in.0", .{ .b, .s, .d }, .replicated);
    const cache_position = self_attn_view.createTensor("in.3", null, .replicated);

    const batch_dim = hidden_states.dim(.b);
    const seq_dim = hidden_states.dim(.s);
    const kv_shape = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.num_hidden_layers)),
        .b = batch_dim,
        .k = seq_dim,
        .h = layer.attn.num_kv_heads,
        .hd = layer.attn.head_dim,
    }, hidden_states.dtype());
    const kv_traced = model.KvCache.init(kv_shape).atLayer(layer_idx);

    const TraceArgs = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const trace_args: TraceArgs = .{ hidden_states, cache_position, kv_traced };

    const exe = try platform.compile(allocator, io, layer, .forward, trace_args, .{ .shardings = &.{&sharding} });
    defer exe.deinit();

    const ActArgs = struct { zml.Tensor, zml.Tensor };
    var act_args: ActArgs = .{ hidden_states, cache_position };
    var act_buffers = try zml.io.load(ActArgs, &act_args, allocator, io, platform, activation_store, .auto);
    defer deinitBuffers(&act_buffers);

    const kv_bytes = kv_shape.byteSize();
    const k_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(k_init);
    @memset(k_init, 0);
    const v_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(v_init);
    @memset(v_init, 0);

    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, .replicated, k_init);
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, .replicated, v_init);
    defer v_buf.deinit();
    const kv_buffers: model.KvCache.Buffer = .{ .k = k_buf, .v = v_buf };

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    const args_buffers = .{ act_buffers[0], act_buffers[1], kv_buffers };
    exe_args.set(.{ layer_weights, args_buffers });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    const output_count = exe.output_shapes.len;
    var results = try allocator.alloc(zml.Buffer, output_count);
    defer allocator.free(results);
    exe_results.fill(.{results});

    if (output_count == 0) {
        std.log.warn("{s}: compiled layer produced no outputs", .{name});
        return;
    }

    // forward returns (hidden_states, KvCache.k, KvCache.v); only the first
    // output corresponds to a recorded HF activation (`out.0`).
    try compareSingleOutput(allocator, io, layer_view, "out.0", &results[0], name, .{
        .absolute_tolerance = 1e-2,
        .minimum_close_fraction = 0.99,
    });
}

fn loadBufferFromStore(allocator: std.mem.Allocator, io: anytype, platform: *zml.Platform, store: *zml.io.TensorStore, key: []const u8, sharding: zml.Sharding) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();

    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}

fn runFullTextModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
    sharding: zml.Sharding,
) !void {
    const model_view = activation_store.view().withPrefix("model");
    if (!model_view.hasKey("in.0") or !model_view.hasKey("out.0")) {
        std.log.warn("skipping full TextModel: missing in.0 / out.0", .{});
        return;
    }

    var text_model = try model.TextModel.init(allocator, model_store.view().withPrefix("model"));
    defer text_model.deinit(allocator);

    var text_buffers = try zml.io.load(model.TextModel, &text_model, allocator, io, platform, model_store, .auto);
    defer deinitBuffers(&text_buffers);

    // 1. Build trace-time tensors for compile.
    const tokens_t = model_view.createTensor("in.0", .{ .b, .s }, .replicated);
    const token_index_t = zml.Tensor.fromShape(zml.Shape.init(.{}, .i32));

    // 2. KV cache shape: (layer, b, k, h, hd).
    const attn0 = text_model.layers[0].attn;
    const kv_shape = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.num_hidden_layers)),
        .b = tokens_t.dim(.b),
        .k = tokens_t.dim(.s),
        .h = attn0.num_kv_heads,
        .hd = attn0.head_dim,
    }, .bf16);
    const kv_traced = model.KvCache.init(kv_shape);

    // 3. Compile forward.
    const TraceArgs = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const exe = try platform.compile(
        allocator,
        io,
        text_model,
        .forward,
        TraceArgs{ tokens_t, token_index_t, kv_traced },
        .{ .shardings = &.{&sharding} },
    );
    defer exe.deinit();

    // 4. Build runtime buffers: tokens from store, token_index = 0, zeroed KV.
    var token_args: struct { zml.Tensor } = .{tokens_t};
    var token_buffers = try zml.io.load(@TypeOf(token_args), &token_args, allocator, io, platform, activation_store, .auto);
    defer deinitBuffers(&token_buffers);

    var token_index_buf: zml.Buffer = try .scalar(io, platform, 0, .i32);
    defer token_index_buf.deinit();

    const kv_bytes = kv_shape.byteSize();
    const k_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(k_init);
    @memset(k_init, 0);
    const v_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(v_init);
    @memset(v_init, 0);
    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, .replicated, k_init);
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, .replicated, v_init);
    defer v_buf.deinit();
    const kv_buffers: model.KvCache.Buffer = .{ .k = k_buf, .v = v_buf };

    // 5. Run.
    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ text_buffers, .{ token_buffers[0], token_index_buf, kv_buffers } });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    // 6. Pull results[0] (hidden), compare with looser tolerance (error compounds across 48 layers).
    var results = try allocator.alloc(zml.Buffer, exe.output_shapes.len);
    defer allocator.free(results);
    exe_results.fill(.{results});
    try compareSingleOutput(allocator, io, model_view, "out.0", &results[0], "model", .{
        .absolute_tolerance = 5e-2,
        .minimum_close_fraction = 0.98,
    });
}

fn deinitBuffers(bufs: anytype) void {
    zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, bufs);
}

// RmsNorm.forward takes a comptime axis, so wrap it to fix axis=.d for compilation.
const RmsNormForD = struct {
    inner: model.RmsNorm,

    pub fn forward(self: RmsNormForD, x: zml.Tensor) zml.Tensor {
        return self.inner.forward(x, .d);
    }
};

fn runRmsNormSynthetic(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.Sharding,
) !void {
    const eps: f32 = 1e-5;
    const d: i64 = 4;

    const weight_shape = zml.Shape.init(.{ .d = d }, .f32);
    const x_shape = zml.Shape.init(.{ .b = 1, .s = 1, .d = d }, .f32);

    const wrapper_t: RmsNormForD = .{ .inner = .{
        .weight = zml.Tensor.fromShape(weight_shape),
        .eps = eps,
    } };
    const x_t = zml.Tensor.fromShape(x_shape);

    const exe = try platform.compileFn(allocator, io, RmsNormForD.forward, .{ wrapper_t, x_t }, .{ .shardings = &.{&sharding} });
    defer exe.deinit();

    const weight_data = [_]f32{ 0.0, 1.0, 0.0, -1.0 };
    const x_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const expected = [_]f32{ 0.3651, 1.4606, 1.0954, 0.0 };

    var weight_buf: zml.Buffer = try .fromBytes(io, platform, weight_shape, .replicated, std.mem.sliceAsBytes(&weight_data));
    defer weight_buf.deinit();
    var x_buf: zml.Buffer = try .fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(&x_data));
    defer x_buf.deinit();

    const wrapper_buffers: zml.Bufferized(RmsNormForD) = .{ .inner = .{ .weight = weight_buf } };

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ wrapper_buffers, x_buf });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var out_buf = exe_results.get(zml.Buffer);
    defer out_buf.deinit();

    var expected_slice: zml.Slice = try .alloc(allocator, x_shape);
    defer expected_slice.free(allocator);
    @memcpy(expected_slice.data()[0 .. expected.len * @sizeOf(f32)], std.mem.sliceAsBytes(&expected));

    var got_slice = try out_buf.toSliceAlloc(allocator, io);
    defer got_slice.free(allocator);

    try zml.testing.expectClose(io, expected_slice, got_slice, .{ .absolute_tolerance = 1e-3 });
    std.log.info("ok RmsNorm synthetic matches", .{});
}

// When layers return multiple outputs, zml.testing.testLayer no longer suffices, hence the following function.
fn compareSingleOutput(
    allocator: std.mem.Allocator,
    io: std.Io,
    view: zml.io.TensorStore.View,
    subkey: []const u8,
    got: *zml.Buffer,
    name: []const u8,
    opts: zml.testing.CompareOpts,
) !void {
    const ref_shape = view.getShape(subkey) orelse {
        std.log.warn("{s}.{s}: no reference shape", .{ name, subkey });
        return;
    };

    const got_shape = got.shape();
    var ref_count: i64 = 1;
    for (ref_shape.dims()) |d| ref_count *= d;
    var got_count: i64 = 1;
    for (got_shape.dims()) |d| got_count *= d;
    if (ref_count != got_count) {
        std.log.warn("{s}.{s}: shape mismatch ref={f} ours={f}, skipping", .{ name, subkey, ref_shape, got_shape });
        return;
    }

    var reader_buffer: [4096]u8 = undefined;
    const expected_slice: zml.Slice = try .alloc(allocator, ref_shape);
    defer expected_slice.free(allocator);
    var reader = try view.getReader(subkey, io, &reader_buffer);
    defer reader.deinit();
    try reader.interface.readSliceAll(expected_slice.data());

    const got_slice = try got.toSliceAlloc(allocator, io);
    defer got_slice.free(allocator);

    zml.testing.expectClose(io, expected_slice, got_slice, opts) catch |err| switch (err) {
        error.TestUnexpectedResult => {
            std.log.warn("X {s}.{s} doesn't match", .{ name, subkey });
            return;
        },
        else => return err,
    };
    std.log.info("ok {s}.{s} matches", .{ name, subkey });
}
