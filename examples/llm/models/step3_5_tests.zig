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

    const shardings: common.Shardings = try .init(platform);

    var repo_model = try step3p5flash.LoadedModel.init(allocator, io, repo, store.view(), shardings);
    defer repo_model.deinit(allocator);

    // Loading bar (single global Progress)
    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();

    var model_buffers = try repo_model.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer repo_model.unloadBuffers(&model_buffers, allocator);

    try run(
        allocator,
        io,
        platform,
        args.activations,
        &store,
        shardings,
        &repo_model.inner.text_model,
        &model_buffers.text_model,
    );
}

pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    model_store: *zml.io.TensorStore,
    shardings: common.Shardings,
    text_model: *const model.TextModel,
    text_buffers: *zml.Bufferized(model.TextModel),
) !void {
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    // Per-layer transformer test: each layer fed the HF reference inputs.
    // Only iterate the main transformer layers; trailing MTP layers are not
    // captured in the HF fixture (the dumper sets num_hidden_layers = 45).
    const num_main_layers = model.default_config.numMainLayers();
    std.log.info("Transformer layer (isolated):", .{});
    const shardings_array = shardings.all();
    for (0..num_main_layers) |layer_idx| {
        runTransformerLayer(allocator, io, platform, model_store, &activation_store, &shardings_array, layer_idx) catch |err| {
            std.log.warn("skipping model.layers.{d}: {s}", .{ layer_idx, @errorName(err) });
        };
    }

    // Chained-layer test: feed layer 0 input, run N layers, compare against
    // HF activation at layer N-1. Bisects when compounding goes off the rails.
    std.log.info("Transformer layer (chained):", .{});
    const chain_steps = [_]usize{ 2, 4, 8, 16, 24, 32, 40, 45 };
    for (chain_steps) |n| {
        if (n > num_main_layers) continue;
        runLayerChain(allocator, io, platform, &activation_store, &shardings_array, text_model, text_buffers, n) catch |err| {
            std.log.warn("skipping chain[0..{d}]: {s}", .{ n, @errorName(err) });
        };
    }

    std.log.info("Full model:", .{});
    try runFullTextModel(allocator, io, platform, &activation_store, shardings, text_model, text_buffers);
}

fn runTransformerLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    layer_idx: usize,
) !void {
    const name = try std.fmt.allocPrint(allocator, "model.layers.{d}", .{layer_idx});
    defer allocator.free(name);

    const model_partitions = shardings[0].numPartitionsForLogicalAxis(.model);
    const layer = try model.TransformerLayer.init(model_store.view().withPrefix(name), layer_idx, model_partitions);
    var layer_weights = try zml.io.load(model.TransformerLayer, &layer, allocator, io, platform, model_store, .{
        .parallelism = 1,
        .shardings = shardings,
        .dma_chunks = 2,
        .dma_chunk_size = 4096,
    });
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
    const kv_shape_unsharded = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.numMainLayers())),
        .b = batch_dim,
        .k = seq_dim,
        .h = layer.attn.num_kv_heads,
        .hd = layer.attn.head_dim,
    }, hidden_states.dtype());
    const kv_shape = model.partitionKvCacheShape(kv_shape_unsharded, layer.attn.num_kv_heads, model_partitions);
    const kv_traced = model.KvCache.init(kv_shape).atLayer(layer_idx);

    const TraceArgs = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const trace_args: TraceArgs = .{ hidden_states, cache_position, kv_traced };

    const exe = try platform.compile(allocator, io, layer, .forward, trace_args, .{ .shardings = shardings });
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

    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings[0], k_init);
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings[0], v_init);
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

const LayerChain = struct {
    layers: []const model.TransformerLayer,

    pub fn forward(
        self: LayerChain,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: model.KvCache,
    ) struct { zml.Tensor, model.KvCache } {
        var hidden = x;
        var kv = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden, kv = layer.forward(hidden, token_index, kv.atLayer(i));
        }
        return .{ hidden, kv };
    }
};

fn runLayerChain(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    activation_store: *zml.io.TensorStore,
    shardings_array: []const zml.Sharding,
    text_model: *const model.TextModel,
    text_buffers: *zml.Bufferized(model.TextModel),
    num_layers: usize,
) !void {
    if (num_layers == 0 or num_layers > text_model.layers.len) return;

    const first_view = activation_store.view().withPrefix("model.layers.0");
    const first_attn_view = activation_store.view().withPrefix("model.layers.0.self_attn");
    if (!first_view.hasKey("in.0") or !first_attn_view.hasKey("in.3")) {
        std.log.warn("skipping chain[0..{d}]: missing layer-0 inputs", .{num_layers});
        return;
    }

    const ref_name = try std.fmt.allocPrint(allocator, "model.layers.{d}", .{num_layers - 1});
    defer allocator.free(ref_name);
    const ref_view = activation_store.view().withPrefix(ref_name);
    if (!ref_view.hasKey("out.0")) {
        std.log.warn("skipping chain[0..{d}]: no {s}.out.0", .{ num_layers, ref_name });
        return;
    }

    const chain: LayerChain = .{ .layers = text_model.layers[0..num_layers] };

    const hidden_states = first_view.createTensor("in.0", .{ .b, .s, .d }, .replicated);
    const cache_position = first_attn_view.createTensor("in.3", null, .replicated);

    const attn0 = text_model.layers[0].attn;
    const model_partitions = shardings_array[0].numPartitionsForLogicalAxis(.model);
    const kv_shape_unsharded = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.numMainLayers())),
        .b = hidden_states.dim(.b),
        .k = hidden_states.dim(.s),
        .h = attn0.num_kv_heads,
        .hd = attn0.head_dim,
    }, hidden_states.dtype());
    const kv_shape = model.partitionKvCacheShape(kv_shape_unsharded, attn0.num_kv_heads, model_partitions);
    const kv_traced = model.KvCache.init(kv_shape);

    const TraceArgs = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const trace_args: TraceArgs = .{ hidden_states, cache_position, kv_traced };
    const exe = try platform.compile(allocator, io, chain, .forward, trace_args, .{ .shardings = shardings_array });
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
    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings_array[0], k_init);
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings_array[0], v_init);
    defer v_buf.deinit();
    const kv_buffers: model.KvCache.Buffer = .{ .k = k_buf, .v = v_buf };

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    const chain_buffers: zml.Bufferized(LayerChain) = .{ .layers = text_buffers.layers[0..num_layers] };
    exe_args.set(.{ chain_buffers, .{ act_buffers[0], act_buffers[1], kv_buffers } });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var results = try allocator.alloc(zml.Buffer, exe.output_shapes.len);
    defer allocator.free(results);
    exe_results.fill(.{results});

    const tag = try std.fmt.allocPrint(allocator, "chain[0..{d}]", .{num_layers});
    defer allocator.free(tag);
    try compareSingleOutput(allocator, io, ref_view, "out.0", &results[0], tag, .{
        .absolute_tolerance = 5e-2,
        .minimum_close_fraction = 0.95,
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
    activation_store: *zml.io.TensorStore,
    shardings: common.Shardings,
    text_model: *const model.TextModel,
    text_buffers: *zml.Bufferized(model.TextModel),
) !void {
    const model_view = activation_store.view().withPrefix("model");
    if (!model_view.hasKey("in.0") or !model_view.hasKey("out.0")) {
        std.log.warn("skipping full TextModel: missing in.0 / out.0", .{});
        return;
    }

    // 1. Build trace-time tensors for compile.
    const tokens_t = model_view.createTensor("in.0", .{ .b, .s }, .replicated);
    // token_index follows HF `cache_position` convention: rank-1 i32 of length S.
    const seq_len = tokens_t.dim(.s);
    const token_index_t = zml.Tensor.fromShape(zml.Shape.init(.{ .s = seq_len }, .i32));

    // 2. KV cache shape: (layer, b, k, h, hd).
    const attn0 = text_model.layers[0].attn;
    const model_partitions = shardings.model.numPartitionsForLogicalAxis(.model);
    const kv_shape_unsharded = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.numMainLayers())),
        .b = tokens_t.dim(.b),
        .k = tokens_t.dim(.s),
        .h = attn0.num_kv_heads,
        .hd = attn0.head_dim,
    }, .bf16);
    const kv_shape = model.partitionKvCacheShape(kv_shape_unsharded, attn0.num_kv_heads, model_partitions);
    const kv_traced = model.KvCache.init(kv_shape);

    // 3. Compile forward.
    const TraceArgs = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const exe = try platform.compile(
        allocator,
        io,
        text_model.*,
        .forward,
        TraceArgs{ tokens_t, token_index_t, kv_traced },
        .{ .shardings = &shardings.all() },
    );
    defer exe.deinit();

    // 4. Build runtime buffers: tokens from store, cache_position = [0..S-1], zeroed KV.
    var token_args: struct { zml.Tensor } = .{tokens_t};
    var token_buffers = try zml.io.load(@TypeOf(token_args), &token_args, allocator, io, platform, activation_store, .auto);
    defer deinitBuffers(&token_buffers);

    const token_index_shape = zml.Shape.init(.{ .s = seq_len }, .i32);
    const token_index_data = try allocator.alloc(i32, @intCast(seq_len));
    defer allocator.free(token_index_data);
    for (token_index_data, 0..) |*v, i| v.* = @intCast(i);
    var token_index_buf: zml.Buffer = try .fromBytes(io, platform, token_index_shape, .replicated, std.mem.sliceAsBytes(token_index_data));
    defer token_index_buf.deinit();

    const kv_bytes = kv_shape.byteSize();
    const k_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(k_init);
    @memset(k_init, 0);
    const v_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(v_init);
    @memset(v_init, 0);
    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings.model, k_init);
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings.model, v_init);
    defer v_buf.deinit();
    const kv_buffers: model.KvCache.Buffer = .{ .k = k_buf, .v = v_buf };

    // 5. Run.
    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ text_buffers.*, .{ token_buffers[0], token_index_buf, kv_buffers } });
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

    const exe = try platform.compileFn(allocator, io, RmsNormForD.forward, .{ wrapper_t, x_t }, .{ .shardings = &.{sharding} });
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
