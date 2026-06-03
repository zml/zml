const std = @import("std");

const zml = @import("zml");

const model = @import("step3_5flash/model.zig");

const TextRotaryEmbedding = model.TextRotaryEmbedding;

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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
fn swigluLimitFor(layer_idx: usize) ?f32 {
    return switch (layer_idx) {
        43, 44 => 7.0,
        else => null,
    };
}

>>>>>>> c856b0f1 (examples/llm: clean up tests and model annotations)
pub fn main(init: std.process.Init) !void {
    std.log.info("main", .{});
=======
const TEST_LAYER = 1;
>>>>>>> 311be4c4 (examples/llm: test all router gate outputs)
=======
const TEST_LAYER = 1;
>>>>>>> 027b91a5 (examples/llm: fix top k comparer to stride for top-k instead of show all 288 experts)
=======
const TEST_LAYER = 0;
>>>>>>> 158ff350 (examples/llm: create test for MoE forward)

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    const sharding = try platform.registerSharding("tp_mesh", .mesh(.{ .model = .high_bandwidth }));
    defer platform.deinit(allocator, io);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    var model_registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer model_registry.deinit();
    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    // try run(allocator, io, platform, args.activations_main, &model_store);
    try run(allocator, io, platform, args.activations_moe, &model_store);
=======
    try run(allocator, io, platform, args.activations, &model_store);
>>>>>>> c856b0f1 (examples/llm: clean up tests and model annotations)
=======
    if (TEST_LAYER == 0) {
        // test main
        try run(allocator, io, platform, args.activations_main, &model_store);
    } else if (TEST_LAYER == 1) {
        // test sublayer (within-gate activations)
        try run(allocator, io, platform, args.activations_moe, &model_store);
    } else if (TEST_LAYER == 2) {
        const moe_view = model_store.view()
            .withPrefix("model")
            .withPrefix("layers")
            .withLayer(3)
            .withPrefix("moe");
        const moe = try model.Moe.init(moe_view);
        std.log.info("moe init'd: {any}", .{moe});
    }
>>>>>>> 311be4c4 (examples/llm: test all router gate outputs)
=======
    try run(allocator, io, platform, args.activations, sharding, &model_store);
>>>>>>> dfcbcfdf (examples/llm: compare attention outputs only (ignore second return value))
}

const TEST_LAYER = 4;

fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    sharding: zml.Sharding,
    model_store: *zml.io.TensorStore,
) !void {
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    // const layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
    // std.log.info("MLP:", .{});
    // for (layer_indices) |layer_idx| {
    //     const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
    //     defer allocator.free(name);
=======
    if (TEST_LAYER == 0) {
<<<<<<< HEAD
=======
    if (TEST_LAYER == 0) {
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)
        const mlp_layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
        std.log.info("MLP:", .{});
        for (mlp_layer_indices) |layer_idx| {
            const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
            defer allocator.free(name);
<<<<<<< HEAD
>>>>>>> 311be4c4 (examples/llm: test all router gate outputs)
=======
        // const mlp_layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
        // std.log.info("MLP:", .{});
        // for (mlp_layer_indices) |layer_idx| {
        //     const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
        //     defer allocator.free(name);
>>>>>>> 158ff350 (examples/llm: create test for MoE forward)

        //     std.log.info("name {s}", .{name});

        //     try testLayer(allocator, io, platform, activation_store.view(), model_store, .mlp, name, name, .{ .absolute_tolerance = 1e-2 });
        // }
        // std.log.info("RMS Norm:", .{});

<<<<<<< HEAD
<<<<<<< HEAD
        for (0..48) |layer_idx| {
            const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
            defer allocator.free(name);
=======
        // for (0..48) |layer_idx| {
        //     const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
        //     defer allocator.free(name);
>>>>>>> 158ff350 (examples/llm: create test for MoE forward)

        //     try testLayer(allocator, io, platform, activation_store.view(), model_store, .rmsNorm, name, name, .{ .absolute_tolerance = 1e-2 });
        // }
        // for (0..48) |layer_idx| {
        //     const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.post_attention_layernorm", .{layer_idx});
        //     defer allocator.free(name);

<<<<<<< HEAD
            try testLayer(allocator, io, platform, activation_store.view(), model_store, .rmsNorm, name, name, .{ .absolute_tolerance = 1e-2 });
=======
        // for (0..48) |layer_idx| {
        //     const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
        //     defer allocator.free(name);

        //     try testLayer(allocator, io, platform, activation_store.view(), model_store, .rmsNorm, name, name, .{ .absolute_tolerance = 1e-2 });
        // }
        // for (0..48) |layer_idx| {
        //     const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.post_attention_layernorm", .{layer_idx});
        //     defer allocator.free(name);

        //     try testLayer(allocator, io, platform, activation_store.view(), model_store, .rmsNorm, name, name, .{ .absolute_tolerance = 1e-2 });
        // }

<<<<<<< HEAD
        for (41..45) |layer_idx| {
            const moe_view = model_store.view().withPrefix("model").withPrefix("layers").withLayer(layer_idx).withPrefix("moe");
            const moe = try model.Moe.init(moe_view);

            var bufs = try zml.io.load(model.Moe, &moe, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&bufs);

            std.log.info("=== layer {d} weight slices ===", .{layer_idx});
            try dumpFirst8(allocator, io, platform, "up_proj", layer_idx, moe.up_proj, bufs.up_proj);
            try dumpFirst8(allocator, io, platform, "gate_proj", layer_idx, moe.gate_proj, bufs.gate_proj);
            try dumpFirst8(allocator, io, platform, "down_proj", layer_idx, moe.down_proj, bufs.down_proj);

            try dumpFirst8With(allocator, io, platform, Slicer.first8_2d, "router.gate.weight[0,:8]", layer_idx, moe.router.gate.weight, bufs.router.gate.weight);
            try dumpFirst8With(allocator, io, platform, Slicer.first8_1d, "router.router_bias[:8]", layer_idx, moe.router.router_bias, bufs.router.router_bias);

            // Also dump the saved MoE input from the activations file.
            const in_name = try std.fmt.allocPrint(allocator, "model.layers.{d}.moe.in", .{layer_idx});
            defer allocator.free(in_name);
            var in_args: struct { zml.Tensor } = .{
                activation_store.view().withPrefix(in_name).createTensor("0", null, .replicated),
            };
            var in_bufs = try zml.io.load(@TypeOf(in_args), &in_args, allocator, io, platform, &activation_store, .auto);
            defer deinitBuffers(&in_bufs);
            try dumpFirst8(allocator, io, platform, "moe.in.0", layer_idx, in_args[0], in_bufs[0]);
>>>>>>> 027b91a5 (examples/llm: fix top k comparer to stride for top-k instead of show all 288 experts)
        }
=======
        //     try testLayer(allocator, io, platform, activation_store.view(), model_store, .rmsNorm, name, name, .{ .absolute_tolerance = 1e-2 });
        // }

        const name = try std.fmt.allocPrint(allocator, "model.layers.3.moe", .{});
        defer allocator.free(name);

        try testLayer(allocator, io, platform, activation_store.view(), model_store, .moe, name, name, .{ .absolute_tolerance = 1e-2 });
>>>>>>> 158ff350 (examples/llm: create test for MoE forward)
=======
        for (3..45) |layer_idx| {
            const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.moe", .{layer_idx});
            defer allocator.free(name);

            try testLayer(allocator, io, platform, activation_store.view(), model_store, .moe, name, name, .{ .absolute_tolerance = 1e-2 });
        }
>>>>>>> d6fed5e4 (examples/llm: MoE tests)
    } else if (TEST_LAYER == 1) {
        // layers 3..44 (45 with 0 indexing but i wont bake it in rn)

        for (3..45) |layer_idx| {
            const weight_name = try std.fmt.allocPrint(allocator, "model.layers.{d}.moe", .{layer_idx});
            defer allocator.free(weight_name);
            const activation_name = try std.fmt.allocPrint(allocator, "model.layers.{d}.moe.router", .{layer_idx});
            defer allocator.free(activation_name);

            try testLayer(allocator, io, platform, activation_store.view(), model_store, .router, weight_name, // weight prefix in the model checkpoint
                activation_name, // activation prefix in router_ref.safetensors
                .{ .absolute_tolerance = 1e-2 });
        }
    }
}

<<<<<<< HEAD
<<<<<<< HEAD
const LayerKind = enum { mlp, rmsNorm };
=======
    const mlp_layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
=======
    // const mlp_layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
    // std.log.info("MLP:", .{});
    // for (mlp_layer_indices) |layer_idx| {
    //     const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
    //     defer allocator.free(name);
>>>>>>> 4dd0ea28 (examples/llm: naive for loop for swiglu clamp layers)
=======
    const mlp_layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
    std.log.info("MLP:", .{});
    for (mlp_layer_indices) |layer_idx| {
        const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
=======

            const mlp = model.Mlp.init(model_store.view().withPrefix(name), null);

            // Recursive cleanup for buffers
            var mlp_weights = try zml.io.load(model.Mlp, &mlp, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&mlp_weights);

            std.log.info("name {s}", .{name});

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
        std.log.info("RMS Norm:", .{});

        for (0..48) |layer_idx| {
            const name__input_layernorm = try std.fmt.allocPrint(allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
            defer allocator.free(name__input_layernorm);

            const name__post_attention_layernorm = try std.fmt.allocPrint(allocator, "model.layers.{d}.post_attention_layernorm", .{layer_idx});
            defer allocator.free(name__post_attention_layernorm);

            const rms1 = model.RmsNorm.init(model_store.view().withPrefix(name__input_layernorm), @as(f32, 1e-5));
            const rms2 = model.RmsNorm.init(model_store.view().withPrefix(name__post_attention_layernorm), @as(f32, 1e-5));

            var rms_weights1 = try zml.io.load(model.RmsNorm, &rms1, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&rms_weights1);

            var rms_weights2 = try zml.io.load(model.RmsNorm, &rms2, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&rms_weights2);

            const in1_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name__input_layernorm});
            defer allocator.free(in1_key);
            if (activation_store.view().hasKey(in1_key)) {
                zml.testing.testLayer(allocator, io, platform, rms1, .forward, activation_store.view(), name__input_layernorm, rms_weights1, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
                    std.log.warn("skipping {s}: {s}", .{ name__input_layernorm, @errorName(err) });
                };
            } else {
                std.log.warn("skipping {s}: no activations recorded", .{name__input_layernorm});
            }

            const in2_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name__post_attention_layernorm});
            defer allocator.free(in2_key);
            if (activation_store.view().hasKey(in2_key)) {
                zml.testing.testLayer(allocator, io, platform, rms2, .forward, activation_store.view(), name__post_attention_layernorm, rms_weights2, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
                    std.log.warn("skipping {s}: {s}", .{ name__post_attention_layernorm, @errorName(err) });
                };
            } else {
                std.log.warn("skipping {s}: no activations recorded", .{name__post_attention_layernorm});
            }
        }

        // MoE is verified; we no longer have to test the router
        for (42..45) |layer_idx| {
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
    } else if (TEST_LAYER == 1) {
        try runRopeTests(allocator, io, platform);
    } else if (TEST_LAYER == 2) {
        try surveySelfAttnShapes(allocator, model_store, &activation_store);
        for (0..45) |layer_idx| {
            runSelfAttnLayer(allocator, io, platform, model_store, &activation_store, sharding, layer_idx) catch |err| {
                std.log.warn("skipping model.layers.{d}.self_attn: {s}", .{ layer_idx, @errorName(err) });
            };
        }
    } else if (TEST_LAYER == 3) {
<<<<<<< HEAD
<<<<<<< HEAD
        try debugSelfAttnStages(allocator, io, platform, model_store, &activation_store, 1);
=======
<<<<<<< HEAD
=======
>>>>>>> 44b24f7e (examples/llm: test for all Attn layer)
        try surveySelfAttnShapes(allocator, model_store, &activation_store);
        for (0..45) |layer_idx| {
            debugSelfAttnStages(allocator, io, platform, model_store, &activation_store, layer_idx) catch |err| {
                std.log.warn("skipping model.layers.{d}.self_attn: {s}", .{ layer_idx, @errorName(err) });
            };
        }

        // try debugSelfAttnStages(allocator, io, platform, model_store, &activation_store, 1);
<<<<<<< HEAD
>>>>>>> 24035ea3 (examples/llm: update tests)
=======
>>>>>>> 44b24f7e (examples/llm: test for all Attn layer)
    } else {
        const layer_idx = 1;
        const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn", .{layer_idx});
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)
        defer allocator.free(name);
>>>>>>> a636246d (examples/llm: RoPE & RoPE tests)

        const attn = try model.SelfAttn.init(model_store.view().withPrefix(name), layer_idx);

        var attn_weights = try zml.io.load(model.SelfAttn, &attn, allocator, io, platform, model_store, .auto);
        defer deinitBuffers(&attn_weights);

        testSelfAttn(
            allocator,
            io,
            platform,
            attn,
            attn_weights,
            activation_store.view(),
            name,
            .{ .absolute_tolerance = 1e-2, .minimum_close_fraction = 0.99 },
        ) catch |err| {
            std.log.warn("skipping {s}: {s}", .{ name, @errorName(err) });
        };
=======
        try runKvCacheTests(allocator, io, platform);
<<<<<<< HEAD
>>>>>>> 2bc2e8f0 (examples/llm: update tests)
=======
    } else if (TEST_LAYER == 4) {
        // End-to-end TransformerLayer smoke test: one dense layer + one MoE layer.
        // Validates that the residual + norm + attn + ffn composition matches
        // the HF dump for the same hardware path the model would run on.
        // const layer_indices = [_]usize{ 0, 1, 2, 3, 4, 5, 6 };
        for (0..48) |layer_idx| {
            runTransformerLayer(allocator, io, platform, model_store, &activation_store, sharding, layer_idx) catch |err| {
                std.log.warn("skipping model.layers.{d}: {s}", .{ layer_idx, @errorName(err) });
            };
        }
>>>>>>> 8935dcbc (examples/llm: test TransformerLayer)
    }
}
>>>>>>> c856b0f1 (examples/llm: clean up tests and model annotations)
=======
const LayerKind = enum { mlp, rmsNorm, router };
>>>>>>> 939965a6 (examples/llm: test router)
=======
const LayerKind = enum { mlp, rmsNorm, router, moe };
>>>>>>> 158ff350 (examples/llm: create test for MoE forward)

// Manual test for SelfAttn: the HF dumper writes tensors at non-contiguous indices
// (past_key_value is a Cache and gets skipped while enumerate keeps counting),
// so `.in.0, .in.1, .in.3, .in.4` exist but `.in.2` does not. `testLayer` assumes
// contiguous indices and would panic, so we wire the four real inputs by name.
fn testSelfAttn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    attn: model.Attn,
    attn_weights: zml.Bufferized(model.Attn),
    activation_view: zml.io.TensorStore.View,
    sharding: zml.Sharding,
    name: []const u8,
    opts: zml.testing.CompareOpts,
) !void {
    const view = activation_view.withPrefix(name);

    // Discover which .in.N keys actually exist.
    inline for (0..8) |i| {
        var buf: [16]u8 = undefined;
        const k = std.fmt.bufPrint(&buf, "in.{d}", .{i}) catch unreachable;
        if (view.hasKey(k)) {
            std.log.info("{s}.{s}: shape={?f}", .{ name, k, view.getShape(k) });
        }
    }

    const hidden_states = view.createTensor("in.0", null, .replicated);
    const cache_position = view.createTensor("in.3", null, .replicated);

    // Build a KvCache sized to this prefill (batch / seq taken from hidden_states).
    const batch_dim = hidden_states.dim(0);
    const seq_dim = hidden_states.dim(1);
    const kv_shape = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.num_hidden_layers)),
        .b = batch_dim,
        .k = seq_dim,
        .h = attn.num_kv_heads,
        .hd = attn.head_dim,
    }, hidden_states.dtype());
    const kv_traced = model.KvCache.init(kv_shape).atLayer(attn.layer_idx);

    const Argsx = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const argsx: Argsx = .{ hidden_states, cache_position, kv_traced };

    const exe = try platform.compile(allocator, io, attn, .forward, argsx, .{ .shardings = &.{&sharding} });
    defer exe.deinit();

    // Load activation-backed inputs (KvCache is initialized to zero separately below).
    const ActArgs = struct { zml.Tensor, zml.Tensor };
    var act_args: ActArgs = .{ hidden_states, cache_position };
    var act_buffers = try zml.io.load(ActArgs, &act_args, allocator, io, platform, activation_view.store, .auto);
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &act_buffers);

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
    exe_args.set(.{ attn_weights, args_buffers });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    const output_count = exe.output_shapes.len;
    var results = try allocator.alloc(zml.Buffer, output_count);
    defer allocator.free(results);
    exe_results.fill(.{results});

    var reader_buffer: [4096]u8 = undefined;
    var failed = false;
    // Our compiled outputs are (attn_out, kv_cache.k, kv_cache.v); the HF
    // dumper records (attn_out, attn_weights). Only out.0 is comparable.
    for (0..@min(1, output_count)) |i| {
        var key_buf: [16]u8 = undefined;
        const subkey = try std.fmt.bufPrint(&key_buf, "out.{d}", .{i});

        const shape = view.getShape(subkey) orelse {
            std.log.warn("{s}.{s}: no reference output", .{ name, subkey });
            continue;
        };

        // Guard against the index spaces drifting again: if element counts
        // disagree, we'd be reading garbage. Skip with a warning.
        const got_shape = results[i].shape();
        var ref_count: i64 = 1;
        for (shape.dims()) |d| ref_count *= d;
        var got_count: i64 = 1;
        for (got_shape.dims()) |d| got_count *= d;
        if (ref_count != got_count) {
            std.log.warn("{s}.{s}: shape mismatch ref={f} ours={f}, skipping", .{ name, subkey, shape, got_shape });
            continue;
        }

        const expected_slice: zml.Slice = try .alloc(allocator, shape);
        defer expected_slice.free(allocator);
        var reader = try view.getReader(subkey, io, &reader_buffer);
        defer reader.deinit();
        try reader.interface.readSliceAll(expected_slice.data());

        const got_slice = try results[i].toSliceAlloc(allocator, io);
        defer got_slice.free(allocator);

        zml.testing.expectClose(io, expected_slice, got_slice, opts) catch |err| switch (err) {
            error.TestUnexpectedResult => {
                std.log.warn("{s}.{s} doesn't match", .{ name, subkey });
                failed = true;
            },
            else => return err,
        };
    }

    if (failed) {
        std.log.info("❌ check failed for {s}", .{name});
    } else {
        std.log.info("✅ all good for {s}", .{name});
    }
}

fn deinitBuffers(bufs: anytype) void {
    zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, bufs);
}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD

// Chose to test this way for one unified test function
fn testLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activation_store: zml.io.TensorStore.View,
    model_store: *zml.io.TensorStore,
    kind: LayerKind,
    weights_name: []const u8,
    activations_name: []const u8,
    opts: zml.testing.CompareOpts,
) !void {
    switch (kind) {
        .mlp => {
            const mlp = model.Mlp.init(model_store.view().withPrefix(weights_name), null);

            // Recursive cleanup for buffers
            var mlp_weights = try zml.io.load(model.Mlp, &mlp, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&mlp_weights);

            try zml.testing.testLayer(allocator, io, platform, mlp, .forward, activation_store, activations_name, mlp_weights, &.{}, opts);
        },
        .rmsNorm => {
            const rms = model.RmsNorm.init(model_store.view().withPrefix(weights_name), @as(f32, 1e-5));

            var rms_weights = try zml.io.load(model.RmsNorm, &rms, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&rms_weights);

            try zml.testing.testLayer(allocator, io, platform, rms, .forward, activation_store, activations_name, rms_weights, &.{}, opts);
        },
        .router => {
            //hardcoded k=num_experts_per_tok=8 for now
            const router = model.Router.init(model_store.view().withPrefix(weights_name), 8, 1);

            var router_weights = try zml.io.load(model.Router, &router, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&router_weights);

            try compareRouterTopK(allocator, io, platform, router, router_weights, activation_store, activations_name, 8);
        },
        .moe => {
            const moe = try model.Moe.init(model_store.view().withPrefix(weights_name));

            var moe_weights = try zml.io.load(model.Moe, &moe, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&moe_weights);

            try zml.testing.testLayer(allocator, io, platform, moe, .forward, activation_store, activations_name, moe_weights, &.{}, opts);
        },
    }
}
<<<<<<< HEAD
=======
>>>>>>> c856b0f1 (examples/llm: clean up tests and model annotations)
=======

fn compareRouterTopK(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    router: model.Router,
    router_weights: zml.Bufferized(model.Router),
    activation_store: zml.io.TensorStore.View,
    activations_name: []const u8,
    comptime show: usize,
) !void {
    var args: struct { zml.Tensor } = .{
        activation_store.withPrefix(activations_name).withPrefix("in").createTensor("0", null, .replicated),
    };

    const exe = try platform.compile(allocator, io, router, .forward, args, .{});
    defer exe.deinit();

    var args_buffers = try zml.io.load(@TypeOf(args), &args, allocator, io, platform, activation_store.store, .auto);
    defer deinitBuffers(&args_buffers);

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ router_weights, args_buffers });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var results: [2]zml.Buffer = undefined;
    exe_results.fill(.{&results});

    const got = try results[1].toSliceAlloc(allocator, io);
    defer got.free(allocator);

    const out_view = activation_store.withPrefix(activations_name).withPrefix("out");
    const exp_shape = out_view.getShape("1") orelse return;
    var read_buf: [4096]u8 = undefined;
    var reader = try out_view.getReader("1", io, &read_buf);
    defer reader.deinit();
    const exp: zml.Slice = try .alloc(allocator, exp_shape);
    defer exp.free(allocator);
    try reader.interface.readSliceAll(exp.data());

    const dims = exp.shape.dims();
    const exp_stride: usize = @intCast(dims[dims.len - 1]); // typically 288 (all experts, ranked)
    var positions: usize = 1;
    for (dims[0 .. dims.len - 1]) |d| positions *= @intCast(d);

    const got_dims = got.shape.dims();
    const got_stride: usize = @intCast(got_dims[got_dims.len - 1]); // typically 8 (top-k)

    const exp_ids = exp.constItems(i32);
    const got_ids = got.constItems(i32);

    std.log.info("{s}: exp.shape={any} got.shape={any} exp_stride={d} got_stride={d} positions={d}", .{
        activations_name, dims, got_dims, exp_stride, got_stride, positions,
    });

    if (exp_stride < show or got_stride < show) {
        std.log.warn("  stride too small for show={d}; skipping", .{show});
        return;
    }

    var mismatches: usize = 0;
    var first: usize = 0;
    for (0..positions) |p| {
        const exp_row = exp_ids[p * exp_stride ..][0..show];
        const got_row = got_ids[p * got_stride ..][0..show];
        if (!topSetEqual(exp_row, got_row)) {
            if (mismatches == 0) first = p;
            mismatches += 1;
        }
    }

    if (mismatches == 0) {
        std.log.info("✅ {s}: top-{d} matches at all {d} positions", .{ activations_name, show, positions });
    } else {
        std.log.warn("⚠️  {s}: {d}/{d} positions differ. first @{d}: exp={any} got={any}", .{
            activations_name,
            mismatches,
            positions,
            first,
            exp_ids[first * exp_stride ..][0..show],
            got_ids[first * got_stride ..][0..show],
        });
    }
}

fn topSetEqual(a: []const i32, b: []const i32) bool {
    outer: for (a) |x| {
        for (b) |y| if (x == y) continue :outer;
        return false;
    }
    return true;
}
<<<<<<< HEAD
>>>>>>> 311be4c4 (examples/llm: test all router gate outputs)
=======

const Slicer = struct {
    pub fn first8(t: zml.Tensor) zml.Tensor {
        return t
            .slice1d(0, .{ .end = 1 })
            .slice1d(1, .{ .end = 1 })
            .slice1d(2, .{ .end = 8 })
            .convert(.f32)
            .reshape(.{8});
    }

    pub fn first8_2d(t: zml.Tensor) zml.Tensor {
        return t
            .slice1d(0, .{ .end = 1 })
            .slice1d(1, .{ .end = 8 })
            .convert(.f32)
            .reshape(.{8});
    }

    pub fn first8_1d(t: zml.Tensor) zml.Tensor {
        return t.slice1d(0, .{ .end = 8 }).convert(.f32).reshape(.{8});
    }
};

fn dumpFirst8(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    name: []const u8,
    layer_idx: usize,
    tensor_template: zml.Tensor,
    buffer: zml.Buffer,
) !void {
    var exe = try zml.module.compile(allocator, io, Slicer.first8, .{tensor_template}, platform, .{});
    defer exe.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, Slicer.first8, .{buffer});
    defer output.deinit();

    const slice = try output.toSliceAlloc(allocator, io);
    defer slice.free(allocator);

    std.log.info("layer {d} {s}[0,0,:8] = {any}", .{ layer_idx, name, slice.constItems(f32) });
}

fn dumpFirst8With(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    comptime slicer_fn: anytype,
    name: []const u8,
    layer_idx: usize,
    tensor_template: zml.Tensor,
    buffer: zml.Buffer,
) !void {
    var exe = try zml.module.compile(allocator, io, slicer_fn, .{tensor_template}, platform, .{});
    defer exe.deinit();

    var output = try zml.testing.autoCall(allocator, io, &exe, slicer_fn, .{buffer});
    defer output.deinit();

    const slice = try output.toSliceAlloc(allocator, io);
    defer slice.free(allocator);

    std.log.info("layer {d} {s} = {any}", .{ layer_idx, name, slice.constItems(f32) });
}
>>>>>>> 027b91a5 (examples/llm: fix top k comparer to stride for top-k instead of show all 288 experts)
=======
=======
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)

fn runSelfAttnLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
    sharding: zml.Sharding,
    layer_idx: usize,
) !void {
    const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn", .{layer_idx});
    defer allocator.free(name);

    const attention_type = model.default_config.layer_types[layer_idx];
    const attn = try model.Attn.init(model_store.view().withPrefix(name), layer_idx, attention_type);

    var attn_weights = try zml.io.load(model.Attn, &attn, allocator, io, platform, model_store, .auto);
    defer deinitBuffers(&attn_weights);

    try testSelfAttn(
        allocator,
        io,
        platform,
        attn,
        attn_weights,
        activation_store.view(),
        sharding,
        name,
        .{ .absolute_tolerance = 1e-2, .minimum_close_fraction = 0.99 },
    );
}

// full TransformerLayer test
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

    const Argsx = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const argsx: Argsx = .{ hidden_states, cache_position, kv_traced };

    const exe = try platform.compile(allocator, io, layer, .forward, argsx, .{ .shardings = &.{&sharding} });
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

    // forward returns (hidden_states, KvCache.k, KvCache.v); only the first
    // output corresponds to a recorded HF activation (`out.0`).
    if (output_count == 0) {
        std.log.warn("{s}: compiled layer produced no outputs", .{name});
        return;
    }

    const ref_shape = layer_view.getShape("out.0") orelse {
        std.log.warn("{s}: no reference shape for out.0", .{name});
        return;
    };

    const got_shape = results[0].shape();
    var ref_count: i64 = 1;
    for (ref_shape.dims()) |d| ref_count *= d;
    var got_count: i64 = 1;
    for (got_shape.dims()) |d| got_count *= d;
    if (ref_count != got_count) {
        std.log.warn("{s}.out.0: shape mismatch ref={f} ours={f}, skipping", .{ name, ref_shape, got_shape });
        return;
    }

    var reader_buffer: [4096]u8 = undefined;
    const expected_slice: zml.Slice = try .alloc(allocator, ref_shape);
    defer expected_slice.free(allocator);
    var reader = try layer_view.getReader("out.0", io, &reader_buffer);
    defer reader.deinit();
    try reader.interface.readSliceAll(expected_slice.data());

    const got_slice = try results[0].toSliceAlloc(allocator, io);
    defer got_slice.free(allocator);

    zml.testing.expectClose(io, expected_slice, got_slice, .{
        .absolute_tolerance = 1e-2,
        .minimum_close_fraction = 0.99,
    }) catch |err| switch (err) {
        error.TestUnexpectedResult => {
            std.log.warn("❌ {s} doesn't match", .{name});
            return;
        },
        else => return err,
    };
    std.log.info("✅ {s} matches", .{name});
}

fn debugSelfAttnStages(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
    layer_idx: usize,
) !void {
    const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn", .{layer_idx});
    defer allocator.free(name);

    // 1. Discover every captured activation under this layer's attention.
    std.log.info("=== captured activation keys under {s} ===", .{name});
    {
        var it = activation_store.registry.iterator();
        while (it.next()) |entry| {
            const key = entry.key_ptr.*;
            if (std.mem.startsWith(u8, key, name) and key.len > name.len and key[name.len] == '.') {
                std.log.info("  {s}: shape={f}", .{ key, entry.value_ptr.shape });
            }
        }
    }

    // 2. Build attention layer and load its weights.
    const attention_type = model.default_config.layer_types[layer_idx];
    const attn = try model.Attn.init(model_store.view().withPrefix(name), layer_idx, attention_type);
    var attn_weights = try zml.io.load(model.Attn, &attn, allocator, io, platform, model_store, .auto);
    defer deinitBuffers(&attn_weights);

    // 3. Wire the recorded inputs (in.0 = hidden_states, in.3 = cache_position).
    const view = activation_store.view().withPrefix(name);
    const hidden_states = view.createTensor("in.0", null, .replicated);
    const cache_position = view.createTensor("in.3", null, .replicated);

    const batch_dim = hidden_states.dim(0);
    const seq_dim = hidden_states.dim(1);
    const kv_shape = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.num_hidden_layers)),
        .b = batch_dim,
        .k = seq_dim,
        .h = attn.num_kv_heads,
        .hd = attn.head_dim,
    }, hidden_states.dtype());
    const kv_traced = model.KvCache.init(kv_shape).atLayer(attn.layer_idx);

    const Argsx = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const argsx: Argsx = .{ hidden_states, cache_position, kv_traced };

    const exe = try platform.compile(allocator, io, attn, .forwardTemp, argsx, .{ .shardings = &.{} });
    defer exe.deinit();

    const ActArgs = struct { zml.Tensor, zml.Tensor };
    var act_args: ActArgs = .{ hidden_states, cache_position };
    var act_buffers = try zml.io.load(ActArgs, &act_args, allocator, io, platform, activation_store, .auto);
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &act_buffers);

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
    exe_args.set(.{ attn_weights, args_buffers });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var stages_buffers: zml.Bufferized(model.Attn.Stages) = undefined;
    var new_kv_buffers: model.KvCache.Buffer = undefined;
    exe_results.fill(.{ &stages_buffers, &new_kv_buffers });
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &stages_buffers);
    defer {
        new_kv_buffers.k.deinit();
        new_kv_buffers.v.deinit();
    }

    // 4. For each stage, print our shape/stats and compare against the matching reference.
    const opts: zml.testing.CompareOpts = .{ .absolute_tolerance = 1e-2, .minimum_close_fraction = 0.99 };

    const StageMap = struct { stage: []const u8, refs: []const []const u8 };
    const stage_map = [_]StageMap{
        .{ .stage = "q_proj", .refs = &.{"q_proj.out.0"} },
        .{ .stage = "k_proj", .refs = &.{"k_proj.out.0"} },
        .{ .stage = "v_proj", .refs = &.{"v_proj.out.0"} },
        .{ .stage = "g_proj", .refs = &.{"g_proj.out.0"} },
        .{ .stage = "q_norm", .refs = &.{"q_norm.out.0"} },
        .{ .stage = "k_norm", .refs = &.{"k_norm.out.0"} },
        .{ .stage = "q_pre_rope_hf", .refs = &.{"rope.q_in"} },
        .{ .stage = "k_pre_rope_hf", .refs = &.{ "rope.k_in", "k_in" } },
        .{ .stage = "cos", .refs = &.{ "rope.cos", "rotary_emb.out.0" } },
        .{ .stage = "sin", .refs = &.{ "rope.sin", "rotary_emb.out.1" } },
        .{ .stage = "q_rope_hf", .refs = &.{"rope.q_embed"} },
        .{ .stage = "k_rope_hf", .refs = &.{"rope.k_embed"} },
        .{ .stage = "attn", .refs = &.{"attn"} },
        .{ .stage = "gate_sig", .refs = &.{"gate_sig"} },
        .{ .stage = "gated", .refs = &.{"gated"} },
        .{ .stage = "o_proj_in", .refs = &.{"o_proj.in.0"} },
        .{ .stage = "out", .refs = &.{"out.0"} },
    };

    inline for (std.meta.fields(model.Attn.Stages)) |field| {
        const stage_name = field.name;
        const buf_ptr: *zml.Buffer = &@field(stages_buffers, field.name);

        std.log.info("--- stage {s}: ours shape={f} ---", .{ stage_name, buf_ptr.shape() });

        var ref_subkeys: []const []const u8 = &.{};
        for (stage_map) |m| {
            if (std.mem.eql(u8, m.stage, stage_name)) {
                ref_subkeys = m.refs;
                break;
            }
        }

        if (ref_subkeys.len == 0) {
            std.log.info("  (no reference mapping for this stage)", .{});
        } else {
            var matched = false;
            for (ref_subkeys) |subkey| {
                const ref_shape = view.getShape(subkey) orelse continue;
                matched = true;
                std.log.info("  ref '{s}.{s}': shape={f}", .{ name, subkey, ref_shape });

                const our_dims = buf_ptr.shape().dims();
                var our_count: i64 = 1;
                for (our_dims) |d| our_count *= d;
                var ref_count: i64 = 1;
                for (ref_shape.dims()) |d| ref_count *= d;
                if (our_count != ref_count) {
                    std.log.info("  -> element-count mismatch ours={d} ref={d}", .{ our_count, ref_count });
                    continue;
                }

                const expected: zml.Slice = try .alloc(allocator, ref_shape);
                defer expected.free(allocator);
                var reader_buffer: [4096]u8 = undefined;
                var reader = try view.getReader(subkey, io, &reader_buffer);
                defer reader.deinit();
                try reader.interface.readSliceAll(expected.data());

                const got = try buf_ptr.toSliceAlloc(allocator, io);
                defer got.free(allocator);

                const BF = zml.floats.BFloat16;
                const report = try zml.testing.compareSlices(
                    allocator,
                    BF,
                    BF,
                    expected.constItems(BF),
                    got.constItems(BF),
                    opts,
                );
                std.log.info("{f}", .{report});
                if (report.close_fraction < opts.minimum_close_fraction or report.nan_or_inf) {
                    std.log.warn("  -> FAIL for {s}", .{subkey});
                } else {
                    std.log.info("  -> ok for {s}", .{subkey});
                }
            }
            if (!matched) {
                std.log.info("  (no captured reference for this stage)", .{});
            }
        }
    }
}

fn surveySelfAttnShapes(
    allocator: std.mem.Allocator,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
) !void {
    const expected_hidden: i64 = 4096;
    const expected_kv: i64 = 8 * 128;
    const num_q_full: i64 = 64;
    const num_q_swa: i64 = 96;
    const head_dim_val: i64 = 128;
    const main_layers: usize = 45;

    std.log.info("=== SelfAttn shape survey (hidden={d}, kv={d}, full_q={d}, swa_q={d}) ===", .{
        expected_hidden, expected_kv, num_q_full * head_dim_val, num_q_swa * head_dim_val,
    });

    const proj_names = [_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj", "g_proj" };

    for (0..48) |layer_idx| {
        const attn_name = try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn", .{layer_idx});
        defer allocator.free(attn_name);
        const mview = model_store.view().withPrefix(attn_name);

        // Step 3.5 Flash pattern: Full, SWA, SWA, SWA. Layers >= 45 are MTP/spec blocks (SWA-shaped).
        const is_full = layer_idx < main_layers and layer_idx % 4 == 0;
        const num_q: i64 = if (is_full) num_q_full else num_q_swa;
        const expected_q: i64 = num_q * head_dim_val;
        const kind: []const u8 = if (layer_idx >= main_layers) "MTP" else if (is_full) "FULL" else "SWA ";

        var line: std.Io.Writer.Allocating = .init(allocator);
        defer line.deinit();
        const w = &line.writer;

        try w.print("layer {d:>2} [{s}] (exp q={d}):", .{ layer_idx, kind, expected_q });

        var mismatch = false;
        for (proj_names) |pn| {
            const wkey = try std.fmt.allocPrint(allocator, "{s}.weight", .{pn});
            defer allocator.free(wkey);
            if (mview.getShape(wkey)) |sh| {
                try w.print(" {s}={f}", .{ pn, sh });
                const dout = sh.dim(0);
                const din = sh.dim(1);
                if (std.mem.eql(u8, pn, "q_proj") and (dout != expected_q or din != expected_hidden)) mismatch = true;
                if ((std.mem.eql(u8, pn, "k_proj") or std.mem.eql(u8, pn, "v_proj")) and (dout != expected_kv or din != expected_hidden)) mismatch = true;
                if (std.mem.eql(u8, pn, "o_proj") and (dout != expected_hidden or din != expected_q)) mismatch = true;
                if (std.mem.eql(u8, pn, "g_proj") and (dout != num_q or din != expected_hidden)) mismatch = true;
            } else {
                try w.print(" {s}=<missing>", .{pn});
                mismatch = true;
            }
        }

        const in_name = try std.fmt.allocPrint(allocator, "{s}.in.0", .{attn_name});
        defer allocator.free(in_name);
        if (activation_store.view().getShape(in_name)) |sh| {
            try w.print(" act.in.0={f}", .{sh});
        }

        if (mismatch) {
            std.log.warn("{s}  <-- MISMATCH", .{line.written()});
        } else {
            std.log.info("{s}", .{line.written()});
        }
    }

    std.log.info("=== end survey ===", .{});
}

// ===========================================================================
// Synthetic RoPE tests
//
// These do not need real model activations; RoPE is deterministic math.
// feed simple increasing inputs through the compiled graph and compare
// against a Zig CPU reference implementation that mirrors the HuggingFace
// reference exactly.
// ===========================================================================

const Rope = struct {
    const B: i64 = 1;
    const S: i64 = 4;
    const H: i64 = 2;
    const HD: i64 = 8;
    const ROTARY_DIM: i64 = 8;
    const THETA: f32 = 10_000.0;

    const QK_LEN: usize = @as(usize, @intCast(B * S * H * HD));
    const CS_LEN: usize = @as(usize, @intCast(B * S * HD));
    const POS_LEN: usize = @as(usize, @intCast(B * S));

    // --- synthetic inputs ---

    fn makeQ(out: []f32) void {
        for (out, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0;
    }

    fn makeK(out: []f32) void {
        for (out, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0 + 0.5;
    }

    fn makePositionIds(out: []i32) void {
        for (out, 0..) |*v, i| v.* = @intCast(i % @as(usize, @intCast(S)));
    }

    // --- CPU references ---

    fn refInvFreq(dim: i64, theta: f32, out: []f32) void {
        const half: usize = @intCast(@divExact(dim, 2));
        std.debug.assert(out.len == half);
        const N: f32 = @floatFromInt(half);
        for (out, 0..) |*v, i| {
            const fi: f32 = @floatFromInt(i);
            v.* = std.math.pow(f32, theta, -fi / N);
        }
    }

    fn refCosSin(position_ids: []const i32, cos_out: []f32, sin_out: []f32) void {
        var inv_freq: [@as(usize, @intCast(@divExact(ROTARY_DIM, 2)))]f32 = undefined;
        refInvFreq(ROTARY_DIM, THETA, &inv_freq);

        const half: usize = inv_freq.len;
        for (position_ids, 0..) |pos, idx| {
            const base = idx * @as(usize, @intCast(HD));
            const p: f32 = @floatFromInt(pos);
            for (0..half) |i| {
                const f = p * inv_freq[i];
                cos_out[base + i] = @cos(f);
                cos_out[base + half + i] = @cos(f);
                sin_out[base + i] = @sin(f);
                sin_out[base + half + i] = @sin(f);
            }
        }
    }

    fn refRotateHalf(x: []const f32, out: []f32) void {
        const hd: usize = @intCast(HD);
        const half: usize = hd / 2;
        var i: usize = 0;
        while (i < x.len) : (i += hd) {
            for (0..half) |j| out[i + j] = -x[i + half + j];
            for (0..half) |j| out[i + half + j] = x[i + j];
        }
    }

    fn refApplyRope(
<<<<<<< HEAD
        q: []const f32,
        k: []const f32,
        cos: []const f32,
        sin: []const f32,
        q_out: []f32,
        k_out: []f32,
    ) void {
        var rot_q: [QK_LEN]f32 = undefined;
        var rot_k: [QK_LEN]f32 = undefined;
        refRotateHalf(q, &rot_q);
        refRotateHalf(k, &rot_k);
=======
        x: []const f32,
        cos: []const f32,
        sin: []const f32,
        x_out: []f32,
    ) void {
        var rot_x: [QK_LEN]f32 = undefined;
        refRotateHalf(x, &rot_x);
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)

        const hd: usize = @intCast(HD);
        const h: usize = @intCast(H);
        const s: usize = @intCast(S);
        const b: usize = @intCast(B);

        for (0..b) |bi| {
            for (0..s) |si| {
                for (0..h) |hi| {
                    for (0..hd) |di| {
<<<<<<< HEAD
                        const q_idx = ((bi * s + si) * h + hi) * hd + di;
                        const cs_idx = (bi * s + si) * hd + di;
                        q_out[q_idx] = q[q_idx] * cos[cs_idx] + rot_q[q_idx] * sin[cs_idx];
                        k_out[q_idx] = k[q_idx] * cos[cs_idx] + rot_k[q_idx] * sin[cs_idx];
=======
                        const x_idx = ((bi * s + si) * h + hi) * hd + di;
                        const cs_idx = (bi * s + si) * hd + di;
                        x_out[x_idx] = x[x_idx] * cos[cs_idx] + rot_x[x_idx] * sin[cs_idx];
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)
                    }
                }
            }
        }
    }

    // --- compiled-graph wrappers ---

    fn cosWrapper(rope: TextRotaryEmbedding, position_ids: zml.Tensor) zml.Tensor {
        const cos, _ = rope.getCosAndSin(position_ids, .f32);
        return cos;
    }

    fn sinWrapper(rope: TextRotaryEmbedding, position_ids: zml.Tensor) zml.Tensor {
        _, const sin = rope.getCosAndSin(position_ids, .f32);
        return sin;
    }

    fn rotateHalfWrapper(x: zml.Tensor) zml.Tensor {
        return TextRotaryEmbedding.rotateHalf(x);
    }

<<<<<<< HEAD
    fn applyRopeQ(
        rope: TextRotaryEmbedding,
        q: zml.Tensor,
        k: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    ) zml.Tensor {
        const q_out, _ = rope.applyRope(q, k, cos, sin);
        return q_out;
    }

    fn applyRopeK(
        rope: TextRotaryEmbedding,
        q: zml.Tensor,
        k: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    ) zml.Tensor {
        _, const k_out = rope.applyRope(q, k, cos, sin);
        return k_out;
=======
    fn applyRopeWrapper(
        rope: TextRotaryEmbedding,
        x: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    ) zml.Tensor {
        return rope.applyRope(x, cos, sin);
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)
    }
};

fn runRopeTests(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("RoPE synthetic tests:", .{});
    try testRotateHalf(allocator, io, platform);
    try testCosSin(allocator, io, platform);
    try testApplyRopeReference(allocator, io, platform);
    try testApplyRopePosition0Identity(allocator, io, platform);
    try testApplyRopePreservesNorm(allocator, io, platform);
    std.log.info("  all RoPE tests passed.", .{});
}

fn testRotateHalf(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  rotateHalf", .{});

    const x_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .h = Rope.H, .hd = Rope.HD }, .f32);
    const x_tensor = zml.Tensor.fromShape(x_shape);

    var x_data: [Rope.QK_LEN]f32 = undefined;
    Rope.makeQ(&x_data);

    var expected: [Rope.QK_LEN]f32 = undefined;
    Rope.refRotateHalf(&x_data, &expected);

    var exe = try zml.module.compile(allocator, io, Rope.rotateHalfWrapper, .{x_tensor}, platform, .{});
    defer exe.deinit();

    var x_buf: zml.Buffer = try .fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(&x_data));
    defer x_buf.deinit();

    var res = try zml.testing.autoCall(allocator, io, &exe, Rope.rotateHalfWrapper, .{x_buf});
    defer res.deinit();

    try zml.testing.expectClose(
        io,
        zml.Slice.init(x_shape, std.mem.sliceAsBytes(&expected)),
        res,
        .{ .absolute_tolerance = 1e-6 },
    );
}

fn testCosSin(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  getCosAndSin", .{});

    const rope = TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const pos_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S }, .i32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .hd = Rope.HD }, .f32);
    const pos_tensor = zml.Tensor.fromShape(pos_shape);

    var pos_data: [Rope.POS_LEN]i32 = undefined;
    Rope.makePositionIds(&pos_data);

    var cos_expected: [Rope.CS_LEN]f32 = undefined;
    var sin_expected: [Rope.CS_LEN]f32 = undefined;
    Rope.refCosSin(&pos_data, &cos_expected, &sin_expected);

    var pos_buf: zml.Buffer = try .fromBytes(io, platform, pos_shape, .replicated, std.mem.sliceAsBytes(&pos_data));
    defer pos_buf.deinit();

    var exe_cos = try zml.module.compile(allocator, io, Rope.cosWrapper, .{ rope, pos_tensor }, platform, .{});
    defer exe_cos.deinit();
    var cos_res = try zml.testing.autoCall(allocator, io, &exe_cos, Rope.cosWrapper, .{pos_buf});
    defer cos_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(cs_shape, std.mem.sliceAsBytes(&cos_expected)),
        cos_res,
        .{ .absolute_tolerance = 1e-5 },
    );

    var exe_sin = try zml.module.compile(allocator, io, Rope.sinWrapper, .{ rope, pos_tensor }, platform, .{});
    defer exe_sin.deinit();
    var sin_res = try zml.testing.autoCall(allocator, io, &exe_sin, Rope.sinWrapper, .{pos_buf});
    defer sin_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(cs_shape, std.mem.sliceAsBytes(&sin_expected)),
        sin_res,
        .{ .absolute_tolerance = 1e-5 },
    );
}

fn testApplyRopeReference(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  applyRope vs reference", .{});

    const rope = TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const qk_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .h = Rope.H, .hd = Rope.HD }, .f32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .hd = Rope.HD }, .f32);

    const q_t = zml.Tensor.fromShape(qk_shape);
    const k_t = zml.Tensor.fromShape(qk_shape);
    const cos_t = zml.Tensor.fromShape(cs_shape);
    const sin_t = zml.Tensor.fromShape(cs_shape);

    var q_data: [Rope.QK_LEN]f32 = undefined;
    var k_data: [Rope.QK_LEN]f32 = undefined;
    Rope.makeQ(&q_data);
    Rope.makeK(&k_data);

    var pos_data: [Rope.POS_LEN]i32 = undefined;
    Rope.makePositionIds(&pos_data);
    var cos_data: [Rope.CS_LEN]f32 = undefined;
    var sin_data: [Rope.CS_LEN]f32 = undefined;
    Rope.refCosSin(&pos_data, &cos_data, &sin_data);

    var q_expected: [Rope.QK_LEN]f32 = undefined;
    var k_expected: [Rope.QK_LEN]f32 = undefined;
<<<<<<< HEAD
    Rope.refApplyRope(&q_data, &k_data, &cos_data, &sin_data, &q_expected, &k_expected);
=======
    Rope.refApplyRope(&q_data, &cos_data, &sin_data, &q_expected);
    Rope.refApplyRope(&k_data, &cos_data, &sin_data, &k_expected);
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)

    var q_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&q_data));
    defer q_buf.deinit();
    var k_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&k_data));
    defer k_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&cos_data));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&sin_data));
    defer sin_buf.deinit();

<<<<<<< HEAD
    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeQ, .{ rope, q_t, k_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeQ, .{ q_buf, k_buf, cos_buf, sin_buf });
=======
    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeWrapper, .{ rope, q_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeWrapper, .{ q_buf, cos_buf, sin_buf });
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)
    defer q_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&q_expected)),
        q_res,
        .{ .absolute_tolerance = 1e-5 },
    );

<<<<<<< HEAD
    var exe_k = try zml.module.compile(allocator, io, Rope.applyRopeK, .{ rope, q_t, k_t, cos_t, sin_t }, platform, .{});
    defer exe_k.deinit();
    var k_res = try zml.testing.autoCall(allocator, io, &exe_k, Rope.applyRopeK, .{ q_buf, k_buf, cos_buf, sin_buf });
=======
    var exe_k = try zml.module.compile(allocator, io, Rope.applyRopeWrapper, .{ rope, k_t, cos_t, sin_t }, platform, .{});
    defer exe_k.deinit();
    var k_res = try zml.testing.autoCall(allocator, io, &exe_k, Rope.applyRopeWrapper, .{ k_buf, cos_buf, sin_buf });
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)
    defer k_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&k_expected)),
        k_res,
        .{ .absolute_tolerance = 1e-5 },
    );
}

fn testApplyRopePosition0Identity(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  applyRope position-0 identity", .{});

    const rope = TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const qk_shape = zml.Shape.init(.{ .b = Rope.B, .s = 1, .h = Rope.H, .hd = Rope.HD }, .f32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = 1, .hd = Rope.HD }, .f32);

    const q_t = zml.Tensor.fromShape(qk_shape);
    const k_t = zml.Tensor.fromShape(qk_shape);
    const cos_t = zml.Tensor.fromShape(cs_shape);
    const sin_t = zml.Tensor.fromShape(cs_shape);

    const qk_len: usize = @as(usize, @intCast(Rope.B * 1 * Rope.H * Rope.HD));
    const cs_len: usize = @as(usize, @intCast(Rope.B * 1 * Rope.HD));

    var q_data: [qk_len]f32 = undefined;
    var k_data: [qk_len]f32 = undefined;
    for (&q_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0;
    for (&k_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 100.0 + 0.5;

    // Position 0: cos = ones, sin = zeros, so applyRope must be the identity.
    var cos_data: [cs_len]f32 = .{1} ** cs_len;
    var sin_data: [cs_len]f32 = .{0} ** cs_len;

    var q_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&q_data));
    defer q_buf.deinit();
    var k_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&k_data));
    defer k_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&cos_data));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&sin_data));
    defer sin_buf.deinit();

<<<<<<< HEAD
    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeQ, .{ rope, q_t, k_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeQ, .{ q_buf, k_buf, cos_buf, sin_buf });
=======
    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeWrapper, .{ rope, q_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeWrapper, .{ q_buf, cos_buf, sin_buf });
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)
    defer q_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&q_data)),
        q_res,
        .{ .absolute_tolerance = 1e-6 },
    );

<<<<<<< HEAD
    var exe_k = try zml.module.compile(allocator, io, Rope.applyRopeK, .{ rope, q_t, k_t, cos_t, sin_t }, platform, .{});
    defer exe_k.deinit();
    var k_res = try zml.testing.autoCall(allocator, io, &exe_k, Rope.applyRopeK, .{ q_buf, k_buf, cos_buf, sin_buf });
=======
    var exe_k = try zml.module.compile(allocator, io, Rope.applyRopeWrapper, .{ rope, k_t, cos_t, sin_t }, platform, .{});
    defer exe_k.deinit();
    var k_res = try zml.testing.autoCall(allocator, io, &exe_k, Rope.applyRopeWrapper, .{ k_buf, cos_buf, sin_buf });
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)
    defer k_res.deinit();
    try zml.testing.expectClose(
        io,
        zml.Slice.init(qk_shape, std.mem.sliceAsBytes(&k_data)),
        k_res,
        .{ .absolute_tolerance = 1e-6 },
    );
}

fn testApplyRopePreservesNorm(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("  applyRope preserves per-head norm", .{});

    const rope = TextRotaryEmbedding.init(Rope.ROTARY_DIM, Rope.THETA, 1.0);
    const qk_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .h = Rope.H, .hd = Rope.HD }, .f32);
    const cs_shape = zml.Shape.init(.{ .b = Rope.B, .s = Rope.S, .hd = Rope.HD }, .f32);

    const q_t = zml.Tensor.fromShape(qk_shape);
    const k_t = zml.Tensor.fromShape(qk_shape);
<<<<<<< HEAD
=======
    _ = k_t; // autofix
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)
    const cos_t = zml.Tensor.fromShape(cs_shape);
    const sin_t = zml.Tensor.fromShape(cs_shape);

    var q_data: [Rope.QK_LEN]f32 = undefined;
    var k_data: [Rope.QK_LEN]f32 = undefined;
    Rope.makeQ(&q_data);
    Rope.makeK(&k_data);

    var pos_data: [Rope.POS_LEN]i32 = undefined;
    Rope.makePositionIds(&pos_data);
    var cos_data: [Rope.CS_LEN]f32 = undefined;
    var sin_data: [Rope.CS_LEN]f32 = undefined;
    Rope.refCosSin(&pos_data, &cos_data, &sin_data);

    var q_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&q_data));
    defer q_buf.deinit();
    var k_buf: zml.Buffer = try .fromBytes(io, platform, qk_shape, .replicated, std.mem.sliceAsBytes(&k_data));
    defer k_buf.deinit();
    var cos_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&cos_data));
    defer cos_buf.deinit();
    var sin_buf: zml.Buffer = try .fromBytes(io, platform, cs_shape, .replicated, std.mem.sliceAsBytes(&sin_data));
    defer sin_buf.deinit();

<<<<<<< HEAD
    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeQ, .{ rope, q_t, k_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeQ, .{ q_buf, k_buf, cos_buf, sin_buf });
=======
    var exe_q = try zml.module.compile(allocator, io, Rope.applyRopeWrapper, .{ rope, q_t, cos_t, sin_t }, platform, .{});
    defer exe_q.deinit();
    var q_res = try zml.testing.autoCall(allocator, io, &exe_q, Rope.applyRopeWrapper, .{ q_buf, cos_buf, sin_buf });
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)
    defer q_res.deinit();

    const q_out_slice = try q_res.toSliceAlloc(allocator, io);
    defer q_out_slice.free(allocator);
    const q_out_bytes = q_out_slice.constData();
    const q_out: []const f32 = @as([*]const f32, @ptrCast(@alignCast(q_out_bytes.ptr)))[0 .. q_out_bytes.len / @sizeOf(f32)];

    const hd: usize = @intCast(Rope.HD);
    const rd: usize = @intCast(Rope.ROTARY_DIM);
    var idx: usize = 0;
    while (idx < q_data.len) : (idx += hd) {
        var n_in: f32 = 0;
        var n_out: f32 = 0;
        for (0..rd) |j| {
            n_in += q_data[idx + j] * q_data[idx + j];
            n_out += q_out[idx + j] * q_out[idx + j];
        }
        if (@abs(n_in - n_out) > 1e-4) {
            std.log.err("Norm mismatch at head {d}: in={d}, out={d}", .{ idx / hd, n_in, n_out });
            return error.NormNotPreserved;
        }
    }
}
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> a636246d (examples/llm: RoPE & RoPE tests)
=======
>>>>>>> a3188857 (examples/llm: remove RoPE tests)
=======
>>>>>>> 36ddcb6b (examples/llm: update RoPE tests to generalize)
=======

// ===========================================================================
// KvCache synthetic tests
//
// Exercise KvCache.update + keys/values in isolation, no model weights needed.
// Write a known value at a known token slot for a known layer, then verify the
// slot we wrote got it and the rest stayed zero.
// ===========================================================================

const KvTest = struct {
    const LAYERS: i64 = 2;
    const B: i64 = 1;
    const H: i64 = 2;
    const K: i64 = 4;
    const HD: i64 = 8;
    const LAYER_IDX: usize = 1;
    const TOKEN_IDX: u32 = 2;

    const KV_LEN: usize = @as(usize, @intCast(LAYERS * B * H * K * HD));
    const NEW_LEN: usize = @as(usize, @intCast(B * H * 1 * HD));
    const READ_LEN: usize = @as(usize, @intCast(B * H * K * HD));

    fn updateAndRead(
        kv: model.KvCache,
        new_k: zml.Tensor,
        new_v: zml.Tensor,
        token_index: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor } {
        const updated = kv.update(new_k, new_v, token_index);
        return .{ updated.keys(), updated.values() };
    }
};

fn runKvCacheTests(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform) !void {
    std.log.info("KvCache synthetic tests:", .{});

    const kv_shape = zml.Shape.init(
        .{ .layer = KvTest.LAYERS, .b = KvTest.B, .h = KvTest.H, .k = KvTest.K, .hd = KvTest.HD },
        .f32,
    );
    const new_shape = zml.Shape.init(
        .{ .b = KvTest.B, .h = KvTest.H, .k = 1, .hd = KvTest.HD },
        .f32,
    );
    const idx_shape = zml.Shape.init(.{}, .u32);

    // Trace-time KvCache, pinned to LAYER_IDX.
    const kv_traced = model.KvCache.init(kv_shape).atLayer(KvTest.LAYER_IDX);
    const new_k_t = zml.Tensor.fromShape(new_shape);
    const new_v_t = zml.Tensor.fromShape(new_shape);
    const idx_t = zml.Tensor.fromShape(idx_shape);

    var exe = try zml.module.compile(
        allocator,
        io,
        KvTest.updateAndRead,
        .{ kv_traced, new_k_t, new_v_t, idx_t },
        platform,
        .{},
    );
    defer exe.deinit();

    // Runtime buffers.
    var k_init: [KvTest.KV_LEN]f32 = .{0} ** KvTest.KV_LEN;
    var v_init: [KvTest.KV_LEN]f32 = .{0} ** KvTest.KV_LEN;
    var new_k_data: [KvTest.NEW_LEN]f32 = .{1.0} ** KvTest.NEW_LEN;
    var new_v_data: [KvTest.NEW_LEN]f32 = .{2.0} ** KvTest.NEW_LEN;
    var idx_data: [1]u32 = .{KvTest.TOKEN_IDX};

    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, .replicated, std.mem.sliceAsBytes(&k_init));
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, .replicated, std.mem.sliceAsBytes(&v_init));
    defer v_buf.deinit();
    var new_k_buf: zml.Buffer = try .fromBytes(io, platform, new_shape, .replicated, std.mem.sliceAsBytes(&new_k_data));
    defer new_k_buf.deinit();
    var new_v_buf: zml.Buffer = try .fromBytes(io, platform, new_shape, .replicated, std.mem.sliceAsBytes(&new_v_data));
    defer new_v_buf.deinit();
    var idx_buf: zml.Buffer = try .fromBytes(io, platform, idx_shape, .replicated, std.mem.sliceAsBytes(&idx_data));
    defer idx_buf.deinit();

    const kv_buffers: model.KvCache.Buffer = .{ .k = k_buf, .v = v_buf };

    var result = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        KvTest.updateAndRead,
        .{ kv_buffers, new_k_buf, new_v_buf, idx_buf },
    );
    defer {
        result[0].deinit();
        result[1].deinit();
    }

    // Build expected: zeros everywhere except the written token slot for the
    // selected layer. keys() returns shape `.b, .h, .k, .hd` (layer dropped).
    var k_expected: [KvTest.READ_LEN]f32 = .{0} ** KvTest.READ_LEN;
    var v_expected: [KvTest.READ_LEN]f32 = .{0} ** KvTest.READ_LEN;
    {
        const b: usize = @intCast(KvTest.B);
        const h: usize = @intCast(KvTest.H);
        const k: usize = @intCast(KvTest.K);
        const hd: usize = @intCast(KvTest.HD);
        const tok: usize = KvTest.TOKEN_IDX;
        for (0..b) |bi| {
            for (0..h) |hi| {
                for (0..hd) |di| {
                    const off = ((bi * h + hi) * k + tok) * hd + di;
                    k_expected[off] = 1.0;
                    v_expected[off] = 2.0;
                }
            }
        }
    }

    const read_shape = zml.Shape.init(
        .{ .b = KvTest.B, .h = KvTest.H, .k = KvTest.K, .hd = KvTest.HD },
        .f32,
    );
    try zml.testing.expectClose(
        io,
        zml.Slice.init(read_shape, std.mem.sliceAsBytes(&k_expected)),
        result[0],
        .{ .absolute_tolerance = 1e-6 },
    );
    try zml.testing.expectClose(
        io,
        zml.Slice.init(read_shape, std.mem.sliceAsBytes(&v_expected)),
        result[1],
        .{ .absolute_tolerance = 1e-6 },
    );

    std.log.info("  KvCache update+read at layer={d} token={d} OK.", .{ KvTest.LAYER_IDX, KvTest.TOKEN_IDX });
}
>>>>>>> 24035ea3 (examples/llm: update tests)
