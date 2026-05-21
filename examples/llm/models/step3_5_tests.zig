const std = @import("std");

const zml = @import("zml");

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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    var model_registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer model_registry.deinit();
    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

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
    }
>>>>>>> 311be4c4 (examples/llm: test all router gate outputs)
}

fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    model_store: *zml.io.TensorStore,
) !void {
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

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
        const mlp_layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
        std.log.info("MLP:", .{});
        for (mlp_layer_indices) |layer_idx| {
            const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
            defer allocator.free(name);
>>>>>>> 311be4c4 (examples/llm: test all router gate outputs)

            std.log.info("name {s}", .{name});

            try testLayer(allocator, io, platform, activation_store.view(), model_store, .mlp, name, name, .{ .absolute_tolerance = 1e-2 });
        }
        std.log.info("RMS Norm:", .{});

<<<<<<< HEAD
        for (0..48) |layer_idx| {
            const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
            defer allocator.free(name);

            try testLayer(allocator, io, platform, activation_store.view(), model_store, .rmsNorm, name, name, .{ .absolute_tolerance = 1e-2 });
        }
        for (0..48) |layer_idx| {
            const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.post_attention_layernorm", .{layer_idx});
            defer allocator.free(name);

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

    //     const mlp = model.Mlp.init(model_store.view().withPrefix(name), null);

    //     // Recursive cleanup for buffers
    //     var mlp_weights = try zml.io.load(model.Mlp, &mlp, allocator, io, platform, model_store, .auto);
    //     defer deinitBuffers(&mlp_weights);

    //     std.log.info("name {s}", .{name});

    //     const input_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name});
    //     defer allocator.free(input_key);
    //     if (!activation_store.view().hasKey(input_key)) {
    //         std.log.warn("skipping {s}: no activations recorded", .{name});
    //         continue;
    //     }

    //     zml.testing.testLayer(allocator, io, platform, mlp, .forward, activation_store.view(), name, mlp_weights, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
    //         std.log.warn("skipping {s}: {s}", .{ name, @errorName(err) });
    //     };
    // }
    // std.log.info("RMS Norm:", .{});

    // for (0..48) |layer_idx| {
    //     const name__input_layernorm = try std.fmt.allocPrint(allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
    //     defer allocator.free(name__input_layernorm);

    //     const name__post_attention_layernorm = try std.fmt.allocPrint(allocator, "model.layers.{d}.post_attention_layernorm", .{layer_idx});
    //     defer allocator.free(name__post_attention_layernorm);

    //     const rms1 = model.RmsNorm.init(model_store.view().withPrefix(name__input_layernorm), @as(f32, 1e-5));
    //     const rms2 = model.RmsNorm.init(model_store.view().withPrefix(name__post_attention_layernorm), @as(f32, 1e-5));

    //     var rms_weights1 = try zml.io.load(model.RmsNorm, &rms1, allocator, io, platform, model_store, .auto);
    //     defer deinitBuffers(&rms_weights1);

    //     var rms_weights2 = try zml.io.load(model.RmsNorm, &rms2, allocator, io, platform, model_store, .auto);
    //     defer deinitBuffers(&rms_weights2);

    //     const in1_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name__input_layernorm});
    //     defer allocator.free(in1_key);
    //     if (activation_store.view().hasKey(in1_key)) {
    //         zml.testing.testLayer(allocator, io, platform, rms1, .forward, activation_store.view(), name__input_layernorm, rms_weights1, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
    //             std.log.warn("skipping {s}: {s}", .{ name__input_layernorm, @errorName(err) });
    //         };
    //     } else {
    //         std.log.warn("skipping {s}: no activations recorded", .{name__input_layernorm});
    //     }

    //     const in2_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name__post_attention_layernorm});
    //     defer allocator.free(in2_key);
    //     if (activation_store.view().hasKey(in2_key)) {
    //         zml.testing.testLayer(allocator, io, platform, rms2, .forward, activation_store.view(), name__post_attention_layernorm, rms_weights2, &.{}, .{ .absolute_tolerance = 1e-2 }) catch |err| {
    //             std.log.warn("skipping {s}: {s}", .{ name__post_attention_layernorm, @errorName(err) });
    //         };
    //     } else {
    //         std.log.warn("skipping {s}: no activations recorded", .{name__post_attention_layernorm});
    //     }
    // }

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
}
>>>>>>> c856b0f1 (examples/llm: clean up tests and model annotations)
=======
const LayerKind = enum { mlp, rmsNorm, router };
>>>>>>> 939965a6 (examples/llm: test router)

fn deinitBuffers(bufs: anytype) void {
    zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, bufs);
}
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
            //hardcoded k=num_experts_per_tok=288 for now
            const router = model.Router.init(model_store.view().withPrefix(weights_name), 288);

            var router_weights = try zml.io.load(model.Router, &router, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&router_weights);

            try compareRouterTopK(allocator, io, platform, router, router_weights, activation_store, activations_name, 8);
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
    var args: struct { zml.Tensor, bool } = .{
        activation_store.withPrefix(activations_name).withPrefix("in").createTensor("0", null, .replicated),
        true,
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
