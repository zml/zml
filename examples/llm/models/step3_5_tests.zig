const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const Step3p5Flash = @import("step3_5flash.zig");
const model = @import("step3_5flash/model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations_main: []const u8,
    activations_moe: []const u8,

    pub const help =
        \\Use step3_5_tests --model=<path> --activations=<path>
        \\
        \\ Validate the Step 3.5 Flash implementation against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>            Path to the model repository
        \\   --activations=<path>      Path to activation safetensors
        \\
    ;
};

const TEST_LAYER = 1;

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

    if (TEST_LAYER == 0) {
        // test main
        try run(allocator, io, platform, args.activations_main, &model_store);
    } else if (TEST_LAYER == 1) {
        // test sublayer (within-gate activations)
        try run(allocator, io, platform, args.activations_moe, &model_store);
    }
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

    if (TEST_LAYER == 0) {
        const mlp_layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
        std.log.info("MLP:", .{});
        for (mlp_layer_indices) |layer_idx| {
            const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
            defer allocator.free(name);

            std.log.info("name {s}", .{name});

            try testLayer(allocator, io, platform, activation_store.view(), model_store, .mlp, name, name, .{ .absolute_tolerance = 1e-2 });
        }
        std.log.info("RMS Norm:", .{});

        for (0..48) |layer_idx| {
            const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
            defer allocator.free(name);

            try testLayer(allocator, io, platform, activation_store.view(), model_store, .rmsNorm, name, name, .{ .absolute_tolerance = 1e-2 });
        }
        for (0..48) |layer_idx| {
            const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.post_attention_layernorm", .{layer_idx});
            defer allocator.free(name);

            try testLayer(allocator, io, platform, activation_store.view(), model_store, .rmsNorm, name, name, .{ .absolute_tolerance = 1e-2 });
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

const LayerKind = enum { mlp, rmsNorm, router };

fn deinitBuffers(bufs: anytype) void {
    zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, bufs);
}

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
    const k: usize = @intCast(dims[dims.len - 1]);
    var positions: usize = 1;
    for (dims[0 .. dims.len - 1]) |d| positions *= @intCast(d);

    const exp_ids = exp.constItems(i32);
    const got_ids = got.constItems(i32);

    var mismatches: usize = 0;
    var first: usize = 0;
    for (0..positions) |p| {
        if (!topSetEqual(exp_ids[p * k ..][0..show], got_ids[p * k ..][0..show])) {
            if (mismatches == 0) first = p;
            mismatches += 1;
        }
    }

    if (mismatches == 0) {
        std.log.info("✅ {s}: top-{d} matches at all {d} positions", .{ activations_name, show, positions });
    } else {
        std.log.warn("⚠️  {s}: {d}/{d} positions differ. first @{d}: exp={any} got={any}", .{
            activations_name,               first,
            mismatches,                     positions,
            exp_ids[first * k ..][0..show], got_ids[first * k ..][0..show],
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
