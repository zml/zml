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

pub fn main(init: std.process.Init) !void {
    std.log.info("main", .{});

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

    // try run(allocator, io, platform, args.activations_main, &model_store);
    try run(allocator, io, platform, args.activations_moe, &model_store);
}

fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    model_store: *zml.io.TensorStore,
) !void {
    std.log.info("run", .{});
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    std.log.info("activation store", .{});
    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    // const layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
    // std.log.info("MLP:", .{});
    // for (layer_indices) |layer_idx| {
    //     const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
    //     defer allocator.free(name);

    //     std.log.info("name {s}", .{name});

    //     try testLayer(allocator, io, platform, activation_store.view(), model_store, .mlp, name, .{ .absolute_tolerance = 1e-2 });
    // }
    // std.log.info("RMS Norm:", .{});

    // for (0..48) |layer_idx| {
    //     const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
    //     defer allocator.free(name);

    //     try testLayer(allocator, io, platform, activation_store.view(), model_store, .rmsNorm, name, .{ .absolute_tolerance = 1e-2 });
    // }
    // for (0..48) |layer_idx| {
    //     const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.post_attention_layernorm", .{layer_idx});
    //     defer allocator.free(name);

    //     try testLayer(allocator, io, platform, activation_store.view(), model_store, .rmsNorm, name, .{ .absolute_tolerance = 1e-2 });
    // }

    try testLayer(allocator, io, platform, activation_store.view(), model_store, .rmsNorm, "model.norm", .{ .absolute_tolerance = 1e-2 });
}

const LayerKind = enum { mlp, rmsNorm };

fn deinitBuffers(bufs: anytype) void {
    std.log.info("deinit buffers", .{});
    zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
            std.log.info("deinit", .{});
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
    name: []const u8,
    opts: zml.testing.CompareOpts,
) !void {
    switch (kind) {
        .mlp => {
            std.log.info("testlayer mlp {s}", .{name});
            const mlp = model.Mlp.init(model_store.view().withPrefix(name), null);

            std.log.info("before recursive cleanup for buffers", .{});
            // Recursive cleanup for buffers
            var mlp_weights = try zml.io.load(model.Mlp, &mlp, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&mlp_weights);

            try zml.testing.testLayer(allocator, io, platform, mlp, .forward, activation_store, name, mlp_weights, &.{}, opts);
        },
        .rmsNorm => {
            const rms = model.RmsNorm.init(model_store.view().withPrefix(name), @as(f32, 1e-5));

            var rms_weights = try zml.io.load(model.RmsNorm, &rms, allocator, io, platform, model_store, .auto);
            defer deinitBuffers(&rms_weights);

            try zml.testing.testLayer(allocator, io, platform, rms, .forward, activation_store, name, rms_weights, &.{}, opts);
        },
    }
}
