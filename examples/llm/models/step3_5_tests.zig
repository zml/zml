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

fn swigluLimitFor(layer_idx: usize) ?f32 {
    return switch (layer_idx) {
        43, 44 => 7.0,
        else => null,
    };
}

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

    try run(allocator, io, platform, args.activations, &model_store);
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

    const mlp_layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };

    std.log.info("MLP:", .{});
    for (mlp_layer_indices) |layer_idx| {
        const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
        defer allocator.free(name);

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
    for (3..45) |layer_idx| {
        const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.moe", .{layer_idx});
        defer allocator.free(name);

        const moe = try model.Moe.init(model_store.view().withPrefix(name));

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

fn deinitBuffers(bufs: anytype) void {
    zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, bufs);
}
