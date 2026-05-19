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
    activations: []const u8,

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

const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)
    limit: ?i32,

    pub fn init(store: zml.io.TensorStore.View, swiglu_limit: ?i32) Mlp {
        return .{
            .up_proj = .init(store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .replicated }), null, .d),
            .gate_proj = .init(store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .replicated }), null, .d),
            .down_proj = .init(store.createTensor("down_proj.weight", .{ .d, .dout }, .{ .d = .replicated }), null, .dout),
            .limit = swiglu_limit,
        };
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        // Add tags to input before providing to our layer.
        const input = x.withTags(.{ .b, .s, .d });

        var up_proj = self.up_proj.forward(input);
        var gate = self.gate_proj.forward(input);
        gate = gate.silu();

        // Step 3.5 Flash clamps gate projection asymmetrically
        if (self.limit) |limit| {
            if (limit != 0) {
                const lim_f = @as(f32, @floatFromInt(limit));
                const max_t = zml.Tensor.scalar(lim_f, gate.dtype());
                const min_t = zml.Tensor.scalar(-lim_f, gate.dtype());

                // Step 3.5 Flash has asymmetric clamping of gate projection
                gate = gate.minimum(max_t);
                up_proj = up_proj.clamp(min_t, max_t);
            }
        }
        return self.down_proj.forward(gate.mul(up_proj));
    }
};

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

    const layer_indices = [_]usize{ 0, 1, 2, 45, 46, 47 };
    for (layer_indices) |layer_idx| {
        const name = try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp", .{layer_idx});
        defer allocator.free(name);

        try testLayer(allocator, io, platform, activation_store.view(), model_store, .mlp, name, .{ .absolute_tolerance = 1e-2 });
    }
}

const LayerKind = enum { mlp };

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
            const mlp = Mlp.init(model_store.view().withPrefix(name), null);

            var mlp_weights = try zml.io.load(Mlp, &mlp, allocator, io, platform, model_store, .auto);
            defer zml.meta.visit(struct {
                fn cb(_: void, b: *zml.Buffer) void {
                    b.deinit();
                }
            }.cb, {}, &mlp_weights);

            try zml.testing.testLayer(allocator, io, platform, mlp, .forward, activation_store, name, mlp_weights, &.{}, opts);
        },
    }
}
