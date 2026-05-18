const std = @import("std");
const zml = @import("zml");

// Multilayer Perceptron
const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        // Add tags to input before providing to our layer.
        x = x.withTags(.{ .b, .s, .d });

        const up_proj = self.up_proj.forward(x);
        var gate = self.gate_proj.forward(x);
        gate = gate.silu();

        return self.down_proj.forward(gate.mul(up_proj));
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;

    var vfs = try zml.io.VFS.init(allocator, io);
    defer vfs.deinit();
    const vfs_io = vfs.io();

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3) return;

    // Load + resolve paths (Zig 0.16.0 treats relative paths as maybes, which breaks GPU)
    const weights_path_raw = args[1];
    const activations_path_raw = args[2];
    const cwd = std.Io.Dir.cwd();
    const model_path = try cwd.realPathFileAlloc(io, weights_path_raw, arena);
    const activations_path = try cwd.realPathFileAlloc(io, activations_path_raw, arena);

    // Registries
    var model_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, vfs_io, model_path);
    defer model_registry.deinit();
    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, vfs_io, activations_path);
    defer activations_registry.deinit();

    // Connect to GPUs, load drivers, create memory map
    const platform = try zml.Platform.auto(allocator, vfs_io, .{});
    defer platform.deinit(allocator, vfs_io);

    // Stores
    var model_store = zml.io.TensorStore.fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    var activation_store = zml.io.TensorStore.fromRegistry(allocator, &activations_registry);
    defer activation_store.deinit();

    std.log.info("All captured activations loaded", .{});

    ////////////////////////////////////////////////////////////////
    const current_layer = "model.layers.46.mlp";
    ////////////////////////////////////////////////////////////////
    const mlp_view = model_store.view().withPrefix(current_layer);

    // start with initialized version zand then change it to the pub const later
    const mlp: Mlp = .{
        .up_proj = .init(mlp_view.withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }), mlp_view.withPrefix("up_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
        .gate_proj = .init(mlp_view.withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .replicated, .d = .replicated }), mlp_view.withPrefix("gate_proj").maybeCreateTensor("bias", .{.dout}, .{ .dout = .model }), .d),
        .down_proj = .init(mlp_view.withPrefix("down_proj").createTensor("weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .replicated }), mlp_view.withPrefix("down_proj").maybeCreateTensor("bias", .{.d}, .{ .d = .replicated }), .dout),
    };

    // at this point, weights are on disk and layer is initialized
    const mlp_weights = try zml.io.load(Mlp, &mlp, allocator, vfs_io, platform, &model_store, .auto);
    try zml.testing.testLayer(arena, vfs_io, platform, mlp, .forward, activation_store.view(), current_layer, .{ .mlp = mlp_weights }, &.{}, .{});
}
