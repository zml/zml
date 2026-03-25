const std = @import("std");

const zml = @import("zml");
const Tensor = zml.Tensor;

const log = std.log.scoped(.general_triton_test);

fn loadBufferFromStore(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, store: *zml.io.TensorStore, key: []const u8) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;
    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    const sharding_replicated = try zml.sharding.replicatedSharding(platform);

    return zml.Buffer.fromBytes(io, platform, shape, sharding_replicated, host_bytes);
}

const Fwd = struct {
    pub fn forward(hidden_states: Tensor, w1: Tensor, w2: Tensor, topk_weights: Tensor, topk_ids: Tensor) Tensor {
        return zml.general_triton_moe.fusedExpertsImpl(hidden_states, w1, w2, topk_weights, topk_ids, .{}) catch |err| {
            std.debug.panic("general Triton MoE failed: {}", .{err});
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.c_allocator;
    const io = init.io;

    var it = init.minimal.args.iterate();
    const prog = it.next() orelse "general_triton_test";
    const dump_path = it.next() orelse "/home/louis/moe_fused_experts_impl_first_layer_first_iter_io.safetensors";
    if (it.next() != null) {
        log.err("usage: {s} [dump_path]", .{prog});
        return;
    }

    var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, dump_path);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    if (platform.target != .cuda) {
        log.warn("platform is not CUDA, skipping execution", .{});
        return;
    }

    var hidden_states = try loadBufferFromStore(allocator, io, platform, &store, "hidden_states_input");
    defer hidden_states.deinit();
    var w1 = try loadBufferFromStore(allocator, io, platform, &store, "w1_input");
    defer w1.deinit();
    var w2 = try loadBufferFromStore(allocator, io, platform, &store, "w2_input");
    defer w2.deinit();
    var topk_weights = try loadBufferFromStore(allocator, io, platform, &store, "topk_weights_input");
    defer topk_weights.deinit();
    var topk_ids = try loadBufferFromStore(allocator, io, platform, &store, "topk_ids_input");
    defer topk_ids.deinit();
    var expected = try loadBufferFromStore(allocator, io, platform, &store, "hidden_states_output");
    defer expected.deinit();

    log.info("hidden_states shape: {f}", .{hidden_states.shape()});
    log.info("w1 shape: {f}", .{w1.shape()});
    log.info("w2 shape: {f}", .{w2.shape()});
    log.info("topk_weights shape: {f}", .{topk_weights.shape()});
    log.info("topk_ids shape: {f}", .{topk_ids.shape()});
    log.info("expected output shape: {f}", .{expected.shape()});

    const sharding_replicated = try zml.sharding.replicatedSharding(platform);

    var exe = try zml.module.compile(
        allocator,
        io,
        Fwd.forward,
        .{
            Tensor.fromShape(hidden_states.shape()),
            Tensor.fromShape(w1.shape()),
            Tensor.fromShape(w2.shape()),
            Tensor.fromShape(topk_weights.shape()),
            Tensor.fromShape(topk_ids.shape()),
        },
        platform,
        .{ .shardings = &.{sharding_replicated} },
    );
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ hidden_states, w1, w2, topk_weights, topk_ids });

    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    exe.callOpts(io, args, &results, .{ .wait = true });

    var actual = results.get(zml.Buffer);
    defer actual.deinit();
    log.info("computation done, validating results against expected output from dump", .{});

    const expected_host = try expected.toSliceAlloc(allocator, io);
    defer expected_host.free(allocator);

    log.info("actual output shape: {f}", .{actual.shape()});
    log.info("expected output shape: {f}", .{expected_host.shape});
    try zml.testing.expectClose(io, actual, expected_host, .{});
    log.info("validation succeeded for {s}", .{dump_path});
}
