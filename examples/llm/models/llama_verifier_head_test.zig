const std = @import("std");
const zml = @import("zml");
const model = @import("llama/model.zig");

test "all-row greedy Llama head preserves row identity and smallest-index ties" {
    if (comptime !zml.Target.tenstorrent.isEnabled()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .tenstorrent, .{ .tenstorrent = .{ .device_index = 0 } });
    defer platform.deinit(allocator, io);
    const bf16 = zml.DataType.bf16.toZigType();
    var embedding: [32 * 32]bf16 = @splat(.fromF32(0));
    embedding[0] = .fromF32(1);
    embedding[32] = .fromF32(1);
    embedding[3 * 32 + 2] = .fromF32(1);
    var hidden: [2 * 32]bf16 = @splat(.fromF32(0));
    hidden[0] = .fromF32(1);
    hidden[32 + 2] = .fromF32(1);
    const norm: [32]bf16 = @splat(.fromF32(1));
    const head: model.LmHead = .{
        .lm_head = null,
        .embed_tokens = .{ .weight = .init(.{ .voc = 32, .d = 32 }, .bf16) },
        .norm = .{ .weight = .init(.{ .d = 32 }, .bf16), .eps = 1e-5 },
        .gen_opts = .{ .topk = 1 },
    };
    const Check = struct {
        fn run(h: model.LmHead, x: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor } {
            const first = model.LmHead.forward(.{ .lm_head = h, .hidden = x, .active_length = .scalar(1, .u32), .rng = zml.Tensor.Rng.init() }).tokens;
            const last = model.LmHead.forward(.{ .lm_head = h, .hidden = x, .active_length = .scalar(2, .u32), .rng = zml.Tensor.Rng.init() }).tokens;
            return .{ h.forwardAllGreedy(x), first, last };
        }
    };
    const exe = try platform.compileFn(allocator, io, Check.run, .{ head, zml.Tensor.init(.{ .s = 2, .d = 32 }, .bf16) }, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var embedding_buffer = try zml.Buffer.fromBytes(io, platform, head.embed_tokens.weight.shape(), std.mem.asBytes(&embedding));
    defer embedding_buffer.deinit();
    var norm_buffer = try zml.Buffer.fromBytes(io, platform, head.norm.weight.shape(), std.mem.asBytes(&norm));
    defer norm_buffer.deinit();
    var hidden_buffer = try zml.Buffer.fromBytes(io, platform, .init(.{ .s = 2, .d = 32 }, .bf16), std.mem.asBytes(&hidden));
    defer hidden_buffer.deinit();
    const head_buffers: zml.Bufferized(model.LmHead) = .{ .lm_head = null, .embed_tokens = .{ .weight = embedding_buffer }, .norm = .{ .weight = norm_buffer } };
    var all: zml.Buffer = undefined;
    var first: zml.Buffer = undefined;
    var last: zml.Buffer = undefined;
    runner.run(io, .{ head_buffers, hidden_buffer }, .{ &all, &first, &last }, .{ .wait = true });
    defer all.deinit();
    defer first.deinit();
    defer last.deinit();
    var actual: [2]u32 = undefined;
    try all.toSlice(io, .init(all.shape(), std.mem.asBytes(&actual)));
    try std.testing.expectEqualSlices(u32, &.{ 0, 3 }, &actual);
    try std.testing.expectEqual(actual[0], try first.getValue(u32, io));
    try std.testing.expectEqual(actual[1], try last.getValue(u32, io));
}
