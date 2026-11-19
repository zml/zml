// Merge cleanup: keep this S4000 hardware regression test with the MUSA plugin integration.
const std = @import("std");

const zml = @import("zml/core");
const attention = @import("zml/kernels/paged_attention");

fn matmul(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.dot(rhs, .contract).reuseBuffer(lhs);
}

test "MUSA matmul replaces the donated input in device memory" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = try zml.Platform.init(allocator, io, .musa, .{});
    defer platform.deinit(allocator, io);
    const lhs: zml.Tensor = .init(.{ .row = 2, .contract = 2 }, .f32);
    const rhs: zml.Tensor = .init(.{ .contract = 2, .col = 2 }, .f32);
    const exe = try platform.compileFn(allocator, io, matmul, .{ lhs, rhs }, .{});
    defer exe.deinit();
    try std.testing.expectEqualSlices(u32, &.{0}, exe.donated_input_indices);

    var left = try zml.Buffer.fromBytes(io, platform, lhs.shape(), std.mem.asBytes(&[4]f32{ 1, 2, 3, 4 }));
    defer left.deinit();
    var right = try zml.Buffer.fromBytes(io, platform, rhs.shape(), std.mem.asBytes(&[4]f32{ 2, 0, 1, 3 }));
    defer right.deinit();
    const address = left.opaqueDevicePtr(0);
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    runner.run(io, .{ left, right }, .{&left}, .{ .wait = true });

    try std.testing.expectEqual(address, left.opaqueDevicePtr(0));
    try std.testing.expectEqual([4]f32{ 4, 6, 10, 12 }, try left.getValue([4]f32, io));
}

const Inputs = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
    table: zml.Tensor,
    lengths: zml.Tensor,
    starts: zml.Tensor,
};

fn causalAttention(i: Inputs) zml.Tensor {
    const platform = zml.Compiler.current().platform;
    const p: attention.Parameters = .{ .stablehlo = .{
        .block_table = i.table,
        .seq_lens = i.lengths,
        .query_start_len = i.starts,
        .options_ = .{ .batch_size = 1, .max_num_pages = 2, .max_seqlen_q = 2, .is_prefill = true },
    } };
    const kv: attention.KvCache = .{ .split = .{ .k = i.k, .v = i.v } };
    const opts: attention.AttentionOptions = .{ .scale = @log(@as(f32, 2)) };
    return attention.pagedAttention(p, i.q, i.k, i.v, kv, opts, platform.replicated_mesh, .{
        .in_specs = .{
            .parameters = zml.ops.partitionSpecs(p, .replicated),
            .q = .replicated,
            .kv_cache = zml.ops.partitionSpecs(kv, .replicated),
            .opts = zml.ops.partitionSpecs(opts, .replicated),
        },
        .out_specs = .replicated,
    });
}

test "MUSA causal split-KV attention combines two physical pages in logical order" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = try zml.Platform.init(allocator, io, .musa, .{});
    defer platform.deinit(allocator, io);
    try std.testing.expectEqual(attention.Backend.stablehlo, attention.Backend.auto(platform));
    const cache: zml.Tensor = .init(.{ .page = 2, .k_chunk = 2, .hkv = 1, .hd = 2 }, .f32);
    const inputs: Inputs = .{
        .q = .init(.{ .b = 2, .hkv = 1, .hg = 1, .hd = 2 }, .f32),
        .k = cache,
        .v = .fromShape(cache.shape()),
        .table = .init(.{ .b = 1, .p = 2 }, .i32),
        .lengths = .init(.{ .b = 1 }, .i32),
        .starts = .init(.{ .b = 2 }, .i32),
    };
    const exe = try platform.compileFn(allocator, io, causalAttention, .{inputs}, .{});
    defer exe.deinit();

    // Queries are at logical positions 2 and 3. The table maps pages in reverse order.
    const host: struct { q: [4]f32, k: [8]f32, v: [8]f32, table: [2]i32, lengths: [1]i32, starts: [2]i32 } = .{
        .q = .{ 1, 0, 0, 1 },
        .k = .{ 1, 1, 0, 0, 0, 1, 1, 0 },
        .v = .{ 12, 5, 24, 7, 0, 1, 6, 3 },
        .table = .{ 1, 0 },
        .lengths = .{4},
        .starts = .{ 0, 2 },
    };
    var buffers: zml.Bufferized(Inputs) = undefined;
    const fields = @typeInfo(Inputs).@"struct".field_names;
    var uploaded: usize = 0;
    defer {
        inline for (fields, 0..) |field, index| {
            if (index < uploaded) @field(buffers, field).deinit();
        }
    }
    inline for (fields) |field| {
        @field(buffers, field) = try .fromBytes(io, platform, @field(inputs, field).shape(), std.mem.asBytes(&@field(host, field)));
        uploaded += 1;
    }
    var output = try exe.eval(allocator, io, .{buffers});
    defer output.deinit();

    // Scaling by ln(2) gives unnormalized weights [1,2,2,0] and [2,1,2,1].
    const expected = [4]f32{ 36.0 / 5.0, 17.0 / 5.0, 9, 22.0 / 6.0 };
    const actual = try output.getValue([4]f32, io);
    for (expected, actual) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-5);
}
