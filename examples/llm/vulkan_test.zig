const std = @import("std");
const zml = @import("zml");

const BFloat16 = zml.floats.BFloat16;

fn abs(x: zml.Tensor) zml.Tensor {
    return x.abs();
}

fn flatten(x: zml.Tensor) zml.Tensor {
    return x.flatten();
}

fn convertU8ToF32(x: zml.Tensor) zml.Tensor {
    return x.convert(.f32);
}

// fn convertI32ToU8(x: zml.Tensor) zml.Tensor {
//     return x.convert(.u8);
// }

fn dot(weight: zml.Tensor, input: zml.Tensor) zml.Tensor {
    return weight.dot(input, .d);
}

fn prefillAttentionScoreDot(query: zml.Tensor, key: zml.Tensor) zml.Tensor {
    return query.dot(key, .hd);
}

fn failingPrefillVanillaAttention(
    query: zml.Tensor,
    key: zml.Tensor,
    value: zml.Tensor,
    token_index: zml.Tensor,
    output_weight: zml.Tensor,
) zml.Tensor {
    const seq_len = key.dim(.k);
    var mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, query.dtype(), null);
    mask = mask.gatherSlices(
        zml.Shape.init(.{ .q = query.dim(.q) }, mask.dtype()),
        token_index.reshape(.{ .coord = 1 }),
        .{},
    );
    const attention = zml.nn.sdpa(query, key, value, .{ .attn_mask = mask })
        .merge(.{ .d = .{ .h, .hd } });
    return attention.dot(output_weight, .d).rename(.{ .dout = .d });
}

fn add(input: zml.Tensor, bias: zml.Tensor) zml.Tensor {
    return input.add(bias);
}

fn subtract(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.sub(rhs);
}

fn multiply(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.mul(rhs);
}

fn divide(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.div(rhs);
}

fn remainder(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.remainder(rhs);
}

fn negate(x: zml.Tensor) zml.Tensor {
    return x.negate();
}

fn minimum(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.minimum(rhs);
}

fn maximum(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.maximum(rhs);
}

fn lessThan(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.cmp(.LT, rhs);
}

fn relu(x: zml.Tensor) zml.Tensor {
    return x.relu();
}

fn argMax(x: zml.Tensor) zml.Tensor {
    return x.argMax(.d).indices;
}

fn queryIota() zml.Tensor {
    return zml.Tensor.iota(.init(.{ .q = 4, .k = 4 }, .i32), .q);
}

fn keyIota() zml.Tensor {
    return zml.Tensor.iota(.init(.{ .q = 4, .k = 4 }, .i32), .k);
}

fn compareCausal(key: zml.Tensor, query: zml.Tensor) zml.Tensor {
    return key.cmp(.LE, query);
}

fn selectCausal(mask: zml.Tensor, zeros: zml.Tensor, minus_inf: zml.Tensor) zml.Tensor {
    return zml.Tensor.select(mask, zeros, minus_inf);
}

fn gatherCausal(mask: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    return mask.gatherSlices(
        zml.Shape.init(.{ .q = 2 }, mask.dtype()),
        token_index.reshape(.{ .coord = 1 }),
        .{},
    );
}

fn broadcastCausal(mask: zml.Tensor) zml.Tensor {
    return mask.broad(.init(.{ .h = 2, .q = 4, .hq = 2, .k = 4 }, mask.dtype()));
}

fn addCausal(scores: zml.Tensor, mask: zml.Tensor) zml.Tensor {
    return scores.add(mask);
}

fn fusedCausalMask(scores: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
    const mask_shape = zml.Shape.init(.{ .q = 32, .k = 32 }, .bf16);
    const query = zml.Tensor.iota(mask_shape, .q);
    const key = zml.Tensor.iota(mask_shape, .k);
    const causal = key.cmp(.LE, query);
    const zeros = zml.Tensor.constant(zml.DataType.bf16.zero()).broad(mask_shape);
    const minus_inf = zml.Tensor.constant(zml.DataType.bf16.minValue()).broad(mask_shape);
    var mask = zml.Tensor.select(causal, zeros, minus_inf);
    mask = mask.gatherSlices(
        zml.Shape.init(.{ .q = 32 }, .bf16),
        token_index.reshape(.{ .coord = 1 }),
        .{},
    );
    mask = mask.broad(scores.shape());
    return scores.add(mask);
}

fn expectOutput(
    comptime Output: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    operation_name: []const u8,
    output: zml.Buffer,
    expected: []const Output,
) !void {
    const host_output = try allocator.alloc(Output, expected.len);
    defer allocator.free(host_output);
    try output.toSlice(
        io,
        zml.Slice.init(output.shape(), std.mem.sliceAsBytes(host_output)),
    );
    if (host_output.len <= 64) {
        std.debug.print("  {s} output: {any}\n", .{ operation_name, host_output });
    }
    try std.testing.expectEqualSlices(Output, expected, host_output);
}

fn printSpirv(
    allocator: std.mem.Allocator,
    io: std.Io,
    dump_dir: []const u8,
) !void {
    const result = try std.process.run(allocator, io, .{
        .argv = &.{
            "/bin/bash",
            "-c",
            \\set -euo pipefail
            \\spirv_file="$(find "$1" -maxdepth 1 -type f -name '*.vulkan.spv' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
            \\test -n "$spirv_file"
            \\patched_file="$(mktemp /tmp/zml-causal-select-spirv.XXXXXX.spv)"
            \\trap 'rm -f "$patched_file"' EXIT
            \\cp "$spirv_file" "$patched_file"
            \\offset=20
            \\size="$(stat -c %s "$patched_file")"
            \\while [ "$offset" -lt "$size" ]; do
            \\  instruction="$(od -An -tu4 -j "$offset" -N4 "$patched_file")"
            \\  word_count=$((instruction >> 16))
            \\  opcode=$((instruction & 65535))
            \\  test "$word_count" -gt 0
            \\  if [ "$opcode" -eq 17 ]; then
            \\    capability="$(od -An -tu4 -j "$((offset + 4))" -N4 "$patched_file")"
            \\    if [ "$capability" -eq 5116 ]; then
            \\      printf '\001\000\000\000' | dd of="$patched_file" bs=1 seek="$((offset + 4))" conv=notrunc status=none
            \\    fi
            \\  fi
            \\  offset=$((offset + word_count * 4))
            \\done
            \\echo "SPIR-V executed by causal_select: $spirv_file"
            \\echo "(BFloat16TypeKHR capability patched only in the temporary copy for this older spirv-dis.)"
            \\spirv-dis "$patched_file" -o -
            ,
            "print-causal-select-spirv",
            dump_dir,
        },
        .stdout_limit = .limited(1024 * 1024),
        .stderr_limit = .limited(64 * 1024),
        .reserve_amount = 16 * 1024,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    if (result.stdout.len != 0) std.debug.print("{s}", .{result.stdout});
    if (result.stderr.len != 0) std.debug.print("{s}", .{result.stderr});
    switch (result.term) {
        .exited => |code| if (code != 0) return error.SpirvDisassemblyFailed,
        else => return error.SpirvDisassemblyFailed,
    }
}

fn runNullary(
    comptime Output: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime operation_name: []const u8,
    comptime dump_dir: []const u8,
    comptime function: anytype,
    expected: []const Output,
) !void {
    std.debug.print("Vulkan operation: {s}\n", .{operation_name});

    var exe = try platform.compileFn(allocator, io, function, .{}, .{
        .program_name = operation_name,
        .xla_dump_to = dump_dir,
        .xla_dump_emitter_re = "mlir-fusion",
    });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{});

    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    exe.call(args, &results);

    var output: zml.Buffer = results.get(zml.Buffer);
    defer output.deinit();
    try expectOutput(Output, allocator, io, operation_name, output, expected);
}

fn runUnary(
    comptime Input: type,
    comptime Output: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime operation_name: []const u8,
    comptime dump_dir: []const u8,
    comptime function: anytype,
    input_shape: zml.Tensor,
    host_input: []const Input,
    expected: []const Output,
) !void {
    std.debug.print("Vulkan operation: {s}\n", .{operation_name});

    var exe = try platform.compileFn(allocator, io, function, .{input_shape}, .{
        .program_name = operation_name,
        .xla_dump_to = dump_dir,
        .xla_dump_emitter_re = "mlir-fusion",
    });
    defer exe.deinit();

    var input = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(input_shape.shape(), std.mem.sliceAsBytes(host_input)),
        .replicated,
    );
    defer input.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{input});

    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    exe.call(args, &results);

    var output: zml.Buffer = results.get(zml.Buffer);
    defer output.deinit();
    try expectOutput(Output, allocator, io, operation_name, output, expected);
}

fn runBinary(
    comptime Lhs: type,
    comptime Rhs: type,
    comptime Output: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime operation_name: []const u8,
    comptime dump_dir: []const u8,
    comptime function: anytype,
    lhs_shape: zml.Tensor,
    lhs_host: []const Lhs,
    rhs_shape: zml.Tensor,
    rhs_host: []const Rhs,
    expected: []const Output,
) !void {
    std.debug.print("Vulkan operation: {s}\n", .{operation_name});

    var exe = try platform.compileFn(allocator, io, function, .{ lhs_shape, rhs_shape }, .{
        .program_name = operation_name,
        .xla_dump_to = dump_dir,
        .xla_dump_emitter_re = "mlir-fusion",
    });
    defer exe.deinit();

    var lhs = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(lhs_shape.shape(), std.mem.sliceAsBytes(lhs_host)),
        .replicated,
    );
    defer lhs.deinit();
    var rhs = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(rhs_shape.shape(), std.mem.sliceAsBytes(rhs_host)),
        .replicated,
    );
    defer rhs.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ lhs, rhs });

    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    exe.call(args, &results);

    var output: zml.Buffer = results.get(zml.Buffer);
    defer output.deinit();
    try expectOutput(Output, allocator, io, operation_name, output, expected);
}

fn runTernary(
    comptime First: type,
    comptime Second: type,
    comptime Third: type,
    comptime Output: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime operation_name: []const u8,
    comptime dump_dir: []const u8,
    comptime function: anytype,
    first_shape: zml.Tensor,
    first_host: []const First,
    second_shape: zml.Tensor,
    second_host: []const Second,
    third_shape: zml.Tensor,
    third_host: []const Third,
    expected: []const Output,
) !void {
    std.debug.print("Vulkan operation: {s}\n", .{operation_name});

    var exe = try platform.compileFn(allocator, io, function, .{ first_shape, second_shape, third_shape }, .{
        .program_name = operation_name,
        .xla_dump_to = dump_dir,
        .xla_dump_emitter_re = "mlir-fusion",
    });
    defer exe.deinit();

    var first = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(first_shape.shape(), std.mem.sliceAsBytes(first_host)),
        .replicated,
    );
    defer first.deinit();
    var second = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(second_shape.shape(), std.mem.sliceAsBytes(second_host)),
        .replicated,
    );
    defer second.deinit();
    var third = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(third_shape.shape(), std.mem.sliceAsBytes(third_host)),
        .replicated,
    );
    defer third.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ first, second, third });

    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    exe.call(args, &results);

    try printSpirv(allocator, io, dump_dir);

    var output: zml.Buffer = results.get(zml.Buffer);
    defer output.deinit();
    try expectOutput(Output, allocator, io, operation_name, output, expected);
}

fn bf16Slice(comptime values: []const f32) [values.len]BFloat16 {
    var result: [values.len]BFloat16 = undefined;
    for (values, 0..) |value, i| {
        result[i] = BFloat16.fromF32(value);
    }
    return result;
}

fn fillFusedExpected(
    scores: []BFloat16,
    expected: []BFloat16,
) void {
    var finite_count: usize = 0;
    var masked_count: usize = 0;
    var i: usize = 0;
    for (0..8) |_| {
        for (0..32) |q| {
            for (0..4) |_| {
                for (0..32) |k| {
                    scores[i] = BFloat16.fromF32(@floatFromInt(i % 97 + 1));
                    if (k <= q) {
                        expected[i] = scores[i];
                        finite_count += 1;
                    } else {
                        expected[i] = BFloat16.minus_inf;
                        masked_count += 1;
                    }
                    i += 1;
                }
            }
        }
    }
    std.debug.assert(finite_count == 16_896);
    std.debug.assert(masked_count == 15_872);
}

fn runFusedCausalMask(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
) !void {
    const element_count = 8 * 32 * 4 * 32;
    const scores = try allocator.alloc(BFloat16, element_count);
    defer allocator.free(scores);
    const expected = try allocator.alloc(BFloat16, element_count);
    defer allocator.free(expected);
    fillFusedExpected(scores, expected);

    try runBinary(
        BFloat16,
        i32,
        BFloat16,
        allocator,
        io,
        platform,
        "causal_mask_fused_prefill",
        "/tmp/xla-vulkan/causal_mask_fused_prefill",
        fusedCausalMask,
        .init(.{ .h = 8, .q = 32, .hq = 4, .k = 32 }, .bf16),
        scores,
        .init(.{}, .i32),
        &.{0},
        expected,
    );
    std.debug.print("  causal_mask_fused_prefill verified 16,896 finite and 15,872 -inf entries\n", .{});
}

fn runFailingPrefillVanillaAttention(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
) !void {
    const query: zml.Tensor = .init(.{ .q = 12_000, .h = 32, .hd = 64 }, .bf16);
    const key: zml.Tensor = .init(.{ .k = 12_000, .h = 8, .hd = 64 }, .bf16);
    const value: zml.Tensor = .init(.{ .k = 12_000, .h = 8, .hd = 64 }, .bf16);
    const token_index: zml.Tensor = .init(.{}, .u32);
    const output_weight: zml.Tensor = .init(.{ .dout = 2_048, .d = 2_048 }, .bf16);
    const dump_path = switch (platform.target) {
        .cuda => "/tmp/xla-cuda/failing_prefill_vanilla_attention",
        .vulkan => "/tmp/xla-vulkan/failing_prefill_vanilla_attention",
        else => "/tmp/xla-other/failing_prefill_vanilla_attention",
    };

    var exe = try platform.compileFn(allocator, io, failingPrefillVanillaAttention, .{ query, key, value, token_index, output_weight }, .{
        .program_name = "failing_prefill_vanilla_attention",
        .xla_dump_to = dump_path,
        .xla_dump_emitter_re = "mlir-fusion",
    });
    defer exe.deinit();

    try std.testing.expectEqual(@as(usize, 1), exe.output_shapes.len);
    try zml.testing.expectEqualShapes(
        .init(.{ .q = 12_000, .d = 2_048 }, .bf16),
        exe.output_shapes[0],
    );

    const query_host = try allocator.alloc(BFloat16, query.shape().count());
    defer allocator.free(query_host);
    @memset(query_host, BFloat16.zero);
    const key_host = try allocator.alloc(BFloat16, key.shape().count());
    defer allocator.free(key_host);
    @memset(key_host, BFloat16.zero);
    const value_host = try allocator.alloc(BFloat16, value.shape().count());
    defer allocator.free(value_host);
    @memset(value_host, BFloat16.zero);
    const output_weight_host = try allocator.alloc(BFloat16, output_weight.shape().count());
    defer allocator.free(output_weight_host);
    @memset(output_weight_host, BFloat16.zero);

    var query_buffer = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(query.shape(), std.mem.sliceAsBytes(query_host)),
        .replicated,
    );
    defer query_buffer.deinit();
    var key_buffer = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(key.shape(), std.mem.sliceAsBytes(key_host)),
        .replicated,
    );
    defer key_buffer.deinit();
    var value_buffer = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(value.shape(), std.mem.sliceAsBytes(value_host)),
        .replicated,
    );
    defer value_buffer.deinit();
    var token_index_buffer = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(token_index.shape(), std.mem.sliceAsBytes(&[_]u32{0})),
        .replicated,
    );
    defer token_index_buffer.deinit();
    var output_weight_buffer = try zml.Buffer.fromSlice(
        io,
        platform,
        zml.Slice.init(output_weight.shape(), std.mem.sliceAsBytes(output_weight_host)),
        .replicated,
    );
    defer output_weight_buffer.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ query_buffer, key_buffer, value_buffer, token_index_buffer, output_weight_buffer });

    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    std.debug.print("Executing exact Llama vanilla-attention failure reproduction...\n", .{});
    exe.call(args, &results);

    var output: zml.Buffer = results.get(zml.Buffer);
    defer output.deinit();
    var output_host = try zml.Slice.alloc(allocator, output.shape());
    defer output_host.free(allocator);
    try output.toSlice(io, output_host);
    std.debug.print("Unexpected success, output: {f}\n", .{output.shape()});
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const platform: *zml.Platform = try .init(allocator, io, .vulkan, .{});
    defer platform.deinit(allocator, io);

    // Existing Vulkan operation tests are disabled while isolating the Llama
    // prefill attention score dot reproduction.
    if (true) {
        try runUnary(f32, f32, allocator, io, platform, "abs", "/tmp/xla-vulkan/abs", abs, .init(.{4}, .f32), &.{ -1, 2, -3, 4 }, &.{ 1, 2, 3, 4 });
        try runUnary(f32, f32, allocator, io, platform, "flatten", "/tmp/xla-vulkan/flatten", flatten, .init(.{ 2, 2 }, .f32), &.{ 1, 2, 3, 4 }, &.{ 1, 2, 3, 4 });
        try runUnary(u8, f32, allocator, io, platform, "convert_u8_to_f32", "/tmp/xla-vulkan/convert_u8_to_f32", convertU8ToF32, .init(.{4}, .u8), &.{ 0, 1, 127, 255 }, &.{ 0, 1, 127, 255 });
        // try runUnary(i32, u8, allocator, io, platform, "convert_i32_to_u8", "/tmp/xla-vulkan/convert_i32_to_u8", convertI32ToU8, .init(.{4}, .i32), &.{ 0, 1, 127, 255 }, &.{ 0, 1, 127, 255 });
        try runBinary(f32, f32, f32, allocator, io, platform, "dot", "/tmp/xla-vulkan/dot", dot, .init(.{ .d_out = 2, .d = 3 }, .f32), &.{ 1, 2, 3, 4, 5, 6 }, .init(.{ .d = 3 }, .f32), &.{ 2, 1, -1 }, &.{ 1, 7 });
        try runBinary(f32, f32, f32, allocator, io, platform, "add", "/tmp/xla-vulkan/add", add, .init(.{ .d = 4 }, .f32), &.{ 1, 2, 3, 4 }, .init(.{ .d = 4 }, .f32), &.{ 10, -2, 0, 0.5 }, &.{ 11, 0, 3, 4.5 });
        try runUnary(f32, f32, allocator, io, platform, "relu", "/tmp/xla-vulkan/relu", relu, .init(.{5}, .f32), &.{ -3, -0.5, 0, 2, 7 }, &.{ 0, 0, 0, 2, 7 });
        try runUnary(f32, i32, allocator, io, platform, "argMax", "/tmp/xla-vulkan/argMax", argMax, .init(.{ .d = 4 }, .f32), &.{ 1, 7, 7, 3 }, &.{1});

        const query_iota_expected = [_]i32{
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3,
        };
        try runNullary(i32, allocator, io, platform, "causal_query_iota", "/tmp/xla-vulkan/causal_query_iota", queryIota, &query_iota_expected);

        const key_iota_expected = [_]i32{
            0, 1, 2, 3,
            0, 1, 2, 3,
            0, 1, 2, 3,
            0, 1, 2, 3,
        };
        try runNullary(i32, allocator, io, platform, "causal_key_iota", "/tmp/xla-vulkan/causal_key_iota", keyIota, &key_iota_expected);

        const causal_bool = [_]bool{
            true, false, false, false,
            true, true,  false, false,
            true, true,  true,  false,
            true, true,  true,  true,
        };
        try runBinary(i32, i32, bool, allocator, io, platform, "causal_compare", "/tmp/xla-vulkan/causal_compare", compareCausal, .init(.{ .q = 4, .k = 4 }, .i32), &key_iota_expected, .init(.{ .q = 4, .k = 4 }, .i32), &query_iota_expected, &causal_bool);

        const bf16_zeros = [_]BFloat16{BFloat16.zero} ** 16;
        const bf16_minus_inf = [_]BFloat16{BFloat16.minus_inf} ** 16;
        var causal_bf16 = bf16_minus_inf;
        for (&causal_bf16, causal_bool) |*value, keep| {
            if (keep) value.* = BFloat16.zero;
        }
        try runTernary(bool, BFloat16, BFloat16, BFloat16, allocator, io, platform, "causal_select", "/tmp/reese-vulkan/dump/causal-select", selectCausal, .init(.{ .q = 4, .k = 4 }, .bool), &causal_bool, .init(.{ .q = 4, .k = 4 }, .bf16), &bf16_zeros, .init(.{ .q = 4, .k = 4 }, .bf16), &bf16_minus_inf, &causal_bf16);

        const tagged_mask = bf16Slice(&.{
            10, 11, 12, 13,
            20, 21, 22, 23,
            30, 31, 32, 33,
            40, 41, 42, 43,
        });
        try runBinary(BFloat16, i32, BFloat16, allocator, io, platform, "causal_gather", "/tmp/xla-vulkan/causal_gather", gatherCausal, .init(.{ .q = 4, .k = 4 }, .bf16), &tagged_mask, .init(.{}, .i32), &.{1}, tagged_mask[4..12]);

        var broadcast_expected: [2 * 4 * 2 * 4]BFloat16 = undefined;
        var broadcast_i: usize = 0;
        for (0..2) |_| {
            for (0..4) |q| {
                for (0..2) |_| {
                    for (0..4) |k| {
                        broadcast_expected[broadcast_i] = causal_bf16[q * 4 + k];
                        broadcast_i += 1;
                    }
                }
            }
        }
        try runUnary(BFloat16, BFloat16, allocator, io, platform, "causal_broadcast", "/tmp/xla-vulkan/causal_broadcast", broadcastCausal, .init(.{ .q = 4, .k = 4 }, .bf16), &causal_bf16, &broadcast_expected);

        const small_scores = bf16Slice(&.{
            1,  2,  3,  4,
            5,  6,  7,  8,
            9,  10, 11, 12,
            13, 14, 15, 16,
        });
        var add_expected = small_scores;
        for (&add_expected, causal_bool) |*value, keep| {
            if (!keep) value.* = BFloat16.minus_inf;
        }
        try runBinary(BFloat16, BFloat16, BFloat16, allocator, io, platform, "causal_add", "/tmp/xla-vulkan/causal_add", addCausal, .init(.{ .q = 4, .k = 4 }, .bf16), &small_scores, .init(.{ .q = 4, .k = 4 }, .bf16), &causal_bf16, &add_expected);

        const arithmetic_lhs = bf16Slice(&.{ 6, -6, 5, -5, 0 });
        const arithmetic_rhs = bf16Slice(&.{ 2, 2, -2, -2, 3 });
        const negate_expected = bf16Slice(&.{ -6, 6, -5, 5, -0.0 });
        const add_arithmetic_expected = bf16Slice(&.{ 8, -4, 3, -7, 3 });
        const subtract_expected = bf16Slice(&.{ 4, -8, 7, -3, -3 });
        const multiply_expected = bf16Slice(&.{ 12, -12, -10, 10, 0 });
        const divide_expected = bf16Slice(&.{ 3, -3, -2.5, 2.5, 0 });
        const remainder_expected = bf16Slice(&.{ 0, -0.0, 1, -1, 0 });
        const minimum_expected = bf16Slice(&.{ 2, -6, -2, -5, 0 });
        const maximum_expected = bf16Slice(&.{ 6, 2, 5, -2, 3 });
        const less_than_expected = [_]bool{ false, true, false, true, true };
        try runUnary(BFloat16, BFloat16, allocator, io, platform, "bf16_negate", "/tmp/xla-vulkan/bf16_negate", negate, .init(.{ .d = 5 }, .bf16), &arithmetic_lhs, &negate_expected);
        try runBinary(BFloat16, BFloat16, BFloat16, allocator, io, platform, "bf16_add", "/tmp/xla-vulkan/bf16_add", add, .init(.{ .d = 5 }, .bf16), &arithmetic_lhs, .init(.{ .d = 5 }, .bf16), &arithmetic_rhs, &add_arithmetic_expected);
        try runBinary(BFloat16, BFloat16, BFloat16, allocator, io, platform, "bf16_subtract", "/tmp/xla-vulkan/bf16_subtract", subtract, .init(.{ .d = 5 }, .bf16), &arithmetic_lhs, .init(.{ .d = 5 }, .bf16), &arithmetic_rhs, &subtract_expected);
        try runBinary(BFloat16, BFloat16, BFloat16, allocator, io, platform, "bf16_multiply", "/tmp/xla-vulkan/bf16_multiply", multiply, .init(.{ .d = 5 }, .bf16), &arithmetic_lhs, .init(.{ .d = 5 }, .bf16), &arithmetic_rhs, &multiply_expected);
        try runBinary(BFloat16, BFloat16, BFloat16, allocator, io, platform, "bf16_divide", "/tmp/xla-vulkan/bf16_divide", divide, .init(.{ .d = 5 }, .bf16), &arithmetic_lhs, .init(.{ .d = 5 }, .bf16), &arithmetic_rhs, &divide_expected);
        try runBinary(BFloat16, BFloat16, BFloat16, allocator, io, platform, "bf16_remainder", "/tmp/xla-vulkan/bf16_remainder", remainder, .init(.{ .d = 5 }, .bf16), &arithmetic_lhs, .init(.{ .d = 5 }, .bf16), &arithmetic_rhs, &remainder_expected);
        try runBinary(BFloat16, BFloat16, BFloat16, allocator, io, platform, "bf16_minimum", "/tmp/xla-vulkan/bf16_minimum", minimum, .init(.{ .d = 5 }, .bf16), &arithmetic_lhs, .init(.{ .d = 5 }, .bf16), &arithmetic_rhs, &minimum_expected);
        try runBinary(BFloat16, BFloat16, BFloat16, allocator, io, platform, "bf16_maximum", "/tmp/xla-vulkan/bf16_maximum", maximum, .init(.{ .d = 5 }, .bf16), &arithmetic_lhs, .init(.{ .d = 5 }, .bf16), &arithmetic_rhs, &maximum_expected);
        try runBinary(BFloat16, BFloat16, bool, allocator, io, platform, "bf16_less_than", "/tmp/xla-vulkan/bf16_less_than", lessThan, .init(.{ .d = 5 }, .bf16), &arithmetic_lhs, .init(.{ .d = 5 }, .bf16), &arithmetic_rhs, &less_than_expected);

        try runFusedCausalMask(allocator, io, platform);
    }

    var gemm_lhs: [3 * 17]BFloat16 = undefined;
    @memset(&gemm_lhs, BFloat16.fromF32(1));
    var gemm_rhs: [17 * 5]BFloat16 = undefined;
    for (0..17) |k| {
        for (0..5) |n| {
            gemm_rhs[k * 5 + n] = BFloat16.fromF32(@floatFromInt(n + 1));
        }
    }
    var gemm_expected: [3 * 5]BFloat16 = undefined;
    for (0..3) |m| {
        for (0..5) |n| {
            gemm_expected[m * 5 + n] = BFloat16.fromF32(@floatFromInt(17 * (n + 1)));
        }
    }
    try runBinary(BFloat16, BFloat16, BFloat16, allocator, io, platform, "bf16_gemm_3x17x5", "/tmp/xla-vulkan/bf16_gemm_3x17x5", gemmBf16, .init(.{ .m = 3, .k = 17 }, .bf16), &gemm_lhs, .init(.{ .k = 17, .n = 5 }, .bf16), &gemm_rhs, &gemm_expected);

    // try runFailingPrefillVanillaAttention(allocator, io, platform);
}

fn gemmBf16(lhs: zml.Tensor, rhs: zml.Tensor) zml.Tensor {
    return lhs.dot(rhs, .k);
}
