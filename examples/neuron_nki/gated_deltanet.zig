const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const zml = @import("zml");
const stdx = zml.stdx;
const bf16 = zml.floats.BFloat16;

const log = std.log.scoped(.neuron_nki_gated_deltanet);

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        .{ .scope = .@"zml/module", .level = .info },
    },
};

const batch_size = 1;
const seq_len = 16;
const num_q_heads = 16;
const num_value_heads = 32;
const qk_head_repetition = @divExact(num_value_heads, num_q_heads);
const key_dim = 128;
const value_dim = 128;

const Args = struct {
    warmups: usize = 2,
    iterations: usize = 5,

    pub const help =
        \\ neuron_nki gated_deltanet --iterations=20
        \\
        \\ Options:
        \\   --warmups=<n>    Untimed execution warmups.
        \\   --iterations=<n> Timed execution iterations.
        \\
    ;
};

const Program = struct {
    source: []const u8,

    fn forward(self: Program, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, g: zml.Tensor, beta: zml.Tensor, h0: zml.Tensor) [2]zml.Tensor {
        return zml.ops.neuronNki(.{ q, k, v, g, beta, h0 }, .{ outputShape(), h0.shape() }, .{
            .name = "gated_deltanet",
            .entrypoint = "gated_deltanet",
            .source = self.source,
        });
    }
};

const ReferenceProgram = struct {
    fn forward(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, g: zml.Tensor, beta: zml.Tensor, h0: zml.Tensor) [2]zml.Tensor {
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(key_dim)));
        const query_heads = zml.nn.normalizeL2(q.rename(.{ .d = .k }).convert(.f32), 1.0e-6).scale(scale);
        const key_heads = zml.nn.normalizeL2(k.rename(.{ .d = .k }).convert(.f32), 1.0e-6);
        const queries = query_heads.stutter1d(@intCast(query_heads.axis(.h)), qk_head_repetition);
        const keys = key_heads.stutter1d(@intCast(key_heads.axis(.h)), qk_head_repetition);
        const values = v.rename(.{ .vh = .h, .vd = .v }).convert(.f32);
        const alphas = g.rename(.{ .vh = .h }).convert(.f32).exp();
        const betas = beta.rename(.{ .vh = .h }).convert(.f32);
        const initial_state = h0.rename(.{ .vh = .h, .vd = .v, .d = .k });

        const result = zml.nn.GatedDeltaNet.forward(queries, keys, values, alphas, betas, .{ .s = initial_state });
        return .{
            result.outputs.rename(.{ .h = .vh, .v = .vd }),
            result.state.s.rename(.{ .h = .vh, .v = .vd, .k = .d }),
        };
    }
};

const KernelBuffers = struct {
    output: zml.Buffer,
    final_state: zml.Buffer,

    fn deinit(self: *KernelBuffers) void {
        self.output.deinit();
        self.final_state.deinit();
    }
};

const BenchmarkResult = struct {
    buffers: KernelBuffers,
    total: stdx.time.Duration,
    average: stdx.time.Duration,
};

const ExpectedSlices = struct {
    output: zml.Slice,
    final_state: zml.Slice,

    fn free(self: *ExpectedSlices, allocator: std.mem.Allocator) void {
        self.output.free(allocator);
        self.final_state.free(allocator);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, Args);
    stdx.debug.assert(args.iterations > 0, "gated_deltanet benchmark expects at least one timed iteration", .{});

    const runfiles = try zml.bazel.runfiles(bazel_builtin.current_repository);

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const kernel_path = try runfiles.rlocation("zml/examples/neuron_nki/gated_deltanet.py", &path_buf);
    const kernel_source = try std.Io.Dir.readFileAlloc(.cwd(), io, kernel_path.?, allocator, .unlimited);
    defer allocator.free(kernel_source);
    const program: Program = .{ .source = kernel_source };

    const platform: *zml.Platform = try .init(allocator, io, .neuron, .{});
    defer platform.deinit(allocator, io);

    const sharding = try zml.sharding.replicatedSharding(platform);

    const q: zml.Tensor = .init(qShape(.bf16), .bf16);
    const k: zml.Tensor = .init(qShape(.bf16), .bf16);
    const v: zml.Tensor = .init(vShape(.bf16), .bf16);
    const g: zml.Tensor = .init(gateShape(.f32), .f32);
    const beta: zml.Tensor = .init(gateShape(.bf16), .bf16);
    const h0: zml.Tensor = .init(stateShape(.f32), .f32);

    var nki_exe = try platform.compileFn(allocator, io, Program.forward, .{ program, q, k, v, g, beta, h0 }, .{ .shardings = &.{sharding} });
    defer nki_exe.deinit();

    var reference_exe = try platform.compileFn(allocator, io, ReferenceProgram.forward, .{ q, k, v, g, beta, h0 }, .{ .shardings = &.{sharding} });
    defer reference_exe.deinit();

    var q_slice = try zml.Slice.alloc(allocator, q.shape());
    defer q_slice.free(allocator);
    var k_slice = try zml.Slice.alloc(allocator, k.shape());
    defer k_slice.free(allocator);
    var v_slice = try zml.Slice.alloc(allocator, v.shape());
    defer v_slice.free(allocator);
    var g_slice = try zml.Slice.alloc(allocator, g.shape());
    defer g_slice.free(allocator);
    var beta_slice = try zml.Slice.alloc(allocator, beta.shape());
    defer beta_slice.free(allocator);
    var h0_slice = try zml.Slice.alloc(allocator, h0.shape());
    defer h0_slice.free(allocator);

    fillInputs(q_slice, k_slice, v_slice, g_slice, beta_slice, h0_slice);

    var expected = try computeExpected(allocator, q_slice, k_slice, v_slice, g_slice, beta_slice, h0_slice);
    defer expected.free(allocator);

    var q_buffer: zml.Buffer = try .fromSlice(io, platform, q_slice, sharding);
    defer q_buffer.deinit();
    var k_buffer: zml.Buffer = try .fromSlice(io, platform, k_slice, sharding);
    defer k_buffer.deinit();
    var v_buffer: zml.Buffer = try .fromSlice(io, platform, v_slice, sharding);
    defer v_buffer.deinit();
    var g_buffer: zml.Buffer = try .fromSlice(io, platform, g_slice, sharding);
    defer g_buffer.deinit();
    var beta_buffer: zml.Buffer = try .fromSlice(io, platform, beta_slice, sharding);
    defer beta_buffer.deinit();
    var h0_buffer: zml.Buffer = try .fromSlice(io, platform, h0_slice, sharding);
    defer h0_buffer.deinit();

    var reference_bench = try benchmarkExecutable(allocator, io, &reference_exe, .{
        q_buffer,
        k_buffer,
        v_buffer,
        g_buffer,
        beta_buffer,
        h0_buffer,
    }, args.warmups, args.iterations);
    defer reference_bench.buffers.deinit();

    var nki_bench = try benchmarkExecutable(allocator, io, &nki_exe, .{
        q_buffer,
        k_buffer,
        v_buffer,
        g_buffer,
        beta_buffer,
        h0_buffer,
    }, args.warmups, args.iterations);
    defer nki_bench.buffers.deinit();

    const compare_opts: zml.testing.CompareOpts = .{
        .absolute_tolerance = 5e-3,
        .relative_tolerance = 5e-2,
        .minimum_close_fraction = 0.999,
    };
    try zml.testing.expectClose(io, reference_bench.buffers.output, expected.output, compare_opts);
    try zml.testing.expectClose(io, reference_bench.buffers.final_state, expected.final_state, compare_opts);
    try zml.testing.expectClose(io, nki_bench.buffers.output, reference_bench.buffers.output, compare_opts);
    try zml.testing.expectClose(io, nki_bench.buffers.final_state, reference_bench.buffers.final_state, compare_opts);

    log.info("Bench gated_deltanet warmups={} iterations={} seq_len={} key_dim={} value_dim={}", .{
        args.warmups,
        args.iterations,
        seq_len,
        key_dim,
        value_dim,
    });
    log.info("ZML nn reference total={d:.3}ms avg={d:.3}ms", .{
        nsToMs(reference_bench.total.ns),
        nsToMs(reference_bench.average.ns),
    });
    log.info("Neuron NKI total={d:.3}ms avg={d:.3}ms reference_avg={d:.3}ms speedup={d:.3}x output={f} final_state={f}", .{
        nsToMs(nki_bench.total.ns),
        nsToMs(nki_bench.average.ns),
        nsToMs(reference_bench.average.ns),
        @as(f64, @floatFromInt(reference_bench.average.ns)) / @as(f64, @floatFromInt(nki_bench.average.ns)),
        nki_bench.buffers.output.shape(),
        nki_bench.buffers.final_state.shape(),
    });
}

fn benchmarkExecutable(allocator: std.mem.Allocator, io: std.Io, exe: anytype, inputs: anytype, warmups: usize, iterations: usize) !BenchmarkResult {
    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    exe_args.set(inputs);

    for (0..warmups) |_| {
        exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });
        var warmup_result = exe_results.get(KernelBuffers);
        warmup_result.deinit();
    }

    var total: stdx.time.Duration = .{};
    var last_output: ?KernelBuffers = null;
    errdefer if (last_output) |*output| output.deinit();

    for (0..iterations) |_| {
        const start: std.Io.Timestamp = .now(io, .awake);
        exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });
        total.ns += @intCast(start.untilNow(io, .awake).toNanoseconds());

        if (last_output) |*output| output.deinit();
        last_output = exe_results.get(KernelBuffers);
    }

    return .{
        .buffers = last_output.?,
        .total = total,
        .average = total.div(iterations),
    };
}

fn computeExpected(
    allocator: std.mem.Allocator,
    q: zml.Slice,
    k: zml.Slice,
    v: zml.Slice,
    g: zml.Slice,
    beta: zml.Slice,
    h0: zml.Slice,
) !ExpectedSlices {
    var expected: ExpectedSlices = .{
        .output = try zml.Slice.alloc(allocator, outputShape()),
        .final_state = try zml.Slice.alloc(allocator, stateShape(.f32)),
    };
    errdefer expected.free(allocator);

    const q_values = q.constItems(bf16);
    const k_values = k.constItems(bf16);
    const v_values = v.constItems(bf16);
    const g_values = g.constItems(f32);
    const beta_values = beta.constItems(bf16);
    const h0_values = h0.constItems(f32);
    const out_values = expected.output.items(f32);
    const ht_values = expected.final_state.items(f32);

    var state = try allocator.alloc(f32, num_value_heads * value_dim * key_dim);
    defer allocator.free(state);
    @memcpy(state, h0_values[0..state.len]);

    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(key_dim)));
    var q_norm: [num_q_heads][key_dim]f32 = undefined;
    var k_norm: [num_q_heads][key_dim]f32 = undefined;

    for (0..seq_len) |t| {
        for (0..num_q_heads) |qh| {
            var q_sum_sq: f32 = 0;
            var k_sum_sq: f32 = 0;
            for (0..key_dim) |ki| {
                const q_value = q_values[qkIndex(t, qh, ki)].toF32();
                const k_value = k_values[qkIndex(t, qh, ki)].toF32();
                q_norm[qh][ki] = q_value;
                k_norm[qh][ki] = k_value;
                q_sum_sq += q_value * q_value;
                k_sum_sq += k_value * k_value;
            }

            const q_inv_norm: f32 = 1.0 / std.math.sqrt(q_sum_sq + 1.0e-6);
            const k_inv_norm: f32 = 1.0 / std.math.sqrt(k_sum_sq + 1.0e-6);
            for (0..key_dim) |ki| {
                q_norm[qh][ki] *= q_inv_norm * scale;
                k_norm[qh][ki] *= k_inv_norm;
            }
        }

        for (0..num_value_heads) |vh| {
            const qh = @divTrunc(vh, qk_head_repetition);
            const decay: f32 = @exp(g_values[gateIndex(t, vh)]);
            const beta_value = beta_values[gateIndex(t, vh)].toF32();

            for (0..value_dim) |vi| {
                for (0..key_dim) |ki| {
                    state[stateIndex(vh, vi, ki)] *= decay;
                }

                var predicted_v: f32 = 0;
                for (0..key_dim) |ki| {
                    predicted_v += state[stateIndex(vh, vi, ki)] * k_norm[qh][ki];
                }

                const delta_v = beta_value * (v_values[inputValueIndex(t, vh, vi)].toF32() - predicted_v);
                for (0..key_dim) |ki| {
                    state[stateIndex(vh, vi, ki)] += delta_v * k_norm[qh][ki];
                }

                var output: f32 = 0;
                for (0..key_dim) |ki| {
                    output += state[stateIndex(vh, vi, ki)] * q_norm[qh][ki];
                }
                out_values[outputIndex(t, vh, vi)] = output;
            }
        }
    }

    @memcpy(ht_values[0..state.len], state);
    return expected;
}

fn fillInputs(q: zml.Slice, k: zml.Slice, v: zml.Slice, g: zml.Slice, beta: zml.Slice, h0: zml.Slice) void {
    const q_values = q.items(bf16);
    const k_values = k.items(bf16);
    const v_values = v.items(bf16);
    const g_values = g.items(f32);
    const beta_values = beta.items(bf16);
    const h0_values = h0.items(f32);

    for (0..seq_len) |t| {
        for (0..num_q_heads) |qh| {
            for (0..key_dim) |ki| {
                q_values[qkIndex(t, qh, ki)] = bf16.fromF32(0.015 + @as(f32, @floatFromInt((t + qh + ki) % 17)) * 0.001);
                k_values[qkIndex(t, qh, ki)] = bf16.fromF32(0.020 + @as(f32, @floatFromInt((3 * t + qh + ki) % 19)) * 0.001);
            }
        }

        for (0..num_value_heads) |vh| {
            for (0..value_dim) |vi| {
                v_values[inputValueIndex(t, vh, vi)] = bf16.fromF32(0.030 + @as(f32, @floatFromInt((5 * t + vh + vi) % 23)) * 0.001);
            }
            g_values[gateIndex(t, vh)] = -0.05 - @as(f32, @floatFromInt((t + vh) % 5)) * 0.005;
            beta_values[gateIndex(t, vh)] = bf16.fromF32(0.35 + @as(f32, @floatFromInt((t + vh) % 4)) * 0.025);
        }
    }

    for (0..num_value_heads) |vh| {
        for (0..value_dim) |vi| {
            for (0..key_dim) |ki| {
                h0_values[stateIndex(vh, vi, ki)] = 0.001 * @as(f32, @floatFromInt((vh + vi + 2 * ki) % 29));
            }
        }
    }
}

fn qShape(dtype: zml.DataType) zml.Shape {
    return zml.Shape.init(.{ .b = batch_size, .s = seq_len, .h = num_q_heads, .d = key_dim }, dtype);
}

fn vShape(dtype: zml.DataType) zml.Shape {
    return zml.Shape.init(.{ .b = batch_size, .s = seq_len, .vh = num_value_heads, .vd = value_dim }, dtype);
}

fn gateShape(dtype: zml.DataType) zml.Shape {
    return zml.Shape.init(.{ .b = batch_size, .s = seq_len, .vh = num_value_heads }, dtype);
}

fn stateShape(dtype: zml.DataType) zml.Shape {
    return zml.Shape.init(.{ .b = batch_size, .vh = num_value_heads, .vd = value_dim, .d = key_dim }, dtype);
}

fn outputShape() zml.Shape {
    return zml.Shape.init(.{ .b = batch_size, .s = seq_len, .vh = num_value_heads, .vd = value_dim }, .f32);
}

fn qkIndex(t: usize, h: usize, k: usize) usize {
    return ((t * num_q_heads + h) * key_dim) + k;
}

fn inputValueIndex(t: usize, h: usize, v: usize) usize {
    return ((t * num_value_heads + h) * value_dim) + v;
}

fn outputIndex(t: usize, h: usize, v: usize) usize {
    return ((t * num_value_heads + h) * value_dim) + v;
}

fn gateIndex(t: usize, h: usize) usize {
    return t * num_value_heads + h;
}

fn stateIndex(h: usize, v: usize, k: usize) usize {
    return ((h * value_dim + v) * key_dim) + k;
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

test "gated deltanet host reference" {
    const allocator = std.testing.allocator;

    var q = try zml.Slice.alloc(allocator, qShape(.bf16));
    defer q.free(allocator);
    var k = try zml.Slice.alloc(allocator, qShape(.bf16));
    defer k.free(allocator);
    var v = try zml.Slice.alloc(allocator, vShape(.bf16));
    defer v.free(allocator);
    var g = try zml.Slice.alloc(allocator, gateShape(.f32));
    defer g.free(allocator);
    var beta = try zml.Slice.alloc(allocator, gateShape(.bf16));
    defer beta.free(allocator);
    var h0 = try zml.Slice.alloc(allocator, stateShape(.f32));
    defer h0.free(allocator);

    fillInputs(q, k, v, g, beta, h0);

    var expected = try computeExpected(allocator, q, k, v, g, beta, h0);
    defer expected.free(allocator);

    try std.testing.expect(std.math.isFinite(expected.output.items(f32)[0]));
    try std.testing.expect(std.math.isFinite(expected.final_state.items(f32)[0]));
}
