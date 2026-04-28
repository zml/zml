const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const bf16 = zml.floats.BFloat16;

// NEURON_RT_LOG_LEVEL=debug NEURON_RT_VISIBLE_CORES=0 bazel run --config=remote --@zml//platforms:neuron=true     --@zml//platforms:cpu=false     //examples/llm --     --model=/var/models/meta-llama/Llama-3.2-1B-Instruct     --prompt="Write one sentence about Zig."     --seqlen=128     --backend=vanilla --topk=1
// NEURON_RT_VISIBLE_CORES=0 bazel run --config=remote --@zml//platforms:neuron=true --@zml//platforms:cpu=false //examples/neuron_nki:attention

// seq_len=128
// q=Tensor({q=1/replicated,h=32/model,hd=64/replicated,bf16})
// k=Tensor({k=128/replicated,h=8/model,hd=64/replicated,bf16})
// v=Tensor({k=128/replicated,h=8/model,hd=64/replicated,bf16})
// attn_mask=Tensor({q=1,k=128,bf16})
// token_index=Tensor({coord=1,u32})
// attn_output=Tensor({q=1,h=32,hd=64,bf16})

/// Self zml.attention.attention.
///   - If token_index is set, x is assumed to be the representation of one new token,
/// and kv_cache will be read for the previous tokens.
///   - If token_index is not set, x is assumed to be the representation of all tokens
const llama_num_heads = 32;
const llama_num_kv_heads = 8;
const llama_head_dim = 64;
const decode_seq_len = 128;

const Args = struct {
    kernel: zml.attention.neuron.Kernel = .decode_tkg,
    token_index: u32 = decode_seq_len - 1,
    warmups: usize = 3,
    iterations: usize = 20,

    pub const help =
        \\ neuron_nki attention --kernel=decode_tkg --iterations=20
        \\
        \\ Options:
        \\   --kernel=<text>  Neuron attention kernel to test: decode_tkg or decode_inhouse.
        \\   --token-index=<n> Decode position inside the fixed 128-token cache.
        \\   --warmups=<n>    Untimed execution warmups for each program.
        \\   --iterations=<n> Timed execution iterations for each program.
        \\
    ;
};

const BenchmarkResult = struct {
    output: zml.Buffer,
    total: stdx.time.Duration,
    average: stdx.time.Duration,
};

const VanillaDecodeAttention = struct {
    fn forward(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
        return zml.attention.attention.attention(
            q,
            k,
            v,
            token_index,
            .{ .vanilla = {} },
            .{ .vanilla = {} },
        );
    }
};

const NeuronAttention = struct {
    kernel: zml.attention.neuron.Kernel,

    fn forward(self: NeuronAttention, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, token_index: zml.Tensor) zml.Tensor {
        return zml.attention.attention.attention(
            q,
            k,
            v,
            token_index,
            .{ .neuron = {} },
            .{ .neuron = .{ .kernel = self.kernel } },
        );
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .init(allocator, io, .neuron, .{});
    defer platform.deinit(allocator, io);

    const sharding = try zml.sharding.replicatedSharding(platform);

    try runDecode(allocator, io, platform, sharding, args);
}

fn runDecode(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, args: Args) !void {
    stdx.debug.assert(args.iterations > 0, "attention benchmark expects at least one timed iteration", .{});
    stdx.debug.assert(args.token_index < decode_seq_len, "token_index ({}) must be less than seq_len ({})", .{ args.token_index, decode_seq_len });

    const q: zml.Tensor = .init(.{
        .q = 1,
        .h = llama_num_heads,
        .hd = llama_head_dim,
    }, .bf16);
    const k: zml.Tensor = .init(.{
        .k = decode_seq_len,
        .h = llama_num_kv_heads,
        .hd = llama_head_dim,
    }, .bf16);
    const v: zml.Tensor = .init(.{
        .k = decode_seq_len,
        .h = llama_num_kv_heads,
        .hd = llama_head_dim,
    }, .bf16);
    const token_index: zml.Tensor = .init(.{}, .u32);

    var vanilla_exe = try platform.compileFn(allocator, io, VanillaDecodeAttention.forward, .{ q, k, v, token_index }, .{ .shardings = &.{sharding} });
    defer vanilla_exe.deinit();

    const neuron_attention: NeuronAttention = .{ .kernel = args.kernel };
    var neuron_exe = try platform.compileFn(allocator, io, NeuronAttention.forward, .{ neuron_attention, q, k, v, token_index }, .{ .shardings = &.{sharding} });
    defer neuron_exe.deinit();

    const q_host = try allocator.alloc(bf16, llama_num_heads * llama_head_dim);
    defer allocator.free(q_host);
    const k_host = try allocator.alloc(bf16, decode_seq_len * llama_num_kv_heads * llama_head_dim);
    defer allocator.free(k_host);
    const v_host = try allocator.alloc(bf16, decode_seq_len * llama_num_kv_heads * llama_head_dim);
    defer allocator.free(v_host);

    fillBf16(q_host, 0.01);
    fillBf16(k_host, 0.02);
    fillBf16(v_host, 0.03);

    var q_buffer = try zml.Buffer.fromBytes(io, platform, q.shape(), sharding, std.mem.sliceAsBytes(q_host));
    defer q_buffer.deinit();
    var k_buffer = try zml.Buffer.fromBytes(io, platform, k.shape(), sharding, std.mem.sliceAsBytes(k_host));
    defer k_buffer.deinit();
    var v_buffer = try zml.Buffer.fromBytes(io, platform, v.shape(), sharding, std.mem.sliceAsBytes(v_host));
    defer v_buffer.deinit();

    var token_index_buffer = try zml.Buffer.scalar(io, platform, args.token_index, .u32, sharding);
    defer token_index_buffer.deinit();

    var vanilla_bench = try benchmarkExecutable(allocator, io, &vanilla_exe, .{
        q_buffer,
        k_buffer,
        v_buffer,
        token_index_buffer,
    }, args.warmups, args.iterations);
    defer vanilla_bench.output.deinit();

    var neuron_bench = try benchmarkExecutable(allocator, io, &neuron_exe, .{
        q_buffer,
        k_buffer,
        v_buffer,
        token_index_buffer,
    }, args.warmups, args.iterations);
    defer neuron_bench.output.deinit();

    try zml.testing.expectClose(io, vanilla_bench.output, neuron_bench.output, .{
        .absolute_tolerance = 1e-2,
        .relative_tolerance = 1e-2,
        .minimum_close_fraction = 0.9995,
    });

    std.log.info("Bench decode attention warmups={} iterations={} token_index={} seq_len={} heads={} kv_heads={} head_dim={}", .{
        args.warmups,
        args.iterations,
        args.token_index,
        decode_seq_len,
        llama_num_heads,
        llama_num_kv_heads,
        llama_head_dim,
    });
    std.log.info("Vanilla total={d:.3}ms avg={d:.3}ms", .{ nsToMs(vanilla_bench.total.ns), nsToMs(vanilla_bench.average.ns) });
    std.log.info("Neuron kernel={} total={d:.3}ms avg={d:.3}ms vanilla_avg={d:.3}ms speedup={d:.3}x output={f}", .{
        args.kernel,
        nsToMs(neuron_bench.total.ns),
        nsToMs(neuron_bench.average.ns),
        nsToMs(vanilla_bench.average.ns),
        @as(f64, @floatFromInt(vanilla_bench.average.ns)) / @as(f64, @floatFromInt(neuron_bench.average.ns)),
        neuron_bench.output.shape(),
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
        var output = exe_results.get(zml.Buffer);
        output.deinit();
    }

    var total: stdx.time.Duration = .{};
    var last_output: ?zml.Buffer = null;
    errdefer if (last_output) |*output| output.deinit();

    for (0..iterations) |_| {
        const start: std.Io.Timestamp = .now(io, .awake);
        exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });
        total.ns += @intCast(start.untilNow(io, .awake).toNanoseconds());

        if (last_output) |*output| output.deinit();
        last_output = exe_results.get(zml.Buffer);
    }

    return .{
        .output = last_output.?,
        .total = total,
        .average = total.div(iterations),
    };
}

fn fillBf16(data: []bf16, offset: f32) void {
    for (data, 0..) |*elem, i| {
        const x: f32 = @as(f32, @floatFromInt(@mod(i, 31))) * 0.001 + offset;
        elem.* = bf16.fromF32(x);
    }
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}
