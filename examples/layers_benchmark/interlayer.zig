const std = @import("std");

const clap = @import("clap");
const zml = @import("zml");
const Shape = zml.Shape;

const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const floats = zml.floats;

const log = std.log.scoped(.qwen);
pub const std_options: std.Options = .{
    .log_level = .info,
};

const builtin = @import("builtin");

const CudaTracer = struct {

    // Those symbols are defined in cudaProfiler.h but their implementation is in libcuda.so
    // They will be bound at call time after libcuda.so is loaded (as a needed dependency of libpjrt_cuda.so).
    const cuProfilerStart = @extern(*const fn () callconv(.c) c_int, .{ .name = "cuProfilerStart", .linkage = .weak }) orelse unreachable;
    const cuProfilerStop = @extern(*const fn () callconv(.c) c_int, .{ .name = "cuProfilerStop", .linkage = .weak }) orelse unreachable;

    // Those symbols are defined in nvToolsExt.h which we don't want to provide.
    // However, we link with libnvToolsExt.so which provides them.
    // They will be bound at call time after libnvToolsExt.so is loaded (manually dlopen'ed by us).
    const nvtxMarkA = @extern(*const fn ([*:0]const u8) callconv(.c) void, .{ .name = "nvtxMarkA", .linkage = .weak }) orelse unreachable;
    const nvtxRangeStartA = @extern(*const fn ([*:0]const u8) callconv(.c) c_int, .{ .name = "nvtxRangeStartA", .linkage = .weak }) orelse unreachable;
    const nvtxRangeEnd = @extern(*const fn (c_int) callconv(.c) void, .{ .name = "nvtxRangeEnd", .linkage = .weak }) orelse unreachable;

    pub fn init(name: [:0]const u8) CudaTracer {
        _ = name;
        _ = cuProfilerStart();
        return .{};
    }

    pub fn deinit(self: *const CudaTracer) void {
        _ = self;
        _ = cuProfilerStop();
    }

    pub fn event(self: *const CudaTracer, message: [:0]const u8) void {
        _ = self;
        nvtxMarkA(message.ptr);
    }

    pub fn frameStart(self: *const CudaTracer, message: [:0]const u8) u64 {
        _ = self;
        return @intCast(nvtxRangeStartA(message.ptr));
    }

    pub fn frameEnd(self: *const CudaTracer, interval_id: u64, message: [:0]const u8) void {
        _ = self;
        _ = message;
        nvtxRangeEnd(@intCast(interval_id));
        return;
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-5,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensorWithTags("weight", .{.d}), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    const Config = struct {
        up_proj_shape: Shape,
        gate_proj_shape: Shape,
        down_proj_shape: Shape,
    };

    pub fn init(store: zml.io.TensorStore.View) !Mlp {
        const up_proj = store.createTensorWithTags("model.layers.0.mlp.up_proj.weight", .{ .dout, .d });
        const gate_proj = store.createTensorWithTags("model.layers.0.mlp.gate_proj.weight", .{ .dout, .d });
        const down_proj = store.createTensorWithTags("model.layers.0.mlp.down_proj.weight", .{ .d, .dout });
        return Mlp{
            .up_proj = zml.nn.Linear.init(up_proj, null, .d),
            .gate_proj = zml.nn.Linear.init(gate_proj, null, .d),
            .down_proj = zml.nn.Linear.init(down_proj, null, .dout),
        };
    }

    pub fn deinit(self: *Mlp) void {
        _ = self;
    }

    pub fn loadBuffers(self: Mlp, allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, platform: zml.Platform) !zml.Bufferized(Mlp) {
        const up_proj = try zml.io.loadBuffersFromId(allocator, io, self.up_proj, store.withPrefix("model.layers.0.mlp.up_proj.weight"), platform);
        const gate_proj = try zml.io.loadBuffersFromId(allocator, io, self.gate_proj, store.withPrefix("model.layers.0.mlp.gate_proj.weight"), platform);
        const down_proj = try zml.io.loadBuffersFromId(allocator, io, self.down_proj, store.withPrefix("model.layers.0.mlp.down_proj.weight"), platform);

        return .{
            .up_proj = up_proj,
            .gate_proj = gate_proj,
            .down_proj = down_proj,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        self.up_proj.weight.deinit();
        self.gate_proj.weight.deinit();
        self.down_proj.weight.deinit();
    }

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const proj = self.up_proj.forward(x);
        var output = self.gate_proj.forward(x);
        log.info("output shape: {f}", .{output.shape()});
        output = output.silu().mul(proj);
        log.info("output after silu and mul shape: {f}", .{output.shape()});
        const result = self.down_proj.forward(output);
        log.info("result shape: {f}", .{result.shape()});
        return result;
    }
};

pub const InterLayer = struct {
    o_proj: zml.nn.Linear,
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,

    post_attention_layernorm: RmsNorm,
    pre_attn_layer_norm: RmsNorm,

    mlp: Mlp,

    num_heads: u32 = 64,
    num_kv_heads: u32 = 8,
    head_dim: u32 = 128,

    pub fn init(store: zml.io.TensorStore.View) !InterLayer {
        const q_w = store.createTensorWithTags("model.layers.0.self_attn.q_proj.weight", .{ .d, .d });
        const k_w = store.createTensorWithTags("model.layers.0.self_attn.k_proj.weight", .{ .dh, .d });
        const v_w = store.createTensorWithTags("model.layers.0.self_attn.v_proj.weight", .{ .dh, .d });
        const o_w = store.createTensorWithTags("model.layers.0.self_attn.o_proj.weight", .{ .d, .d });

        const mlp = try Mlp.init(store);

        return InterLayer{
            .q_proj = zml.nn.Linear.init(
                q_w,
                null,
                .d,
            ),
            .k_proj = zml.nn.Linear.init(
                k_w,
                null,
                .d,
            ),
            .v_proj = zml.nn.Linear.init(
                v_w,
                null,
                .d,
            ),
            .o_proj = zml.nn.Linear.init(
                o_w,
                null,
                .d,
            ),

            .pre_attn_layer_norm = RmsNorm.init(
                store.withPrefix("model.layers.0.input_layernorm"),
                1e-6,
            ),
            .post_attention_layernorm = RmsNorm.init(
                store.withPrefix("model.layers.0.post_attention_layernorm"),
                1e-6,
            ),

            .mlp = mlp,
        };
    }

    pub fn deinit(self: *InterLayer) void {
        _ = self;
    }

    pub fn loadBuffers(self: InterLayer, allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, platform: zml.Platform) !zml.Bufferized(InterLayer) {
        const q_proj = try zml.io.loadBuffersFromId(allocator, io, self.q_proj, store.withPrefix("model.layers.0.self_attn.q_proj.weight"), platform);
        const k_proj = try zml.io.loadBuffersFromId(allocator, io, self.k_proj, store.withPrefix("model.layers.0.self_attn.k_proj.weight"), platform);
        const v_proj = try zml.io.loadBuffersFromId(allocator, io, self.v_proj, store.withPrefix("model.layers.0.self_attn.v_proj.weight"), platform);
        const o_proj = try zml.io.loadBuffersFromId(allocator, io, self.o_proj, store.withPrefix("model.layers.0.self_attn.o_proj.weight"), platform);

        const pre_attn_layer_norm = try zml.io.loadBuffersFromId(allocator, io, self.pre_attn_layer_norm, store.withPrefix("model.layers.0.input_layernorm.weight"), platform);
        const post_attention_layernorm = try zml.io.loadBuffersFromId(allocator, io, self.post_attention_layernorm, store.withPrefix("model.layers.0.post_attention_layernorm.weight"), platform);

        const mlp = try self.mlp.loadBuffers(allocator, io, store, platform);

        return .{
            .q_proj = q_proj,
            .k_proj = k_proj,
            .v_proj = v_proj,
            .o_proj = o_proj,
            .pre_attn_layer_norm = pre_attn_layer_norm,
            .post_attention_layernorm = post_attention_layernorm,
            .mlp = mlp,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(InterLayer)) void {
        self.q_proj.weight.deinit();
        self.k_proj.weight.deinit();
        self.v_proj.weight.deinit();
        self.o_proj.weight.deinit();
        self.pre_attn_layer_norm.weight.deinit();
        self.post_attention_layernorm.weight.deinit();
        Mlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(
        self: InterLayer,
        x0: Tensor, // [B,  d]
        attn: Tensor, // [B, h, hd]
        token_pos: Tensor, // [B, S]
    ) struct { Tensor, Tensor, Tensor, Tensor } {
        const attn_flat = attn.mergeTranspose(.{ .h, .hd }, .d);
        const x1 = x0.add(self.o_proj.forward(attn_flat));

        const x1_norm = self.post_attention_layernorm.forward(x1);
        const x2 = self.mlp.forward(x1_norm).add(x1);

        log.info("x2 shape after MLP: {f}", .{x2.shape()});

        const x_next = self.pre_attn_layer_norm.forward(x2);

        log.info("x_next shape after Pre-Attn Norm: {f}", .{x_next.shape()});

        var q = self.q_proj.forward(x_next);
        var k = self.k_proj.forward(x_next);
        const v = self.v_proj.forward(x_next);

        log.info("q shape after Q_Proj: {f}", .{q.shape()});
        log.info("k shape after K_Proj: {f}", .{k.shape()});
        log.info("v shape after V_Proj: {f}", .{v.shape()});

        q = q.splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim });
        k = k.splitAxis(.dh, .{ .h = self.num_kv_heads, .hd = self.head_dim });

        const rope_opts = zml.nn.RopeOpts{ .freq_base = 10000.0 };
        q = zml.nn.rope(q, token_pos, rope_opts);
        k = zml.nn.rope(k, token_pos, rope_opts);

        return .{ x2, q, k, v };
    }
};

const params = clap.parseParamsComptime(
    \\--help                      print this help
    \\--hf-model-path  <STRING>   path to the directory containing model weights, config and tokenizer
    \\--seq-len        <UINT>     sequence length (default: 512)
);

fn bool_parser(in: []const u8) error{}!bool {
    return std.mem.indexOfScalar(u8, "tTyY1", in[0]) != null;
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    zml.init();
    defer zml.deinit();

    const parsers = comptime .{
        .BOOL = bool_parser,
        .UINT = clap.parsers.int(u32, 0),
        .STRING = clap.parsers.string,
        .PATH = clap.parsers.string,
    };
    var diag: clap.Diagnostic = .{};
    var stderr_buffer: [1024]u8 = undefined;
    var stderr = std.Io.File.stderr().writer(io, &stderr_buffer);
    defer stderr.interface.flush() catch {};

    var cli = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        diag.report(&stderr.interface, err) catch {};
        stderr.interface.writeAll("usage: ") catch {};
        clap.usage(&stderr.interface, clap.Help, &params) catch {};
        stderr.interface.writeAll("\n") catch {};
        return;
    };
    defer cli.deinit();

    if (cli.args.help != 0) {
        clap.help(&stderr.interface, clap.Help, &params, .{}) catch {};
        return;
    }

    const hf_model_path = cli.args.@"hf-model-path" orelse {
        log.err("Missing --hf-model-path", .{});
        return;
    };

    const compilation_options = zml.CompilationOptions{
        //.xla_dump_to = "/tmp/zml/mlp_capacity",
        .sharding_enabled = false,
    };
    _ = compilation_options; // autofix

    // const config_json_file = try repo.dir.openFile(io, "config.json", .{});
    // defer config_json_file.close(io);
    // var config_json_buffer: [256]u8 = undefined;
    // var config_reader = config_json_file.reader(io, &config_json_buffer);
    // var reader: std.json.Reader = .init(allocator, &config_reader.interface);
    // defer reader.deinit();

    // const parsed_config = try std.json.parseFromTokenSource(Mlp.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
    // defer parsed_config.deinit();
    // const config = parsed_config.value;
    // _ = config; // autofix

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, hf_model_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var platform: zml.Platform = try .auto(io, .{});
    defer platform.deinit();

    var layers_tensors: InterLayer = try .init(store.view());

    defer layers_tensors.deinit();

    var layers_model_buffers = try layers_tensors.loadBuffers(allocator, io, store.view(), platform);
    defer InterLayer.unloadBuffers(&layers_model_buffers);

    const up_proj_shape = layers_tensors.mlp.up_proj.weight.shape();
    log.info("MLP shapes: {f}", .{up_proj_shape});

    // const mlp_shapes = try zml.Tensor.shapesOf(mlp_tensors, compiler_arena.allocator());
    const batch_sizes = [_]u32{ 1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048 };

    const in_features = up_proj_shape.dim(1);

    const tracer = CudaTracer.init("mlp_tracer");
    defer tracer.deinit();

    for (batch_sizes, 0..) |batch_size, j| {
        log.info(" Test avec batch_size={d}...", .{batch_size});
        log.info(" Weight dtype: {f}", .{up_proj_shape});

        var rng_state = std.Random.DefaultPrng.init(42);
        const random = rng_state.random();
        var current_random_data: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .b = batch_size, .d = in_features }, .bf16));
        defer current_random_data.free(allocator);
        var attn_random_data: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .b = batch_size, .h = layers_tensors.num_heads, .hd = layers_tensors.head_dim }, .bf16));
        defer attn_random_data.free(allocator);
        var token_pos_data: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .b = batch_size }, .u32));
        defer token_pos_data.free(allocator);
        log.info(" Filling input with random data...", .{});

        for (current_random_data.items(floats.BFloat16)) |*val| {
            val.* = floats.BFloat16.fromF32(random.float(f32) * 2.0 - 1.0);
        }
        for (attn_random_data.items(floats.BFloat16)) |*val| {
            val.* = floats.BFloat16.fromF32(random.float(f32) * 2.0 - 1.0);
        }
        for (token_pos_data.items(u32)) |*val| {
            val.* = 1;
        }

        log.info("current_data content : {any}", .{current_random_data.items(floats.BFloat16)[0..10]});
        log.info("attn_random_data content : {any}", .{attn_random_data.items(floats.BFloat16)[0..10]});
        log.info("token_pos_data content : {any}", .{token_pos_data.items(u32)[0..1]});

        log.info("Buffer preparation done.", .{});

        const input_buffer: zml.Buffer = try .fromSlice(io, platform, current_random_data);
        defer input_buffer.deinit();

        const attn_buffer: zml.Buffer = try .fromSlice(io, platform, attn_random_data);
        defer attn_buffer.deinit();

        const token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_data);
        defer token_pos_buffer.deinit();

        const input_tensor: zml.Tensor = .init(current_random_data.shape, .bf16);
        const attn_tensor: zml.Tensor = .init(attn_random_data.shape, .bf16);
        const token_pos_tensor: zml.Tensor = .init(token_pos_data.shape, .u32);

        log.info("Compilation started...", .{});
        log.info(" Input tensor shape: {f}", .{input_tensor.shape()});
        var exe = try platform.compileModel(allocator, io, InterLayer.forward, layers_tensors, .{ input_tensor, attn_tensor, token_pos_tensor });
        defer exe.deinit();
        log.info("Compilation finished.", .{});

        var args = try exe.args(allocator);
        defer args.deinit(allocator);

        var results = try exe.results(allocator);
        defer results.deinit(allocator);

        args.set(.{ layers_model_buffers, input_buffer, attn_buffer, token_pos_buffer });

        log.info("Execution started...", .{});

        // Warmup
        {
            const message: [:0]const u8 = "Warmup Run";
            const trace = tracer.frameStart(message);
            exe.call(args, &results);
            const output = results.get(zml.Buffer);
            defer output.deinit();
            tracer.frameEnd(trace, message);
        }

        const num_runs = 30;
        const tracer_buffer = try allocator.alloc(u8, 1024);
        defer allocator.free(tracer_buffer);

        for (0..num_runs) |i| {
            _ = i; // autofix

            args = try exe.args(allocator);

            results = try exe.results(allocator);

            args.set(.{ layers_model_buffers, input_buffer, attn_buffer, token_pos_buffer });

            const message: [:0]const u8 = try std.fmt.bufPrintZ(tracer_buffer, "{} : Mlp with batch_size={}", .{ j, batch_size });
            const trace = tracer.frameStart(message);

            exe.call(args, &results);
            const output = results.get(zml.Buffer);
            defer output.deinit();

            const output_slice = try output.toSliceAlloc(allocator, io);
            defer output_slice.free(allocator);
            tracer.frameEnd(trace, message);
        }

        log.info("Batch {d} finished", .{batch_size});
    }

    log.info("Iteration terminee", .{});
}
