const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const log = std.log.scoped(.tpu);

const Args = struct {
    model: []const u8,
    prompt: ?[]const u8 = null,
    seqlen: u32 = 2048,
    backend: ?zml.attention.attention.Backend = null,
    single: bool = false,

    pub const help =
        \\ Use llm --model=<path> [options]
        \\
        \\ Run text generation with a model selected from `model_type` in the `config.json`.
        \\
        \\ Options:
        \\   --model=<path>      Path to the model repository (required)
        \\   --prompt=<string>   Prompt to use for generation (default: none)
        \\   --seqlen=<number>   Sequence length (default: 2048)
        \\   --backend=<text>    Attention backend to use ([vanilla, cuda_fa2, cuda_fa3], default: auto-selection)
        \\   --single            Create a single kernel encompassing all the layers when supported 
        \\                       (only used by LFM2 which uses multiple kernels by default)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // `bazel run` executes binaries from Bazel's runfiles tree by default.
    // If available, switch back to the shell's original working directory.
    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);

    //
    // Virtual File Systems
    //
    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer http_client.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var s3_vfs: zml.io.VFS.S3 = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer s3_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());
    try vfs.register("s3", s3_vfs.io());

    const io = vfs.io();

    //
    // Platform and Backend Selection
    //
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    log.info("\n{f}", .{platform.fmtVerbose()});

    //
    // Model initialization
    //
    log.info("Resolving model repository..", .{});
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    log.info("Initializing model..", .{});
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model_mesh: zml.sharding.LogicalMesh = try .init("model", .{ .model = .high_bandwidth });
    const model_sharding_strategy: zml.sharding.Strategy = try .suggest(model_mesh, platform.physical_mesh);
    const sharding: zml.sharding.Sharding = try .initFromStrategy(platform, model_mesh, model_sharding_strategy);

    const layer: SimplifiedLayer = .init(store.view().withPrefix("model.layers.0"));

    var layer_buffers = try layer.loadBuffers(allocator, io, platform, &store, &.{sharding});
    defer SimplifiedLayer.unloadBuffers(&layer_buffers);

    const input: zml.Tensor = .init(.{ .b = 1024, .d = 2048 }, .bf16);

    var exe = try platform.compile(allocator, io, layer, .forward, .{input}, .{ .shardings = &.{sharding}, .program_name = "simplified_layer" });
    defer exe.deinit();

    const rng: zml.Tensor.Rng = .init();
    var rng_buffer = try zml.Tensor.Rng.initBuffer(platform, 0, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffer);

    var input_buffer = try generateRandomBuffer(allocator, io, rng, &rng_buffer, input.shape(), platform, sharding);
    defer input_buffer.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    for (0..10000) |_| {
        exe_args.set(.{ layer_buffers, input_buffer });
        exe.call(exe_args, &exe_results);
        exe_results.fill(.{&input_buffer});
    }
}

pub fn generateRandomBufferKernel(rng: zml.Tensor.Rng, shape: zml.Shape) struct { zml.Tensor.Rng, zml.Tensor } {
    const new_rng, const tensor = rng.uniform(shape, .{});

    return .{ new_rng, tensor };
}

pub fn generateRandomBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    rng: zml.Tensor.Rng,
    rng_buffer: *zml.Bufferized(zml.Tensor.Rng),
    shape: zml.Shape,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    var exe = try platform.compileFn(allocator, io, generateRandomBufferKernel, .{ rng, shape }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(.{rng_buffer});
    exe.call(args, &results);

    var out: zml.Buffer = undefined;
    results.fill(.{ rng_buffer, &out });

    return out;
}

const SimplifiedLayer = struct {
    mlp: Mlp,
    norm: RmsNorm,

    pub fn init(store: zml.io.TensorStore.View) SimplifiedLayer {
        return .{
            .mlp = .init(store.withPrefix("mlp")),
            .norm = .init(store.withPrefix("input_layernorm"), 1e-6),
        };
    }

    pub fn loadBuffers(
        self: *const SimplifiedLayer,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(SimplifiedLayer) {
        return zml.io.load(SimplifiedLayer, self, allocator, io, platform, store, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .shardings = shardings,
            .parallelism = 16,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SimplifiedLayer)) void {
        Mlp.unloadBuffers(&self.mlp);
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(self: SimplifiedLayer, x: zml.Tensor) zml.Tensor {
        const normalized = self.norm.forward(x);
        const out = self.mlp.forward(normalized);
        return out.reuseBuffer(x);
    }
};

const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = .init(store.createTensor("up_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .gate_proj = .init(store.createTensor("gate_proj.weight", .{ .dout, .d }, .{ .dout = .model }), null, .d),
            .down_proj = .init(store.createTensor("down_proj.weight", .{ .dout, .d }, .{ .d = .model }), null, .d),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        self.up_proj.weight.deinit();
        if (self.up_proj.bias) |*bias| bias.deinit();
        self.gate_proj.weight.deinit();
        if (self.gate_proj.bias) |*bias| bias.deinit();
        self.down_proj.weight.deinit();
        if (self.down_proj.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const proj = self.up_proj.forward(x);
        var output = self.gate_proj.forward(x);
        output = output.silu().mul(proj).rename(.{ .dout = .d });
        return self.down_proj.forward(output);
    }
};
