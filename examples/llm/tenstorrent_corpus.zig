//! Op-selection corpus for the Tenstorrent backend (DESIGN 12.2).
//!
//! Compiles the building blocks of the Llama path, a two-layer Llama-3.1-8B
//! configuration of this example, and the one-layer prefill and decode
//! programs of M1 (llama_layer.zig), for tensor-parallel degrees 1, 2, 4 and 8
//! against the plugin's device-free topologies. Every program is saved as ZML
//! hands it to PJRT, before the plugin compiles it (`pjrt_program_dump_dir`),
//! and `<topology>/manifest.jsonl` records each one with the plugin's answer:
//! compiled, or the exact error. Most programs are refused today; that is part
//! of the record.

const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const zml = @import("zml");
const Tensor = zml.Tensor;

const llama_layer = @import("llama_layer.zig");
const common = @import("models/common.zig");
const llama = @import("models/llama.zig");
const pjrt_error_log = @import("pjrt_error_log.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = pjrt_error_log.logFn,
};

const Args = struct {
    out: []const u8,
    tp: ?u32 = null,

    pub const help =
        \\ Use tenstorrent_corpus --out=<dir> [--tp=<1|2|4|8>]
        \\
        \\ Writes <dir>/<topology>/<program>.mlirbc, .mlir and .compile_options.pb, and <dir>/<topology>/manifest.jsonl,
        \\ for every topology (tp1, tp2, tp4, tp4_2x2, tp8) or those of tensor-parallel degree --tp.
        \\
    ;
};

/// Llama-3.1-8B, as its config.json, cut to two layers.
const llama_config_json =
    \\{
    \\  "bos_token_id": 128000,
    \\  "eos_token_id": [128001, 128008, 128009],
    \\  "head_dim": 128,
    \\  "hidden_size": 4096,
    \\  "intermediate_size": 14336,
    \\  "max_position_embeddings": 131072,
    \\  "num_attention_heads": 32,
    \\  "num_hidden_layers": 2,
    \\  "num_key_value_heads": 8,
    \\  "rms_norm_eps": 1e-05,
    \\  "rope_scaling": {
    \\    "factor": 8.0,
    \\    "high_freq_factor": 4.0,
    \\    "low_freq_factor": 1.0,
    \\    "original_max_position_embeddings": 8192,
    \\    "rope_type": "llama3"
    \\  },
    \\  "rope_theta": 500000.0,
    \\  "tie_word_embeddings": false,
    \\  "vocab_size": 128256
    \\}
;
const intermediate_size = 14336;
const vocab_size = 128256;
/// A multiple of 32, as the Tenstorrent arm of this example requires.
const seqlen = 1024;
/// examples/llm's default sampling.
const topk = 4;

const Topology = struct {
    name: []const u8,
    tp: u32,
    fixture: []const u8,
    /// Forces the view; empty takes the plugin's default.
    mesh_shape: []const i64 = &.{},
};

const topologies = [_]Topology{
    .{ .name = "tp1", .tp = 1, .fixture = "blackhole_P150.yaml" },
    .{ .name = "tp2", .tp = 2, .fixture = "box_2xp150_capture.yaml" },
    .{ .name = "tp4", .tp = 4, .fixture = "blackhole_4xP150.yaml" },
    // The same four cards as a 2x2 view: tensor parallelism folds both axes.
    .{ .name = "tp4_2x2", .tp = 4, .fixture = "blackhole_4xP150.yaml", .mesh_shape = &.{ 2, 2 } },
    .{ .name = "tp8", .tp = 8, .fixture = "blackhole_8xP150.yaml" },
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(io, build_working_directory, .{});
        defer working_dir.close(io);
        try std.process.setCurrentDir(io, working_dir);
    }
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    var out_dir = try std.Io.Dir.cwd().createDirPathOpen(io, args.out, .{});
    out_dir.close(io);
    const out_path = try std.Io.Dir.cwd().realPathFileAlloc(io, args.out, allocator);
    defer allocator.free(out_path);

    for (topologies) |topology| {
        if (args.tp) |tp| if (tp != topology.tp) continue;
        try captureTopology(allocator, io, out_path, topology);
    }
}

fn captureTopology(allocator: std.mem.Allocator, io: std.Io, out_path: []const u8, topology: Topology) !void {
    var fixture_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const r = try zml.bazel.runfiles(bazel_builtin.current_repository);
    var name_buf: [256]u8 = undefined;
    const fixture_name = try std.fmt.bufPrint(&name_buf, "zml/platforms/tenstorrent/fixtures/{s}", .{topology.fixture});
    const fixture = try r.rlocation(fixture_name, &fixture_buf) orelse return error.FileNotFound;

    const dir_name = topology.name;
    const dump_dir = try std.Io.Dir.path.join(allocator, &.{ out_path, dir_name });
    defer allocator.free(dump_dir);

    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, dump_dir, .{});
    defer dir.close(io);
    const manifest_file = try dir.createFile(io, "manifest.jsonl", .{});
    defer manifest_file.close(io);
    var manifest_buf: [4096]u8 = undefined;
    var manifest_writer = manifest_file.writer(io, &manifest_buf);
    var manifest: Manifest = .{ .writer = &manifest_writer.interface };

    const platform: *zml.Platform = try .init(allocator, io, .tenstorrent, .{
        .tenstorrent = .{ .runtime = .fake, .cluster_descriptor = fixture, .mesh_shape = topology.mesh_shape },
        .pjrt_program_dump_dir = dump_dir,
    });
    defer platform.deinit(allocator, io);
    if (platform.devices.len != topology.tp) {
        std.log.err("{s} serves {d} devices, expected {d}", .{ topology.fixture, platform.devices.len, topology.tp });
        return error.UnexpectedTopology;
    }
    const meshes: common.Meshes = try .init(platform);
    if (meshes.model.numPartitions() != topology.tp) {
        std.log.err("{s}: the model mesh has {d} partitions, expected {d}", .{ topology.name, meshes.model.numPartitions(), topology.tp });
        return error.UnexpectedTopology;
    }

    const info: zml.platform.tenstorrent.DeviceInfo = try .init(platform, platform.devices[0]);
    var ctx: Context = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .meshes = meshes,
        .topology = topology,
        .dir_name = dir_name,
        .view = info.view,
        .manifest = &manifest,
    };
    try ctx.captureBlocks();
    try ctx.captureLlama();
    try ctx.captureLayer();
}

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    meshes: common.Meshes,
    topology: Topology,
    dir_name: []const u8,
    view: zml.platform.tenstorrent.View,
    manifest: *Manifest,

    fn model(self: *const Context) *const zml.Mesh {
        return self.meshes.model;
    }

    /// Compiles `func` on `args` and records the outcome; a refusal is data, not a failure.
    fn capture(self: *Context, comptime kind: []const u8, name: []const u8, comptime func: anytype, args: std.meta.ArgsTuple(@TypeOf(func))) !void {
        pjrt_error_log.reset();
        const result = self.platform.compileFn(self.allocator, self.io, func, args, .{
            .meshes = &self.meshes.all(),
            .program_name = name,
        });
        try self.record(kind, name, tensorShapes(args), result);
    }

    fn record(self: *Context, comptime kind: []const u8, name: []const u8, inputs: ShapeList, result: anytype) !void {
        var outcome: Outcome = .compiled;
        if (result) |exe| {
            var compiled = exe;
            compiled.deinit();
        } else |err| {
            const message = pjrt_error_log.get();
            // An error raised before PJRT (ZML tracing, sharding) carries no plugin message.
            outcome = .{ .refused = .{ .err = err, .message = if (message.len > 0) message else "(no PJRT message: the compile failed in ZML before reaching the plugin)" } };
        }
        try self.manifest.write(self, kind, name, inputs, outcome);
        switch (outcome) {
            .compiled => std.log.info("{s} {s}: compiled", .{ self.topology.name, name }),
            .refused => |r| std.log.info("{s} {s}: {s} {s}", .{ self.topology.name, name, @errorName(r.err), r.message }),
        }
    }

    fn captureBlocks(self: *Context) !void {
        const model_mesh = self.model();
        const dtype: zml.DataType = .bf16;
        const d = 4096;
        const h = 32;
        const hkv = 8;
        const hd = 128;

        for ([_]struct { []const u8, i64 }{ .{ "decode", 1 }, .{ "prefill", seqlen } }) |phase| {
            const phase_name, const tokens = phase;
            var name_buf: [64]u8 = undefined;

            // The residual stream is replicated before every norm on the Llama path.
            const x = Tensor.fromShape(zml.Shape.init(.{ .s = tokens, .d = d }, dtype).withPartitioning(model_mesh, .{ .d = .replicated }));
            const norm_weight = Tensor.fromShape(zml.Shape.init(.{ .d = d }, dtype).withPartitioning(model_mesh, .{ .d = .replicated }));
            try self.capture("block", try std.fmt.bufPrint(&name_buf, "block_rmsnorm_{s}", .{phase_name}), Blocks.rmsNorm, .{ x, norm_weight });

            const q = Tensor.fromShape(zml.Shape.init(.{ .s = tokens, .h = h, .hd = hd }, dtype).withPartitioning(model_mesh, .{ .h = .model }));
            const k = Tensor.fromShape(zml.Shape.init(.{ .s = tokens, .h = hkv, .hd = hd }, dtype).withPartitioning(model_mesh, .{ .h = .model }));
            const token_index = Tensor.init(.{}, .u32);
            try self.capture("block", try std.fmt.bufPrint(&name_buf, "block_rotary_{s}", .{phase_name}), Blocks.rotary, .{ q, k, token_index });

            const q_attn = Tensor.fromShape(zml.Shape.init(.{ .q = tokens, .h = h, .hd = hd }, dtype).withPartitioning(model_mesh, .{ .h = .model }));
            // Keys and values as SelfAttn reads them from one layer of the cache: heads before positions.
            const kv_attn = llama.model.KvCache.platformShape(.tenstorrent, zml.Shape.init(.{ .layer = 1, .k = seqlen, .h = hkv, .hd = hd }, dtype))
                .drop(.layer).withPartitioning(model_mesh, .{ .h = .model });
            try self.capture("block", try std.fmt.bufPrint(&name_buf, "block_attention_{s}", .{phase_name}), Blocks.attention, .{ q_attn, Tensor.fromShape(kv_attn), Tensor.fromShape(kv_attn), token_index });

            const embedding = Tensor.fromShape(zml.Shape.init(.{ .voc = vocab_size, .d = d }, dtype).withPartitioning(model_mesh, .{ .d = .model }));
            const token_ids = Tensor.init(.{ .s = tokens }, .u32);
            try self.capture("block", try std.fmt.bufPrint(&name_buf, "block_embedding_{s}", .{phase_name}), Blocks.embedding, .{ embedding, token_ids });

            const cache = llama.model.KvCache.init(
                llama.model.KvCache.platformShape(.tenstorrent, zml.Shape.init(.{ .layer = 2, .k = seqlen, .h = hkv, .hd = hd }, dtype)),
                model_mesh,
            );
            const new_kv = zml.Shape.init(.{ .k = tokens, .h = hkv, .hd = hd }, dtype).withPartitioning(model_mesh, .{ .h = .model });
            try self.capture("block", try std.fmt.bufPrint(&name_buf, "block_kv_cache_update_{s}", .{phase_name}), Blocks.kvCacheUpdate, .{ cache, Tensor.fromShape(new_kv), Tensor.fromShape(new_kv), token_index, 1 });
        }

        // Sampling reads logits as the LM head leaves them, vocabulary sharded,
        // and replicated as a smaller variant of the same idioms.
        for ([_]bool{ true, false }) |sharded| {
            const variant_name = if (sharded) "vocab_sharded" else "replicated";
            var name_buf: [64]u8 = undefined;
            const logits_shape = zml.Shape.init(.{ .s = 1, .voc = vocab_size }, .bf16);
            const logits = Tensor.fromShape(if (sharded)
                logits_shape.withPartitioning(model_mesh, .{ .voc = .model })
            else
                logits_shape.withPartitioning(model_mesh, .{ .voc = .replicated }));
            try self.capture("block", try std.fmt.bufPrint(&name_buf, "block_topk_{s}", .{variant_name}), Blocks.topK, .{logits});
            try self.capture("block", try std.fmt.bufPrint(&name_buf, "block_argmax_{s}", .{variant_name}), Blocks.argMax, .{logits});
            try self.capture("block", try std.fmt.bufPrint(&name_buf, "block_sampling_{s}", .{variant_name}), Blocks.sampling, .{ logits, Tensor.Rng.init() });
        }
    }

    fn captureLlama(self: *Context) !void {
        const allocator = self.allocator;
        const parsed = try std.json.parseFromSlice(llama.Config, allocator, llama_config_json, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const config = parsed.value;

        var registry = try llamaRegistry(allocator, config);
        defer registry.deinit();
        var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &self.meshes.all());
        defer store.deinit();

        const mdl: llama.Model = try .init(allocator, store.view(), config, .{
            .sampling_strategy = .{ .topk = topk },
            .max_seq_len = config.max_position_embeddings,
        });
        defer mdl.deinit(allocator);

        const backend: zml.attention.Backend = .auto(self.platform);
        const params: llama.CompilationParameters = .init(mdl, config, seqlen, backend, self.meshes, self.platform.target);
        var progress: std.Progress.Node = .none;
        for ([_]common.Phase{ .prefill, .decode }) |phase| {
            pjrt_error_log.reset();
            const tokens: usize = if (phase == .prefill) seqlen else 1;
            const attention_parameters = if (phase == .prefill) params.prefill_attention_parameters else params.decode_attention_parameters;
            const result = llama.inference.compileKernel(allocator, self.io, self.platform, mdl, params, tokens, attention_parameters, phase, &progress);
            const inputs: ShapeList = tensorShapes(.{ Tensor.init(.{ .s = tokens }, .u32), params.token_index, params.kv_cache, params.rng });
            try self.record("llama_2_layers", phase.programName("llama", "forward"), inputs, result);
        }
    }

    /// The M1 programs: layer 0 alone, prefill of 128 tokens and decode of one.
    fn captureLayer(self: *Context) !void {
        const allocator = self.allocator;
        const parsed = try std.json.parseFromSlice(llama.Config, allocator, llama_config_json, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const config = parsed.value;

        var registry = try llamaRegistry(allocator, config);
        defer registry.deinit();
        var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &self.meshes.all());
        defer store.deinit();
        const layer: llama.model.TransformerLayer = try .init(store.view().withPrefix("model").withPrefix("layers").withLayer(0), config);

        const backend: zml.attention.Backend = .auto(self.platform);
        for ([_]common.Phase{ .prefill, .decode }) |phase| {
            const input: llama_layer.Input = .init(layer, config, phase, .bf16, self.meshes, self.platform.target, backend);
            try self.capture("llama_layer", llama_layer.programName(phase), llama_layer.forward, .{input});
        }
    }
};

const Blocks = struct {
    const eps = 1e-5;

    /// examples/llm RmsNorm.forward.
    fn rmsNorm(x: Tensor, weight: Tensor) Tensor {
        return zml.nn.rmsNorm(x, .d, eps).mul(weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }

    /// examples/llm SelfAttn.forward up to the cache: positions from the token index, HF halves layout.
    fn rotary(q: Tensor, k: Tensor, token_index: Tensor) [2]Tensor {
        const pos_index = b: {
            const temp = Tensor.arange(.{ .end = q.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = q.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };
        const opts: zml.nn.RopeOpts = .{
            .layout = .real_im_pass,
            .scaling = .{ .llama3 = .{
                .factor = 8,
                .high_freq_factor = 4,
                .low_freq_factor = 1,
                .original_max_position_embeddings = 8192,
                .rope_theta = 500000,
            } },
        };
        // Heads stay sharded, as they flow into the cache and attention on the Llama path.
        return .{
            zml.nn.rope(q, pos_index, opts).withPartitioning(.model, .{ .s = .replicated, .h = .model, .hd = .replicated }),
            zml.nn.rope(k, pos_index, opts).withPartitioning(.model, .{ .s = .replicated, .h = .model, .hd = .replicated }),
        };
    }

    /// zml.attention with the plain-StableHLO backend (nn.sdpa), causal, heads sharded.
    fn attention(q: Tensor, k: Tensor, v: Tensor, token_index: Tensor) Tensor {
        return zml.attention.attention(q, k, v, token_index, .{ .stablehlo = {} }, .{ .stablehlo = {} }, zml.Compiler.current().mesh(.model), .head(.model));
    }

    /// examples/llm EmbedTokens.forward.
    fn embedding(weight: Tensor, tokens: Tensor) Tensor {
        const table: zml.nn.TokenEmbedding = .{ .weight = weight };
        return table.forward(tokens.withPartialTags(.{.s})).withPartialTags(.{.d}).replicate();
    }

    /// examples/llm KvCache.updateAt for one layer, in place.
    fn kvCacheUpdate(cache: llama.model.KvCache, k: Tensor, v: Tensor, token_index: Tensor, layer: u32) llama.model.KvCache {
        return cache.updateAt(k, v, token_index, .scalar(layer, .u32)).reuseBuffer(cache);
    }

    fn topK(logits: Tensor) [2]Tensor {
        const result = logits.topK(.{ .topk = .voc }, topk, .{});
        return .{ result.values, result.indices };
    }

    fn argMax(logits: Tensor) [2]Tensor {
        const result = logits.argMax(.voc);
        return .{ result.values, result.indices };
    }

    /// examples/llm LmHead.forward after the projection.
    fn sampling(logits: Tensor, rng: Tensor.Rng) struct { Tensor, Tensor.Rng } {
        const tokens, const next_rng = zml.nn.sampleTokens(logits, .{ .topk = topk }, rng);
        return .{ tokens.convert(.u32), next_rng };
    }
};

/// Two-layer Llama weights: shapes only, nothing is ever read.
fn llamaRegistry(allocator: std.mem.Allocator, config: llama.Config) !zml.safetensors.TensorRegistry {
    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    errdefer registry.deinit();
    const d: i64 = config.hidden_size;
    const hd: i64 = config.head_dim.?;
    const q_out: i64 = config.num_attention_heads * hd;
    const kv_out: i64 = config.num_key_value_heads * hd;

    const Entry = struct { []const u8, []const i64 };
    const add = struct {
        fn add(reg: *zml.safetensors.TensorRegistry, name: []const u8, dims: []const i64) !void {
            try reg.registerTensor(.{ .file_uri = "", .name = name, .shape = .init(dims, .bf16), .offset = 0 });
        }
    }.add;

    for ([_]Entry{
        .{ "model.embed_tokens.weight", &.{ vocab_size, d } },
        .{ "model.norm.weight", &.{d} },
        .{ "lm_head.weight", &.{ vocab_size, d } },
    }) |entry| try add(&registry, entry[0], entry[1]);

    for (0..config.num_hidden_layers) |layer| {
        for ([_]Entry{
            .{ "input_layernorm.weight", &.{d} },
            .{ "post_attention_layernorm.weight", &.{d} },
            .{ "self_attn.q_proj.weight", &.{ q_out, d } },
            .{ "self_attn.k_proj.weight", &.{ kv_out, d } },
            .{ "self_attn.v_proj.weight", &.{ kv_out, d } },
            .{ "self_attn.o_proj.weight", &.{ d, q_out } },
            .{ "mlp.gate_proj.weight", &.{ intermediate_size, d } },
            .{ "mlp.up_proj.weight", &.{ intermediate_size, d } },
            .{ "mlp.down_proj.weight", &.{ d, intermediate_size } },
        }) |entry| {
            var name_buf: [128]u8 = undefined;
            try add(&registry, try std.fmt.bufPrint(&name_buf, "model.layers.{d}.{s}", .{ layer, entry[0] }), entry[1]);
        }
    }
    return registry;
}

const ShapeList = zml.stdx.BoundedArray(zml.Shape, 64);

/// Every tensor argument of a compiled function, in argument order.
fn tensorShapes(args: anytype) ShapeList {
    var shapes: ShapeList = .empty;
    zml.meta.visit(struct {
        fn cb(list: *ShapeList, t: *const Tensor) void {
            list.appendAssumeCapacity(t.shape());
        }
    }.cb, &shapes, &args);
    return shapes;
}

const Outcome = union(enum) {
    compiled,
    refused: struct { err: anyerror, message: []const u8 },
};

const Manifest = struct {
    writer: *std.Io.Writer,

    fn write(self: *Manifest, ctx: *const Context, kind: []const u8, name: []const u8, inputs: ShapeList, outcome: Outcome) !void {
        var w: std.json.Stringify = .{ .writer = self.writer };
        try w.beginObject();
        var text: [512]u8 = undefined;
        try w.objectField("module");
        try w.write(try std.fmt.bufPrint(&text, "{s}/{s}", .{ ctx.dir_name, name }));
        try w.objectField("kind");
        try w.write(kind);
        try w.objectField("tp");
        try w.write(ctx.topology.tp);
        try w.objectField("fixture");
        try w.write(ctx.topology.fixture);
        try w.objectField("view");
        try w.beginArray();
        for (ctx.view.axes.constSlice()) |ax| {
            try w.beginObject();
            try w.objectField("extent");
            try w.write(ax.extent);
            try w.objectField("kind");
            try w.write(@tagName(ax.kind));
            try w.objectField("links_per_hop");
            try w.write(ax.links_per_hop);
            try w.endObject();
        }
        try w.endArray();
        try w.objectField("files");
        try w.beginArray();
        for ([_][]const u8{ "mlirbc", "mlir", "compile_options.pb" }) |ext| {
            try w.write(try std.fmt.bufPrint(&text, "{s}/{s}.{s}", .{ ctx.dir_name, name, ext }));
        }
        try w.endArray();
        try w.objectField("inputs");
        try w.beginArray();
        for (inputs.constSlice()) |shape| try w.write(try std.fmt.bufPrint(&text, "{f}", .{shape}));
        try w.endArray();
        try w.objectField("result");
        switch (outcome) {
            .compiled => try w.write("compiled"),
            .refused => |r| {
                try w.write("error");
                try w.objectField("error_code");
                try w.write(@errorName(r.err));
                try w.objectField("error");
                try w.write(r.message);
            },
        }
        try w.endObject();
        try self.writer.writeByte('\n');
        try self.writer.flush();
    }
};
