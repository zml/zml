const std = @import("std");
const builtin = @import("builtin");
const log = std.log;
const cfg = @import("config.zig");
const Config = cfg.Config;

const zml = @import("zml");
const stdx = zml.stdx;
const Tensor = zml.Tensor;

const mel = @import("mel_spectrogram.zig");
const LogMelSpectrogram = mel.LogMelSpectrogram;

const enc = @import("encoder.zig");
const Encoder = enc.Encoder;

const dec = @import("decoder.zig");
const common = @import("common.zig");

const CausalConv1d = struct {
    weight: Tensor,
    bias: Tensor,
    stride: u32,

    fn init(store: zml.io.TensorStore.View, stride: u32) CausalConv1d {
        return .{
            .weight = store.createTensor("weight", .{ .cout, .cin, .k }, .replicated),
            .bias = store.createTensor("bias", .{.channels}, .replicated),
            .stride = stride,
        };
    }

    fn forward(self: CausalConv1d, x: Tensor) Tensor {
        const y = x.conv1d(self.weight, .{ .window_strides = self.stride, .padding = &.{ 2, 0 } });
        return y.add(self.bias.broad(y.shape()));
    }

    fn unload(buffers: *zml.Bufferized(CausalConv1d)) void {
        common.deinitBufferized(buffers);
    }
};

fn testLayer(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, layer: anytype, comptime method: std.meta.DeclEnum(@TypeOf(layer)), store: zml.io.TensorStore.View, name: []const u8, weights: zml.Bufferized(@TypeOf(layer)), opts: zml.testing.CompareOpts) !void {
    const forward = @field(@TypeOf(layer), @tagName(method));
    const Args = stdx.meta.Tail(std.meta.ArgsTuple(@TypeOf(forward)));
    var args: Args = undefined;
    const input_name = try std.fmt.allocPrint(allocator, "{s}.in", .{name});
    defer allocator.free(input_name);
    zml.meta.forEachVisit(&args, *Tensor, struct {
        fn init(i: usize, tensor: *Tensor, input_store: zml.io.TensorStore.View) void {
            var buf: [32]u8 = undefined;
            tensor.* = input_store.createTensor(std.fmt.bufPrint(&buf, "{d}", .{i}) catch unreachable, null, .replicated);
        }
    }.init, .{store.withPrefix(input_name)});

    const exe = try platform.compile(allocator, io, layer, method, args, .{ .program_name = name });
    defer exe.deinit();
    var progress = std.Progress.start(io, .{ .root_name = name });
    defer progress.end();
    var inputs = try common.loadModel(Args, &args, allocator, io, platform, store.store, &progress, "reference inputs");
    defer common.deinitBufferized(&inputs);
    var arguments = try exe.args(allocator);
    defer arguments.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    arguments.set(.{ weights, inputs });
    exe.callOpts(io, arguments, &results, .{ .wait = true });
    const outputs = try allocator.alloc(zml.Buffer, exe.output_shapes.len);
    defer allocator.free(outputs);
    results.fill(.{outputs});
    defer for (outputs) |*output| output.deinit();
    for (outputs, 0..) |output, i| {
        const key = try std.fmt.allocPrint(allocator, "{s}.out.{d}", .{ name, i });
        defer allocator.free(key);
        const expected = try zml.Slice.alloc(allocator, store.getShape(key) orelse return error.MissingReference);
        defer expected.free(allocator);
        var buffer: [4096]u8 = undefined;
        var reader = try store.getReader(key, io, &buffer);
        defer reader.deinit();
        try reader.interface.readSliceAll(expected.data());
        try zml.testing.expectClose(io, expected, output, opts);
    }
    log.info("Passed {s}", .{name});
}

const CliArgs = struct {
    reference: []const u8,
    model: []const u8,
};

pub fn main(init: std.process.Init) !void {
    log.info("Start of Voxtral test", .{});

    var dbg = std.heap.DebugAllocator(.{ .thread_safe = true }).init;
    defer if (builtin.mode == .Debug) std.debug.assert(dbg.deinit() == .ok);

    const allocator = switch (builtin.mode) {
        .Debug => dbg.allocator(),
        else => std.heap.c_allocator,
    };

    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    const io = init.io;

    var ref_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.reference);
    defer ref_registry.deinit();

    var ref_store: zml.io.TensorStore = .fromRegistry(allocator, &ref_registry);
    defer ref_store.deinit();

    const model_dir = try zml.safetensors.resolveModelRepo(io, args.model);
    defer model_dir.close(io);
    const model_file = try model_dir.openFile(io, if (std.mem.endsWith(u8, args.model, ".safetensors")) std.fs.path.basename(args.model) else "consolidated.safetensors", .{});
    defer model_file.close(io);
    var model_registry = try zml.safetensors.fetchRegistry(allocator, io, model_dir, model_file);
    defer model_registry.deinit();

    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    // Parse config from model directory
    var parsed_config = try cfg.parseConfig(allocator, io, model_dir);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    const encoder_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder";
    var progress = std.Progress.start(io, .{ .root_name = "Voxtral tests" });
    defer progress.end();

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    log.info("Selected platform {f}\n", .{platform.fmtVerbose()});

    // -- Mel spectrogram test
    {
        var melspectro_model: LogMelSpectrogram = .init(config);
        var mel_spectrum_buffers_future = try io.concurrent(LogMelSpectrogram.load, .{ &melspectro_model, io, platform });
        var mel_spectrum_buffers = try mel_spectrum_buffers_future.await(io);
        defer LogMelSpectrogram.unload(&mel_spectrum_buffers);

        try testLayer(allocator, io, platform, MelTestHarness{ .inner = melspectro_model }, .forward, ref_store.view(), "mel", .{ .inner = mel_spectrum_buffers }, .{});
    }

    // -- Conv stem test
    {
        const conv_store = model_store.view().withPrefix(encoder_prefix).withPrefix("conv_layers");
        const conv_stem = ConvStemTestHarness{
            .conv0 = CausalConv1d.init(conv_store.withLayer(0).withPrefix("conv"), 1),
            .conv1 = CausalConv1d.init(conv_store.withLayer(1).withPrefix("conv"), 2),
        };
        var conv_buffers = try common.loadModel(ConvStemTestHarness, &conv_stem, allocator, io, platform, &model_store, &progress, "test");
        defer ConvStemTestHarness.unload(&conv_buffers);

        try testLayer(allocator, io, platform, conv_stem, .forward, ref_store.view(), "conv_stem", conv_buffers, .{});
    }

    // -- Adapter test
    {
        const adapter = AdapterTestHarness{
            .inner = dec.Adapter.init(model_store.view(), config),
        };
        var adapter_buffers = try common.loadModel(AdapterTestHarness, &adapter, allocator, io, platform, &model_store, &progress, "test");
        defer AdapterTestHarness.unload(&adapter_buffers);

        try testLayer(allocator, io, platform, adapter, .forward, ref_store.view(), "adapter", adapter_buffers, .{ .minimum_close_fraction = 0.8 });
    }

    // -- Decoder Layer tests
    {
        const dec_layer_store = model_store.view().withPrefix("layers");

        // Load t_cond from reference activations (shared across all layers)
        const t_cond_tensor = ref_store.view().withPrefix("t_cond.out").createTensor("0", .{.d}, .replicated);
        const TCond = struct { t_cond: Tensor };
        const t_cond_spec = TCond{ .t_cond = t_cond_tensor };
        var t_cond_buf = try common.loadModel(TCond, &t_cond_spec, allocator, io, platform, &ref_store, &progress, "reference");
        defer t_cond_buf.t_cond.deinit();

        inline for (0..32) |i| {
            if (i < config.n_layers) {
                const layer = DecoderLayerTestHarness{
                    .inner = dec.DecoderLayer.init(dec_layer_store.withLayer(i), config),
                    .t_cond = t_cond_tensor,
                    .config = config,
                };

                var inner_buffers = try common.loadModel(dec.DecoderLayer, &layer.inner, allocator, io, platform, &model_store, &progress, "test");
                defer dec.DecoderLayer.unload(&inner_buffers);

                const layer_buffers: zml.Bufferized(DecoderLayerTestHarness) = .{
                    .inner = inner_buffers,
                    .t_cond = t_cond_buf.t_cond,
                };

                try testLayer(allocator, io, platform, layer, .forward, ref_store.view(), std.fmt.comptimePrint("decoder.prefill.layers.{d}", .{i}), layer_buffers, .{ .minimum_close_fraction = 0.85 });
            }
        }
    }
}

// ================================================================
// Test harnesses
// ================================================================

const MelTestHarness = struct {
    inner: LogMelSpectrogram,

    pub fn forward(self: MelTestHarness, waveform: Tensor) Tensor {
        return self.inner.melStep(waveform.withTags(.{.samples}));
    }
};

const ConvStemTestHarness = struct {
    conv0: CausalConv1d,
    conv1: CausalConv1d,

    pub fn unload(self: *zml.Bufferized(ConvStemTestHarness)) void {
        CausalConv1d.unload(&self.conv0);
        CausalConv1d.unload(&self.conv1);
    }

    pub fn forward(self: ConvStemTestHarness, mel_input: Tensor) Tensor {
        const dtype = self.conv0.weight.dtype();
        var h = mel_input.convert(dtype).withTags(.{ .channels, .time }).insertAxes(.channels, .{.batch});

        h = self.conv0.forward(h).gelu();
        h = self.conv1.forward(h).gelu();
        h = h.squeeze(.batch);

        return h.transpose(.{ .time, .channels }).convert(.f32);
    }
};

const AdapterTestHarness = struct {
    inner: dec.Adapter,

    pub fn unload(self: *zml.Bufferized(AdapterTestHarness)) void {
        dec.Adapter.unload(&self.inner);
    }

    pub fn forward(self: AdapterTestHarness, encoder_out: Tensor) Tensor {
        return self.inner.forward(encoder_out.convert(.bf16).withTags(.{ .s, .d })).convert(.f32);
    }
};

const DecoderLayerTestHarness = struct {
    inner: dec.DecoderLayer,
    t_cond: Tensor,
    config: Config,

    pub fn forward(self: DecoderLayerTestHarness, h: Tensor) Tensor {
        const dtype = self.inner.attention.wq.weight.dtype();
        const h_typed = h.convert(dtype).withTags(.{ .s, .d });

        // Create zero KV cache for prefill (empty cache for this layer)
        const attn_config = self.config.attentionConfig();
        const kv_shape = zml.Shape.init(.{
            .layer = 1,
            .k = h_typed.dim(.s),
            .h = @as(i64, @intCast(attn_config.n_kv_heads)),
            .hd = @as(i64, @intCast(attn_config.head_dim)),
        }, dtype);

        const kv_cache = dec.KvCache{
            .k = Tensor.zeroes(kv_shape),
            .v = Tensor.zeroes(kv_shape),
            .layer_index = Tensor.scalar(0, .u32),
        };

        const token_index = Tensor.scalar(0, .u32);
        const attention_metadata: zml.attention.Metadata = .{ .vanilla = {} };
        const attention_parameters: zml.attention.Parameters = .{ .vanilla = {} };
        const result = self.inner.forward(h_typed, token_index, kv_cache, self.t_cond.convert(dtype), attn_config, attention_metadata, attention_parameters);

        return result[0].convert(.f32);
    }
};
