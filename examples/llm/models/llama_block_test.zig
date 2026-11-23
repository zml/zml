//! Compare a trained Llama block and its KV cache against the CPU fixture.
const std = @import("std");
const zml = @import("zml");

const common = @import("common.zig");
const llama = @import("llama/model.zig");

const Args = struct {
    fixture: []const u8,
    target: zml.Target,
    pub const help = "Usage: llama_block_test --target=musa --fixture=/path/to/generated/fixture\n";
};

const BlockInput = struct {
    layer: llama.TransformerLayer,
    hidden: zml.Tensor,
    index: zml.Tensor,
    cache: llama.KvCache,
};

fn block(input: BlockInput) llama.TransformerLayer.Output {
    return llama.TransformerLayer.forward(.{
        .layer = input.layer,
        .hidden = input.hidden,
        .token_index = input.index,
        .kv_cache = input.cache,
        .kv_cache_index = .scalar(0, .u32),
        .attention_metadata = .{ .stablehlo = {} },
        .attention_parameters = .{ .stablehlo = {} },
    });
}

const Exe = zml.FnExe(block);

fn compile(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, mesh: *const zml.Mesh, layer: llama.TransformerLayer, cache: llama.KvCache, length: i64, width: u32) !Exe {
    return Exe.compile(allocator, io, platform, .{
        .meshes = &.{mesh},
        .program_name = if (length == 1) "llama_block_decode" else "llama_block_prefill",
    }, .{.{
        .layer = layer,
        .hidden = .init(.{ .s = length, .d = width }, .bf16),
        .index = .init(.{}, .u32),
        .cache = cache,
    }});
}

fn read(allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, name: []const u8) !zml.Slice {
    const shape = store.getShape(name) orelse return error.MissingFixtureTensor;
    const data = try zml.Slice.alloc(allocator, shape);
    errdefer data.free(allocator);
    var scratch: [8192]u8 = undefined;
    var reader = try store.getReader(name, io, &scratch);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(data.data());
    return data;
}

// Check every BF16 element, including the entire unwritten part of both caches.
// Eager CPU attention and ZML may apply the attention scale on different sides of the dot.
const Tolerance = struct { abs: f32, rel: f32 };
const History = struct { previous: []u8, prefix_elements: usize };

fn check(allocator: std.mem.Allocator, io: std.Io, step: usize, store: zml.io.TensorStore.View, name: []const u8, buffer: zml.Buffer, live_elements: usize, tolerance: Tolerance, history: ?History) !bool {
    const expected = try read(allocator, io, store, name);
    defer expected.free(allocator);
    if (expected.dtype() != .bf16 or !expected.shape.eql(buffer.shape())) return error.UnexpectedShape;
    const actual = try buffer.toSliceAlloc(allocator, io);
    defer actual.free(allocator);
    const BF16 = comptime zml.DataType.bf16.toZigType();
    var errors: usize = 0;
    var max_error: f32 = 0;
    var squared_error: f64 = 0;
    for (actual.constItems(BF16), expected.constItems(BF16), 0..) |lhs, rhs, index| {
        const value = lhs.toF32();
        const reference = rhs.toF32();
        const delta = @abs(value - reference);
        const valid = std.math.isFinite(value) and std.math.isFinite(reference) and
            (if (index >= live_elements) value == 0 and reference == 0 else delta <= tolerance.abs + tolerance.rel * @abs(reference));
        if (!valid) {
            if (errors == 0) std.log.err("step {d} {s}[{d}] got {d}, expected {d}", .{ step, name, index, value, reference });
            errors += 1;
        }
        max_error = @max(delta, max_error);
        squared_error += @as(f64, delta) * delta;
    }
    if (history) |cache| {
        const prefix_bytes = cache.prefix_elements * @sizeOf(BF16);
        if (!std.mem.eql(u8, cache.previous[0..prefix_bytes], actual.constData()[0..prefix_bytes])) {
            std.log.err("step {d} {s} modified the previously written cache prefix", .{ step, name });
            return false;
        }
        @memcpy(cache.previous, actual.constData());
    }
    const count = expected.shape.count();
    std.log.info("step {d} {s}: {d}/{d} within tolerance, max_abs={d:.6} rms={d:.6}", .{ step, name, count - errors, count, max_error, @sqrt(squared_error / @as(f64, @floatFromInt(count))) });
    return errors == 0;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const platform = try zml.Platform.init(allocator, io, args.target, .{ .cpu = .{ .device_count = 1 } });
    defer platform.deinit(allocator, io);
    if (platform.devices.len != 1) return error.ExpectedOneVisibleDevice;
    std.log.info("Running Llama block on {t}: {s}", .{ platform.target, platform.devices[0].kind() });
    const mesh = try platform.registerMesh("model", .mesh(.{ .model = .high_bandwidth }));

    const dir = try std.Io.Dir.cwd().openDir(io, args.fixture, .{});
    defer dir.close(io);
    const config = try common.parseConfig(llama.Config, allocator, io, dir);
    defer config.deinit();
    const cfg = config.value;
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, dir);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &.{mesh});
    defer store.deinit();
    const layer = try llama.TransformerLayer.init(store.view().withPrefix("model.layers.0"), cfg);
    var weights = try zml.mem.bufferize(allocator, llama.TransformerLayer, &layer);
    defer llama.TransformerLayer.unloadBuffers(&weights);
    {
        var loader = try zml.io.Loader.init(allocator, platform, .{ .parallelism = 2, .dma_chunks = 4, .dma_chunk_size = 4 * zml.MiB });
        defer loader.deinit();
        try loader.load(io, llama.TransformerLayer, &layer, &weights, &store, .{});
        try loader.await(io);
    }
    const steps = store.view().withPrefix("steps");
    const first = steps.withLayer(0);
    const cache_dims = (first.getShape("k") orelse return error.MissingFixtureTensor).withTags(.{ .layer, .k, .h, .hd });
    const cache = llama.KvCache.init(cache_dims, mesh);
    const zeros = try allocator.alloc(u8, cache_dims.byteSize());
    defer allocator.free(zeros);
    @memset(zeros, 0);
    var cache_buffers: llama.KvCache.Buffer = .{
        .k = try .fromBytes(io, platform, cache.k.shape(), zeros),
        .v = try .fromBytes(io, platform, cache.v.shape(), zeros),
    };
    defer llama.KvCache.deinitBuffer(&cache_buffers);
    const prefill_tokens = (first.getShape("input") orelse return error.MissingFixtureTensor).dim(0);
    if (prefill_tokens < 2) return error.ExpectedMultiTokenPrefill;
    const prefill = try compile(allocator, io, platform, mesh, layer, cache, prefill_tokens, cfg.hidden_size);
    defer prefill.deinit();
    const decode = try compile(allocator, io, platform, mesh, layer, cache, 1, cfg.hidden_size);
    defer decode.deinit();
    var prefill_runner: Exe.Runner(.{.layer}) = try .init(&prefill, allocator, .{ .layer = weights });
    defer prefill_runner.deinit(allocator);
    var decode_runner: Exe.Runner(.{.layer}) = try .init(&decode, allocator, .{ .layer = weights });
    defer decode_runner.deinit(allocator);

    const previous_keys = try allocator.alloc(u8, cache_dims.byteSize());
    defer allocator.free(previous_keys);
    const previous_values = try allocator.alloc(u8, cache_dims.byteSize());
    defer allocator.free(previous_values);
    const decode_steps = 4;
    var position: u32 = 0;
    var passed = true;
    for (0..1 + decode_steps) |step| {
        const fixture = steps.withLayer(step);
        const input = try read(allocator, io, fixture, "input");
        defer input.free(allocator);
        const length: u32 = @intCast(input.shape.dim(0));
        if (step != 0 and length != 1) return error.ExpectedSingleTokenDecode;
        if (position + length > cache_dims.dim(.k)) return error.CacheCapacityExceeded;
        var hidden = try zml.Buffer.fromSlice(io, platform, input);
        defer hidden.deinit();
        var index = try zml.Buffer.scalar(io, platform, position, .u32);
        defer index.deinit();
        const runner = if (step == 0) &prefill_runner else &decode_runner;
        runner.run(io, .{
            .inputs = .{ .hidden = hidden, .index = index, .cache = cache_buffers },
            .outputs = .{ .hidden = &hidden, .kv_cache = &cache_buffers },
        }, .{ .wait = true });
        const row_size: usize = @intCast(cache_dims.dim(.h) * cache_dims.dim(.hd));
        const prefix = position * row_size;
        position += length;
        const used = position * row_size;
        const output_ok = try check(allocator, io, step, fixture, "output", hidden, hidden.shape().count(), .{ .abs = 0.006, .rel = 0.04 }, null);
        const keys_ok = try check(allocator, io, step, fixture, "k", cache_buffers.k, used, .{ .abs = 0.04, .rel = 0.025 }, .{ .previous = previous_keys, .prefix_elements = prefix });
        const values_ok = try check(allocator, io, step, fixture, "v", cache_buffers.v, used, .{ .abs = 0.002, .rel = 0.025 }, .{ .previous = previous_values, .prefix_elements = prefix });
        passed = passed and output_ok and keys_ok and values_ok;
    }
    if (!passed) return error.ReferenceMismatch;
    std.log.info("PASS: Llama block, {d}-token prefill and {d} consecutive decode steps", .{ prefill_tokens, decode_steps });
}
