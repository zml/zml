const std = @import("std");

const zml = @import("zml");

const weights = @import("../recipe/weights.zig");

const log = std.log.scoped(.minimax_h3);

// =============================================================================
// draft/taeh3.zig — tiny autoencoder for H3 latents
//
// Draft RGB at 16px / latent cell. Feeds the Stage 2 handoff.
// =============================================================================

const Conv = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor = null,
};

const Mem = struct {
    c0: Conv,
    c2: Conv,
    c4: Conv,
};

pub const Model = struct {
    batch: i64,
    time: i64,
    stem: Conv,
    m0: [3]Mem,
    grow0: Conv,
    down0: Conv,
    m1: [3]Mem,
    grow1: Conv,
    down1: Conv,
    m2: [3]Mem,
    grow2: Conv,
    down2: Conv,
    head: Conv,

    pub fn init(store: zml.io.TensorStore.View, batch: i64, time: i64) Model {
        const d = store.withPrefix("decoder");
        return .{
            .batch = batch,
            .time = time,
            .stem = conv(d, "1", true),
            .m0 = .{ mem(d, "3"), mem(d, "4"), mem(d, "5") },
            .grow0 = conv1(d, "7.conv"),
            .down0 = conv(d, "8", false),
            .m1 = .{ mem(d, "9"), mem(d, "10"), mem(d, "11") },
            .grow1 = conv1(d, "13.conv"),
            .down1 = conv(d, "14", false),
            .m2 = .{ mem(d, "15"), mem(d, "16"), mem(d, "17") },
            .grow2 = conv1(d, "19.conv"),
            .down2 = conv(d, "20", false),
            .head = conv(d, "22", true),
        };
    }
};

fn conv(store: zml.io.TensorStore.View, id: []const u8, bias: bool) Conv {
    var wname: [32]u8 = undefined;
    var bname: [32]u8 = undefined;
    const w = std.fmt.bufPrint(&wname, "{s}.weight", .{id}) catch unreachable;
    const b = std.fmt.bufPrint(&bname, "{s}.bias", .{id}) catch unreachable;
    return .{
        .weight = store.createTensor(w, .{ .co, .ci, .kh, .kw }, .replicated),
        .bias = if (bias) store.createTensor(b, .{.co}, .replicated) else null,
    };
}

fn conv1(store: zml.io.TensorStore.View, id: []const u8) Conv {
    var wname: [40]u8 = undefined;
    const w = std.fmt.bufPrint(&wname, "{s}.weight", .{id}) catch unreachable;
    return .{
        .weight = store.createTensor(w, .{ .co, .ci, .kh, .kw }, .replicated),
        .bias = null,
    };
}

fn mem(store: zml.io.TensorStore.View, id: []const u8) Mem {
    var p0: [40]u8 = undefined;
    var p2: [40]u8 = undefined;
    var p4: [40]u8 = undefined;
    return .{
        .c0 = convNamed(store, std.fmt.bufPrint(&p0, "{s}.conv.0", .{id}) catch unreachable, true),
        .c2 = convNamed(store, std.fmt.bufPrint(&p2, "{s}.conv.2", .{id}) catch unreachable, true),
        .c4 = convNamed(store, std.fmt.bufPrint(&p4, "{s}.conv.4", .{id}) catch unreachable, true),
    };
}

fn convNamed(store: zml.io.TensorStore.View, id: []const u8, bias: bool) Conv {
    var wname: [48]u8 = undefined;
    var bname: [48]u8 = undefined;
    const w = std.fmt.bufPrint(&wname, "{s}.weight", .{id}) catch unreachable;
    const b = std.fmt.bufPrint(&bname, "{s}.bias", .{id}) catch unreachable;
    return .{
        .weight = store.createTensor(w, .{ .co, .ci, .kh, .kw }, .replicated),
        .bias = if (bias) store.createTensor(b, .{.co}, .replicated) else null,
    };
}

fn asNchw(x: zml.Tensor) zml.Tensor {
    return x.withTags(.{ .n, .c, .h, .w });
}

fn asDt(x: zml.Tensor, t: zml.Tensor) zml.Tensor {
    return if (x.dtype() == t.dtype()) x else x.convert(t.dtype());
}

fn applyConv(x: zml.Tensor, c: Conv) zml.Tensor {
    var y = asNchw(x.conv2d(asDt(c.weight, x), .{ .padding = &.{ 1, 1, 1, 1 } }));
    if (c.bias) |b| y = y.add(asDt(b, y).rename(.{ .co = .c }).broad(y.shape()));
    return y;
}

fn applyConv1(x: zml.Tensor, c: Conv) zml.Tensor {
    return asNchw(x.conv2d(asDt(c.weight, x), .{}));
}

fn memblock(m: Mem, x: zml.Tensor, batch: i64, time: i64) zml.Tensor {
    const xt = x.reshape(.{ .b = batch, .t = time, .c = x.dim(.c), .h = x.dim(.h), .w = x.dim(.w) });
    const past = xt.pad(0, .{ .t = zml.Tensor.Pad{ .low = 1 } }).slice(.t, .{ .start = 0, .end = time });
    const p4 = past.reshape(.{ .n = x.dim(.n), .c = x.dim(.c), .h = x.dim(.h), .w = x.dim(.w) });
    const cat = zml.Tensor.concatenate(&.{ x, p4 }, .c);
    const y = applyConv(applyConv(applyConv(cat, m.c0).relu(), m.c2).relu(), m.c4);
    return y.add(x).relu();
}

fn tgrow(c: Conv, x: zml.Tensor, stride: i64) zml.Tensor {
    const y = applyConv1(x, c);
    if (stride == 1) return y;
    return y.reshape(.{
        .n = y.dim(.n) * stride,
        .c = @divExact(y.dim(.c), stride),
        .h = y.dim(.h),
        .w = y.dim(.w),
    });
}

fn up2(x: zml.Tensor) zml.Tensor {
    return asNchw(zml.nn.upsample(x, .{ .mode = .nearest, .scale_factor = &.{ 2, 2 } }));
}

/// PyTorch `F.pixel_shuffle(x, 2)` on NCHW.
fn pixelShuffle2(x: zml.Tensor) zml.Tensor {
    const n = x.dim(.n);
    const h = x.dim(.h);
    const w = x.dim(.w);
    const y = x.reshape(.{ .n = n, .c = 3, .rh = 2, .rw = 2, .h = h, .w = w });
    const z = y.transpose(.{ .n, .c, .h, .rh, .w, .rw });
    return asNchw(z.reshape(.{ .n = n, .c = 3, .h = h * 2, .w = w * 2 }));
}

pub const DecodeInput = struct {
    model: Model,
    latents: zml.Tensor,
};

pub const DecodeOutput = struct {
    rgb: zml.Tensor,
};

pub fn decode(input: DecodeInput) DecodeOutput {
    const model = input.model;
    const b = model.batch;
    var t = model.time;
    const three = zml.Tensor.scalar(3.0, input.latents.dtype());
    var x = input.latents.div(three).tanh().mul(three);
    x = applyConv(x, model.stem).relu();
    for (model.m0) |m| x = memblock(m, x, b, t);
    x = up2(x);
    x = tgrow(model.grow0, x, 1);
    x = applyConv(x, model.down0);
    for (model.m1) |m| x = memblock(m, x, b, t);
    x = up2(x);
    x = tgrow(model.grow1, x, 2);
    t *= 2;
    x = applyConv(x, model.down1);
    for (model.m2) |m| x = memblock(m, x, b, t);
    x = up2(x);
    x = tgrow(model.grow2, x, 2);
    x = applyConv(x, model.down2).relu();
    x = applyConv(x, model.head);
    x = pixelShuffle2(x);
    return .{ .rgb = x };
}

pub const spatial_scale: u32 = 16;
pub const temporal_expand: i64 = 4;

pub fn timeOut(time: i64) i64 {
    return time * temporal_expand;
}

pub fn pixelExtent(latent: u32) u32 {
    return latent * spatial_scale;
}

pub const Compiled = struct {
    decode: zml.FnExe(decode),
    bufs: zml.Bufferized(Model),
    batch: u32,
    time: u32,
    latent_h: u32,
    latent_w: u32,
    owns_bufs: bool = true,

    pub fn deinit(self: *Compiled) void {
        if (self.owns_bufs) zml.Buffer.deinitAll(Model, &self.bufs);
        self.decode.deinit();
    }
};

pub const Loaded = struct {
    store: zml.io.TensorStore,
    registry: *zml.safetensors.TensorRegistry,

    pub fn deinit(self: *Loaded, allocator: std.mem.Allocator) void {
        self.store.deinit();
        self.registry.deinit();
        allocator.destroy(self.registry);
    }

    pub fn bind(self: *const Loaded, allocator: std.mem.Allocator) zml.io.TensorStore {
        return .fromRegistry(allocator, self.registry);
    }
};

pub fn open(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !Loaded {
    const registry = try allocator.create(zml.safetensors.TensorRegistry);
    errdefer allocator.destroy(registry);
    registry.* = try zml.safetensors.TensorRegistry.fromPath(allocator, io, path);
    errdefer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, registry);
    errdefer store.deinit();
    return .{
        .store = store,
        .registry = registry,
    };
}

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    loaded: *Loaded,
    batch: u32,
    time: u32,
    latent_h: u32,
    latent_w: u32,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    reuse: ?*const Compiled,
) !Compiled {
    progress.increaseEstimatedTotalItems(1);
    var store = loaded.bind(allocator);
    defer store.deinit();
    const model = Model.init(store.view(), batch, time);
    const bt = model.batch * model.time;
    const exe = try zml.FnExe(decode).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_taeh3",
    }, .{.{
        .model = model,
        .latents = .init(.{ .n = bt, .c = 24, .h = latent_h, .w = latent_w }, .f32),
    }});
    const load_start: std.Io.Timestamp = .now(io, .awake);
    const bufs = if (reuse) |src| src.bufs else try weights.load(allocator, io, platform, &store, shardings, Model, &model, progress, null);
    log.info("compile TAEH3: ok {d}x{d} t={d}{s} [{f}]", .{
        latent_w,
        latent_h,
        model.time,
        if (reuse != null) " reuse weights" else " load",
        load_start.untilNow(io, .awake),
    });
    return .{
        .decode = exe,
        .bufs = bufs,
        .batch = @intCast(model.batch),
        .time = @intCast(model.time),
        .latent_h = latent_h,
        .latent_w = latent_w,
        .owns_bufs = reuse == null,
    };
}

fn thwcToNchw(dst: []f32, src: []const f32, t: u32, h: u32, w: u32, c: u32) void {
    var tt: u32 = 0;
    while (tt < t) : (tt += 1) {
        var hh: u32 = 0;
        while (hh < h) : (hh += 1) {
            var ww: u32 = 0;
            while (ww < w) : (ww += 1) {
                var cc: u32 = 0;
                while (cc < c) : (cc += 1) {
                    dst[((tt * c + cc) * h + hh) * w + ww] = src[((tt * h + hh) * w + ww) * c + cc];
                }
            }
        }
    }
}

/// H3 chunk trim: 20-frame groups, drop 3 prefix per group, then drop last 12.
pub fn takeFrameIndex(f: u32, time: u32) u32 {
    if (time == 0) return 0;
    return @min(time - 1, f + 3 + (f / 17) * 3);
}

/// Decode TAEH3 and keep RGB `{n,c,h,w}` on device for the GPU handoff.
pub fn runDevice(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const Compiled,
    latents_thwc: []const f32,
) !zml.Buffer {
    const t = compiled.time;
    const h = compiled.latent_h;
    const w = compiled.latent_w;
    const nchw = try allocator.alloc(f32, t * 24 * h * w);
    defer allocator.free(nchw);
    thwcToNchw(nchw, latents_thwc, t, h, w, 24);

    var lat_buf = try weights.fromItems(io, platform, .init(.{ .n = t, .c = 24, .h = h, .w = w }, .f32), nchw);
    defer lat_buf.deinit();
    const exec_start: std.Io.Timestamp = .now(io, .awake);
    var runner = try zml.FnExe(decode).Runner(.{.model}).init(&compiled.decode, allocator, .{ .model = compiled.bufs });
    defer runner.deinit(allocator);
    var rgb: zml.Buffer = undefined;
    runner.run(io, .{
        .inputs = .{ .latents = lat_buf },
        .outputs = .{ .rgb = &rgb },
        .opts = .{ .wait = true },
    });
    log.info("TAEH3 exec [{f}]", .{exec_start.untilNow(io, .awake)});
    return rgb;
}

/// H3 chunk trim: 20-frame groups, drop 3 prefix per group, then drop last 12.
pub fn takeFrames(allocator: std.mem.Allocator, ncht: []const f32, time: u32, h: u32, w: u32, frames: u32) ![]f32 {
    const plane = h * w;
    const out = try allocator.alloc(f32, 3 * frames * plane);
    var f: u32 = 0;
    while (f < frames) : (f += 1) {
        const src_t = takeFrameIndex(f, time);
        var c: u32 = 0;
        while (c < 3) : (c += 1) {
            const src = ((src_t * 3 + c) * plane);
            const dst = (c * frames + f) * plane;
            @memcpy(out[dst..][0..plane], ncht[src..][0..plane]);
        }
    }
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        out[i] = std.math.clamp(out[i], 0.0, 1.0);
    }
    return out;
}
