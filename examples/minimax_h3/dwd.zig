//! Replicated Pi-H3 DWD pixel head, independent of the original tiled VAE.
//! Based on LakonLab@2ee2b334374ec91b7cee371a85a524bf4ba644d3,
//! sol_h3/{model,sol_h3}.py and autoencoders/color_encoders.py (Apache-2.0),
//! Copyright (c) 2026 Hansheng Chen; upstream Sol-H3-VAE@6a8a33f.

const std = @import("std");
const zml = @import("zml");
const ops = @import("ops.zig");
const config = @import("config.zig");
const dwd_plan = @import("dwd_plan.zig");

const Tensor = zml.Tensor;
const Grid = dwd_plan.Grid;
const View = zml.io.TensorStore.View;
const width = 2048;
const feature_width = 5376;
const heads = 32;
const hd = 64;
const eps = 1e-5;
const patch = 16;
const layers = 8;

fn parameter(store: View, name: []const u8, tags: anytype, dims: []const i64) !Tensor {
    const shape = store.getShape(name) orelse return error.MissingDwdWeight;
    if (shape.dtype() != .f16 or !std.mem.eql(i64, shape.dims(), dims))
        return error.InvalidDwdWeight;
    return store.createTensor(name, tags, .replicated, .{});
}

fn expandGrid(x: Tensor, grid: Grid, dim: i64) Tensor {
    // The four projected subvectors are (dy,dx), not a contiguous raster
    // run of four: each input patch becomes a 2x2 tile in the expanded grid.
    return x.reshape(.{ .t = grid.t, .y = grid.h / 2, .x = grid.w / 2, .dy = 2, .dx = 2, .d = dim })
        .transpose(.{ .t, .y, .dy, .x, .dx, .d })
        .reshape(.{ .s = grid.videoRows(), .d = dim });
}

const Expand = struct {
    proj: zml.nn.Linear,
    registers: Tensor,

    const Input = struct { model: Expand, features: Tensor, grid: Grid };
    const Output = struct { hidden: Tensor };

    fn forward(input: Input) Output {
        const projected = ops.bf16Linear(input.model.proj, input.features).rename(.{ .dout = .d });
        const video = expandGrid(projected, input.grid, width);
        const regs = input.model.registers.convert(.bf16).reshape(.{ .r = 4, .d = width });
        const one_group = Tensor.concatenate(&.{ regs, Tensor.zeroes(.init(.{ .r = 1, .d = width }, .bf16)) }, .r);
        const globals = one_group.broad(.init(.{ .g = input.grid.groups(), .r = dwd_plan.globals, .d = width }, .bf16))
            .reshape(.{ .s = input.grid.groups() * dwd_plan.globals, .d = width });
        return .{ .hidden = Tensor.concatenate(&.{ video, globals }, .s) };
    }
};

fn rms(x: Tensor, weight: Tensor) Tensor {
    const normed = zml.nn.rmsNorm(x.convert(.f32), .d, eps);
    return normed.mul(weight.convert(.f32).broad(normed.shape())).convert(.bf16);
}

const Qkv = struct { q: Tensor, k: Tensor, v: Tensor };

fn splitQkv(projected: Tensor, h: i64, d: i64) Qkv {
    // The head is OUTSIDE the q/k/v triplet in the fused projection.
    const x = projected.splitAxis(.d, .{ .h = h, .part = 3, .hd = d });
    return .{ .q = x.slice(.part, .single(0)), .k = x.slice(.part, .single(1)), .v = x.slice(.part, .single(2)) };
}

const Rotary = struct { cos: Tensor, sin: Tensor };

fn localAngles(comptime axis: @EnumLiteral(), n: usize, cycle: i32, shift: i32, inv: Tensor) Tensor {
    const id = Tensor.arange(.{ .end = @intCast(n) }, .i32).withTags(.{axis}).addConstant(shift).remainderConst(cycle);
    const pos = id.convert(.f32).addConstant(0.5).divByConst(@as(f32, @floatFromInt(cycle))).scale(2).addConstant(-1);
    return pos.scale(2.0 * std.math.pi).outer(inv);
}

fn rotary(grid: Grid, shifted: bool) Rotary {
    const exponent = Tensor.arange(.{ .end = 8 }, .f32).withTags(.{.f}).scale(0.125);
    const inv = Tensor.scalar(1, .f32).div(Tensor.scalar(100, .f32).pow(exponent));
    const st: i32 = if (shifted) 2 else 0;
    const sp: i32 = if (shifted) 8 else 0;
    const shape: zml.Shape = .init(.{ .t = grid.t, .y = grid.h, .x = grid.w, .f = 8 }, .f32);
    const half = Tensor.concatenate(&.{
        localAngles(.t, grid.t, 5, st, inv).broad(shape),
        localAngles(.y, grid.h, 16, sp, inv).broad(shape),
        localAngles(.x, grid.w, 16, sp, inv).broad(shape),
    }, .f).reshape(.{ .s = grid.videoRows(), .f = 24 });
    // ops.applyRotary consumes duplicated sin/cos halves for a 48-of-64 prefix.
    const angle = Tensor.concatenate(&.{ half, half }, .f);
    return .{ .cos = angle.cos().convert(.bf16), .sin = angle.sin().convert(.bf16) };
}

fn rotateVideo(x: Tensor, pe: Rotary) Tensor {
    const n = pe.cos.dim(.s);
    const v = ops.applyRotary(x.slice(.s, .{ .end = n }), pe.cos, pe.sin);
    return Tensor.concatenate(&.{ v, x.slice(.s, .{ .start = n }) }, .s);
}

fn scatterAttention(out_: Tensor, qkv: Qkv, buckets: []const dwd_plan.Bucket, backend: zml.attention.Backend) Tensor {
    var out = out_;
    const replicated: zml.attention.Partitioning = .{
        .in_specs = .{ .q = .replicated, .k = .replicated, .v = .replicated },
        .out_specs = .replicated,
    };
    for (buckets) |b| {
        const count = b.q.items.len / b.nq;
        // Bound each gathered K or V to ~64 MiB in BF16 at the native width,
        // except when a single global-query sequence itself exceeds that bound.
        const limit = @max(1, 16384 / b.nk);
        var i: usize = 0;
        while (i < count) {
            const batch = @min(limit, count - i);
            const qs = b.q.items[i * b.nq ..][0 .. batch * b.nq];
            const ks = b.k.items[i * b.nk ..][0 .. batch * b.nk];
            const qi = Tensor.constantTensor(.init(.{ .b = batch, .q = b.nq }, .i32), std.mem.sliceAsBytes(qs));
            const ki = Tensor.constantTensor(.init(.{ .b = batch, .k = b.nk }, .i32), std.mem.sliceAsBytes(ks));
            const values = zml.attention.attentionOpts(
                qkv.q.gather(.{ .s = qi }, .{}),
                qkv.k.gather(.{ .s = ki }, .{}),
                qkv.v.gather(.{ .s = ki }, .{}),
                Tensor.scalar(0, .i32),
                .init(.fromBackend(backend)),
                .init(.fromBackend(backend)),
                .{ .causal = false },
                zml.Compiler.current().mesh(.model),
                replicated,
            );
            // No two sequences in these plans write the same canonical row.
            out = out.scatterSlices(.{ .s = qi }, values, .{
                .indices_are_unique = true,
                .update_fn = Tensor.ScatterOpts.override,
            });
            i += batch;
        }
    }
    return out;
}

fn swiglu(x: Tensor) Tensor {
    const gate, const up = x.chunkExact(.d, 2);
    // LakonLab's CUDA silu_mul_ uses FP32 intermediates and a single BF16 store.
    return gate.convert(.f32).silu().mul(up.convert(.f32)).convert(.bf16);
}

const Block = struct {
    norm1: Tensor,
    qkv: zml.nn.Linear,
    attn_out: zml.nn.Linear,
    scale1: Tensor,
    norm2: Tensor,
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,
    scale2: Tensor,

    const Input = struct {
        model: Block,
        hidden: Tensor,
        grid: Grid,
        shifted: bool,
        windows: []const dwd_plan.Bucket,
        global: []const dwd_plan.Bucket,
        backend: zml.attention.Backend,
    };
    const Output = struct { hidden: Tensor };

    fn init(p: View) !Block {
        return .{
            .norm1 = try parameter(p, "norm1.weight", .{.d}, &.{width}),
            .qkv = ops.linear(p, "attn.to_qkv.weight", "attn.to_qkv.bias", .replicated, .{}),
            .attn_out = ops.linear(p, "attn.to_out.weight", "attn.to_out.bias", .replicated, .{}),
            .scale1 = try parameter(p, "scale1", .{.d}, &.{width}),
            .norm2 = try parameter(p, "norm2.weight", .{.d}, &.{width}),
            .w1 = ops.linear(p, "ff.w1.weight", "ff.w1.bias", .replicated, .{}),
            .w2 = ops.linear(p, "ff.w2.weight", "ff.w2.bias", .replicated, .{}),
            .scale2 = try parameter(p, "scale2", .{.d}, &.{width}),
        };
    }

    fn forward(in: Input) Output {
        const m = in.model;
        var p = splitQkv(ops.bf16Linear(m.qkv, rms(in.hidden, m.norm1)).rename(.{ .dout = .d }), heads, hd);
        const pe = rotary(in.grid, in.shifted);
        p.q = rotateVideo(zml.nn.rmsNorm(p.q, .hd, eps), pe);
        p.k = rotateVideo(zml.nn.rmsNorm(p.k, .hd, eps), pe);
        const g = scatterAttention(Tensor.zeroes(p.q.shape()), p, in.global, in.backend);
        const all = scatterAttention(g, p, in.windows, in.backend).merge(.{ .d = .{ .h, .hd } });
        const attn = ops.bf16Linear(m.attn_out, all).rename(.{ .dout = .d });
        const x = in.hidden.add(attn.mul(m.scale1.convert(.bf16).broad(attn.shape())));
        const up = ops.bf16Linear(m.w1, rms(x, m.norm2)).rename(.{ .dout = .d });
        const ff = ops.bf16Linear(m.w2, swiglu(up)).rename(.{ .dout = .d });
        return .{ .hidden = x.add(ff.mul(m.scale2.convert(.bf16).broad(ff.shape()))).reuseBuffer(in.hidden) };
    }
};

fn unpatch(projected: Tensor, temporal: i64, spatial: i64) Tensor {
    // Channel precedes temporal subframe in each checkpoint projection.
    return projected.splitAxis(.d, .{ .c = 3, .u = temporal, .dy = spatial, .dx = spatial })
        .transpose(.{ .c, .g, .l, .u, .y, .dy, .x, .dx })
        .merge(.{ .f = .{ .l, .u }, .h = .{ .y, .dy }, .w = .{ .x, .dx } });
}

fn rgb01(x: Tensor) Tensor {
    const y = ops.denorm(x.convert(.f32), &.{ -0.03, -0.088, -0.188 }, &.{ 0.458, 0.448, 0.45 });
    return y.clamp(Tensor.scalar(-1, .f32), Tensor.scalar(1, .f32)).addConstant(1).scale(0.5);
}

const Finish = struct {
    weight: Tensor,
    bias: Tensor,
    first: zml.nn.Linear,
    rest: zml.nn.Linear,

    const Input = struct { model: Finish, hidden: Tensor, grid: Grid };
    const Output = struct { rgb: Tensor };

    fn project(m: Finish, x: Tensor, grid: Grid, n: usize, t: usize) Tensor {
        const groups = x.reshape(.{ .g = n, .l = t, .y = grid.h, .x = grid.w, .d = width });
        const first = unpatch(ops.bf16Linear(m.first, groups.slice(.l, .{ .end = 1 })).rename(.{ .dout = .d }), 1, patch);
        const frames = if (t == 1) first else Tensor.concatenate(&.{ first, unpatch(ops.bf16Linear(m.rest, groups.slice(.l, .{ .start = 1 })).rename(.{ .dout = .d }), 4, patch) }, .f);
        return frames.reshape(.{ .c = 3, .f = n * (4 * t - 3), .h = grid.h * patch, .w = grid.w * patch });
    }

    fn forward(in: Input) Output {
        const m = in.model;
        const grid = in.grid;
        const v = in.hidden.slice(.s, .{ .end = @intCast(grid.videoRows()) });
        const normed = zml.nn.normalizeVariance(v.convert(.f32), eps)
            .mul(m.weight.convert(.f32).broad(v.shape()))
            .add(m.bias.convert(.f32).broad(v.shape())).convert(.bf16);
        const x = normed.reshape(.{ .t = grid.t, .y = grid.h, .x = grid.w, .d = width });
        const n = grid.t / dwd_plan.period;
        const tail = grid.t % dwd_plan.period;
        const boundary: i64 = @intCast(n * dwd_plan.period);
        const full = if (n > 0) m.project(x.slice(.t, .{ .end = boundary }), grid, n, dwd_plan.period) else null;
        const pixels = if (tail > 0) blk: {
            const end = m.project(x.slice(.t, .{ .start = boundary }), grid, 1, tail);
            break :blk if (full) |a| Tensor.concatenate(&.{ a, end }, .f) else end;
        } else full.?;
        return .{ .rgb = rgb01(pixels) };
    }
};

const Compiled = struct {
    grid: Grid,
    expand: zml.FnExe(Expand.forward),
    blocks: [2]zml.FnExe(Block.forward),
    finish: zml.FnExe(Finish.forward),

    fn deinit(c: *Compiled) void {
        c.expand.deinit();
        for (&c.blocks) |*e| e.deinit();
        c.finish.deinit();
    }
};

pub const Dwd = struct {
    expand: Expand,
    blocks: [layers]Block,
    finish: Finish,
    compiled: ?Compiled = null,

    pub fn init(store: View) !Dwd {
        const p = store.withPrefix("dwd_pixel_refiner");
        var blocks: [layers]Block = undefined;
        for (&blocks, 0..) |*b, i| b.* = try Block.init(p.withPrefix("blocks").withLayer(i));
        return .{
            .expand = .{
                .proj = ops.linear(p, "feature_expand.weight", "feature_expand.bias", .replicated, .{}),
                .registers = try parameter(p, "register_tokens", .{ .b, .r, .d }, &.{ 1, 4, width }),
            },
            .blocks = blocks,
            .finish = .{
                .weight = try parameter(p, "norm_out.weight", .{.d}, &.{width}),
                .bias = try parameter(p, "norm_out.bias", .{.d}, &.{width}),
                .first = ops.linear(p, "first_frame_proj_out.weight", "first_frame_proj_out.bias", .replicated, .{}),
                .rest = ops.linear(p, "proj_out.weight", "proj_out.bias", .replicated, .{}),
            },
        };
    }

    pub fn deinit(self: *Dwd) void {
        if (self.compiled) |*c| c.deinit();
        self.compiled = null;
    }

    pub fn compile(self: *Dwd, run: *const ops.Run, geo: config.Geometry) !void {
        return self.compileBackend(run, geo, try ops.piAttentionBackend(run.platform));
    }

    fn compileBackend(self: *Dwd, run: *const ops.Run, geo: config.Geometry, backend: zml.attention.Backend) !void {
        const grid: Grid = .{ .t = geo.latent_t, .h = geo.latent_h, .w = geo.latent_w };
        if (grid.t == 0 or grid.h == 0 or grid.w == 0 or grid.h % 2 != 0 or grid.w % 2 != 0 or
            grid.h * patch != geo.pixel_h or grid.w * patch != geo.pixel_w or
            grid.frames() != geo.frames or grid.videoRows() / 4 != geo.video_tokens) return error.InvalidDwdGeometry;
        const a = run.allocator;
        const hidden: Tensor = .init(.{ .s = grid.rows(), .d = width }, .bf16);
        const expand = try zml.FnExe(Expand.forward).compile(a, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "pi_h3_dwd_expand",
        }, .{.{ .model = self.expand, .features = .init(.{ .b = 1, .s = geo.video_tokens, .d = feature_width }, .bf16), .grid = grid }});
        errdefer expand.deinit();
        // Canonical row arrays are embedded as compile-time constants. Only two
        // block executables survive compilation, independent of the eight weight sets.
        var global = try dwd_plan.Plan.global(a, grid);
        defer global.deinit(a);
        var blocks: [2]zml.FnExe(Block.forward) = undefined;
        var done: usize = 0;
        errdefer for (blocks[0..done]) |*e| e.deinit();
        for (&blocks, 0..) |*exe, i| {
            const shifted = i != 0;
            var plan = try dwd_plan.Plan.windows(a, grid, shifted);
            defer plan.deinit(a);
            exe.* = try zml.FnExe(Block.forward).compile(a, run.io, run.platform, .{
                .meshes = &run.mesh,
                .program_name = if (shifted) "pi_h3_dwd_odd" else "pi_h3_dwd_even",
            }, .{.{
                .model = self.blocks[i],
                .hidden = hidden,
                .grid = grid,
                .shifted = shifted,
                .windows = plan.buckets.items,
                .global = global.buckets.items,
                .backend = backend,
            }});
            done += 1;
        }
        const finish = try zml.FnExe(Finish.forward).compile(a, run.io, run.platform, .{
            .meshes = &run.mesh,
            .program_name = "pi_h3_dwd_pixels",
        }, .{.{ .model = self.finish, .hidden = hidden, .grid = grid }});
        self.compiled = .{ .grid = grid, .expand = expand, .blocks = blocks, .finish = finish };
    }

    /// Borrow replicated [1,T*h*w,5376] final-evaluation video features.
    /// Return caller-owned, FP32 planar RGB (C,F,H,W), ready for writeOutputs.
    pub fn decode(self: *const Dwd, run: *const ops.Run, store: *zml.io.TensorStore, features: zml.Buffer) ![]f32 {
        const c = if (self.compiled) |*v| v else return error.NotCompiled;
        const a = run.allocator;
        if (features.shape().dtype() != .bf16 or
            !std.mem.eql(i64, features.shape().dims(), &.{ 1, @intCast(c.grid.videoRows() / 4), feature_width }))
            return error.InvalidDwdFeatures;

        var loader: zml.io.Loader = try .init(a, run.platform, ops.loader_opts);
        defer loader.deinit();
        var hidden: zml.Buffer = undefined;
        {
            var model = try ops.load(run, store, Expand, &self.expand, &loader);
            defer zml.Buffer.deinitAll(Expand, &model);
            var runner = try c.expand.runner(a);
            defer runner.deinit(a);
            runner.run(run.io, .{ .inputs = .{ .model = model, .features = features }, .outputs = .{ .hidden = &hidden } }, .{ .wait = true });
        }
        defer hidden.deinit();
        for (&self.blocks, 0..) |*block, i| {
            var model = try ops.load(run, store, Block, block, &loader);
            defer zml.Buffer.deinitAll(Block, &model);
            var runner = try c.blocks[i % 2].runner(a);
            defer runner.deinit(a);
            // Runner destroys the donated hidden handle and installs the new owner.
            runner.run(run.io, .{ .inputs = .{ .model = model, .hidden = hidden }, .outputs = .{ .hidden = &hidden } }, .{ .wait = true });
        }
        var model = try ops.load(run, store, Finish, &self.finish, &loader);
        defer zml.Buffer.deinitAll(Finish, &model);
        var runner = try c.finish.runner(a);
        defer runner.deinit(a);
        var rgb: zml.Buffer = undefined;
        runner.run(run.io, .{ .inputs = .{ .model = model, .hidden = hidden }, .outputs = .{ .rgb = &rgb } }, .{ .wait = true });
        defer rgb.deinit();
        const out = try a.alloc(f32, rgb.shape().count());
        errdefer a.free(out);
        try rgb.toSlice(run.io, .init(rgb.shape(), std.mem.sliceAsBytes(out)));
        return out;
    }
};
