//! The tiny growing Metal ops example. Starts at the f32 elementwise family and
//! grows one op at a time as the Metal AIR emitter gains coverage (then
//! reductions, …) — this is the regression suite and dev driver, NOT an
//! add-specific demo. See metal-xla-docs/PLAN.md.
//!
//! Each op runs on the CPU backend (the correctness oracle) and on Metal (via
//! our XLA fork's PJRT plugin + the new AIR-native emitter); the results must
//! match within a per-op tolerance (exact ops ~0; fast-math transcendentals
//! looser).
//!
//! Run: bazel run //examples/metal_ops --//platforms:metal=true

const std = @import("std");
const log = std.log;

const zml = @import("zml");

// --- the ops under test (elementwise f32) --------------------------------
fn add(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.add(b);
}
fn sub(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.sub(b);
}
fn mul(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.mul(b);
}
fn div(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.div(b);
}
fn maximum(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.maximum(b);
}
fn minimum(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.minimum(b);
}
fn fma3(a: zml.Tensor, b: zml.Tensor, c: zml.Tensor) zml.Tensor {
    return a.add(b).mul(c); // (a+b)*c — should fuse into ONE kFusion kernel
}
// Broadcast INSIDE a fusion (E5.2): x*scale + bias with scale,bias [d] broadcast
// to x's [b,d]. Fuses to one kFusion containing two broadcasts (the bias/scale
// pattern). Exercises the index-remap in the fused elemental emitter.
fn affineBcast(x: zml.Tensor, scale: zml.Tensor, bias: zml.Tensor) zml.Tensor {
    const xs = x.shape();
    return x.mul(scale.broad(xs)).add(bias.broad(xs));
}
// Matmul: contract the shared .k axis. With a={.m,.k} and b={.k,.n} this is a
// plain row-major NN matmul; with b={.n,.k} it's the y = x·Wᵀ linear (rhs
// contracts its inner dim) — same forward fn, the input tags pick NN vs NT.
fn matmul(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k);
}
// Multi-op module (whole-graph execution): y = x·W + bias. The add consumes the
// dot's RESULT, not an entry parameter — so this is NOT a single-op module and
// routes through the thunk-sequence graph executable: an MPSGraph matmul writing
// an intermediate buffer, then an elementwise-add kernel reading that + the bias.
fn linearBias(a: zml.Tensor, b: zml.Tensor, c: zml.Tensor) zml.Tensor {
    return a.dot(b, .k).add(c);
}
// Matmul feeding a FUSION (not a bare op): abs(negate(x·W)). negate∘abs fuse into
// one kFusion that CONSUMES the dot's result — exercises the graph path's fusion
// thunk with a non-parameter operand (bound by buffer slot). The dot stays a
// separate MPSGraph thunk (dots are kept out of fusions).
fn matNegAbs(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k).negate().abs();
}
// Matmul → indexed transform: transpose of the dot result. Exercises the graph
// path's indexed-copy thunk reading a computed (non-parameter) buffer.
// dot=[[22,28],[49,64]] → transpose = [[22,49],[28,64]].
fn matT(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k).transpose(.{ .n, .m });
}
// Matmul → reduce: sum the dot result over its n axis. Exercises the graph
// path's reduce thunk reading a computed buffer (and a top-level init constant).
// sum_n([[22,28],[49,64]]) = [50,113].
fn matSum(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k).sum(.n);
}
// Deeper chain: abs(x·W1)·W2 — a matmul, then abs (elementwise), then a matmul
// whose lhs is the abs RESULT (a computed buffer, not a parameter). Three thunks,
// two intermediates — exercises a matmul thunk reading an intermediate.
fn matAbsMat(x: zml.Tensor, w1: zml.Tensor, w2: zml.Tensor) zml.Tensor {
    return x.dot(w1, .k).abs().dot(w2, .n);
}
// Deeper chain (5 thunks: 3 matmuls + 2 abs kernels) — abs(abs(x·W1)·W2)·W3.
// Stresses command-buffer pipelining: many launches with no per-thunk block and
// several live intermediate + params buffers held until the single final block.
fn mlpDeep(x: zml.Tensor, w1: zml.Tensor, w2: zml.Tensor, w3: zml.Tensor) zml.Tensor {
    return x.dot(w1, .k).abs().dot(w2, .n).abs().dot(w3, .p);
}
// Batched matmul — the multi-head-attention primitive. a{h,m,k}·b{h,k,n}
// contracts .k and batches the shared .h → {h,m,n}: a batched dot_general (batch
// dim .h), lowered via MPSGraph's native batched matmul on the graph path.
fn bmm(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.dot(b, .k);
}
// GQA attention QKᵀ — the GENERAL batched dot the simple matcher can't express:
// the batch dim .hkv sits at a NON-leading operand position and q has TWO free
// dims (.s, .hg). q{s,hkv,hg,hd}·k{t,hkv,hd} contracts .hd, batches shared .hkv →
// {hkv,s,hg,t} (exactly Llama-3.2's shape). MPSGraph permutes each operand to
// [batch,M,K]/[batch,K,N] in-graph; the [batch,M,N] result is XLA's output order.
fn gqa(q: zml.Tensor, k: zml.Tensor) zml.Tensor {
    return q.dot(k, .hd);
}
// Multi-head attention CORE (pre-split heads, no projections): per-head scaled
// dot-product attention. q{h,s,e} k{h,t,e} v{h,t,e} → ctx{h,s,e}. Two BATCHED
// matmuls — scores = q·kᵀ (contract .e), ctx = attn·v (contract .t) — with a
// softmax over .t between; batch dim .h leads throughout. The exact shape real
// multi-head attention produces — proves batched matmul composes with the
// batched reduce/softmax path.
fn mha(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor) zml.Tensor {
    const e: f32 = @floatFromInt(q.dim(.e));
    const scores = q.dot(k, .e).scale(1.0 / @sqrt(e)); // {h,s,t}
    const attn = softmaxAx(scores, .t); // {h,s,t}
    return attn.dot(v, .t); // {h,s,e}
}
// Causal mask built ON-DEVICE from iota — the real use. Keep key j ≤ query i
// (lower triangle incl. diagonal), else −1e9. ZML's iota is s32, so the two
// iotas + the INTEGER compare + select + scalar all FUSE into one kernel: this
// exercises an integer iota and an integer compare INSIDE a fusion
// (EmitFusedValue), the shape real masking produces. Output stays f32 (select).
fn causalMask(x: zml.Tensor) zml.Tensor {
    const i = zml.Tensor.iota(x.shape(), .r); // query (row) index  (s32)
    const j = zml.Tensor.iota(x.shape(), .c); // key (col) index    (s32)
    const keep = j.cmp(.LE, i); // s32 compare: col ≤ row
    return keep.select(x, zml.Tensor.scalar(-1e9, .f32).broad(x.shape()));
}
// CONVERT inside a fusion — the bridge from integer indices to float math (the
// RoPE-positions building block): iota (s32) → convert(.f32) → add x. The
// convert is an s32→f32 cast (sitofp) computed inline. out[r,c] = c + x[r,c].
fn posBias(x: zml.Tensor) zml.Tensor {
    return zml.Tensor.iota(x.shape(), .c).convert(.f32).add(x);
}
// RoPE (rotary position embedding) — the headline op, composed entirely of now-
// supported pieces. With pos_idx=null it builds positions via arange (iota
// s32→convert f32); inv_freq is a host-computed constant. Then: outer(pos,
// inv_freq) = broadcast×broadcast×mul → sin/cos → scale → convert → broadcast;
// splitRealImg STATIC-SLICEs the .hd axis into halves; the rotation is mul/sub +
// mul/add; mergeRealImg CONCATENATEs. Exercises the new kSlice indexed path end
// to end. Sequential layout (HF), default scaling (theta=10000).
fn ropeFwd(x: zml.Tensor) zml.Tensor {
    return zml.nn.rope(x, null, .{ .layout = .sequential, .scaling = .{ .default = .{} } });
}
// RoPE with EXPLICIT non-zero positions (the decode path: positions start at
// token_index, not 0). metal_ops `rope` only covers pos_idx=null (arange from 0),
// so the rope-at-offset math is otherwise untested.
fn ropePosFwd(x: zml.Tensor) zml.Tensor {
    const pos = zml.Tensor.arange(.{ .start = 5, .end = 5 + x.dim(.s) }, .i32).withTags(.{.s});
    return zml.nn.rope(x, pos, .{ .layout = .sequential, .scaling = .{ .default = .{} } });
}
// Mixed-precision elementwise: a*b + a. Verifies f16/bf16 STORAGE through the
// fusion path — load half/bfloat → f32 compute → per-op round → store. The
// interior a*b is rounded to the node dtype before the +a (matching XLA's
// per-op rounding), so a rounding bug would show. Two ops → one fusion.
fn mulAdd(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return a.mul(b).add(a);
}
// KV-cache indexing ops with a RUNTIME offset. dynSlice: read a fixed-length
// window of x at a runtime start. dynUpdate: write `upd` into x at a runtime
// start, returning the updated array (the KV-cache write).
fn dynSlice(x: zml.Tensor, off: zml.Tensor) zml.Tensor {
    return x.dynamicSlice1d(0, .{ .start = off, .len = 3 });
}
// gatherSlices with a RUNTIME index — exactly how zml.attention's vanilla backend
// builds the DECODE attention mask: take the causal-mask ROW at `token_index`
// (slice extent 1 along .q). Reshape the harness's flat x into a [q=4,k=4] mask
// with distinct rows so a wrong row is obvious. off=2 must return row 2.
fn gatherRow(x: zml.Tensor, off: zml.Tensor) zml.Tensor {
    const m = x.reshape(.{ .q = 4, .k = 4 });
    return m.gatherSlices(zml.Shape.init(.{ .q = 1 }, x.dtype()), off.reshape(.{ .coord = 1 }), .{});
}
fn dynUpdate(x: zml.Tensor, upd: zml.Tensor, off: zml.Tensor) zml.Tensor {
    return x.dynamicUpdateSlice1d(upd, 0, off);
}
// KV-cache WRITE as the real model does it: scatterSlices a single .k-slice at a
// RUNTIME index, override, with reuseBuffer (in-place donation). This is what
// KvCache.updateAt uses (model.zig) — distinct from dynamicUpdateSlice (kvwrt).
// off=2 into a [k=4,d=2] zero cache with row {7,8} -> {0,0, 0,0, 7,8, 0,0}.
fn scatterRow(x: zml.Tensor, upd: zml.Tensor, off: zml.Tensor) zml.Tensor {
    const cache = x.reshape(.{ .k = 4, .d = 2 });
    const row = upd.reshape(.{ .k = 1, .d = 2 });
    return cache.scatterSlices(.{ .k = off }, row, .{ .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(cache);
}
// Embedding lookup: gather rows of a table[V,D] by an index vector[N] → [N,D].
// out[n,:] = table[idx[n],:]. The canonical model embedding / vocab gather.
fn embed(table: zml.Tensor, idx: zml.Tensor) zml.Tensor {
    return table.gather(.{ .v = idx }, .{});
}
// KV-cache row write: cache[rows,cols], write row[1,cols] at a runtime position
// along axis 0 (the other axis offset is a 0 constant) → updated cache. Rank-2,
// exercises the DUS kernel's multi-dim coordinate logic.
fn kvWrite(cache: zml.Tensor, row: zml.Tensor, pos: zml.Tensor) zml.Tensor {
    return cache.dynamicUpdateSlice1d(row, 0, pos);
}
// Concatenate two tensors along .j: [2,2] ++ [2,2] -> [2,4]. (RoPE rotate-half /
// KV-cache append use this op.)
fn concatJ(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
    return zml.Tensor.concatenate(&.{ a, b }, .j);
}
// Multi-output module: a single graph with TWO array results — root is a
// tuple(add, mul). Exercises the graph executable's tuple-output path
// (per-leaf buffer + root tuple index table via WriteRootTupleIndexTable).
fn addMul(a: zml.Tensor, b: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
    return .{ a.add(b), a.mul(b) };
}
// Numerically-stable softmax over .j, built from supported ops only (no
// compare/select): exp(x - max) / sum(exp(x - max)). The transformer attention
// weight normalizer.
fn mySoftmax(x: zml.Tensor) zml.Tensor {
    const m = x.max(.j);
    const e = x.sub(m.broad(x.shape())).exp();
    return e.div(e.sum(.j).broad(x.shape()));
}
// ZML's STOCK softmax (not the hand-written one): emits compare + select for the
// all-(-inf)-row NaN guard. Verifies compare/select lower (fused, no PRED buffer).
fn stockSoftmax(a: zml.Tensor) zml.Tensor {
    return a.softmax(.j);
}
// RMSNorm over .j (mul-based, avoids pow): x · rsqrt(mean(x²) + eps).
fn myRmsNorm(x: zml.Tensor) zml.Tensor {
    const d: f32 = @floatFromInt(x.dim(.j));
    const ms = x.mul(x).sum(.j).scale(1.0 / d); // [i] mean square
    const inv = ms.addConstant(1e-6).rsqrt(); // [i]
    return x.mul(inv.broad(x.shape())); // [i,j]
}
// Axis-generic versions of the transformer normalizers (used by the block).
fn softmaxAx(x: zml.Tensor, comptime axis: anytype) zml.Tensor {
    const m = x.max(axis);
    const e = x.sub(m.broad(x.shape())).exp();
    return e.div(e.sum(axis).broad(x.shape()));
}
fn rmsNormAx(x: zml.Tensor, comptime axis: anytype) zml.Tensor {
    const d: f32 = @floatFromInt(x.dim(axis));
    const ms = x.mul(x).sum(axis).scale(1.0 / d);
    const inv = ms.addConstant(1e-6).rsqrt();
    return x.mul(inv.broad(x.shape()));
}

// A full tiny TRANSFORMER block end-to-end: pre-norm single-head self-attention
// with a causal mask, residual, pre-norm MLP (relu), residual. Everything runs
// through the Metal graph path: rmsNorm (reduce+fusion), Q/K/V/O matmuls, scaled
// QKᵀ + mask, softmax (reduce→fusion→reduce→fusion), attn·V, MLP, two residuals.
//   x[s,d]  wq/wk/wv[d,hd]  wo[hd,d]  mask[s,k]  w1[d,ff]  w2[ff,d] -> [s,d]
fn attnBlock(
    x: zml.Tensor,
    wq: zml.Tensor,
    wk: zml.Tensor,
    wv: zml.Tensor,
    wo: zml.Tensor,
    mask: zml.Tensor,
    w1: zml.Tensor,
    w2: zml.Tensor,
) zml.Tensor {
    const hd: f32 = @floatFromInt(wq.dim(.hd));
    const xn = rmsNormAx(x, .d); // [s,d]
    const q = xn.dot(wq, .d); // [s,hd]
    const k = xn.dot(wk, .d).rename(.{ .s = .k }); // [k,hd]
    const v = xn.dot(wv, .d).rename(.{ .s = .k }); // [k,hd]
    const scores = q.dot(k, .hd).scale(1.0 / @sqrt(hd)).add(mask); // [s,k]
    const attn = softmaxAx(scores, .k); // [s,k]
    const proj = attn.dot(v, .k).dot(wo, .hd); // [s,d]
    const x2 = x.add(proj); // residual
    const hn = rmsNormAx(x2, .d);
    const h = hn.dot(w1, .d).relu(); // [s,ff]
    return x2.add(h.dot(w2, .ff)); // residual [s,d]
}
// The same block, now the REAL Llama-decode-layer shape and dtype-generic (one
// forward fn, run at f32/f16/bf16): pre-norm self-attention with RoPE on Q,K and
// an ON-DEVICE causal mask, residual; pre-norm relu MLP, residual. Composes every
// dtype-generic family at once — rmsNorm (reduce+fusion), Q/K/V/O matmuls, RoPE
// (slice/iota/baked-const/sin-cos/concat fusions), scaled QKᵀ, iota×2→cmp→select
// mask, softmax (reduce→fusion→reduce→fusion), attn·V, MLP. No mask input — it is
// generated from iota — so the whole layer is just weights + x.
//   x[s,d]  wq/wk/wv[d,hd]  wo[hd,d]  w1[d,ff]  w2[ff,d] -> [s,d]
fn attnBlockRope(
    x: zml.Tensor,
    wq: zml.Tensor,
    wk: zml.Tensor,
    wv: zml.Tensor,
    wo: zml.Tensor,
    w1: zml.Tensor,
    w2: zml.Tensor,
) zml.Tensor {
    const hd: f32 = @floatFromInt(wq.dim(.hd));
    const xn = rmsNormAx(x, .d); // [s,d]
    const q = zml.nn.rope(xn.dot(wq, .d), null, .{}); // [s,hd] + RoPE
    const k = zml.nn.rope(xn.dot(wk, .d), null, .{}).rename(.{ .s = .k }); // [k,hd]
    const v = xn.dot(wv, .d).rename(.{ .s = .k }); // [k,hd]
    const raw = q.dot(k, .hd).scale(1.0 / @sqrt(hd)); // [s,k]
    // On-device causal mask: keep key ≤ query, else −1e9 (in the score dtype).
    const qi = zml.Tensor.iota(raw.shape(), .s);
    const ki = zml.Tensor.iota(raw.shape(), .k);
    const scores = ki.cmp(.LE, qi).select(raw, zml.Tensor.scalar(-1e9, raw.dtype()).broad(raw.shape()));
    const attn = softmaxAx(scores, .k); // [s,k]
    const proj = attn.dot(v, .k).dot(wo, .hd); // [s,d]
    const x2 = x.add(proj); // residual
    const hn = rmsNormAx(x2, .d);
    const h = hn.dot(w1, .d).relu(); // [s,ff]
    return x2.add(h.dot(w2, .ff)); // residual [s,d]
}
// A tiny REAL model end-to-end: a per-token 2-layer MLP classifier.
//   e      = embed(table, tokens)        // gather rows           [n, d]
//   h      = relu(e·W1 + b1)             // matmul, +bias, relu   [n, h]
//   logits = h·W2 + b2                   // matmul, +bias         [n, c]
// Exercises the whole graph path at once: gather → matmul → fusion(bias-add +
// relu, with a scalar constant inside) → matmul → fusion(bias-add), pipelined.
fn tinyMLP(tokens: zml.Tensor, table: zml.Tensor, w1: zml.Tensor, b1: zml.Tensor, w2: zml.Tensor, b2: zml.Tensor) zml.Tensor {
    const e = table.gather(.{ .v = tokens }, .{}); // [n, d]
    const z1 = e.dot(w1, .d); // [n, h]
    const h = z1.add(b1.broad(z1.shape())).relu(); // [n, h]
    const z2 = h.dot(w2, .h); // [n, c]
    return z2.add(b2.broad(z2.shape())); // [n, c]
}
fn negate(a: zml.Tensor) zml.Tensor {
    return a.negate();
}
fn abs(a: zml.Tensor) zml.Tensor {
    return a.abs();
}
fn exp(a: zml.Tensor) zml.Tensor {
    return a.exp();
}
fn log_(a: zml.Tensor) zml.Tensor {
    return a.log();
}
fn sqrt(a: zml.Tensor) zml.Tensor {
    return a.sqrt();
}
fn rsqrt(a: zml.Tensor) zml.Tensor {
    return a.rsqrt();
}
fn tanh(a: zml.Tensor) zml.Tensor {
    return a.tanh();
}
fn sin_(a: zml.Tensor) zml.Tensor {
    return a.sin();
}
fn cos_(a: zml.Tensor) zml.Tensor {
    return a.cos();
}
fn silu_(a: zml.Tensor) zml.Tensor {
    return a.silu(); // x · sigmoid(x) — exercises the logistic op
}
// SwiGLU MLP (the Llama/Mistral feed-forward): silu(x·Wg) * (x·Wu) · Wd.
fn swiglu(x: zml.Tensor, wg: zml.Tensor, wu: zml.Tensor, wd: zml.Tensor) zml.Tensor {
    return x.dot(wg, .k).silu().mul(x.dot(wu, .k)).dot(wd, .n);
}

fn runBinaryOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_data: []const f32,
    b_data: []const f32,
    out: []f32,
) !void {
    const shape = zml.Shape.init(.{ .n = a_data.len }, .f32);
    const a_t: zml.Tensor = .fromShape(shape);
    const b_t: zml.Tensor = .fromShape(shape);
    var exe = try platform.compileFn(allocator, io, func, .{ a_t, b_t }, .{});
    defer exe.deinit();

    var a_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer a_buf.deinit();
    var b_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(b_data));
    defer b_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ a_buf, b_buf });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn runTernaryOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_data: []const f32,
    b_data: []const f32,
    c_data: []const f32,
    out: []f32,
) !void {
    const shape = zml.Shape.init(.{ .n = a_data.len }, .f32);
    const a_t: zml.Tensor = .fromShape(shape);
    const b_t: zml.Tensor = .fromShape(shape);
    const c_t: zml.Tensor = .fromShape(shape);
    var exe = try platform.compileFn(allocator, io, func, .{ a_t, b_t, c_t }, .{});
    defer exe.deinit();

    var a_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer a_buf.deinit();
    var b_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(b_data));
    defer b_buf.deinit();
    var c_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(c_data));
    defer c_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ a_buf, b_buf, c_buf });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

// dynamic-slice: x[n] f32 + a runtime i32 scalar offset → out[len].
fn runDynSliceOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    x_data: []const f32,
    off_val: i32,
    out: []f32,
) !void {
    const x_shape = zml.Shape.init(.{ .n = x_data.len }, .f32);
    const off_shape = zml.Shape.init(.{}, .i32);
    const xt: zml.Tensor = .fromShape(x_shape);
    const ot: zml.Tensor = .fromShape(off_shape);
    var exe = try platform.compileFn(allocator, io, func, .{ xt, ot }, .{});
    defer exe.deinit();

    var xb = try zml.Buffer.fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(x_data));
    defer xb.deinit();
    const off_arr = [_]i32{off_val};
    var ob = try zml.Buffer.fromBytes(io, platform, off_shape, .replicated, std.mem.sliceAsBytes(&off_arr));
    defer ob.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ xb, ob });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);
    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn checkDynSlice(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    x_data: []const f32,
    off_val: i32,
    n_out: usize,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    runDynSliceOn(allocator, io, cpu, func, x_data, off_val, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runDynSliceOn(allocator, io, metal, func, x_data, off_val, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n_out], mo[0..n_out], 1e-6);
}

// dynamic-update-slice: x[n] f32, upd[m] f32, runtime i32 offset → out[n].
fn runDynUpdateOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    x_data: []const f32,
    upd_data: []const f32,
    off_val: i32,
    out: []f32,
) !void {
    const x_shape = zml.Shape.init(.{ .n = x_data.len }, .f32);
    const upd_shape = zml.Shape.init(.{ .n = upd_data.len }, .f32);
    const off_shape = zml.Shape.init(.{}, .i32);
    const xt: zml.Tensor = .fromShape(x_shape);
    const ut: zml.Tensor = .fromShape(upd_shape);
    const ot: zml.Tensor = .fromShape(off_shape);
    var exe = try platform.compileFn(allocator, io, func, .{ xt, ut, ot }, .{});
    defer exe.deinit();

    var xb = try zml.Buffer.fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(x_data));
    defer xb.deinit();
    var ub = try zml.Buffer.fromBytes(io, platform, upd_shape, .replicated, std.mem.sliceAsBytes(upd_data));
    defer ub.deinit();
    const off_arr = [_]i32{off_val};
    var ob = try zml.Buffer.fromBytes(io, platform, off_shape, .replicated, std.mem.sliceAsBytes(&off_arr));
    defer ob.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ xb, ub, ob });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);
    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn checkDynUpdate(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    x_data: []const f32,
    upd_data: []const f32,
    off_val: i32,
    n_out: usize,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    runDynUpdateOn(allocator, io, cpu, func, x_data, upd_data, off_val, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runDynUpdateOn(allocator, io, metal, func, x_data, upd_data, off_val, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n_out], mo[0..n_out], 1e-6);
}

// Tiny MLP classifier end-to-end: tokens[N] i32 + table[V,D] + W1[D,H] + b1[H] +
// W2[H,C] + b2[C] (all f32) → logits[N,C]. One graph, ~6 thunks.
fn runTinyMLPOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    N: i64,
    V: i64,
    D: i64,
    H: i64,
    C: i64,
    tokens: []const i32,
    table: []const f32,
    w1: []const f32,
    b1: []const f32,
    w2: []const f32,
    b2: []const f32,
    out: []f32,
) !void {
    const tok_shape = zml.Shape.init(.{ .n = N }, .i32);
    const table_shape = zml.Shape.init(.{ .v = V, .d = D }, .f32);
    const w1_shape = zml.Shape.init(.{ .d = D, .h = H }, .f32);
    const b1_shape = zml.Shape.init(.{ .h = H }, .f32);
    const w2_shape = zml.Shape.init(.{ .h = H, .c = C }, .f32);
    const b2_shape = zml.Shape.init(.{ .c = C }, .f32);
    const tokt: zml.Tensor = .fromShape(tok_shape);
    const tat: zml.Tensor = .fromShape(table_shape);
    const w1t: zml.Tensor = .fromShape(w1_shape);
    const b1t: zml.Tensor = .fromShape(b1_shape);
    const w2t: zml.Tensor = .fromShape(w2_shape);
    const b2t: zml.Tensor = .fromShape(b2_shape);
    var exe = try platform.compileFn(allocator, io, tinyMLP, .{ tokt, tat, w1t, b1t, w2t, b2t }, .{});
    defer exe.deinit();

    var tokb = try zml.Buffer.fromBytes(io, platform, tok_shape, .replicated, std.mem.sliceAsBytes(tokens));
    defer tokb.deinit();
    var tab = try zml.Buffer.fromBytes(io, platform, table_shape, .replicated, std.mem.sliceAsBytes(table));
    defer tab.deinit();
    var w1b = try zml.Buffer.fromBytes(io, platform, w1_shape, .replicated, std.mem.sliceAsBytes(w1));
    defer w1b.deinit();
    var b1b = try zml.Buffer.fromBytes(io, platform, b1_shape, .replicated, std.mem.sliceAsBytes(b1));
    defer b1b.deinit();
    var w2b = try zml.Buffer.fromBytes(io, platform, w2_shape, .replicated, std.mem.sliceAsBytes(w2));
    defer w2b.deinit();
    var b2b = try zml.Buffer.fromBytes(io, platform, b2_shape, .replicated, std.mem.sliceAsBytes(b2));
    defer b2b.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ tokb, tab, w1b, b1b, w2b, b2b });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);
    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn checkTinyMLP(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
) usize {
    const N: i64 = 3;
    const V: i64 = 5;
    const D: i64 = 4;
    const H: i64 = 6;
    const C: i64 = 2;
    const tokens = [_]i32{ 2, 0, 4 };
    var table: [20]f32 = undefined; // V*D
    var w1: [24]f32 = undefined; // D*H
    var b1: [6]f32 = undefined; // H
    var w2: [12]f32 = undefined; // H*C
    var b2: [2]f32 = undefined; // C
    fillFrac(f32, &table, 13);
    fillFrac(f32, &w1, 7);
    fillFrac(f32, &b1, 3);
    fillFrac(f32, &w2, 5);
    fillFrac(f32, &b2, 2);
    const n_out: usize = @intCast(N * C);
    var co: [16]f32 = undefined;
    var mo: [16]f32 = undefined;
    runTinyMLPOn(allocator, io, cpu, N, V, D, H, C, &tokens, &table, &w1, &b1, &w2, &b2, co[0..n_out]) catch |e| {
        log.err("tinyMLP  CPU failed: {s}", .{@errorName(e)});
        return 1;
    };
    runTinyMLPOn(allocator, io, metal, N, V, D, H, C, &tokens, &table, &w1, &b1, &w2, &b2, mo[0..n_out]) catch |e| {
        log.err("tinyMLP  Metal failed: {s}", .{@errorName(e)});
        return 1;
    };
    return compare("tinyMLP", co[0..n_out], mo[0..n_out], 1e-4);
}

// Tiny transformer block runner: S tokens, D model dim, Dh head dim, FF mlp dim.
// 8 f32 inputs (x, wq, wk, wv, wo, mask, w1, w2) → [S,D].
fn runAttnBlockOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    S: i64,
    D: i64,
    Dh: i64,
    FF: i64,
    x: []const f32,
    wq: []const f32,
    wk: []const f32,
    wv: []const f32,
    wo: []const f32,
    mask: []const f32,
    w1: []const f32,
    w2: []const f32,
    out: []f32,
) !void {
    const sx = zml.Shape.init(.{ .s = S, .d = D }, .f32);
    const sqkv = zml.Shape.init(.{ .d = D, .hd = Dh }, .f32);
    const swo = zml.Shape.init(.{ .hd = Dh, .d = D }, .f32);
    const smask = zml.Shape.init(.{ .s = S, .k = S }, .f32);
    const sw1 = zml.Shape.init(.{ .d = D, .ff = FF }, .f32);
    const sw2 = zml.Shape.init(.{ .ff = FF, .d = D }, .f32);
    const tx: zml.Tensor = .fromShape(sx);
    const tq: zml.Tensor = .fromShape(sqkv);
    const tk: zml.Tensor = .fromShape(sqkv);
    const tv: zml.Tensor = .fromShape(sqkv);
    const two: zml.Tensor = .fromShape(swo);
    const tm: zml.Tensor = .fromShape(smask);
    const t1: zml.Tensor = .fromShape(sw1);
    const t2: zml.Tensor = .fromShape(sw2);
    var exe = try platform.compileFn(allocator, io, attnBlock, .{ tx, tq, tk, tv, two, tm, t1, t2 }, .{});
    defer exe.deinit();

    var bx = try zml.Buffer.fromBytes(io, platform, sx, .replicated, std.mem.sliceAsBytes(x));
    defer bx.deinit();
    var bq = try zml.Buffer.fromBytes(io, platform, sqkv, .replicated, std.mem.sliceAsBytes(wq));
    defer bq.deinit();
    var bk = try zml.Buffer.fromBytes(io, platform, sqkv, .replicated, std.mem.sliceAsBytes(wk));
    defer bk.deinit();
    var bv = try zml.Buffer.fromBytes(io, platform, sqkv, .replicated, std.mem.sliceAsBytes(wv));
    defer bv.deinit();
    var bwo = try zml.Buffer.fromBytes(io, platform, swo, .replicated, std.mem.sliceAsBytes(wo));
    defer bwo.deinit();
    var bm = try zml.Buffer.fromBytes(io, platform, smask, .replicated, std.mem.sliceAsBytes(mask));
    defer bm.deinit();
    var b1 = try zml.Buffer.fromBytes(io, platform, sw1, .replicated, std.mem.sliceAsBytes(w1));
    defer b1.deinit();
    var b2 = try zml.Buffer.fromBytes(io, platform, sw2, .replicated, std.mem.sliceAsBytes(w2));
    defer b2.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ bx, bq, bk, bv, bwo, bm, b1, b2 });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);
    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn checkAttnBlock(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
) usize {
    const S: i64 = 4;
    const D: i64 = 6;
    const Dh: i64 = 6;
    const FF: i64 = 12;
    var x: [24]f32 = undefined; // S*D
    var wq: [36]f32 = undefined; // D*Dh
    var wk: [36]f32 = undefined;
    var wv: [36]f32 = undefined;
    var wo: [36]f32 = undefined; // Dh*D
    var w1: [72]f32 = undefined; // D*FF
    var w2: [72]f32 = undefined; // FF*D
    fillFrac(f32, &x, 11);
    fillFrac(f32, &wq, 3);
    fillFrac(f32, &wk, 5);
    fillFrac(f32, &wv, 7);
    fillFrac(f32, &wo, 9);
    fillFrac(f32, &w1, 13);
    fillFrac(f32, &w2, 17);
    // Causal additive mask [S,S]: 0 on/below the diagonal, -1e9 above.
    var mask: [16]f32 = undefined;
    for (0..@intCast(S)) |i| {
        for (0..@intCast(S)) |j| {
            mask[i * @as(usize, @intCast(S)) + j] = if (j <= i) 0.0 else -1.0e9;
        }
    }
    const n_out: usize = @intCast(S * D);
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    runAttnBlockOn(allocator, io, cpu, S, D, Dh, FF, &x, &wq, &wk, &wv, &wo, &mask, &w1, &w2, co[0..n_out]) catch |e| {
        log.err("attn  CPU failed: {s}", .{@errorName(e)});
        return 1;
    };
    runAttnBlockOn(allocator, io, metal, S, D, Dh, FF, &x, &wq, &wk, &wv, &wo, &mask, &w1, &w2, mo[0..n_out]) catch |e| {
        log.err("attn  Metal failed: {s}", .{@errorName(e)});
        return 1;
    };
    return compare("attn", co[0..n_out], mo[0..n_out], 1e-3);
}

// Dtype-generic transformer-block-with-RoPE runner (attnBlockRope). 7 typed
// inputs (x, wq, wk, wv, wo, w1, w2) → [S,D]; T = f32/f16/bf16.
fn runAttnBlockRopeOn(
    comptime T: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    S: i64,
    D: i64,
    Dh: i64,
    FF: i64,
    x: []const T,
    wq: []const T,
    wk: []const T,
    wv: []const T,
    wo: []const T,
    w1: []const T,
    w2: []const T,
    out: []T,
) !void {
    const dt = dtOf(T);
    const sx = zml.Shape.init(.{ .s = S, .d = D }, dt);
    const sqkv = zml.Shape.init(.{ .d = D, .hd = Dh }, dt);
    const swo = zml.Shape.init(.{ .hd = Dh, .d = D }, dt);
    const sw1 = zml.Shape.init(.{ .d = D, .ff = FF }, dt);
    const sw2 = zml.Shape.init(.{ .ff = FF, .d = D }, dt);
    const tx: zml.Tensor = .fromShape(sx);
    const tq: zml.Tensor = .fromShape(sqkv);
    const tk: zml.Tensor = .fromShape(sqkv);
    const tv: zml.Tensor = .fromShape(sqkv);
    const two: zml.Tensor = .fromShape(swo);
    const t1: zml.Tensor = .fromShape(sw1);
    const t2: zml.Tensor = .fromShape(sw2);
    var exe = try platform.compileFn(allocator, io, attnBlockRope, .{ tx, tq, tk, tv, two, t1, t2 }, .{});
    defer exe.deinit();

    var bx = try zml.Buffer.fromBytes(io, platform, sx, .replicated, std.mem.sliceAsBytes(x));
    defer bx.deinit();
    var bq = try zml.Buffer.fromBytes(io, platform, sqkv, .replicated, std.mem.sliceAsBytes(wq));
    defer bq.deinit();
    var bk = try zml.Buffer.fromBytes(io, platform, sqkv, .replicated, std.mem.sliceAsBytes(wk));
    defer bk.deinit();
    var bv = try zml.Buffer.fromBytes(io, platform, sqkv, .replicated, std.mem.sliceAsBytes(wv));
    defer bv.deinit();
    var bwo = try zml.Buffer.fromBytes(io, platform, swo, .replicated, std.mem.sliceAsBytes(wo));
    defer bwo.deinit();
    var b1 = try zml.Buffer.fromBytes(io, platform, sw1, .replicated, std.mem.sliceAsBytes(w1));
    defer b1.deinit();
    var b2 = try zml.Buffer.fromBytes(io, platform, sw2, .replicated, std.mem.sliceAsBytes(w2));
    defer b2.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ bx, bq, bk, bv, bwo, b1, b2 });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);
    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(T));
}

fn checkAttnBlockRope(
    comptime T: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    rel_tol: f32,
) usize {
    const S: i64 = 4;
    const D: i64 = 6;
    const Dh: i64 = 6;
    const FF: i64 = 12;
    var x: [24]T = undefined; // S*D
    var wq: [36]T = undefined; // D*Dh
    var wk: [36]T = undefined;
    var wv: [36]T = undefined;
    var wo: [36]T = undefined; // Dh*D
    var w1: [72]T = undefined; // D*FF
    var w2: [72]T = undefined; // FF*D
    fillFrac(T, &x, 11);
    fillFrac(T, &wq, 3);
    fillFrac(T, &wk, 5);
    fillFrac(T, &wv, 7);
    fillFrac(T, &wo, 9);
    fillFrac(T, &w1, 13);
    fillFrac(T, &w2, 17);
    const n_out: usize = @intCast(S * D);
    var co: [64]T = undefined;
    var mo: [64]T = undefined;
    runAttnBlockRopeOn(T, allocator, io, cpu, S, D, Dh, FF, &x, &wq, &wk, &wv, &wo, &w1, &w2, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runAttnBlockRopeOn(T, allocator, io, metal, S, D, Dh, FF, &x, &wq, &wk, &wv, &wo, &w1, &w2, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    var cof: [64]f32 = undefined;
    var mof: [64]f32 = undefined;
    for (0..n_out) |i| {
        cof[i] = toF32(T, co[i]);
        mof[i] = toF32(T, mo[i]);
    }
    // A full attention block composes ~15 ops with residual cancellation; the
    // Metal-vs-CPU divergence bottoms out at ~1 output-dtype ULP (proven: the
    // identical f16 kernels pass at 6.9e-3, 14x under tol; the bf16 worst error
    // is exactly 2^-7). Use a dtype-scaled abs floor so a 1-ULP error on a
    // near-zero post-cancellation lane isn't read as a 10% relative miss.
    return compareAbs(name, cof[0..n_out], mof[0..n_out], rel_tol, absFloorFor(T));
}

// Embedding lookup: table[V,D] f32 + idx[N] i32 → out[N,D] f32.
fn runEmbedOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    vocab: i64,
    dim: i64,
    table_data: []const f32,
    idx_data: []const i32,
    out: []f32,
) !void {
    const table_shape = zml.Shape.init(.{ .v = vocab, .d = dim }, .f32);
    const idx_shape = zml.Shape.init(.{ .n = idx_data.len }, .i32);
    const tt: zml.Tensor = .fromShape(table_shape);
    const it: zml.Tensor = .fromShape(idx_shape);
    var exe = try platform.compileFn(allocator, io, embed, .{ tt, it }, .{});
    defer exe.deinit();

    var tb = try zml.Buffer.fromBytes(io, platform, table_shape, .replicated, std.mem.sliceAsBytes(table_data));
    defer tb.deinit();
    var ib = try zml.Buffer.fromBytes(io, platform, idx_shape, .replicated, std.mem.sliceAsBytes(idx_data));
    defer ib.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ tb, ib });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);
    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn checkEmbed(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    vocab: i64,
    dim: i64,
    table_data: []const f32,
    idx_data: []const i32,
) usize {
    const n: usize = @intCast(@as(i64, @intCast(idx_data.len)) * dim);
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    runEmbedOn(allocator, io, cpu, vocab, dim, table_data, idx_data, co[0..n]) catch |e| {
        log.err("embed  CPU failed: {s}", .{@errorName(e)});
        return 1;
    };
    runEmbedOn(allocator, io, metal, vocab, dim, table_data, idx_data, mo[0..n]) catch |e| {
        log.err("embed  Metal failed: {s}", .{@errorName(e)});
        return 1;
    };
    return compare("embed", co[0..n], mo[0..n], 1e-6);
}

// KV-cache row write: cache[rows,cols] f32 + row[1,cols] f32 + runtime i32 pos
// → updated cache[rows,cols]. Exercises rank-2 dynamic-update-slice.
fn runKvWriteOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    rows: i64,
    cols: i64,
    cache_data: []const f32,
    row_data: []const f32,
    pos_val: i32,
    out: []f32,
) !void {
    const cache_shape = zml.Shape.init(.{ .r = rows, .c = cols }, .f32);
    const row_shape = zml.Shape.init(.{ .r = 1, .c = cols }, .f32);
    const pos_shape = zml.Shape.init(.{}, .i32);
    const ct: zml.Tensor = .fromShape(cache_shape);
    const rt: zml.Tensor = .fromShape(row_shape);
    const pt: zml.Tensor = .fromShape(pos_shape);
    var exe = try platform.compileFn(allocator, io, kvWrite, .{ ct, rt, pt }, .{});
    defer exe.deinit();

    var cb = try zml.Buffer.fromBytes(io, platform, cache_shape, .replicated, std.mem.sliceAsBytes(cache_data));
    defer cb.deinit();
    var rb = try zml.Buffer.fromBytes(io, platform, row_shape, .replicated, std.mem.sliceAsBytes(row_data));
    defer rb.deinit();
    const pos_arr = [_]i32{pos_val};
    var pb = try zml.Buffer.fromBytes(io, platform, pos_shape, .replicated, std.mem.sliceAsBytes(&pos_arr));
    defer pb.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ cb, rb, pb });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);
    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn checkKvWrite(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    rows: i64,
    cols: i64,
    cache_data: []const f32,
    row_data: []const f32,
    pos_val: i32,
) usize {
    const n: usize = @intCast(rows * cols);
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    runKvWriteOn(allocator, io, cpu, rows, cols, cache_data, row_data, pos_val, co[0..n]) catch |e| {
        log.err("kvwrt  CPU failed: {s}", .{@errorName(e)});
        return 1;
    };
    runKvWriteOn(allocator, io, metal, rows, cols, cache_data, row_data, pos_val, mo[0..n]) catch |e| {
        log.err("kvwrt  Metal failed: {s}", .{@errorName(e)});
        return 1;
    };
    return compare("kvwrt", co[0..n], mo[0..n], 1e-6);
}

// Run a binary func returning TWO tensors; fill out0/out1 from the two result
// buffers. Drives the tuple-output graph path.
fn runTwoOutOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_data: []const f32,
    b_data: []const f32,
    out0: []f32,
    out1: []f32,
) !void {
    const shape = zml.Shape.init(.{ .n = a_data.len }, .f32);
    const a_t: zml.Tensor = .fromShape(shape);
    const b_t: zml.Tensor = .fromShape(shape);
    var exe = try platform.compileFn(allocator, io, func, .{ a_t, b_t }, .{});
    defer exe.deinit();

    var a_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer a_buf.deinit();
    var b_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(b_data));
    defer b_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ a_buf, b_buf });
    exe.call(exe_args, &exe_results);

    var results = exe_results.get(struct { zml.Buffer, zml.Buffer });
    defer results[0].deinit();
    defer results[1].deinit();
    _ = try results[0].await(io);
    _ = try results[1].await(io);

    const s0 = try results[0].toSliceAlloc(allocator, io);
    defer s0.free(allocator);
    const s1 = try results[1].toSliceAlloc(allocator, io);
    defer s1.free(allocator);
    @memcpy(out0, s0.items(f32));
    @memcpy(out1, s1.items(f32));
}

fn checkTwoOut(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a: []const f32,
    b: []const f32,
    rel_tol: f32,
) usize {
    var c0: [64]f32 = undefined;
    var c1: [64]f32 = undefined;
    var m0: [64]f32 = undefined;
    var m1: [64]f32 = undefined;
    const n = a.len;
    runTwoOutOn(allocator, io, cpu, func, a, b, c0[0..n], c1[0..n]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runTwoOutOn(allocator, io, metal, func, a, b, m0[0..n], m1[0..n]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, c0[0..n], m0[0..n], rel_tol) +
        compare(name, c1[0..n], m1[0..n], rel_tol);
}

fn runUnaryOn(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_data: []const f32,
    out: []f32,
) !void {
    const shape = zml.Shape.init(.{ .n = a_data.len }, .f32);
    const a_t: zml.Tensor = .fromShape(shape);
    var exe = try platform.compileFn(allocator, io, func, .{a_t}, .{});
    defer exe.deinit();

    var a_buf = try zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer a_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{a_buf});
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

// Shape transforms (E3, via the indexing-map / indexed-copy kernel).
fn transpose23(a: zml.Tensor) zml.Tensor {
    return a.transpose(.{ .j, .i }); // [2,3] -> [3,2]
}
fn bcast(a: zml.Tensor) zml.Tensor {
    return a.broad(zml.Shape.init(.{ .i = 4, .j = 3 }, .f32)); // [3] -> [4,3]
}
fn reshape23(a: zml.Tensor) zml.Tensor {
    return a.reshape(.{ .i = 2, .j = 3 }); // [6] -> [2,3]
}
fn sumj(a: zml.Tensor) zml.Tensor {
    return a.sum(.j); // [2,3] -> [2] (reduce over axis 1)
}
// Transpose / reshape INSIDE a fusion (E5.2): a data-movement op feeding an
// elementwise op must fuse into one kFusion (negate is exact, keeps the check
// tight). If XLA keeps them separate these hit Unimplemented and fail loudly.
fn transposeNeg(a: zml.Tensor) zml.Tensor {
    return a.transpose(.{ .j, .i }).negate(); // [2,3]->[3,2] then negate
}
fn reshapeNeg(a: zml.Tensor) zml.Tensor {
    return a.reshape(.{ .i = 3, .j = 2 }).negate(); // [2,3]->[3,2] (flat) then negate
}
fn sumAll(a: zml.Tensor) zml.Tensor {
    return a.sum(.n); // [n] -> scalar (num_out=1: the tree-reduction case)
}
// MAX reductions for the dtype-generic tree/two-level path: max is order-
// independent and its result is one of the inputs, so a bf16/f16 max is EXACT vs
// the oracle regardless of accumulation order — cleanly isolates the kernel's
// load/store-dtype boundary from any sum-accumulation-precision question.
fn maxj(a: zml.Tensor) zml.Tensor {
    return a.max(.j); // [i,j] -> [i] (tree when extent>=256, num_out>1)
}
fn maxAll(a: zml.Tensor) zml.Tensor {
    return a.max(.n); // [n] -> scalar (tree/two-level by extent)
}
// Reduce of a COMPUTED elementwise (RMSNorm's reduce(square(x)) shape): the
// reduce's input is x*x, not a parameter. If XLA fuses the multiply into the
// reduce, this is a reduction kFusion with a non-parameter reduce input — the
// fused-reduction kernel (inline elementwise prologue in the reduce loop).
fn sumSq(a: zml.Tensor) zml.Tensor {
    return a.mul(a).sum(.j); // [i,j] -> [i], out_i = Σ_j a[i,j]^2
}
// The SAME variance, but squared via power(x, 2) — how XLA/Llama RMSNorm actually
// lowers it (a kPower with a constant exponent feeding the reduce, not mul(x,x)).
// Exercises kPower in the fused-reduction prologue. Negative inputs verify x^2 is
// computed as repeated multiply (correct for x<0), not a pow/exp·log path (NaN).
fn sumPow2(a: zml.Tensor) zml.Tensor {
    const two = zml.Tensor.scalar(2.0, a.dtype()).broad(a.shape());
    return a.pow(two).sum(.j);
}
// Multi-axis reduce (one kReduce over several axes): zml.sum is single-axis, so
// drop to ops.reduce. Only contiguous reduced axes are supported (they merge to
// one extent/stride). sumJK reduces the trailing [j,k]; sumAll3 reduces all.
fn sumAxes(a: zml.Tensor, axes: []const i64) zml.Tensor {
    return zml.ops.reduce(.{a}, .{zml.Tensor.constant(a.dtype().zero())}, axes, struct {
        pub fn acc(args: zml.ops.ReduceArgs) struct { zml.Tensor } {
            return .{args.right.add(args.left.convert(args.right.dtype()))};
        }
    }.acc, .{})[0];
}
fn sumJK(a: zml.Tensor) zml.Tensor {
    return sumAxes(a, &.{ 1, 2 }); // [i,j,k] -> [i]
}
fn sumAll3(a: zml.Tensor) zml.Tensor {
    return sumAxes(a, &.{ 0, 1, 2 }); // [i,j,k] -> scalar
}
const iota24: [24]f32 = blk: {
    var a: [24]f32 = undefined;
    for (&a, 0..) |*e, i| e.* = @floatFromInt(i);
    break :blk a;
};
// Reduce-window via cumulativeSum (a stride-1, [N-1,0]-padded, single-axis
// window): 1D prefix sum, and 2D over the inner axis (exercises the kept-axis
// base offset).
fn cumsum1(a: zml.Tensor) zml.Tensor {
    return a.cumulativeSum(.n); // [n] -> [n]
}
fn cumsum2(a: zml.Tensor) zml.Tensor {
    return a.cumulativeSum(.j); // [i,j] -> [i,j], prefix sum along j
}

// Inputs large enough (extent >= 256) to exercise the threadgroup tree
// reduction rather than the serial-per-thread kernel.
const vec1024: [1024]f32 = blk: {
    @setEvalBranchQuota(4000);
    var a: [1024]f32 = undefined;
    for (&a, 0..) |*e, i| e.* = @floatFromInt(i % 13);
    break :blk a;
};
const mat3x512: [1536]f32 = blk: {
    @setEvalBranchQuota(4000);
    var a: [1536]f32 = undefined;
    for (&a, 0..) |*e, i| e.* = @floatFromInt((i % 11) + 1);
    break :blk a;
};

/// Run a unary shape-transform `func` (input `in_shape`) on `platform`,
/// writing the flattened result into `out`.
fn runUnaryShaped(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    in_shape: zml.Shape,
    in_data: []const f32,
    out: []f32,
) !void {
    const a_t: zml.Tensor = .fromShape(in_shape);
    var exe = try platform.compileFn(allocator, io, func, .{a_t}, .{});
    defer exe.deinit();

    var a_buf = try zml.Buffer.fromBytes(io, platform, in_shape, .replicated, std.mem.sliceAsBytes(in_data));
    defer a_buf.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{a_buf});
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

// Typed (f16/bf16/f32) 1-input runner — for dtype tests of unary ops (transpose,
// reductions, …). Output converted to f32 by the caller (checkUnaryT) to compare.
fn runUnaryT(
    comptime T: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    in_shape: zml.Shape,
    in_data: []const T,
    out: []T,
) !void {
    const a_t: zml.Tensor = .fromShape(in_shape);
    var exe = try platform.compileFn(allocator, io, func, .{a_t}, .{});
    defer exe.deinit();
    var a_buf = try zml.Buffer.fromBytes(io, platform, in_shape, .replicated, std.mem.sliceAsBytes(in_data));
    defer a_buf.deinit();
    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{a_buf});
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);
    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(T));
}

fn checkUnaryT(
    comptime T: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    in_shape: zml.Shape,
    in_data: []const T,
    n_out: usize,
    rel_tol: f32,
) usize {
    var co: [64]T = undefined;
    var mo: [64]T = undefined;
    runUnaryT(T, allocator, io, cpu, func, in_shape, in_data, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runUnaryT(T, allocator, io, metal, func, in_shape, in_data, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    var cof: [64]f32 = undefined;
    var mof: [64]f32 = undefined;
    for (0..n_out) |i| {
        cof[i] = toF32(T, co[i]);
        mof[i] = toF32(T, mo[i]);
    }
    return compare(name, cof[0..n_out], mof[0..n_out], rel_tol);
}

fn checkShaped(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    in_shape: zml.Shape,
    in_data: []const f32,
    n_out: usize,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    runUnaryShaped(allocator, io, cpu, func, in_shape, in_data, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runUnaryShaped(allocator, io, metal, func, in_shape, in_data, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n_out], mo[0..n_out], 1e-6);
}

/// Pass if every lane is within abs floor `abs_tol` OR relative tolerance.
/// `abs_tol` is also the small-value regularizer in the relative metric, so it
/// must reflect the OUTPUT dtype's representable precision (~1 ULP at the data's
/// magnitude). The f32-scale 1e-4 default is right for f32/f16 outputs ~O(1);
/// a composed bf16 graph with cancellation produces a worst error of ~1 bf16 ULP
/// (2^-7 ≈ 7.8e-3 at O(1)) that can land on a near-zero post-cancellation output
/// — a ~10% "miss" under a pure relative metric that NO bf16 kernel can avoid,
/// so such tests pass a bf16-scale floor (≈2 ULP).
fn compareAbs(name: []const u8, cpu_out: []const f32, metal_out: []const f32, rel_tol: f32, abs_tol: f32) usize {
    var max_abs: f32 = 0;
    var worst_rel: f32 = 0;
    for (cpu_out, metal_out) |c, m| {
        const e = @abs(m - c);
        if (e > max_abs) max_abs = e;
        const rel = e / (@abs(c) + abs_tol);
        if (rel > worst_rel) worst_rel = rel;
    }
    if (max_abs <= abs_tol or worst_rel <= rel_tol) {
        log.info("{s:>6}  OK     max_abs={e} max_rel={e}", .{ name, max_abs, worst_rel });
        return 0;
    }
    log.err("{s:>6}  FAIL   max_abs={e} max_rel={e}", .{ name, max_abs, worst_rel });
    log.err("        cpu  ={any}", .{cpu_out});
    log.err("        metal={any}", .{metal_out});
    return 1;
}
/// Default abs floor (1e-4), appropriate for f32/f16 outputs ~O(1).
fn compare(name: []const u8, cpu_out: []const f32, metal_out: []const f32, rel_tol: f32) usize {
    return compareAbs(name, cpu_out, metal_out, rel_tol, 1e-4);
}
/// ~2 ULP at O(1) for the output dtype — the absolute precision floor of a
/// composed graph, used where cancellation makes a pure relative metric
/// meaningless near zero (see [[compareAbs]]).
fn absFloorFor(comptime T: type) f32 {
    return switch (T) {
        f32 => 1e-4,
        f16 => 2e-3, // 2 * 2^-10 at O(1)
        else => 1.6e-2, // bf16: 2 * 2^-7 at O(1)
    };
}

fn checkBinary(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a: []const f32,
    b: []const f32,
    rel_tol: f32,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    const n = a.len;
    runBinaryOn(allocator, io, cpu, func, a, b, co[0..n]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runBinaryOn(allocator, io, metal, func, a, b, mo[0..n]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n], mo[0..n], rel_tol);
}

fn checkTernary(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a: []const f32,
    b: []const f32,
    c: []const f32,
    rel_tol: f32,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    const n = a.len;
    runTernaryOn(allocator, io, cpu, func, a, b, c, co[0..n]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runTernaryOn(allocator, io, metal, func, a, b, c, mo[0..n]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n], mo[0..n], rel_tol);
}

/// Run a `func(x, scale, bias)` where x has `x_shape` and scale/bias share the
/// smaller `d_shape` (broadcast inside the fusion). Writes the flat result.
fn runAffineBcast(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    x_shape: zml.Shape,
    x_data: []const f32,
    d_shape: zml.Shape,
    scale_data: []const f32,
    bias_data: []const f32,
    out: []f32,
) !void {
    const xt: zml.Tensor = .fromShape(x_shape);
    const st: zml.Tensor = .fromShape(d_shape);
    const bt: zml.Tensor = .fromShape(d_shape);
    var exe = try platform.compileFn(allocator, io, func, .{ xt, st, bt }, .{});
    defer exe.deinit();

    var xb = try zml.Buffer.fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(x_data));
    defer xb.deinit();
    var sb = try zml.Buffer.fromBytes(io, platform, d_shape, .replicated, std.mem.sliceAsBytes(scale_data));
    defer sb.deinit();
    var bb = try zml.Buffer.fromBytes(io, platform, d_shape, .replicated, std.mem.sliceAsBytes(bias_data));
    defer bb.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ xb, sb, bb });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn checkAffineBcast(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    x_shape: zml.Shape,
    x_data: []const f32,
    d_shape: zml.Shape,
    scale_data: []const f32,
    bias_data: []const f32,
    n_out: usize,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    runAffineBcast(allocator, io, cpu, func, x_shape, x_data, d_shape, scale_data, bias_data, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runAffineBcast(allocator, io, metal, func, x_shape, x_data, d_shape, scale_data, bias_data, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n_out], mo[0..n_out], 1e-6);
}

// Two differently-shaped inputs (a:[M,K], b:[K,N] or [N,K]) → out:[M,N].
fn runMatmul(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_shape: zml.Shape,
    a_data: []const f32,
    b_shape: zml.Shape,
    b_data: []const f32,
    out: []f32,
) !void {
    const at: zml.Tensor = .fromShape(a_shape);
    const bt: zml.Tensor = .fromShape(b_shape);
    var exe = try platform.compileFn(allocator, io, func, .{ at, bt }, .{});
    defer exe.deinit();

    var ab = try zml.Buffer.fromBytes(io, platform, a_shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer ab.deinit();
    var bb = try zml.Buffer.fromBytes(io, platform, b_shape, .replicated, std.mem.sliceAsBytes(b_data));
    defer bb.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ ab, bb });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn checkMatmul(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a_shape: zml.Shape,
    a_data: []const f32,
    b_shape: zml.Shape,
    b_data: []const f32,
    n_out: usize,
    rel_tol: f32,
) usize {
    var co: [256]f32 = undefined;
    var mo: [256]f32 = undefined;
    runMatmul(allocator, io, cpu, func, a_shape, a_data, b_shape, b_data, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runMatmul(allocator, io, metal, func, a_shape, a_data, b_shape, b_data, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n_out], mo[0..n_out], rel_tol);
}

// Three differently-shaped inputs (a:[M,K], b:[K,N], bias:[M,N]) → out:[M,N],
// for the multi-op linearBias graph (matmul thunk → add kernel).
fn run3Shaped(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_shape: zml.Shape,
    a_data: []const f32,
    b_shape: zml.Shape,
    b_data: []const f32,
    c_shape: zml.Shape,
    c_data: []const f32,
    out: []f32,
) !void {
    const at: zml.Tensor = .fromShape(a_shape);
    const bt: zml.Tensor = .fromShape(b_shape);
    const ct: zml.Tensor = .fromShape(c_shape);
    var exe = try platform.compileFn(allocator, io, func, .{ at, bt, ct }, .{});
    defer exe.deinit();

    var ab = try zml.Buffer.fromBytes(io, platform, a_shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer ab.deinit();
    var bb = try zml.Buffer.fromBytes(io, platform, b_shape, .replicated, std.mem.sliceAsBytes(b_data));
    defer bb.deinit();
    var cb = try zml.Buffer.fromBytes(io, platform, c_shape, .replicated, std.mem.sliceAsBytes(c_data));
    defer cb.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ ab, bb, cb });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn check3Shaped(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a_shape: zml.Shape,
    a_data: []const f32,
    b_shape: zml.Shape,
    b_data: []const f32,
    c_shape: zml.Shape,
    c_data: []const f32,
    n_out: usize,
    rel_tol: f32,
) usize {
    var co: [256]f32 = undefined;
    var mo: [256]f32 = undefined;
    run3Shaped(allocator, io, cpu, func, a_shape, a_data, b_shape, b_data, c_shape, c_data, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    run3Shaped(allocator, io, metal, func, a_shape, a_data, b_shape, b_data, c_shape, c_data, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n_out], mo[0..n_out], rel_tol);
}

// Four differently-shaped inputs → out, for the deep MLP pipelining stress test.
fn run4Shaped(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    shapes: [4]zml.Shape,
    data: [4][]const f32,
    out: []f32,
) !void {
    const t0: zml.Tensor = .fromShape(shapes[0]);
    const t1: zml.Tensor = .fromShape(shapes[1]);
    const t2: zml.Tensor = .fromShape(shapes[2]);
    const t3: zml.Tensor = .fromShape(shapes[3]);
    var exe = try platform.compileFn(allocator, io, func, .{ t0, t1, t2, t3 }, .{});
    defer exe.deinit();

    var b0 = try zml.Buffer.fromBytes(io, platform, shapes[0], .replicated, std.mem.sliceAsBytes(data[0]));
    defer b0.deinit();
    var b1 = try zml.Buffer.fromBytes(io, platform, shapes[1], .replicated, std.mem.sliceAsBytes(data[1]));
    defer b1.deinit();
    var b2 = try zml.Buffer.fromBytes(io, platform, shapes[2], .replicated, std.mem.sliceAsBytes(data[2]));
    defer b2.deinit();
    var b3 = try zml.Buffer.fromBytes(io, platform, shapes[3], .replicated, std.mem.sliceAsBytes(data[3]));
    defer b3.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ b0, b1, b2, b3 });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);
    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(f32));
}

fn check4Shaped(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    shapes: [4]zml.Shape,
    data: [4][]const f32,
    n_out: usize,
    rel_tol: f32,
) usize {
    var co: [256]f32 = undefined;
    var mo: [256]f32 = undefined;
    run4Shaped(allocator, io, cpu, func, shapes, data, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    run4Shaped(allocator, io, metal, func, shapes, data, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n_out], mo[0..n_out], rel_tol);
}

// dtype-generic matmul (f16 = native f16, bf16 = zml.floats.BFloat16). Same as
// runMatmul but typed; the output is converted to f32 by the caller for compare.
fn runMatmulT(
    comptime T: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    comptime func: anytype,
    a_shape: zml.Shape,
    a_data: []const T,
    b_shape: zml.Shape,
    b_data: []const T,
    out: []T,
) !void {
    const at: zml.Tensor = .fromShape(a_shape);
    const bt: zml.Tensor = .fromShape(b_shape);
    var exe = try platform.compileFn(allocator, io, func, .{ at, bt }, .{});
    defer exe.deinit();

    var ab = try zml.Buffer.fromBytes(io, platform, a_shape, .replicated, std.mem.sliceAsBytes(a_data));
    defer ab.deinit();
    var bb = try zml.Buffer.fromBytes(io, platform, b_shape, .replicated, std.mem.sliceAsBytes(b_data));
    defer bb.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ ab, bb });
    exe.call(exe_args, &exe_results);
    var result = exe_results.get(zml.Buffer);
    defer result.deinit();
    _ = try result.await(io);

    const slice = try result.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    @memcpy(out, slice.items(T));
}

fn toF32(comptime T: type, v: T) f32 {
    return if (T == f32) v else if (T == f16) @floatCast(v) else v.toF32();
}

// Build a comptime bf16 array from f32 literals (for inline typed test data).
fn bf16a(comptime vals: anytype) [vals.len]zml.floats.BFloat16 {
    var out: [vals.len]zml.floats.BFloat16 = undefined;
    inline for (vals, 0..) |v, i| out[i] = zml.floats.BFloat16.fromF32(v);
    return out;
}

fn checkMatmulT(
    comptime T: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a_shape: zml.Shape,
    a_data: []const T,
    b_shape: zml.Shape,
    b_data: []const T,
    n_out: usize,
    rel_tol: f32,
) usize {
    var co: [16]T = undefined;
    var mo: [16]T = undefined;
    runMatmulT(T, allocator, io, cpu, func, a_shape, a_data, b_shape, b_data, co[0..n_out]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runMatmulT(T, allocator, io, metal, func, a_shape, a_data, b_shape, b_data, mo[0..n_out]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    var cof: [16]f32 = undefined;
    var mof: [16]f32 = undefined;
    for (0..n_out) |i| {
        cof[i] = toF32(T, co[i]);
        mof[i] = toF32(T, mo[i]);
    }
    return compare(name, cof[0..n_out], mof[0..n_out], rel_tol);
}

fn dtOf(comptime T: type) zml.DataType {
    return if (T == f32) .f32 else if (T == f16) .f16 else .bf16;
}

fn fillFrac(comptime T: type, s: []T, seed: usize) void {
    for (s, 0..) |*e, i| {
        const x: f32 = 0.5 - @as(f32, @floatFromInt((i * seed + 7) % 97)) / 97.0;
        e.* = if (T == f32) x else if (T == f16) @floatCast(x) else zml.floats.BFloat16.fromF32(x);
    }
}

// Heap-backed, dtype+layout-generic matmul check for medium shapes (the [16]
// buffers of checkMatmulT can't hold them). transpose_b picks NN (b={k,n}) vs
// NT (b={n,k}, the y=x·Wᵀ linear). Fractional values stress tiling/accumulation.
fn checkMM(
    comptime T: type,
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    M: i64,
    K: i64,
    N: i64,
    transpose_b: bool,
    rel_tol: f32,
) usize {
    const dt = dtOf(T);
    const a_shape = zml.Shape.init(.{ .m = M, .k = K }, dt);
    const b_shape = if (transpose_b)
        zml.Shape.init(.{ .n = N, .k = K }, dt)
    else
        zml.Shape.init(.{ .k = K, .n = N }, dt);
    const mk: usize = @intCast(M * K);
    const kn: usize = @intCast(K * N);
    const mn: usize = @intCast(M * N);
    const a = allocator.alloc(T, mk) catch return 1;
    defer allocator.free(a);
    const b = allocator.alloc(T, kn) catch return 1;
    defer allocator.free(b);
    fillFrac(T, a, 31);
    fillFrac(T, b, 17);
    const co = allocator.alloc(T, mn) catch return 1;
    defer allocator.free(co);
    const mo = allocator.alloc(T, mn) catch return 1;
    defer allocator.free(mo);
    runMatmulT(T, allocator, io, cpu, matmul, a_shape, a, b_shape, b, co) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runMatmulT(T, allocator, io, metal, matmul, a_shape, a, b_shape, b, mo) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    const cof = allocator.alloc(f32, mn) catch return 1;
    defer allocator.free(cof);
    const mof = allocator.alloc(f32, mn) catch return 1;
    defer allocator.free(mof);
    for (0..mn) |i| {
        cof[i] = toF32(T, co[i]);
        mof[i] = toF32(T, mo[i]);
    }
    return compare(name, cof, mof, rel_tol);
}

fn checkUnary(
    allocator: std.mem.Allocator,
    io: std.Io,
    cpu: *zml.Platform,
    metal: *zml.Platform,
    name: []const u8,
    comptime func: anytype,
    a: []const f32,
    rel_tol: f32,
) usize {
    var co: [64]f32 = undefined;
    var mo: [64]f32 = undefined;
    const n = a.len;
    runUnaryOn(allocator, io, cpu, func, a, co[0..n]) catch |e| {
        log.err("{s:>6}  CPU failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    runUnaryOn(allocator, io, metal, func, a, mo[0..n]) catch |e| {
        log.err("{s:>6}  Metal failed: {s}", .{ name, @errorName(e) });
        return 1;
    };
    return compare(name, co[0..n], mo[0..n], rel_tol);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const cpu: *zml.Platform = try .init(allocator, io, .cpu, .{});
    defer cpu.deinit(allocator, io);
    const metal: *zml.Platform = try .init(allocator, io, .metal, .{});
    defer metal.deinit(allocator, io);
    log.info("CPU oracle vs Metal: {f}", .{metal.fmtVerbose()});

    // Binary inputs: negatives + positives; no zeros in b (for div).
    const a_bin = [_]f32{ 1, -2, 3, -4, 5, -6, 7, -8 };
    const b_bin = [_]f32{ 10, 20, -30, 40, -50, 60, 70, 80 };
    // Unary inputs: strictly positive so log/sqrt/rsqrt are valid.
    const a_un = [_]f32{ 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };

    const exact = 1e-6; // a single fast-math FP op is bit-identical to plain
    const transc = 1e-2; // fast-math exp/log/sqrt/rsqrt/tanh approximations

    var failures: usize = 0;
    inline for (.{
        .{ "add", add, exact }, .{ "sub", sub, exact },
        .{ "mul", mul, exact }, .{ "div", div, 1e-3 },
        .{ "max", maximum, exact }, .{ "min", minimum, exact },
    }) |c| {
        failures += checkBinary(allocator, io, cpu, metal, c[0], c[1], &a_bin, &b_bin, c[2]);
    }

    // Multi-op fusion (E5): (a+b)*c must compile into ONE kFusion kernel.
    {
        const c_bin = [_]f32{ 2, 2, 2, 2, 2, 2, 2, 2 };
        failures += checkTernary(allocator, io, cpu, metal, "fma", fma3, &a_bin, &b_bin, &c_bin, exact);
    }

    // Broadcast inside a fusion (E5.2): x*scale + bias, scale/bias [d]->[2,3].
    // = [[11,202,3003],[41,502,6003]]. A kFusion with two broadcasts inside.
    failures += checkAffineBcast(allocator, io, cpu, metal, "affine", affineBcast,
        zml.Shape.init(.{ .b = 2, .d = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 },
        zml.Shape.init(.{ .d = 3 }, .f32), &[_]f32{ 10, 100, 1000 }, &[_]f32{ 1, 2, 3 }, 6);
    inline for (.{
        .{ "neg", negate, exact }, .{ "abs", abs, exact },
        .{ "exp", exp, transc }, .{ "log", log_, transc },
        .{ "sqrt", sqrt, transc }, .{ "rsqrt", rsqrt, transc },
        .{ "tanh", tanh, transc }, .{ "sin", sin_, transc },
        .{ "cos", cos_, transc }, .{ "silu", silu_, transc },
    }) |c| {
        failures += checkUnary(allocator, io, cpu, metal, c[0], c[1], &a_un, c[2]);
    }

    // Shape transforms (E3): pure data movement, exact vs CPU.
    failures += checkShaped(allocator, io, cpu, metal, "transp", transpose23,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);
    failures += checkShaped(allocator, io, cpu, metal, "bcast", bcast,
        zml.Shape.init(.{ .j = 3 }, .f32), &[_]f32{ 10, 20, 30 }, 12);
    failures += checkShaped(allocator, io, cpu, metal, "reshape", reshape23,
        zml.Shape.init(.{ .n = 6 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);

    // Transpose / reshape inside a fusion (E5.2 finish): negate∘transpose and
    // negate∘reshape, [2,3]->[3,2], must each fuse into one kFusion.
    failures += checkShaped(allocator, io, cpu, metal, "tneg", transposeNeg,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);
    failures += checkShaped(allocator, io, cpu, metal, "rneg", reshapeNeg,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);

    // Reduction (E4): sum over axis 1, [2,3] -> [2] = [3, 12].
    failures += checkShaped(allocator, io, cpu, metal, "sum", sumj,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 2);

    // Reduce of a computed elementwise (RMSNorm reduce(square(x)) shape):
    // [2,3] -> [2], out_i = Σ_j a[i,j]^2 = [0+1+4, 9+16+25] = [5, 50].
    failures += checkShaped(allocator, io, cpu, metal, "sumsq", sumSq,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 2);
    // Variance via power(x,2) (the real Llama RMSNorm lowering, kPower). NEGATIVE
    // inputs: out_i = Σ_j x^2 = [4+1+0, 1+4+9] = [5, 14] (proves x<0 → x², not NaN).
    failures += checkShaped(allocator, io, cpu, metal, "sumpow2", sumPow2,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ -2, -1, 0, 1, 2, 3 }, 2);
    // Same fused reduce at a larger reduced extent (512): num_out=3, exercises
    // the serial-per-output fused kernel over a long reduce loop. Values 1..11
    // → Σ of ≤512 squares ≤ 61952, exact in f32.
    failures += checkShaped(allocator, io, cpu, metal, "sumsqB", sumSq,
        zml.Shape.init(.{ .i = 3, .j = 512 }, .f32), &mat3x512, 3);

    // Multi-axis reduce (contiguous): [2,3,4] reduce {j,k} -> [2] = [66, 210];
    // reduce-all {i,j,k} -> [] = 276. Both merge to one extent/stride.
    failures += checkShaped(allocator, io, cpu, metal, "sumjk", sumJK,
        zml.Shape.init(.{ .i = 2, .j = 3, .k = 4 }, .f32), &iota24, 2);
    failures += checkShaped(allocator, io, cpu, metal, "sumall3", sumAll3,
        zml.Shape.init(.{ .i = 2, .j = 3, .k = 4 }, .f32), &iota24, 1);

    // Reduce-window (E4+): cumulative sum. 1D [6]->[0,1,3,6,10,15]; 2D prefix
    // sum along j (the windowed axis), i kept (tests the base offset).
    failures += checkShaped(allocator, io, cpu, metal, "cumsum", cumsum1,
        zml.Shape.init(.{ .n = 6 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);
    failures += checkShaped(allocator, io, cpu, metal, "cumsum2", cumsum2,
        zml.Shape.init(.{ .i = 2, .j = 4 }, .f32), &[_]f32{ 0, 1, 2, 3, 10, 20, 30, 40 }, 8);
    // Reduce-to-scalar via the TREE kernel (E4 harden): [1024] -> [], num_out=1,
    // extent=1024 (>=256) so this is one threadgroup cooperatively reducing
    // (base=0, exercises the grid-stride accumulate + tree). Serial-per-thread
    // would run it on a single thread.
    failures += checkShaped(allocator, io, cpu, metal, "sumall", sumAll,
        zml.Shape.init(.{ .n = 1024 }, .f32), &vec1024, 1);
    // Tree reduction with num_out>1: [3,512] -> [3]. base != 0, so this also
    // exercises the tree kernel's output-index delinearization.
    failures += checkShaped(allocator, io, cpu, metal, "sumtree", sumj,
        zml.Shape.init(.{ .i = 3, .j = 512 }, .f32), &mat3x512, 3);

    // Two-level reduction (extent >= 65536, tiny num_out): pass 1 runs many
    // threadgroups per output into partials, pass 2 reduces them. sumbig is
    // reduce-to-scalar (base=0); sum2dbig has num_out=4 (base!=0, exercises the
    // pass-1 base delinearization). Runtime-allocated (too big for comptime).
    {
        const big = try allocator.alloc(f32, 65536);
        defer allocator.free(big);
        for (big, 0..) |*e, i| e.* = @floatFromInt(i % 13);
        failures += checkShaped(allocator, io, cpu, metal, "sumbig", sumAll,
            zml.Shape.init(.{ .n = 65536 }, .f32), big, 1);

        const big2d = try allocator.alloc(f32, 4 * 65536);
        defer allocator.free(big2d);
        for (big2d, 0..) |*e, i| e.* = @floatFromInt((i % 13) + 1);
        failures += checkShaped(allocator, io, cpu, metal, "sum2dbig", sumj,
            zml.Shape.init(.{ .i = 4, .j = 65536 }, .f32), big2d, 4);
    }

    // The cooperative tree + two-level kernels now carry f16/bf16 (load→f32
    // accumulate→store; two-level keeps f32 partials between passes). MAX
    // reductions → exact vs the oracle (result is an input value), isolating the
    // dtype boundary. bf16max: [3,512]→[3] tree (extent 512≥256, num_out 3).
    // bf16maxbig: [65536]→[1] two-level (extent≥65536). Distinct bf16 values via
    // a stride so the max is unambiguous; n_out≤64.
    {
        const T = zml.floats.BFloat16;
        var m3x512: [1536]T = undefined;
        for (&m3x512, 0..) |*e, i| e.* = T.fromF32(@as(f32, @floatFromInt((i * 7) % 251)) / 32.0);
        failures += checkUnaryT(T, allocator, io, cpu, metal, "bf16max", maxj,
            zml.Shape.init(.{ .i = 3, .j = 512 }, .bf16), &m3x512, 3, 1e-2);

        const big = try allocator.alloc(T, 65536);
        defer allocator.free(big);
        for (big, 0..) |*e, i| e.* = T.fromF32(@as(f32, @floatFromInt((i * 7) % 251)) / 32.0);
        failures += checkUnaryT(T, allocator, io, cpu, metal, "bf16maxbig", maxAll,
            zml.Shape.init(.{ .n = 65536 }, .bf16), big, 1, 1e-2);
    }

    // Matmul. Backend mirrors XLA_METAL_MATMUL: "" (default = MPSGraph, modern,
    // f32/f16/bf16) | "naive" (f32) | "metalblas" | "mpsmatrix" (legacy MPS,
    // f32/f16, aborts on bf16). Tiny NN/NT (N<32) route, under metalBLAS, to its
    // m5/gemv kernels (not wired into the plugin yet → loud Unimplemented), so
    // skip them there; mmMed (96x48, K=64) is m5_tensor-eligible and runs on
    // every backend.
    const mm_backend: []const u8 = blk: {
        const v = std.c.getenv("XLA_METAL_MATMUL") orelse break :blk "";
        break :blk std.mem.span(v);
    };
    const mm_is_metalblas = std.mem.eql(u8, mm_backend, "metalblas");
    const mm_is_naive = std.mem.eql(u8, mm_backend, "naive");
    const mm_is_mpsmatrix = std.mem.eql(u8, mm_backend, "mpsmatrix");  // legacy
    // Default path (empty/unknown) = modern MPSGraph: f32/f16/bf16, any transpose.
    const mm_is_mpsgraph = !mm_is_metalblas and !mm_is_naive and !mm_is_mpsmatrix;

    // NN: [2,3]·[3,2] = [[22,28],[49,64]]. NT (the y = x·Wᵀ linear, rhs contracts
    // its inner dim): [2,3]·[2,3]ᵀ with W=[[1,0,1],[0,1,0]] = [[4,2],[10,5]].
    // Small integers → exact in f32.
    if (!mm_is_metalblas) {
        failures += checkMatmul(allocator, io, cpu, metal, "mmNN", matmul,
            zml.Shape.init(.{ .m = 2, .k = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 },
            zml.Shape.init(.{ .k = 3, .n = 2 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 }, 4, exact);
        failures += checkMatmul(allocator, io, cpu, metal, "mmNT", matmul,
            zml.Shape.init(.{ .m = 2, .k = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 },
            zml.Shape.init(.{ .n = 2, .k = 3 }, .f32), &[_]f32{ 1, 0, 1, 0, 1, 0 }, 4, exact);
    }
    // Medium NN ([96,64]·[64,48]) and NT (the y=x·Wᵀ linear: [96,64]·[48,64]ᵀ),
    // fractional values. NN runs on every backend (metalBLAS → m5_tensor); NT
    // under metalBLAS would route to m5_gemm, which isn't wired yet (it produced
    // wrong output — simdgroup-builtin issue), so NT is MPS/naive only.
    failures += checkMM(f32, allocator, io, cpu, metal, "mmMed", 96, 64, 48, false, 1e-4);
    if (!mm_is_metalblas)
        failures += checkMM(f32, allocator, io, cpu, metal, "mmNTmed", 96, 64, 48, true, 1e-4);

    // f16 — the MPS family (MPSGraph default + legacy mpsmatrix); naive is
    // f32-only, metalBLAS's tiny path isn't wired. MPSGraph also does f16 NT.
    if (mm_is_mpsgraph or mm_is_mpsmatrix) {
        failures += checkMatmulT(f16, allocator, io, cpu, metal, "mmF16", matmul, zml.Shape.init(.{ .m = 2, .k = 3 }, .f16), &[_]f16{ 1, 2, 3, 4, 5, 6 }, zml.Shape.init(.{ .k = 3, .n = 2 }, .f16), &[_]f16{ 1, 2, 3, 4, 5, 6 }, 4, 1e-2);
        failures += checkMM(f16, allocator, io, cpu, metal, "mmF16med", 96, 64, 48, false, 2e-2);
    }

    // bf16 — MPSGraph (default) does it natively; this is the headline win over
    // the legacy MPSMatrix path (which ABORTS on bf16). metalBLAS does NN via
    // m5_tensor. naive/mpsmatrix can't. MPSGraph also handles bf16 NT (the
    // y=x·Wᵀ linear), which metalBLAS can't yet (NT→m5_gemm, not wired).
    if (mm_is_mpsgraph or mm_is_metalblas) {
        failures += checkMM(zml.floats.BFloat16, allocator, io, cpu, metal, "bf16NN", 96, 64, 48, false, 3e-2);
    }
    if (mm_is_mpsgraph)
        failures += checkMM(zml.floats.BFloat16, allocator, io, cpu, metal, "bf16NT", 96, 64, 48, true, 3e-2);

    // GEMV-shaped NN matmuls route to the dedicated metalBLAS GEMV kernels via the
    // decide()-driven path: M==1 → gemv_t (x·W, every dtype), thin M (2..16) low-
    // precision → gemv_bt (one B stream feeds all rows). N>=16 & K>=64 clear the
    // GEMV regime gates. metalBLAS only — MPSGraph handles these shapes its own way.
    if (mm_is_metalblas) {
        failures += checkMM(f32, allocator, io, cpu, metal, "gvT32", 1, 256, 64, false, 1e-4);
        failures += checkMM(zml.floats.BFloat16, allocator, io, cpu, metal, "gvTbf", 1, 256, 512, false, 3e-2);
        failures += checkMM(f16, allocator, io, cpu, metal, "gvT16", 1, 256, 512, false, 2e-2);
        failures += checkMM(zml.floats.BFloat16, allocator, io, cpu, metal, "gvBt4", 4, 128, 512, false, 3e-2);
        failures += checkMM(zml.floats.BFloat16, allocator, io, cpu, metal, "gvBt16", 16, 128, 512, false, 3e-2);
        failures += checkMM(f16, allocator, io, cpu, metal, "gvBtF16", 8, 256, 256, false, 2e-2);
        // gemv_nt: x · Wᵀ — the STANDARD decode/lm_head linear (weight [out,in]=[N,K],
        // so trans_b), M==1. This is the path real decode projections take.
        failures += checkMM(f32, allocator, io, cpu, metal, "gvNT32", 1, 256, 512, true, 1e-4);
        failures += checkMM(zml.floats.BFloat16, allocator, io, cpu, metal, "gvNTbf", 1, 256, 512, true, 3e-2);
        failures += checkMM(f16, allocator, io, cpu, metal, "gvNT16", 1, 256, 512, true, 2e-2);
    }

    // Whole-graph execution (first multi-op module): y = x·W + bias. The add
    // consumes the dot's result, so this routes through the thunk-sequence graph
    // executable — MPSGraph matmul into an intermediate buffer, then an
    // elementwise-add kernel. The dot always uses MPSGraph here (graph path is
    // backend-independent), so run it on every XLA_METAL_MATMUL setting.
    // [2,3]·[3,2]=[[22,28],[49,64]], + [[100,200],[300,400]] = [[122,228],[349,464]].
    failures += check3Shaped(allocator, io, cpu, metal, "linbias", linearBias,
        zml.Shape.init(.{ .m = 2, .k = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 },
        zml.Shape.init(.{ .k = 3, .n = 2 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 },
        zml.Shape.init(.{ .m = 2, .n = 2 }, .f32), &[_]f32{ 100, 200, 300, 400 }, 4, exact);

    // Batched matmul (multi-head attention's QKᵀ / AV primitive): a{h,m,k}·b{h,k,n}
    // → {h,m,n}, batch dim .h. Distinct per-head data so a bug that collapses or
    // mixes batch slices is caught vs the oracle. Graph path → MPSGraph batched
    // matmul (backend-independent). Two heads of [2,3]·[3,2]; small ints → exact.
    failures += checkMatmul(allocator, io, cpu, metal, "bmm", bmm,
        zml.Shape.init(.{ .h = 2, .m = 2, .k = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
        zml.Shape.init(.{ .h = 2, .k = 3, .n = 2 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1 }, 8, exact);

    // GQA attention QKᵀ — the GENERAL batched dot (Llama-3.2 layout): batch .hkv at
    // a NON-leading operand position + two lhs free dims (.s, .hg).
    // q{s,hkv,hg,hd}·k{t,hkv,hd} contract .hd → {hkv,s,hg,t}. 2·2·2·2=16 out, K=2
    // (exact). Verifies the in-graph permute→[batch,M,K]/[batch,K,N] matmul.
    failures += checkMatmul(allocator, io, cpu, metal, "gqa", gqa,
        zml.Shape.init(.{ .s = 2, .hkv = 2, .hg = 2, .hd = 2 }, .f32),
        &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 },
        zml.Shape.init(.{ .t = 2, .hkv = 2, .hd = 2 }, .f32),
        &[_]f32{ 1, 0, 0, 1, 2, 1, 1, 2 }, 16, exact);

    // Multi-head attention core: per-head q·kᵀ → softmax(.t) → ·v. Two batched
    // matmuls + a batched softmax (reduce over .t of a {h,s,t} tensor) — the real
    // MHA shape. h=2 heads, s=2 queries, t=3 keys, e=2 head dim. Distinct heads.
    failures += check3Shaped(allocator, io, cpu, metal, "mha", mha,
        zml.Shape.init(.{ .h = 2, .s = 2, .e = 2 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 },
        zml.Shape.init(.{ .h = 2, .t = 3, .e = 2 }, .f32), &[_]f32{ 1, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 0 },
        zml.Shape.init(.{ .h = 2, .t = 3, .e = 2 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1 }, 8, 1e-5);

    // DECODE-shape MHA: a SINGLE query (s=1) against t=3 keys/values, batched over
    // h=2 heads. M=1 makes XLA's DotStrengthReduction rewrite the batched q·kᵀ and
    // attn·v into broadcast-multiply-reduce (the path the full-Llama decode loop
    // takes; distinct from the s>1 prefill batched dot above). Reproduces the
    // decode attention math in isolation.
    failures += check3Shaped(allocator, io, cpu, metal, "mhaDec", mha,
        zml.Shape.init(.{ .h = 2, .s = 1, .e = 2 }, .f32), &[_]f32{ 1, 2, 3, 4 },
        zml.Shape.init(.{ .h = 2, .t = 3, .e = 2 }, .f32), &[_]f32{ 1, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 0 },
        zml.Shape.init(.{ .h = 2, .t = 3, .e = 2 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1 }, 4, 1e-5);

    // Causal mask FUSED from iota (s32 iota×2 → integer cmp → select → scalar):
    // lower triangle keeps x, upper → −1e9. Exercises an integer iota + integer
    // compare inside a fusion (the on-device masking shape). Exact (select picks).
    failures += checkShaped(allocator, io, cpu, metal, "causal", causalMask,
        zml.Shape.init(.{ .r = 3, .c = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, 9);
    // convert (s32→f32) inside a fusion: iota positions cast to f32, added to x
    // (the RoPE-positions bridge). out[r,c] = c + x[r,c]. Integer→f32 exact.
    failures += checkShaped(allocator, io, cpu, metal, "cvtpos", posBias,
        zml.Shape.init(.{ .r = 2, .c = 4 }, .f32), &[_]f32{ 10, 20, 30, 40, 50, 60, 70, 80 }, 8);
    // RoPE end to end (.s=4 positions, .hd=8 head dim → 32 outputs). The full
    // chain: arange→outer→sin/cos→scale→convert→broadcast→SLICE(split)→
    // mul/sub,mul/add→CONCAT(merge). sin/cos are fast-math → transc tolerance.
    failures += checkUnaryT(f32, allocator, io, cpu, metal, "rope", ropeFwd,
        zml.Shape.init(.{ .s = 4, .hd = 8 }, .f32),
        &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2 },
        32, transc);
    failures += checkUnaryT(f32, allocator, io, cpu, metal, "ropePos", ropePosFwd,
        zml.Shape.init(.{ .s = 4, .hd = 8 }, .f32),
        &[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2 },
        32, transc);
    // f16 / bf16 STORAGE through the fusion path: a*b + a (load half/bfloat →
    // f32 compute → per-op round → store). Fractional data so the interior a*b
    // rounding is exercised. The graph is no longer f32-only. [[C12]]
    failures += checkMatmulT(f16, allocator, io, cpu, metal, "f16ew", mulAdd,
        zml.Shape.init(.{ .n = 4 }, .f16), &[_]f16{ 0.1, 0.7, 1.3, 2.1 },
        zml.Shape.init(.{ .n = 4 }, .f16), &[_]f16{ 3.0, 1.1, 0.6, 0.9 }, 4, 1e-2);
    const bf_a = bf16a(.{ 0.1, 0.7, 1.3, 2.1 });
    const bf_b = bf16a(.{ 3.0, 1.1, 0.6, 0.9 });
    failures += checkMatmulT(zml.floats.BFloat16, allocator, io, cpu, metal, "bf16ew", mulAdd,
        zml.Shape.init(.{ .n = 4 }, .bf16), &bf_a,
        zml.Shape.init(.{ .n = 4 }, .bf16), &bf_b, 4, 3e-2);
    // bf16 DATA MOVEMENT (raw copy, no arithmetic): a standalone transpose
    // through the dtype-generic indexed-copy kernel at bf16 storage. Bit-exact.
    const bf_t = bf16a(.{ 1, 2, 3, 4, 5, 6 });
    failures += checkUnaryT(zml.floats.BFloat16, allocator, io, cpu, metal, "bf16T", transpose23,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .bf16), &bf_t, 6, 1e-2);
    // bf16 REDUCTIONS — accumulate in f32, narrow to bf16 on store. Serial sum
    // (sumj) and a fused reduce(square) (sumSq, RMSNorm's shape). The tree/two-
    // level paths are f32-only, so a bf16 reduce uses the dtype-generic
    // serial/fused kernels. Small data; bf16 tolerance.
    const bf_r = bf16a(.{ 0.5, 1.5, 2.0, 0.25, 1.25, 3.0 });
    failures += checkUnaryT(zml.floats.BFloat16, allocator, io, cpu, metal, "bf16sum", sumj,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .bf16), &bf_r, 2, 3e-2);
    failures += checkUnaryT(zml.floats.BFloat16, allocator, io, cpu, metal, "bf16ssq", sumSq,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .bf16), &bf_r, 2, 3e-2);
    // bf16 BARE elementwise (single op → the non-fused kernel): |x| at bf16
    // storage (load→f32 compute→store). Bit-exact (abs preserves the value).
    const bf_e = bf16a(.{ -1.5, 2.5, -0.25, 3.0 });
    failures += checkUnaryT(zml.floats.BFloat16, allocator, io, cpu, metal, "bf16abs", abs,
        zml.Shape.init(.{ .n = 4 }, .bf16), &bf_e, 4, 1e-2);

    // Deeper chain (3 thunks, 2 intermediates): abs(x·W1)·W2. The SECOND matmul
    // reads a COMPUTED buffer (the abs result), not a parameter — the matmul-
    // reading-an-intermediate path. dot1=[[10,12],[-19,-24]], abs=[[10,12],[19,24]],
    // ·W2[[1,2],[3,4]] = [[46,68],[91,134]]. Exact integers.
    failures += check3Shaped(allocator, io, cpu, metal, "mlpAbs", matAbsMat,
        zml.Shape.init(.{ .m = 2, .k = 3 }, .f32), &[_]f32{ 1, -2, 3, -4, 5, -6 },
        zml.Shape.init(.{ .k = 3, .n = 2 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 },
        zml.Shape.init(.{ .n = 2, .p = 2 }, .f32), &[_]f32{ 1, 2, 3, 4 }, 4, exact);

    // Matmul → FUSION (abs∘negate consuming the dot result): the other graph-path
    // shape — a fusion thunk with a non-parameter operand. Mixed-sign inputs make
    // abs meaningful. dot=[[10,12],[-19,-24]] → |−dot| = [[10,12],[19,24]].
    failures += checkMatmul(allocator, io, cpu, metal, "matneg", matNegAbs,
        zml.Shape.init(.{ .m = 2, .k = 3 }, .f32), &[_]f32{ 1, -2, 3, -4, 5, -6 },
        zml.Shape.init(.{ .k = 3, .n = 2 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 }, 4, exact);

    // Matmul → indexed (transpose of the dot result): a third graph-path shape —
    // an indexed-copy thunk reading a computed buffer. dotᵀ = [[22,49],[28,64]].
    failures += checkMatmul(allocator, io, cpu, metal, "matT", matT,
        zml.Shape.init(.{ .m = 2, .k = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 },
        zml.Shape.init(.{ .k = 3, .n = 2 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 }, 4, exact);

    // Matmul → reduce (sum of the dot result over n): a fourth graph-path shape —
    // a reduce thunk reading a computed buffer. sum_n = [50, 113].
    failures += checkMatmul(allocator, io, cpu, metal, "matSum", matSum,
        zml.Shape.init(.{ .m = 2, .k = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 },
        zml.Shape.init(.{ .k = 3, .n = 2 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 }, 2, exact);

    // Deep chain (5 thunks: 3 matmuls + 2 abs) abs(abs(x·W1)·W2)·W3 — command-
    // buffer pipelining stress (many un-blocked launches, several live buffers).
    failures += check4Shaped(allocator, io, cpu, metal, "mlpDeep", mlpDeep, .{
        zml.Shape.init(.{ .m = 2, .k = 3 }, .f32),
        zml.Shape.init(.{ .k = 3, .n = 2 }, .f32),
        zml.Shape.init(.{ .n = 2, .p = 2 }, .f32),
        zml.Shape.init(.{ .p = 2, .q = 2 }, .f32),
    }, .{
        &[_]f32{ 1, -2, 3, -4, 5, -6 }, &[_]f32{ 1, 2, 3, 4, 5, 6 },
        &[_]f32{ 1, 2, 3, 4 },          &[_]f32{ 1, 2, 3, 4 },
    }, 4, exact);

    // Multi-output module (tuple root): one graph, two array results
    // (add, mul) returned together via the root tuple index table.
    failures += checkTwoOut(allocator, io, cpu, metal, "twoout", addMul, &a_bin, &b_bin, exact);

    // Concatenate along .j: [[1,2],[3,4]] ++ [[5,6],[7,8]] = [[1,2,5,6],[3,4,7,8]].
    failures += checkMatmul(allocator, io, cpu, metal, "concat", concatJ,
        zml.Shape.init(.{ .i = 2, .j = 2 }, .f32), &[_]f32{ 1, 2, 3, 4 },
        zml.Shape.init(.{ .i = 2, .j = 2 }, .f32), &[_]f32{ 5, 6, 7, 8 }, 8, exact);
    // bf16 concat — raw data movement (select-chain) at bf16 storage. The other
    // indexed kernels (DUS/gather) share this exact load/store path. Bit-exact.
    const cf_a = bf16a(.{ 1, 2, 3, 4 });
    const cf_b = bf16a(.{ 5, 6, 7, 8 });
    failures += checkMatmulT(zml.floats.BFloat16, allocator, io, cpu, metal, "bf16cat", concatJ,
        zml.Shape.init(.{ .i = 2, .j = 2 }, .bf16), &cf_a,
        zml.Shape.init(.{ .i = 2, .j = 2 }, .bf16), &cf_b, 8, 1e-2);

    // KV-cache indexing (runtime offset). dynSlice: x[6] at start=2, len=3 →
    // [12,13,14]. dynUpdate: write [7,8] into zeros[6] at start=3 → [0,0,0,7,8,0].
    failures += checkDynSlice(allocator, io, cpu, metal, "dslice", dynSlice,
        &[_]f32{ 10, 11, 12, 13, 14, 15 }, 2, 3);
    // gatherSlices row at runtime index 2 of a 4x4 → expect row 2 = {20,21,22,23}
    // (the zml.attention vanilla DECODE mask-row gather, in isolation).
    failures += checkDynSlice(allocator, io, cpu, metal, "gathRow", gatherRow,
        &[_]f32{ 0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33 }, 2, 4);
    failures += checkDynUpdate(allocator, io, cpu, metal, "dupd", dynUpdate,
        &[_]f32{ 0, 0, 0, 0, 0, 0 }, &[_]f32{ 7, 8 }, 3, 6);
    // scatterSlices at a runtime index — the REAL KV-cache write (KvCache.updateAt).
    failures += checkDynUpdate(allocator, io, cpu, metal, "scatRow", scatterRow,
        &[_]f32{ 0, 0, 0, 0, 0, 0, 0, 0 }, &[_]f32{ 7, 8 }, 2, 8);
    // Rank-2 KV-cache write: cache[4,3], write row [100,101,102] at pos=2.
    failures += checkKvWrite(allocator, io, cpu, metal, 4, 3,
        &[_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }, &[_]f32{ 100, 101, 102 }, 2);
    // Embedding lookup (gather rows): table[4,3], idx=[2,0,3] → rows 2,0,3 =
    // [[6,7,8],[0,1,2],[9,10,11]].
    failures += checkEmbed(allocator, io, cpu, metal, 4, 3,
        &[_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 }, &[_]i32{ 2, 0, 3 });

    // Transformer building blocks (own composites, supported ops only).
    // softmax over .j; rmsNorm over .j — both [2,3]->[2,3].
    failures += checkShaped(allocator, io, cpu, metal, "smax", mySoftmax,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);
    failures += checkShaped(allocator, io, cpu, metal, "rmsn", myRmsNorm,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 1, 2, 3, 4, 5, 6 }, 6);
    // ZML's stock softmax (compare + select for the masked-row guard).
    failures += checkShaped(allocator, io, cpu, metal, "smaxZ", stockSoftmax,
        zml.Shape.init(.{ .i = 2, .j = 3 }, .f32), &[_]f32{ 0, 1, 2, 3, 4, 5 }, 6);

    // ===== Tiny REAL model end-to-end: a per-token 2-layer MLP classifier =====
    // embed → matmul+bias → relu → matmul+bias → logits, in one Metal graph.
    failures += checkTinyMLP(allocator, io, cpu, metal);

    // SwiGLU feed-forward (Llama/Mistral MLP): silu(x·Wg) * (x·Wu) · Wd. Two
    // matmuls → silu(logistic fusion) * → matmul. [2,3]·... → [2,2].
    failures += check4Shaped(allocator, io, cpu, metal, "swiglu", swiglu, .{
        zml.Shape.init(.{ .m = 2, .k = 3 }, .f32),
        zml.Shape.init(.{ .k = 3, .n = 2 }, .f32),
        zml.Shape.init(.{ .k = 3, .n = 2 }, .f32),
        zml.Shape.init(.{ .n = 2, .p = 2 }, .f32),
    }, .{
        &[_]f32{ 1, -2, 3, -4, 5, -6 }, &[_]f32{ 1, 2, 3, 4, 5, 6 },
        &[_]f32{ 6, 5, 4, 3, 2, 1 },    &[_]f32{ 1, 2, 3, 4 },
    }, 4, 1e-2);

    // ===== Capstone: a full tiny TRANSFORMER block end-to-end =====
    // pre-norm single-head causal self-attention + residual + pre-norm MLP +
    // residual — rmsNorm, Q/K/V/O matmuls, scaled QKᵀ+mask, softmax, attn·V, MLP.
    failures += checkAttnBlock(allocator, io, cpu, metal);

    // The real Llama-decode-layer shape, dtype-generic: same block + RoPE on Q/K
    // + an on-device causal mask, run at f32/f16/bf16. This is the headline —
    // every dtype-generic family composing into one low-precision transformer
    // layer, matching the CPU oracle. bf16's coarse mantissa compounds ~1 ULP
    // per op through a deep chain → looser tol at lower precision.
    failures += checkAttnBlockRope(f32, allocator, io, cpu, metal, "ropeF32", 2e-3);
    failures += checkAttnBlockRope(f16, allocator, io, cpu, metal, "ropeF16", 2e-2);
    failures += checkAttnBlockRope(zml.floats.BFloat16, allocator, io, cpu, metal, "ropeBf16", 8e-2);

    if (failures != 0) {
        log.err("❌ {d} op(s) mismatched Metal vs CPU", .{failures});
        return error.MetalMismatch;
    }
    log.info("✅ PASS: all Metal ops match the CPU oracle (elementwise + transpose/bcast/reshape + sum-reduce)", .{});
}
