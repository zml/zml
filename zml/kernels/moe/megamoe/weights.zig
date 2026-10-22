//! DeepSeek-V4.1 checkpoint expert weights to the MegaMoE kernel layouts (`megamoe.Weights`),
//! as tensor ops: they run as small XLA programs at weight-load time.
//!
//! The reference is DeepGEMM's `transform_weights_for_mega_moe` (vLLM's vendored copy,
//! `vllm/third_party/deep_gemm/mega/__init__.py:120-172`) on 1x32 UE8M0 scales packed four per
//! int32, which vLLM's MegaMoE loader writes straight from the checkpoint shards
//! (`vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py:433-448, 523-604`). The
//! reference dumps' `LAYOUTS.md` §4 documents the results:
//! - weights keep their bytes (packed e2m1, low nibble first, or e4m3); the L1 weights interleave
//!   gate (w1) and up (w3) rows by 8 (`l1WeightRow`);
//! - scales become int32 words of four UE8M0 exponents (K-group `4kq + j` in byte `j`,
//!   little-endian), with rows interleaved like the L1 weights, then permuted within each
//!   128-row block for UTCCP (`utccpRow`, `l1ScaleRow`), stored MN-major: memory `[.., K/128, MN]`.
//!
//! The results are `megamoe.Weights` fields as `forward`/`forwardStaged` take them. Routed
//! tensors keep the expert axis as an independent leading axis: every op acts within one expert,
//! so expert-sharded inputs give expert-sharded outputs without communication.
const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const Tensor = zml.Tensor;

pub const Routed = struct {
    /// `l1_w` `[E, 2I, H/2]` or `l2_w` `[E, H, I/2]`: u8, packed e2m1.
    w: Tensor,
    /// `l1_sf` `[E, H/128, 2I]` or `l2_sf` `[E, I/128, H]`: i32, MN-major UE8M0 words.
    sf: Tensor,
};

/// Routed L1 (gate/up) weights of the local experts.
/// `w1` (gate), `w3` (up): `[E, I, H/2]` i8/u8, packed e2m1 (the checkpoint bytes).
/// `w1_scale`, `w3_scale`: `[E, I, H/32]` f8e8m0/u8, 1x32 UE8M0.
/// Returns `l1_w` `[E, 2I, H/2]` u8 (gate/up interleaved by 8 rows) and `l1_sf` `[E, H/128, 2I]`
/// i32 (MN-major words of 4 UE8M0, rows interleaved like `l1_w`, then UTCCP-permuted within each
/// 128-row block). `w` keeps `w1`'s tags; `sf` takes `w1_scale`'s with its last two axes swapped.
pub fn routedL1(w1: Tensor, w1_scale: Tensor, w3: Tensor, w3_scale: Tensor) Routed {
    expectRouted("w1", w1, w1_scale);
    expectRouted("w3", w3, w3_scale);
    stdx.debug.assert(std.mem.eql(i64, w1.dims(), w3.dims()), "MegaMoE routedL1 expects w1 and w3 of the same shape, got {f} and {f}", .{ w1, w3 });
    expectBlocks("routedL1", w1.dim(1), 64, 2 * w1.dim(2));

    // `transform_weights_for_mega_moe` (`mega/__init__.py:161-165`): `_interleave_weights` on the
    // weights, `_transpose_sf_for_utccp(_interleave_weights(sf))` on the scales.
    const w = interleaveGateUp(bytes(w1), bytes(w3));
    const sf = utccpMnMajor(interleaveGateUp(scaleWords(w1_scale), scaleWords(w3_scale)));
    return .{ .w = w.withTags(w1.shape()), .sf = sf.withTags(w1_scale.shape().transpose(.{ 0, 2, 1 })) };
}

/// Routed L2 (down) weights of the local experts.
/// `w2`: `[E, H, I/2]` i8/u8, packed e2m1. `w2_scale`: `[E, H, I/32]` f8e8m0/u8, 1x32 UE8M0.
/// Returns `l2_w` `[E, H, I/2]` u8 (bytes unchanged) and `l2_sf` `[E, I/128, H]` i32 (MN-major
/// words, UTCCP-permuted rows). Tags as for `routedL1`.
pub fn routedL2(w2: Tensor, w2_scale: Tensor) Routed {
    expectRouted("w2", w2, w2_scale);
    expectBlocks("routedL2", w2.dim(1), 128, 2 * w2.dim(2));

    // `mega/__init__.py:166-167`: L2 weights unchanged, scales transposed for UTCCP.
    return .{ .w = bytes(w2).withTags(w2.shape()), .sf = utccpMnMajor(scaleWords(w2_scale)).withTags(w2_scale.shape().transpose(.{ 0, 2, 1 })) };
}

pub const Shared = struct {
    /// `shared_l1_w` `[2I, H]` or `shared_l2_w` `[H, I]`: u8, e4m3 bytes.
    w: Tensor,
    /// `shared_l1_sf` `[H/128, 2I]` or `shared_l2_sf` `[I/128, H]`: i32, MN-major UE8M0 words.
    sf: Tensor,
};

/// Shared-expert L1 (gate/up) weights, for one shared expert (`n_shared = 1`).
/// `w1` (gate), `w3` (up): `[I, H]` f8e4m3fn/u8. `w1_scale`, `w3_scale`: f8e8m0/u8 UE8M0 block
/// scales tiling `[I, H/32]`: the checkpoint's 32x32 blocks are `[I/32, H/32]`; 1x32 scales
/// (`[I, H/32]`) pass through. They are expanded to 1x32 per row like vLLM's `_pack_scale`
/// (`deep_gemm_mega_moe.py:572-582`), then transformed like one routed expert
/// (`deep_gemm_mega_moe.py:427-431, 537-570`).
/// Returns `shared_l1_w` `[2I, H]` u8 (gate/up interleaved by 8 rows) and `shared_l1_sf`
/// `[H/128, 2I]` i32. Tags as for `routedL1`.
pub fn sharedL1(w1: Tensor, w1_scale: Tensor, w3: Tensor, w3_scale: Tensor) Shared {
    expectShared("w1", w1, w1_scale);
    expectShared("w3", w3, w3_scale);
    stdx.debug.assert(std.mem.eql(i64, w1.dims(), w3.dims()), "MegaMoE sharedL1 expects w1 and w3 of the same shape, got {f} and {f}", .{ w1, w3 });
    const rows = w1.dim(0);
    const k = w1.dim(1);
    expectBlocks("sharedL1", rows, 64, k);

    const w = interleaveGateUp(oneExpert(bytes(w1)), oneExpert(bytes(w3)));
    const sf = utccpMnMajor(interleaveGateUp(
        oneExpert(scaleWords(expandScales(w1_scale, rows, @divExact(k, 32)))),
        oneExpert(scaleWords(expandScales(w3_scale, rows, @divExact(k, 32)))),
    ));
    return .{
        .w = w.reshape(.{ 2 * rows, k }).withTags(w1.shape()),
        .sf = sf.reshape(.{ @divExact(k, 128), 2 * rows }).withTags(w1_scale.shape().transpose(.{ 1, 0 })),
    };
}

/// Shared-expert L2 (down) weights, for one shared expert.
/// `w2`: `[H, I]` f8e4m3fn/u8. `w2_scale`: f8e8m0/u8 block scales tiling `[H, I/32]` (checkpoint
/// 32x32 blocks: `[H/32, I/32]`), expanded like `sharedL1`'s.
/// Returns `shared_l2_w` `[H, I]` u8 (bytes unchanged) and `shared_l2_sf` `[I/128, H]` i32.
pub fn sharedL2(w2: Tensor, w2_scale: Tensor) Shared {
    expectShared("w2", w2, w2_scale);
    const rows = w2.dim(0);
    const k = w2.dim(1);
    expectBlocks("sharedL2", rows, 128, k);

    const sf = utccpMnMajor(oneExpert(scaleWords(expandScales(w2_scale, rows, @divExact(k, 32)))));
    return .{
        .w = bytes(w2).withTags(w2.shape()),
        .sf = sf.reshape(.{ @divExact(k, 128), rows }).withTags(w2_scale.shape().transpose(.{ 1, 0 })),
    };
}

/// Row of `l1_w` (and of `shared_l1_w`) holding row `r` of w1 (`half` 0, gate) or w3 (`half` 1,
/// up): DeepGEMM's `_interleave_weights(gran=8)` (`mega/__init__.py:120-134`), i.e. vLLM's
/// `r -> (r // 8) * 16 + 8 * half + r % 8` (`deep_gemm_mega_moe.py:599-601`).
pub fn l1WeightRow(r: u32, half: u1) u32 {
    return (r / 8) * 16 + 8 * @as(u32, half) + r % 8;
}

/// Row of a scale tensor after DeepGEMM's `_transpose_sf_for_utccp` (`mega/__init__.py:137-151`,
/// `reshape(G, -1, 4, 32, K).transpose(2, 3)`): row `128b + 32j + i` moves to `128b + 4i + j`.
/// This is the MN index of row `r` of the w2 scales in `l2_sf` (`deep_gemm_mega_moe.py:596-597`).
pub fn utccpRow(r: u32) u32 {
    return (r / 128) * 128 + (r % 32) * 4 + (r % 128) / 32;
}

/// MN index in `l1_sf` of row `r` of the w1 (`half` 0) or w3 (`half` 1) scales, i.e.
/// `utccpRow(l1WeightRow(r, half))`, in vLLM's closed form
/// `64b + 16j + 8q + m -> 128b + 64q + 32 half + 4m + j` (`deep_gemm_mega_moe.py:602-604`).
pub fn l1ScaleRow(r: u32, half: u1) u32 {
    const b = r / 64;
    const j = (r / 16) % 4;
    const q = (r / 8) % 2;
    const m = r % 8;
    return 128 * b + 64 * q + 32 * @as(u32, half) + 4 * m + j;
}

/// `[E, N, K]` gate and up rows interleaved by 8: `[g0..7, u0..7, g8..15, u8..15, ..]`, i.e.
/// `torch.stack([gate, up], dim=2)` on `[E, N/8, 8, K]` views (`mega/__init__.py:129-133`).
fn interleaveGateUp(gate: Tensor, up: Tensor) Tensor {
    const e = gate.dim(0);
    const n = gate.dim(1);
    const k = gate.dim(2);
    const blocks = @divExact(n, 8);
    const halves = [2]Tensor{ gate.reshape(.{ e, blocks, 1, 8, k }), up.reshape(.{ e, blocks, 1, 8, k }) };
    return Tensor.concatenate(&halves, 2).reshape(.{ e, 2 * n, k });
}

/// `[E, MN, K/128]` i32 scale words, rows permuted for UTCCP (`_transpose_sf_for_utccp`), in
/// MN-major memory order `[E, K/128, MN]`: `[E, MN/128, 4, 32, Kq]` read as `[E, Kq, MN/128, 32, 4]`.
fn utccpMnMajor(sf: Tensor) Tensor {
    const e = sf.dim(0);
    const mn = sf.dim(1);
    const kq = sf.dim(2);
    return sf.reshape(.{ e, @divExact(mn, 128), 4, 32, kq }).transpose(.{ 0, 4, 1, 3, 2 }).reshape(.{ e, kq, mn });
}

/// `[.., G]` UE8M0 bytes as `[.., G/4]` int32 words, byte `j` of word `kq` = group `4kq + j`:
/// vLLM's `.view(torch.int32)` of the contiguous bytes (`deep_gemm_mega_moe.py:582`).
fn scaleWords(scale: Tensor) Tensor {
    const b = bytes(scale);
    return b.splitAxis(-1, .{ @divExact(b.dim(-1), 4), 4 }).bitCast(.i32);
}

/// Block scales repeated to `[rows, cols]` (`repeat_interleave` on rows then columns), vLLM's
/// `_pack_scale` (`deep_gemm_mega_moe.py:572-582`).
fn expandScales(scale: Tensor, rows: i64, cols: i64) Tensor {
    return bytes(scale).stutter(&.{ @intCast(@divExact(rows, scale.dim(0))), @intCast(@divExact(cols, scale.dim(1))) });
}

/// `[N, K]` as a single-expert `[1, N, K]`: the shared expert has the layout of one routed
/// expert slice (`deep_gemm_mega_moe.py:427-431`).
fn oneExpert(t: Tensor) Tensor {
    return t.reshape(.{ 1, t.dim(0), t.dim(1) });
}

fn bytes(t: Tensor) Tensor {
    return if (t.dtype() == .u8) t else t.bitCast(.u8);
}

fn expectRouted(comptime name: []const u8, w: Tensor, scale: Tensor) void {
    stdx.debug.assert(w.rank() == 3 and (w.dtype() == .i8 or w.dtype() == .u8), "MegaMoE expects " ++ name ++ " as [E, N, K/2] i8/u8 packed e2m1, got {f}", .{w});
    const groups = std.math.divExact(i64, 2 * w.dim(2), 32) catch 0;
    const ok = scale.rank() == 3 and scale.dim(0) == w.dim(0) and scale.dim(1) == w.dim(1) and scale.dim(2) == groups;
    stdx.debug.assert(ok and isScaleDtype(scale), "MegaMoE expects " ++ name ++ "_scale as [E, N, K/32] = [{d}, {d}, {d}] f8e8m0/u8 (1x32 UE8M0), got {f}", .{ w.dim(0), w.dim(1), groups, scale });
}

fn expectShared(comptime name: []const u8, w: Tensor, scale: Tensor) void {
    stdx.debug.assert(w.rank() == 2 and (w.dtype() == .f8e4m3fn or w.dtype() == .u8), "MegaMoE expects shared " ++ name ++ " as [N, K] f8e4m3fn/u8, got {f}", .{w});
    const groups = std.math.divExact(i64, w.dim(1), 32) catch 0;
    const ok = scale.rank() == 2 and groups != 0 and @mod(w.dim(0), scale.dim(0)) == 0 and @mod(groups, scale.dim(1)) == 0;
    stdx.debug.assert(ok and isScaleDtype(scale), "MegaMoE expects shared " ++ name ++ "_scale as f8e8m0/u8 block scales tiling [N, K/32] = [{d}, {d}], got {f}", .{ w.dim(0), groups, scale });
}

fn isScaleDtype(scale: Tensor) bool {
    return scale.dtype() == .f8e8m0 or scale.dtype() == .u8;
}

/// `rows` (the MN extent) in whole UTCCP 128-row blocks after the gate/up interleave, and `k`
/// in whole int32 words of 4 x 32-element groups.
fn expectBlocks(comptime op: []const u8, rows: i64, row_multiple: i64, k: i64) void {
    stdx.debug.assert(@mod(rows, row_multiple) == 0 and @mod(k, 128) == 0, "MegaMoE " ++ op ++ " expects rows ({d}) a multiple of {d} and K ({d}) a multiple of 128", .{ rows, row_multiple, k });
}
