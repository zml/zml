//! DeepGEMM's MegaMoE (`sm100_fp8_fp4_mega_moe_impl`) as a zml op: expert-parallel
//! dispatch, FP8 x FP4 grouped GEMMs with the SwiGLU epilogue, the fused FP8 shared
//! expert and the combine, in one persistent CuTe Program per rank.
//!
//! Tensors are DeepGEMM's kernel inputs in memory order (row-major shapes).
//! The weight layouts are DeepGEMM's post-`transform_weights_for_mega_moe`
//! layouts documented in the reference dumps' `LAYOUTS.md`.
//!
//! One custom call per op: a CuTe Program (`megamoe/program.zig`) whose host function launches
//! kernel #0, the staging of the inputs into this rank's symmetric buffer (`staging.zig`), then
//! kernel #1, the MegaMoE kernel (`kernel.zig`), with DeepGEMM's launch configuration. BLOCK_M
//! and every other template argument are resolved when the op is built, from the static token
//! count (DeepGEMM's heuristics, `layout.zig`).
const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const lib = @import("zml/kernels/moe/megamoe/lib");

const program = lib.program;

/// DeepSeek-V4.1 checkpoint expert weights to the `Weights` layouts (load-time tensor ops).
pub const weights = @import("weights.zig");

/// `layout::kLCMCandidateBlockM`: the least common multiple of DeepGEMM's candidate
/// BLOCK_M values, used to align `num_max_tokens_per_rank`.
pub const token_alignment: u32 = lib.layout.token_alignment;

pub const Problem = struct {
    num_ranks: u32,
    num_experts: u32,
    num_topk: u32,
    hidden: u32,
    intermediate: u32,
    num_shared_experts: u32,
    /// As requested; aligned internally like DeepGEMM (8192 -> 9600).
    num_max_tokens_per_rank: u32,
    activation_clamp: f32 = 10.0,
    fast_math: bool = true,

    pub fn numExpertsPerRank(self: Problem) u32 {
        return @divExact(self.num_experts, self.num_ranks);
    }

    /// `SymmBuffer.num_max_tokens_per_rank`: the token capacity every buffer region is sized for.
    pub fn numMaxTokensPerRankAligned(self: Problem) u32 {
        return lib.layout.alignTokens(self.num_max_tokens_per_rank);
    }

    pub fn validate(self: Problem) void {
        stdx.debug.assert(self.num_ranks >= 1 and self.num_experts % self.num_ranks == 0, "MegaMoE expects experts ({d}) divisible by ranks ({d})", .{ self.num_experts, self.num_ranks });
        stdx.debug.assert(self.num_topk >= 1, "MegaMoE expects top-k >= 1", .{});
        // BLOCK_K = 128 for both GEMMs, and 2-CTA BLOCK_N = 2 x 128 for both N dimensions.
        stdx.debug.assert(self.hidden % 256 == 0 and (2 * self.intermediate) % 256 == 0 and self.intermediate % 128 == 0, "MegaMoE expects hidden ({d}) and 2 x intermediate ({d}) multiples of 256", .{ self.hidden, 2 * self.intermediate });
        stdx.debug.assert(self.num_max_tokens_per_rank >= 1, "MegaMoE expects num_max_tokens_per_rank >= 1", .{});
        stdx.debug.assert(self.activation_clamp >= 0, "MegaMoE expects a non-negative activation clamp, got {d}", .{self.activation_clamp});
        // DeepGEMM's `kFastMath = false` SwiGLU (`expf`, IEEE division) is not ported.
        stdx.debug.assert(self.fast_math, "MegaMoE only implements fast_math = true", .{});
    }

    /// The kernel instantiation for a call with `num_tokens` tokens per rank on `num_sms` SMs.
    pub fn instance(self: Problem, num_tokens: u32, num_sms: u32) program.Instance {
        return .{
            .num_ranks = self.num_ranks,
            .num_experts = self.num_experts,
            .num_topk = self.num_topk,
            .hidden = self.hidden,
            .intermediate_hidden = self.intermediate,
            .num_shared_experts = self.num_shared_experts,
            .num_max_tokens_per_rank = self.num_max_tokens_per_rank,
            .num_tokens = num_tokens,
            .num_sms = num_sms,
            .weight_type = .fp4_unpacked,
            .activation_clamp = self.activation_clamp,
            .fast_math = self.fast_math,
        };
    }
};

pub const Workspace = union(enum) {
    /// Single-rank testing before XLA support: a persistent u8 buffer the caller zeroes once and
    /// threads through calls (input aliased to the `workspace` output). Requires num_ranks == 1.
    operand: zml.Tensor,
    /// XLA-owned symmetric workspace shared by every call with this name (Design A).
    symmetric: struct { name: []const u8 },
};

/// Kernel inputs exactly as DeepGEMM consumes them. Shapes are memory order, row-major.
pub const StagedInputs = struct {
    /// `[T, H]` u8: e4m3 bytes.
    x: zml.Tensor,
    /// `[T, H/128]` i32: packed UE8M0, 4 per int, token-major.
    x_sf: zml.Tensor,
    /// `[T, K]` i64 (-1 = no route).
    topk_idx: zml.Tensor,
    /// `[T, K]` f32.
    topk_weights: zml.Tensor,
};

/// Local experts only (E_local = num_experts / num_ranks), in DeepGEMM's
/// post-`transform_weights_for_mega_moe` layout and memory order. Scale factors are
/// MN-major int32 words of 4 UE8M0 exponents (32-element K groups), i.e. memory order
/// `[.., K/128, MN]`, with rows permuted for UTCCP (and interleaved like the weights for L1).
pub const Weights = struct {
    /// `[E_local, 2I, H/2]` packed e2m1 (low nibble first), gate/up interleaved by 8 rows.
    l1_w: zml.Tensor,
    /// `[E_local, H/128, 2I]` i32.
    l1_sf: zml.Tensor,
    /// `[E_local, H, I/2]` packed e2m1.
    l2_w: zml.Tensor,
    /// `[E_local, I/128, H]` i32.
    l2_sf: zml.Tensor,
    /// `[2SI, H]` e4m3 bytes, gate/up interleaved by 8 rows.
    shared_l1_w: ?zml.Tensor = null,
    /// `[H/128, 2SI]` i32.
    shared_l1_sf: ?zml.Tensor = null,
    /// `[H, SI]` e4m3 bytes.
    shared_l2_w: ?zml.Tensor = null,
    /// `[SI/128, H]` i32.
    shared_l2_sf: ?zml.Tensor = null,
};

pub const Result = struct { y: zml.Tensor, workspace: ?zml.Tensor };

/// Runs on one rank's data (call inside zml.ops.manualComputation for num_ranks > 1).
/// T = x.dim(0) is static; BLOCK_M etc. are resolved at compile time from T like DeepGEMM does per call.
/// Returns `y`: `[T, H]` bf16, with `x`'s tags.
pub fn forwardStaged(
    problem: Problem,
    inputs: StagedInputs,
    weights_: Weights,
    /// `[num_sms]` u8, indexed by `%smid`: `num_sms` (the grid size) must be the device's SM
    /// count (DeepGEMM sizes this table by the device's SM count, independently of the grid).
    sm_locality_domains: zml.Tensor,
    workspace: Workspace,
) Result {
    problem.validate();
    const tokens = inputs.x.dim(0);
    validateTokens(problem, inputs.x, inputs.topk_weights);
    expectShape("x", inputs.x, &.{ tokens, problem.hidden }, &.{ .u8, .f8e4m3fn });
    expectShape("x_sf", inputs.x_sf, &.{ tokens, @divExact(problem.hidden, 128) }, &.{.i32});
    expectShape("topk_idx", inputs.topk_idx, &.{ tokens, problem.num_topk }, &.{.i64});
    validateWeights(problem, weights_);
    validateWorkspace(problem, sm_locality_domains, workspace);

    return call(.staged, problem, .{
        .x = byteView(inputs.x),
        .x_sf = inputs.x_sf,
        .topk_idx = inputs.topk_idx,
        .topk_weights = inputs.topk_weights,
    }, weights_, sm_locality_domains, workspace);
}

/// Same, from bf16 activations: staging (FP8 1x32 UE8M0 quantization, i32->i64 ids, shared-expert SF
/// layout) runs as kernel #0 of the same Program and must be bit-exact with DeepGEMM's
/// `per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)`.
pub fn forward(
    problem: Problem,
    /// `[T, H]` bf16.
    x: zml.Tensor,
    /// `[T, K]` i32 (-1 = no route).
    topk_idx: zml.Tensor,
    /// `[T, K]` f32.
    topk_weights: zml.Tensor,
    weights_: Weights,
    /// `[num_sms]` u8, indexed by `%smid`: `num_sms` (the grid size) must be the device's SM
    /// count (DeepGEMM sizes this table by the device's SM count, independently of the grid).
    sm_locality_domains: zml.Tensor,
    workspace: Workspace,
) Result {
    problem.validate();
    const tokens = x.dim(0);
    validateTokens(problem, x, topk_weights);
    expectShape("x", x, &.{ tokens, problem.hidden }, &.{.bf16});
    expectShape("topk_idx", topk_idx, &.{ tokens, problem.num_topk }, &.{.i32});
    validateWeights(problem, weights_);
    validateWorkspace(problem, sm_locality_domains, workspace);

    return call(.bf16, problem, .{
        .x = x,
        .x_sf = null,
        .topk_idx = topk_idx,
        .topk_weights = topk_weights,
    }, weights_, sm_locality_domains, workspace);
}

/// The CuTe Program of one input kind and workspace mode (`program.emitProgram`); its custom
/// call's operands and results are `program.params`.
fn MegaMoEProgram(comptime inputs: program.InputKind, comptime workspace: program.WorkspaceKind) type {
    return struct {
        const name = "megamoe_" ++ @tagName(inputs) ++ "_" ++ @tagName(workspace);

        fn run(b: *zml.kernel.cute.Builder, spec: program.Spec) zml.kernel.cute.FinishError!void {
            return program.emitProgram(b, spec, name);
        }

        pub const P = zml.kernel.cute.Program(program.Spec, .{
            .name = name,
            .inputs = program.paramNames(inputs, workspace, false),
            .outputs = program.paramNames(inputs, workspace, true),
            .run = run,
        });
    };
}

const Activations = struct {
    x: zml.Tensor,
    x_sf: ?zml.Tensor,
    topk_idx: zml.Tensor,
    topk_weights: zml.Tensor,
};

/// FP8 tensors as the byte tensors the host function declares (`u8` storage of e4m3).
fn byteView(t: zml.Tensor) zml.Tensor {
    return if (t.dtype() == .f8e4m3fn) t.bitCast(.u8) else t;
}

fn call(
    comptime inputs: program.InputKind,
    problem: Problem,
    acts: Activations,
    weights_: Weights,
    sm_locality_domains: zml.Tensor,
    workspace: Workspace,
) Result {
    const tokens: u32 = @intCast(acts.x.dim(0));
    const num_sms: u32 = @intCast(sm_locality_domains.dim(0));
    // Without a shared expert the Program still takes the shared operands; the routed tensors
    // stand in (DeepGEMM likewise copies the routed tensor maps), the kernel never reads them.
    const operands = .{
        .x = acts.x,
        .x_sf = acts.x_sf,
        .topk_idx = acts.topk_idx,
        .topk_weights = acts.topk_weights,
        .l1_w = weights_.l1_w,
        .l1_sf = weights_.l1_sf,
        .l2_w = weights_.l2_w,
        .l2_sf = weights_.l2_sf,
        .shared_l1_w = byteView(weights_.shared_l1_w orelse weights_.l1_w),
        .shared_l1_sf = weights_.shared_l1_sf orelse weights_.l1_sf,
        .shared_l2_w = byteView(weights_.shared_l2_w orelse weights_.l2_w),
        .shared_l2_sf = weights_.shared_l2_sf orelse weights_.l2_sf,
        .sm_locality_domains = sm_locality_domains,
    };
    const y_shape = acts.x.shape().withDtype(.bf16);
    switch (workspace) {
        .operand => |ws| {
            const P = MegaMoEProgram(inputs, .operand).P;
            var in: P.Inputs = undefined;
            fillOperands(P.Inputs, &in, operands);
            in.workspace = ws;
            const r = P.call(in, .{ .y = y_shape, .workspace_out = ws.shape() }, .{
                .cfg = .{ .instance = problem.instance(tokens, num_sms), .inputs = inputs, .workspace = .operand },
                .output_operand_aliases = .{ .workspace_out = .workspace },
            });
            return .{ .y = r.y, .workspace = r.workspace_out };
        },
        .symmetric => |s| {
            const P = MegaMoEProgram(inputs, .symmetric).P;
            var in: P.Inputs = undefined;
            fillOperands(P.Inputs, &in, operands);
            const r = P.call(in, .{ .y = y_shape }, .{
                .cfg = .{ .instance = problem.instance(tokens, num_sms), .inputs = inputs, .workspace = .symmetric },
                .symmetric_workspace = .{ .name = s.name, .bytes = @intCast(workspaceBytes(problem, num_sms)) },
            });
            return .{ .y = r.y, .workspace = null };
        },
    }
}

fn fillOperands(comptime Inputs: type, in: *Inputs, operands: anytype) void {
    inline for (@typeInfo(Inputs).@"struct".field_names) |f| {
        if (comptime std.mem.eql(u8, f, "workspace")) continue;
        const v = @field(operands, f);
        @field(in, f) = if (@TypeOf(v) == ?zml.Tensor) v.? else v;
    }
}

fn validateTokens(problem: Problem, x: zml.Tensor, topk_weights: zml.Tensor) void {
    stdx.debug.assert(x.rank() == 2, "MegaMoE expects x [T, H], got {f}", .{x});
    const tokens = x.dim(0);
    stdx.debug.assert(tokens <= problem.numMaxTokensPerRankAligned(), "MegaMoE got {d} tokens, more than num_max_tokens_per_rank {d} (aligned {d})", .{ tokens, problem.num_max_tokens_per_rank, problem.numMaxTokensPerRankAligned() });
    expectShape("topk_weights", topk_weights, &.{ tokens, problem.num_topk }, &.{.f32});
}

fn validateWeights(problem: Problem, weights_: Weights) void {
    const experts = problem.numExpertsPerRank();
    const h = problem.hidden;
    const i = problem.intermediate;
    expectShape("l1_w", weights_.l1_w, &.{ experts, 2 * i, h / 2 }, &.{ .u8, .i8 });
    expectShape("l2_w", weights_.l2_w, &.{ experts, h, i / 2 }, &.{ .u8, .i8 });
    expectShape("l1_sf", weights_.l1_sf, &.{ experts, h / 128, 2 * i }, &.{.i32});
    expectShape("l2_sf", weights_.l2_sf, &.{ experts, i / 128, h }, &.{.i32});

    const shared = [_]?zml.Tensor{ weights_.shared_l1_w, weights_.shared_l1_sf, weights_.shared_l2_w, weights_.shared_l2_sf };
    var present: u32 = 0;
    for (shared) |t| present += @intFromBool(t != null);
    if (problem.num_shared_experts == 0) {
        stdx.debug.assert(present == 0, "MegaMoE got shared expert weights but num_shared_experts = 0", .{});
        return;
    }
    stdx.debug.assert(present == shared.len, "MegaMoE with {d} shared experts needs all four shared weight tensors, got {d}", .{ problem.num_shared_experts, present });
    const si = i * problem.num_shared_experts;
    expectShape("shared_l1_w", weights_.shared_l1_w.?, &.{ 2 * si, h }, &.{ .u8, .f8e4m3fn });
    expectShape("shared_l2_w", weights_.shared_l2_w.?, &.{ h, si }, &.{ .u8, .f8e4m3fn });
    expectShape("shared_l1_sf", weights_.shared_l1_sf.?, &.{ h / 128, 2 * si }, &.{.i32});
    expectShape("shared_l2_sf", weights_.shared_l2_sf.?, &.{ si / 128, h }, &.{.i32});
}

fn validateWorkspace(problem: Problem, sm_locality_domains: zml.Tensor, workspace: Workspace) void {
    stdx.debug.assert(sm_locality_domains.rank() == 1 and sm_locality_domains.dtype() == .u8, "MegaMoE expects sm_locality_domains [num_sms] u8, got {f}", .{sm_locality_domains});
    const num_sms: u32 = @intCast(sm_locality_domains.dim(0));
    // The kernel runs one 2-CTA cluster per SM pair.
    stdx.debug.assert(num_sms >= 2 and num_sms % 2 == 0, "MegaMoE expects an even SM count, got {d}", .{num_sms});
    switch (workspace) {
        .operand => |ws| {
            stdx.debug.assert(problem.num_ranks == 1, "MegaMoE Workspace.operand requires num_ranks == 1, got {d}; use Workspace.symmetric", .{problem.num_ranks});
            stdx.debug.assert(ws.rank() == 1 and ws.dtype() == .u8, "MegaMoE expects the workspace operand as [bytes] u8, got {f}", .{ws});
            const required = workspaceBytes(problem, num_sms);
            stdx.debug.assert(ws.byteSize() >= required, "MegaMoE workspace operand has {d} bytes, needs {d}", .{ ws.byteSize(), required });
        },
        .symmetric => |s| {
            stdx.debug.assert(s.name.len != 0, "MegaMoE expects a symmetric workspace name", .{});
            if (problem.num_ranks > 1) {
                stdx.debug.assert(zml.Compiler.current().manual_computation_depth > 0, "MegaMoE with {d} ranks must be called inside zml.ops.manualComputation", .{problem.num_ranks});
            }
        },
    }
}

fn expectShape(comptime name: []const u8, t: zml.Tensor, dims: []const i64, dtypes: []const zml.DataType) void {
    var ok = t.rank() == dims.len;
    if (ok) for (dims, 0..) |d, axis| {
        ok = ok and t.dim(axis) == d;
    };
    stdx.debug.assert(ok, "MegaMoE expects " ++ name ++ " with dims {any}, got {f}", .{ dims, t });
    stdx.debug.assert(std.mem.findScalar(zml.DataType, dtypes, t.dtype()) != null, "MegaMoE expects " ++ name ++ " dtype in {any}, got {f}", .{ dtypes, t });
}

/// DeepGEMM's `get_symm_buffer_size_for_mega_moe` (FP8 x FP4, SwiGLU): the bytes of one rank's
/// symmetric buffer, i.e. the workspace (signals, routing metadata), the staged inputs, the
/// L1/L2 activation rings and the combine slots (`layout.SymmBufferLayout`).
pub fn workspaceBytes(problem: Problem, num_sms: u32) u64 {
    problem.validate();
    return problem.instance(1, num_sms).symmLayout().numBytes();
}

test workspaceBytes {
    const dsv41: Problem = .{
        .num_ranks = 2,
        .num_experts = 384,
        .num_topk = 6,
        .hidden = 5120,
        .intermediate = 2304,
        .num_shared_experts = 1,
        .num_max_tokens_per_rank = 4096,
    };
    // DeepGEMM `SymmBuffer.buffer.nbytes` (reference dump r2_t4096, 152 SMs).
    try std.testing.expectEqual(@as(u64, 680837504), workspaceBytes(dsv41, 152));
    try std.testing.expectEqual(@as(u32, 5760), dsv41.numMaxTokensPerRankAligned());
    var vllm = dsv41;
    vllm.num_max_tokens_per_rank = 8192;
    try std.testing.expectEqual(@as(u32, 9600), vllm.numMaxTokensPerRankAligned());
}
