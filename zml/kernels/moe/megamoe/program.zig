//! Host side of the MegaMoE `cute.Program`: what DeepGEMM's `sm100_fp8_fp4_mega_moe.hpp` does
//! before `launch_kernel`, plus the staging the caller does before calling it (vLLM's
//! `prepare_megamoe_inputs`, DeepGEMM's test).
//!
//! `emitProgram` emits the whole program: kernel #0 (`staging.zig`: the inputs into this
//! rank's symmetric buffer) and kernel #1 (`kernel.zig`: `sm100_fp8_fp4_mega_moe_impl`), then
//! the host function. The host function builds the 18 tensor maps (`tma.Map` over this rank's
//! symmetric buffer and the weights, `layout.TensorMaps` parameters), the `SymBuffer` values
//! (this rank's buffer and every peer's offset from it), and launches kernel #0 (no PDL), then
//! kernel #1 with DeepGEMM's launch configuration: grid `kNumSMs`, cluster (2, 1, 1),
//! `kNumThreads` threads, the kernel's shared memory, PDL (vLLM enables it).
//!
//! Host-function parameters (`params`): the custom call's operands then results (`Param`), and
//! for `Workspace.symmetric` what XLA appends (`symmetric_workspace` backend config): every
//! rank's workspace base in rank order, then the rank (i32, the partition id).
const std = @import("std");

const cute = @import("kernels/cute/builder");

const kernel = @import("kernel.zig");
const layout = @import("layout.zig");
const staging = @import("staging.zig");

const B = cute.Builder;
const V = cute.Value;

/// A MegaMoE instantiation: model shape, ranks, per-rank token count of the call (selects
/// BLOCK_M like DeepGEMM), SM count, routed weight type, the activation template arguments.
pub const Instance = struct {
    num_ranks: u32 = 2,
    num_experts: u32 = 384,
    num_topk: u32 = 6,
    hidden: u32 = 5120,
    intermediate_hidden: u32 = 2304,
    num_shared_experts: u32 = 1,
    /// As requested (`deep_gemm.mega.SymmBuffer` aligns it, `shape()`).
    num_max_tokens_per_rank: u32 = 8192,
    num_tokens: u32 = 4096,
    num_sms: u32 = 152,
    weight_type: layout.WeightType = .fp4_unpacked,
    activation_clamp: f32 = 10.0,
    fast_math: bool = true,
    timeout_cycles: i64 = @import("device.zig").default_timeout_cycles,

    pub fn shape(self: Instance) layout.Shape {
        return .{
            .num_ranks = self.num_ranks,
            .num_experts = self.num_experts,
            .num_topk = self.num_topk,
            .hidden = self.hidden,
            .intermediate_hidden = self.intermediate_hidden,
            .num_shared_experts = self.num_shared_experts,
            .num_max_tokens_per_rank = layout.alignTokens(self.num_max_tokens_per_rank),
        };
    }

    /// `get_symm_buffer_size_for_mega_moe`'s layout for this shape and SM count.
    pub fn symmLayout(self: Instance) layout.SymmBufferLayout {
        return .init(self.shape(), self.num_sms);
    }

    /// The kernel config (DeepGEMM's heuristics) with DeepGEMM's default weight layout.
    pub fn config(self: Instance) kernel.Config {
        const s = self.shape();
        const symm = self.symmLayout();
        const k: layout.Kernel = .init(s, layout.config(s, self.num_tokens, symm), self.num_sms, self.weight_type);
        const weights: layout.DefaultWeights = .init(s, self.weight_type);
        return .{
            .kernel = k,
            .maps = .init(k, symm.buffer, weights.tensors(s.num_shared_experts > 0)),
            .activation_clamp = self.activation_clamp,
            .fast_math = self.fast_math,
            .timeout_cycles = self.timeout_cycles,
        };
    }

    /// The staging kernel's config for this call.
    pub fn stagingConfig(self: Instance, inputs: staging.Inputs) staging.Config {
        return .init(self.config().kernel, self.symmLayout().buffer, inputs, self.num_tokens);
    }
};

/// Where the symmetric buffer lives.
pub const Symmetric = union(enum) {
    /// Single rank: this buffer (a global pointer); `offsets = {0}`, `rank = 0`.
    single: V,
    /// Every rank's buffer (global pointers, rank order) and this rank's index (i32).
    ranks: struct { buffers: []const V, rank: V },
};

/// `SymBuffer(buffer_ptrs, rank_idx)`: `base = buffers[rank]` (i64), `offsets[i] = buffers[i] -
/// base`, `rank` (i32).
pub const SymValues = struct { base: V, offsets: []V, rank: V };

pub fn symValues(b: *B, num_ranks: u32, symmetric: Symmetric) SymValues {
    const R = num_ranks;
    const offsets = b.arena.allocator().alloc(V, R) catch @panic("OOM");
    switch (symmetric) {
        .single => |ptr| {
            std.debug.assert(R == 1);
            offsets[0] = b.cst(.i64, 0);
            return .{ .base = b.ptrToInt(ptr, .i64), .offsets = offsets, .rank = b.cst(.i32, 0) };
        },
        .ranks => |r| {
            std.debug.assert(r.buffers.len == R);
            const addrs = b.arena.allocator().alloc(V, R) catch @panic("OOM");
            for (addrs, r.buffers) |*a, p| a.* = b.ptrToInt(p, .i64);
            var base = addrs[0];
            for (addrs[1..], 1..) |a, i| base = b.select(r.rank.eq(@as(i32, @intCast(i))), a, base);
            for (offsets, addrs) |*o, a| o.* = a.sub(base);
            return .{ .base = base, .offsets = offsets, .rank = r.rank };
        },
    }
}

/// The host function's values the MegaMoE launch needs (global pointers unless noted).
pub const HostArgs = struct {
    /// `[num_tokens, hidden]` bf16 output.
    y: V,
    symmetric: Symmetric,
    /// Weight tensors in `layout.TensorMapSpec.Base` order (`l1_weights` .. `shared_l2_weights_sf`).
    /// Without a shared expert, the shared entries are unused (their maps copy the routed ones).
    weights: [8]V,
    /// `[kNumSMs]` u8 SM locality domains.
    sm_locality_domains: V,
    /// `num_tokens` (i32).
    num_tokens: V,
    /// `int*` cumulative per-expert receive counts (i64 address), 0 for `nullptr`.
    cumulative_local_expert_recv_stats: ?V = null,
};

fn baseOf(b: *B, args: HostArgs, base: layout.TensorMapSpec.Base, local: V) V {
    return switch (base) {
        .symm_buffer => local,
        inline else => |w| b.ptrToInt(args.weights[@backingInt(w) - 1], .i64),
    };
}

/// Launches `kernel_name` (emitted by `kernel.emit(b, cfg, kernel_name)`) from the current
/// host function; returns the launch status (i32).
pub fn emitLaunch(b: *B, cfg: kernel.Config, kernel_name: []const u8, args: HostArgs) V {
    const k = cfg.kernel;
    const R = k.shape.num_ranks;
    const sym = symValues(b, R, args.symmetric);

    // The 18 tensor maps.
    const maps = kernel.tmaMaps(b, cfg);
    const atoms = b.arena.allocator().alloc(V, maps.len) catch @panic("OOM");
    inline for (layout.TensorMaps.names, 0..) |name, i| {
        const spec: layout.TensorMapSpec = @field(cfg.maps, name);
        const base = baseOf(b, args, spec.base, sym.base);
        const ptr = b.intToPtr(if (spec.base_offset == 0) base else base.add(@as(i64, @intCast(spec.base_offset))), .i8, .gmem, 16);
        const atom = b.makeRawTmaAtom(ptr, maps[i].raw());
        atoms[i] = .{ .inner = atom.atom.inner, .kernel = b };
    }

    // Kernel arguments, in `kernel.argSpecs` order.
    const values = b.arena.allocator().alloc(V, kernel.Args.count(R)) catch @panic("OOM");
    values[kernel.Args.y] = args.y;
    values[kernel.Args.cumulative_local_expert_recv_stats] = args.cumulative_local_expert_recv_stats orelse b.cst(.i64, 0);
    values[kernel.Args.num_tokens] = args.num_tokens;
    values[kernel.Args.sym_base] = sym.base;
    for (sym.offsets, 0..) |o, r| values[kernel.Args.sym_offsets + r] = o;
    values[kernel.Args.symRank(R)] = sym.rank;
    for (atoms, 0..) |a, i| values[kernel.Args.map(R, i)] = a;
    values[kernel.Args.smLocalityDomains(R)] = args.sm_locality_domains;

    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.cst(.i32, @as(i32, @intCast(k.num_sms))), one, one },
        .block = .{ b.cst(.i32, @as(i32, @intCast(kernel.numThreads(cfg)))), one, one },
        .dynamic_smem = b.kernelSmemSize(kernel_name),
        .stream = b.cudaStream(),
        .cluster = .{ b.cst(.i32, 2), one, one },
        .use_pdl = true,
    });
    return b.cudaResultStatus(b.launchEx(kernel_name, config, values));
}

// ==================== the whole Program ====================

pub const InputKind = staging.Inputs;

pub const WorkspaceKind = enum {
    /// The workspace is a custom-call operand aliased to the `workspace_out` result (one rank).
    operand,
    /// XLA's symmetric workspace (Design A): `R` workspace pointers and the rank follow the
    /// custom call's buffers.
    symmetric,
};

/// The Program's configuration (`zml.kernel.cute.Program`'s `Config`).
pub const Spec = struct {
    instance: Instance,
    inputs: InputKind,
    workspace: WorkspaceKind,
};

/// Custom-call operands then results, in host-parameter order (`params`).
pub const Param = enum {
    // Operands.
    x,
    x_sf,
    topk_idx,
    topk_weights,
    l1_w,
    l1_sf,
    l2_w,
    l2_sf,
    shared_l1_w,
    shared_l1_sf,
    shared_l2_w,
    shared_l2_sf,
    sm_locality_domains,
    workspace,
    // Results.
    y,
    workspace_out,

    pub fn isResult(self: Param) bool {
        return @backingInt(self) >= @backingInt(Param.y);
    }

    /// The pointer's element type in the host function (XLA checks it against the buffer's:
    /// byte buffers match any 8-bit type).
    pub fn dtype(self: Param, inputs: InputKind) cute.DType {
        return switch (self) {
            .x => if (inputs == .bf16) .bf16 else .i8,
            .x_sf, .l1_sf, .l2_sf, .shared_l1_sf, .shared_l2_sf => .i32,
            .topk_idx => if (inputs == .bf16) .i32 else .i64,
            .topk_weights => .f32,
            .l1_w, .l2_w, .shared_l1_w, .shared_l2_w, .sm_locality_domains, .workspace, .workspace_out => .i8,
            .y => .bf16,
        };
    }
};

/// The host function's buffer parameters (the custom call's operands, then results). Without a
/// shared expert the `shared_*` operands are still present (the caller passes the routed
/// tensors; the kernel does not read them).
pub fn params(comptime inputs: InputKind, comptime workspace: WorkspaceKind) []const Param {
    return comptime blk: {
        var out: []const Param = &.{};
        for (std.enums.values(Param)) |p| {
            if (p == .x_sf and inputs == .bf16) continue;
            if ((p == .workspace or p == .workspace_out) and workspace == .symmetric) continue;
            out = out ++ &[_]Param{p};
        }
        break :blk out;
    };
}

/// `params` names (operands, or results) for `zml.kernel.cute.Program`'s spec.
pub fn paramNames(comptime inputs: InputKind, comptime workspace: WorkspaceKind, comptime results: bool) []const [:0]const u8 {
    return comptime blk: {
        var out: []const [:0]const u8 = &.{};
        for (params(inputs, workspace)) |p| {
            if (p.isResult() == results) out = out ++ &[_][:0]const u8{@tagName(p)};
        }
        break :blk out;
    };
}

pub const megamoe_kernel_name = "megamoe_kernel";
pub const staging_kernel_name = "megamoe_staging_kernel";

/// Emits kernel #0, kernel #1 and the host function `host_name` of the program open in `b`.
pub fn emitProgram(b: *B, spec: Spec, host_name: []const u8) cute.FinishError!void {
    const inst = spec.instance;
    const cfg = inst.config();
    const R = cfg.kernel.shape.num_ranks;
    std.debug.assert(spec.workspace == .symmetric or R == 1);
    const scfg = inst.stagingConfig(spec.inputs);
    try kernel.emit(b, cfg, megamoe_kernel_name);
    try staging.emit(b, scfg, staging_kernel_name);

    b.beginFunction(host_name, .host);
    const list = switch (spec.inputs) {
        inline else => |i| switch (spec.workspace) {
            inline else => |w| params(i, w),
        },
    };
    const arena = b.arena.allocator();
    const num_ws: usize = if (spec.workspace == .symmetric) R + 1 else 0;
    const specs = arena.alloc(cute.ArgSpec, list.len + num_ws) catch @panic("OOM");
    for (specs[0..list.len], list) |*s, p| s.* = .{ .name = @tagName(p), .kind = .{ .ptr = p.dtype(spec.inputs) } };
    if (spec.workspace == .symmetric) {
        // `Builder.declareArgsWithSymmetricWorkspace`'s parameters (XLA's convention).
        for (specs[list.len..][0..R], 0..) |*s, r| {
            s.* = .{ .name = std.fmt.allocPrint(arena, "workspace{d}", .{r}) catch @panic("OOM"), .kind = .{ .ptr = .i8 } };
        }
        specs[list.len + R] = .{ .name = "rank", .kind = .{ .mlir_type = .{ .type_ = cute.DType.i32.toMlir(b.ctx) } } };
    }
    try b.declareArgList(specs);

    var at: [std.enums.values(Param).len]?V = @splat(null);
    for (list, 0..) |p, i| at[@backingInt(p)] = b.arg(i);
    const get = struct {
        fn f(a: []const ?V, p: Param) V {
            return a[@backingInt(p)].?;
        }
    }.f;

    const symmetric: Symmetric = switch (spec.workspace) {
        .operand => .{ .single = get(&at, .workspace_out) },
        .symmetric => blk: {
            const buffers = arena.alloc(V, R) catch @panic("OOM");
            for (buffers, 0..) |*w, r| w.* = b.arg(list.len + r);
            break :blk .{ .ranks = .{ .buffers = buffers, .rank = b.arg(list.len + R) } };
        },
    };

    // Kernel #0: staging into this rank's buffer.
    const sym = symValues(b, R, symmetric);
    const zero = b.cst(.i64, 0);
    const staged = staging.emitLaunch(b, scfg, staging_kernel_name, .{
        .x = b.ptrToInt(get(&at, .x), .i64),
        .x_sf = if (spec.inputs == .staged) b.ptrToInt(get(&at, .x_sf), .i64) else zero,
        .topk_idx = b.ptrToInt(get(&at, .topk_idx), .i64),
        .topk_weights = b.ptrToInt(get(&at, .topk_weights), .i64),
        .sym_base = sym.base,
    });

    // Kernel #1: MegaMoE.
    const launched = emitLaunch(b, cfg, megamoe_kernel_name, .{
        .y = get(&at, .y),
        .symmetric = symmetric,
        .weights = .{
            get(&at, .l1_w),        get(&at, .l1_sf),        get(&at, .l2_w),        get(&at, .l2_sf),
            get(&at, .shared_l1_w), get(&at, .shared_l1_sf), get(&at, .shared_l2_w), get(&at, .shared_l2_sf),
        },
        .sm_locality_domains = get(&at, .sm_locality_domains),
        .num_tokens = b.cst(.i32, @as(i32, @intCast(inst.num_tokens))),
    });
    b.returnHostStatus(b.select(staged.ne(0), staged, launched));
    b.endFunction(null);
}

test "host parameters" {
    try std.testing.expectEqual(@as(usize, 16), params(.staged, .operand).len);
    try std.testing.expectEqual(@as(usize, 13), params(.bf16, .symmetric).len);
    try std.testing.expectEqualStrings("x", paramNames(.bf16, .symmetric, false)[0]);
    try std.testing.expectEqualStrings("topk_idx", paramNames(.bf16, .symmetric, false)[1]);
    try std.testing.expectEqual(@as(usize, 1), paramNames(.bf16, .symmetric, true).len);
    try std.testing.expectEqualStrings("workspace_out", paramNames(.staged, .operand, true)[1]);
}
