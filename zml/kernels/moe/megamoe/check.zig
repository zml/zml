//! MegaMoE validation harness: runs `zml.moe.megamoe` on DeepGEMM reference dumps
//! (written by the reference harness `megamoe_ref.py`; layouts in `<dumps>/LAYOUTS.md`)
//! and compares `y` bit for bit with DeepGEMM's, checks that warmed replays stay
//! bit-identical to the first call, and times the op.
//!
//! A dump is `meta.json` plus, per rank, `rank<r>.json` and `rank<r>.safetensors`
//! (staged inputs, `x_bf16`, `y`, `sm_locality_domains`); weights live in the rank file
//! or in the `weights_file` (per rank) and `shared_weights_file` (all ranks) it names.
//! Tensors are saved in memory order.
//!
//! One device per rank: R = 1 runs a single-device program with a zeroed
//! `Workspace.operand` threaded through the calls; R > 1 runs one
//! `zml.ops.manualComputation` partition per rank on an `ep` mesh with
//! `Workspace.symmetric`, every per-rank tensor concatenated along its leading
//! axis and sharded by rank, and tensors common to all ranks replicated.
//!
//! Build (zml workspace root). `--config=remote` downloads nothing by default, so ask for
//! the binary and its runfiles to run it outside `bazel run`:
//!   bazel build --config=remote --config=release --@zml//platforms:cuda=true \
//!     --repo_env=RULES_ZIG_CACHE_PREFIX_LINUX=/tmp/benjamin-zig-cache \
//!     --remote_download_outputs=toplevel --build_runfile_links //zml/kernels/moe/megamoe:check
//! Run through the GPU wrapper (CUDA_VISIBLE_DEVICES must expose exactly R GPUs). Use the
//! release output path: `bazel-bin` follows the last build's configuration.
//!   B=bazel-out/aarch64-opt/bin/zml/kernels/moe/megamoe/check
//!   gpu_run.sh 0 $B --dump=<dumps>/r1_t512_s1
//!   gpu_run.sh 0,1 $B --dump=<dumps>/r2_t4096_s1 --verify-load
//!   gpu_run.sh 0 $B --dump=<dumps>/r2_t4096_s1 --ranks=1
//!     (plumbing only: the first R ranks' tensors with num_experts = E_local * R)
//! Unit tests: `bazel test ... //zml/kernels/moe/megamoe:check_test`.
//!
//! Timing is host wall time (zml exposes no device timer): a synchronous loop (each call
//! waits) and a pipelined batch (back-to-back calls, one wait), which approaches device time
//! once the kernel outlasts host dispatch (~50-100 us per call).
//!
//! Exit status: 0 when every rank's `y` matches bit for bit and replays are stable,
//! 2 on a mismatch, 1 on errors.
const std = @import("std");

const zml = @import("zml");
const megamoe = zml.moe.megamoe;

const log = std.log.scoped(.megamoe_check);

const Args = struct {
    pub const help =
        \\MegaMoE validation harness (bit-exact y against DeepGEMM dumps)
        \\  --dump=<dir>            reference dump directory (meta.json, rank<r>.json, rank<r>.safetensors)
        \\  --iters=20              timed calls (synchronous loop, then the same count pipelined)
        \\  --warmup=3              untimed calls after the first one
        \\  --replays=3             warmed calls compared bit for bit with the first call
        \\  --from-bf16             call megamoe.forward on x_bf16 instead of forwardStaged
        \\  --ranks=<R>             override the dump's rank count (R <= dump ranks; R < dump ranks is plumbing only)
        \\  --workspace-bytes=<n>   override the workspace operand size (R = 1)
        \\  --num-sms=<n>           override the SM count (default: the dump's)
        \\  --xla-dump=<dir>        XLA dump directory
        \\  --verify-load           read every loaded device shard back and compare it with the dump files
        \\  --verbose               print more mismatch coordinates and per-call timings
        \\
    ;
    dump: []const u8,
    iters: u32 = 20,
    warmup: u32 = 3,
    replays: u32 = 3,
    from_bf16: bool = false,
    ranks: ?u32 = null,
    workspace_bytes: ?u64 = null,
    num_sms: ?u32 = null,
    xla_dump: ?[]const u8 = null,
    verify_load: bool = false,
    verbose: bool = false,
};

/// `meta.json`: the reference harness arguments.
const Meta = struct {
    num_ranks: u32,
    hidden: u32,
    intermediate_hidden: u32,
    num_experts: u32,
    num_topk: u32,
    num_shared_experts: u32,
    num_max_tokens_per_rank: u32,
    activation_clamp: f32,
    fast_math: u32,
};

/// `rank<r>.json`: what DeepGEMM did on that rank. Covers both dump formats: the first one
/// (weights inside the rank file, `buffer_bytes`, `time_us`) and the current one (weights in
/// `weights_file` per rank plus a `shared_weights_file` common to all ranks).
const RankInfo = struct {
    rank: u32,
    num_tokens: u32,
    block_m: u32,
    deterministic: bool,
    num_max_tokens_per_rank_aligned: u32,
    num_sms: u32,
    /// Relative to the dump directory.
    weights_file: ?[]const u8 = null,
    /// Relative to the dump directory; the same file for every rank.
    shared_weights_file: ?[]const u8 = null,
    y_sha256: ?[]const u8 = null,
    buffer_bytes: ?u64 = null,
    symm_buffer_layout: ?struct { num_bytes: u64 } = null,
    time_us: ?f64 = null,
    timing: ?struct {
        kineto_us: ?f64 = null,
        graph: ?struct { moe_only: ?struct { us_per_call: f64 } = null } = null,
    } = null,

    /// DeepGEMM's `SymmBuffer.buffer.nbytes`.
    fn bufferBytes(self: RankInfo) ?u64 {
        if (self.symm_buffer_layout) |l| return l.num_bytes;
        return self.buffer_bytes;
    }

    fn kinetoUs(self: RankInfo) ?f64 {
        if (self.timing) |t| return t.kineto_us;
        return self.time_us;
    }

    fn graphUs(self: RankInfo) ?f64 {
        const t = self.timing orelse return null;
        const g = t.graph orelse return null;
        const m = g.moe_only orelse return null;
        return m.us_per_call;
    }
};

/// Per-rank tensors of one call, in memory order. The program sees the
/// concatenation of all ranks along the leading axis, sharded by rank.
const Operands = struct {
    /// `x_fp8` `[T, H]` u8, or `x_bf16` `[T, H]` bf16 with --from-bf16.
    x: zml.Tensor,
    /// `[T, H/128]` i32, staged calls only.
    x_sf: ?zml.Tensor,
    /// `[T, K]` i64, converted to i32 in the graph for `forward`.
    topk_idx: zml.Tensor,
    topk_weights: zml.Tensor,
    weights: megamoe.Weights,
    sm_locality_domains: zml.Tensor,
};

const workspace_name = "megamoe_check";

const Program = struct {
    problem: megamoe.Problem,
    from_bf16: bool,

    fn call(p: Program, ops: Operands, workspace: megamoe.Workspace) megamoe.Result {
        if (p.from_bf16) return megamoe.forward(p.problem, ops.x, ops.topk_idx.convert(.i32), ops.topk_weights, ops.weights, ops.sm_locality_domains, workspace);
        return megamoe.forwardStaged(p.problem, .{
            .x = ops.x,
            .x_sf = ops.x_sf.?,
            .topk_idx = ops.topk_idx,
            .topk_weights = ops.topk_weights,
        }, ops.weights, ops.sm_locality_domains, workspace);
    }

    /// R = 1: returns `{ y, workspace }`, the workspace aliased to its input.
    fn single(p: Program, ops: Operands, workspace: zml.Tensor) [2]zml.Tensor {
        const r = p.call(ops, .{ .operand = workspace });
        return .{ r.y, r.workspace.? };
    }

    const Partition = struct { program: Program, operands: Operands };

    /// R > 1: one manual partition per rank; `y` is sharded like `x`.
    fn multi(p: Program, ops: Operands) zml.Tensor {
        return zml.ops.manualComputation(partition, .{ .program = p, .operands = ops }, ops.x.shape().withDtype(.bf16));
    }

    fn partition(ctx: Partition, _: zml.Shape) zml.Tensor {
        return ctx.program.call(ctx.operands, .{ .symmetric = .{ .name = workspace_name } }).y;
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    var stdout_buffer: [16 * 1024]u8 = undefined;
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);
    const out = &stdout_file.interface;

    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const meta = try readJson(Meta, arena, io, try std.fs.path.join(arena, &.{ args.dump, "meta.json" }));
    const ranks = args.ranks orelse meta.num_ranks;
    if (ranks == 0 or ranks > meta.num_ranks) zml.stdx.flags.fatal("--ranks={d} but the dump has {d} ranks", .{ ranks, meta.num_ranks });
    const partial = ranks != meta.num_ranks;

    const infos = try arena.alloc(RankInfo, ranks);
    for (infos, 0..) |*info, r| {
        info.* = try readJson(RankInfo, arena, io, try std.fmt.allocPrint(arena, "{s}/rank{d}.json", .{ args.dump, r }));
        if (info.num_tokens != infos[0].num_tokens) zml.stdx.flags.fatal("rank {d} has {d} tokens, rank 0 has {d}: ranks with different token counts are not supported (static T per partition)", .{ r, info.num_tokens, infos[0].num_tokens });
    }
    var dump: Dump = try .open(allocator, arena, io, args.dump, infos);
    defer dump.deinit(allocator);
    const tokens = infos[0].num_tokens;
    const num_sms = args.num_sms orelse infos[0].num_sms;

    const experts_per_rank = @divExact(meta.num_experts, meta.num_ranks);
    const problem: megamoe.Problem = .{
        .num_ranks = ranks,
        .num_experts = experts_per_rank * ranks,
        .num_topk = meta.num_topk,
        .hidden = meta.hidden,
        .intermediate = meta.intermediate_hidden,
        .num_shared_experts = meta.num_shared_experts,
        .num_max_tokens_per_rank = meta.num_max_tokens_per_rank,
        .activation_clamp = meta.activation_clamp,
        .fast_math = meta.fast_math != 0,
    };
    problem.validate();
    if (problem.numMaxTokensPerRankAligned() != infos[0].num_max_tokens_per_rank_aligned)
        log.warn("aligned num_max_tokens_per_rank {d} differs from DeepGEMM's {d}", .{ problem.numMaxTokensPerRankAligned(), infos[0].num_max_tokens_per_rank_aligned });
    const required_bytes = megamoe.workspaceBytes(problem, num_sms);
    const workspace_bytes = args.workspace_bytes orelse required_bytes;

    try out.print("megamoe check: dump={s} ranks={d}{s} tokens/rank={d} mode={s} block_m(DeepGEMM)={d}\n", .{ args.dump, ranks, if (partial) " (PARTIAL: first ranks of a larger dump, y not expected to match)" else "", tokens, if (args.from_bf16) "forward(x_bf16)" else "forwardStaged", infos[0].block_m });
    try out.print("problem: R={d} E={d} (local {d}) K={d} H={d} I={d} S={d} max_tokens/rank={d} (aligned {d}) clamp={d} fast_math={} num_sms={d}\n", .{ problem.num_ranks, problem.num_experts, problem.numExpertsPerRank(), problem.num_topk, problem.hidden, problem.intermediate, problem.num_shared_experts, problem.num_max_tokens_per_rank, problem.numMaxTokensPerRankAligned(), problem.activation_clamp, problem.fast_math, num_sms });
    const deepgemm_bytes = infos[0].bufferBytes();
    const size_check = if (partial) "n/a (partial)" else if (deepgemm_bytes == null) "n/a" else if (required_bytes == deepgemm_bytes.?) "match" else "WORKSPACE-SIZE MISMATCH";
    try out.print("workspace: workspaceBytes={d} DeepGEMM buffer bytes={?d} ({s}){s}\n", .{ required_bytes, deepgemm_bytes, size_check, if (ranks == 1) try std.fmt.allocPrint(arena, "; operand bytes={d}", .{workspace_bytes}) else "" });
    try out.flush();

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    if (platform.devices.len != ranks) zml.stdx.flags.fatal("{d} ranks need exactly {d} visible devices, platform has {d} ({t}); set CUDA_VISIBLE_DEVICES", .{ ranks, ranks, platform.devices.len, platform.target });
    for (platform.devices, 0..) |device, i| {
        const cores = device.pjrt_desc.attribute(platform.pjrt_api, "core_count");
        const sms: i64 = if (cores) |c| (if (c == .int64) c.int64 else -1) else -1;
        try out.print("device {d}: {s}, {d} SMs{s}\n", .{ i, device.kind(), sms, if (sms != -1 and sms != num_sms) " (differs from --num-sms / the dump)" else "" });
    }
    try out.flush();

    const mesh: ?*const zml.Mesh = if (ranks > 1) try platform.registerMesh("ep", .mesh(.{ .ep = .high_bandwidth })) else null;

    // Shapes and sources: per-rank tensors are concatenated along their leading axis and
    // sharded by rank, tensors of a file common to all ranks are replicated.
    var loading: Loading = .{ .allocator = allocator, .arena = arena, .io = io, .platform = platform, .dump = &dump, .mesh = mesh };
    const shared = problem.num_shared_experts != 0;
    const has_domains = dump.ranks[0].contains("sm_locality_domains");
    const operands: Operands = .{
        .x = if (args.from_bf16) try loading.tensor("x", &.{"x_bf16"}) else try loading.tensor("x", &.{ "x", "x_fp8" }),
        .x_sf = if (args.from_bf16) null else try loading.tensor("x_sf", &.{"x_sf"}),
        .topk_idx = try loading.tensor("topk_idx", &.{"topk_idx"}),
        .topk_weights = try loading.tensor("topk_weights", &.{"topk_weights"}),
        .weights = .{
            .l1_w = try loading.tensor("l1_w", &.{"l1_w"}),
            .l1_sf = try loading.tensor("l1_sf", &.{"l1_sf"}),
            .l2_w = try loading.tensor("l2_w", &.{"l2_w"}),
            .l2_sf = try loading.tensor("l2_sf", &.{"l2_sf"}),
            .shared_l1_w = if (shared) try loading.tensor("shared_l1_w", &.{"shared_l1_w"}) else null,
            .shared_l1_sf = if (shared) try loading.tensor("shared_l1_sf", &.{"shared_l1_sf"}) else null,
            .shared_l2_w = if (shared) try loading.tensor("shared_l2_w", &.{"shared_l2_w"}) else null,
            .shared_l2_sf = if (shared) try loading.tensor("shared_l2_sf", &.{"shared_l2_sf"}) else null,
        },
        .sm_locality_domains = if (has_domains)
            try loading.tensor("sm_locality_domains", &.{"sm_locality_domains"})
        else
            .fromShape(loading.global(.init(.{num_sms}, .u8))),
    };
    if (operands.sm_locality_domains.count() != @as(usize, num_sms) * (if (mesh != null) ranks else 1))
        zml.stdx.flags.fatal("sm_locality_domains is {f}, expected {d} SMs per rank", .{ operands.sm_locality_domains, num_sms });

    // Device buffers, loaded concurrently with direct DMA writers.
    const load_start: std.Io.Timestamp = .now(io, .awake);
    var buffers = try zml.mem.bufferize(allocator, Operands, &operands);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Operands, &buffers);
    {
        const w = &buffers.weights;
        var targets: std.ArrayList(Loading.Target) = .empty;
        defer targets.deinit(allocator);
        try targets.appendSlice(allocator, &.{
            .{ .name = "x", .buffer = &buffers.x },
            .{ .name = "topk_idx", .buffer = &buffers.topk_idx },
            .{ .name = "topk_weights", .buffer = &buffers.topk_weights },
            .{ .name = "l1_w", .buffer = &w.l1_w },
            .{ .name = "l1_sf", .buffer = &w.l1_sf },
            .{ .name = "l2_w", .buffer = &w.l2_w },
            .{ .name = "l2_sf", .buffer = &w.l2_sf },
        });
        if (buffers.x_sf) |*b| try targets.append(allocator, .{ .name = "x_sf", .buffer = b });
        if (w.shared_l1_w) |*b| try targets.append(allocator, .{ .name = "shared_l1_w", .buffer = b });
        if (w.shared_l1_sf) |*b| try targets.append(allocator, .{ .name = "shared_l1_sf", .buffer = b });
        if (w.shared_l2_w) |*b| try targets.append(allocator, .{ .name = "shared_l2_w", .buffer = b });
        if (w.shared_l2_sf) |*b| try targets.append(allocator, .{ .name = "shared_l2_sf", .buffer = b });
        if (has_domains) try targets.append(allocator, .{ .name = "sm_locality_domains", .buffer = &buffers.sm_locality_domains });
        loading.loadAll(targets.items) catch |err| zml.stdx.flags.fatal("loading {s} failed: {t}", .{ args.dump, err });
        const load_s = seconds(load_start.untilNow(io, .awake));
        try out.print("load: {d:.2} GiB in {d:.2} s ({d:.2} GiB/s host->devices)\n", .{ gib(loading.bytes), load_s, gib(loading.bytes) / load_s });
        if (args.verify_load) {
            for (targets.items) |target| {
                const ok = try loading.verify(target);
                try out.print("verify-load {s}: {s}\n", .{ target.name, if (ok) "ok" else "MISMATCH" });
                if (!ok) zml.stdx.flags.fatal("device copy of {s} differs from the dump", .{target.name});
            }
        }
        try out.flush();
    }
    if (!has_domains) {
        // DeepGEMM's `get_even_sm_locality_domains` with 2 domains (TPC index % 2), used when MLOPart
        // localization is unavailable. It only steers scheduling, not numerics.
        const domains = try arena.alloc(u8, operands.sm_locality_domains.count());
        for (domains, 0..) |*d, i| d.* = @intCast(((i % num_sms) / 2) % 2);
        buffers.sm_locality_domains = try .fromBytes(io, platform, operands.sm_locality_domains.shape(), domains);
    }
    try out.print("sm_locality_domains: {s}\n", .{if (has_domains) "from dump" else "even map (TPC % 2), not in dump"});

    // Reference y, rank-major like the program's output.
    const y_shape = operands.x.shape().withDtype(.bf16);
    const expected = try allocator.alloc(u8, y_shape.byteSize());
    defer allocator.free(expected);
    const rank_bytes = expected.len / ranks;
    for (0..ranks) |r| {
        const t = dump.ranks[r].get("y") orelse zml.stdx.flags.fatal("rank {d} has no y", .{r});
        if (t.shape.byteSize() != rank_bytes or t.shape.dtype() != .bf16) zml.stdx.flags.fatal("rank {d} y is {f}, expected {d} bytes of bf16", .{ r, t.shape, rank_bytes });
        const dst = expected[r * rank_bytes ..][0..rank_bytes];
        try readTensor(io, t, dst);
        if (infos[r].y_sha256) |want| {
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(dst, &digest, .{});
            if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), want)) zml.stdx.flags.fatal("rank {d} y does not match its y_sha256 {s}", .{ r, want });
        }
    }

    const program: Program = .{ .problem = problem, .from_bf16 = args.from_bf16 };
    const compile_start: std.Io.Timestamp = .now(io, .awake);
    const meshes: []const *const zml.Mesh = if (mesh) |m| &.{m} else &.{};
    const compile_opts: zml.Compiler.Options = .{ .meshes = meshes, .program_name = "megamoe_check", .xla_dump_to = args.xla_dump };
    const exe = if (ranks == 1)
        try platform.compileFn(allocator, io, Program.single, .{ program, operands, zml.Tensor.init(.{@as(i64, @intCast(workspace_bytes))}, .u8) }, compile_opts)
    else
        try platform.compileFn(allocator, io, Program.multi, .{ program, operands }, compile_opts);
    defer exe.deinit();
    try out.print("compile: {d:.2} s\n", .{seconds(compile_start.untilNow(io, .awake))});
    try out.flush();

    var calls: Calls = .{ .allocator = allocator, .io = io, .operands = &buffers, .workspace = null };
    calls.runner = try exe.runner(allocator);
    defer calls.runner.deinit(allocator);
    if (ranks == 1) {
        // Zeroed once, then threaded through every call like DeepGEMM's SymmBuffer.
        const zeros = try allocator.alloc(u8, workspace_bytes);
        defer allocator.free(zeros);
        @memset(zeros, 0);
        calls.workspace = try .fromBytes(io, platform, .init(.{@as(i64, @intCast(workspace_bytes))}, .u8), zeros);
    }
    defer if (calls.workspace) |*w| w.deinit();

    // First call: the bit-exact comparison.
    const first = try calls.runToHost(arena);
    var all_match = true;
    for (0..ranks) |r| {
        const comparison = compare(first[r * rank_bytes ..][0..rank_bytes], expected[r * rank_bytes ..][0..rank_bytes], problem.hidden);
        all_match = all_match and comparison.mismatched == 0;
        try comparison.report(out, r, if (args.verbose) 16 else 4);
    }
    try out.flush();

    // Warmup, then synchronous timing (each call waits for completion).
    for (0..args.warmup) |_| try calls.runAndDrop(true);
    const sync_ns = try arena.alloc(f64, args.iters);
    for (sync_ns) |*ns| {
        const start: std.Io.Timestamp = .now(io, .awake);
        try calls.runAndDrop(true);
        ns.* = @floatFromInt(start.untilNow(io, .awake).toNanoseconds());
    }
    if (args.verbose) for (sync_ns, 0..) |ns, i| try out.print("  call {d}: {d:.1} us\n", .{ i, ns / 1e3 });

    // Pipelined timing: back-to-back dispatch, one wait at the end. The first batch is untimed:
    // it grows the device allocator to hold all outputs in flight.
    var pipelined: Pipelined = undefined;
    for (0..2) |_| pipelined = try calls.pipelined(args.iters);
    if (args.verbose) try out.print("  pipelined: dispatch {d:.1} us, total {d:.1} us for {d} calls\n", .{ pipelined.dispatch_ns / 1e3, pipelined.total_ns / 1e3, args.iters });
    const pipelined_ns = pipelined.total_ns;

    // Warmed replays must stay bit-identical to the first call.
    var stable: u32 = 0;
    for (0..args.replays) |_| {
        const again = try calls.runToHost(arena);
        if (std.mem.eql(u8, again, first)) stable += 1;
    }
    try out.print("replay: {d}/{d} warmed calls bit-identical to the first call{s}\n", .{ stable, args.replays, if (stable == args.replays) "" else " (REPLAY MISMATCH)" });

    if (args.iters != 0) {
        const s = summarize(sync_ns);
        try out.print("timing (host wall time; zml exposes no device timer): sync median {d:.1} us, min {d:.1} us, mean {d:.1} us per call ({d} calls, run+wait each); pipelined {d:.1} us per call ({d} back-to-back calls, one wait)\n", .{ s.median / 1e3, s.min / 1e3, s.mean / 1e3, args.iters, pipelined_ns / 1e3 / @as(f64, @floatFromInt(args.iters)), args.iters });
    }
    for (infos) |info| try out.print("DeepGEMM rank {d}: kineto {?d:.1} us per call, CUDA graph {?d:.1} us per call, deterministic={}\n", .{ info.rank, info.kinetoUs(), info.graphUs(), info.deterministic });

    const ok = all_match and stable == args.replays;
    try out.print("RESULT: {s}\n", .{if (ok) "MATCH" else "MISMATCH"});
    try out.flush();
    if (!ok) std.process.exit(2);
}

const Pipelined = struct { dispatch_ns: f64, total_ns: f64 };

/// Calls of the compiled program with the R = 1 workspace threaded through.
const Calls = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    operands: *const zml.Bufferized(Operands),
    workspace: ?zml.Buffer,
    runner: zml.Exe.Runner = undefined,

    /// Returns `y`; the caller owns it.
    fn run(self: *Calls, wait: bool) zml.Buffer {
        var y: zml.Buffer = undefined;
        if (self.workspace) |*w| {
            // The workspace input is donated: its output replaces the handle.
            self.runner.run(self.io, .{ self.operands.*, w.* }, .{ &y, w }, .{ .wait = wait });
        } else {
            self.runner.run(self.io, .{self.operands.*}, .{&y}, .{ .wait = wait });
        }
        return y;
    }

    /// `iters` calls without waiting, then one wait for the last output.
    fn pipelined(self: *Calls, iters: u32) !Pipelined {
        const ys = try self.allocator.alloc(zml.Buffer, iters);
        defer self.allocator.free(ys);
        var launched: usize = 0;
        defer for (ys[0..launched]) |*y| y.deinit();
        const start: std.Io.Timestamp = .now(self.io, .awake);
        for (ys) |*y| {
            y.* = self.run(false);
            launched += 1;
        }
        const dispatch: f64 = @floatFromInt(start.untilNow(self.io, .awake).toNanoseconds());
        if (launched != 0) try ys[launched - 1].await(self.io);
        if (self.workspace) |w| try w.await(self.io);
        return .{ .dispatch_ns = dispatch, .total_ns = @floatFromInt(start.untilNow(self.io, .awake).toNanoseconds()) };
    }

    fn runAndDrop(self: *Calls, wait: bool) !void {
        var y = self.run(wait);
        y.deinit();
    }

    fn runToHost(self: *Calls, arena: std.mem.Allocator) ![]const u8 {
        var y = self.run(true);
        defer y.deinit();
        const host = try y.toSliceAlloc(self.allocator, self.io);
        defer host.free(self.allocator);
        return try arena.dupe(u8, host.constData());
    }
};

/// The dump's safetensors files: per rank, `rank<r>.safetensors` and its weights file; plus the
/// shared-expert weights file common to every rank.
const Dump = struct {
    const Tensors = std.StringArrayHashMapUnmanaged(zml.safetensors.Tensor);

    registries: std.ArrayList(zml.safetensors.TensorRegistry) = .empty,
    ranks: []Tensors,
    common: Tensors = .empty,

    fn open(allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, dir: []const u8, infos: []const RankInfo) !Dump {
        var dump: Dump = .{ .ranks = try arena.alloc(Tensors, infos.len) };
        errdefer dump.deinit(allocator);
        @memset(dump.ranks, .empty);
        for (infos, dump.ranks, 0..) |info, *tensors, r| {
            try dump.add(allocator, arena, io, try std.fmt.allocPrint(arena, "{s}/rank{d}.safetensors", .{ dir, r }), tensors);
            if (info.weights_file) |file| try dump.add(allocator, arena, io, try std.fs.path.join(arena, &.{ dir, file }), tensors);
            if (info.shared_weights_file) |file| {
                if (!std.mem.eql(u8, file, infos[0].shared_weights_file orelse "")) zml.stdx.flags.fatal("rank {d} uses shared weights {s}, rank 0 {?s}", .{ r, file, infos[0].shared_weights_file });
                if (r == 0) try dump.add(allocator, arena, io, try std.fs.path.join(arena, &.{ dir, file }), &dump.common);
            }
        }
        return dump;
    }

    fn add(dump: *Dump, allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, path: []const u8, tensors: *Tensors) !void {
        try dump.registries.append(allocator, try openRegistry(allocator, io, path));
        var it = dump.registries.items[dump.registries.items.len - 1].iterator();
        while (it.next()) |entry| {
            const gop = try tensors.getOrPut(arena, entry.key_ptr.*);
            if (gop.found_existing) zml.stdx.flags.fatal("tensor {s} appears twice ({s})", .{ entry.key_ptr.*, path });
            gop.value_ptr.* = entry.value_ptr.*;
        }
    }

    fn deinit(dump: *Dump, allocator: std.mem.Allocator) void {
        for (dump.registries.items) |*r| r.deinit();
        dump.registries.deinit(allocator);
    }
};

/// Global shapes and DMA loading of dump tensors.
const Loading = struct {
    allocator: std.mem.Allocator,
    /// Owns `sources`.
    arena: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    dump: *const Dump,
    mesh: ?*const zml.Mesh,
    sources: std.StringArrayHashMapUnmanaged(Source) = .empty,
    bytes: u64 = 0,

    const Source = union(enum) {
        /// One tensor per rank, concatenated along the leading axis and sharded by rank.
        per_rank: []const zml.safetensors.Tensor,
        /// One tensor for all ranks, replicated.
        common: zml.safetensors.Tensor,
    };

    const Target = struct { name: []const u8, buffer: *zml.Buffer };

    fn ranks(self: *const Loading) usize {
        return self.dump.ranks.len;
    }

    /// The per-rank shape with the leading axis multiplied by R, sharded by rank.
    fn global(self: *const Loading, local: zml.Shape) zml.Shape {
        const m = self.mesh orelse return local;
        const axis = m.resolveLogicalAxis("ep").?;
        return local.setDim(0, local.dim(0) * @as(i64, @intCast(self.ranks()))).withPartitioning(m, zml.Sharding.Partitioning.init(&.{.sharded(@intCast(axis))}));
    }

    /// The program operand for the first of `aliases` found in the dump, recorded as `name`.
    fn tensor(self: *Loading, name: []const u8, aliases: []const []const u8) !zml.Tensor {
        for (aliases) |alias| {
            if (self.dump.ranks[0].get(alias)) |t0| {
                const per_rank = try self.arena.alloc(zml.safetensors.Tensor, self.ranks());
                for (self.dump.ranks, per_rank, 0..) |tensors, *t, r| {
                    t.* = tensors.get(alias) orelse zml.stdx.flags.fatal("rank {d} has no tensor {s}", .{ r, alias });
                    if (!t.shape.eql(t0.shape)) zml.stdx.flags.fatal("{s}: rank {d} has {f}, rank 0 has {f}", .{ alias, r, t.shape, t0.shape });
                }
                try self.sources.put(self.arena, name, .{ .per_rank = per_rank });
                return .fromShape(self.global(t0.shape));
            }
            if (self.dump.common.get(alias)) |t| {
                try self.sources.put(self.arena, name, .{ .common = t });
                return .fromShape(t.shape);
            }
        }
        zml.stdx.flags.fatal("dump has no tensor {s} (looked for {any})", .{ name, aliases });
    }

    const Task = struct {
        loading: *Loading,
        loader: *zml.io.Loader,
        name: []const u8,
        source: Source,
        buffer: *zml.Buffer,
        err: ?anyerror = null,

        fn bytes(task: Task) u64 {
            return switch (task.source) {
                .per_rank => |ts| ts[0].byteSize() * ts.len,
                .common => |t| t.byteSize(),
            };
        }
    };

    fn loadAll(self: *Loading, targets: []const Target) !void {
        var loader: zml.io.Loader = try .init(self.allocator, self.platform, .{ .parallelism = 4, .dma_chunks = 8, .dma_chunk_size = 64 * zml.MiB });
        defer loader.deinit();
        const tasks = try self.allocator.alloc(Task, targets.len);
        defer self.allocator.free(tasks);
        for (tasks, targets) |*task, target| task.* = .{ .loading = self, .loader = &loader, .name = target.name, .source = self.sources.get(target.name).?, .buffer = target.buffer };

        // Largest first, so the long transfers start early.
        std.mem.sort(Task, tasks, {}, struct {
            fn larger(_: void, a: Task, b: Task) bool {
                return a.bytes() > b.bytes();
            }
        }.larger);
        for (tasks) |*task| loader.group.async(self.io, loadTask, .{task});
        try loader.await(self.io);
        for (tasks) |task| {
            if (task.err) |err| {
                log.err("loading {s} failed: {t}", .{ task.name, err });
                return err;
            }
            self.bytes += task.bytes();
        }
    }

    fn loadTask(task: *Task) void {
        task.loading.loadOne(task.loader, task.source, task.buffer) catch |err| {
            task.err = err;
        };
    }

    /// Reads every device shard of a loaded target back and compares it with its file bytes.
    fn verify(self: *Loading, target: Target) !bool {
        const source = self.sources.get(target.name).?;
        const buffer = target.buffer.*;
        const placement: zml.Sharding.Placement = try .init(&buffer.shape());
        const local = placement.localShape(buffer.shape());
        const actual = try self.allocator.alloc(u8, local.byteSize());
        defer self.allocator.free(actual);
        const expected = try self.allocator.alloc(u8, local.byteSize());
        defer self.allocator.free(expected);
        for (0..buffer.numShards()) |device_index| {
            const file_tensor = switch (source) {
                .common => |t| t,
                .per_rank => |ts| ts[@intCast(@divExact(placement.slices(device_index, buffer.shape()).get(0).start, local.dim(0)))],
            };
            if (file_tensor.byteSize() != actual.len) return false;
            try readTensor(self.io, file_tensor, expected);
            const event = try buffer._shards.get(device_index).toHostBuffer(self.platform.pjrt_api, actual);
            if (event) |e| {
                defer e.deinit(self.platform.pjrt_api);
                try e.await(self.platform.pjrt_api, self.io);
            }
            if (!std.mem.eql(u8, actual, expected)) return false;
        }
        return true;
    }

    /// Streams the source file regions into one device buffer. Per-rank tensors become the
    /// consecutive slices of the leading axis, which the direct writer routes to the device
    /// holding each slice; common tensors are mirrored to every device.
    fn loadOne(self: *Loading, loader: *zml.io.Loader, source: Source, buffer: *zml.Buffer) !void {
        const one = [1]zml.safetensors.Tensor{if (source == .common) source.common else undefined};
        const parts: []const zml.safetensors.Tensor = switch (source) {
            .per_rank => |ts| ts,
            .common => &one,
        };
        const shape: zml.Shape = switch (source) {
            .per_rank => |ts| self.global(ts[0].shape),
            .common => |t| t.shape,
        };
        var writer: zml.io.MemoryWriter = try .init(self.allocator, self.io, self.platform, loader.pinned_buffer_pools, loader.dma_allocators, loader.dma_chunk_size, shape, buffer, .default);
        defer writer.deinit(self.allocator);
        for (parts) |part| {
            var reader = try part.reader(self.io, &.{}, .{});
            defer reader.deinit();
            _ = try reader.interface.streamRemaining(writer.interface());
        }
        try writer.interface().flush();
    }
};

fn openRegistry(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !zml.safetensors.TensorRegistry {
    const file = std.Io.Dir.openFile(.cwd(), io, path, .{ .mode = .read_only }) catch |err| zml.stdx.flags.fatal("cannot open {s}: {t}", .{ path, err });
    defer file.close(io);
    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    errdefer registry.deinit();
    try zml.safetensors.parseSafetensors(allocator, io, &registry, file);
    return registry;
}

fn readTensor(io: std.Io, t: zml.safetensors.Tensor, dst: []u8) !void {
    std.debug.assert(dst.len == t.byteSize());
    const file = try std.Io.Dir.openFile(.cwd(), io, t.file_uri, .{ .mode = .read_only });
    defer file.close(io);
    if (try file.readPositionalAll(io, dst, t.offset) != dst.len) return error.EndOfStream;
}

fn readJson(comptime T: type, arena: std.mem.Allocator, io: std.Io, path: []const u8) !T {
    const bytes = zml.stdx.Io.Dir.readFileAlloc(.cwd(), io, path, arena, .limited(16 * zml.MiB)) catch |err| zml.stdx.flags.fatal("cannot read {s}: {t}", .{ path, err });
    return std.json.parseFromSliceLeaky(T, arena, bytes, .{ .ignore_unknown_fields = true });
}

fn seconds(d: std.Io.Duration) f64 {
    return @as(f64, @floatFromInt(d.toNanoseconds())) / 1e9;
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(zml.GiB));
}

const Summary = struct { median: f64, min: f64, mean: f64 };

fn summarize(samples: []f64) Summary {
    std.mem.sort(f64, samples, {}, std.sort.asc(f64));
    var sum: f64 = 0;
    for (samples) |s| sum += s;
    const n = samples.len;
    const median = if (n % 2 == 1) samples[n / 2] else (samples[n / 2 - 1] + samples[n / 2]) / 2;
    return .{ .median = median, .min = samples[0], .mean = sum / @as(f64, @floatFromInt(n)) };
}

const max_reported_mismatches = 16;

/// Bitwise comparison of one rank's `y` (bf16, `[T, H]`).
const Comparison = struct {
    elements: usize = 0,
    mismatched: usize = 0,
    nonfinite_actual: usize = 0,
    nonfinite_expected: usize = 0,
    /// Over elements where both are finite.
    max_abs: f64 = 0,
    max_abs_at: ?usize = null,
    /// Over elements where both are finite and the expected value is nonzero.
    max_rel: f64 = 0,
    max_rel_at: ?usize = null,
    first: [max_reported_mismatches]usize = undefined,
    first_len: usize = 0,
    hidden: usize,
    sha_actual: [32]u8,
    sha_expected: [32]u8,
    actual: []align(1) const u16,
    expected: []align(1) const u16,

    fn report(c: Comparison, out: *std.Io.Writer, rank: usize, max_lines: usize) !void {
        const status = if (c.mismatched == 0) "MATCH" else "MISMATCH";
        try out.print("rank {d}: {s} {d}/{d} elements differ bitwise (nonfinite actual {d}, expected {d}); max_abs {e:.4}", .{ rank, status, c.mismatched, c.elements, c.nonfinite_actual, c.nonfinite_expected, c.max_abs });
        if (c.max_abs_at) |i| try out.print(" at (t={d},h={d})", .{ i / c.hidden, i % c.hidden });
        try out.print("; max_rel {e:.4}", .{c.max_rel});
        if (c.max_rel_at) |i| try out.print(" at (t={d},h={d})", .{ i / c.hidden, i % c.hidden });
        try out.print("\n  sha256 actual={s} expected={s}\n", .{ std.fmt.bytesToHex(c.sha_actual, .lower), std.fmt.bytesToHex(c.sha_expected, .lower) });
        for (c.first[0..@min(c.first_len, max_lines)]) |i| {
            try out.print("  mismatch (t={d},h={d}): actual {e:.6} (0x{x:0>4}) expected {e:.6} (0x{x:0>4})\n", .{ i / c.hidden, i % c.hidden, bf16ToF32(c.actual[i]), c.actual[i], bf16ToF32(c.expected[i]), c.expected[i] });
        }
    }
};

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

fn compare(actual_bytes: []const u8, expected_bytes: []const u8, hidden: usize) Comparison {
    std.debug.assert(actual_bytes.len == expected_bytes.len);
    const actual = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, actual_bytes));
    const expected = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, expected_bytes));
    var c: Comparison = .{ .hidden = hidden, .sha_actual = undefined, .sha_expected = undefined, .actual = actual, .expected = expected };
    std.crypto.hash.sha2.Sha256.hash(actual_bytes, &c.sha_actual, .{});
    std.crypto.hash.sha2.Sha256.hash(expected_bytes, &c.sha_expected, .{});
    c.elements = actual.len;
    for (actual, expected, 0..) |a_bits, e_bits, i| {
        if (a_bits == e_bits) continue;
        c.mismatched += 1;
        if (c.first_len < max_reported_mismatches) {
            c.first[c.first_len] = i;
            c.first_len += 1;
        }
        const a = bf16ToF32(a_bits);
        const e = bf16ToF32(e_bits);
        const a_finite = std.math.isFinite(a);
        const e_finite = std.math.isFinite(e);
        c.nonfinite_actual += @intFromBool(!a_finite);
        c.nonfinite_expected += @intFromBool(!e_finite);
        if (!a_finite or !e_finite) continue;
        const diff: f64 = @abs(@as(f64, a) - @as(f64, e));
        if (diff > c.max_abs) {
            c.max_abs = diff;
            c.max_abs_at = i;
        }
        if (e != 0) {
            const rel = diff / @abs(@as(f64, e));
            if (rel > c.max_rel) {
                c.max_rel = rel;
                c.max_rel_at = i;
            }
        }
    }
    return c;
}

test compare {
    const one: u16 = 0x3f80; // 1.0
    const two: u16 = 0x4000; // 2.0
    const nan: u16 = 0x7fc0;
    const expected = [_]u16{ one, two, one, 0 };
    const actual = [_]u16{ one, one, nan, 0x8000 }; // -0 differs from +0 bitwise
    const c = compare(std.mem.sliceAsBytes(&actual), std.mem.sliceAsBytes(&expected), 2);
    try std.testing.expectEqual(@as(usize, 4), c.elements);
    try std.testing.expectEqual(@as(usize, 3), c.mismatched);
    try std.testing.expectEqual(@as(usize, 1), c.nonfinite_actual);
    try std.testing.expectEqual(@as(f64, 1), c.max_abs);
    try std.testing.expectEqual(@as(?usize, 1), c.max_abs_at);
    try std.testing.expectEqual(@as(f64, 0.5), c.max_rel);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, c.first[0..c.first_len]);
    const same = compare(std.mem.sliceAsBytes(&expected), std.mem.sliceAsBytes(&expected), 2);
    try std.testing.expectEqual(@as(usize, 0), same.mismatched);
    try std.testing.expectEqualSlices(u8, &same.sha_actual, &same.sha_expected);
}

test summarize {
    var samples = [_]f64{ 4, 1, 3, 2 };
    const s = summarize(&samples);
    try std.testing.expectEqual(@as(f64, 2.5), s.median);
    try std.testing.expectEqual(@as(f64, 1), s.min);
    try std.testing.expectEqual(@as(f64, 2.5), s.mean);
}

test "workspace sizes match DeepGEMM's SymmBuffer sizes" {
    // `SymmBuffer.buffer.nbytes` reported by the reference dumps on 152 SMs.
    var problem: megamoe.Problem = .{
        .num_ranks = 2,
        .num_experts = 384,
        .num_topk = 6,
        .hidden = 5120,
        .intermediate = 2304,
        .num_shared_experts = 1,
        .num_max_tokens_per_rank = 4096,
    };
    try std.testing.expectEqual(@as(u64, 680837504), megamoe.workspaceBytes(problem, 152)); // r2_t4096
    problem.num_max_tokens_per_rank = 8192;
    try std.testing.expectEqual(@as(u64, 1070359424), megamoe.workspaceBytes(problem, 152)); // r2_t4096_s1
    problem.num_ranks = 1;
    try std.testing.expectEqual(@as(u64, 1048832384), megamoe.workspaceBytes(problem, 152)); // r1_t512_s1
    problem.num_shared_experts = 0;
    try std.testing.expectEqual(@as(u64, 892774784), megamoe.workspaceBytes(problem, 152)); // r1_t512_s0
    problem = .{ .num_ranks = 1, .num_experts = 32, .num_topk = 4, .hidden = 1024, .intermediate = 512, .num_shared_experts = 1, .num_max_tokens_per_rank = 8192 };
    try std.testing.expectEqual(@as(u64, 179563904), megamoe.workspaceBytes(problem, 152)); // r1_small
    problem.num_experts = 128;
    problem.num_topk = 6;
    try std.testing.expectEqual(@as(u64, 223309184), megamoe.workspaceBytes(problem, 152)); // r1_small_e128
}
