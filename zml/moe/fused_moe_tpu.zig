const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const CompilationContext = @import("../module.zig").CompilationContext;
const Tensor = zml.Tensor;
const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");

const mtt = @import("kernels/mosaic_tpu/builder");

const log = std.log.scoped(.moe_fused_moe_tpu);

var gate_up_transpose: std.atomic.Value(bool) = .init(false);
var down_transpose: std.atomic.Value(bool) = .init(false);

const CacheEntry = struct {
    cfg: Cfg,
    ir: [:0]u8,
};

var cache_mutex: std.Io.Mutex = .init;
var cache_entries: [64]CacheEntry = undefined;
var cache_len: usize = 0;

pub const ActivationMode = enum {
    silu,
    relu,
    gelu,
    quick_gelu_plus_one,
};

pub const Options = struct {
    activation: ActivationMode = .gelu,
    num_experts_per_tok: u32,
};

pub const Parameters = struct {
    num_experts_per_tok: u32,
    activation: ActivationMode,

    pub const InitOptions = struct {
        num_experts_per_tok: u32,
        activation: ActivationMode = .gelu,
    };

    pub fn init(opts: InitOptions) Parameters {
        return .{
            .num_experts_per_tok = opts.num_experts_per_tok,
            .activation = opts.activation,
        };
    }
};

pub const Metadata = struct {
    scratch: ?Tensor,

    pub const InitOptions = struct {
        scratch_shape: ?zml.Shape = null,
    };

    pub fn init(opts: InitOptions) Metadata {
        return .{
            .scratch = if (opts.scratch_shape) |scratch_shape| .fromShape(scratch_shape) else null,
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        return .{
            .scratch = if (self.scratch) |scratch| try zml.Buffer.uninitialized(io, platform, scratch.shape(), .replicated, .{}) else null,
        };
    }
};

pub fn deinitBuffer(bufferized: *zml.Bufferized(Metadata)) void {
    if (bufferized.scratch) |*scratch| scratch.deinit();
}

fn warningTranspose(flag: *std.atomic.Value(bool), projection_name: []const u8, expected_shape: []const u8) void {
    if (!flag.swap(true, .monotonic)) {
        log.warn("{s} MoE weights are transposed at runtime; weights must be loaded with the correct shape {s} to avoid the transpose at runtime.", .{ projection_name, expected_shape });
    }
}

fn canonicalizeGateUp(w1: Tensor, hidden_size: i64) !Tensor {
    if (w1.rank() != 3) return error.InvalidShape;

    if (w1.dim(1) == hidden_size) return w1.withTags(.{ .expert, .in, .out });
    if (w1.dim(2) == hidden_size) {
        warningTranspose(&gate_up_transpose, "gate_up_proj", "[expert, in, out]");
        return w1.withTags(.{ .expert, .out, .in }).transpose(.{ .expert, .in, .out });
    }
    return error.InvalidShape;
}

fn canonicalizeDown(w2: Tensor, hidden_size: i64, gate_up_out: i64) !Tensor {
    if (w2.rank() != 3) return error.InvalidShape;

    const mid_size = @divFloor(gate_up_out, 2);
    if (w2.dim(1) == mid_size and w2.dim(2) == hidden_size) return w2.withTags(.{ .expert, .mid, .out });
    if (w2.dim(1) == hidden_size and w2.dim(2) == mid_size) {
        warningTranspose(&down_transpose, "down_proj", "[expert, mid, out]");
        return w2.withTags(.{ .expert, .out, .mid }).transpose(.{ .expert, .mid, .out });
    }
    return error.InvalidShape;
}

fn alignTo(x: i64, y: i64) i64 {
    return @divFloor(x + y - 1, y) * y;
}

fn minInt(a: i64, b: i64) i64 {
    return if (a < b) a else b;
}

fn maxInt(a: i64, b: i64) i64 {
    return if (a > b) a else b;
}

fn cfgDType(dtype: zml.DataType) mtt.DType {
    return switch (zml.kernel.mosaic_tpu.from(dtype)) {
        .bf16 => .bf16,
        .f16 => .f16,
        .f32 => .f32,
        else => stdx.debug.panic("fused_moe_tpu expects bf16/f16/f32 tensors, got {}", .{dtype}),
    };
}

fn tokenPacking(dtype: zml.DataType) i64 {
    return switch (zml.kernel.mosaic_tpu.from(dtype)) {
        .bf16, .f16 => 2,
        .f32, .i32 => 1,
        .i8 => 4,
        else => 1,
    };
}

fn tokenBlockSize(local_num_tokens: i64) i64 {
    if (local_num_tokens >= 32 and @rem(local_num_tokens, 32) == 0) return 32;
    if (local_num_tokens >= 16 and @rem(local_num_tokens, 16) == 0) return 16;
    return 8;
}

pub fn scratchShape(num_tokens: i64, hidden_size: i64, num_experts: i64, ep_size: i64, dtype: zml.DataType) zml.Shape {
    if (ep_size <= 0) stdx.debug.panic("fused_moe_tpu scratch ep_size must be positive, got {d}", .{ep_size});

    const min_tokens = ep_size * 8;
    const padded_tokens = alignTo(@max(num_tokens, min_tokens), ep_size * 8);
    const local_padded_tokens = @divExact(padded_tokens, ep_size);
    const scratch_pack = tokenPacking(dtype);

    return zml.Shape.init(.{
        .expert = num_experts,
        .bt = tokenBlockSize(local_padded_tokens),
        .pack = scratch_pack,
        .hidden = @divExact(hidden_size, scratch_pack),
    }, dtype).withPartitioning(.{
        .expert = .replicated,
        .bt = .replicated,
        .pack = .replicated,
        .hidden = .replicated,
    });
}

const Cfg = struct {
    pub const ActFn = enum {
        silu,
        gelu,
        swigluoai,
    };

    pub const ScoringFn = enum {
        softmax,
        sigmoid,
    };

    num_tokens: i64,
    hidden_size: i64,
    intermediate_size: i64,
    num_experts: i64,
    top_k: i64,
    ep_size: i64,
    token_dtype: mtt.DType = .bf16,
    weight_dtype: mtt.DType = .bf16,
    renormalize_topk_logits: bool = false,
    act_fn: ActFn = .silu,
    scoring_fn: ScoringFn = .softmax,
    bt: i64,
    bf: i64,
    bd1: i64,
    bd2: i64,
    btc: i64,
    bfc: i64,
    bd1c: i64,
    bd2c: i64,

    pub fn tokenPacking(self: Cfg) i64 {
        return switch (self.token_dtype) {
            .bf16, .f16 => 2,
            .f32, .i32 => 1,
            .i8 => 4,
            else => 1,
        };
    }
};

fn fusedAct(mode: ActivationMode) Cfg.ActFn {
    return switch (mode) {
        .silu => .silu,
        .gelu => .gelu,
        .relu => stdx.debug.panic("fused_moe TPU kernel does not support relu activation", .{}),
        .quick_gelu_plus_one => .swigluoai,
    };
}

fn fusedCfg(
    tokens: Tensor,
    w1: Tensor,
    w2: Tensor,
    gating: Tensor,
    opts: Options,
    ep_size: i64,
) Cfg {
    const hidden_size = tokens.dim(.d);
    const intermediate_size = w2.dim(.mid);
    const top_k: i64 = @intCast(opts.num_experts_per_tok);
    const local_num_tokens = tokens.dim(.token);
    const num_tokens = local_num_tokens * ep_size;

    const bt = tokenBlockSize(local_num_tokens);
    const btc = minInt(bt, 32);
    const bf: i64 = if (@rem(intermediate_size, 512) == 0) 512 else 256;
    const bd1: i64 = if (@rem(hidden_size, 512) == 0) 512 else 256;
    const bd2: i64 = if (@rem(hidden_size, 512) == 0) 512 else 256;

    return .{
        .num_tokens = num_tokens,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
        .num_experts = gating.dim(.expert),
        .top_k = top_k,
        .ep_size = ep_size,
        .token_dtype = cfgDType(tokens.dtype()),
        .weight_dtype = cfgDType(w1.dtype()),
        .renormalize_topk_logits = true,
        .act_fn = fusedAct(opts.activation),
        .scoring_fn = .softmax,
        .bt = bt,
        .bf = bf,
        .bd1 = bd1,
        .bd2 = bd2,
        .btc = btc,
        .bfc = minInt(bf, 256),
        .bd1c = minInt(bd1, 256),
        .bd2c = minInt(bd2, 256),
    };
}

fn boolArg(value: bool) [:0]const u8 {
    return if (value) "1" else "0";
}

fn intArg(allocator: std.mem.Allocator, value: i64) [:0]const u8 {
    const tmp = std.fmt.allocPrint(allocator, "{d}", .{value}) catch @panic("fused_moe_tpu int arg OOM");
    defer allocator.free(tmp);
    return allocator.dupeZ(u8, tmp) catch @panic("fused_moe_tpu int arg OOM");
}

fn emitterPath(allocator: std.mem.Allocator) ![:0]const u8 {
    const runfiles = bazel.runfiles(bazel_builtin.current_repository) catch
        return try allocator.dupeZ(u8, "/home/gcpuser/new-backend/zml/bazel-bin/zml/fused_moe_tpu_emit");

    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = runfiles.rlocation("zml/zml/fused_moe_tpu_emit", &path_buf) catch null orelse
        runfiles.rlocation("zml/fused_moe_tpu_emit", &path_buf) catch |err| {
            log.err("Failed to resolve fused_moe_tpu emitter in runfiles: {}", .{err});
            return err;
        };

    const resolved = path orelse "/home/gcpuser/new-backend/zml/bazel-bin/zml/fused_moe_tpu_emit";
    return try allocator.dupeZ(u8, resolved);
}

const EmitterRunResult = struct {
    stdout: []u8,
    stderr: []u8,
    term: std.process.Child.Term,
};

fn closeLinuxFd(fd: i32) void {
    _ = std.os.linux.close(fd);
}

fn linuxPipe() ![2]i32 {
    var fds: [2]i32 = undefined;
    const rc = std.os.linux.pipe(&fds);
    return switch (std.posix.errno(rc)) {
        .SUCCESS => fds,
        .MFILE => error.ProcessFdQuotaExceeded,
        .NFILE => error.SystemFdQuotaExceeded,
        .NOMEM => error.SystemResources,
        else => error.Unexpected,
    };
}

fn readLinuxPipe(allocator: std.mem.Allocator, fd: i32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    var buf: [16 * 1024]u8 = undefined;
    while (true) {
        const rc = std.os.linux.read(fd, buf[0..].ptr, buf.len);
        switch (std.posix.errno(rc)) {
            .SUCCESS => {
                const n: usize = @intCast(rc);
                if (n == 0) break;
                try out.appendSlice(allocator, buf[0..n]);
            },
            .INTR => continue,
            else => return error.EmitterReadFailed,
        }
    }

    return try out.toOwnedSlice(allocator);
}

fn waitLinuxPid(pid: std.os.linux.pid_t) !std.process.Child.Term {
    var status: u32 = 0;
    while (true) {
        const rc = std.os.linux.waitpid(pid, &status, 0);
        switch (std.posix.errno(rc)) {
            .SUCCESS => break,
            .INTR => continue,
            else => return error.EmitterWaitFailed,
        }
    }

    if ((status & 0x7f) == 0) return .{ .exited = @intCast((status >> 8) & 0xff) };
    return .{ .unknown = status };
}

fn runEmitterLinux(allocator: std.mem.Allocator, argv: []const [:0]const u8) !EmitterRunResult {
    var argv_buf = try allocator.allocSentinel(?[*:0]const u8, argv.len, null);
    defer allocator.free(argv_buf);
    for (argv, 0..) |arg, i| argv_buf[i] = arg.ptr;
    const empty_env = [_:null]?[*:0]const u8{};

    const stdout_pipe = try linuxPipe();
    errdefer closeLinuxFd(stdout_pipe[0]);
    errdefer closeLinuxFd(stdout_pipe[1]);

    const stderr_pipe = try linuxPipe();
    errdefer closeLinuxFd(stderr_pipe[0]);
    errdefer closeLinuxFd(stderr_pipe[1]);

    const fork_rc = std.os.linux.fork();
    switch (std.posix.errno(fork_rc)) {
        .SUCCESS => {},
        .AGAIN => return error.SystemResources,
        .NOMEM => return error.SystemResources,
        else => return error.Unexpected,
    }

    const pid: std.os.linux.pid_t = @intCast(fork_rc);
    if (pid == 0) {
        closeLinuxFd(stdout_pipe[0]);
        closeLinuxFd(stderr_pipe[0]);
        if (std.posix.errno(std.os.linux.dup2(stdout_pipe[1], std.posix.STDOUT_FILENO)) != .SUCCESS) std.os.linux.exit(127);
        if (std.posix.errno(std.os.linux.dup2(stderr_pipe[1], std.posix.STDERR_FILENO)) != .SUCCESS) std.os.linux.exit(127);
        closeLinuxFd(stdout_pipe[1]);
        closeLinuxFd(stderr_pipe[1]);
        _ = std.os.linux.execve(argv_buf[0].?, argv_buf.ptr, &empty_env);
        std.os.linux.exit(127);
    }

    closeLinuxFd(stdout_pipe[1]);
    closeLinuxFd(stderr_pipe[1]);

    const stdout = try readLinuxPipe(allocator, stdout_pipe[0]);
    errdefer allocator.free(stdout);
    closeLinuxFd(stdout_pipe[0]);

    const stderr = try readLinuxPipe(allocator, stderr_pipe[0]);
    errdefer allocator.free(stderr);
    closeLinuxFd(stderr_pipe[0]);

    return .{
        .stdout = stdout,
        .stderr = stderr,
        .term = try waitLinuxPid(pid),
    };
}

fn runEmitter(allocator: std.mem.Allocator, io: std.Io, argv: []const [:0]const u8) !EmitterRunResult {
    if (builtin.os.tag == .linux) return runEmitterLinux(allocator, argv);

    const result = try std.process.run(allocator, io, .{
        .argv = argv,
        .stdout_limit = .unlimited,
        .stderr_limit = .limited(16 * 1024),
    });
    return .{
        .stdout = result.stdout,
        .stderr = result.stderr,
        .term = result.term,
    };
}

fn emitExternal(allocator: std.mem.Allocator, io: std.Io, cfg: Cfg) ![:0]const u8 {
    cache_mutex.lockUncancelable(io);
    defer cache_mutex.unlock(io);

    for (cache_entries[0..cache_len]) |entry| {
        if (std.meta.eql(entry.cfg, cfg)) return try allocator.dupeZ(u8, entry.ir);
    }

    const emitter = try emitterPath(allocator);
    defer allocator.free(emitter);
    const int_args = [_][:0]const u8{
        intArg(allocator, cfg.num_tokens),
        intArg(allocator, cfg.hidden_size),
        intArg(allocator, cfg.intermediate_size),
        intArg(allocator, cfg.num_experts),
        intArg(allocator, cfg.top_k),
        intArg(allocator, cfg.ep_size),
        intArg(allocator, cfg.bt),
        intArg(allocator, cfg.bf),
        intArg(allocator, cfg.bd1),
        intArg(allocator, cfg.bd2),
        intArg(allocator, cfg.btc),
        intArg(allocator, cfg.bfc),
        intArg(allocator, cfg.bd1c),
        intArg(allocator, cfg.bd2c),
    };
    defer {
        for (int_args) |arg_value| allocator.free(arg_value);
    }

    const argv = [_][:0]const u8{
        emitter,
        int_args[0],
        int_args[1],
        int_args[2],
        int_args[3],
        int_args[4],
        int_args[5],
        @tagName(cfg.token_dtype),
        @tagName(cfg.weight_dtype),
        boolArg(cfg.renormalize_topk_logits),
        @tagName(cfg.act_fn),
        @tagName(cfg.scoring_fn),
        int_args[6],
        int_args[7],
        int_args[8],
        int_args[9],
        int_args[10],
        int_args[11],
        int_args[12],
        int_args[13],
    };
    const process_allocator = std.heap.c_allocator;
    const result = try runEmitter(process_allocator, io, &argv);
    defer process_allocator.free(result.stdout);
    defer process_allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code == 0) {
            if (cache_len == cache_entries.len) return error.FusedMoeEmitterCacheFull;
            const cached = try process_allocator.dupeZ(u8, result.stdout);
            cache_entries[cache_len] = .{ .cfg = cfg, .ir = cached };
            cache_len += 1;
            return try allocator.dupeZ(u8, cached);
        },
        else => {},
    }
    log.err("fused_moe_tpu emitter failed: {s}", .{result.stderr});
    return error.FusedMoeEmitterFailed;
}

pub fn forward(
    hidden_states: Tensor,
    router_logits_: Tensor,
    w1: Tensor,
    w2: Tensor,
    metadata: Metadata,
    opts: Options,
) !Tensor {
    if (hidden_states.dtype() != w1.dtype() or hidden_states.dtype() != w2.dtype()) return error.UnsupportedType;

    const b = hidden_states.dim(.b);
    const s = hidden_states.dim(.s);
    const hidden = hidden_states.reshape(.{ .token = b * s, .d = hidden_states.dim(.d) }).withTags(.{ .token, .d });
    const gate_up = try canonicalizeGateUp(w1, hidden.dim(.d));
    const down = try canonicalizeDown(w2, hidden.dim(.d), gate_up.dim(.out));

    const router_logits = router_logits_.reshape(.{ .token = b * s, .expert = router_logits_.dim(.expert) }).withTags(.{ .token, .expert });
    if (router_logits.dtype() != .f32 and router_logits.dtype() != .bf16 and router_logits.dtype() != .f16) return error.UnsupportedType;
    if (router_logits.dim(.token) != hidden.dim(.token)) return error.InvalidShape;
    if (router_logits.dim(.expert) != gate_up.dim(.expert) or gate_up.dim(.expert) != down.dim(.expert)) return error.InvalidShape;

    const ctx = CompilationContext.current();
    const ep_size = ctx.partitioning.numPartitionsForLogicalAxis(gate_up.shape(), .experts) catch @as(i64, ctx.partitioning.numPartitions());
    if (ep_size <= 0) return error.InvalidShape;

    const min_tokens = ep_size * 8;
    const padded_tokens = alignTo(@max(hidden.dim(.token), min_tokens), ep_size * 8);
    const pad_tokens = padded_tokens - hidden.dim(.token);
    const padded_hidden = hidden.pad(0, .{ .token = Tensor.Pad{ .low = 0, .high = pad_tokens, .interior = 0 } }).withPartitioning(.{ .token = .experts, .d = .replicated });
    const padded_logits = router_logits.pad(0, .{ .token = Tensor.Pad{ .low = 0, .high = pad_tokens, .interior = 0 } })
        .convert(hidden.dtype())
        .withPartitioning(.{ .token = .experts, .expert = .replicated });

    const w1_kernel = gate_up
        .splitAxis(.out, .{ .proj = 2, .mid = @divExact(gate_up.dim(.out), 2) })
        .transpose(.{ .expert, .proj, .in, .mid })
        .withPartitioning(.{ .expert = .experts, .proj = .replicated, .in = .replicated, .mid = .replicated });
    const w2_kernel = down.withPartitioning(.{ .expert = .experts, .mid = .replicated, .out = .replicated });

    const output_shape = zml.Shape.init(.{ .token = padded_tokens, .d = hidden.dim(.d) }, hidden.dtype())
        .withPartitioning(.{ .token = .experts, .d = .replicated });
    const metadata_scratch = metadata.scratch orelse return error.InvalidMetadata;
    const expected_scratch_shape = scratchShape(padded_tokens, hidden.dim(.d), router_logits.dim(.expert), ep_size, hidden.dtype());
    if (!metadata_scratch.shape().eql(expected_scratch_shape)) return error.InvalidShape;
    const scratch = metadata_scratch.withPartitioning(.{
        .expert = .replicated,
        .bt = .replicated,
        .pack = .replicated,
        .hidden = .replicated,
    });

    const output = zml.ops.manualComputation(
        .{ padded_hidden, w1_kernel, w2_kernel, padded_logits, scratch },
        output_shape,
        .{ .opts = opts, .ep_size = ep_size },
        (struct {
            fn body(body_context: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, local_output_shape: zml.Shape) zml.Tensor {
                const local_tokens = sharded_inputs[0].dim(.token);
                const cfg = fusedCfg(
                    sharded_inputs[0],
                    sharded_inputs[1],
                    sharded_inputs[2],
                    sharded_inputs[3],
                    body_context.opts,
                    body_context.ep_size,
                );
                const tokens_hbm = sharded_inputs[0].reshape(.{
                    .token = local_tokens,
                    .pack = cfg.tokenPacking(),
                    .hidden = @divExact(cfg.hidden_size, cfg.tokenPacking()),
                });
                const cur = CompilationContext.current();
                const allocator = cur.allocator;
                const ir = emitExternal(allocator, cur.io, cfg) catch |err|
                    stdx.debug.panic("fused_moe TPU kernel emit failed: {}", .{err});
                defer allocator.free(ir);
                return zml.kernel.mosaic_tpu.callRaw(
                    ir,
                    &.{ tokens_hbm, sharded_inputs[1], sharded_inputs[2], sharded_inputs[3], sharded_inputs[4] },
                    &.{local_output_shape},
                    .{ .has_communication = true },
                )[0];
            }
        }).body,
    );

    return output
        .slice1d(.token, .{ .end = hidden.dim(.token) })
        .reshape(.{ .b = b, .s = s, .d = hidden.dim(.d) });
}
