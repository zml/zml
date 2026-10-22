//! End-to-end check of `zml.kernel.cute.Program.CallOpts.symmetric_workspace`
//! on every visible GPU: one rank per partition.
//!
//! Each call is one CuTe program whose kernel increments an epoch counter kept
//! in this rank's workspace, writes `(rank + 1) + 1000 * epoch` into slot
//! `rank` of every rank's workspace (double-buffered by epoch parity), then
//! waits until its own workspace holds every rank's value for this epoch. The
//! output row of a rank is `[slot 0, .., slot R-1, epoch, calls, rank]`, plus
//! zml's `partitionId()`, which XLA's rank must equal. Two
//! executables chain 6 and 1 calls: the first runs as a command buffer (CUDA
//! graph), the second as plain thunks, and they share one workspace, so the
//! epoch continues across executions and executables. The first is then
//! replayed on a fixed input, and a third one, which donates its input, is
//! fed its own output so its command buffer replays without updates.
const std = @import("std");

const dialects = @import("mlir/dialects");
const zml = @import("zml");
const cute = zml.kernel.cute;
const B = cute.Builder;

const log = std.log.scoped(.symmetric_workspace);

const max_ranks = 8;
const workspace: zml.ops.CuteOps.SymmetricWorkspace = .{
    .name = "zml.kernels.symmetric_workspace.check",
    .bytes = 1 << 20,
};
/// The epoch counter, past the slots.
const epoch_offset = 4096;
/// Polls before a rank gives up on a peer (the check then fails, not hangs).
const max_polls = 1 << 24;
const kernel_name = "symmetric_workspace_check_kernel";

const Config = struct { num_ranks: usize };

const Check = cute.Program(Config, .{
    .name = "symmetric_workspace_check",
    .inputs = &.{"prev"},
    .outputs = &.{"out"},
    .run = program,
});

/// What the kernel writes.
fn kernelRowLength(num_ranks: usize) usize {
    return num_ranks + 3;
}

/// The kernel's row and the partition id.
fn rowLength(num_ranks: usize) usize {
    return kernelRowLength(num_ranks) + 1;
}

fn program(b: *B, cfg: Config) cute.FinishError!void {
    std.debug.assert(cfg.num_ranks >= 1 and cfg.num_ranks <= max_ranks);
    b.beginFunction(kernel_name, .cuda_kernel);
    try kernel(b, cfg);
    b.endFunction(.{ 32, 1, 1 });

    b.beginFunction(Check.name, .host);
    const host = try b.declareArgsWithSymmetricWorkspace(.{
        .prev = .{ .ptr = cute.DType.i32 },
        .out = .{ .ptr = cute.DType.i32 },
    }, cfg.num_ranks);
    // The kernel takes `max_ranks` workspace pointers and reads the first
    // `num_ranks`.
    var ws: [max_ranks]cute.Value = undefined;
    for (&ws, 0..) |*w, r| w.* = host.workspaces[@min(r, cfg.num_ranks - 1)];
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ one, one, one },
        .block = .{ b.cst(.i32, 32), one, one },
        .dynamic_smem = b.kernelSmemSize(kernel_name),
        .stream = b.cudaStream(),
    });
    const launched = b.launchEx(kernel_name, config, .{
        host.args.prev, host.args.out, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7], host.rank,
    });
    b.returnHostStatus(b.cudaResultStatus(launched));
    b.endFunction(null);
}

fn stReleaseSys(b: *B, address: cute.Value, value: cute.Value) void {
    b.emitVoid(dialects.llvm.inline_asm(b.ctx, "st.release.sys.global.b32 [$0], $1;", &.{ address.inner, value.inner }, null, .{ .constraints = "l,r,~{memory}", .has_side_effects = true }, b.loc()));
}

fn ldAcquireSys(b: *B, address: cute.Value) cute.Value {
    return b.emit(dialects.llvm.inline_asm(b.ctx, "ld.acquire.sys.global.b32 $0, [$1];", &.{address.inner}, cute.DType.i32.toMlir(b.ctx), .{ .constraints = "=r,l,~{memory}", .has_side_effects = true }, b.loc()));
}

fn kernel(b: *B, cfg: Config) cute.FinishError!void {
    const ptr: cute.DType = .i8;
    const a = try b.declareArgs(.{
        .prev = .{ .ptr = cute.DType.i32 },
        .out = .{ .ptr = cute.DType.i32 },
        .w0 = .{ .ptr = ptr },
        .w1 = .{ .ptr = ptr },
        .w2 = .{ .ptr = ptr },
        .w3 = .{ .ptr = ptr },
        .w4 = .{ .ptr = ptr },
        .w5 = .{ .ptr = ptr },
        .w6 = .{ .ptr = ptr },
        .w7 = .{ .ptr = ptr },
        .rank = .{ .mlir_type = .{ .type_ = cute.DType.i32.toMlir(b.ctx) } },
    });
    const ranks: i32 = @intCast(cfg.num_ranks);
    const all = [max_ranks]cute.Value{ a.w0, a.w1, a.w2, a.w3, a.w4, a.w5, a.w6, a.w7 };
    var peers: [max_ranks]cute.Value = undefined;
    for (peers[0..cfg.num_ranks], all[0..cfg.num_ranks]) |*p, w| p.* = b.ptrToInt(w, .i64);
    var own = peers[0];
    for (1..cfg.num_ranks) |r| own = b.select(a.rank.eq(@as(i32, @intCast(r))), peers[r], own);

    var leader = b.openIf(b.threadIdx().x.eq(0));
    const epoch_address = own.add(@as(i64, epoch_offset));
    const epoch = b.loadWords(1, epoch_address, .global, .{})[0].add(1);
    b.storeWords(1, epoch_address, .global, .{epoch});
    const parity_base = epoch.bitAnd(1).mul(ranks);
    const value = a.rank.add(1).add(epoch.mul(1000));
    const slot = a.rank.add(parity_base).to(.i64).mul(4);
    for (peers[0..cfg.num_ranks]) |peer| stReleaseSys(b, peer.add(slot), value);

    const out = b.ptrToInt(a.out, .i64);
    const i32_type = cute.DType.i32.toMlir(b.ctx);
    for (0..cfg.num_ranks) |q| {
        const address = own.add(parity_base.add(@as(i32, @intCast(q))).to(.i64).mul(4));
        const expected = epoch.mul(1000).add(@as(i32, @intCast(q + 1)));
        var poll = b.openWhile(.{ b.cst(.i32, 0), b.cst(.i32, 0) }, .{ i32_type, i32_type });
        const seen = ldAcquireSys(b, address);
        const polls = poll.before_carried[1];
        poll.yieldBefore(seen.ne(expected).bitAnd(polls.lt(@as(i32, max_polls))), .{ seen, polls });
        poll.yieldAfter(.{ poll.after_carried[0], poll.after_carried[1].add(1) });
        b.storeWords(1, out.add(@as(i64, @intCast(4 * q))), .global, .{poll.results[0]});
    }
    const tail = out.add(@as(i64, @intCast(4 * cfg.num_ranks)));
    const calls = b.loadWords(1, b.ptrToInt(a.prev, .i64).add(@as(i64, @intCast(4 * (cfg.num_ranks + 1)))), .global, .{})[0];
    b.storeWords(1, tail, .global, .{epoch});
    b.storeWords(1, tail.add(4), .global, .{calls.add(1)});
    b.storeWords(1, tail.add(8), .global, .{a.rank});
    leader.yieldThen(.{});
}

/// `calls` chained programs per rank; the row of each rank is threaded through.
/// `donate` makes the output reuse the input's buffer, so a command buffer
/// fed its own output replays without updates.
fn chain(prev: zml.Tensor, num_ranks: usize, calls: usize, donate: bool) zml.Tensor {
    const out = zml.ops.manualComputation(
        (struct {
            prev: zml.Tensor,
            num_ranks: usize,
            calls: usize,

            fn body(ctx: @This(), _: zml.Shape) zml.Tensor {
                const row: zml.Shape = .init(.{@as(i64, @intCast(kernelRowLength(ctx.num_ranks)))}, .i32);
                var x = ctx.prev.reshape(.{@as(i64, @intCast(rowLength(ctx.num_ranks)))});
                for (0..ctx.calls) |_| {
                    x = Check.call(.{ .prev = x }, .{ .out = row }, .{
                        .cfg = .{ .num_ranks = ctx.num_ranks },
                        .symmetric_workspace = workspace,
                    }).out;
                }
                const partition = zml.ops.partitionId().convert(.i32).reshape(.{1});
                // The local shape: one row.
                return zml.Tensor.concatenate(&.{ x, partition }, 0).reshape(ctx.prev.shape());
            }
        }).body,
        .{ .prev = prev, .num_ranks = num_ranks, .calls = calls },
        prev.shape(),
    );
    return if (donate) out.reuseBuffer(prev) else out;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NeedsCuda;
    const num_ranks = platform.devices.len;
    if (num_ranks > max_ranks) return error.TooManyDevices;
    log.info("{d} rank(s)", .{num_ranks});

    const mesh = try platform.registerMesh("ranks", .mesh(.{ .model = .high_bandwidth }));
    const width = rowLength(num_ranks);
    const shape = zml.Shape.init(.{ .rank = @as(i64, @intCast(num_ranks)), .x = @as(i64, @intCast(width)) }, .i32)
        .withPartitioning(mesh, .{ .rank = .model, .x = .replicated });

    const chains = [_]usize{ 6, 1, 6 };
    const donates = [_]bool{ false, false, true };
    var exes: [chains.len]zml.Exe = undefined;
    for (&exes, chains, donates) |*exe, calls, donate| {
        exe.* = try platform.compileFn(allocator, io, chain, .{ zml.Tensor.fromShape(shape), num_ranks, calls, donate }, .{ .meshes = &.{mesh} });
    }
    defer for (&exes) |*exe| exe.deinit();

    const zeros = try zml.Slice.alloc(allocator, shape);
    defer zeros.free(allocator);
    @memset(zeros.items(i32), 0);
    var prev = try zml.Buffer.fromSlice(io, platform, zeros);
    defer prev.deinit();

    var epoch: i32 = 0;
    var calls_so_far: i32 = 0;
    var failures: usize = 0;
    // Alternate the executables, threading each output into the next call,
    // then replay the command-buffer one on a fixed input: the epoch lives on
    // the device and keeps counting.
    const steps = 8;
    const replays = 4;
    const donated = 4;
    for (0..steps + replays + donated) |step| {
        const which: usize = if (step < steps) step % 2 else if (step < steps + replays) 0 else 2;
        const threaded = step < steps or step >= steps + replays;
        const exe = &exes[which];
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{prev});
        exe.call(io, args, &results, .{});
        var out = results.get(zml.Buffer);

        epoch += @intCast(chains[which]);
        const calls = calls_so_far + @as(i32, @intCast(chains[which]));
        const got = try out.toSliceAlloc(allocator, io);
        defer got.free(allocator);
        for (0..num_ranks) |r| {
            const row = got.items(i32)[r * width ..][0..width];
            var expected: [max_ranks + 4]i32 = undefined;
            for (0..num_ranks) |q| expected[q] = @as(i32, @intCast(q + 1)) + 1000 * epoch;
            expected[num_ranks] = epoch;
            expected[num_ranks + 1] = calls;
            expected[num_ranks + 2] = @intCast(r);
            expected[num_ranks + 3] = @intCast(r);
            const ok = std.mem.eql(i32, row, expected[0..width]);
            if (!ok) failures += 1;
            log.info("step {d} ({d} calls{s}) rank {d}: {any} {s}", .{ step, chains[which], if (threaded) "" else ", same input", r, row, if (ok) "ok" else "MISMATCH" });
            if (!ok) log.err("  expected {any}", .{expected[0..width]});
        }
        if (threaded) {
            prev.deinit();
            prev = out;
            calls_so_far = calls;
        } else {
            out.deinit();
        }
    }
    if (failures > 0) {
        log.err("{d} mismatching rows", .{failures});
        return error.CheckFailed;
    }
    log.info("symmetric workspace check passed on {d} rank(s)", .{num_ranks});
}
