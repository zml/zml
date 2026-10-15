//! Shared public-entry-point tests. Hardware absence is the only runtime skip:
//! supported cases are explicit, and errors in those cases must fail the test.
const std = @import("std");
const zml = @import("zml");
const fixtures = @import("fixtures.zig");
const Case = fixtures.Case;
const Matrix = fixtures.Matrix;

test {
    _ = zml.bazel;
}

const Options = struct {
    producers: usize = 16,
    consumers: usize = 1,
    case_queue_capacity: usize = 16,
    executable_queue_capacity: usize = 16,
};

const CaseJob = struct {
    index: usize,
    case: Case,
};

const ExecutableJob = struct {
    job: CaseJob,
    exe: zml.Exe,
};

const Result = struct {
    case: Case,
    outcome: union(enum) {
        passed,
        skipped,
        compilation_failed: anyerror,
        check_failed: anyerror,
    },
};

const Pipeline = struct {
    platform: *const zml.Platform,

    fn generateCases(self: *const Pipeline, io: std.Io, backend: zml.moe.Backend, matrix: Matrix, case_queue: *std.Io.Queue(CaseJob), results: []?Result) std.Io.Cancelable!void {
        defer case_queue.close(io);
        const available = backend.isAvailable(self.platform);

        var index: usize = 0;
        for (0..matrix.count()) |local_index| {
            const c = matrix.caseAt(backend, local_index);
            if (available) {
                case_queue.putOne(io, .{ .index = index, .case = c }) catch |err| return switch (err) {
                    error.Closed => {},
                    error.Canceled => error.Canceled,
                };
            } else {
                results[index] = .{ .case = c, .outcome = .skipped };
            }
            index += 1;
        }
    }

    fn compileExecutablesWorker(self: *const Pipeline, io: std.Io, allocator: std.mem.Allocator, case_queue: *std.Io.Queue(CaseJob), executable_queue: *std.Io.Queue(ExecutableJob), results: []?Result) std.Io.Cancelable!void {
        while (true) {
            const job = case_queue.getOne(io) catch |err| return switch (err) {
                error.Closed => {},
                error.Canceled => error.Canceled,
            };
            var exe = job.case.compile(allocator, io, self.platform) catch |err| {
                if (err == error.Canceled) return error.Canceled;
                results[job.index] = .{ .case = job.case, .outcome = .{ .compilation_failed = err } };
                continue;
            };
            // Ownership transfers only on a successful enqueue.
            executable_queue.putOne(io, .{ .job = job, .exe = exe }) catch |err| {
                exe.deinit();
                return switch (err) {
                    error.Closed => {},
                    error.Canceled => error.Canceled,
                };
            };
        }
    }

    fn runTestCaseWorker(self: *const Pipeline, io: std.Io, allocator: std.mem.Allocator, executable_queue: *std.Io.Queue(ExecutableJob), results: []?Result) std.Io.Cancelable!void {
        while (true) {
            var compiled = executable_queue.getOne(io) catch |err| return switch (err) {
                error.Closed => {},
                error.Canceled => error.Canceled,
            };
            defer compiled.exe.deinit();
            const job = compiled.job;
            job.case.check(allocator, io, self.platform, &compiled.exe) catch |err| {
                if (err == error.Canceled) return error.Canceled;
                results[job.index] = .{ .case = job.case, .outcome = .{ .check_failed = err } };
                continue;
            };
            results[job.index] = .{ .case = job.case, .outcome = .passed };
        }
    }

    fn run(self: *const Pipeline, io: std.Io, allocator: std.mem.Allocator, backend: zml.moe.Backend, matrix: Matrix, options: Options) !void {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();

        const cases_buffer = try arena.allocator().alloc(CaseJob, options.case_queue_capacity);
        const executables_buffer = try arena.allocator().alloc(ExecutableJob, options.executable_queue_capacity);
        const results = try arena.allocator().alloc(?Result, matrix.count());
        @memset(results, null);

        var case_queue: std.Io.Queue(CaseJob) = .init(cases_buffer);
        var executable_queue: std.Io.Queue(ExecutableJob) = .init(executables_buffer);

        var enumerator: std.Io.Group = .init;
        var producers: std.Io.Group = .init;
        var consumers: std.Io.Group = .init;
        defer {
            case_queue.close(io);
            executable_queue.close(io);
            enumerator.cancel(io);
            producers.cancel(io);
            consumers.cancel(io);
            // On partial startup or cancellation, queued executables still own resources.
            while (executable_queue.getOneUncancelable(io)) |compiled| {
                compiled.exe.deinit();
            } else |_| {}
        }

        for (0..options.consumers) |_| try consumers.concurrent(io, runTestCaseWorker, .{ self, io, allocator, &executable_queue, results });
        for (0..options.producers) |_| try producers.concurrent(io, compileExecutablesWorker, .{ self, io, allocator, &case_queue, &executable_queue, results });
        try enumerator.concurrent(io, generateCases, .{ self, io, backend, matrix, &case_queue, results });

        try enumerator.await(io);
        try producers.await(io);
        // Closed queues drain before getOne returns error.Closed.
        executable_queue.close(io);
        try consumers.await(io);

        try report(results);
    }

    fn report(results: []const ?Result) !void {
        var failed: usize = 0;
        for (results, 0..) |result, index| {
            const r = result orelse return error.MissingCaseResult;
            switch (r.outcome) {
                .passed, .skipped => {},
                inline .compilation_failed, .check_failed => |err, stage| {
                    failed += 1;
                    std.debug.print("MoE case {}/{} {s}: {any}\nError: {}\n", .{
                        index + 1, results.len, @tagName(stage), r.case, err,
                    });
                },
            }
        }
        if (failed != 0) return error.MoeComplianceFailed;
    }
};

test "Triton compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    try pipeline.run(std.testing.io, std.testing.allocator, .triton, fixtures.matrices.triton_base, .{});

    if (platform.target != .cuda) {
        // Other platforms retain the general MXFP4 path and its full contract.
        try pipeline.run(std.testing.io, std.testing.allocator, .triton, fixtures.matrices.triton_mxfp4_general, .{});
        return;
    }

    // Preserve the 12 supported cases from the former general MXFP4 matrix.
    // Its other 252 cases require activations or projection bias unsupported
    // by the simplified path. The former specialized matrix remains below.
    try pipeline.run(std.testing.io, std.testing.allocator, .triton, fixtures.matrices.triton_mxfp4_cuda_small, .{});
}

// Keep triton mxfp4 separate because it only supports a subset for now
test "Triton MXFP4 compatibility matrix" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;
    const pipeline: Pipeline = .{ .platform = platform };

    // The merged backend uses the former specialized MXFP4 path: clipped
    // SwiGLU without projection bias, with unconditional MXFP8 activations.
    try pipeline.run(std.testing.io, std.testing.allocator, .triton, fixtures.matrices.triton_mxfp4_cuda, .{});
}

test "Triton MoE prefill launch buckets fit shared memory" {
    const platform = zml.testing.env();
    if (!zml.moe.Backend.triton.isAvailable(platform) or platform.target != .rocm) return error.SkipZigTest;

    for ([_]i64{ 8, 32, 96, 128, 256, 512, 1024, 2048, 3072, 4096 }) |tokens| {
        const case: Case = .{
            .backend = .triton,
            .tokens = tokens,
            .width = 256,
            .intermediate = 128,
            .topk = 2,
        };
        var exe = try case.compile(std.testing.allocator, std.testing.io, platform);
        defer exe.deinit();
        try case.check(std.testing.allocator, std.testing.io, platform, &exe);
    }
}

test "FlashInfer compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    // Only default SwiGLU, no linear bias, and routing after the down projection.
    try pipeline.run(std.testing.io, std.testing.allocator, .flashinfer_cutlass, fixtures.matrices.flashinfer, .{});
}

test "specialized CuTe MXFP4 compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    try pipeline.run(std.testing.io, std.testing.allocator, .cute, fixtures.matrices.cute, .{});
}

test "Fly compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    // These constraints keep execution on Fly rather than its Triton fallback.
    try pipeline.run(std.testing.io, std.testing.allocator, .fly, fixtures.matrices.fly, .{});
}

test "Metal compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    // Quantized Metal kernels require BF16 activations and do not support bias.
    try pipeline.run(std.testing.io, std.testing.allocator, .metal, fixtures.matrices.metal, .{});
}

test "Mosaic TPU compatibility matrix" {
    const platform = zml.testing.env();

    const pipeline: Pipeline = .{ .platform = platform };

    // canonicalizeDown currently requires a gated projection; quantized weights
    // are not supported by gmmDType. Plain activations need a backend fix first.
    try pipeline.run(std.testing.io, std.testing.allocator, .mosaic_tpu, fixtures.matrices.mosaic_tpu, .{});
}
