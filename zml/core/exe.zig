const std = @import("std");
const builtin = @import("builtin");

const meta = @import("zml/meta");
const pjrt = @import("pjrt");
const stdx = @import("stdx");
const tracer = @import("zml/profiling").tracer;

const Buffer = @import("buffer.zig").Buffer;
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const Sharding = @import("Sharding.zig");
const Tensor = @import("tensor.zig").Tensor;

pub const Exe = struct {
    platform: *const Platform,
    exe: *pjrt.LoadedExecutable,

    context: ?*pjrt.ExecuteContext = null,

    input_shapes: []const Shape,
    output_shapes: []const Shape,

    /// Inputs whose buffers are donated to an output (`reuseBuffer`).
    /// Runners destroy their PJRT handles after the raw call returns.
    donated_input_indices: []const u32,

    num_devices: u32,
    num_partitions: u32,

    arena: std.heap.ArenaAllocator,

    pub fn init(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        exe: *pjrt.LoadedExecutable,
        num_devices: u32,
        num_partitions: u32,
        input_shapes: []const Shape,
        output_shapes: []const Shape,
        input_aliasing: []const ?u32,
    ) !Exe {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const input_shapes_copy = try arena.allocator().dupe(Shape, input_shapes);
        const output_shapes_copy = try arena.allocator().dupe(Shape, output_shapes);

        var donated_count: usize = 0;
        for (input_aliasing) |aliasing| donated_count += @intFromBool(aliasing != null);

        const donated_input_indices = try arena.allocator().alloc(u32, donated_count);

        var donated_index: usize = 0;
        for (input_aliasing, 0..) |aliasing, input_index| {
            if (aliasing == null) continue;
            donated_input_indices[donated_index] = @intCast(input_index);
            donated_index += 1;
        }

        return .{
            .platform = platform,
            .exe = exe,
            .input_shapes = input_shapes_copy,
            .output_shapes = output_shapes_copy,
            .donated_input_indices = donated_input_indices,
            .num_devices = num_devices,
            .num_partitions = num_partitions,
            .arena = arena,
        };
    }

    pub fn deinit(self: *const Exe) void {
        if (self.context) |context| context.deinit(self.platform.pjrt_api);
        self.exe.deinit(self.platform.pjrt_api);
        self.arena.deinit();
    }

    pub fn args(self: *const Exe, allocator: std.mem.Allocator) !Arguments {
        return Arguments.init(allocator, self.input_shapes, self.num_devices);
    }

    pub fn results(self: *const Exe, allocator: std.mem.Allocator) !Results {
        return Results.init(allocator, self.output_shapes, self.platform, self.num_devices);
    }

    pub const FlatBuffers = struct {
        buffers: []const [*]*pjrt.Buffer,
        raw_buffers: []const *pjrt.Buffer,

        num_devices: usize,

        pub fn init(allocator: std.mem.Allocator, count: usize, num_devices: usize) !FlatBuffers {
            const raw_buffers = try allocator.alloc(*pjrt.Buffer, num_devices * count);
            errdefer allocator.free(raw_buffers);

            const buffers = try allocator.alloc([*]*pjrt.Buffer, num_devices);
            errdefer allocator.free(buffers);

            for (0..num_devices) |i| {
                buffers[i] = raw_buffers[i * count ..].ptr;
            }

            return .{
                .buffers = buffers,
                .raw_buffers = raw_buffers,
                .num_devices = num_devices,
            };
        }

        pub fn deinit(self: *const FlatBuffers, allocator: std.mem.Allocator) void {
            allocator.free(self.buffers);
            allocator.free(self.raw_buffers);
        }
    };

    pub const Arguments = struct {
        flat_buffers: FlatBuffers,
        expected_shapes: []const Shape,
        baked_count: usize = 0,

        pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, num_devices: usize) error{OutOfMemory}!Arguments {
            const flat_buffers = try FlatBuffers.init(allocator, shapes.len, num_devices);
            errdefer flat_buffers.deinit(allocator);

            const expected_shapes = try allocator.dupe(Shape, shapes);
            errdefer allocator.free(expected_shapes);

            return .{
                .flat_buffers = flat_buffers,
                .expected_shapes = expected_shapes,
            };
        }

        pub fn deinit(self: *const Arguments, allocator: std.mem.Allocator) void {
            allocator.free(self.expected_shapes);
            self.flat_buffers.deinit(allocator);
        }

        pub fn set(self: *Arguments, v: anytype) void {
            return self.setPartial(v, 0);
        }

        pub fn setPartial(self: *Arguments, v: anytype, offset: usize) void {
            const LocalContext = struct {
                self: *Arguments,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .self = self, .current_index = offset + self.baked_count };
            meta.visit(struct {
                fn cb(context_: *LocalContext, buffer: *const Buffer) void {
                    stdx.debug.assert(
                        context_.self.expected_shapes[context_.current_index].eql(buffer.shape()),
                        "Expected argument {} to have shape {f}, got {f}",
                        .{ context_.current_index, context_.self.expected_shapes[context_.current_index], buffer.shape() },
                    );

                    const expected = context_.self.flat_buffers.num_devices;
                    const shard_count = buffer._shards.len;

                    stdx.debug.assert(
                        shard_count == expected,
                        "Argument {} has {d} shards but executable expects {d}",
                        .{ context_.current_index, shard_count, expected },
                    );

                    for (0..expected) |device_index| {
                        context_.self.flat_buffers.buffers[device_index][context_.current_index] =
                            buffer._shards.get(device_index);
                    }

                    context_.current_index += 1;
                }
            }.cb, &context, &v);
        }

        pub fn bake(self: *Arguments, v: anytype) void {
            const LocalContext = struct {
                self: *Arguments,

                current_index: usize = 0,
            };

            var context: LocalContext = .{ .self = self, .current_index = self.baked_count };

            meta.visit(struct {
                fn cb(context_: *LocalContext, buffer: *const Buffer) void {
                    stdx.debug.assert(context_.self.expected_shapes[context_.current_index].eql(buffer.shape()), "Expected argument {} to have shape {f}, got {f}", .{ context_.current_index, context_.self.expected_shapes[context_.current_index], buffer.shape() });

                    for (0..context_.self.flat_buffers.num_devices) |device_index| {
                        context_.self.flat_buffers.buffers[device_index][context_.current_index] = buffer._shards.get(device_index);
                    }

                    context_.current_index += 1;
                }
            }.cb, &context, &v);

            self.baked_count = context.current_index;
        }
    };

    pub const Results = struct {
        platform: *const Platform,
        flat_buffers: FlatBuffers,

        expected_shapes: []const Shape,

        pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, platform: *const Platform, num_devices: usize) !Results {
            const flat_buffers = try FlatBuffers.init(allocator, shapes.len, num_devices);
            errdefer flat_buffers.deinit(allocator);

            const expected_shapes = try allocator.dupe(Shape, shapes);
            errdefer allocator.free(expected_shapes);

            return .{
                .platform = platform,
                .flat_buffers = flat_buffers,
                .expected_shapes = expected_shapes,
            };
        }

        pub fn deinit(self: *const Results, allocator: std.mem.Allocator) void {
            allocator.free(self.expected_shapes);
            self.flat_buffers.deinit(allocator);
        }

        pub fn get(self: *Results, comptime T: type) T {
            var result: T = undefined;
            const LocalContext = struct {
                self: *Results,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .self = self, .current_index = 0 };
            meta.visit(struct {
                fn cb(context_: *LocalContext, buffer: *Buffer) void {
                    var shards: Buffer.Shards = .empty;
                    for (0..context_.self.flat_buffers.num_devices) |device_index| {
                        shards.appendAssumeCapacity(context_.self.flat_buffers.buffers[device_index][context_.current_index]);
                    }
                    buffer.* = Buffer.fromPjrtBuffers(context_.self.platform, context_.self.expected_shapes[context_.current_index], shards.constSlice());
                    context_.current_index += 1;
                }
            }.cb, &context, &result);
            return result;
        }

        pub fn fill(self: *Results, v: anytype) void {
            const LocalContext = struct {
                results: *Results,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .results = self, .current_index = 0 };
            meta.visit(struct {
                fn cb(ctx: *LocalContext, buffer: *Buffer) void {
                    //stdx.debug.assert(ctx.results.expected_shapes[ctx.current_index].eql(buffer.shape()), "Expected result {} to have shape {f}, got {f}", .{ ctx.current_index, ctx.results.expected_shapes[ctx.current_index], buffer.shape() });
                    var shards: Buffer.Shards = .empty;
                    for (0..ctx.results.flat_buffers.num_devices) |device_index| {
                        shards.appendAssumeCapacity(ctx.results.flat_buffers.buffers[device_index][ctx.current_index]);
                    }
                    buffer.* = Buffer.fromPjrtBuffers(ctx.results.platform, ctx.results.expected_shapes[ctx.current_index], shards.constSlice());
                    ctx.current_index += 1;
                }
            }.cb, &context, &v);
        }
    };

    /// Executes a single-output executable and waits for its result.
    /// The caller owns the returned buffer. Donated inputs are consumed.
    pub fn eval(self: *const Exe, allocator: std.mem.Allocator, io: std.Io, input_values: anytype) !Buffer {
        stdx.debug.assert(self.output_shapes.len == 1, "eval expects one output, got {d}; use a runner for multiple outputs", .{self.output_shapes.len});
        var result: Buffer = undefined;
        var r = try self.runner(allocator);
        defer r.deinit(allocator);
        r.run(io, input_values, .{&result}, .{ .wait = true });
        return result;
    }

    pub fn runner(self: *const Exe, allocator: std.mem.Allocator) !Runner {
        return .init(self, allocator);
    }

    /// Reusable argument and result storage for a borrowed executable.
    /// The executable must outlive the runner, and the runner must not be used concurrently.
    pub const Runner = struct {
        exe: *const Exe,
        args: Arguments,
        results: Results,

        pub fn init(exe: *const Exe, allocator: std.mem.Allocator) !Runner {
            var arguments = try exe.args(allocator);
            errdefer arguments.deinit(allocator);
            var results_ = try exe.results(allocator);
            errdefer results_.deinit(allocator);
            return .{
                .exe = exe,
                .args = arguments,
                .results = results_,
            };
        }

        pub fn deinit(self: *Runner, allocator: std.mem.Allocator) void {
            self.results.deinit(allocator);
            self.args.deinit(allocator);
        }

        /// Destroys donated input handles and fills the output destinations.
        /// Use the donated buffer's address as its output destination to replace it safely.
        /// Non-donated inputs remain owned by the caller.
        pub fn run(self: *Runner, io: std.Io, input_values: anytype, output_refs: anytype, opts: CallOpts) void {
            self.args.set(input_values);
            self.exe.call(io, self.args, &self.results, opts);
            self.exe.deinitDonatedInputs(self.args);
            self.results.fill(output_refs);
        }
    };

    pub const CallOpts = struct {
        /// Waits for execution to complete before returning.
        wait: bool = false,
    };

    fn deinitDonatedInputs(self: *const Exe, arguments: Arguments) void {
        // Execution deletes donated inputs, but leaves their PJRT handles allocated.
        // Arguments retain the old handles even when outputs replace the caller's buffers.
        for (self.donated_input_indices) |input_index| {
            for (arguments.flat_buffers.buffers[0..self.num_devices]) |device_buffers| {
                device_buffers[input_index].deinit(self.platform.pjrt_api);
            }
        }
    }

    /// Executes with optional waiting. The caller owns all input handles, including donated inputs.
    /// Donated input contents cannot be used after the call; output buffers have their own handles.
    /// Use a runner to destroy donated handles automatically and replace buffers through output destinations.
    pub fn call(self: *const Exe, io: std.Io, arguments: Arguments, results_: *Results, opts: CallOpts) void {
        var span = tracer.span("zml.exe.call", .{
            .wait = opts.wait,
            .arg_count = arguments.expected_shapes.len,
            .result_count = results_.expected_shapes.len,
        });
        defer span.end();

        var events: [Platform.MAX_NUM_DEVICES]?*pjrt.Event = @splat(null);

        const partition_events = events[0..@intCast(self.num_partitions)];
        const events_slice: ?[]?*pjrt.Event = switch (self.platform.target) {
            .neuron => partition_events,
            .cpu, .cuda, .rocm, .tpu, .oneapi, .metal, .furiosa => if (opts.wait) partition_events else null,
        };

        self.exe.execute(self.platform.pjrt_api, .{
            .arguments = arguments.flat_buffers.buffers,
            .num_args = arguments.expected_shapes.len,
            .results = results_.flat_buffers.buffers,
            .events = events_slice,
            // TODO: Set this flag for undonatable buffers.
            .non_donatable_input_indices = &.{},
            .context = self.context,
        }) catch |err| {
            std.debug.panic("PJRT_LoadedExecutable_Execute failed with: {}", .{err});
        };

        switch (self.platform.target) {
            .neuron => {
                for (events_slice.?) |e| {
                    if (e) |ev| {
                        if (opts.wait) {
                            ev.await(self.platform.pjrt_api, io) catch |err| {
                                std.debug.panic("PJRT execution failed with: {}", .{err});
                            };
                        }
                        ev.deinit(self.platform.pjrt_api);
                    }
                }
            },
            .cpu, .cuda, .rocm, .tpu, .oneapi, .metal, .furiosa => if (opts.wait) {
                for (events_slice.?) |e| {
                    if (e) |ev| {
                        ev.await(self.platform.pjrt_api, io) catch |err| {
                            std.debug.panic("PJRT execution failed with: {}", .{err});
                        };
                        ev.deinit(self.platform.pjrt_api);
                    }
                }
            },
        }
    }

    pub fn memoryStats(self: *const Exe) pjrt.ApiError!pjrt.CompiledMemoryStats {
        const exe = try self.exe.executable(self.platform.pjrt_api);
        return try exe.getCompiledMemoryStats(self.platform.pjrt_api);
    }
};

test "Exe raw calls leave donated input handle cleanup to the caller" {
    const platform = @import("testing.zig").env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Functions = struct {
        fn forward(value: Tensor, weight: Tensor) Tensor {
            return value.add(weight).reuseBuffer(value);
        }
    };
    const value: Tensor = .init(.{2}, .i32);
    const weight: Tensor = .init(.{2}, .i32);
    const exe = try platform.compileFn(allocator, io, Functions.forward, .{ value, weight }, .{});
    defer exe.deinit();
    try std.testing.expectEqualSlices(u32, &.{0}, exe.donated_input_indices);

    var weight_buffer = try Buffer.fromBytes(io, platform, weight.shape(), std.mem.asBytes(&[2]i32{ 10, 20 }));
    defer weight_buffer.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    for ([_]bool{ false, true }) |wait| {
        var value_buffer = try Buffer.fromBytes(io, platform, value.shape(), std.mem.asBytes(&[2]i32{ 1, 2 }));
        defer value_buffer.deinit();
        args.set(.{ value_buffer, weight_buffer });
        exe.call(io, args, &results, .{ .wait = wait });
        var output = results.get(Buffer);
        defer output.deinit();

        // The donated handle is still valid for cleanup even though its contents are deleted.
        for (value_buffer._shards.constSlice()) |shard| {
            try std.testing.expect(shard.isDeleted(platform.pjrt_api));
        }
        value_buffer.deinit();
        try std.testing.expectEqual([2]i32{ 11, 22 }, try output.getValue([2]i32, io));
        try std.testing.expectEqual([2]i32{ 10, 20 }, try weight_buffer.getValue([2]i32, io));
    }
}

test "Exe runner replaces donated buffers with and without waiting" {
    const platform = @import("testing.zig").env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Functions = struct {
        fn forward(weight: Tensor, value: Tensor) Tensor {
            return value.add(weight).reuseBuffer(value);
        }
    };
    const weight: Tensor = .init(.{2}, .i32);
    const value: Tensor = .init(.{2}, .i32);
    const exe = try platform.compileFn(allocator, io, Functions.forward, .{ weight, value }, .{});
    defer exe.deinit();

    var weight_buffer = try Buffer.fromBytes(io, platform, weight.shape(), std.mem.asBytes(&[2]i32{ 10, 20 }));
    defer weight_buffer.deinit();
    var value_buffer = try Buffer.fromBytes(io, platform, value.shape(), std.mem.asBytes(&[2]i32{ 1, 2 }));
    defer value_buffer.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    runner.args.bake(weight_buffer);

    runner.run(io, .{value_buffer}, .{&value_buffer}, .{});
    try std.testing.expectEqual([2]i32{ 11, 22 }, try value_buffer.getValue([2]i32, io));

    runner.run(io, .{value_buffer}, .{&value_buffer}, .{ .wait = true });
    try std.testing.expectEqual([2]i32{ 21, 42 }, try value_buffer.getValue([2]i32, io));
}

test "Exe runner replaces unchanged donated buffers" {
    const platform = @import("testing.zig").env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Functions = struct {
        fn forward(value: Tensor) Tensor {
            return value.reuseBuffer(value);
        }
    };
    const value: Tensor = .init(.{2}, .i32);
    const exe = try platform.compileFn(allocator, io, Functions.forward, .{value}, .{});
    defer exe.deinit();
    try std.testing.expectEqualSlices(u32, &.{0}, exe.donated_input_indices);

    var value_buffer = try Buffer.fromBytes(io, platform, value.shape(), std.mem.asBytes(&[2]i32{ 1, 2 }));
    defer value_buffer.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    for ([_]bool{ false, true }) |wait| {
        runner.run(io, .{value_buffer}, .{&value_buffer}, .{ .wait = wait });
        try std.testing.expectEqual([2]i32{ 1, 2 }, try value_buffer.getValue([2]i32, io));
    }
}
