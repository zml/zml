const std = @import("std");
const builtin = @import("builtin");

const meta = @import("zml/meta");
const pjrt = @import("pjrt");
const stdx = @import("stdx");
const tracer = @import("zml/profiling").tracer;

const Buffer = @import("buffer.zig").Buffer;
const Compiler = @import("Compiler.zig");
const mem = @import("mem.zig");
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const Sharding = @import("Sharding.zig");
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/exe");

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
        self.tryCall(io, arguments, results_, opts) catch |err| {
            std.debug.panic("PJRT execution failed with: {}", .{err});
        };
    }

    /// `call` that returns a failed execution instead of panicking, for a caller
    /// that must release its devices before exiting (an aborted process leaves
    /// Tenstorrent cards unusable until they are reset).
    pub fn tryCall(self: *const Exe, io: std.Io, arguments: Arguments, results_: *Results, opts: CallOpts) !void {
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
            .cpu, .cuda, .rocm, .tpu, .oneapi, .metal, .furiosa, .tenstorrent => if (opts.wait) partition_events else null,
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
            log.err("PJRT_LoadedExecutable_Execute failed with: {}", .{err});
            return err;
        };

        // Neuron always returns events, awaited only on request; the other
        // targets return them only when asked to wait.
        const events_ = events_slice orelse return;
        for (events_, 0..) |e, i| {
            const ev = e orelse continue;
            if (opts.wait) ev.await(self.platform.pjrt_api, io) catch |err| {
                for (events_[i..]) |rest| if (rest) |r| r.deinit(self.platform.pjrt_api);
                return err;
            };
            ev.deinit(self.platform.pjrt_api);
        }
    }

    /// The device layouts of the inputs that hold `model`'s tensors, for a
    /// function whose first argument is `model` (a model's weights): one per
    /// tensor, in `meta.visit` order, allocated in `arena`. The loader uploads
    /// each tensor in its layout (`zml.io.Loader.LoadOpts.layouts`). They are
    /// the platform's default layouts unless the executable was compiled with
    /// `weight_layouts = .executable` and its compiler chose another (on
    /// Tenstorrent, a matmul's weight in the order the matmul reads). Null on
    /// a platform without the Layouts extension, whose buffers always take
    /// the default layout.
    pub fn modelLayouts(self: *const Exe, arena: std.mem.Allocator, model: anytype) !?[]const pjrt.MemoryLayout {
        if (self.platform.pjrt_api.extension(.layouts) == null) return null;
        const executable = try self.exe.executable(self.platform.pjrt_api);
        defer executable.deinit(self.platform.pjrt_api);
        const parsed = try executable.parameterLayouts(self.platform.pjrt_api, arena);
        const count = meta.count(Tensor, model);
        if (count > parsed.len or count > self.input_shapes.len) {
            log.err("{} model tensors for an executable of {} inputs", .{ count, self.input_shapes.len });
            return error.ModelIsNotTheFirstArgument;
        }
        const Context = struct { exe: *const Exe, mismatch: ?usize = null };
        var context: Context = .{ .exe = self };
        meta.forEachVisit(model, *const Tensor, struct {
            fn call(i: usize, tensor: *const Tensor, ctx: *Context) void {
                if (ctx.mismatch == null and !ctx.exe.input_shapes[i].eql(tensor.shape())) ctx.mismatch = i;
            }
        }.call, .{&context});
        if (context.mismatch) |i| {
            log.err("model tensor {} does not match input {} {f} of the executable", .{ i, i, self.input_shapes[i] });
            return error.ModelIsNotTheFirstArgument;
        }
        const layouts = try arena.alloc(pjrt.MemoryLayout, count);
        for (layouts, parsed[0..count]) |*layout, *default| layout.* = default.toMemoryLayout();
        return layouts;
    }

    /// The layouts every executable of `exes` takes `model`'s tensors in
    /// (`modelLayouts`), which must agree: one buffer feeds all of them. A
    /// tensor two executables want in different layouts is an error naming
    /// it and both layouts.
    pub fn commonModelLayouts(arena: std.mem.Allocator, exes: []const *const Exe, model: anytype) !?[]const pjrt.MemoryLayout {
        if (exes.len == 0) return null;
        const first = try exes[0].modelLayouts(arena, model) orelse return null;
        for (exes[1..], 1..) |exe, e| {
            const other = (try exe.modelLayouts(arena, model)).?;
            for (first, other, 0..) |a, b, i| {
                if (!a.eql(b)) {
                    log.err("model tensor {} {f} is wanted in layout {f} by executable 0 and {f} by executable {}; compile them with one layout for it", .{ i, exes[0].input_shapes[i], a, b, e });
                    return error.ModelLayoutsDiffer;
                }
            }
        }
        return first;
    }

    pub fn memoryStats(self: *const Exe) pjrt.ApiError!pjrt.CompiledMemoryStats {
        const exe = try self.exe.executable(self.platform.pjrt_api);
        return try exe.getCompiledMemoryStats(self.platform.pjrt_api);
    }
};

/// A typed executable whose input and output buffer structures are derived from
/// the function used to compile it.
pub fn FnExe(comptime function_: anytype) type {
    const function_info = switch (@typeInfo(@TypeOf(function_))) {
        .@"fn" => |info| info,
        else => @compileError("FnExe expects a function, got " ++ @typeName(@TypeOf(function_))),
    };
    if (function_info.attrs.varargs or function_info.param_types.len != 1) {
        @compileError("FnExe function must accept exactly one input struct");
    }

    const FunctionInput = typedFunctionInput(function_);
    const FunctionOutput = typedFunctionOutput(function_);

    return struct {
        raw: Exe,

        pub const Input = mem.Bufferized(FunctionInput);
        pub const Output = typedOutputDestinations(mem.Bufferized(FunctionOutput));

        const Self = @This();

        pub fn init(raw: Exe) Self {
            return .{ .raw = raw };
        }

        pub fn compile(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const Platform,
            opts: Compiler.Options,
            args: std.meta.ArgsTuple(@TypeOf(function_)),
        ) Compiler.Error!Self {
            return .{ .raw = try Compiler.Typed(function_).compile(allocator, io, platform, opts, args) };
        }

        pub fn deinit(self: *const Self) void {
            self.raw.deinit();
        }

        pub fn runner(self: *const Self, allocator: std.mem.Allocator) !Runner(.{}) {
            return .init(self, allocator, .{});
        }

        /// Reusable argument and result storage for a borrowed executable.
        /// The executable must outlive the runner, and the runner must not be used concurrently.
        pub fn Runner(comptime baked_fields: anytype) type {
            const count = countBackedFields(Input, baked_fields);

            return struct {
                exe: *const Exe,
                args: Exe.Arguments,
                results: Exe.Results,

                const RunnerSelf = @This();

                pub const BakedInput = structFieldRange(Input, 0, count);
                pub const NonBakedInput = structFieldRange(Input, count, @typeInfo(Input).@"struct".field_names.len);

                pub fn init(exe: *const Self, allocator: std.mem.Allocator, baked: BakedInput) !RunnerSelf {
                    var arguments = try exe.raw.args(allocator);
                    errdefer arguments.deinit(allocator);
                    var results = try exe.raw.results(allocator);
                    errdefer results.deinit(allocator);
                    arguments.bake(baked);
                    return .{ .exe = &exe.raw, .args = arguments, .results = results };
                }

                pub fn deinit(self: *RunnerSelf, allocator: std.mem.Allocator) void {
                    self.results.deinit(allocator);
                    self.args.deinit(allocator);
                }

                pub const Call = struct {
                    inputs: NonBakedInput,
                    outputs: Output,
                };

                /// Destroys donated input handles before filling outputs; non-donated inputs remain caller-owned.
                pub fn run(self: *RunnerSelf, io: std.Io, call: Call, opts: Exe.CallOpts) void {
                    self.args.set(call.inputs);
                    self.exe.call(io, self.args, &self.results, opts);
                    self.exe.deinitDonatedInputs(self.args);
                    self.results.fill(call.outputs);
                }

                /// `run` that returns a failed execution (see `Exe.tryCall`); the
                /// inputs then stay caller-owned and the outputs are not filled.
                pub fn tryRun(self: *RunnerSelf, io: std.Io, call: Call, opts: Exe.CallOpts) !void {
                    self.args.set(call.inputs);
                    try self.exe.tryCall(io, self.args, &self.results, opts);
                    self.exe.deinitDonatedInputs(self.args);
                    self.results.fill(call.outputs);
                }
            };
        }
    };
}

fn countBackedFields(comptime Input: type, comptime baked_fields: anytype) usize {
    const baked_info = switch (@typeInfo(@TypeOf(baked_fields))) {
        .@"struct" => |info| info,
        else => @compileError("FnExe baked fields must be a tuple of enum literals"),
    };
    if (!baked_info.is_tuple) {
        @compileError("FnExe baked fields must be a tuple of enum literals");
    }

    const input_info = @typeInfo(Input).@"struct";
    if (baked_info.field_names.len > input_info.field_names.len) {
        @compileError("FnExe baked fields must be a prefix of its bufferized input fields");
    }

    inline for (baked_info.field_names, 0..) |tuple_field_name, i| {
        const baked_field = @field(baked_fields, tuple_field_name);
        if (@TypeOf(baked_field) != @EnumLiteral()) {
            @compileError("FnExe baked fields must be enum literals");
        }
        if (!std.mem.eql(u8, @tagName(baked_field), input_info.field_names[i])) {
            @compileError("FnExe baked fields must be an ordered prefix of its bufferized input fields");
        }
    }
    return baked_info.field_names.len;
}

fn structFieldRange(comptime Struct: type, comptime start: usize, comptime end: usize) type {
    const info = @typeInfo(Struct).@"struct";
    return @Struct(.auto, null, info.field_names[start..end], info.field_types[start..end], info.field_attrs[start..end]);
}

fn typedFunctionInput(comptime function_: anytype) type {
    const Input = @typeInfo(@TypeOf(function_)).@"fn".param_types[0] orelse
        @compileError("FnExe function must have a concrete input type");
    validateFnExeStruct("input", Input);
    return Input;
}

fn typedFunctionOutput(comptime function_: anytype) type {
    const Output = @typeInfo(@TypeOf(function_)).@"fn".return_type orelse @compileError("FnExe function must return an output struct");
    validateFnExeStruct("output", Output);
    return Output;
}

fn validateFnExeStruct(comptime role: []const u8, comptime T: type) void {
    const info = switch (@typeInfo(T)) {
        .@"struct" => |info| info,
        else => @compileError("FnExe " ++ role ++ " must be a struct, got " ++ @typeName(T)),
    };
    if (info.is_tuple) {
        @compileError("FnExe " ++ role ++ " must use named fields");
    }
}

fn typedOutputDestinations(comptime BufferizedOutput: type) type {
    const output_info = @typeInfo(BufferizedOutput).@"struct";
    var field_types: [output_info.field_names.len]type = undefined;
    var field_attrs: [output_info.field_names.len]std.builtin.Type.Struct.FieldAttributes = undefined;
    for (&field_types, &field_attrs, output_info.field_types) |*T, *attrs, Field| {
        T.* = *Field;
        attrs.* = .{ .@"align" = @alignOf(*Field) };
    }
    return @Struct(.auto, null, output_info.field_names, &field_types, &field_attrs);
}

test "FnExe derives baked and runtime calls from a function" {
    const Inputs = struct {
        weights: struct {
            tensor: Tensor,
            scale: f32,
        },
        bias: Tensor,
        tokens: Tensor,
        token_count: usize,
    };
    const Outputs = struct {
        hidden: Tensor,
        cache: struct {
            key: Tensor,
            length: usize,
        },
        metadata: usize,
    };
    const Functions = struct {
        fn forward(inputs: Inputs) Outputs {
            _ = inputs;
            return undefined;
        }
    };

    const Model = FnExe(Functions.forward);
    try std.testing.expect(@hasField(Model.Input, "weights"));
    try std.testing.expect(@hasField(Model.Input, "bias"));
    try std.testing.expect(@hasField(Model.Input, "tokens"));
    try std.testing.expect(!@hasField(Model.Input, "token_count"));
    try std.testing.expect(!@hasField(@FieldType(Model.Input, "weights"), "scale"));

    const Baked = Model.Runner(.{ .weights, .bias }).BakedInput;
    try std.testing.expect(@hasField(Baked, "weights"));
    try std.testing.expect(@hasField(Baked, "bias"));
    try std.testing.expect(!@hasField(Baked, "tokens"));

    const Runtime = Model.Runner(.{ .weights, .bias }).NonBakedInput;
    try std.testing.expect(!@hasField(Runtime, "weights"));
    try std.testing.expect(!@hasField(Runtime, "bias"));
    try std.testing.expectEqual(Buffer, @FieldType(Runtime, "tokens"));

    try std.testing.expectEqual(*Buffer, @FieldType(Model.Output, "hidden"));
    try std.testing.expectEqual(
        *mem.Bufferized(@FieldType(Outputs, "cache")),
        @FieldType(Model.Output, "cache"),
    );
    try std.testing.expect(!@hasField(Model.Output, "metadata"));
}

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

test "FnExe runner replaces donated buffers with and without waiting" {
    const platform = @import("testing.zig").env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Functions = struct {
        fn forward(inputs: struct { weight: Tensor, value: Tensor }) struct { value: Tensor } {
            return .{ .value = inputs.value.add(inputs.weight).reuseBuffer(inputs.value) };
        }
    };
    const Model = FnExe(Functions.forward);
    const weight: Tensor = .init(.{2}, .i32);
    const value: Tensor = .init(.{2}, .i32);
    const exe = try Model.compile(allocator, io, platform, .{}, .{.{ .weight = weight, .value = value }});
    defer exe.deinit();

    var weight_buffer = try Buffer.fromBytes(io, platform, weight.shape(), std.mem.asBytes(&[2]i32{ 10, 20 }));
    defer weight_buffer.deinit();
    var value_buffer = try Buffer.fromBytes(io, platform, value.shape(), std.mem.asBytes(&[2]i32{ 1, 2 }));
    defer value_buffer.deinit();
    var runner = try Model.Runner(.{.weight}).init(&exe, allocator, .{ .weight = weight_buffer });
    defer runner.deinit(allocator);

    runner.run(io, .{
        .inputs = .{ .value = value_buffer },
        .outputs = .{ .value = &value_buffer },
    }, .{});
    try std.testing.expectEqual([2]i32{ 11, 22 }, try value_buffer.getValue([2]i32, io));

    runner.run(io, .{
        .inputs = .{ .value = value_buffer },
        .outputs = .{ .value = &value_buffer },
    }, .{ .wait = true });
    try std.testing.expectEqual([2]i32{ 21, 42 }, try value_buffer.getValue([2]i32, io));
}

test "FnExe runner replaces unchanged donated buffers" {
    const platform = @import("testing.zig").env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Functions = struct {
        fn forward(inputs: struct { value: Tensor }) struct { value: Tensor } {
            return .{ .value = inputs.value.reuseBuffer(inputs.value) };
        }
    };
    const Model = FnExe(Functions.forward);
    const value: Tensor = .init(.{2}, .i32);
    const exe = try Model.compile(allocator, io, platform, .{}, .{.{ .value = value }});
    defer exe.deinit();
    try std.testing.expectEqualSlices(u32, &.{0}, exe.raw.donated_input_indices);

    var value_buffer = try Buffer.fromBytes(io, platform, value.shape(), std.mem.asBytes(&[2]i32{ 1, 2 }));
    defer value_buffer.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    for ([_]bool{ false, true }) |wait| {
        runner.run(io, .{
            .inputs = .{ .value = value_buffer },
            .outputs = .{ .value = &value_buffer },
        }, .{ .wait = wait });
        try std.testing.expectEqual([2]i32{ 1, 2 }, try value_buffer.getValue([2]i32, io));
    }
}
