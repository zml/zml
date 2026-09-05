const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
const Compiler = @import("Compiler.zig");
const mem = @import("mem.zig");
const meta = @import("meta.zig");
const Platform = @import("platform.zig").Platform;
const tracer = @import("profiling/tracer.zig");
const Shape = @import("shape.zig").Shape;
const Sharding = @import("Sharding.zig");
const Tensor = @import("tensor.zig").Tensor;

pub const Exe = struct {
    platform: *const Platform,
    exe: *pjrt.LoadedExecutable,

    context: ?*pjrt.ExecuteContext = null,

    input_shapes: []const Shape,
    output_shapes: []const Shape,

    input_shardings: []const Sharding,
    output_shardings: []const Sharding,

    num_devices: usize,
    num_partitions: i32,

    arena: std.heap.ArenaAllocator,

    pub fn init(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        exe: *pjrt.LoadedExecutable,
        num_devices: usize,
        num_partitions: i32,
        input_shapes: []const Shape,
        output_shapes: []const Shape,
        input_shardings: []const Sharding,
        output_shardings: []const Sharding,
    ) !Exe {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const input_shapes_copy = try arena.allocator().dupe(Shape, input_shapes);
        const output_shapes_copy = try arena.allocator().dupe(Shape, output_shapes);

        // Re-home sharding pointers into arena-owned values so exe doesn't depend on caller lifetimes.
        const input_shardings_copy = try arena.allocator().dupe(Sharding, input_shardings);
        const output_shardings_copy = try arena.allocator().dupe(Sharding, output_shardings);

        return .{
            .platform = platform,
            .exe = exe,
            .input_shapes = input_shapes_copy,
            .output_shapes = output_shapes_copy,
            .input_shardings = input_shardings_copy,
            .output_shardings = output_shardings_copy,
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
        return Arguments.init(allocator, self.input_shapes, self.input_shardings, self.num_devices);
    }

    pub fn results(self: *const Exe, allocator: std.mem.Allocator) !Results {
        return Results.init(allocator, self.output_shapes, self.output_shardings, self.platform, self.num_devices);
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
        shardings: []const Sharding,

        pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, shardings: []const Sharding, num_devices: usize) error{OutOfMemory}!Arguments {
            const flat_buffers = try FlatBuffers.init(allocator, shapes.len, num_devices);
            errdefer flat_buffers.deinit(allocator);

            const expected_shapes = try allocator.dupe(Shape, shapes);
            errdefer allocator.free(expected_shapes);

            return .{
                .flat_buffers = flat_buffers,
                .expected_shapes = expected_shapes,
                .shardings = shardings,
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
        shardings: []const Sharding,

        pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, shardings: []const Sharding, platform: *const Platform, num_devices: usize) !Results {
            const flat_buffers = try FlatBuffers.init(allocator, shapes.len, num_devices);
            errdefer flat_buffers.deinit(allocator);

            const expected_shapes = try allocator.dupe(Shape, shapes);
            errdefer allocator.free(expected_shapes);

            return .{
                .platform = platform,
                .flat_buffers = flat_buffers,
                .expected_shapes = expected_shapes,
                .shardings = shardings,
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
                    buffer.* = Buffer.fromPjrtBuffers(context_.self.platform, context_.self.expected_shapes[context_.current_index], context_.self.shardings[context_.current_index], shards.constSlice());
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
                    buffer.* = Buffer.fromPjrtBuffers(ctx.results.platform, ctx.results.expected_shapes[ctx.current_index], ctx.results.shardings[ctx.current_index], shards.constSlice());
                    ctx.current_index += 1;
                }
            }.cb, &context, &v);
        }
    };

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

        pub fn run(self: *Runner, input_values: anytype, output_values: anytype) void {
            self.args.set(input_values);
            self.exe.call(self.args, &self.results);
            self.results.fill(output_values);
        }

        pub fn runOpts(self: *Runner, io: std.Io, input_values: anytype, output_values: anytype, opts: CallOpts) void {
            self.args.set(input_values);
            self.exe.callOpts(io, self.args, &self.results, opts);
            self.results.fill(output_values);
        }
    };

    pub fn internalCall(self: *const Exe, io: ?std.Io, arguments: Arguments, results_: *Results, opts: CallOpts) void {
        stdx.debug.assert(opts.wait == false or io != null, "io should not be null when waiting for execution completion", .{});
        var events: [Platform.MAX_NUM_DEVICES]?*pjrt.Event = @splat(null);

        const partition_events = events[0..@intCast(self.num_partitions)];
        const events_slice: ?[]?*pjrt.Event = switch (self.platform.target) {
            .neuron => partition_events,
            .cpu, .cuda, .rocm, .tpu, .oneapi, .metal => if (opts.wait) partition_events else null,
        };

        self.exe.execute(self.platform.pjrt_api, .{
            .arguments = arguments.flat_buffers.buffers,
            .num_args = arguments.expected_shapes.len,
            .results = results_.flat_buffers.buffers,
            .events = events_slice,
            // this allows to tell a specific buffer shouldn't be donated,
            // even if it has been marked as "can be donated" during compiler.
            // TODO: expose it ?
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
                            ev.await(self.platform.pjrt_api, io.?) catch |err| {
                                std.debug.panic("PJRT execution failed with: {}", .{err});
                            };
                        }
                        ev.deinit(self.platform.pjrt_api);
                    }
                }
            },
            .cpu, .cuda, .rocm, .tpu, .oneapi, .metal => if (opts.wait) {
                for (events_slice.?) |e| {
                    if (e) |ev| {
                        ev.await(self.platform.pjrt_api, io.?) catch |err| {
                            std.debug.panic("PJRT execution failed with: {}", .{err});
                        };
                        ev.deinit(self.platform.pjrt_api);
                    }
                }
            },
        }
    }

    pub const CallOpts = struct {
        wait: bool = false,
    };

    pub fn callOpts(self: *const Exe, io: std.Io, arguments: Arguments, results_: *Results, opts: CallOpts) void {
        var span = tracer.span("zml.exe.call", .{
            .wait = opts.wait,
            .arg_count = arguments.expected_shapes.len,
            .result_count = results_.expected_shapes.len,
        });
        defer span.end();
        return self.internalCall(io, arguments, results_, opts);
    }

    pub fn call(self: *const Exe, arguments: Arguments, results_: *Results) void {
        var span = tracer.span("zml.exe.call", .{
            .wait = false,
            .arg_count = arguments.expected_shapes.len,
            .result_count = results_.expected_shapes.len,
        });
        defer span.end();
        return self.internalCall(null, arguments, results_, .{});
    }
};

/// A typed executable whose input and output buffer structures are derived from
/// the function used to compile it.
pub fn FnExe(comptime function_: anytype) type {
    const function_info = switch (@typeInfo(@TypeOf(function_))) {
        .@"fn" => |info| info,
        else => @compileError("FnExe expects a function, got " ++ @typeName(@TypeOf(function_))),
    };
    if (function_info.is_var_args or function_info.params.len != 1) {
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
                pub const NonBakedInput = structFieldRange(Input, count, @typeInfo(Input).@"struct".fields.len);

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

                pub fn rebake(self: *RunnerSelf, baked: BakedInput) void {
                    self.args.baked_count = 0;
                    self.args.bake(baked);
                }

                pub fn run(self: *RunnerSelf, io: std.Io, call: struct {
                    inputs: NonBakedInput,
                    outputs: Output,
                    opts: Exe.CallOpts = .{},
                }) void {
                    self.args.set(call.inputs);
                    self.exe.callOpts(io, self.args, &self.results, call.opts);
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

    const input_fields = @typeInfo(Input).@"struct".fields;
    if (baked_info.fields.len > input_fields.len) {
        @compileError("FnExe baked fields must be a prefix of its bufferized input fields");
    }

    inline for (baked_info.fields, 0..) |tuple_field, i| {
        const baked_field = @field(baked_fields, tuple_field.name);
        if (@TypeOf(baked_field) != @EnumLiteral()) {
            @compileError("FnExe baked fields must be enum literals");
        }
        if (!std.mem.eql(u8, @tagName(baked_field), input_fields[i].name)) {
            @compileError("FnExe baked fields must be an ordered prefix of its bufferized input fields");
        }
    }
    return baked_info.fields.len;
}

fn structFieldRange(comptime Struct: type, comptime start: usize, comptime end: usize) type {
    const fields = @typeInfo(Struct).@"struct".fields[start..end];
    var field_names: [fields.len][]const u8 = undefined;
    var field_types: [fields.len]type = undefined;
    var field_attrs: [fields.len]std.builtin.Type.StructField.Attributes = undefined;
    for (&field_names, &field_types, &field_attrs, fields) |*name, *T, *attrs, field| {
        name.* = field.name;
        T.* = field.type;
        attrs.* = .{
            .@"comptime" = field.is_comptime,
            .@"align" = field.alignment,
            .default_value_ptr = field.default_value_ptr,
        };
    }
    return @Struct(.auto, null, &field_names, &field_types, &field_attrs);
}

fn typedFunctionInput(comptime function_: anytype) type {
    const Input = @typeInfo(@TypeOf(function_)).@"fn".params[0].type orelse
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
    const fields = output_info.fields;
    var field_names: [fields.len][]const u8 = undefined;
    var field_types: [fields.len]type = undefined;
    var field_attrs: [fields.len]std.builtin.Type.StructField.Attributes = undefined;
    for (&field_names, &field_types, &field_attrs, fields) |*name, *T, *attrs, field| {
        name.* = field.name;
        T.* = *field.type;
        attrs.* = .{ .@"align" = @alignOf(*field.type) };
    }
    return @Struct(.auto, null, &field_names, &field_types, &field_attrs);
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
