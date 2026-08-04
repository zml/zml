const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");

const Buffer = @import("buffer.zig").Buffer;
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

        pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, shardings: []const Sharding, num_devices: usize) !Arguments {
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

    /// An executable together with reusable argument and result storage.
    ///
    /// A runner owns all three values and must not be used concurrently.
    pub const Runner = struct {
        exe: Exe,
        args: Arguments,
        results: Results,

        /// Takes ownership of the executable and allocates reusable argument and result storage.
        pub fn init(exe: Exe, allocator: std.mem.Allocator) !Runner {
            errdefer exe.deinit();
            var arguments = try exe.args(allocator);
            errdefer arguments.deinit(allocator);
            var results_ = try exe.results(allocator);
            errdefer results_.deinit(allocator);
            return .{ .exe = exe, .args = arguments, .results = results_ };
        }

        pub fn deinit(self: *Runner, allocator: std.mem.Allocator) void {
            self.results.deinit(allocator);
            self.args.deinit(allocator);
            self.exe.deinit();
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
        var events = [_]?*pjrt.Event{null} ** Platform.MAX_NUM_DEVICES;

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
            // even if it has been marked as "can be donated" during compilation.
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
                            ev.await(self.platform.pjrt_api, io.?) catch unreachable;
                        }
                        ev.deinit(self.platform.pjrt_api);
                    }
                }
            },
            .cpu, .cuda, .rocm, .tpu, .oneapi, .metal => if (opts.wait) {
                for (events_slice.?) |e| {
                    if (e) |ev| {
                        ev.await(self.platform.pjrt_api, io.?) catch unreachable;
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

/// A named collection of executables with matching argument and result sets.
///
/// The executable names are declared once as an enum:
///
/// ```zig
/// const Model = MultiExe(enum { embed, layer, sample });
/// const model = Model.init(.{
///     .embed = embed_exe,
///     .layer = layer_exe,
///     .sample = sample_exe,
/// });
/// ```
pub fn MultiExe(comptime Fields: type) type {
    const fieldInfo = switch (@typeInfo(Fields)) {
        .@"enum" => |info| info,
        else => @compileError("MultiExe expects an enum, got " ++ @typeName(Fields)),
    };
    if (!fieldInfo.is_exhaustive) {
        @compileError("MultiExe requires an exhaustive enum, got " ++ @typeName(Fields));
    }

    return struct {
        executables: Executables,

        pub const Field = Fields;
        pub const Executables = std.EnumArray(Field, Exe);

        const Self = @This();

        pub const Args = struct {
            values: std.EnumArray(Field, Exe.Arguments),

            pub fn deinit(self: *const Args, allocator: std.mem.Allocator) void {
                for (&self.values.values) |*arguments| {
                    arguments.deinit(allocator);
                }
            }

            pub fn at(self: *Args, field: Field) *Exe.Arguments {
                return self.values.getPtr(field);
            }

            pub fn atConst(self: *const Args, field: Field) *const Exe.Arguments {
                return self.values.getPtrConst(field);
            }

            pub fn set(self: *Args, field: Field, value: anytype) void {
                self.at(field).set(value);
            }

            pub fn setPartial(self: *Args, field: Field, value: anytype, offset: usize) void {
                self.at(field).setPartial(value, offset);
            }

            pub fn bake(self: *Args, field: Field, value: anytype) void {
                self.at(field).bake(value);
            }
        };

        pub const Results = struct {
            values: std.EnumArray(Field, Exe.Results),

            pub fn deinit(self: *const Results, allocator: std.mem.Allocator) void {
                for (&self.values.values) |*results_| {
                    results_.deinit(allocator);
                }
            }

            pub fn at(self: *Results, field: Field) *Exe.Results {
                return self.values.getPtr(field);
            }

            pub fn atConst(self: *const Results, field: Field) *const Exe.Results {
                return self.values.getPtrConst(field);
            }

            pub fn get(self: *Results, field: Field, comptime T: type) T {
                return self.at(field).get(T);
            }

            pub fn fill(self: *Results, field: Field, value: anytype) void {
                self.at(field).fill(value);
            }
        };

        /// A collection of executables together with reusable argument and result storage.
        ///
        /// A runner owns all three values and must not be used concurrently.
        pub const Runner = struct {
            exe: Self,
            args: Args,
            results: Results,

            /// Takes ownership of the executable collection and allocates reusable argument and result storage.
            pub fn init(exe: Self, allocator: std.mem.Allocator) !Runner {
                errdefer exe.deinit();
                var arguments = try exe.args(allocator);
                errdefer arguments.deinit(allocator);
                var results_ = try exe.results(allocator);
                errdefer results_.deinit(allocator);
                return .{ .exe = exe, .args = arguments, .results = results_ };
            }

            pub fn deinit(self: *Runner, allocator: std.mem.Allocator) void {
                self.results.deinit(allocator);
                self.args.deinit(allocator);
                self.exe.deinit();
            }

            pub fn run(self: *Runner, field: Field, input_values: anytype, output_values: anytype) void {
                self.args.set(field, input_values);
                self.exe.call(field, &self.args, &self.results);
                self.results.fill(field, output_values);
            }

            pub fn runOpts(self: *Runner, field: Field, io: std.Io, input_values: anytype, output_values: anytype, opts: Exe.CallOpts) void {
                self.args.set(field, input_values);
                self.exe.callOpts(field, io, &self.args, &self.results, opts);
                self.results.fill(field, output_values);
            }
        };

        pub fn init(executables: std.enums.EnumFieldStruct(Field, Exe, null)) Self {
            return .{ .executables = .init(executables) };
        }

        pub fn deinit(self: *const Self) void {
            for (&self.executables.values) |*exe| {
                exe.deinit();
            }
        }

        pub fn at(self: *const Self, field: Field) *const Exe {
            return self.executables.getPtrConst(field);
        }

        pub fn args(self: *const Self, allocator: std.mem.Allocator) !Args {
            var args_: Args = .{ .values = .initUndefined() };
            var initialized: usize = 0;
            errdefer for (args_.values.values[0..initialized]) |*arguments| {
                arguments.deinit(allocator);
            };

            for (&self.executables.values, &args_.values.values) |exe, *arguments| {
                arguments.* = try exe.args(allocator);
                initialized += 1;
            }
            return args_;
        }

        pub fn results(self: *const Self, allocator: std.mem.Allocator) !Results {
            var results_: Results = .{ .values = .initUndefined() };
            var initialized: usize = 0;
            errdefer for (results_.values.values[0..initialized]) |*result| {
                result.deinit(allocator);
            };

            for (&self.executables.values, &results_.values.values) |exe, *result| {
                result.* = try exe.results(allocator);
                initialized += 1;
            }
            return results_;
        }

        pub fn call(self: *const Self, field: Field, arguments: *const Args, results_: *Results) void {
            self.at(field).call(arguments.atConst(field).*, results_.at(field));
        }

        pub fn callOpts(self: *const Self, field: Field, io: std.Io, arguments: *const Args, results_: *Results, opts: Exe.CallOpts) void {
            self.at(field).callOpts(io, arguments.atConst(field).*, results_.at(field), opts);
        }
    };
}

/// A typed facade over `MultiExe` whose input and output buffer structures are
/// derived from the functions used to compile each executable.
pub fn TypedMultiExe(comptime functions: anytype) type {
    const function_map_info = switch (@typeInfo(@TypeOf(functions))) {
        .@"struct" => |info| info,
        else => @compileError("TypedMultiExe expects a struct function map, got " ++ @typeName(@TypeOf(functions))),
    };
    if (function_map_info.is_tuple) {
        @compileError("TypedMultiExe expects a function map with named fields");
    }

    inline for (function_map_info.fields) |field| {
        validateTypedFunction(field.name, @field(functions, field.name));
    }

    const FunctionMap = @TypeOf(functions);
    const FunctionField = std.meta.FieldEnum(FunctionMap);
    const RawMultiExe = MultiExe(FunctionField);

    return struct {
        raw: Raw,

        pub const Field = FunctionField;
        pub const Raw = RawMultiExe;

        const Self = @This();

        pub fn function(comptime field: Field) @TypeOf(@field(functions, @tagName(field))) {
            return @field(functions, @tagName(field));
        }

        /// Runtime input buffers for one executable. Non-Tensor function input
        /// fields are omitted by `Bufferized`.
        pub fn Inputs(comptime field: Field) type {
            return mem.Bufferized(typedFunctionInput(function(field)));
        }

        /// Mutable destinations for the top-level runtime output fields.
        pub fn OutputDestinations(comptime field: Field) type {
            return typedOutputDestinations(mem.Bufferized(typedFunctionOutput(function(field))));
        }

        pub fn Call(comptime field: Field) type {
            return struct {
                inputs: Inputs(field),
                outputs: OutputDestinations(field),
                opts: Exe.CallOpts = .{},
            };
        }

        pub fn init(executables: std.enums.EnumFieldStruct(Field, Exe, null)) Self {
            return .{ .raw = .init(executables) };
        }

        pub fn deinit(self: *const Self) void {
            self.raw.deinit();
        }

        pub const Runner = struct {
            raw: Raw.Runner,

            /// Takes ownership of the executable collection and allocates reusable argument and result storage.
            pub fn init(exe: Self, allocator: std.mem.Allocator) !Runner {
                return .{ .raw = try Raw.Runner.init(exe.raw, allocator) };
            }

            pub fn deinit(self: *Runner, allocator: std.mem.Allocator) void {
                self.raw.deinit(allocator);
            }

            pub fn run(self: *Runner, comptime field: Field, io: std.Io, call: Call(field)) void {
                self.raw.runOpts(field, io, call.inputs, call.outputs, call.opts);
            }
        };
    };
}

fn validateTypedFunction(comptime field_name: []const u8, comptime function_: anytype) void {
    const function_info = switch (@typeInfo(@TypeOf(function_))) {
        .@"fn" => |info| info,
        else => @compileError("TypedMultiExe field '" ++ field_name ++ "' must map to a function"),
    };
    if (function_info.is_var_args or function_info.params.len != 1) {
        @compileError("TypedMultiExe function '" ++ field_name ++ "' must accept exactly one input struct");
    }

    const Input = function_info.params[0].type orelse
        @compileError("TypedMultiExe function '" ++ field_name ++ "' must have a concrete input type");
    validateTypedNamedStruct(field_name, "input", Input);

    const Output = function_info.return_type orelse
        @compileError("TypedMultiExe function '" ++ field_name ++ "' must return an output struct");
    validateTypedNamedStruct(field_name, "output", Output);
}

fn validateTypedNamedStruct(comptime field_name: []const u8, comptime role: []const u8, comptime T: type) void {
    const info = switch (@typeInfo(T)) {
        .@"struct" => |info| info,
        else => @compileError("TypedMultiExe function '" ++ field_name ++ "' " ++ role ++ " must be a struct, got " ++ @typeName(T)),
    };
    if (info.is_tuple) {
        @compileError("TypedMultiExe function '" ++ field_name ++ "' " ++ role ++ " must use named fields");
    }
}

fn typedFunctionInput(comptime function_: anytype) type {
    return @typeInfo(@TypeOf(function_)).@"fn".params[0].type.?;
}

fn typedFunctionOutput(comptime function_: anytype) type {
    return @typeInfo(@TypeOf(function_)).@"fn".return_type.?;
}

fn typedOutputDestinations(comptime BufferizedOutput: type) type {
    if (BufferizedOutput == void) return struct {};

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

test "MultiExe manages matching argument and result sets" {
    const Model = MultiExe(enum { embed, layer, sample });
    const empty_shapes: []const Shape = &.{};
    const empty_shardings: []const Sharding = &.{};

    const Helpers = struct {
        fn emptyExe() Exe {
            return .{
                .platform = undefined,
                .exe = undefined,
                .input_shapes = empty_shapes,
                .output_shapes = empty_shapes,
                .input_shardings = empty_shardings,
                .output_shardings = empty_shardings,
                .num_devices = 0,
                .num_partitions = 0,
                .arena = .init(std.testing.allocator),
            };
        }
    };

    const model: Model = .init(.{
        .embed = Helpers.emptyExe(),
        .layer = Helpers.emptyExe(),
        .sample = Helpers.emptyExe(),
    });
    var runner = try Model.Runner.init(model, std.testing.allocator);
    defer {
        runner.results.deinit(std.testing.allocator);
        runner.args.deinit(std.testing.allocator);
        for (&runner.exe.executables.values) |*exe| exe.arena.deinit();
    }
    runner.args.bake(.embed, .{});
    _ = runner.results.get(.sample, struct {});

    try std.testing.expectEqual(@as(usize, 0), runner.args.at(.layer).baked_count);
    try std.testing.expect(runner.exe.at(.embed) == runner.exe.executables.getPtrConst(.embed));
}

test "TypedMultiExe derives named runtime calls from functions" {
    const EmbedInputs = struct {
        weights: struct {
            tensor: Tensor,
            scale: f32,
        },
        tokens: Tensor,
        token_count: usize,
    };
    const EmbedOutputs = struct {
        hidden: Tensor,
        metadata: usize,
    };
    const LayerInputs = struct {
        hidden: Tensor,
        layer_index: usize,
    };
    const LayerOutputs = struct {
        hidden: Tensor,
        cache: struct { key: Tensor, length: usize },
    };
    const Functions = struct {
        fn embed(inputs: EmbedInputs) EmbedOutputs {
            _ = inputs;
            return undefined;
        }

        fn layer(inputs: LayerInputs) LayerOutputs {
            _ = inputs;
            return undefined;
        }
    };

    const Model = TypedMultiExe(.{
        .embed = Functions.embed,
        .layer = Functions.layer,
    });

    const EmbedRuntimeInputs = Model.Inputs(.embed);
    try std.testing.expect(@hasField(EmbedRuntimeInputs, "weights"));
    try std.testing.expect(@hasField(EmbedRuntimeInputs, "tokens"));
    try std.testing.expect(!@hasField(EmbedRuntimeInputs, "token_count"));
    try std.testing.expect(!@hasField(@FieldType(EmbedRuntimeInputs, "weights"), "scale"));
    try std.testing.expectEqual(Buffer, @FieldType(EmbedRuntimeInputs, "tokens"));

    const LayerOutputDestinations = Model.OutputDestinations(.layer);
    try std.testing.expectEqual(*Buffer, @FieldType(LayerOutputDestinations, "hidden"));
    try std.testing.expectEqual(*mem.Bufferized(@FieldType(LayerOutputs, "cache")), @FieldType(LayerOutputDestinations, "cache"));

    const call: Model.Call(.embed) = .{
        .inputs = undefined,
        .outputs = undefined,
    };
    try std.testing.expect(!call.opts.wait);
    try std.testing.expectEqual(@TypeOf(Functions.embed), @TypeOf(Model.function(.embed)));
}
