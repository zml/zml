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

/// An untyped executable: inputs and outputs are flattened buffer lists.
/// Prefer the typed `Exe(function)`; this is its `raw` field.
pub const RawExe = struct {
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
    ) !RawExe {
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

    pub fn deinit(self: *const RawExe) void {
        if (self.context) |context| context.deinit(self.platform.pjrt_api);
        self.exe.deinit(self.platform.pjrt_api);
        self.arena.deinit();
    }

    pub fn args(self: *const RawExe, allocator: std.mem.Allocator) !Arguments {
        return Arguments.init(allocator, self.input_shapes, self.num_devices);
    }

    pub fn results(self: *const RawExe, allocator: std.mem.Allocator) !Results {
        return Results.init(allocator, self.output_shapes, self.platform, self.num_devices);
    }

    /// PJRT buffer handles for one side of a call (arguments or results), laid out as
    /// `PJRT_LoadedExecutable_Execute` expects: one row per device, one column per
    /// flattened tensor, in `meta.visit` order.
    /// Only handles are stored; their buffers are owned by the caller's `Buffer`s.
    pub const FlatBuffers = struct {
        /// Row pointers into `raw_buffers`: `buffers[device][index]`.
        buffers: []const [*]*pjrt.Buffer,
        /// Backing storage for all rows, `num_devices * count` handles, device-major.
        raw_buffers: []const *pjrt.Buffer,

        num_devices: usize,

        /// Allocates `count` handle slots per device. The slots are undefined until set.
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
    pub fn eval(self: *const RawExe, allocator: std.mem.Allocator, io: std.Io, input_values: anytype) !Buffer {
        stdx.debug.assert(self.output_shapes.len == 1, "eval expects one output, got {d}; use a runner for multiple outputs", .{self.output_shapes.len});
        var result: Buffer = undefined;
        var r = try self.runner(allocator);
        defer r.deinit(allocator);
        r.run(io, input_values, .{&result}, .{ .wait = true });
        return result;
    }

    pub fn runner(self: *const RawExe, allocator: std.mem.Allocator) !Runner {
        return .init(self, allocator);
    }

    /// Reusable argument and result storage for a borrowed executable.
    /// The executable must outlive the runner, and the runner must not be used concurrently.
    pub const Runner = struct {
        exe: *const RawExe,
        args: Arguments,
        results: Results,

        pub fn init(exe: *const RawExe, allocator: std.mem.Allocator) !Runner {
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

    fn deinitDonatedInputs(self: *const RawExe, arguments: Arguments) void {
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
    pub fn call(self: *const RawExe, io: std.Io, arguments: Arguments, results_: *Results, opts: CallOpts) void {
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

    pub fn memoryStats(self: *const RawExe) pjrt.ApiError!pjrt.CompiledMemoryStats {
        const exe = try self.exe.executable(self.platform.pjrt_api);
        return try exe.getCompiledMemoryStats(self.platform.pjrt_api);
    }
};

/// Deprecated alias of `Exe`.
pub const FnExe = Exe;

/// A typed executable whose input and output buffer structures are derived from
/// the function used to compile it.
///
/// A function taking a single named struct (`fn (inputs: Inputs) Outputs`) is called
/// with `Bufferized(Inputs)`, and its leading fields can be baked by name:
/// `Runner(.{ .weights })`.
/// Any other function (`fn (self: Model, x: Tensor, cache: Cache) Outputs`) is called
/// with the tuple `Bufferized(ArgsTuple(function))`. Arguments without tensors are
/// dropped and the remaining ones renumbered, so `Runner(.{.@"0"})` bakes the first
/// argument that holds tensors.
///
/// Use `raw` for untyped access, e.g. to store executables of different functions together.
pub fn Exe(comptime function_: anytype) type {
    const Function = @TypeOf(function_);
    const function_info = switch (@typeInfo(Function)) {
        .@"fn" => |info| info,
        else => @compileError("Exe expects a function, got " ++ @typeName(Function)),
    };
    if (function_info.attrs.varargs) {
        @compileError("Exe function can't be variadic: " ++ @typeName(Function));
    }
    const FunctionOutput = function_info.return_type orelse
        @compileError("Exe function must have a concrete return type: " ++ @typeName(Function));

    return struct {
        raw: RawExe,

        /// The function this executable was compiled from.
        pub const function = function_;
        /// Buffers passed to `eval` and `Runner.run`.
        pub const Input = NonVoid(mem.Bufferized(FunctionInput(function_)));
        /// Buffers returned by `eval`.
        pub const Result = mem.Bufferized(FunctionOutput);
        /// Destinations filled by `Runner.run`.
        pub const Output = OutputDestinations(Result);

        const Self = @This();

        pub fn compile(
            allocator: std.mem.Allocator,
            io: std.Io,
            platform: *const Platform,
            opts: Compiler.Options,
            args: std.meta.ArgsTuple(Function),
        ) Compiler.Error!Self {
            return Compiler.compile(allocator, io, platform, function_, args, opts);
        }

        pub fn deinit(self: *const Self) void {
            self.raw.deinit();
        }

        pub fn memoryStats(self: *const Self) pjrt.ApiError!pjrt.CompiledMemoryStats {
            return self.raw.memoryStats();
        }

        /// Executes and waits for the result. The caller owns the returned buffers.
        /// Donated inputs are consumed.
        pub fn eval(self: *const Self, allocator: std.mem.Allocator, io: std.Io, inputs: Input) !Result {
            if (Output == Result) @compileError("Exe.eval can't size a slice result, use a runner: " ++ @typeName(Result));
            var r = try self.runner(allocator);
            defer r.deinit(allocator);
            r.args.set(inputs);
            self.raw.call(io, r.args, &r.results, .{ .wait = true });
            self.raw.deinitDonatedInputs(r.args);
            var result: Result = undefined;
            r.results.fill(&result);
            return result;
        }

        pub fn runner(self: *const Self, allocator: std.mem.Allocator) !Runner(.{}) {
            return .init(self, allocator, .{});
        }

        /// Reusable argument and result storage for a borrowed executable.
        /// `baked_fields` is an ordered prefix of `Input` field names, set once at init.
        /// The executable must outlive the runner, and the runner must not be used concurrently.
        pub fn Runner(comptime baked_fields: anytype) type {
            const count = countBakedFields(Input, baked_fields);

            return struct {
                exe: *const RawExe,
                args: RawExe.Arguments,
                results: RawExe.Results,

                const RunnerSelf = @This();

                pub const BakedInput = structFieldRange(Input, 0, count);
                pub const NonBakedInput = if (count == 0) Input else structFieldRange(Input, count, @typeInfo(Input).@"struct".field_names.len);

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

                /// Destroys donated input handles and fills the output destinations.
                /// Use the donated buffer's address as its output destination to replace it safely.
                /// Non-donated inputs remain owned by the caller.
                pub fn run(self: *RunnerSelf, io: std.Io, inputs: NonBakedInput, outputs: Output, opts: RawExe.CallOpts) void {
                    self.args.set(inputs);
                    self.exe.call(io, self.args, &self.results, opts);
                    self.exe.deinitDonatedInputs(self.args);
                    self.results.fill(outputs);
                }
            };
        }
    };
}

/// The value bufferized into an executable's input: the single named struct argument,
/// or the arguments tuple otherwise.
fn FunctionInput(comptime function_: anytype) type {
    const Function = @TypeOf(function_);
    const param_types = @typeInfo(Function).@"fn".param_types;
    for (param_types) |Param| {
        if (Param == null) @compileError("Exe function must have concrete parameter types: " ++ @typeName(Function));
    }
    if (param_types.len == 1 and isNamedInputStruct(param_types[0].?)) return param_types[0].?;
    return std.meta.ArgsTuple(Function);
}

fn isNamedInputStruct(comptime T: type) bool {
    if (T == Tensor) return false;
    return switch (@typeInfo(T)) {
        .@"struct" => |info| !info.is_tuple,
        else => false,
    };
}

/// Functions without tensor inputs are called with `.{}`.
fn NonVoid(comptime T: type) type {
    return if (T == void) @Tuple(&.{}) else T;
}

/// Destinations mirror the top level of the result: a struct or tuple of pointers to
/// each field, an array of pointers to each element, the slice itself (pre-sized by the caller),
/// or a pointer to the whole result otherwise (`*Buffer` for a `Tensor`).
fn OutputDestinations(comptime Result: type) type {
    if (Result == void) return @Tuple(&.{});
    const info = switch (@typeInfo(Result)) {
        .@"struct" => |info| if (Result == Buffer) return *Result else info,
        .array => |info| return [info.len]*info.child,
        .pointer => |info| if (info.size == .slice and !info.attrs.@"const") return Result else return *Result,
        else => return *Result,
    };
    var field_types: [info.field_names.len]type = undefined;
    var field_attrs: [info.field_names.len]std.builtin.Type.Struct.FieldAttributes = undefined;
    for (&field_types, &field_attrs, info.field_types) |*T, *attrs, Field| {
        T.* = *Field;
        attrs.* = .{ .@"align" = @alignOf(*Field) };
    }
    if (info.is_tuple) return @Tuple(&field_types);
    return @Struct(.auto, null, info.field_names, &field_types, &field_attrs);
}

fn countBakedFields(comptime Input: type, comptime baked_fields: anytype) usize {
    const baked_info = switch (@typeInfo(@TypeOf(baked_fields))) {
        .@"struct" => |info| info,
        else => @compileError("Exe baked fields must be a tuple of enum literals"),
    };
    if (!baked_info.is_tuple) {
        @compileError("Exe baked fields must be a tuple of enum literals");
    }
    if (baked_info.field_names.len == 0) return 0;

    const input_info = @typeInfo(Input).@"struct";
    if (baked_info.field_names.len > input_info.field_names.len) {
        @compileError("Exe baked fields must be a prefix of its bufferized input fields");
    }

    inline for (baked_info.field_names, 0..) |tuple_field_name, i| {
        const baked_field = @field(baked_fields, tuple_field_name);
        if (@TypeOf(baked_field) != @EnumLiteral()) {
            @compileError("Exe baked fields must be enum literals");
        }
        if (!std.mem.eql(u8, @tagName(baked_field), input_info.field_names[i])) {
            @compileError("Exe baked fields must be an ordered prefix of its bufferized input fields, expected ." ++ input_info.field_names[i] ++ ", got ." ++ @tagName(baked_field));
        }
    }
    return baked_info.field_names.len;
}

fn structFieldRange(comptime Struct: type, comptime start: usize, comptime end: usize) type {
    const info = @typeInfo(Struct).@"struct";
    if (info.is_tuple) return @Tuple(info.field_types[start..end]);
    return @Struct(.auto, null, info.field_names[start..end], info.field_types[start..end], info.field_attrs[start..end]);
}

test "Exe derives baked and runtime calls from a struct function" {
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

    const Model = Exe(Functions.forward);
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

test "Exe derives baked and runtime calls from a positional function" {
    const Model = struct {
        weight: Tensor,
        scale: f32,
    };
    const Cache = struct { key: Tensor };
    const Functions = struct {
        fn forward(self: Model, token_count: usize, x: Tensor, cache: Cache) struct { Tensor, Cache } {
            _ = self;
            _ = token_count;
            _ = x;
            _ = cache;
            return undefined;
        }
        fn single(x: Tensor) Tensor {
            return x;
        }
        fn noInput(count: usize) Tensor {
            _ = count;
            return undefined;
        }
    };

    const Forward = Exe(Functions.forward);
    // `token_count` holds no tensor: it is dropped and the following arguments renumbered.
    try std.testing.expectEqual(3, @typeInfo(Forward.Input).@"struct".field_names.len);
    try std.testing.expectEqual(mem.Bufferized(Model), @FieldType(Forward.Input, "0"));
    try std.testing.expectEqual(Buffer, @FieldType(Forward.Input, "1"));
    try std.testing.expectEqual(mem.Bufferized(Cache), @FieldType(Forward.Input, "2"));

    const Runner = Forward.Runner(.{.@"0"});
    try std.testing.expectEqual(@Tuple(&.{mem.Bufferized(Model)}), Runner.BakedInput);
    try std.testing.expectEqual(@Tuple(&.{ Buffer, mem.Bufferized(Cache) }), Runner.NonBakedInput);
    try std.testing.expectEqual(@Tuple(&.{ *Buffer, *mem.Bufferized(Cache) }), Forward.Output);

    const Single = Exe(Functions.single);
    try std.testing.expectEqual(@Tuple(&.{Buffer}), Single.Input);
    try std.testing.expectEqual(*Buffer, Single.Output);
    try std.testing.expectEqual(Buffer, Single.Result);

    try std.testing.expectEqual(@Tuple(&.{}), Exe(Functions.noInput).Input);
}

test "RawExe calls leave donated input handle cleanup to the caller" {
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
    try std.testing.expectEqualSlices(u32, &.{0}, exe.raw.donated_input_indices);

    var weight_buffer = try Buffer.fromBytes(io, platform, weight.shape(), std.mem.asBytes(&[2]i32{ 10, 20 }));
    defer weight_buffer.deinit();
    var args = try exe.raw.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.raw.results(allocator);
    defer results.deinit(allocator);

    for ([_]bool{ false, true }) |wait| {
        var value_buffer = try Buffer.fromBytes(io, platform, value.shape(), std.mem.asBytes(&[2]i32{ 1, 2 }));
        defer value_buffer.deinit();
        args.set(.{ value_buffer, weight_buffer });
        exe.raw.call(io, args, &results, .{ .wait = wait });
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
        fn forward(inputs: struct { weight: Tensor, value: Tensor }) struct { value: Tensor } {
            return .{ .value = inputs.value.add(inputs.weight).reuseBuffer(inputs.value) };
        }
    };
    const Model = Exe(Functions.forward);
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

    runner.run(io, .{ .value = value_buffer }, .{ .value = &value_buffer }, .{});
    try std.testing.expectEqual([2]i32{ 11, 22 }, try value_buffer.getValue([2]i32, io));

    runner.run(io, .{ .value = value_buffer }, .{ .value = &value_buffer }, .{ .wait = true });
    try std.testing.expectEqual([2]i32{ 21, 42 }, try value_buffer.getValue([2]i32, io));
}

test "Exe runner bakes positional arguments" {
    const platform = @import("testing.zig").env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Model = struct {
        weight: Tensor,

        pub fn forward(self: @This(), scale: i32, value: Tensor) struct { Tensor, Tensor } {
            const sum = value.add(self.weight);
            return .{ sum.reuseBuffer(value), sum.scale(scale) };
        }
    };
    const model: Model = .{ .weight = .init(.{2}, .i32) };
    const value: Tensor = .init(.{2}, .i32);
    const exe = try platform.compile(allocator, io, model, .forward, .{ 3, value }, .{});
    defer exe.deinit();

    var weight_buffer = try Buffer.fromBytes(io, platform, model.weight.shape(), std.mem.asBytes(&[2]i32{ 10, 20 }));
    defer weight_buffer.deinit();
    var value_buffer = try Buffer.fromBytes(io, platform, value.shape(), std.mem.asBytes(&[2]i32{ 1, 2 }));
    defer value_buffer.deinit();
    var scaled: Buffer = undefined;

    var runner = try @TypeOf(exe).Runner(.{.@"0"}).init(&exe, allocator, .{.{ .weight = weight_buffer }});
    defer runner.deinit(allocator);
    runner.run(io, .{value_buffer}, .{ &value_buffer, &scaled }, .{ .wait = true });
    defer scaled.deinit();
    try std.testing.expectEqual([2]i32{ 11, 22 }, try value_buffer.getValue([2]i32, io));
    try std.testing.expectEqual([2]i32{ 33, 66 }, try scaled.getValue([2]i32, io));

    var result = try exe.eval(allocator, io, .{ .{ .weight = weight_buffer }, value_buffer });
    defer result[1].deinit();
    value_buffer = result[0];
    try std.testing.expectEqual([2]i32{ 21, 42 }, try value_buffer.getValue([2]i32, io));
    try std.testing.expectEqual([2]i32{ 63, 126 }, try result[1].getValue([2]i32, io));
}

test "Exe runner replaces unchanged donated buffers" {
    const platform = @import("testing.zig").env();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Functions = struct {
        fn forward(inputs: struct { value: Tensor }) struct { value: Tensor } {
            return .{ .value = inputs.value.reuseBuffer(inputs.value) };
        }
    };
    const Model = Exe(Functions.forward);
    const value: Tensor = .init(.{2}, .i32);
    const exe = try Model.compile(allocator, io, platform, .{}, .{.{ .value = value }});
    defer exe.deinit();
    try std.testing.expectEqualSlices(u32, &.{0}, exe.raw.donated_input_indices);

    var value_buffer = try Buffer.fromBytes(io, platform, value.shape(), std.mem.asBytes(&[2]i32{ 1, 2 }));
    defer value_buffer.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    for ([_]bool{ false, true }) |wait| {
        runner.run(io, .{ .value = value_buffer }, .{ .value = &value_buffer }, .{ .wait = wait });
        try std.testing.expectEqual([2]i32{ 1, 2 }, try value_buffer.getValue([2]i32, io));
    }
}
