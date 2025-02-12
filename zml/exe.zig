const std = @import("std");
const stdx = @import("stdx");

const aio = @import("aio.zig");
const meta = @import("meta.zig");
const pjrt = @import("pjrtx.zig");

const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("tensor.zig").Bufferized;
const CompilationContext = @import("module.zig").CompilationContext;
const Platform = @import("platform.zig").Platform;
const Shape = @import("shape.zig").Shape;
const ShapeOf = @import("tensor.zig").ShapeOf;

const log = std.log.scoped(.zml);

test {
    std.testing.refAllDecls(@This());
}

/// Compiles a Model struct with the given configuration and shapes, for the given platform.
/// The steps are:
/// * lookup at tensors available in the store and create a `model: Model` struct with them
/// * call `model.init(init_args)` to fields of the model that aren't Tensor, ie hyperparemeters/config
/// * generate MLIR by calling `model.forward` with tensor of the given shapes and other arguments
pub fn compile(
    allocator: std.mem.Allocator,
    comptime func: anytype,
    init_args: anytype,
    args_shapes: ShapeOf(ModuleSignature(func).ArgsT),
    buffer_store: aio.BufferStore,
    platform: Platform,
) !FnExe(func) {
    const ModelT = ModuleSignature(func).ModelT;

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var model = try aio.populateModel(ModelT, arena, buffer_store);

    // If the Model has a "init" function, call it with the given parameters.
    if (@hasDecl(ModelT, "init")) {
        // TODO(Corentin,@Improvement): Add a warning/error if there is no init function but init_args is non-void.
        @call(.auto, ModelT.init, .{@as(*ModelT, &model)} ++ init_args);
    }

    return compileModel(allocator, func, model, args_shapes, platform);
}

/// Compiles a Model struct with the given configuration and shapes, for the given platform.
/// Generate MLIR by calling `model.forward` with tensor of the given shapes and other arguments
pub fn compileModel(
    allocator: std.mem.Allocator,
    comptime func: anytype,
    model: ModuleSignature(func).ModelT,
    args_shapes: ShapeOf(ModuleSignature(func).ArgsT),
    platform: Platform,
) !FnExe(func) {
    const ModelT = ModuleSignature(func).ModelT;
    const name = @typeName(ModelT) ++ ".forward";
    log.info("Compiling {s} with {}", .{ name, args_shapes });

    var context = try CompilationContext.init(allocator, name, platform);
    defer context.deinit();

    return .{ .inner = try context.compileInternal(allocator, func, .{model} ++ args_shapes) };
}

/// Compiles a function with the given configuration and shapes, for the given platform.
/// Generate MLIR by calling the given function with tensor of the given shapes.
pub fn compileFn(
    allocator: std.mem.Allocator,
    comptime func: anytype,
    args: ShapeOf(stdx.meta.FnArgs(func)),
    platform: Platform,
) !FnExe(func) {
    var pretty_name = try prettyFnName(func, allocator);
    defer pretty_name.deinit(allocator);
    var context = try CompilationContext.init(allocator, pretty_name.items, platform);
    defer context.deinit();

    return .{ .inner = try context.compileInternal(allocator, func, args) };
}

pub fn FnExe(comptime func: anytype) type {
    return Exe(stdx.meta.FnArgs(func), stdx.meta.FnResult(func));
}

/// Represents a ZML model, compiled into a PJRT executable, and ready to call.
/// The buffers for the model weights are saved inside the struct and will be used in `call`.
/// You only need to pass the remaining arguments.
/// Creating a `ModuleExe` is a two steps proccess:
///
/// ```
/// const exe: zml.FnExe(MyModel.forward) = try zml.compile(allocator, MyModel.forward, init_args, model_shapes, buffer_store, platform);`
/// const module: zml.ModuleExe(MyModel.forward) = exe.prepare(model_buffers);
/// ```
pub fn ModuleExe(comptime func: anytype) type {
    const AllArgs = stdx.meta.FnArgs(func);
    const len = @typeInfo(AllArgs).Struct.fields.len;
    stdx.debug.assertComptime(len > 0, "ModuleExe expects a function with at least one argument where the first one is treated as the module, got {}", .{func});
    return Exe(stdx.meta.Tail(AllArgs), stdx.meta.FnResult(func));
}

// making this a struct force all fields to be evaluted on creation,
// which gives a better error stacktrace
// than delaying the error to when the object fields are read.
const Sign = struct {
    ModelT: type,
    ArgsT: type,
    ReturnT: type,
};

pub fn ModuleSignature(comptime func: anytype) Sign {
    const AllArgsT = stdx.meta.FnArgs(func);
    const len = @typeInfo(AllArgsT).@"struct".fields.len;
    stdx.debug.assertComptime(len > 0, "ModuleExe expects a function with at least one argument where the first one is treated as the module, got {}", .{func});

    return .{
        .ModelT = stdx.meta.Head(AllArgsT),
        .ArgsT = stdx.meta.Tail(AllArgsT),
        .ReturnT = stdx.meta.FnResult(func),
    };
}

/// Represents an MLIR module compiled into a PJRT executable.
/// The BaseExe is a plain old struct and doesn't have information about Zig types.
///
/// It also contains pre-allocated buffers so that we can pass them to PJRT_LoadedExecutable_Execute
/// without allocations.
pub const BaseExe = struct {
    /// The platform for which this module was compiled.
    platform: Platform,

    /// The PJRT executable representing the compiled module.
    exe: *pjrt.LoadedExecutable,

    /// Pre-allocated slice of buffers to use as inputs when the module is called.
    input_per_device: []const [*]*pjrt.Buffer,

    /// Pre-allocated slice of buffers to use as outputs when the module is called.
    output_per_device: []const [*]*pjrt.Buffer,

    /// Number of buffers already fed to the executable.
    ready_buffer_count: u32,

    /// Total number of buffers needed by this executable.
    input_buffer_count: u32,

    result_shapes: []Shape,

    /// Num devices used (>1 for sharded executable)
    num_devices: u8,

    /// Allocator backing memory
    _arena: std.heap.ArenaAllocator,

    pub fn init(parent_allocator: std.mem.Allocator, platform: Platform, exe: *pjrt.LoadedExecutable, args: struct { n_in: u32, result_shapes: []const Shape, n_devices: u8 }) !BaseExe {
        var arena = std.heap.ArenaAllocator.init(parent_allocator);
        errdefer arena.deinit();
        const allocator = arena.allocator();
        const n_out = args.result_shapes.len;
        const n_devices = args.n_devices;
        // Allocate once for all the *pjrt.Buffer we need to store ...
        const all_buffers = try allocator.alloc(*pjrt.Buffer, (args.n_in + n_out) * n_devices);
        const all_input_buffers, const all_output_buffers = splitBuffer(*pjrt.Buffer, all_buffers, .{ args.n_in * n_devices, n_out * n_devices });

        // ... and once for all the [*]*pjrt.Buffer.
        const all_per_device = try allocator.alloc([*]*pjrt.Buffer, 2 * n_devices);
        const input_per_device, const output_per_device = splitBuffer([*]*pjrt.Buffer, all_per_device, .{ n_devices, n_devices });

        for (0..n_devices) |i| {
            input_per_device[i] = all_input_buffers[i * args.n_in ..].ptr;
            output_per_device[i] = all_output_buffers[i * n_out ..].ptr;
        }

        return .{
            .platform = platform,
            .exe = exe,
            .ready_buffer_count = 0,
            .input_buffer_count = args.n_in,
            .num_devices = args.n_devices,
            .input_per_device = input_per_device,
            .output_per_device = output_per_device,
            .result_shapes = try allocator.dupe(Shape, args.result_shapes),
            ._arena = arena,
        };
    }

    pub fn deinit(self: BaseExe) void {
        self._arena.deinit();
    }

    pub fn call(self: BaseExe) void {
        stdx.debug.assert(self.input_buffer_count == self.ready_buffer_count, "BaseExe isn't ready to be called, expected {} buffer inputs got {}", .{ self.input_buffer_count, self.ready_buffer_count });
        return self._unsafeCall();
    }

    pub fn _unsafeCall(self: BaseExe) void {
        var events = [_]?*pjrt.Event{null} ** Platform.MAX_NUM_DEVICES;
        const sharding = self.platform.sharding();

        self.exe.execute(self.platform.pjrt_api, .{
            .arguments = self.input_per_device,
            .num_args = self.input_buffer_count,
            .results = self.output_per_device,
            .events = events[0..sharding.num_partitions],
            // this allows to tell a specific buffer shouldn't be donated,
            // even if it has been marked as "can be donated" during compilation.
            // TODO: expose it ?
            .non_donatable_input_indices = &.{},
        }) catch unreachable;

        for (events[0..sharding.num_partitions]) |e| {
            if (e) |ev| {
                ev.await_(self.platform.pjrt_api) catch unreachable;
            }
        }
    }

    pub fn serialize(self: BaseExe, writer: anytype) !void {
        var executable = try self.exe.getExecutable(self.platform.pjrt_api);
        var serialize_result = try executable.serialize(self.platform.pjrt_api);
        defer serialize_result.deinit();
        try writer.writeAll(serialize_result.bytes);
    }

    // pub fn deserialize(allocator: std.mem.Allocator, platform: Platform, reader: anytype) !Self {
    //     const bytes = try reader.readToEndAlloc(allocator, max_pjrt_executable_size);
    //     defer allocator.free(bytes);
    //     return platform.pjrt_client.deserializeAndLoad(platform.pjrt_api, bytes);
    // }

    pub fn prepare(self: *BaseExe, x: anytype) void {
        const n = fillBuffers(&x, self.input_per_device, self.ready_buffer_count);
        self.ready_buffer_count += n;
    }

    pub fn getOutputBuffer(self: BaseExe, i: usize) Buffer {
        var shards: Buffer.Shards = .{};
        for (self.output_per_device) |dev_out| {
            shards.appendAssumeCapacity(dev_out[i]);
        }

        return Buffer.fromPjrtBuffers(self.platform, self.result_shapes[i], shards.constSlice());
    }
};

/// Represents a ZML function, compiled into a PJRT executable.
/// The signature of the Exe reflects the arguments that are needed for `call`.
pub fn Exe(ArgsT: type, ReturnT: type) type {
    return struct {
        const Self = @This();

        /// The raw untyped compiled module.
        inner: BaseExe,

        pub fn deinit(self: Self) void {
            self.inner.deinit();
        }

        /// Hardcode the first argument of the function to the given buffers.
        /// Returns an Exe with one less argument in `call`.
        /// In functional languages this is known as partial application.
        ///
        /// **Warning:** the new Exe reuses the underlying memory of the previous one.
        /// The caller is responsible to come up with a strategy to call `deinit` exactly once.
        pub fn prepare(self: Self, first_arg: Bufferized(stdx.meta.Head(ArgsT))) Exe(stdx.meta.Tail(ArgsT), ReturnT) {
            var new: Exe(stdx.meta.Tail(ArgsT), ReturnT) = .{ .inner = self.inner };
            new.inner.prepare(first_arg);
            return new;
        }

        pub fn serialize(self: Self, writer: anytype) !void {
            return try self.inner.serialize(writer);
        }

        pub fn platform(self: Self) Platform {
            return self.inner.platform;
        }

        pub fn call(self: Self, args: Bufferized(ArgsT)) Bufferized(ReturnT) {
            const total_ready = fillBuffers(&args, self.inner.input_per_device, self.inner.ready_buffer_count);
            std.debug.assert(total_ready == self.inner.input_buffer_count);
            self.inner._unsafeCall();
            var result: Bufferized(ReturnT) = undefined;
            assignRawBuffers(&result, self.inner.platform, self.inner.output_per_device, self.inner.result_shapes);
            return result;
        }
    };
}

fn splitBuffer(T: type, buffer: []T, lengths: anytype) [lengths.len][]T {
    var res: [lengths.len][]T = undefined;
    var i: usize = 0;
    inline for (&res, lengths) |*r, len| {
        r.* = buffer[i .. i + len];
        i += len;
    }
    std.debug.assert(i == buffer.len);
    return res;
}

/// Visit the given struct and fill the `buffers` slice with the buffer associated with encountered Tensor.
fn fillBuffers(v: anytype, buffers: []const [*]*pjrt.Buffer, start: u32) u32 {
    const LocalContext = struct {
        index: u32,
        buffers: []const [*]*pjrt.Buffer,
    };
    var context: LocalContext = .{
        .index = start,
        .buffers = buffers,
    };
    meta.visit((struct {
        fn cb(ctx: *LocalContext, buffer: *const Buffer) void {
            // stdx.debug.assert(!buffer._data.isDeleted(), "Can't use {} (argument buffer {}) because its pjrt buffer has been donated", .{ buffer, ctx.index });
            const model_sharding = ctx.buffers.len;
            stdx.debug.assert(buffer._shards.len == model_sharding, "Can't feed a {}-sharded tensor into a {}-sharded model", .{ buffer._shards.len, ctx.buffers.len });
            for (buffer._shards.constSlice(), 0..) |shard, d| {
                ctx.buffers[d][ctx.index] = shard;
            }
            ctx.index += 1;
        }
    }).cb, &context, v);
    return context.index;
}

/// Visit the given struct and override tensors by creating a new one using the provided PJRT buffers.
fn assignRawBuffers(v: anytype, platform: Platform, buffers: []const [*]*pjrt.Buffer, buffer_shapes: []Shape) void {
    const LocalContext = struct {
        index: u32,
        platform: Platform,
        buffers: []const [*]*pjrt.Buffer,
        buffer_shapes: []Shape,
    };
    var local_ctx: LocalContext = .{
        .index = 0,
        .platform = platform,
        .buffers = buffers,
        .buffer_shapes = buffer_shapes,
    };
    meta.visit((struct {
        fn cb(ctx: *LocalContext, buffer: *Buffer) void {
            const i = ctx.index;
            ctx.index += 1;
            if (i >= ctx.buffer_shapes.len) return;

            var shards: Buffer.Shards = .{};
            for (ctx.buffers) |buff| {
                shards.appendAssumeCapacity(buff[i]);
            }
            buffer.* = Buffer.fromPjrtBuffers(ctx.platform, ctx.buffer_shapes[i], shards.constSlice());
        }
    }).cb, &local_ctx, v);
    stdx.debug.internalAssert(local_ctx.index == buffer_shapes.len, "Pjrt call returned {} tensors, but the return type {s}, contains {} Buffers. Note that modules need to have a comptime know number of returned tensors.", .{ buffers.len, @typeName(@TypeOf(v)), local_ctx.index });
}

fn prettyFnName(
    comptime func: anytype,
    allocator: std.mem.Allocator,
) !std.ArrayListUnmanaged(u8) {
    const full_noisy_name = @typeName(@TypeOf(func));
    const og_len = full_noisy_name.len;
    const buffer = try allocator.alloc(u8, og_len);
    errdefer comptime unreachable; // No errors below this point.
    var out: []u8 = buffer;

    {
        const verbose = "tensor.Tensor";
        const compact = "Tensor";
        const num_replacements = std.mem.replace(u8, full_noisy_name, verbose, compact, buffer);
        out.len = out.len + num_replacements * compact.len - num_replacements * verbose.len;
    }

    {
        const verbose = "tensor.Tensor.";
        const compact = "";
        const num_replacements = std.mem.replace(u8, out, verbose, compact, buffer);
        out.len = out.len + num_replacements * compact.len - num_replacements * verbose.len;
    }

    {
        const verbose = "shape.Shape";
        const compact = "Shape";
        const num_replacements = std.mem.replace(u8, out, verbose, compact, buffer);
        out.len = out.len + num_replacements * compact.len - num_replacements * verbose.len;
    }

    return .{ .items = out, .capacity = og_len };
}
