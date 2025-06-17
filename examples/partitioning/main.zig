const std = @import("std");
const zml = @import("zml");
const stdx = @import("stdx");
const asynk = @import("async");
const flags = stdx.flags;

// set log level to debug to print the generated IR
pub const std_options: std.Options = .{
    .log_level = .debug,
    .logFn = asynk.logFn(std.log.defaultLog),
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .@"zml/async", .level = .info },
    },
};

const log = std.log.scoped(.@"examples/partitioning");

const Model = struct {
    pub fn forward(self: Model, a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        _ = self; // autofix
        // return a.add(b);
        return a.add(b); // .withSharding(mesh, .{ .y = .m });
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const CliArgs = struct {
        pub const help =
            \\ partitioning --size=64 --dtype=i32
        ;
        size: usize = 8,
        dtype: zml.DataType = .i32,
    };

    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);

    const mesh: zml.Mesh = .init(.{ .x = 2, .y = 2 });

    const platform = context.autoPlatform(.{ .cpu = .{ .cpu_device_count = mesh.numRequiredDevices() } }).withCompilationOptions(.{
        .sharding_enabled = true,
    });
    context.printAvailablePlatforms(platform);

    const shape: zml.Shape = .init(.{ cli_args.size, cli_args.size }, cli_args.dtype);

    const a_shape = shape.withTags(.{ .m, .k }).withPartitionning(.{ .x = .m, .y = .k });
    const b_shape = shape.withTags(.{ .k, .n }).withPartitionning(.{ .x = .k, .y = .n });

    const buffers: zml.aio.BufferStore.Buffers = .{};

    const buffer_store: zml.aio.BufferStore = .{
        .arena = arena_state,
        .buffers = buffers,
    };

    const model_shapes = try zml.aio.populateModel(Model, allocator, buffer_store);

    var compilation = try asynk.asyncc(zml.compileModel, .{ allocator, Model.forward, model_shapes, .{ a_shape, b_shape }, mesh, platform });

    var model_weights = try zml.aio.loadModelBuffers(Model, model_shapes, buffer_store, arena, platform);
    defer zml.aio.unloadBuffers(&model_weights);
    log.info("✅ Loaded weights", .{});

    const compiled = try compilation.awaitt();
    var executable = compiled.prepare(model_weights);
    defer executable.deinit();
    log.info("✅ Compiled model", .{});

    const sharding: zml.Sharding = .init(mesh, a_shape);
    log.info("Sharding: {}", .{sharding});

    const a_buffer_slice = try zml.slice.arange(allocator, a_shape, .{});
    defer allocator.free(a_buffer_slice);
    log.debug("Created arange slice of size {d} ptr {*} bytes for shape: {s}: {}", .{
        a_buffer_slice.len,
        a_buffer_slice.ptr,
        a_shape,
        zml.HostBuffer.fromBytes(a_shape, a_buffer_slice).pretty(),
    });

    const a_buffer = try zml.Buffer.from(platform, sharding, a_buffer_slice, .{});
    defer a_buffer.deinit();

    const b_buffer_slice = try allocateArangeSlice(allocator, a_shape);
    defer allocator.free(b_buffer_slice);
    log.debug("Created arange slice of size {d} ptr {*} bytes for shape: {s}: {}", .{
        b_buffer_slice.len,
        b_buffer_slice.ptr,
        a_shape,
        zml.HostBuffer.fromBytes(a_shape, b_buffer_slice).pretty(),
    });

    const b_buffer = try zml.Buffer.from(platform, sharding, b_buffer_slice, .{});
    defer b_buffer.deinit();

    log.info("✅ Running model....", .{});
    var result: zml.Buffer = executable.call(.{ a_buffer, b_buffer });
    defer result.deinit();

    // --- 2. Call the high-level reassembly function ---
    const reassembled_buffer = try allocator.alloc(u8, sharding.shape.byteSize());
    defer allocator.free(reassembled_buffer);

    try sharding.reassembleFromPjrtBuffers(
        platform,
        result._shards.constSlice(),
        reassembled_buffer,
        allocator,
    );

    log.warn("Reassembled buffer: {any}", .{
        // items(reassembled_buffer, sharding.shape, i32),
        zml.HostBuffer.fromBytes(sharding.shape, reassembled_buffer).pretty(),
    });

    // const cpu_result = try result.toHostAlloc(allocator);
    // defer cpu_result.deinit(allocator);

    // log.info("Result: {any}", .{cpu_result.pretty()});
}

pub fn allocateArangeSlice(allocator: std.mem.Allocator, shape: zml.Shape) ![]u8 {
    const start: i64 = 0;
    const step: i64 = 1;
    const slice = try allocator.alloc(u8, shape.byteSize());
    errdefer allocator.free(slice);

    switch (shape.dtype()) {
        inline else => |d| if (comptime d.class() != .integer) {
            stdx.debug.assert(shape.dtype().class() == .integer, "arange expects type to be integer, got {} instead.", .{shape.dtype()});
        } else {
            const Zt = d.toZigType();
            var j: i64 = start;
            for (zml.Shaped(Zt, shape, slice)) |*val| {
                val.* = @intCast(j);
                j += step;
            }
        },
    }

    return slice;
}
