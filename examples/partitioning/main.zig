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
        return a.dot(b, .{.k}); // .withSharding(mesh, .{ .y = .m });
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const CliArgs = struct {
        pub const help =
            \\ partitioning --size=4096 --dtype=f16
        ;
        size: usize = 4096,
        dtype: zml.DataType = .f16,
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

    const mesh: zml.Mesh = .init(.{ .y = 4 });

    const platform = context.autoPlatform(.{ .cpu = .{ .cpu_device_count = mesh.numRequiredDevices() } }).withCompilationOptions(.{
        .sharding_enabled = true,
    });
    context.printAvailablePlatforms(platform);

    const shape: zml.Shape = .init(.{ cli_args.size, cli_args.size }, cli_args.dtype);

    const a_shape = shape.withTags(.{ .m, .k }).withPartitionning(.{ .y = .k });
    const b_shape = shape.withTags(.{ .k, .n }).withPartitionning(.{ .y = .k });

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

    const a_buffer_alloc = try allocator.alloc(u8, a_shape.byteSize());
    defer allocator.free(a_buffer_alloc);
    const a_slice = std.mem.bytesAsSlice(f16, a_buffer_alloc);
    @memset(a_slice, 1.0);
    log.debug("Created buffer of size {d} ptr {*} bytes for shape: {s}", .{ a_buffer_alloc.len, a_buffer_alloc.ptr, a_shape });

    const a_buffer = try zml.Buffer.from(platform, sharding, a_buffer_alloc, .{});
    defer a_buffer.deinit();

    const b_buffer_alloc = try allocator.alloc(u8, b_shape.byteSize());
    defer allocator.free(b_buffer_alloc);
    const b_slice = std.mem.bytesAsSlice(f16, b_buffer_alloc);
    @memset(b_slice, 1.0);
    log.debug("Created buffer of size {d} ptr {*} bytes for shape: {s}", .{ b_buffer_alloc.len, b_buffer_alloc.ptr, b_shape });

    const b_buffer = try zml.Buffer.from(platform, sharding, b_buffer_alloc, .{});
    defer b_buffer.deinit();

    log.info("✅ Running model....", .{});
    var result: zml.Buffer = executable.call(.{ a_buffer, b_buffer });
    defer result.deinit();

    const cpu_result = try result.toHostAlloc(allocator);
    defer cpu_result.deinit(allocator);

    log.info("Result: {any}", .{cpu_result.pretty()});
}
