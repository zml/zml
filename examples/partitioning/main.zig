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

// const ModuleMesh = zml.Mesh.init(.{ .a = 1, .b = 2 });
// return a.add(b);
// return a.add(b).dot(b, .{.k}); // .withSharding(mesh, .{ .y = .m });

const ModuleWithWeight = struct {
    weight: zml.Tensor,

    pub fn init(self: *ModuleWithWeight) void {
        self.weight = self.weight.withTags(.{ .rows, .cols });
    }

    pub fn forward(self: ModuleWithWeight, a: zml.Tensor) zml.Tensor {
        return a.add(self.weight.withPartitionning(.{ .grid_x = .rows, .grid_y = .cols }));
    }
};

const Model = struct {
    module_with_weights: ModuleWithWeight,
    mesh: ?zml.Mesh,

    pub fn init(self: *Model, mesh: zml.Mesh) void {
        self.mesh = mesh;
        self.module_with_weights.init();
    }

    pub fn forward(self: Model, a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        const a_add_b = a.add(b);
        const weighted_a: zml.Tensor = zml.call(self.module_with_weights, .forward, .{a.replicated()});
        zml.pushMesh(self.mesh.?);
        const weighted_b: zml.Tensor = zml.call(self.module_with_weights, .forward, .{weighted_a.withPartitionning(.{ .grid_x = .m, .grid_y = .k })});
        zml.popMesh();
        zml.pushMesh(self.mesh.?);
        const weighted_c: zml.Tensor = zml.call(self.module_with_weights, .forward, .{weighted_b.withPartitionning(.{ .grid_x = .m, .grid_y = .k })});
        zml.popMesh();
        return a_add_b.add(weighted_a).add(weighted_b).add(weighted_c).withPartitionning(.{ .grid_x = .m, .grid_y = .k });
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

    const mesh: zml.Mesh = .init(.{ .grid_x = 2, .grid_y = 2 });

    const platform = context.autoPlatform(.{ .cpu = .{ .cpu_device_count = mesh.numRequiredDevices() } }).withCompilationOptions(.{
        .sharding_enabled = true,
        .xla_dump_to = "/tmp/zml/partitioning",
    });
    context.printAvailablePlatforms(platform);

    const shape: zml.Shape = .init(.{ cli_args.size, cli_args.size }, cli_args.dtype);

    const a_shape = shape.withTags(.{ .m, .k }).withPartitionning(.{ .grid_x = .m, .grid_y = .k });
    const b_shape = shape.withTags(.{ .k, .n }).withPartitionning(.{ .grid_x = .k, .grid_y = .n });

    var buffers: zml.aio.BufferStore.Buffers = .{};
    const weight = try zml.slice.arange(allocator, shape, .{});
    defer allocator.free(weight);
    try buffers.put(arena, "module_with_weights.weight", zml.HostBuffer.fromBytes(shape, weight));

    const buffer_store: zml.aio.BufferStore = .{
        .arena = arena_state,
        .buffers = buffers,
    };

    var model_shapes = try zml.aio.populateModel(Model, allocator, buffer_store);
    model_shapes.init(mesh);

    var compilation = try asynk.asyncc(zml.compileModel, .{ allocator, Model.forward, model_shapes, .{ a_shape, b_shape }, mesh, platform });

    var model_weights = try zml.aio.loadModelBuffers(Model, model_shapes, buffer_store, arena, platform, mesh);
    defer zml.aio.unloadBuffers(&model_weights);
    log.info("✅ Loaded weights", .{});

    const compiled = try compilation.awaitt();
    var executable = compiled.prepare(model_weights);
    defer executable.deinit();
    log.info("✅ Compiled model", .{});

    const a_sharding: zml.Sharding = .init(mesh, a_shape);
    log.info("a_sharding: {}", .{a_sharding});

    const a_buffer_slice = try zml.slice.arange(allocator, a_shape, .{});
    defer allocator.free(a_buffer_slice);
    log.debug("a_buffer_slice: {}", .{zml.HostBuffer.fromBytes(a_shape, a_buffer_slice).pretty()});

    const a_buffer = try zml.Buffer.from(platform, a_sharding, a_buffer_slice, .{});
    defer a_buffer.deinit();

    const b_sharding: zml.Sharding = .init(mesh, a_shape);
    log.info("b_sharding: {}", .{b_sharding});

    const b_buffer_slice = try zml.slice.arange(allocator, b_shape, .{});
    defer allocator.free(b_buffer_slice);
    log.debug("b_buffer_slice: {}", .{zml.HostBuffer.fromBytes(b_shape, b_buffer_slice).pretty()});

    const b_buffer = try zml.Buffer.from(platform, b_sharding, b_buffer_slice, .{});
    defer b_buffer.deinit();

    log.info("✅ Running model....", .{});
    var result: zml.Buffer = executable.call(.{ a_buffer, b_buffer });
    defer result.deinit();

    // --- 2. Call the high-level reassembly function ---
    const reassembled_buffer = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(reassembled_buffer);

    log.warn("output shape: {any}", .{result.shape()});

    const replicated_output_shape = result.shape().withTags(.{ .m, .n });
    const output_sharding = zml.Sharding.init(mesh, replicated_output_shape);

    try output_sharding.reassembleFromPjrtBuffers(
        platform,
        result._shards.constSlice(),
        reassembled_buffer,
        allocator,
    );

    log.warn("Reassembled buffer: {any}", .{
        zml.HostBuffer.fromBytes(b_sharding.global_shape, reassembled_buffer).pretty(),
    });
}
