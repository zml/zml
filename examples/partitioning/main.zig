const std = @import("std");
const zml = @import("zml");
const stdx = @import("stdx");
const asynk = @import("async");

// set log level to debug to print the generated IR
pub const std_options: std.Options = .{
    .log_level = .debug,
    .logFn = asynk.logFn(std.log.defaultLog),
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .@"zml/async", .level = .info },
    },
};

const log = std.log.scoped(.@"examples/partitioning");

const ThirdPartyModule = struct {
    weight: zml.Tensor,

    pub fn init(self: *ThirdPartyModule, mesh: zml.Mesh) void {
        self.weight = self.weight.withTags(.{ .rows, .cols }).withMesh(mesh).withSharding(.{ .rows = .x });
    }

    pub fn forward(self: ThirdPartyModule, a: zml.Tensor) zml.Tensor {
        log.debug("Forwarding a: {} and weight: {}", .{ a, self.weight });

        zml.pushMesh(a.mesh().flatten(.x));
        const x = a.withSharding(.{ .m = .x }).add(self.weight.withSharding(.{ .rows = .x }));
        zml.popMesh();

        return x.withSharding(.{ .m = .x, .n = .y });
    }
};

const Model = struct {
    module_with_weights: ThirdPartyModule,
    mesh: zml.Mesh = undefined,

    pub fn init(self: *Model, main_mesh: zml.Mesh, third_party_mesh: zml.Mesh) void {
        self.mesh = main_mesh;
        self.module_with_weights.init(third_party_mesh);
    }

    pub fn forward(self: Model, a: zml.Tensor, b: zml.Tensor) zml.Tensor {
        const a_add_b = a.add(b);
        const weighted_a: zml.Tensor = zml.call(self.module_with_weights, .forward, .{a});
        return a_add_b.add(weighted_a);
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var context = try zml.Context.init();
    defer context.deinit();

    const platform = context.autoPlatform(.{ .cpu = .{ .cpu_device_count = 12 } }).withCompilationOptions(.{
        .sharding_enabled = true,
        .xla_dump_to = "/tmp/zml/partitioning",
    });
    context.printAvailablePlatforms(platform);

    // const mesh: zml.Mesh = .init(.{ .x = 4, .y = 3 });
    const mesh: zml.Mesh = .auto(platform);
    // const mesh: zml.Mesh = .single();
    log.info("{}", .{mesh});

    const shape = zml.Shape.init(.{ .m = 12, .n = 6 }, .i32);

    const a_shape = shape.withTags(.{ .m, .n }).withPartitioning(.{ .m = .x, .n = .y });
    const b_shape = shape.withTags(.{ .n, .m }).withPartitioning(.{ .n = .x, .m = .y });

    var buffers: zml.aio.BufferStore.Buffers = .{};
    const weight = try zml.slice.arange(allocator, shape, .{});
    defer allocator.free(weight);
    try buffers.put(arena, "module_with_weights.weight", zml.HostBuffer.fromBytes(shape, weight));

    const buffer_store: zml.aio.BufferStore = .{
        .arena = arena_state,
        .buffers = buffers,
    };

    var model_shapes = try zml.aio.populateModel(Model, allocator, buffer_store);
    model_shapes.init(mesh, mesh.flatten(.x));

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
    log.debug("a_buffer_slice: {}", .{zml.slice.pretty(a_shape, a_buffer_slice)});

    const a_buffer = try zml.Buffer.from(platform, a_sharding, a_buffer_slice, .{});
    defer a_buffer.deinit();

    const b_sharding: zml.Sharding = .init(mesh, a_shape);
    log.info("b_sharding: {}", .{b_sharding});

    const b_buffer_slice = try zml.slice.arange(allocator, b_shape, .{});
    defer allocator.free(b_buffer_slice);
    log.debug("b_buffer_slice: {}", .{zml.slice.pretty(b_shape, b_buffer_slice)});

    const b_buffer = try zml.Buffer.from(platform, b_sharding, b_buffer_slice, .{});
    defer b_buffer.deinit();

    log.info("✅ Running model....", .{});
    var result_buffer: zml.Buffer = executable.call(.{ a_buffer, b_buffer });
    defer result_buffer.deinit();

    const result = try result_buffer.toHost(allocator);
    defer allocator.free(result);

    log.info("Result buffer: {}", .{zml.slice.pretty(result_buffer.shape(), result)});
}
