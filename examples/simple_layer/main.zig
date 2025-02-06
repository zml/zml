const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");

const Tensor = zml.Tensor;

pub fn conditionalForward(cache: Tensor, indices: Tensor, input: Tensor) Tensor {
    const pages = cache.gatherValues(.{.page}, indices.appendAxes(.{.coord}), .{});
    var result = pages.dot(input, .{.hd});
    const zeros = zml.Tensor.constant(result.shape(), result.dtype().zero());
    const cutoff = zml.Tensor.constant(indices.shape(), indices.dtype().constant(cache.dim(.page)));
    const mask = zml.Tensor.cmp(indices, .LT, cutoff).broad(result.shape());
    result = mask.select(result, zeros);
    return result;
}

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Arena allocator for BufferStore etc.
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    _ = arena; // autofix

    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/simple_layer",
    };

    const platform = context.autoPlatform(.{}).withCompilationOptions(compilation_options);
    context.printAvailablePlatforms(platform);

    var profiler = platform.getProfiler(null);
    defer profiler.deinit();

    profiler.start();
    defer {
        profiler.stop();
        profiler.dumpAsJsonTo(allocator, std.fs.cwd(), "trace.json") catch unreachable;
    }

    const dtype = .bf16;
    const batch_size = 1;
    const page_count = 8192;
    const max_page_count = 2048;
    const chunk_size = 16;
    const num_heads = 8;
    const head_dim = 8192;

    const cache_shape = zml.Shape.init(.{ .page = page_count, .k_chunk = chunk_size, .h = num_heads, .hd = head_dim }, dtype);
    const indices_shape = zml.Shape.init(.{ .b = batch_size, .max_page = max_page_count }, .u32);
    const input_shape = zml.Shape.init(.{ .b = batch_size, .q = 1, .h = num_heads, .hd = head_dim }, dtype);
    const mod = try zml.compileFn(allocator, conditionalForward, .{ cache_shape, indices_shape, input_shape }, platform);
    defer mod.deinit();

    var generator = std.Random.DefaultPrng.init(0);
    const random = generator.random();

    var cache = try createRandomBuffer(allocator, platform, cache_shape, random);
    defer cache.deinit();
    var input = try createRandomBuffer(allocator, platform, input_shape, random);
    defer input.deinit();

    const indices_data = try allocator.alloc(u32, max_page_count);
    defer allocator.free(indices_data);

    //@memset(indices_data, std.math.maxInt(u32));
    @memset(indices_data, 1);
    //indices_data[0] = 1;
    //@memcpy(indices_data[0..16], &[_]u32{ 3, 5, 17, 546, 43, 68, 22, 23, 24, 25, 26, 27, 89, 90, 91, 92 });
    var indices = try zml.Buffer.fromSlice(platform, indices_shape, indices_data);
    _ = try indices.awaitt();

    // call our executable module
    var result: zml.Buffer = mod.call(.{ cache, indices, input });
    defer result.deinit();
}

fn createRandomBuffer(allocator: std.mem.Allocator, platform: zml.Platform, shape: zml.Shape, random: std.Random) !zml.Buffer {
    const data = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(data);

    switch (shape.dtype()) {
        inline else => |v| {
            const ZigType = v.toZigType();
            switch (comptime v.class()) {
                .bool => unreachable,
                .integer => {
                    for (std.mem.bytesAsSlice(ZigType, data)) |*e| e.* = random.int(ZigType);
                },
                .float => {
                    const value = random.float(f64);
                    for (std.mem.bytesAsSlice(ZigType, data)) |*e| e.* = if (ZigType == f64)
                        value
                    else if (ZigType == f32)
                        @floatCast(value)
                    else if (ZigType == f16)
                        @floatCast(value)
                    else
                        @bitCast(random.int(std.meta.Int(.unsigned, @bitSizeOf(ZigType))));
                },
                .complex => unreachable,
            }
        },
    }

    var temp = try zml.Buffer.fromBytes(platform, shape, data);
    _ = try temp.awaitt();
    return temp;
}
