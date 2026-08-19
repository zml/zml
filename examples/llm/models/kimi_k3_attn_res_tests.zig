const std = @import("std");

const zml = @import("zml");
const attn_res = @import("kimi_k3/attn_res.zig");

comptime {
    @setEvalBranchQuota(100_000);
}

pub const std_options: std.Options = .{ .log_level = .info };

const Args = struct {
    fixture: []const u8,

    pub const help =
        \\Use kimi_k3_attn_res_tests --fixture=<attn-res-reference.safetensors>
        \\
        \\Run Kimi K3 Attention Residual differential tests on NVIDIA CUDA only.
        \\
    ;
};

const strict: zml.testing.CompareOpts = .{
    .absolute_tolerance = 1e-5,
    .relative_tolerance = 1e-5,
    .minimum_close_fraction = 1.0,
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: zml.io.TensorStore.View,
    sharding: zml.Sharding,
    stdout: *std.Io.Writer,

    fn runCase(self: *Context, name: []const u8) !void {
        @setEvalBranchQuota(100_000);
        const prefix_key = try std.fmt.allocPrint(self.allocator, "{s}.prefix", .{name});
        defer self.allocator.free(prefix_key);
        const blocks_key = try std.fmt.allocPrint(self.allocator, "{s}.blocks", .{name});
        defer self.allocator.free(blocks_key);
        const active_key = try std.fmt.allocPrint(self.allocator, "{s}.active", .{name});
        defer self.allocator.free(active_key);
        const norm_key = try std.fmt.allocPrint(self.allocator, "{s}.norm_weight", .{name});
        defer self.allocator.free(norm_key);
        const projection_key = try std.fmt.allocPrint(self.allocator, "{s}.projection_weight", .{name});
        defer self.allocator.free(projection_key);

        var prefix_buffer = try self.load(prefix_key);
        defer prefix_buffer.deinit();
        var blocks_buffer = try self.load(blocks_key);
        defer blocks_buffer.deinit();
        var active_buffer = try self.load(active_key);
        defer active_buffer.deinit();
        var norm_buffer = try self.load(norm_key);
        defer norm_buffer.deinit();
        var projection_buffer = try self.load(projection_key);
        defer projection_buffer.deinit();

        const prefix = zml.Tensor.fromShape(prefix_buffer.shape()).withTags(.{ .token, .d });
        const blocks = zml.Tensor.fromShape(blocks_buffer.shape()).withTags(.{ .token, .source, .d });
        const active = zml.Tensor.fromShape(active_buffer.shape()).withTags(.{.source});
        const norm = zml.Tensor.fromShape(norm_buffer.shape()).withTags(.{.d});
        const projection = zml.Tensor.fromShape(projection_buffer.shape()).withTags(.{.d});
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            attn_res.selectEps1e6,
            .{ prefix, blocks, active, norm, projection },
            .{ .shardings = &.{self.sharding} },
        );
        defer exe.deinit();

        const started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        var actual = try zml.testing.autoCall(
            self.allocator,
            self.io,
            &exe,
            attn_res.selectEps1e6,
            .{ prefix_buffer, blocks_buffer, active_buffer, norm_buffer, projection_buffer },
        );
        defer actual.output.deinit();
        defer actual.candidates.deinit();
        defer actual.scores.deinit();
        defer actual.probabilities.deinit();

        try self.compareExpected(name, "expected.output", actual.output, strict);
        try self.compareExpected(name, "expected.candidates", actual.candidates, zml.testing.CompareOpts.exact_match);
        try self.compareExpected(name, "expected.scores", actual.scores, strict);
        try self.compareExpected(name, "expected.probabilities", actual.probabilities, strict);
        const elapsed_us = @divTrunc(
            std.Io.Clock.now(.real, self.io).toNanoseconds() - started,
            1000,
        );
        // KIMI_K3_TEMP_REMOVE_M20: selector candidate/probability shapes and
        // synchronized timing are bring-up diagnostics removed at cleanup.
        try self.stdout.print(
            "KIMI_K3_ATTN_RES_PASS name={s} elapsed_us={} candidates={f} probabilities={f}\n",
            .{ name, elapsed_us, actual.candidates.shape(), actual.probabilities.shape() },
        );
        try self.stdout.flush();
    }

    fn compareExpected(
        self: *Context,
        name: []const u8,
        suffix: []const u8,
        actual: zml.Buffer,
        opts: zml.testing.CompareOpts,
    ) !void {
        const key = try std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ name, suffix });
        defer self.allocator.free(key);
        var expected = try self.load(key);
        defer expected.deinit();
        try zml.testing.expectClose(self.io, actual, expected, opts);
    }

    fn load(self: *Context, key: []const u8) !zml.Buffer {
        const shape = self.store.getShape(key) orelse return error.MissingAttentionResidualFixture;
        const bytes = try self.allocator.alloc(u8, shape.byteSize());
        defer self.allocator.free(bytes);
        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.store.getReader(key, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(bytes);
        return zml.Buffer.fromBytes(self.io, self.platform, shape, self.sharding, bytes);
    }
};

fn testWorkspace(stdout: *std.Io.Writer) !void {
    var workspace = attn_res.DepthWorkspace.reset();
    for (0..14) |layer| {
        workspace.beginLayer();
        const boundary = workspace.appendBoundary(layer, 12);
        if (boundary != (layer == 0 or layer == 12)) return error.AttentionResidualBoundaryMismatch;
        workspace.addBranch();
        const expected_blocks: usize = if (layer < 12) 1 else 2;
        if (workspace.active_blocks != expected_blocks or !workspace.prefix_valid) {
            return error.AttentionResidualWorkspaceMismatch;
        }
    }
    workspace = attn_res.DepthWorkspace.reset();
    if (workspace.active_blocks != 0 or workspace.prefix_valid) return error.StaleAttentionResidualWorkspace;
    try stdout.writeAll("KIMI_K3_ATTN_RES_WORKSPACE_PASS boundaries=0,12 reset_active_blocks=0\n");
    try stdout.flush();
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const platform: *zml.Platform = try .init(allocator, io, .cuda, .{
        .xla_gpu = .{ .allocator = .{ .bfc = .{ .preallocate = false, .memory_fraction = 0.25 } } },
    });
    defer platform.deinit(allocator, io);
    if (platform.target != .cuda) return error.NvidiaCudaRequired;

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.fixture);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    var stdout_file = std.Io.File.stdout().writerStreaming(io, &.{});
    try testWorkspace(&stdout_file.interface);

    var context: Context = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .store = store.view(),
        .sharding = platform.replicated_sharding,
        .stdout = &stdout_file.interface,
    };
    const cases = [_][]const u8{
        "synthetic.one_source",
        "synthetic.multiple_sources",
        "synthetic.all_sources",
        "synthetic.inactive_stale",
        "real.layer0.mlp",
        "real.layer1.self_attention",
        "real.layer1.mlp",
        "real.layer2.self_attention",
        "real.layer2.mlp",
        "real.layer3.self_attention",
        "real.layer3.mlp",
        "real.output",
    };
    for (cases) |name| try context.runCase(name);
    try stdout_file.interface.writeAll("KIMI_K3_ATTN_RES_ALL_PASS count=12 backend=cuda\n");
    try stdout_file.interface.flush();
}
