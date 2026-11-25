const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const main_ = @import("main.zig");
const masking = @import("masking.zig");
const model = @import("model.zig");

const Tensor = zml.Tensor;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: []const u8,

    pub const help =
        \\Use test-activations --model=<path> --activations=<path>
        \\
        \\ Validate the Timestral implementation against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>            Path to the model repository
        \\   --activations=<path>      Path to activation safetensors
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    var loaded: LoadedModel = undefined;
    try loaded.init(allocator, io, platform, args.model);
    defer loaded.deinit();

    var activation_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.activations);
    defer activation_registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &activation_registry, &.{platform.replicated_mesh});
    defer activation_store.deinit();

    const ctx: TestContext = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .activations = activation_store.view(),
    };

    try ctx.testLayer(PreprocessorTargetsSeriesTest{}, "preprocessor_targets", {});
    try ctx.testLayer(PreprocessorRowsTest{}, "preprocessor_rows", {});
    try ctx.testLayer(PatchBuilderTest{ .inner = loaded.model.patch_builder }, "model.patch_builder", {});
    try ctx.testLayer(AttentionLocatorTest{ .inner = loaded.model.attn_masking }, "model.attn_masking", {});
    try ctx.testLayer(
        PatchRepresentationTest{ .inner = loaded.model.representation },
        "model.patch_representation",
        .{ .inner = loaded.weights.representation },
    );
}

/// Arguments shared by every `zml.testing.testLayer` call.
const TestContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    activations: zml.io.TensorStore.View,

    fn testLayer(self: TestContext, layer: anytype, name: []const u8, layer_weights: zml.Bufferized(@TypeOf(layer))) !void {
        return zml.testing.testLayer(
            self.allocator,
            self.io,
            self.platform,
            layer,
            .forward,
            self.activations,
            name,
            layer_weights,
            &.{},
            .{},
        );
    }
};

/// Symbolic model with its weights loaded on the platform.
const LoadedModel = struct {
    config: std.json.Parsed(model.Config),
    registry: zml.safetensors.TensorRegistry,
    store: zml.io.TensorStore,
    model: model.Model,
    weights: zml.Bufferized(model.Model),

    /// Initialized in place because `store` keeps a pointer to `registry`.
    fn init(self: *LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, path: []const u8) !void {
        var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
        defer dir.close(io);

        self.config = try main_.parseConfig(model.Config, allocator, io, dir);
        errdefer self.config.deinit();

        self.registry = try .fromPath(allocator, io, path);
        errdefer self.registry.deinit();

        self.store = .fromRegistry(allocator, &self.registry, &.{platform.replicated_mesh});
        errdefer self.store.deinit();

        // Keeps slices into `config`, which must outlive it.
        self.model = .init(self.store.view(), self.config.value);

        self.weights = try zml.mem.bufferize(allocator, model.Model, &self.model);
        errdefer deinitBuffers(&self.weights);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.load(io, model.Model, &self.model, &self.weights, &self.store, .{});
        try loader.await(io);
    }

    fn deinit(self: *LoadedModel) void {
        deinitBuffers(&self.weights);
        self.store.deinit();
        self.registry.deinit();
        self.config.deinit();
    }

    fn deinitBuffers(weights: *zml.Bufferized(model.Model)) void {
        zml.meta.visit(struct {
            fn cb(_: void, b: *zml.Buffer) void {
                b.deinit();
            }
        }.cb, {}, weights);
    }
};

/// `testLayer` creates untagged inputs and passes the layer by value.
const PreprocessorTargetsSeriesTest = struct {
    pub fn forward(_: PreprocessorTargetsSeriesTest, targets: Tensor) model.PreProcessor.TargetsSeries {
        return .make(targets.withTags(.{ .batch, .variate, .time }));
    }
};

/// `testLayer` creates untagged inputs and passes the layer by value.
const PreprocessorRowsTest = struct {
    const Rows = @typeInfo(@TypeOf(model.PreProcessor.forward)).@"fn".return_type.?;

    pub fn forward(_: PreprocessorRowsTest, values: Tensor, mask: Tensor) Rows {
        return .make(.{
            .values = values.withTags(.{ .batch, .variate, .time }),
            .mask = mask.withTags(.{ .batch, .variate, .time }),
        });
    }
};

/// `testLayer` creates untagged inputs and passes the layer by value.
const PatchBuilderTest = struct {
    inner: model.PatchBuilder,

    pub fn forward(self: PatchBuilderTest, values: Tensor, mask: Tensor) @typeInfo(@TypeOf(model.PatchBuilder.forward)).@"fn".return_type.? {
        return self.inner.forward(.{
            .values = values.withTags(.{ .row, .time }),
            .mask = mask.withTags(.{ .row, .time }),
            // Unused by the scaler.
            .roles = undefined,
            .group = undefined,
        });
    }
};

/// `testLayer` creates untagged inputs and passes the layer by value.
const PatchRepresentationTest = struct {
    inner: @FieldType(model.Model, "representation"),

    pub fn forward(self: PatchRepresentationTest, values: Tensor, mask: Tensor) Tensor {
        return self.inner.forward(.{
            .values = values.withTags(.{ .row, .token, .h }),
            .mask = mask.withTags(.{ .row, .token, .h }),
        });
    }
};

/// `testLayer` creates untagged inputs and passes the layer by value.
const AttentionLocatorTest = struct {
    inner: model.AttentionLocator,

    pub fn forward(self: AttentionLocatorTest, observed: Tensor, group: Tensor) @typeInfo(@TypeOf(model.AttentionLocator.forward)).@"fn".return_type.? {
        return self.inner.forward(
            observed.withTags(.{ .row, .token, .h }),
            group.withTags(.{.row}),
        );
    }
};
