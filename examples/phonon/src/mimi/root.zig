const std = @import("std");
const zml = @import("zml");

const Model = model.Pocket_TTS_Audio_Tokenizer_Model;
const Phonon = @import("../phonon/root.zig").Phonon;
const model = @import("model.zig");
const util = @import("../util.zig");

/// Latent audio to actual audio.
pub const Mimi = struct {
    io: std.Io,
    gpa: std.mem.Allocator,

    platform: *const zml.Platform,

    model: Model,
    weights: zml.Bufferized(Model),

    exe: zml.Exe,
    runner: zml.Exe.Runner,

    pub const sampling_rate = 24_000;
    pub const samples_per_frame = 1920;

    /// Initializes a heap-allocated decoder; `self` may be undefined on entry.
    pub fn init_in_place(
        self: *Mimi,
        io: std.Io,
        gpa: std.mem.Allocator,
        platform: *const zml.Platform,
        trace_context: ?zml.tracer.Context,
    ) !void {
        const here = @typeName(@This()) ++ ".init_in_place";

        self.io = io;
        self.gpa = gpa;
        self.platform = platform;

        var span = zml.tracer.span_at_for(@src(), here, .{}, .{
            .consumer = trace_context,
        });
        defer span.end();

        // TODO: we're doing this at least twice
        const path_phonon = try util.assetPath(gpa, util.gradium_phonon_31m_en);
        defer gpa.free(path_phonon);
        var registry_phonon = try zml.safetensors.TensorRegistry.fromPath(gpa, io, path_phonon);
        defer registry_phonon.deinit();
        var store_phonon: zml.io.TensorStore = .fromRegistry(gpa, &registry_phonon, &.{});
        defer store_phonon.deinit();

        const path_mimi = try util.assetPath(gpa, util.gradium_phonon_mimi);
        defer gpa.free(path_mimi);
        var registry_mimi = try zml.safetensors.TensorRegistry.fromPath(gpa, io, path_mimi);
        defer registry_mimi.deinit();
        var store_mimi: zml.io.TensorStore = .fromRegistry(gpa, &registry_mimi, &.{});
        defer store_mimi.deinit();

        self.model = .init(store_phonon.view(), store_mimi.view());
        try self.load(&store_phonon, &store_mimi);
        errdefer zml.Buffer.deinitAll(Model, &self.weights);

        self.exe = try self.platform.compile(
            self.gpa,
            self.io,
            self.model,
            .forward,
            .{
                zml.Tensor.fromShape(.init(.{2}, .i32)),
                zml.Tensor.fromShape(.init(.{ .b = 1, .a = 34, .ld = 32 }, .f32)),
            },
            .{
                .program_name = @typeName(@This()),
            },
        );
        errdefer self.exe.deinit();
        util.dumpMemoryStats(@typeName(@This()), &self.exe);

        self.runner = try self.exe.runner(gpa);
        errdefer self.runner.deinit(gpa);
    }

    /// Load decoder weights from Mimi and latent normalization statistics from Phonon.
    pub fn load(self: *Mimi, store_phonon: *const zml.io.TensorStore, store_mimi: *const zml.io.TensorStore) !void {
        var span = zml.tracer.span_at(@src(), @typeName(@This()) ++ ".load", .{});
        defer span.end();

        self.weights = try self.model.bufferize(self.gpa);
        errdefer zml.Buffer.deinitAll(Model, &self.weights);
        var loader: zml.io.Loader = try .init(self.gpa, self.platform, .default);
        defer loader.deinit();
        try loader.load(self.io, Model, &self.model, &self.weights, store_mimi, .{});
        loader.loadBuffer(self.io, &self.weights.mean, &self.model.mean, store_phonon, .{});
        loader.loadBuffer(self.io, &self.weights.stddev, &self.model.stddev, store_phonon, .{});
        try loader.await(self.io);
    }

    pub fn deinit(self: *Mimi) void {
        const gpa = self.gpa;

        zml.Buffer.deinitAll(Model, &self.weights);

        self.runner.deinit(gpa);
        self.exe.deinit();
    }

    /// Borrows Phonon's post-step [2] i32 token/frame counts and latent history.
    pub fn forward(self: *Mimi, active_lengths: zml.Buffer, history34: zml.Buffer) !zml.Slice {
        const here = @typeName(@This()) ++ ".forward";

        const io = self.io;

        var span = zml.tracer.span_at(@src(), here, .{});
        defer span.end();

        var waveform: zml.Buffer = undefined;
        defer waveform.deinit();

        self.runner.run(io, .{ self.weights, active_lengths, history34 }, .{&waveform}, .{ .wait = true });

        return waveform.toSliceAlloc(self.gpa, self.io);
    }
};
