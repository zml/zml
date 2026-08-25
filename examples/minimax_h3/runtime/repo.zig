const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("../vae/audio.zig");
const checkpoint = @import("../core/checkpoint.zig");
const config = @import("../core/config.zig");
const dit = @import("../model/dit.zig");
const encoder = @import("../model/encoder.zig");
const sharding = @import("../core/sharding.zig");
const visual_vae = @import("../vae/visual.zig");

const log = std.log.scoped(.minimax_h3);

pub const Bundle = struct {
    task: std.Io.Dir,
    task_owned: bool,
    transformer: std.Io.Dir,
    encoder: std.Io.Dir,
    visual_cfg: std.Io.Dir,
    visual_source: ?std.Io.Dir,
    audio_dir: std.Io.Dir,

    dit_registry: *zml.safetensors.TensorRegistry,
    dit_store: zml.io.TensorStore,
    enc_registry: *zml.safetensors.TensorRegistry,
    enc_store: zml.io.TensorStore,
    visual_registry: *zml.safetensors.TensorRegistry,
    visual_store: zml.io.TensorStore,
    audio_registry: *zml.safetensors.TensorRegistry,
    audio_store: zml.io.TensorStore,

    dit: dit.LoadedModel,
    enc: encoder.LoadedModel,
    visual: visual_vae.LoadedModel,
    audio: audio_vae.LoadedModel,

    pub fn open(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        variant: config.Variant,
        shardings: sharding.Shardings,
    ) !Bundle {
        const task = try config.openTaskDir(io, repo, variant);
        errdefer if (task.owned) task.dir.close(io);

        var transformer = try openTransformer(io, task.dir, repo, variant);
        errdefer transformer.close(io);
        var encoder_dir = openSharedComponent(io, task.dir, repo, "text_encoder") orelse
            return error.EncoderMissing;
        errdefer encoder_dir.close(io);
        var visual_cfg = openSharedComponent(io, task.dir, repo, "video_vae") orelse
            return error.VaeMissing;
        errdefer visual_cfg.close(io);
        var visual_source = openOptionalDir(io, visual_cfg, "source");
        errdefer if (visual_source) |*dir| dir.close(io);
        var audio_dir = openSharedComponent(io, task.dir, repo, "audio_vae") orelse
            return error.VaeMissing;
        errdefer audio_dir.close(io);

        const visual_weights = visual_source orelse visual_cfg;

        const dit_registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(dit_registry);
        dit_registry.* = try .fromRepo(allocator, io, transformer);
        errdefer dit_registry.deinit();
        try refuseUnsupported(dit_registry, allocator);
        var dit_store: zml.io.TensorStore = .fromRegistry(allocator, dit_registry);
        errdefer dit_store.deinit();

        const enc_registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(enc_registry);
        enc_registry.* = try .fromRepo(allocator, io, encoder_dir);
        errdefer enc_registry.deinit();
        var enc_store: zml.io.TensorStore = .fromRegistry(allocator, enc_registry);
        errdefer enc_store.deinit();

        const visual_registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(visual_registry);
        visual_registry.* = try .fromRepo(allocator, io, visual_weights);
        errdefer visual_registry.deinit();
        var visual_store: zml.io.TensorStore = .fromRegistry(allocator, visual_registry);
        errdefer visual_store.deinit();

        const audio_registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(audio_registry);
        audio_registry.* = try .fromRepo(allocator, io, audio_dir);
        errdefer audio_registry.deinit();
        var audio_store: zml.io.TensorStore = .fromRegistry(allocator, audio_registry);
        errdefer audio_store.deinit();
        if (!visual_vae.ready(visual_store.view()) or !audio_vae.decodeReady(audio_store.view()))
            return error.VaeSchemaMismatch;

        var loaded_dit = try dit.LoadedModel.init(allocator, io, transformer, dit_store.view());
        errdefer loaded_dit.deinit(allocator);
        var loaded_enc = try encoder.LoadedModel.init(allocator, io, encoder_dir, enc_store.view());
        errdefer loaded_enc.deinit(allocator);
        try shardings.checkLoaded(loaded_dit.cfg, loaded_enc.cfg);

        var loaded_visual = try visual_vae.LoadedModel.init(allocator, io, visual_cfg, visual_store.view());
        errdefer loaded_visual.deinit(allocator);
        var loaded_audio = try audio_vae.LoadedModel.init(allocator, io, audio_dir, audio_store.view());
        errdefer loaded_audio.deinit(allocator);
        log.info("vae: video+audio graphs ready", .{});

        return .{
            .task = task.dir,
            .task_owned = task.owned,
            .transformer = transformer,
            .encoder = encoder_dir,
            .visual_cfg = visual_cfg,
            .visual_source = visual_source,
            .audio_dir = audio_dir,
            .dit_registry = dit_registry,
            .dit_store = dit_store,
            .enc_registry = enc_registry,
            .enc_store = enc_store,
            .visual_registry = visual_registry,
            .visual_store = visual_store,
            .audio_registry = audio_registry,
            .audio_store = audio_store,
            .dit = loaded_dit,
            .enc = loaded_enc,
            .visual = loaded_visual,
            .audio = loaded_audio,
        };
    }

    pub fn deinit(self: *Bundle, allocator: std.mem.Allocator, io: std.Io) void {
        self.audio.deinit(allocator);
        self.visual.deinit(allocator);
        self.enc.deinit(allocator);
        self.dit.deinit(allocator);
        self.audio_store.deinit();
        self.audio_registry.deinit();
        allocator.destroy(self.audio_registry);
        self.visual_store.deinit();
        self.visual_registry.deinit();
        allocator.destroy(self.visual_registry);
        self.enc_store.deinit();
        self.enc_registry.deinit();
        allocator.destroy(self.enc_registry);
        self.dit_store.deinit();
        self.dit_registry.deinit();
        allocator.destroy(self.dit_registry);
        self.audio_dir.close(io);
        if (self.visual_source) |*dir| dir.close(io);
        self.visual_cfg.close(io);
        self.encoder.close(io);
        self.transformer.close(io);
        if (self.task_owned) self.task.close(io);
    }
};

fn refuseUnsupported(registry: *zml.safetensors.TensorRegistry, allocator: std.mem.Allocator) !void {
    var keys: std.ArrayList([]const u8) = .empty;
    defer keys.deinit(allocator);
    var it = registry.iterator();
    while (it.next()) |e| try keys.append(allocator, e.key_ptr.*);
    if (checkpoint.refuseReason(checkpoint.inspect(keys.items))) |why| {
        log.err("{s}", .{why});
        return error.UnsupportedCheckpoint;
    }
}

pub fn openOptionalDir(io: std.Io, parent: std.Io.Dir, name: []const u8) ?std.Io.Dir {
    return parent.openDir(io, name, .{}) catch null;
}

pub fn openSharedComponent(io: std.Io, task_dir: std.Io.Dir, repo: std.Io.Dir, name: []const u8) ?std.Io.Dir {
    if (openOptionalDir(io, task_dir, name)) |dir| return dir;
    if (openOptionalDir(io, repo, name)) |dir| return dir;
    return openNestedDir(io, repo, "FL2VA", name);
}

pub fn openTransformer(io: std.Io, task_dir: std.Io.Dir, repo: std.Io.Dir, variant: config.Variant) !std.Io.Dir {
    if (openOptionalDir(io, task_dir, "transformer")) |dir| return dir;
    if (variant == .ref2va) {
        if (openOptionalDir(io, repo, "transformer_ref")) |dir| return dir;
        if (openNestedDir(io, repo, "Ref2VA", "transformer")) |dir| return dir;
        return error.Ref2vaTransformerMissing;
    }
    if (openOptionalDir(io, repo, "transformer")) |dir| return dir;
    if (openNestedDir(io, repo, "FL2VA", "transformer")) |dir| return dir;
    return error.TransformerMissing;
}

pub fn loadTokenizer(
    allocator: std.mem.Allocator,
    io: std.Io,
    task_dir: std.Io.Dir,
    repo: std.Io.Dir,
    progress: *std.Progress.Node,
) !zml.tokenizer.Tokenizer {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Loading tokenizer...", 1);
    defer node.end();

    const bytes = try readTokenizerAny(allocator, io, task_dir, repo);
    defer allocator.free(bytes);
    log.info("tokenizer: {d} bytes", .{bytes.len});
    return try .fromBytes(allocator, bytes);
}

fn openNestedDir(io: std.Io, parent: std.Io.Dir, first: []const u8, second: []const u8) ?std.Io.Dir {
    var outer = parent.openDir(io, first, .{}) catch return null;
    const inner = outer.openDir(io, second, .{}) catch {
        outer.close(io);
        return null;
    };
    outer.close(io);
    return inner;
}

fn readTokenizerAny(allocator: std.mem.Allocator, io: std.Io, task_dir: std.Io.Dir, repo: std.Io.Dir) ![]u8 {
    if (readTokenizer(allocator, io, task_dir)) |bytes| return bytes else |err| switch (err) {
        error.MissingTokenizer => {},
        else => return err,
    }
    if (readTokenizer(allocator, io, repo)) |bytes| return bytes else |err| switch (err) {
        error.MissingTokenizer => {},
        else => return err,
    }
    var fl = repo.openDir(io, "FL2VA", .{}) catch return error.MissingTokenizer;
    defer fl.close(io);
    return readTokenizer(allocator, io, fl);
}

fn readTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) ![]u8 {
    const names = [_][]const u8{ "tokenizer/tokenizer.json", "tokenizer.json" };
    for (names) |name| {
        const file = dir.openFile(io, name, .{}) catch continue;
        defer file.close(io);
        var reader = file.reader(io, &.{});
        return try reader.interface.readAlloc(allocator, try file.length(io));
    }
    return error.MissingTokenizer;
}
