const std = @import("std");

const zml = @import("zml");

const audio_vae = @import("../draft/audio.zig");
const config = @import("../recipe/config.zig");
const dit = @import("../draft/dit.zig");
const encoder = @import("../draft/encoder.zig");
const sharding = @import("../recipe/shard.zig");

const log = std.log.scoped(.minimax_h3);

// =============================================================================
// serve/repo.zig — open the H3 / Turbo / encoder / audio checkpoints
// =============================================================================

pub const Open = struct {
    model: []const u8,
    dit: []const u8 = "",
};

pub const FileSource = struct {
    dir: std.Io.Dir,
    dir_owned: bool,
    file: ?[]u8 = null,

    fn deinit(self: *FileSource, allocator: std.mem.Allocator, io: std.Io) void {
        if (self.file) |name| allocator.free(name);
        if (self.dir_owned) self.dir.close(io);
        self.file = null;
        self.dir_owned = false;
    }
};

const Search = struct {
    repo: std.Io.Dir,
    task: std.Io.Dir,
};

pub const Bundle = struct {
    task: std.Io.Dir,
    task_owned: bool,
    dit_src: FileSource,
    enc_src: FileSource,
    audio_src: FileSource,

    dit_registry: *zml.safetensors.TensorRegistry,
    dit_store: zml.io.TensorStore,
    enc_registry: *zml.safetensors.TensorRegistry,
    enc_store: zml.io.TensorStore,
    audio_registry: *zml.safetensors.TensorRegistry,
    audio_store: zml.io.TensorStore,

    dit: dit.LoadedModel,
    enc: encoder.LoadedModel,
    audio: audio_vae.LoadedModel,

    pub fn open(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        shardings: sharding.Shardings,
        opts: Open,
    ) !Bundle {
        const task = try config.openTaskDir(io, repo);
        errdefer if (task.owned) task.dir.close(io);

        const search: Search = .{ .repo = repo, .task = task.dir };

        var dit_src = try resolveDit(allocator, io, search, opts);
        errdefer dit_src.deinit(allocator, io);
        var enc_src = try resolveComponent(io, search, "text_encoder", error.EncoderMissing);
        errdefer enc_src.deinit(allocator, io);
        var audio_src = try resolveComponent(io, search, "audio_vae", error.VaeMissing);
        errdefer audio_src.deinit(allocator, io);

        const dit_registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(dit_registry);
        dit_registry.* = try fetchRegistry(allocator, io, dit_src);
        errdefer dit_registry.deinit();
        try refuseUnsupported(dit_registry, allocator);
        var dit_store: zml.io.TensorStore = .fromRegistry(allocator, dit_registry);
        errdefer dit_store.deinit();

        const enc_registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(enc_registry);
        enc_registry.* = try fetchRegistry(allocator, io, enc_src);
        errdefer enc_registry.deinit();
        var enc_store: zml.io.TensorStore = .fromRegistry(allocator, enc_registry);
        errdefer enc_store.deinit();

        const audio_registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(audio_registry);
        audio_registry.* = try fetchRegistry(allocator, io, audio_src);
        errdefer audio_registry.deinit();
        var audio_store: zml.io.TensorStore = .fromRegistry(allocator, audio_registry);
        errdefer audio_store.deinit();
        if (!audio_vae.decodeReady(audio_store.view()))
            return error.VaeSchemaMismatch;

        var loaded_dit = try dit.LoadedModel.init(allocator, io, dit_src.dir, dit_store.view());
        errdefer loaded_dit.deinit(allocator);
        var loaded_enc = try encoder.LoadedModel.init(allocator, io, enc_src.dir, enc_store.view());
        errdefer loaded_enc.deinit(allocator);
        try shardings.checkLoaded(loaded_dit.cfg, loaded_enc.cfg);

        var loaded_audio = try audio_vae.LoadedModel.init(allocator, io, audio_src.dir, audio_store.view());
        errdefer loaded_audio.deinit(allocator);
        log.info("vae: audio decode graph ready", .{});

        return .{
            .task = task.dir,
            .task_owned = task.owned,
            .dit_src = dit_src,
            .enc_src = enc_src,
            .audio_src = audio_src,
            .dit_registry = dit_registry,
            .dit_store = dit_store,
            .enc_registry = enc_registry,
            .enc_store = enc_store,
            .audio_registry = audio_registry,
            .audio_store = audio_store,
            .dit = loaded_dit,
            .enc = loaded_enc,
            .audio = loaded_audio,
        };
    }

    pub fn deinit(self: *Bundle, allocator: std.mem.Allocator, io: std.Io) void {
        self.audio.deinit(allocator);
        self.enc.deinit(allocator);
        self.dit.deinit(allocator);
        self.audio_store.deinit();
        self.audio_registry.deinit();
        allocator.destroy(self.audio_registry);
        self.enc_store.deinit();
        self.enc_registry.deinit();
        allocator.destroy(self.enc_registry);
        self.dit_store.deinit();
        self.dit_registry.deinit();
        allocator.destroy(self.dit_registry);
        self.audio_src.deinit(allocator, io);
        self.enc_src.deinit(allocator, io);
        self.dit_src.deinit(allocator, io);
        if (self.task_owned) self.task.close(io);
    }
};

fn refuseUnsupported(registry: *zml.safetensors.TensorRegistry, allocator: std.mem.Allocator) !void {
    var keys: std.ArrayList([]const u8) = .empty;
    defer keys.deinit(allocator);
    var it = registry.iterator();
    while (it.next()) |e| try keys.append(allocator, e.key_ptr.*);
    const report = inspect(keys.items);
    if (refuseReason(report)) |why| {
        log.err("{s}", .{why});
        return error.UnsupportedCheckpoint;
    }
}

fn openOptionalDir(io: std.Io, parent: std.Io.Dir, name: []const u8) ?std.Io.Dir {
    return parent.openDir(io, name, .{}) catch null;
}

fn openShared(io: std.Io, task_dir: std.Io.Dir, repo: std.Io.Dir, name: []const u8) ?std.Io.Dir {
    if (openOptionalDir(io, task_dir, name)) |dir| return dir;
    if (openOptionalDir(io, repo, name)) |dir| return dir;
    return openOfficialNested(io, repo, name);
}

fn openOfficialNested(io: std.Io, repo: std.Io.Dir, name: []const u8) ?std.Io.Dir {
    if (openNestedDir(io, repo, config.task_dir, name)) |dir| return dir;
    return null;
}

pub const fused_overlay_name = "fused.safetensors";

fn fetchRegistry(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: FileSource,
) !zml.safetensors.TensorRegistry {
    var registry = if (src.file) |name| blk: {
        const file = try src.dir.openFile(io, name, .{ .mode = .read_only });
        defer file.close(io);
        log.info("weights: {s}", .{name});
        break :blk try zml.safetensors.fetchRegistry(allocator, io, src.dir, file);
    } else blk: {
        for (weight_entrypoints) |name| {
            if (src.dir.openFile(io, name, .{ .mode = .read_only })) |file| {
                defer file.close(io);
                log.info("weights: {s}", .{name});
                break :blk try zml.safetensors.fetchRegistry(allocator, io, src.dir, file);
            } else |_| {}
        }
        return error.FileNotFound;
    };

    // Super overlay dirs keep the official shard index *and* a LoRA-merged
    // `fused.safetensors` (363 tensors). The index alone is the unfused base.
    if (src.dir.openFile(io, fused_overlay_name, .{ .mode = .read_only })) |file| {
        defer file.close(io);
        if (src.file == null or !std.mem.eql(u8, src.file.?, fused_overlay_name)) {
            try zml.safetensors.parseSafetensors(allocator, io, &registry, file);
            log.info("weights: overlay {s}", .{fused_overlay_name});
        }
    } else |_| {}
    return registry;
}

fn fileInDir(io: std.Io, dir: std.Io.Dir, name: []const u8) bool {
    const file = dir.openFile(io, name, .{ .mode = .read_only }) catch return false;
    file.close(io);
    return true;
}

/// HF checkpoints use Transformers (`model.safetensors*`) or Diffusers
/// (`diffusion_pytorch_model*`) names. Empty task folders such as
/// `FL2VA/transformer` exist and must not win over the real shard dir.
pub const weight_entrypoints = [_][]const u8{
    "model.safetensors.index.json",
    "model.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors",
};

fn dirHasWeights(io: std.Io, dir: std.Io.Dir) bool {
    for (weight_entrypoints) |name| {
        if (fileInDir(io, dir, name)) return true;
    }
    return false;
}

fn takeWeightedDir(io: std.Io, dir: ?std.Io.Dir) ?std.Io.Dir {
    const opened = dir orelse return null;
    if (dirHasWeights(io, opened)) return opened;
    opened.close(io);
    return null;
}

fn resolveDit(
    allocator: std.mem.Allocator,
    io: std.Io,
    search: Search,
    opts: Open,
) !FileSource {
    if (opts.dit.len != 0) return openDitOverride(allocator, io, opts.dit);
    if (openOfficialDit(io, search)) |dir| {
        return .{ .dir = dir, .dir_owned = true, .file = null };
    }
    return error.TransformerMissing;
}

fn resolveComponent(io: std.Io, search: Search, official: []const u8, missing: anyerror) !FileSource {
    if (openShared(io, search.task, search.repo, official)) |dir| {
        return .{ .dir = dir, .dir_owned = true, .file = null };
    }
    return missing;
}

fn openDitOverride(allocator: std.mem.Allocator, io: std.Io, path: []const u8) !FileSource {
    const dir = zml.safetensors.resolveModelRepo(io, path) catch return error.TransformerMissing;
    if (std.mem.endsWith(u8, path, ".safetensors") or std.mem.endsWith(u8, path, ".safetensors.index.json")) {
        const name = allocator.dupe(u8, std.fs.path.basename(path)) catch |err| {
            dir.close(io);
            return err;
        };
        return .{ .dir = dir, .dir_owned = true, .file = name };
    }
    if (dirHasWeights(io, dir)) return .{ .dir = dir, .dir_owned = true, .file = null };
    dir.close(io);
    return error.TransformerMissing;
}

fn openOfficialDit(io: std.Io, search: Search) ?std.Io.Dir {
    return takeWeightedDir(io, openOptionalDir(io, search.task, "transformer")) orelse
        takeWeightedDir(io, openOptionalDir(io, search.repo, "transformer")) orelse
        takeWeightedDir(io, openNestedDir(io, search.repo, config.task_dir, "transformer"));
}

pub const tokenizer_relpaths = [_][]const u8{
    "tokenizer/tokenizer.json",
    "tokenizer.json",
};

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

    const bytes = try readTokenizerBytes(allocator, io, task_dir, repo);
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

fn readTokenizerBytes(
    allocator: std.mem.Allocator,
    io: std.Io,
    task_dir: std.Io.Dir,
    repo: std.Io.Dir,
) ![]u8 {
    if (readTokenizerAny(allocator, io, task_dir, repo)) |bytes| return bytes else |err| switch (err) {
        error.MissingTokenizer => {},
        else => return err,
    }
    return readOfficialTokenizer(allocator, io);
}

fn readTokenizerAny(
    allocator: std.mem.Allocator,
    io: std.Io,
    task_dir: std.Io.Dir,
    repo: std.Io.Dir,
) ![]u8 {
    const nearby = [_]std.Io.Dir{ task_dir, repo };
    for (nearby) |dir| {
        if (readTokenizer(allocator, io, dir)) |bytes| return bytes else |err| switch (err) {
            error.MissingTokenizer => {},
            else => return err,
        }
    }
    if (repo.openDir(io, config.task_dir, .{})) |dir| {
        defer dir.close(io);
        if (readTokenizer(allocator, io, dir)) |bytes| return bytes else |err| switch (err) {
            error.MissingTokenizer => {},
            else => return err,
        }
    } else |_| {}
    return error.MissingTokenizer;
}

fn readOfficialTokenizer(allocator: std.mem.Allocator, io: std.Io) ![]u8 {
    var buf: [256]u8 = undefined;
    const path = config.officialTokenizerUri(&buf) catch return error.MissingTokenizer;
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.MissingTokenizer;
    defer file.close(io);
    log.info("tokenizer: {s}", .{path});
    return readTokenizerFile(allocator, io, file);
}

fn readTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) ![]u8 {
    for (tokenizer_relpaths) |name| {
        const file = dir.openFile(io, name, .{}) catch continue;
        defer file.close(io);
        return readTokenizerFile(allocator, io, file);
    }
    return error.MissingTokenizer;
}

fn readTokenizerFile(allocator: std.mem.Allocator, io: std.Io, file: std.Io.File) ![]u8 {
    var reader = file.reader(io, &.{});
    return try reader.interface.readAlloc(allocator, try file.length(io));
}

pub const Report = struct {
    has_adaln_proj: bool = false,
    has_time: bool = false,
};

fn hasKey(keys: []const []const u8, suffix: []const u8) bool {
    for (keys) |key| {
        if (std.mem.endsWith(u8, key, suffix) or std.mem.eql(u8, key, suffix)) return true;
    }
    return false;
}

pub fn inspect(keys: []const []const u8) Report {
    return .{
        .has_adaln_proj = hasKey(keys, "adaln_proj.linear.weight"),
        .has_time = hasKey(keys, "time_embedder.linear_1.weight"),
    };
}

pub fn refuseReason(report: Report) ?[]const u8 {
    if (!report.has_adaln_proj) return "AdaLN projection weights missing; not a recognized H3 DiT";
    if (!report.has_time) return "time_embedder missing; not a recognized H3 DiT";
    return null;
}

/// Read DiT/encoder `config.json` so the physical mesh can even-split the real head counts.
pub fn peekHeadCounts(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo_dir: std.Io.Dir,
    opts: Open,
) !sharding.HeadCounts {
    const task = try config.openTaskDir(io, repo_dir);
    defer if (task.owned) task.dir.close(io);
    const search: Search = .{ .repo = repo_dir, .task = task.dir };
    var dit_src = try resolveDit(allocator, io, search, opts);
    defer dit_src.deinit(allocator, io);
    var enc_src = try resolveComponent(io, search, "text_encoder", error.EncoderMissing);
    defer enc_src.deinit(allocator, io);
    const dit_cfg = try config.loadDitConfig(allocator, io, dit_src.dir);
    const enc_cfg = try config.loadEncoderConfig(allocator, io, enc_src.dir);
    return .{
        .dit = dit_cfg.num_attention_heads,
        .enc = enc_cfg.num_attention_heads,
        .kv = enc_cfg.num_key_value_heads,
    };
}
