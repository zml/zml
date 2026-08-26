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
    extra: ?std.Io.Dir,
    task: std.Io.Dir,
};

pub const Bundle = struct {
    task: std.Io.Dir,
    task_owned: bool,
    dit_src: FileSource,
    enc_src: FileSource,
    visual_src: FileSource,
    audio_src: FileSource,
    visual_source: ?std.Io.Dir,

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
        opts: Open,
    ) !Bundle {
        const task = try config.openTaskDir(io, repo, variant);
        errdefer if (task.owned) task.dir.close(io);

        var extra = openBundleRoot(io, opts.model);
        defer if (extra) |*dir| dir.close(io);
        const search: Search = .{ .repo = repo, .extra = extra, .task = task.dir };

        var dit_src = try resolveDit(allocator, io, search, variant, opts);
        errdefer dit_src.deinit(allocator, io);
        var enc_src = try resolveComponent(allocator, io, search, .{
            .official = "text_encoder",
            .scan = "text_encoders",
            .needles = &.{},
            .missing = error.EncoderMissing,
        });
        errdefer enc_src.deinit(allocator, io);
        var visual_src = try resolveComponent(allocator, io, search, .{
            .official = "video_vae",
            .aliases = &.{ "visual_vae", "vae" },
            .scan = "vae",
            .needles = &.{ "video", "vae" },
            .missing = error.VaeMissing,
        });
        errdefer visual_src.deinit(allocator, io);
        var audio_src = try resolveComponent(allocator, io, search, .{
            .official = "audio_vae",
            .scan = "vae",
            .needles = &.{ "audio", "vae" },
            .missing = error.VaeMissing,
        });
        errdefer audio_src.deinit(allocator, io);

        var visual_source = if (visual_src.file == null) openOptionalDir(io, visual_src.dir, "source") else null;
        errdefer if (visual_source) |*dir| dir.close(io);
        const visual_weights = visual_source orelse visual_src.dir;

        const dit_registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(dit_registry);
        dit_registry.* = try openRegistry(allocator, io, dit_src);
        errdefer dit_registry.deinit();
        try refuseUnsupported(dit_registry, allocator);
        var dit_store: zml.io.TensorStore = .fromRegistry(allocator, dit_registry);
        errdefer dit_store.deinit();

        const enc_registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(enc_registry);
        enc_registry.* = try openRegistry(allocator, io, enc_src);
        errdefer enc_registry.deinit();
        var enc_store: zml.io.TensorStore = .fromRegistry(allocator, enc_registry);
        errdefer enc_store.deinit();

        const visual_registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(visual_registry);
        visual_registry.* = try openRegistry(allocator, io, .{
            .dir = visual_weights,
            .dir_owned = false,
            .file = visual_src.file,
        });
        errdefer visual_registry.deinit();
        var visual_store: zml.io.TensorStore = .fromRegistry(allocator, visual_registry);
        errdefer visual_store.deinit();

        const audio_registry = try allocator.create(zml.safetensors.TensorRegistry);
        errdefer allocator.destroy(audio_registry);
        audio_registry.* = try openRegistry(allocator, io, audio_src);
        errdefer audio_registry.deinit();
        var audio_store: zml.io.TensorStore = .fromRegistry(allocator, audio_registry);
        errdefer audio_store.deinit();
        if (!visual_vae.ready(visual_store.view()) or !audio_vae.decodeReady(audio_store.view()))
            return error.VaeSchemaMismatch;

        var loaded_dit = try dit.LoadedModel.init(allocator, io, dit_src.dir, dit_store.view());
        errdefer loaded_dit.deinit(allocator);
        var loaded_enc = try encoder.LoadedModel.init(allocator, io, enc_src.dir, enc_store.view());
        errdefer loaded_enc.deinit(allocator);
        try shardings.checkLoaded(loaded_dit.cfg, loaded_enc.cfg);

        var loaded_visual = try visual_vae.LoadedModel.init(allocator, io, visual_src.dir, visual_store.view());
        errdefer loaded_visual.deinit(allocator);
        var loaded_audio = try audio_vae.LoadedModel.init(allocator, io, audio_src.dir, audio_store.view());
        errdefer loaded_audio.deinit(allocator);
        log.info("vae: video+audio graphs ready", .{});

        return .{
            .task = task.dir,
            .task_owned = task.owned,
            .dit_src = dit_src,
            .enc_src = enc_src,
            .visual_src = visual_src,
            .audio_src = audio_src,
            .visual_source = visual_source,
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
        if (self.visual_source) |*dir| dir.close(io);
        self.audio_src.deinit(allocator, io);
        self.visual_src.deinit(allocator, io);
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
    const report = checkpoint.inspect(keys.items);
    if (checkpoint.refuseReason(report)) |why| {
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
    for (config.official_task_dirs) |task| {
        if (openNestedDir(io, repo, task, name)) |dir| return dir;
    }
    return null;
}

fn openRegistry(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: FileSource,
) !zml.safetensors.TensorRegistry {
    if (src.file) |name| {
        const file = try src.dir.openFile(io, name, .{ .mode = .read_only });
        defer file.close(io);
        log.info("weights: {s}", .{name});
        return zml.safetensors.fetchRegistry(allocator, io, src.dir, file);
    }
    for (weight_entrypoints) |name| {
        if (src.dir.openFile(io, name, .{ .mode = .read_only })) |file| {
            defer file.close(io);
            log.info("weights: {s}", .{name});
            return zml.safetensors.fetchRegistry(allocator, io, src.dir, file);
        } else |_| {}
    }
    return error.FileNotFound;
}

fn fileInDir(io: std.Io, dir: std.Io.Dir, name: []const u8) bool {
    const file = dir.openFile(io, name, .{ .mode = .read_only }) catch return false;
    file.close(io);
    return true;
}

/// Official HF dumps use either Transformers (`model.safetensors*`) or
/// Diffusers (`diffusion_pytorch_model*`) names. Empty task folders such as
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
    variant: config.Variant,
    opts: Open,
) !FileSource {
    if (opts.dit.len != 0)
        return openFilePath(allocator, io, search, opts.dit, &.{"diffusion_models"}) catch
            return error.TransformerMissing;
    if (std.mem.endsWith(u8, opts.model, ".safetensors")) {
        return .{
            .dir = search.repo,
            .dir_owned = false,
            .file = try allocator.dupe(u8, std.fs.path.basename(opts.model)),
        };
    }
    if (openOfficialDit(io, search, variant)) |dir| {
        return .{ .dir = dir, .dir_owned = true, .file = null };
    }
    const needles: []const []const u8 = switch (variant.taskFamily()) {
        .fl2va => &.{"fl2va"},
        .ref2va => &.{"ref2va"},
    };
    const missing: anyerror = if (variant.taskFamily() == .ref2va)
        error.Ref2vaTransformerMissing
    else
        error.TransformerMissing;
    return takeScan(
        scanIn(allocator, io, search, "diffusion_models", needles, true),
        missing,
        error.AmbiguousDit,
    );
}

const ComponentSpec = struct {
    official: []const u8,
    aliases: []const []const u8 = &.{},
    scan: []const u8,
    needles: []const []const u8,
    missing: anyerror,
};

fn resolveComponent(
    allocator: std.mem.Allocator,
    io: std.Io,
    search: Search,
    spec: ComponentSpec,
) !FileSource {
    if (openShared(io, search.task, search.repo, spec.official)) |dir| {
        return .{ .dir = dir, .dir_owned = true, .file = null };
    }
    for (spec.aliases) |name| {
        if (openShared(io, search.task, search.repo, name)) |dir| {
            return .{ .dir = dir, .dir_owned = true, .file = null };
        }
    }
    const src = (try scanIn(allocator, io, search, spec.scan, spec.needles, false)) orelse return spec.missing;
    return src;
}

fn openOfficialDit(io: std.Io, search: Search, variant: config.Variant) ?std.Io.Dir {
    // Official dump has no Ref2VA/. openTaskDir then falls back to the repo, so
    // task/transformer is the fl2va DiT and must not win for ref2va.
    return switch (variant.taskFamily()) {
        .ref2va => takeWeightedDir(io, openOptionalDir(io, search.repo, "transformer_ref")) orelse
            takeWeightedDir(io, openNestedDir(io, search.repo, config.taskDirName(.ref2va), "transformer")),
        .fl2va => takeWeightedDir(io, openOptionalDir(io, search.task, "transformer")) orelse
            takeWeightedDir(io, openOptionalDir(io, search.repo, "transformer")) orelse
            takeWeightedDir(io, openNestedDir(io, search.repo, config.taskDirName(.fl2va), "transformer")),
    };
}

fn openFilePath(
    allocator: std.mem.Allocator,
    io: std.Io,
    search: Search,
    path: []const u8,
    folders: []const []const u8,
) !FileSource {
    const base = std.fs.path.basename(path);
    if (std.fs.path.dirname(path) != null and (std.mem.indexOfScalar(u8, path, '/') != null or std.mem.indexOfScalar(u8, path, '\\') != null)) {
        const dir = try zml.safetensors.resolveModelRepo(io, path);
        return .{ .dir = dir, .dir_owned = true, .file = try allocator.dupe(u8, base) };
    }
    if (fileInDir(io, search.repo, base)) {
        return .{ .dir = search.repo, .dir_owned = false, .file = try allocator.dupe(u8, base) };
    }
    if (try fileInFolders(allocator, io, search.repo, base, folders)) |src| return src;
    if (search.extra) |root| {
        if (try fileInFolders(allocator, io, root, base, folders)) |src| return src;
    }
    return error.FileNotFound;
}

fn takeScan(result: anytype, missing: anyerror, ambiguous: anyerror) !FileSource {
    const src = result catch |err| switch (err) {
        error.AmbiguousWeights => return ambiguous,
        else => |e| return e,
    };
    return src orelse return missing;
}

fn fileInFolders(
    allocator: std.mem.Allocator,
    io: std.Io,
    root: std.Io.Dir,
    base: []const u8,
    folders: []const []const u8,
) !?FileSource {
    for (folders) |folder| {
        if (openOptionalDir(io, root, folder)) |dir| {
            if (fileInDir(io, dir, base)) {
                return .{ .dir = dir, .dir_owned = true, .file = try allocator.dupe(u8, base) };
            }
            dir.close(io);
        }
    }
    return null;
}

fn scanIn(
    allocator: std.mem.Allocator,
    io: std.Io,
    search: Search,
    folder: []const u8,
    needles: []const []const u8,
    unique: bool,
) !?FileSource {
    if (try scanFolder(allocator, io, search.repo, folder, needles, unique)) |src| return src;
    if (search.extra) |root| {
        if (try scanFolder(allocator, io, root, folder, needles, unique)) |src| return src;
    }
    return null;
}

fn scanFolder(
    allocator: std.mem.Allocator,
    io: std.Io,
    root: std.Io.Dir,
    folder: []const u8,
    needles: []const []const u8,
    unique: bool,
) !?FileSource {
    const dir = root.openDir(io, folder, .{ .iterate = true }) catch return null;
    if (scanFilename(allocator, io, dir, needles, unique)) |name| {
        if (name) |found| return .{ .dir = dir, .dir_owned = true, .file = found };
        dir.close(io);
        return null;
    } else |err| {
        dir.close(io);
        return err;
    }
}

fn scanFilename(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    needles: []const []const u8,
    unique: bool,
) !?[]u8 {
    var it = dir.iterate();
    var found: ?[]u8 = null;
    errdefer if (found) |name| allocator.free(name);
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!checkpoint.safetensorsContains(entry.name, needles)) continue;
        if (found != null) {
            if (unique) return error.AmbiguousWeights;
            continue;
        }
        found = try allocator.dupe(u8, entry.name);
    }
    return found;
}

fn openBundleRoot(io: std.Io, model_path: []const u8) ?std.Io.Dir {
    if (!std.mem.endsWith(u8, model_path, ".safetensors")) return null;
    const parent = std.fs.path.dirname(model_path) orelse return null;
    if (!checkpoint.isBundleLeaf(std.fs.path.basename(parent))) return null;
    const root = std.fs.path.dirname(parent) orelse ".";
    return std.Io.Dir.openDir(.cwd(), io, root, .{}) catch null;
}

pub const tokenizer_relpaths = [_][]const u8{
    "tokenizer/tokenizer.json",
    "processor/tokenizer.json",
    "text_encoder/tokenizer.json",
    "tokenizer.json",
};

pub fn loadTokenizer(
    allocator: std.mem.Allocator,
    io: std.Io,
    task_dir: std.Io.Dir,
    repo: std.Io.Dir,
    model: []const u8,
    progress: *std.Progress.Node,
) !zml.tokenizer.Tokenizer {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Loading tokenizer...", 1);
    defer node.end();

    const bytes = try readTokenizerBytes(allocator, io, task_dir, repo, model);
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
    model: []const u8,
) ![]u8 {
    if (readTokenizerAny(allocator, io, task_dir, repo, model)) |bytes| return bytes else |err| switch (err) {
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
    model_path: []const u8,
) ![]u8 {
    var extra = openBundleRoot(io, model_path);
    defer if (extra) |*dir| dir.close(io);

    const nearby = [_]?std.Io.Dir{ task_dir, repo, extra };
    for (nearby) |maybe| {
        const dir = maybe orelse continue;
        if (readTokenizer(allocator, io, dir)) |bytes| return bytes else |err| switch (err) {
            error.MissingTokenizer => {},
            else => return err,
        }
    }
    for (config.official_task_dirs) |name| {
        var dir = repo.openDir(io, name, .{}) catch continue;
        defer dir.close(io);
        if (readTokenizer(allocator, io, dir)) |bytes| return bytes else |err| switch (err) {
            error.MissingTokenizer => {},
            else => return err,
        }
    }
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
