const std = @import("std");

const zml = @import("zml");

pub const cache_directory = ".zml-kimi-k3-experts-v1";
pub const cache_schema_version: u32 = 1;
pub const canonical_parts: usize = 4;
pub const experts_per_part: usize = 224;
pub const dma_chunk_size: usize = 16 * zml.MiB;
pub const dma_chunks_per_device: usize = 4;

pub const ManifestFile = struct {
    name: []const u8,
    first_expert: usize,
    end_expert: usize,
    payload_bytes: u64,
    file_size: u64,
    sha256: []const u8,
};

pub const ManifestTensorShape = struct {
    projection: []const u8,
    component: []const u8,
    dtype: []const u8,
    shape: []const usize,
};

pub const Manifest = struct {
    schema_version: u32,
    expert_count: usize,
    canonical_parts: usize,
    experts_per_part: usize,
    first_layer: usize,
    end_layer: usize,
    expert_payload_bytes: u64,
    source_index_sha256: []const u8,
    source_config_sha256: []const u8,
    tensor_shapes: []const ManifestTensorShape,
    files: []const ManifestFile,
};

pub const TransferStats = struct {
    extents: u64 = 0,
    bytes: u64 = 0,
    opens: u64 = 0,
    read_ns: u64 = 0,
    upload_wait_ns: u64 = 0,
    total_ns: u64 = 0,
};

fn fileSha256(
    allocator: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    name: []const u8,
) ![64]u8 {
    const stat = try std.Io.Dir.statFile(dir, io, name, .{});
    const bytes = try allocator.alloc(u8, @intCast(stat.size));
    defer allocator.free(bytes);
    const read = try std.Io.Dir.readFile(dir, io, name, bytes);
    if (read.len != bytes.len) return error.KimiK3CacheFingerprintShortRead;
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(bytes);
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

pub const PackedCache = struct {
    io: std.Io,
    dir: std.Io.Dir,
    parsed_manifest: std.json.Parsed(Manifest),
    registry: zml.safetensors.TensorRegistry,

    pub fn open(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
    ) !?PackedCache {
        const dir = repo.openDir(io, cache_directory, .{}) catch |err| switch (err) {
            error.FileNotFound => return null,
            else => return err,
        };
        errdefer dir.close(io);

        const manifest_file = try dir.openFile(io, "manifest.json", .{ .mode = .read_only });
        defer manifest_file.close(io);
        var manifest_buffer: [4096]u8 = undefined;
        var manifest_reader = manifest_file.reader(io, &manifest_buffer);
        var json_reader: std.json.Reader = .init(allocator, &manifest_reader.interface);
        defer json_reader.deinit();
        const parsed_manifest = try std.json.parseFromTokenSource(
            Manifest,
            allocator,
            &json_reader,
            .{ .ignore_unknown_fields = true },
        );
        errdefer parsed_manifest.deinit();

        const manifest = parsed_manifest.value;
        if (manifest.schema_version != cache_schema_version or
            manifest.expert_count != 896 or
            manifest.canonical_parts != canonical_parts or
            manifest.experts_per_part != experts_per_part or
            manifest.first_layer != 1 or
            manifest.end_layer != 93 or
            manifest.expert_payload_bytes != 1_446_456_066_048)
        {
            return error.InvalidKimiK3PackedExpertManifest;
        }

        if (manifest.files.len != canonical_parts)
            return error.InvalidKimiK3PackedExpertManifest;
        const expected_projections = [6][]const u8{ "w1", "w1", "w2", "w2", "w3", "w3" };
        const expected_components = [6][]const u8{
            "weight_packed",
            "weight_scale",
            "weight_packed",
            "weight_scale",
            "weight_packed",
            "weight_scale",
        };
        const expected_shapes = [6][3]usize{
            .{ experts_per_part, 3072, 1792 },
            .{ experts_per_part, 3072, 112 },
            .{ experts_per_part, 3584, 1536 },
            .{ experts_per_part, 3584, 96 },
            .{ experts_per_part, 3072, 1792 },
            .{ experts_per_part, 3072, 112 },
        };
        if (manifest.tensor_shapes.len != expected_shapes.len)
            return error.InvalidKimiK3PackedExpertManifest;
        for (manifest.tensor_shapes, 0..) |tensor_shape, index| {
            if (!std.mem.eql(u8, tensor_shape.projection, expected_projections[index]) or
                !std.mem.eql(u8, tensor_shape.component, expected_components[index]) or
                !std.mem.eql(u8, tensor_shape.dtype, "U8") or
                !std.mem.eql(usize, tensor_shape.shape, &expected_shapes[index]))
            {
                return error.InvalidKimiK3PackedExpertManifest;
            }
        }
        for (manifest.files, 0..) |file_entry, part| {
            var expected_name_buffer: [64]u8 = undefined;
            const expected_name = try std.fmt.bufPrint(
                &expected_name_buffer,
                "experts-{d:0>5}-of-{d:0>5}.safetensors",
                .{ part, canonical_parts },
            );
            if (!std.mem.eql(u8, file_entry.name, expected_name) or
                file_entry.sha256.len != 64)
            {
                return error.InvalidKimiK3PackedExpertFile;
            }
            for (file_entry.sha256) |character| {
                if (!std.ascii.isDigit(character) and
                    !(character >= 'a' and character <= 'f'))
                {
                    return error.InvalidKimiK3PackedExpertFile;
                }
            }
            const stat = try std.Io.Dir.statFile(dir, io, expected_name, .{});
            if (file_entry.first_expert != part * experts_per_part or
                file_entry.end_expert != (part + 1) * experts_per_part or
                file_entry.payload_bytes != 361_614_016_512 or
                file_entry.file_size != stat.size)
            {
                return error.InvalidKimiK3PackedExpertFile;
            }
        }

        const index_hash = try fileSha256(allocator, io, repo, "model.safetensors.index.json");
        const config_hash = try fileSha256(allocator, io, repo, "config.json");
        if (!std.mem.eql(u8, &index_hash, manifest.source_index_sha256) or
            !std.mem.eql(u8, &config_hash, manifest.source_config_sha256))
        {
            return error.StaleKimiK3PackedExpertCache;
        }

        var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, dir);
        errdefer registry.deinit();
        return .{
            .io = io,
            .dir = dir,
            .parsed_manifest = parsed_manifest,
            .registry = registry,
        };
    }

    pub fn deinit(self: *PackedCache) void {
        self.registry.deinit();
        self.parsed_manifest.deinit();
        self.dir.close(self.io);
    }

    pub fn tensor(
        self: *PackedCache,
        layer_index: usize,
        part: usize,
        projection: []const u8,
        component: []const u8,
    ) !zml.safetensors.Tensor {
        var key_buffer: [160]u8 = undefined;
        const key = try std.fmt.bufPrint(
            &key_buffer,
            "layers.{d}.part.{d}.{s}.{s}",
            .{ layer_index, part, projection, component },
        );
        return self.registry.tensors.get(key) orelse error.MissingKimiK3PackedExpertTensor;
    }
};

pub const Resources = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    dma_loader: zml.io.Loader,
    files: std.StringHashMapUnmanaged(std.Io.File) = .empty,
    files_mutex: std.Io.Mutex = .init,
    packed_cache: ?PackedCache,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        repo: ?std.Io.Dir,
    ) !Resources {
        var dma_loader: zml.io.Loader = try .init(allocator, platform, .{
            .parallelism = 2,
            .dma_chunks = dma_chunks_per_device,
            .dma_chunk_size = dma_chunk_size,
        });
        errdefer dma_loader.deinit();
        const packed_cache = if (repo) |model_repo|
            try PackedCache.open(allocator, io, model_repo)
        else
            null;
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .dma_loader = dma_loader,
            .packed_cache = packed_cache,
        };
    }

    pub fn deinit(self: *Resources) void {
        var iterator = self.files.valueIterator();
        while (iterator.next()) |file| file.close(self.io);
        self.files.deinit(self.allocator);
        if (self.packed_cache) |*cache| cache.deinit();
        self.dma_loader.deinit();
    }

    pub fn usesPackedCache(self: *const Resources) bool {
        return self.packed_cache != null;
    }

    const OpenResult = struct {
        file: std.Io.File,
        opened: bool,
    };

    fn openPersistent(self: *Resources, uri: []const u8) !OpenResult {
        self.files_mutex.lockUncancelable(self.io);
        defer self.files_mutex.unlock(self.io);
        if (self.files.get(uri)) |file| return .{ .file = file, .opened = false };
        const file = try std.Io.Dir.openFile(.cwd(), self.io, uri, .{ .mode = .read_only });
        errdefer file.close(self.io);
        try self.files.put(self.allocator, uri, file);
        return .{ .file = file, .opened = true };
    }

    pub fn streamSources(
        self: *Resources,
        sources: []const zml.safetensors.Tensor,
        target: zml.Shape,
        sharding: zml.Sharding,
        output: *zml.Buffer,
        progress: ?*std.Progress.Node,
    ) !TransferStats {
        var writer = try zml.io.MemoryWriter.init(
            self.allocator,
            self.io,
            self.platform,
            self.dma_loader.pinned_buffer_pools,
            self.dma_loader.dma_allocators,
            self.dma_loader.dma_chunk_size,
            target,
            sharding,
            output,
        );
        errdefer output.deinit();
        defer writer.deinit(self.allocator);

        const total_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        var stats: TransferStats = .{};
        for (sources) |source| {
            const read_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
            const opened = try self.openPersistent(source.file_uri);
            stats.opens += @intFromBool(opened.opened);
            var file_reader = opened.file.reader(self.io, &.{});
            try file_reader.seekTo(source.offset);
            // A single Reader.stream call is allowed to transfer only the
            // current DMA window. Pump the complete tensor before advancing
            // to the next source extent.
            try file_reader.interface.streamExact64(
                writer.interface(),
                source.byteSize(),
            );
            stats.read_ns += @intCast(std.Io.Clock.now(.real, self.io).toNanoseconds() - read_started);
            stats.extents += 1;
            if (progress) |node| node.completeOne();
            stats.bytes += source.byteSize();
        }
        const upload_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        try writer.interface().flush();
        stats.upload_wait_ns = @intCast(std.Io.Clock.now(.real, self.io).toNanoseconds() - upload_started);
        stats.total_ns = @intCast(std.Io.Clock.now(.real, self.io).toNanoseconds() - total_started);
        return stats;
    }

    pub fn readExtent(
        self: *Resources,
        source: zml.safetensors.Tensor,
        bytes: []u8,
    ) !TransferStats {
        const started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        const opened = try self.openPersistent(source.file_uri);
        var reader = opened.file.reader(self.io, &.{});
        try reader.seekTo(source.offset);
        try reader.interface.readSliceAll(bytes);
        const elapsed: u64 = @intCast(std.Io.Clock.now(.real, self.io).toNanoseconds() - started);
        return .{
            .extents = 1,
            .bytes = bytes.len,
            .opens = @intFromBool(opened.opened),
            .read_ns = elapsed,
            .total_ns = elapsed,
        };
    }

    pub fn streamMemorySlices(
        self: *Resources,
        slices: []const []const u8,
        target: zml.Shape,
        sharding: zml.Sharding,
        output: *zml.Buffer,
    ) !TransferStats {
        const total_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        var writer = try zml.io.MemoryWriter.init(
            self.allocator,
            self.io,
            self.platform,
            self.dma_loader.pinned_buffer_pools,
            self.dma_loader.dma_allocators,
            self.dma_loader.dma_chunk_size,
            target,
            sharding,
            output,
        );
        errdefer output.deinit();
        defer writer.deinit(self.allocator);
        for (slices) |slice| try writer.interface().writeAll(slice);
        const upload_started = std.Io.Clock.now(.real, self.io).toNanoseconds();
        try writer.interface().flush();
        return .{
            .upload_wait_ns = @intCast(std.Io.Clock.now(.real, self.io).toNanoseconds() - upload_started),
            .total_ns = @intCast(std.Io.Clock.now(.real, self.io).toNanoseconds() - total_started),
        };
    }

    pub fn streamMemoryOffsets(
        self: *Resources,
        bytes: []const u8,
        offsets: *const [896]usize,
        component_bytes: usize,
        target: zml.Shape,
        sharding: zml.Sharding,
        output: *zml.Buffer,
    ) !TransferStats {
        var slices: [896][]const u8 = undefined;
        for (&slices, offsets) |*slice, offset|
            slice.* = bytes[offset..][0..component_bytes];
        return self.streamMemorySlices(&slices, target, sharding, output);
    }

    pub fn streamMemoryStrided(
        self: *Resources,
        bytes: []const u8,
        first_component_offset: usize,
        expert_stride: usize,
        component_bytes: usize,
        target: zml.Shape,
        sharding: zml.Sharding,
        output: *zml.Buffer,
    ) !TransferStats {
        var slices: [896][]const u8 = undefined;
        for (&slices, 0..) |*slice, expert| {
            const offset = first_component_offset + expert * expert_stride;
            slice.* = bytes[offset..][0..component_bytes];
        }
        return self.streamMemorySlices(&slices, target, sharding, output);
    }

    pub fn packedExpertSources(
        self: *Resources,
        layer_index: usize,
        projection: []const u8,
        component: []const u8,
        sources: *[canonical_parts]zml.safetensors.Tensor,
    ) !bool {
        const cache = &(self.packed_cache orelse return false);
        for (0..canonical_parts) |part| {
            sources[part] = try cache.tensor(layer_index, part, projection, component);
        }
        return true;
    }
};
