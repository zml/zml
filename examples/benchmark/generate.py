#!/usr/bin/env python3
"""Generate main.zig, including literal hexadecimal pointer expectations.

Run: python3 examples/benchmark/generate.py
Check reproducibility: python3 examples/benchmark/generate.py --check
Requires the repository's `zig 0.16.0 fmt` formatter (override with --zig).

The oracle uses row-major rectangular slices, not ZML's Placement/AxisSplit.
For each tensor axis, the start is logical_coordinate * dimension / shard_count.
The byte offset is the dot product of those starts and row-major byte strides.
"""

import argparse
from dataclasses import dataclass
from itertools import product
from math import prod
from pathlib import Path
import shlex
import subprocess


PHYSICAL_AXES = ('link_x', 'link_y', 'link_z', 'bus')
TENSOR_AXES = ('m', 'n', 'e', 'p')


@dataclass(frozen=True)
class Layout:
    name: str
    target: str
    sizes: tuple[int, ...]
    reverse_ids: bool = False


LAYOUTS = (
    Layout('cpu-1', 'cpu', (1,)),
    Layout('cuda-8', 'cuda', (8,)),
    Layout('cuda-2x2', 'cuda', (2, 2)),
    Layout('rocm-2x2x2', 'rocm', (2, 2, 2)),
    Layout('neuron-2x2x2x2', 'neuron', (2, 2, 2, 2)),
    Layout('cuda-4x2x2x2-reversed', 'cuda', (4, 2, 2, 2), True),
)


@dataclass(frozen=True)
class LogicalMesh:
    axes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Strategy:
    name: str
    bindings: tuple[tuple[str, tuple[str, ...]], ...] = ()
    mode: str = 'explicit'


MODEL = LogicalMesh((('model', 'high_bandwidth'),))
TWO = LogicalMesh(MODEL.axes + (('data', 'low_bandwidth'),))
THREE = LogicalMesh(TWO.axes + (('experts', 'balanced'),))
FOUR = LogicalMesh(THREE.axes + (('pipeline', 'low_bandwidth'),))

# Each pair is tried on every physical layout that supplies its required axes.
MESH_STRATEGIES = (
    (LogicalMesh(()), Strategy('replicated', mode='replicated')),
    (MODEL, Strategy('partial-model', (('model', ('link_x',)),))),
    (MODEL, Strategy('all-axes', mode='all')),
    (MODEL, Strategy('folded-model', mode='fold')),
    (TWO, Strategy('auto-model-data', mode='suggest')),
    (TWO, Strategy('reordered-2d', (('model', ('link_y',)), ('data', ('link_x',))))),
    (THREE, Strategy('logical-3d', (('model', ('link_x',)), ('data', ('link_y',)), ('experts', ('link_z',))))),
    (FOUR, Strategy('logical-4d', (('model', ('link_x',)), ('data', ('link_y',)), ('experts', ('link_z',)), ('pipeline', ('bus',))))),
)

# Explicit expectations for Strategy.suggest; do not duplicate its heuristic.
SUGGESTED_BINDINGS = {
    'cpu-1': (('model', ('link_x',)), ('data', ('link_x',))),
    'cuda-8': (('model', ('link_x',)), ('data', ('link_x',))),
    'cuda-2x2': (('model', ('link_x',)), ('data', ('link_y',))),
    'rocm-2x2x2': (('model', ('link_x',)), ('data', ('link_z',))),
    'neuron-2x2x2x2': (('model', ('link_x',)), ('data', ('link_z',))),
    'cuda-4x2x2x2-reversed': (('model', ('link_x',)), ('data', ('bus',))),
}

# Include a dimension divisible by 32 to exercise all-axis sharding on 32 devices.
SHAPES = ((64,), (24, 56), (16, 24, 40), (16, 24, 40, 56), (32, 24, 40, 56))


def shape_partitions(logical):
    """Each mesh gets replicated, partial, and rotated full-axis assignments."""
    logical_axes = tuple(name for name, _ in logical.axes)
    for dims in SHAPES:
        if not logical_axes:
            yield dims, (None,) * len(dims)
            continue
        assignments = [(None,) * len(dims)]
        for axis in range(len(dims)):
            partial = [None] * len(dims)
            partial[axis] = logical_axes[0]
            assignments.append(tuple(partial))
        if len(logical_axes) <= len(dims):
            for rotation in range(len(dims)):
                full = [None] * len(dims)
                for i, axis in enumerate(logical_axes):
                    full[(i + rotation) % len(dims)] = axis
                assignments.append(tuple(full))
        for partition in dict.fromkeys(assignments):
            yield dims, partition


def bindings_for(layout, strategy):
    available = PHYSICAL_AXES[:len(layout.sizes)]
    if strategy.mode in ('all', 'fold'):
        return {'model': available}
    if strategy.mode == 'suggest':
        return dict(SUGGESTED_BINDINGS[layout.name])
    if any(axis not in available for _, axes in strategy.bindings for axis in axes):
        return None
    return dict(strategy.bindings)


def expected_offsets(layout, bindings, dims, partition):
    """Return None for indivisible slices; otherwise enumerate slice origins."""
    physical_sizes = dict(zip(PHYSICAL_AXES, layout.sizes))
    consumed = set()
    splits = []
    for dim, logical_axis in zip(dims, partition):
        # Shared physical axes partition the first tensor axis using them;
        # later uses replicate along those axes (e.g. 1D suggested meshes).
        axes = tuple(a for a in bindings.get(logical_axis, ()) if a not in consumed)
        consumed.update(axes)
        count = prod(physical_sizes[a] for a in axes)
        if dim % count:
            return None
        splits.append((axes, dim // count))
    offsets = []
    for coordinates in product(*(range(size) for size in layout.sizes)):
        physical = dict(zip(PHYSICAL_AXES, coordinates))
        starts = []
        for axes, width in splits:
            coordinate = 0
            for axis in axes:
                coordinate = coordinate * physical_sizes[axis] + physical[axis]
            starts.append(coordinate * width)
        # Flatten the tensor element coordinate independently of sharding plans.
        element = 0
        for dim, start in zip(dims, starts):
            element = element * dim + start
        offsets.append(4 * element)
    assert len(offsets) == prod(layout.sizes)
    assert all(0 <= offset < 4 * prod(dims) for offset in offsets)
    return offsets


def zig_struct(fields):
    return '.{ ' + ', '.join(f'.{name} = {value}' for name, value in fields) + ' }'


def strategy_code(logical, strategy, layout):
    if strategy.mode == 'replicated':
        return []
    logical_code = '.mesh(' + zig_struct((name, '.' + intent) for name, intent in logical.axes) + ')'
    lines = [f'const logical: Sharding.LogicalMesh = {logical_code};']
    if strategy.mode == 'suggest':
        lines.append('const strategy: Sharding.Strategy = .suggest(logical, &platform.physical_mesh);')
    else:
        lines.append('var strategy: Sharding.Strategy = .init;')
        bindings = bindings_for(layout, strategy)
        if strategy.mode == 'fold':
            lines.append('strategy.addBinding(.model, .link_x);')
            tags = ', '.join('.' + a for a in PHYSICAL_AXES[:len(layout.sizes)])
            lines.append(f'strategy.addFold(.link_x, &.{{ {tags} }});')
        else:
            for axis, tags in bindings.items():
                for tag in tags:
                    lines.append(f'strategy.addBinding(.{axis}, .{tag});')
    lines += ['const mesh = try allocator.create(zml.Mesh);',
              f'mesh.* = try .init("{strategy.name}", &platform.physical_mesh, logical, strategy);']
    return lines


def generate():
    catalog = ['const layouts = [_]Layout{']
    for layout in LAYOUTS:
        sizes = ', '.join(map(str, layout.sizes))
        catalog.append(f'.{{ .name = "{layout.name}", .target = .{layout.target}, .sizes = &.{{ {sizes} }}, .reverse_ids = {str(layout.reverse_ids).lower()} }},')
    catalog += ['};', f'const maxDataBytes = {max(4 * prod(d) for d in SHAPES)};']
    code = ['fn meshesFor(allocator: std.mem.Allocator, platform: *const zml.Platform, layout: []const u8) ![]const MeshCases {',
            'var meshes: std.ArrayList(MeshCases) = .empty;']
    counts = {}
    skipped = 0
    for layout in LAYOUTS:
        code.append(f'if (std.mem.eql(u8, layout, "{layout.name}")) {{')
        count = 0
        for logical, strategy in MESH_STRATEGIES:
            bindings = bindings_for(layout, strategy)
            if bindings is None:
                continue
            code += ['{'] + strategy_code(logical, strategy, layout)
            code.append(f'try meshes.append(allocator, .{{ .name = "{strategy.name}", .cases = try allocator.dupe(Case, &.{{')
            for dims, partition in shape_partitions(logical):
                offsets = expected_offsets(layout, bindings, dims, partition)
                if offsets is None:
                    skipped += 1
                    continue
                count += 1
                shape = zig_struct(zip(TENSOR_AXES, map(str, dims)))
                part = zig_struct((tag, '.' + axis) for tag, axis in zip(TENSOR_AXES, partition) if axis)
                name = 'x'.join(map(str, dims)) + '/' + '-'.join(axis or 'rep' for axis in partition)
                literal = ', '.join(hex(offset) for offset in offsets)
                shape_code = f'zml.Shape.init({shape}, .f32)'
                if strategy.mode != 'replicated':
                    shape_code += f'.withPartitioning(mesh, {part})'
                code.append(f'.{{ .name = "{name}", .shape = {shape_code}, .offsets = &.{{ {literal} }} }},')
            code += ['}) });', '}']
        code += ['return meshes.items;', '}']
        counts[layout.name] = count
    code += ['return error.UnknownLayout;', '}']
    return RUNTIME.replace('@CATALOG@', '\n'.join(catalog)).replace('@CASES@', '\n'.join(code)), counts, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--zig', default='zig 0.16.0', help='Zig formatter command')
    args = parser.parse_args()
    source, counts, skipped = generate()
    formatted = subprocess.run(shlex.split(args.zig) + ['fmt', '--stdin'], input=source, text=True, capture_output=True, check=True).stdout
    destination = Path(__file__).with_name('main.zig')
    if args.check:
        if destination.read_text() != formatted:
            raise SystemExit('main.zig is out of date; run generate.py')
    else:
        destination.write_text(formatted)
    print(f'{sum(counts.values())} cases; {skipped} indivisible shape/partition combinations excluded')
    for layout, count in counts.items():
        print(f'  {layout}: {count}')


# The generated benchmark remains self-contained in main.zig.
RUNTIME = r'''
//! Generated by generate.py; edit the generator and regenerate.
//! Run with: bazel run --config=release //examples/benchmark
const std = @import("std");
const builtin = @import("builtin");
const zml = @import("zml");
const Sharding = zml.Sharding;
const pjrt = zml.pjrt;

comptime {
    _ = zml.bazel;
}

const CliArgs = struct {
    pub const help = "benchmark --iterations=200 --samples=8 --warmup=100";
    iterations: usize = 200,
    samples: usize = 8,
    warmup: usize = 100,
};

@CATALOG@

pub fn main(init: std.process.Init) !void {
    const args = zml.stdx.flags.parse(init.minimal.args, CliArgs);
    if (args.iterations == 0 or args.samples == 0) return error.InvalidBenchmarkCount;
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const allocator = arena.allocator();

    var buffers: std.ArrayList(zml.Buffer) = .empty;
    try buffers.ensureTotalCapacity(init.gpa, @max(args.iterations, args.warmup));
    defer buffers.deinit(init.gpa);

    const data = try allocator.alloc(u8, maxDataBytes);
    @memset(data, 0);
    std.log.info("Buffer.from metadata only; mode={s}; iterations={d}; samples={d}; warmup={d}", .{ @tagName(builtin.mode), args.iterations, args.samples, args.warmup });
    std.log.info("Synthetic platforms; no PJRT, transfers, events or waits. Setup and pointer checks excluded. Compiler barriers and buffer storage included.", .{});
    const iterations: u64 = @intCast(args.iterations);
    const calls_per_case = iterations * @as(u64, @intCast(args.samples));
    var results: [layouts.len]Result = undefined;
    for (layouts, &results) |layout, *result| {
        const platform = try syntheticPlatform(allocator, layout);
        const meshes = try meshesFor(allocator, platform, layout.name);
        var total: u64 = 0;
        var count: u64 = 0;
        for (meshes) |mesh| {
            for (mesh.cases) |case| {
                const caseData = data[0..case.shape.byteSize()];
                try batch(platform, case.shape, caseData, args.warmup, &buffers);
                try checkBuffers(buffers.items, caseData, layout.name, mesh.name, case);
                buffers.clearRetainingCapacity();
                for (0..args.samples) |_| {
                    const start: std.Io.Timestamp = .now(init.io, .awake);
                    try batch(platform, case.shape, caseData, args.iterations, &buffers);
                    const ns: u64 = @intCast(start.untilNow(init.io, .awake).toNanoseconds());
                    try checkBuffers(buffers.items, caseData, layout.name, mesh.name, case);
                    buffers.clearRetainingCapacity();
                    total += ns;
                }
                count += 1;
            }
        }
        const calls = count * calls_per_case;
        result.* = .{
            .devices = @intCast(platform.devices.len),
            .cases = count,
            .calls = calls,
            .elapsedNs = total,
        };
    }
    summarize(&results);
}

const Result = struct {
    devices: u64,
    cases: u64,
    calls: u64,
    elapsedNs: u64,
};

fn summarize(results: *const [layouts.len]Result) void {
    std.log.info("layout,devices,cases,avg_ns_per_call,avg_ns_per_device", .{});
    var meanDevices: f64 = 0;
    var meanNs: f64 = 0;
    for (layouts, results) |layout, result| {
        const devices: f64 = @floatFromInt(result.devices);
        const nsPerCall = @as(f64, @floatFromInt(result.elapsedNs)) / @as(f64, @floatFromInt(result.calls));
        std.log.info("{s},{d},{d},{d},{d}", .{ layout.name, result.devices, result.cases, result.elapsedNs / result.calls, result.elapsedNs / result.calls / result.devices });
        meanDevices += devices;
        meanNs += nsPerCall;
    }
    meanDevices /= @as(f64, @floatFromInt(results.len));
    meanNs /= @as(f64, @floatFromInt(results.len));

    // Ordinary least squares, with equal weight for each physical layout.
    var covariance: f64 = 0;
    var variance: f64 = 0;
    for (results) |result| {
        const nsPerCall = @as(f64, @floatFromInt(result.elapsedNs)) / @as(f64, @floatFromInt(result.calls));
        const dx = @as(f64, @floatFromInt(result.devices)) - meanDevices;
        covariance += dx * (nsPerCall - meanNs);
        variance += dx * dx;
    }
    if (variance == 0) {
        std.log.info("Linear fit unavailable: need distinct device counts", .{});
        return;
    }
    const perDevice = covariance / variance;
    const base = meanNs - perDevice * meanDevices;
    std.log.info("Linear fit (metadata only, equal weight per layout): Buffer.from = {d:.0}ns + {d:.0}ns * num_devices", .{ @floor(base), @floor(perDevice) });
}

// Mirrors Buffer.from and emptyShell. Keep this in sync with zml/buffer.zig.
// Synthetic handles are never dereferenced or passed to PJRT, including deinit.
// Compiler barriers stand in for the external call consuming each upload's args.
noinline fn bufferFromMetadata(platform: *const zml.Platform, shape: zml.Shape, data: []const u8, opts: zml.Buffer.FromOptions) !zml.Buffer {
    var buffer: zml.Buffer = .{ ._platform = platform, ._shape = shape, ._shards = .empty };
    const packed_shape = shape.packedShape();
    var placed_shape = packed_shape;
    if (placed_shape._sharding.mesh == null) placed_shape._sharding.mesh = platform.replicated_mesh;
    const placement = try Sharding.Placement.init(placed_shape);
    const ty = zml.pjrtx.bufferTypeFromDtype(packed_shape.dtype());
    const layout = platform.defaultMemoryLayout(placement.shape.dims(), packed_shape.dtype());
    zml.stdx.debug.assert(platform.devices[0].memory(opts.memory) != null, "Device doesn't have {} memory", .{opts.memory});
    const slice = zml.Slice.init(packed_shape, data);

    for (platform.physical_mesh.devices_in_canonical_order) |device| {
        const memory = platform.devices[device.id].memory(opts.memory).?;
        const args: pjrt.Client.BufferFromHostBufferArgs = .{
            .data = placement.shardPtr(device.coords, slice),
            .dst = .{ .memory = memory.pjrt_memory },
            .layout = layout,
            .dims = placement.shape.dims(),
            .buffer_type = ty,
            .byte_strides = slice.byte_strides.constSlice(),
            .host_buffer_semantics = .ImmutableUntilTransferCompletes,
        };
        std.mem.doNotOptimizeAway(&args);
        // Smuggle the "data" pointer passed to BufferFromHostBufferArgs into the result Buffer object,
        // to allow checking the correctness of the operation `checkBuffers`
        buffer._shards.appendAssumeCapacity(@ptrCast(@constCast(args.data)));
    }
    return buffer;
}

const Layout = struct {
    name: []const u8,
    target: zml.Target,
    sizes: []const usize,
    reverse_ids: bool = false,
};

const physical_tags = [_]Sharding.PhysicalAxisTag{ .link_x, .link_y, .link_z, .bus };

fn physicalTree(allocator: std.mem.Allocator, sizes: []const usize, depth: usize, next_id: *u32) !Sharding.PhysicalNode {
    if (depth == sizes.len) {
        defer next_id.* += 1;
        // Let PhysicalMesh.fromTree assign coordinates from the tree path.
        return .{ .leaf = .{ .id = next_id.*, .coords = @splat(0xff) } };
    }
    const children = try allocator.alloc(Sharding.PhysicalNode, sizes[depth]);
    for (children) |*child| child.* = try physicalTree(allocator, sizes, depth + 1, next_id);
    return .axis(physical_tags[depth], .{ .mesh = .torus }, children);
}

fn syntheticPlatform(allocator: std.mem.Allocator, layout: Layout) !*zml.Platform {
    const platform = try allocator.create(zml.Platform);
    var next_id: u32 = 0;
    const root = try physicalTree(allocator, layout.sizes, 0, &next_id);
    // Only metadata fields are initialized. Runtime/client fields must never be used.
    platform.* = undefined;
    platform.target = layout.target;
    platform.physical_mesh = try .fromTree(allocator, layout.target, root);
    if (layout.reverse_ids) {
        for (platform.physical_mesh.devices_in_canonical_order) |*device| device.id = next_id - 1 - device.id;
    }
    const devices = try allocator.alloc(zml.platform.Device, next_id);
    const memories = try allocator.alloc(zml.Memory, next_id);
    for (devices, memories, 0..) |*device, *memory, i| {
        memory.* = .{
            .pjrt_memory = @ptrFromInt((i + 1) * 4096),
            .platform = platform,
            .addressable_by_devices = &.{},
        };
        device.* = undefined;
        device.platform = platform;
        device.memory_by_kind = .initFill(memory);
    }
    platform.devices = devices;
    platform.memories = memories;
    const replicated = try allocator.create(zml.Mesh);
    replicated.* = try .init("replicated", &platform.physical_mesh, .mesh(.{ .all = .high_bandwidth }), .parseBindings(.{ .all = .link_x }));
    platform.replicated_mesh = replicated;
    return platform;
}

const Case = struct {
    name: []const u8,
    shape: zml.Shape,
    offsets: []const usize,
};

const MeshCases = struct {
    name: []const u8,
    cases: []const Case,
};

@CASES@

fn batch(platform: *const zml.Platform, shape: zml.Shape, data: []const u8, iterations: usize, out: *std.ArrayList(zml.Buffer)) !void {
    for (0..iterations) |_| {
        const buffer = try bufferFromMetadata(platform, shape, data, .{ .wait = false });
        out.appendAssumeCapacity(buffer);
    }
}

pub fn checkBuffers(buffers: []const zml.Buffer, data: []const u8, layout: []const u8, mesh: []const u8, case: Case) !void {
    const offsets = case.offsets;
    // These shards smuggle host pointers; never call PJRT methods or Buffer.deinit.
    for (buffers) |buffer| {
        if (buffer.numShards() != offsets.len or buffer.numShards() != buffer._platform.devices.len) return error.InvalidShardCount;
        for (buffer._shards.constSlice(), offsets, 0..) |shard, offset, deviceIndex| {
            if (offset >= data.len) return error.InvalidExpectedOffset;
            const actual = @intFromPtr(shard);
            const want = @intFromPtr(data.ptr + offset);
            if (actual != want) {
                std.log.err("{s}/{s}/{s}: canonical device {d}, expected data + 0x{x} ({x}), got {x}", .{ layout, mesh, case.name, deviceIndex, offset, want, actual });
                return error.InvalidShardPointer;
            }
        }
    }
}
'''


if __name__ == '__main__':
    main()
