const std = @import("std");

const artifact = @import("artifact.zig");

const Allocator = std.mem.Allocator;

const CliArgs = struct {
    source: []u8,
    source_map: []u8,
    stablehlo: []u8,
    hlo: []u8,
    output: []u8,

    fn deinit(self: *CliArgs, allocator: Allocator) void {
        allocator.free(self.source);
        allocator.free(self.source_map);
        allocator.free(self.stablehlo);
        allocator.free(self.hlo);
        allocator.free(self.output);
        self.* = undefined;
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args = parseArgs(allocator, init.minimal.args) catch |err| {
        std.log.err(
            "invalid arguments ({s}); expected --source PATH --source-map PATH --stablehlo PATH --hlo PATH --output PATH",
            .{@errorName(err)},
        );
        return err;
    };
    defer args.deinit(allocator);

    const source = try readFile(allocator, io, args.source);
    defer allocator.free(source);
    const source_map = try readFile(allocator, io, args.source_map);
    defer allocator.free(source_map);
    const stablehlo = try readFile(allocator, io, args.stablehlo);
    defer allocator.free(stablehlo);
    const hlo = try readFile(allocator, io, args.hlo);
    defer allocator.free(hlo);

    try artifact.writeBundleFromTexts(
        allocator,
        io,
        args.output,
        source,
        source_map,
        stablehlo,
        hlo,
    );
    std.log.info("Explorer bundle written to {s}", .{args.output});
}

fn readFile(allocator: Allocator, io: std.Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
}

fn parseArgs(allocator: Allocator, process_args: std.process.Args) !CliArgs {
    var iterator = try std.process.Args.Iterator.initAllocator(process_args, allocator);
    defer iterator.deinit();
    _ = iterator.next();

    var source: ?[]u8 = null;
    errdefer if (source) |value| allocator.free(value);
    var source_map: ?[]u8 = null;
    errdefer if (source_map) |value| allocator.free(value);
    var stablehlo: ?[]u8 = null;
    errdefer if (stablehlo) |value| allocator.free(value);
    var hlo: ?[]u8 = null;
    errdefer if (hlo) |value| allocator.free(value);
    var output: ?[]u8 = null;
    errdefer if (output) |value| allocator.free(value);

    while (iterator.next()) |argument_z| {
        const argument: []const u8 = argument_z;
        if (try parseArgument(allocator, &iterator, argument, "--source", &source)) continue;
        if (try parseArgument(allocator, &iterator, argument, "--source-map", &source_map)) continue;
        if (try parseArgument(allocator, &iterator, argument, "--stablehlo", &stablehlo)) continue;
        if (try parseArgument(allocator, &iterator, argument, "--hlo", &hlo)) continue;
        if (try parseArgument(allocator, &iterator, argument, "--output", &output)) continue;
        return error.UnknownArgument;
    }

    return .{
        .source = source orelse return error.MissingSource,
        .source_map = source_map orelse return error.MissingSourceMap,
        .stablehlo = stablehlo orelse return error.MissingStableHlo,
        .hlo = hlo orelse return error.MissingHlo,
        .output = output orelse return error.MissingOutput,
    };
}

fn parseArgument(
    allocator: Allocator,
    iterator: *std.process.Args.Iterator,
    argument: []const u8,
    name: []const u8,
    destination: *?[]u8,
) !bool {
    if (std.mem.eql(u8, argument, name)) {
        if (destination.* != null) return error.DuplicateArgument;
        const value = iterator.next() orelse return error.MissingArgumentValue;
        if (value.len == 0) return error.MissingArgumentValue;
        destination.* = try allocator.dupe(u8, value);
        return true;
    }
    if (std.mem.startsWith(u8, argument, name) and argument.len > name.len and argument[name.len] == '=') {
        if (destination.* != null) return error.DuplicateArgument;
        const value = argument[name.len + 1 ..];
        if (value.len == 0) return error.MissingArgumentValue;
        destination.* = try allocator.dupe(u8, value);
        return true;
    }
    return false;
}
