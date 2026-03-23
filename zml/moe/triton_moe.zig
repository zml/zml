const std = @import("std");

const zml = @import("../zml.zig");
const DataType = zml.DataType;
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const ops = zml.ops;
const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/moe/general_triton_backend");

pub const GenerationConfig = struct {
    a_dtype: DataType,
    b_dtype: DataType,
    c_dtype: DataType,
    a_scale_dtype: ?DataType = null,
    b_scale_dtype: ?DataType = null,
    b_bias_dtype: ?DataType = null,
    topk_weights_dtype: ?DataType = null,
    num_tokens: usize,
    top_k: usize,
    num_experts: usize,
    out_features: usize,
    in_features: usize,
    max_num_tokens_padded: usize,
    num_valid_tokens: usize,
    block_size_m: usize,
    block_size_n: usize,
    block_size_k: usize,
    group_size_m: usize,
    split_k: usize = 1,
    group_n: usize = 0,
    group_k: usize = 0,
    naive_block_assignment: bool = false,
    mul_routed_weight: bool = false,
    compute_type: DataType = .bf16,
    use_fp8_w8a8: bool = false,
    use_int8_w8a8: bool = false,
    use_int8_w8a16: bool = false,
    per_channel_quant: bool = false,
    has_bias: bool = false,
    num_warps: usize,
    num_stages: usize,
};

const AlignBlockSizeKernel = enum {
    align_block_size,
    count_and_sort,

    fn kernelName(self: @This()) []const u8 {
        return switch (self) {
            .align_block_size => "moe_align_block_size_kernel",
            .count_and_sort => "count_and_sort_expert_tokens_kernel",
        };
    }
};

const AlignBlockSizeGenerationConfig = struct {
    kernel_family: []const u8 = "align_block_size",
    kernel_name: []const u8,
    numel: usize,
    num_experts: usize,
    padded_num_experts: usize,
    max_num_tokens_padded: usize,
    max_num_m_blocks: usize,
    block_size_m: usize,
    experts_per_warp: usize,
    hist_block: usize,
    sort_block_size: usize,
    sort_grid_x: usize,
};

fn getGenerateBinPath(allocator: std.mem.Allocator) ![]const u8 {
    const runfiles = bazel.runfiles(bazel_builtin.current_repository) catch |err| {
        log.err("Failed to initialize runfiles for MoE Triton backend: {}", .{err});
        return err;
    };

    var sandbox_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = runfiles.rlocation("zml/zml/moe/triton/sandbox", &sandbox_path_buf) catch |err| {
        log.err("Failed to resolve MoE Triton generator sandbox in runfiles: {}", .{err});
        return err;
    };

    const sandbox_dir = sandbox_path orelse {
        log.err(
            "MoE Triton generator sandbox is missing from runfiles. Expected `zml/zml/moe/triton/sandbox`; add the generator sandbox to the binary's runtime data.",
            .{},
        );
        return error.MissingTritonGeneratorRunfile;
    };

    log.info("Sandbox path for MoE Triton backend: {s}", .{sandbox_dir});

    const path = std.fs.path.join(allocator, &.{ sandbox_dir, "bin", "generate" }) catch |err| {
        log.err("Failed to construct path to MoE Triton generator binary: {}", .{err});
        return err;
    };

    if (!std.fs.path.isAbsolute(path)) {
        log.err("Constructed MoE Triton generator path is not absolute: {s}", .{path});
        return error.InvalidTritonGeneratorPath;
    }

    return path;
}

fn generateTtir(allocator: std.mem.Allocator, io: std.Io, config: GenerationConfig) ![:0]const u8 {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    var list: std.ArrayList([]const u8) = .empty;
    try list.append(arena.allocator(), try getGenerateBinPath(arena.allocator()));
    try list.append(arena.allocator(), "--config");
    try list.append(arena.allocator(), try std.fmt.allocPrint(arena.allocator(), "{f}", .{std.json.fmt(config, .{ .emit_null_optional_fields = false })}));
    const result = try std.process.run(arena.allocator(), io, .{ .argv = list.items });
    switch (result.term) {
        .exited => |exit_code| {
            if (exit_code != 0) {
                std.log.err("Failed to generate MoE matmul TTIR. Stderr: {s}", .{result.stderr});
            }
        },
        else => {},
    }
    return try allocator.dupeZ(u8, result.stdout);
}

fn generateAlignBlockSizeTtir(
    allocator: std.mem.Allocator,
    io: std.Io,
    kernel: AlignBlockSizeKernel,
    numel: i64,
    num_experts: i64,
    padded_num_experts: i64,
    max_num_tokens_padded: i64,
    max_num_m_blocks: i64,
    block_size_m: i64,
    experts_per_warp: i64,
    hist_block: i64,
    sort_block_size: i64,
    sort_grid_x: i64,
) ![:0]const u8 {
    const config = AlignBlockSizeGenerationConfig{
        .kernel_name = kernel.kernelName(),
        .numel = @intCast(numel),
        .num_experts = @intCast(num_experts),
        .padded_num_experts = @intCast(padded_num_experts),
        .max_num_tokens_padded = @intCast(max_num_tokens_padded),
        .max_num_m_blocks = @intCast(max_num_m_blocks),
        .block_size_m = @intCast(block_size_m),
        .experts_per_warp = @intCast(experts_per_warp),
        .hist_block = @intCast(hist_block),
        .sort_block_size = @intCast(sort_block_size),
        .sort_grid_x = @intCast(sort_grid_x),
    };

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    var list: std.ArrayList([]const u8) = .empty;
    try list.append(arena.allocator(), try getGenerateBinPath(arena.allocator()));
    try list.append(arena.allocator(), "--config");
    try list.append(arena.allocator(), try std.fmt.allocPrint(arena.allocator(), "{f}", .{std.json.fmt(config, .{ .emit_null_optional_fields = false })}));
    const result = try std.process.run(arena.allocator(), io, .{ .argv = list.items });
    switch (result.term) {
        .exited => |exit_code| {
            if (exit_code != 0) {
                std.log.err("Failed to generate align block size TTIR ({s}). Stderr: {s}", .{ kernel.kernelName(), result.stderr });
            }
        },
        else => {},
    }
    return try allocator.dupeZ(u8, result.stdout);
}

pub const Options = struct {
    inplace: bool = false,
    activation: []const u8 = "silu",
    apply_router_weight_on_input: bool = false,
    use_fp8_w8a8: bool = false,
    use_int8_w8a8: bool = false,
    use_int8_w8a16: bool = false,
    use_int4_w4a16: bool = false,
    ocp_mx_scheme: ?[]const u8 = null,
    per_channel_quant: bool = false,
    global_num_experts: i64 = -1,
    expert_map: ?Tensor = null,
    w1_scale: ?Tensor = null,
    w2_scale: ?Tensor = null,
    w1_zp: ?Tensor = null,
    w2_zp: ?Tensor = null,
    a1_scale: ?Tensor = null,
    a2_scale: ?Tensor = null,
    block_shape: ?[]const i64 = null,
    w1_bias: ?Tensor = null,
    w2_bias: ?Tensor = null,
    block_size_m: i64 = 16,
    block_size_n: i64 = 64,
    block_size_k: i64 = 32,
    group_size_m: i64 = 1,
    num_warps: i64 = 8,
    num_stages: i64 = 4,
    dynamic_launch_by_num_tokens: bool = true,
};

fn applyTokenBasedLaunchConfig(opts: Options, num_tokens: i64, num_experts: i64) Options {
    var out = opts;
    if (!opts.dynamic_launch_by_num_tokens) return out;

    // General default policy for NVIDIA bf16/fp16 and fp8 per-tensor.
    // Tile sizes scale with batch size: small batches are more memory-bound,
    // while larger batches benefit from wider M/N tiles and more warps.
    if (num_tokens <= 32) {
        out.block_size_m = 16;
    } else if (num_tokens <= 96) {
        out.block_size_m = 32;
    } else if (num_tokens <= 512) {
        out.block_size_m = 64;
    } else {
        out.block_size_m = 128;
    }

    out.block_size_n = if (num_tokens <= 64) 64 else 128;
    out.block_size_k = if (opts.use_fp8_w8a8 or num_tokens <= 64) 128 else 64;

    const tokens_per_expert = @divFloor(num_tokens, @max(num_experts, 1));
    out.group_size_m = if (tokens_per_expert > 128) 16 else 1;
    out.num_warps = if (num_tokens <= 128) 4 else 8;
    out.num_stages = if (num_tokens <= 32) 4 else 3;

    return out;
}

fn getLaunchConfigJsonPath(allocator: std.mem.Allocator) ![]const u8 {
    const runfiles = bazel.runfiles(bazel_builtin.current_repository) catch |err| {
        log.err("Failed to initialize runfiles for MoE launch config: {}", .{err});
        return err;
    };

    var config_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const config_path = runfiles.rlocation("zml/zml/moe/triton/triton_kernels/config.json", &config_path_buf) catch |err| {
        log.err("Failed to resolve MoE launch config in runfiles: {}", .{err});
        return err;
    };

    const config_json = config_path orelse {
        log.err("MoE launch config is missing from runfiles: zml/zml/moe/triton/triton_kernels/config.json", .{});
        return error.MissingLaunchConfigRunfile;
    };

    return try allocator.dupe(u8, config_json);
}

fn jsonValueAsI64(v: std.json.Value) !i64 {
    return switch (v) {
        .integer => |x| x,
        .float => |x| @intFromFloat(x),
        else => error.InvalidLaunchConfigValue,
    };
}

fn applyTokenBasedLaunchConfigFromJson(opts: Options, num_tokens: i64) !Options {
    var out = opts;
    if (!opts.dynamic_launch_by_num_tokens) return out;

    var threaded_io: std.Io.Threaded = .init_single_threaded;
    threaded_io.allocator = std.heap.c_allocator;
    defer threaded_io.deinit();
    const io = threaded_io.io();

    var arena: std.heap.ArenaAllocator = .init(std.heap.c_allocator);
    defer arena.deinit();

    const allocator = arena.allocator();
    const config_path = try getLaunchConfigJsonPath(allocator);
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.openFile(io, config_path, .{ .mode = .read_only });
    defer file.close(io);
    var reader = file.reader(io, &.{});
    const config_json = try reader.interface.readAlloc(allocator, try file.length(io));
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, config_json, .{});

    if (parsed.value != .object) return error.InvalidLaunchConfigShape;

    var it = parsed.value.object.iterator();
    var best_m: ?i64 = null;
    var best_diff: u64 = std.math.maxInt(u64);
    var best_config: ?std.json.Value = null;

    while (it.next()) |entry| {
        const m = std.fmt.parseInt(i64, entry.key_ptr.*, 10) catch continue;
        if (entry.value_ptr.* != .object) continue;

        const diff: u64 = if (m >= num_tokens)
            @intCast(m - num_tokens)
        else
            @intCast(num_tokens - m);

        if (best_m == null or diff < best_diff or (diff == best_diff and m < best_m.?)) {
            best_m = m;
            best_diff = diff;
            best_config = entry.value_ptr.*;
        }
    }

    if (best_config == null) return error.NoMatchingLaunchConfig;

    const cfg = best_config.?.object;
    out.block_size_m = try jsonValueAsI64(cfg.get("BLOCK_SIZE_M") orelse return error.MissingLaunchConfigField);
    out.block_size_n = try jsonValueAsI64(cfg.get("BLOCK_SIZE_N") orelse return error.MissingLaunchConfigField);
    out.block_size_k = try jsonValueAsI64(cfg.get("BLOCK_SIZE_K") orelse return error.MissingLaunchConfigField);
    out.group_size_m = try jsonValueAsI64(cfg.get("GROUP_SIZE_M") orelse return error.MissingLaunchConfigField);
    out.num_warps = try jsonValueAsI64(cfg.get("num_warps") orelse return error.MissingLaunchConfigField);
    out.num_stages = try jsonValueAsI64(cfg.get("num_stages") orelse return error.MissingLaunchConfigField);

    log.info(
        "Loaded MoE launch config from JSON: requested M={}, selected M={}, block_m={}, block_n={}, block_k={}, group_m={}, warps={}, stages={}",
        .{ num_tokens, best_m.?, out.block_size_m, out.block_size_n, out.block_size_k, out.group_size_m, out.num_warps, out.num_stages },
    );

    return out;
}

fn i64s(v: i64) Tensor {
    return Tensor.constant(.{ .i64 = v }).reshape(.{1});
}

fn ceilDiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

fn validateOptions(opts: Options) !void {
    if (opts.inplace) return error.Unimplemented;
    if (!std.mem.eql(u8, opts.activation, "silu")) return error.UnsupportedActivation;
    if (opts.apply_router_weight_on_input) return error.UnsupportedOption;
    if (opts.use_fp8_w8a8 or opts.use_int8_w8a8 or opts.use_int8_w8a16 or opts.use_int4_w4a16) return error.UnsupportedQuantization;
    if (opts.ocp_mx_scheme != null or opts.per_channel_quant) return error.UnsupportedOption;
    if (opts.global_num_experts != -1) return error.UnsupportedOption;
    if (opts.expert_map != null) return error.UnsupportedOption;
    if (opts.w1_scale != null or opts.w2_scale != null or opts.w1_zp != null or opts.w2_zp != null) return error.UnsupportedOption;
    if (opts.a1_scale != null or opts.a2_scale != null or opts.block_shape != null) return error.UnsupportedOption;
    if (opts.w1_bias != null or opts.w2_bias != null) return error.UnsupportedOption;
}

fn alignBlockSizeZml(topk_ids: Tensor, num_experts: i64, block_size_m: i64) struct { Tensor, Tensor, Tensor } {
    const topk_ids_ = topk_ids.withTags(.{ .token, .topk }).convert(.i32);
    const num_tokens = topk_ids_.dim(.token);
    const topk = topk_ids_.dim(.topk);
    const num_assignments = num_tokens * topk;
    const max_num_tokens_padded = if (num_assignments < num_experts)
        num_assignments * block_size_m
    else
        num_assignments + num_experts * (block_size_m - 1);
    const max_num_m_blocks = ceilDiv(max_num_tokens_padded, block_size_m);

    const flat_experts = topk_ids_.reshape(.{ .g = num_assignments });
    const flat_indices = Tensor.arange(.{ .end = num_assignments }, .i32).withTags(.{.g});
    const ones = Tensor.constant(.{ .i32 = 1 }).broad(flat_experts.shape());

    const counts0 = Tensor.zeroes(Shape.init(.{ .expert = num_experts }, .i32));
    // For each expert count how many tokens are assigned to it
    const counts = counts0.scatterSlices(
        .{ .expert = flat_experts.rename(.{ .g = .n }) },
        ones.rename(.{ .g = .n }),
        .{
            .update_fn = Tensor.ScatterOpts.increment,
            .indices_are_unique = false,
        },
    );

    // We pad to the upper block size modulo for each expert
    const padded_counts = counts.addConstant(block_size_m - 1)
        .divByConst(block_size_m)
        .mul(Tensor.scalar(block_size_m, .i32).broad(counts.shape()));
    const zero_i32 = Tensor.zeroes(Shape.init(.{ .expert = 1 }, .i32));
    // offsets of each expert in the flat assignment, and offsets with padding
    const offsets = Tensor.concatenate(&.{ zero_i32, counts.cumulativeSum(.expert) }, 0);
    const padded_offsets = Tensor.concatenate(&.{ zero_i32, padded_counts.cumulativeSum(.expert) }, 0);
    // Retrieve the last offset to get the number of tokens after padding
    const num_tokens_post_padded = padded_offsets.slice1d(.expert, .{ .start = num_experts, .end = num_experts + 1 }).reshape(.{1});

    const sort_perm = flat_experts.argsort(.g, .{}).rename(.{ .g = .n });
    const sorted_experts = flat_experts.gather(.{ .g = sort_perm }, .{});
    const sorted_flat_indices = flat_indices.gather(.{ .g = sort_perm }, .{});
    const sorted_positions = Tensor.arange(.{ .end = num_assignments }, .i32).withTags(.{.g});
    const group_start = offsets.slice1d(.expert, .{ .start = 0, .end = num_experts }).gather(.{ .expert = sorted_experts }, .{}).rename(.{ .n = .g });
    const padded_group_start = padded_offsets.slice1d(.expert, .{ .start = 0, .end = num_experts }).gather(.{ .expert = sorted_experts }, .{}).rename(.{ .n = .g });
    const rank_in_group = sorted_positions.sub(group_start);
    const dest_positions = padded_group_start.add(rank_in_group);

    var sorted_token_ids = Tensor.constant(.{ .i32 = @as(i32, @intCast(num_assignments)) }).broad(Shape.init(.{ .g = max_num_tokens_padded }, .i32));
    sorted_token_ids = sorted_token_ids.scatterSlices(
        .{ .g = dest_positions.rename(.{ .g = .n }) },
        sorted_flat_indices,
        .{ .indices_are_unique = true },
    );

    //create indices of the blocks
    const block_ids = Tensor.arange(.{ .end = max_num_m_blocks }, .i32).withTags(.{.block});
    // For each (block, expert) pair, check whether the block falls within the expert's padded range.
    // broad() uses tag-based axis matching: {block} -> {block, n} and {n} -> {block, n}.
    const joint_shape = Shape.init(.{ .block = max_num_m_blocks, .n = num_experts }, .i32);
    const starts_blocks = padded_offsets.slice1d(.expert, .{ .start = 0, .end = num_experts }).reshape(.{ .expert = num_experts }).divByConst(block_size_m);
    const ends_blocks = padded_offsets.slice1d(.expert, .{ .start = 1, .end = num_experts + 1 }).reshape(.{ .expert = num_experts }).divByConst(block_size_m);
    const ge_starts = block_ids.broad(joint_shape).cmp(.GE, starts_blocks.rename(.{ .expert = .n }).broad(joint_shape));
    const lt_ends = block_ids.broad(joint_shape).cmp(.LT, ends_blocks.rename(.{ .expert = .n }).broad(joint_shape));
    // in_range[block, n] = true iff block belongs to expert n
    const in_range = ge_starts.logical(.AND, lt_ends);
    // expert_fill[block, n] = n (the expert index), or -1 where not in range
    const expert_fill = Tensor.arange(.{ .end = num_experts }, .i32).withTags(.{.n}).broad(joint_shape);
    // Since padded ranges are non-overlapping, at most one expert covers each block.
    // max(.n) returns that expert's id, or -1 if no expert covers the block.
    const expert_ids = Tensor.select(in_range, expert_fill, Tensor.constant(.{ .i32 = -1 }).broad(joint_shape)).max(.n);
    return .{ sorted_token_ids, expert_ids, num_tokens_post_padded };
}

// Here the padding is made so that each token is aligned "in front of" its assigned experts and an expert process a contiguous block of tokens (based on block size m)
fn alignBlockSize(allocator: std.mem.Allocator, io: std.Io, topk_ids: Tensor, num_experts: i64, block_size_m: i64) !struct { Tensor, Tensor, Tensor } {
    const topk_ids_ = topk_ids.withTags(.{ .token, .topk }).convert(.i32);
    const num_tokens = topk_ids_.dim(.token);
    const topk = topk_ids_.dim(.topk);
    const num_assignments = num_tokens * topk;
    const max_num_tokens_padded = if (num_assignments < num_experts)
        num_assignments * block_size_m
    else
        num_assignments + num_experts * (block_size_m - 1);
    const max_num_m_blocks = ceilDiv(max_num_tokens_padded, block_size_m);
    const warp_size: i64 = 32;
    const padded_num_experts = ceilDiv(num_experts, warp_size) * warp_size;
    const experts_per_warp: i64 = warp_size;
    const hist_block: i64 = 256;
    const sort_block_size: i64 = 256;
    const sort_grid_x: i64 = @min(ceilDiv(num_assignments, sort_block_size), 65535);

    log.info(
        "alignBlockSize launch: num_assignments={} num_experts={} padded_num_experts={} block_size_m={} max_num_tokens_padded={} max_num_m_blocks={} sort_grid_x={}",
        .{
            num_assignments,
            num_experts,
            padded_num_experts,
            block_size_m,
            max_num_tokens_padded,
            max_num_m_blocks,
            sort_grid_x,
        },
    );

    const ttir_align = try generateAlignBlockSizeTtir(
        allocator,
        io,
        .align_block_size,
        num_assignments,
        num_experts,
        padded_num_experts,
        max_num_tokens_padded,
        max_num_m_blocks,
        block_size_m,
        experts_per_warp,
        hist_block,
        sort_block_size,
        sort_grid_x,
    );
    defer allocator.free(ttir_align);
    const ttir_count_sort = try generateAlignBlockSizeTtir(
        allocator,
        io,
        .count_and_sort,
        num_assignments,
        num_experts,
        padded_num_experts,
        max_num_tokens_padded,
        max_num_m_blocks,
        block_size_m,
        experts_per_warp,
        hist_block,
        sort_block_size,
        sort_grid_x,
    );
    defer allocator.free(ttir_count_sort);

    const flat_experts = topk_ids_.reshape(.{ .g = num_assignments });
    var cumsums = Tensor.zeroes(Shape.init(.{ .g = num_experts + 1 }, .i32));
    var expert_ids = Tensor.zeroes(Shape.init(.{ .g = max_num_m_blocks }, .i32));
    var sorted_token_ids = Tensor.zeroes(Shape.init(.{ .g = max_num_tokens_padded }, .i32));
    var num_tokens_post_padded = Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32));

    {
        const inputs = .{
            flat_experts,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            cumsums,
        };
        const outputs = ops.triton(inputs, .{ sorted_token_ids.shape(), expert_ids.shape(), num_tokens_post_padded.shape(), cumsums.shape() }, .{
            .name = "moe_align_block_size_kernel",
            .ir = ttir_align,
            .grid = .{ 2, 1, 1 },
            .num_stages = 1,
            .num_warps = 8,
            .output_operand_aliases = &.{ 1, 2, 3, 4 },
        });
        sorted_token_ids = outputs[0];
        expert_ids = outputs[1];
        num_tokens_post_padded = outputs[2];
        cumsums = outputs[3];
    }

    {
        const inputs = .{
            sorted_token_ids,
            flat_experts,
            cumsums,
        };
        const outputs = ops.triton(inputs, .{ sorted_token_ids.shape(), cumsums.shape() }, .{
            .name = "count_and_sort_expert_tokens_kernel",
            .ir = ttir_count_sort,
            .grid = .{ @intCast(sort_grid_x), 1, 1 },
            .num_stages = 1,
            .num_warps = 4,
            .output_operand_aliases = &.{ 0, 2 },
        });
        sorted_token_ids = outputs[0];
        cumsums = outputs[1];
    }

    return .{ sorted_token_ids, expert_ids, num_tokens_post_padded };
}

fn makeGenerationConfig(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    max_num_tokens_padded: i64,
    num_valid_tokens: i64,
    opts: Options,
    naive_block_assignment: bool,
) GenerationConfig {
    return .{
        .a_dtype = a.dtype(),
        .b_dtype = b.dtype(),
        .c_dtype = c.dtype(),
        .num_tokens = @intCast(a.dim(0)),
        .top_k = @intCast(c.dim(1)),
        .num_experts = @intCast(b.dim(0)),
        .out_features = @intCast(b.dim(1)),
        .in_features = @intCast(b.dim(2)),
        .max_num_tokens_padded = @intCast(max_num_tokens_padded),
        .num_valid_tokens = @intCast(num_valid_tokens),
        .block_size_m = @intCast(opts.block_size_m),
        .block_size_n = @intCast(opts.block_size_n),
        .block_size_k = @intCast(opts.block_size_k),
        .group_size_m = @intCast(opts.group_size_m),
        .naive_block_assignment = naive_block_assignment,
        .compute_type = c.dtype(),
        .num_warps = @intCast(opts.num_warps),
        .num_stages = @intCast(opts.num_stages),
    };
}

fn callFusedKernel(
    a: Tensor,
    b: Tensor,
    c: Tensor,
    b_bias: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    topk_weights: Tensor,
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_padded: Tensor,
    ttir: [:0]const u8,
    config: GenerationConfig,
    max_num_tokens_padded: i64,
    num_valid_tokens: i64,
) Tensor {
    const block_size_m: i64 = @intCast(config.block_size_m);
    const block_size_n: i64 = @intCast(config.block_size_n);
    const m_tokens = a.dim(0);
    const em_effective = if (m_tokens < block_size_m)
        @min(max_num_tokens_padded, num_valid_tokens * block_size_m)
    else
        max_num_tokens_padded;
    const grid_x = ceilDiv(em_effective, block_size_m) * ceilDiv(b.dim(1), block_size_n);
    log.info(
        "moe grid:  m_tokens={} topk={} num_experts={} num_valid_tokens={} max_num_tokens_padded={} em_effective={} block_m={} block_n={} bN={} grid_x={}",
        .{
            m_tokens,
            @divExact(num_valid_tokens, m_tokens),
            b.dim(0),
            num_valid_tokens,
            max_num_tokens_padded,
            em_effective,
            block_size_m,
            block_size_n,
            b.dim(1),
            grid_x,
        },
    );
    const inputs = .{
        a,
        b,
        c,
        b_bias,
        a_scale,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        i64s(b.dim(1)),
        i64s(b.dim(2)),
        i64s(max_num_tokens_padded),
        i64s(num_valid_tokens),
        i64s(a.dim(1)),
        i64s(b.dim(1) * b.dim(2)),
        i64s(b.dim(2)),
        i64s(c.dim(2)),
        i64s(0),
        i64s(0),
        i64s(0),
        i64s(0),
        i64s(0),
        i64s(0),
        i64s(0),
    };
    const outputs = ops.triton(inputs, .{c.shape()}, .{
        .name = "fused_moe_kernel",
        .ir = ttir,
        .grid = .{
            @intCast(grid_x),
            1,
            1,
        },
        .num_stages = @intCast(config.num_stages),
        .num_warps = @intCast(config.num_warps),
        .output_operand_aliases = &.{2},
    });
    return outputs[0];
}

// Call the fused moe kernels and reduce across top k experts
pub fn fusedExpertsImpl(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    opts: Options,
) !Tensor {
    try validateOptions(opts);
    const effective_opts = applyTokenBasedLaunchConfigFromJson(opts, hidden_states.dim(0)) catch |err| fallback: {
        log.warn("Failed to load MoE launch config from JSON ({}), falling back to built-in token heuristic", .{err});
        break :fallback applyTokenBasedLaunchConfig(opts, hidden_states.dim(0), w1.dim(0));
    };

    const hidden = hidden_states.withTags(.{ .token, .in });
    const gate_up = w1.withTags(.{ .expert, .out, .in });
    const down = w2.withTags(.{ .expert, .out, .mid });
    const weights = topk_weights.withTags(.{ .token, .topk });
    const ids = topk_ids.withTags(.{ .token, .topk });

    if (hidden.dtype() != .bf16) return error.UnsupportedType;
    if (gate_up.dtype() != .bf16 or down.dtype() != .bf16) return error.UnsupportedType;
    if (weights.dtype() != .f32 and weights.dtype() != .bf16) return error.UnsupportedType;
    if (ids.dtype() != .i32) return error.UnsupportedType;
    if (hidden.dim(.in) != gate_up.dim(.in)) return error.InvalidShape;
    if (@rem(gate_up.dim(.out), 2) != 0) return error.InvalidShape;
    if (down.dim(.mid) != @divFloor(gate_up.dim(.out), 2)) return error.InvalidShape;
    if (ids.dim(.token) != hidden.dim(.token) or weights.dim(.token) != hidden.dim(.token)) return error.InvalidShape;
    if (ids.dim(.topk) != weights.dim(.topk)) return error.InvalidShape;
    if (gate_up.dim(.expert) != down.dim(.expert)) return error.InvalidShape;

    const block_size_m = effective_opts.block_size_m;
    const num_experts = gate_up.dim(.expert);
    const num_assignments = hidden.dim(.token) * ids.dim(.topk);
    const sparsity_factor: i64 = 4;
    const naive_block_assignment = num_assignments * sparsity_factor <= num_experts;

    const max_num_tokens_padded = if (naive_block_assignment)
        num_assignments * block_size_m
    else if (num_assignments < num_experts)
        num_assignments * block_size_m
    else
        num_assignments + num_experts * (block_size_m - 1);

    log.info("moe naive path selected={}", .{naive_block_assignment});

    var threaded_io: std.Io.Threaded = .init_single_threaded;
    threaded_io.allocator = std.heap.c_allocator;
    defer threaded_io.deinit();

    const io = threaded_io.io();

    const sorted_token_ids, const expert_ids, const num_tokens_post_padded = if (naive_block_assignment) blk: {
        // In the naive path each M-block corresponds to exactly one (token, topk) assignment.
        const naive_sorted_ids = Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32));
        const naive_expert_ids = ids.reshape(.{ .g = num_assignments }).convert(.i32);
        const naive_num_tokens_post_padded = Tensor.constant(.{ .i32 = @as(i32, @intCast(max_num_tokens_padded)) }).reshape(.{1});
        break :blk .{ naive_sorted_ids, naive_expert_ids, naive_num_tokens_post_padded };
    } else try alignBlockSize(std.heap.c_allocator, io, ids, num_experts, block_size_m);

    var first_out = Tensor.zeroes(Shape.init(.{ .token = hidden.dim(.token), .topk = ids.dim(.topk), .out = gate_up.dim(.out) }, .bf16));
    const first_generation_config = makeGenerationConfig(hidden, gate_up, first_out, max_num_tokens_padded, num_assignments, effective_opts, naive_block_assignment);

    const ttir_first_matmul = generateTtir(std.heap.c_allocator, io, first_generation_config) catch |err| {
        log.err("Failed to generate TTIR for first MoE matmul: {}", .{err});
        return err;
    };

    defer std.heap.c_allocator.free(ttir_first_matmul);

    var b_bias = opts.w1_bias orelse Tensor.zeroes(Shape.init(.{ .expert = gate_up.dim(.expert), .out = gate_up.dim(.out) }, gate_up.dtype()));
    var a_scale = opts.a1_scale orelse Tensor.scalar(1.0, .f32);
    var b_scale = opts.w1_scale orelse Tensor.scalar(1.0, .f32);

    first_out = callFusedKernel(
        hidden,
        gate_up,
        first_out,
        b_bias,
        a_scale,
        b_scale,
        weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        ttir_first_matmul,
        first_generation_config,
        max_num_tokens_padded,
        num_assignments,
    );

    const first_flat = first_out.reshape(.{ .g = num_assignments, .out = gate_up.dim(.out) });
    const gate = first_flat.slice1d(.out, .{ .start = 0, .end = @divExact(gate_up.dim(.out), 2) });
    const up = first_flat.slice1d(.out, .{ .start = @divExact(gate_up.dim(.out), 2), .end = gate_up.dim(.out) });
    const activated = gate.silu().mul(up);

    var second_out = Tensor.zeroes(Shape.init(.{ .token = hidden.dim(.token), .topk = ids.dim(.topk), .out = down.dim(.out) }, .bf16));
    const second_generation_config = makeGenerationConfig(activated, down, second_out, max_num_tokens_padded, num_assignments, effective_opts, naive_block_assignment);
    const ttir_second_matmul = generateTtir(std.heap.c_allocator, io, second_generation_config) catch |err| {
        log.err("Failed to generate TTIR for second MoE matmul: {}", .{err});
        return err;
    };
    defer std.heap.c_allocator.free(ttir_second_matmul);

    b_bias = opts.w1_bias orelse Tensor.zeroes(Shape.init(.{ .expert = gate_up.dim(.expert), .out = gate_up.dim(.out) }, gate_up.dtype()));
    a_scale = opts.a1_scale orelse Tensor.scalar(1.0, .f32);
    b_scale = opts.w1_scale orelse Tensor.scalar(1.0, .f32);

    second_out = callFusedKernel(
        activated,
        down,
        second_out,
        b_bias,
        a_scale,
        b_scale,
        weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        ttir_second_matmul,
        second_generation_config,
        max_num_tokens_padded,
        num_assignments,
    );

    const weighted = second_out.mul(weights.convert(second_out.dtype()).broad(second_out.shape()));
    return weighted.sum(.topk).squeeze(.topk);
}
