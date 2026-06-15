const std = @import("std");

const fused_moe = @import("mosaic_tpu_kernels/fused_moe.zig");
const mtt = @import("kernels/mosaic_tpu/builder");

const Arg = [:0]const u8;

fn parseEnum(comptime T: type, value: []const u8) !T {
    inline for (std.meta.fields(T)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    return error.InvalidEnumValue;
}

fn arg(args: []const Arg, index: usize) ![]const u8 {
    if (index >= args.len) return error.MissingArgument;
    return args[index];
}

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var args_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_iter.deinit();

    var args_list: std.ArrayList(Arg) = .empty;
    while (args_iter.next()) |next_arg| try args_list.append(allocator, next_arg);
    const args = args_list.items;
    if (args.len != 20) return error.InvalidArgumentCount;

    const cfg = fused_moe.Cfg{
        .num_tokens = try std.fmt.parseInt(i64, try arg(args, 1), 10),
        .hidden_size = try std.fmt.parseInt(i64, try arg(args, 2), 10),
        .intermediate_size = try std.fmt.parseInt(i64, try arg(args, 3), 10),
        .num_experts = try std.fmt.parseInt(i64, try arg(args, 4), 10),
        .top_k = try std.fmt.parseInt(i64, try arg(args, 5), 10),
        .ep_size = try std.fmt.parseInt(i64, try arg(args, 6), 10),
        .token_dtype = try parseEnum(mtt.DType, try arg(args, 7)),
        .weight_dtype = try parseEnum(mtt.DType, try arg(args, 8)),
        .renormalize_topk_logits = try std.fmt.parseInt(u1, try arg(args, 9), 10) != 0,
        .act_fn = try parseEnum(fused_moe.Cfg.ActFn, try arg(args, 10)),
        .scoring_fn = try parseEnum(fused_moe.Cfg.ScoringFn, try arg(args, 11)),
        .bt = try std.fmt.parseInt(i64, try arg(args, 12), 10),
        .bf = try std.fmt.parseInt(i64, try arg(args, 13), 10),
        .bd1 = try std.fmt.parseInt(i64, try arg(args, 14), 10),
        .bd2 = try std.fmt.parseInt(i64, try arg(args, 15), 10),
        .btc = try std.fmt.parseInt(i64, try arg(args, 16), 10),
        .bfc = try std.fmt.parseInt(i64, try arg(args, 17), 10),
        .bd1c = try std.fmt.parseInt(i64, try arg(args, 18), 10),
        .bd2c = try std.fmt.parseInt(i64, try arg(args, 19), 10),
    };

    const ir = try fused_moe.Kernel.emit(allocator, cfg);
    try std.Io.File.writeStreamingAll(.stdout(), init.io, ir);
}
