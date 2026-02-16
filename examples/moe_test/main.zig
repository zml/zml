const std = @import("std");

const zml = @import("zml");
const ops = zml.ops;

const log = std.log.scoped(.moe_test);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const ttir_path: []const u8 =
    "/home/louislechevalier/zml/moe_mxfp4_p_tma_tensorized_prefill_gate_up_test.ttir";

const NUM_SMS: i32 = 170;
const NUM_WARPS: i32 = 8;
const NUM_STAGES: i32 = 3;

const KernelInputs = struct {
    Y__base: zml.Tensor,
    Y__shape_0: zml.Tensor,
    Y__shape_1: zml.Tensor,
    Y__shape_2: zml.Tensor,
    Y__shape_3: zml.Tensor,
    Y__shape_4: zml.Tensor,
    Y__stride_0: zml.Tensor,
    Y__stride_1: zml.Tensor,
    Y__stride_2: zml.Tensor,
    Y__stride_3: zml.Tensor,
    Y__stride_4: zml.Tensor,
    YPtr: zml.Tensor,
    stride_y_k: zml.Tensor,
    stride_y_z: zml.Tensor,
    stride_y_m: zml.Tensor,
    stride_y_n: zml.Tensor,

    X__base: zml.Tensor,
    X__shape_0: zml.Tensor,
    X__shape_1: zml.Tensor,
    X__shape_2: zml.Tensor,
    X__shape_3: zml.Tensor,
    X__shape_4: zml.Tensor,
    X__stride_0: zml.Tensor,
    X__stride_1: zml.Tensor,
    X__stride_2: zml.Tensor,
    X__stride_3: zml.Tensor,
    X__stride_4: zml.Tensor,
    XPtr: zml.Tensor,
    stride_x_z: zml.Tensor,
    stride_x_m: zml.Tensor,
    stride_x_k: zml.Tensor,

    W__base: zml.Tensor,
    W__shape_0: zml.Tensor,
    W__shape_1: zml.Tensor,
    W__shape_2: zml.Tensor,
    W__stride_0: zml.Tensor,
    W__stride_1: zml.Tensor,
    W__stride_2: zml.Tensor,
    WPtr: zml.Tensor,
    stride_w_e: zml.Tensor,
    stride_w_k: zml.Tensor,
    stride_w_n: zml.Tensor,

    WMxScale__base: zml.Tensor,
    WMxScale__shape_0: zml.Tensor,
    WMxScale__shape_1: zml.Tensor,
    WMxScale__shape_2: zml.Tensor,
    WMxScale__stride_0: zml.Tensor,
    WMxScale__stride_1: zml.Tensor,
    WMxScale__stride_2: zml.Tensor,
    stride_w_mx_e: zml.Tensor,
    stride_w_mx_k: zml.Tensor,
    stride_w_mx_n: zml.Tensor,

    B: zml.Tensor,
    stride_b_e: zml.Tensor,

    M: zml.Tensor,
    N: zml.Tensor,
    K: zml.Tensor,
    K_W: zml.Tensor,

    XSliceSizes: zml.Tensor,
    XSliceOffs: zml.Tensor,
    XBlockOffs: zml.Tensor,
    XBlockSchedule: zml.Tensor,

    batch_size: zml.Tensor,
    grid_m: zml.Tensor,
    grid_n: zml.Tensor,
    out_alpha: zml.Tensor,
    reduce_rank: zml.Tensor,

    YOut: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) KernelInputs {
        return .{
            .Y__base = store.createTensor("Y__base"),
            .Y__shape_0 = store.createTensor("Y__shape_0"),
            .Y__shape_1 = store.createTensor("Y__shape_1"),
            .Y__shape_2 = store.createTensor("Y__shape_2"),
            .Y__shape_3 = store.createTensor("Y__shape_3"),
            .Y__shape_4 = store.createTensor("Y__shape_4"),
            .Y__stride_0 = store.createTensor("Y__stride_0"),
            .Y__stride_1 = store.createTensor("Y__stride_1"),
            .Y__stride_2 = store.createTensor("Y__stride_2"),
            .Y__stride_3 = store.createTensor("Y__stride_3"),
            .Y__stride_4 = store.createTensor("Y__stride_4"),
            .YPtr = store.createTensor("YPtr"),
            .stride_y_k = store.createTensor("stride_y_k"),
            .stride_y_z = store.createTensor("stride_y_z"),
            .stride_y_m = store.createTensor("stride_y_m"),
            .stride_y_n = store.createTensor("stride_y_n"),

            .X__base = store.createTensor("X__base"),
            .X__shape_0 = store.createTensor("X__shape_0"),
            .X__shape_1 = store.createTensor("X__shape_1"),
            .X__shape_2 = store.createTensor("X__shape_2"),
            .X__shape_3 = store.createTensor("X__shape_3"),
            .X__shape_4 = store.createTensor("X__shape_4"),
            .X__stride_0 = store.createTensor("X__stride_0"),
            .X__stride_1 = store.createTensor("X__stride_1"),
            .X__stride_2 = store.createTensor("X__stride_2"),
            .X__stride_3 = store.createTensor("X__stride_3"),
            .X__stride_4 = store.createTensor("X__stride_4"),
            .XPtr = store.createTensor("XPtr"),
            .stride_x_z = store.createTensor("stride_x_z"),
            .stride_x_m = store.createTensor("stride_x_m"),
            .stride_x_k = store.createTensor("stride_x_k"),

            .W__base = store.createTensor("W__base"),
            .W__shape_0 = store.createTensor("W__shape_0"),
            .W__shape_1 = store.createTensor("W__shape_1"),
            .W__shape_2 = store.createTensor("W__shape_2"),
            .W__stride_0 = store.createTensor("W__stride_0"),
            .W__stride_1 = store.createTensor("W__stride_1"),
            .W__stride_2 = store.createTensor("W__stride_2"),
            .WPtr = store.createTensor("WPtr"),
            .stride_w_e = store.createTensor("stride_w_e"),
            .stride_w_k = store.createTensor("stride_w_k"),
            .stride_w_n = store.createTensor("stride_w_n"),

            .WMxScale__base = store.createTensor("WMxScale__base"),
            .WMxScale__shape_0 = store.createTensor("WMxScale__shape_0"),
            .WMxScale__shape_1 = store.createTensor("WMxScale__shape_1"),
            .WMxScale__shape_2 = store.createTensor("WMxScale__shape_2"),
            .WMxScale__stride_0 = store.createTensor("WMxScale__stride_0"),
            .WMxScale__stride_1 = store.createTensor("WMxScale__stride_1"),
            .WMxScale__stride_2 = store.createTensor("WMxScale__stride_2"),
            .stride_w_mx_e = store.createTensor("stride_w_mx_e"),
            .stride_w_mx_k = store.createTensor("stride_w_mx_k"),
            .stride_w_mx_n = store.createTensor("stride_w_mx_n"),

            .B = store.createTensor("B"),
            .stride_b_e = store.createTensor("stride_b_e"),

            .M = store.createTensor("M"),
            .N = store.createTensor("N"),
            .K = store.createTensor("K"),
            .K_W = store.createTensor("K_W"),

            .XSliceSizes = store.createTensor("XSliceSizes"),
            .XSliceOffs = store.createTensor("XSliceOffs"),
            .XBlockOffs = store.createTensor("XBlockOffs"),
            .XBlockSchedule = store.createTensor("XBlockSchedule"),

            .batch_size = store.createTensor("batch_size"),
            .grid_m = store.createTensor("grid_m"),
            .grid_n = store.createTensor("grid_n"),
            .out_alpha = store.createTensor("out_alpha"),
            .reduce_rank = store.createTensor("reduce_rank"),

            .YOut = store.createTensor("YOut"),
        };
    }
};

fn loadTtir(path: []const u8) [:0]const u8 {
    var threaded = std.Io.Threaded.init(std.heap.page_allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var buf: [1024 * 1024]u8 = undefined;
    const cwd = std.Io.Dir.cwd();
    var file = cwd.openFile(io, path, .{}) catch |err| {
        std.debug.panic("failed to open TTIR file: {s}", .{@errorName(err)});
    };
    defer file.close(io);

    var reader = file.reader(io, &buf);
    const len = file.length(io) catch |err| {
        std.debug.panic("failed to get TTIR file length: {s}", .{@errorName(err)});
    };
    const ir_buf = reader.interface.readAlloc(zml.module.CompilationContext.current().allocator, len) catch |err| {
        std.debug.panic("failed to read TTIR file: {s}", .{@errorName(err)});
    };
    return zml.module.CompilationContext.current().allocator.dupeZ(u8, ir_buf) catch |err| {
        std.debug.panic("failed to null-terminate TTIR: {s}", .{@errorName(err)});
    };
}

fn runKernel(inputs: KernelInputs) zml.Tensor {
    const ir_z = loadTtir(ttir_path);

    inputs.Y__base.print("Y__base");
    inputs.Y__shape_0.print("Y__shape_0");
    inputs.Y__shape_1.print("Y__shape_1");
    inputs.Y__shape_2.print("Y__shape_2");
    inputs.Y__shape_3.print("Y__shape_3");
    inputs.Y__shape_4.print("Y__shape_4");
    inputs.Y__stride_0.print("Y__stride_0");
    inputs.Y__stride_1.print("Y__stride_1");
    inputs.Y__stride_2.print("Y__stride_2");
    inputs.Y__stride_3.print("Y__stride_3");
    inputs.Y__stride_4.print("Y__stride_4");
    inputs.YPtr.print("YPtr");
    inputs.stride_y_k.print("stride_y_k");
    inputs.stride_y_z.print("stride_y_z");
    inputs.stride_y_m.print("stride_y_m");
    inputs.stride_y_n.print("stride_y_n");

    inputs.X__base.print("X__base");
    inputs.X__shape_0.print("X__shape_0");
    inputs.X__shape_1.print("X__shape_1");
    inputs.X__shape_2.print("X__shape_2");
    inputs.X__shape_3.print("X__shape_3");
    inputs.X__shape_4.print("X__shape_4");
    inputs.X__stride_0.print("X__stride_0");
    inputs.X__stride_1.print("X__stride_1");
    inputs.X__stride_2.print("X__stride_2");
    inputs.X__stride_3.print("X__stride_3");
    inputs.X__stride_4.print("X__stride_4");
    inputs.XPtr.print("XPtr");
    inputs.stride_x_z.print("stride_x_z");
    inputs.stride_x_m.print("stride_x_m");
    inputs.stride_x_k.print("stride_x_k");

    inputs.W__base.print("W__base");
    inputs.W__shape_0.print("W__shape_0");
    inputs.W__shape_1.print("W__shape_1");
    inputs.W__shape_2.print("W__shape_2");
    inputs.W__stride_0.print("W__stride_0");
    inputs.W__stride_1.print("W__stride_1");
    inputs.W__stride_2.print("W__stride_2");
    inputs.WPtr.print("WPtr");
    inputs.stride_w_e.print("stride_w_e");
    inputs.stride_w_k.print("stride_w_k");
    inputs.stride_w_n.print("stride_w_n");

    inputs.WMxScale__base.print("WMxScale__base");
    inputs.WMxScale__shape_0.print("WMxScale__shape_0");
    inputs.WMxScale__shape_1.print("WMxScale__shape_1");
    inputs.WMxScale__shape_2.print("WMxScale__shape_2");
    inputs.WMxScale__stride_0.print("WMxScale__stride_0");
    inputs.WMxScale__stride_1.print("WMxScale__stride_1");
    inputs.WMxScale__stride_2.print("WMxScale__stride_2");
    inputs.stride_w_mx_e.print("stride_w_mx_e");
    inputs.stride_w_mx_k.print("stride_w_mx_k");
    inputs.stride_w_mx_n.print("stride_w_mx_n");

    inputs.B.print("B");
    inputs.stride_b_e.print("stride_b_e");

    inputs.M.print("M");
    inputs.N.print("N");
    inputs.K.print("K");
    inputs.K_W.print("K_W");

    inputs.XSliceSizes.print("XSliceSizes");
    inputs.XSliceOffs.print("XSliceOffs");
    inputs.XBlockOffs.print("XBlockOffs");
    inputs.XBlockSchedule.print("XBlockSchedule");

    inputs.batch_size.print("batch_size");
    inputs.grid_m.print("grid_m");
    inputs.grid_n.print("grid_n");
    inputs.out_alpha.print("out_alpha");
    inputs.reduce_rank.print("reduce_rank");

    inputs.YOut.print("YOut");

    std.log.info("Y__base {f}", .{inputs.Y__base.shape()});
    std.log.info("Y__shape_0 {f}", .{inputs.Y__shape_0.shape()});
    std.log.info("Y__shape_1 {f}", .{inputs.Y__shape_1.shape()});
    std.log.info("Y__shape_2 {f}", .{inputs.Y__shape_2.shape()});
    std.log.info("Y__shape_3 {f}", .{inputs.Y__shape_3.shape()});
    std.log.info("Y__shape_4 {f}", .{inputs.Y__shape_4.shape()});
    std.log.info("Y__stride_0 {f}", .{inputs.Y__stride_0.shape()});
    std.log.info("Y__stride_1 {f}", .{inputs.Y__stride_1.shape()});
    std.log.info("Y__stride_2 {f}", .{inputs.Y__stride_2.shape()});
    std.log.info("Y__stride_3 {f}", .{inputs.Y__stride_3.shape()});
    std.log.info("Y__stride_4 {f}", .{inputs.Y__stride_4.shape()});
    std.log.info("YPtr {f}", .{inputs.YPtr.shape()});
    std.log.info("stride_y_k {f}", .{inputs.stride_y_k.shape()});
    std.log.info("stride_y_z {f}", .{inputs.stride_y_z.shape()});
    std.log.info("stride_y_m {f}", .{inputs.stride_y_m.shape()});
    std.log.info("stride_y_n {f}", .{inputs.stride_y_n.shape()});

    std.log.info("X__base {f}", .{inputs.X__base.shape()});
    std.log.info("X__shape_0 {f}", .{inputs.X__shape_0.shape()});
    std.log.info("X__shape_1 {f}", .{inputs.X__shape_1.shape()});
    std.log.info("X__shape_2 {f}", .{inputs.X__shape_2.shape()});
    std.log.info("X__shape_3 {f}", .{inputs.X__shape_3.shape()});
    std.log.info("X__shape_4 {f}", .{inputs.X__shape_4.shape()});
    std.log.info("X__stride_0 {f}", .{inputs.X__stride_0.shape()});
    std.log.info("X__stride_1 {f}", .{inputs.X__stride_1.shape()});
    std.log.info("X__stride_2 {f}", .{inputs.X__stride_2.shape()});
    std.log.info("X__stride_3 {f}", .{inputs.X__stride_3.shape()});
    std.log.info("X__stride_4 {f}", .{inputs.X__stride_4.shape()});
    std.log.info("XPtr {f}", .{inputs.XPtr.shape()});
    std.log.info("stride_x_z {f}", .{inputs.stride_x_z.shape()});
    std.log.info("stride_x_m {f}", .{inputs.stride_x_m.shape()});
    std.log.info("stride_x_k {f}", .{inputs.stride_x_k.shape()});

    std.log.info("W__base {f}", .{inputs.W__base.shape()});
    std.log.info("W__shape_0 {f}", .{inputs.W__shape_0.shape()});
    std.log.info("W__shape_1 {f}", .{inputs.W__shape_1.shape()});
    std.log.info("W__shape_2 {f}", .{inputs.W__shape_2.shape()});
    std.log.info("W__stride_0 {f}", .{inputs.W__stride_0.shape()});
    std.log.info("W__stride_1 {f}", .{inputs.W__stride_1.shape()});
    std.log.info("W__stride_2 {f}", .{inputs.W__stride_2.shape()});
    std.log.info("WPtr {f}", .{inputs.WPtr.shape()});
    std.log.info("stride_w_e {f}", .{inputs.stride_w_e.shape()});
    std.log.info("stride_w_k {f}", .{inputs.stride_w_k.shape()});
    std.log.info("stride_w_n {f}", .{inputs.stride_w_n.shape()});

    std.log.info("WMxScale__base {f}", .{inputs.WMxScale__base.shape()});
    std.log.info("WMxScale__shape_0 {f}", .{inputs.WMxScale__shape_0.shape()});
    std.log.info("WMxScale__shape_1 {f}", .{inputs.WMxScale__shape_1.shape()});
    std.log.info("WMxScale__shape_2 {f}", .{inputs.WMxScale__shape_2.shape()});
    std.log.info("WMxScale__stride_0 {f}", .{inputs.WMxScale__stride_0.shape()});
    std.log.info("WMxScale__stride_1 {f}", .{inputs.WMxScale__stride_1.shape()});
    std.log.info("WMxScale__stride_2 {f}", .{inputs.WMxScale__stride_2.shape()});
    std.log.info("stride_w_mx_e {f}", .{inputs.stride_w_mx_e.shape()});
    std.log.info("stride_w_mx_k {f}", .{inputs.stride_w_mx_k.shape()});
    std.log.info("stride_w_mx_n {f}", .{inputs.stride_w_mx_n.shape()});

    std.log.info("B {f}", .{inputs.B.shape()});
    std.log.info("stride_b_e {f}", .{inputs.stride_b_e.shape()});

    std.log.info("M {f}", .{inputs.M.shape()});
    std.log.info("N {f}", .{inputs.N.shape()});
    std.log.info("K {f}", .{inputs.K.shape()});
    std.log.info("K_W {f}", .{inputs.K_W.shape()});

    std.log.info("XSliceSizes {f}", .{inputs.XSliceSizes.shape()});
    std.log.info("XSliceOffs {f}", .{inputs.XSliceOffs.shape()});
    std.log.info("XBlockOffs {f}", .{inputs.XBlockOffs.shape()});
    std.log.info("XBlockSchedule {f}", .{inputs.XBlockSchedule.shape()});

    std.log.info("batch_size {f}", .{inputs.batch_size.shape()});
    std.log.info("grid_m {f}", .{inputs.grid_m.shape()});
    std.log.info("grid_n {f}", .{inputs.grid_n.shape()});
    std.log.info("reduce_rank {f}", .{inputs.reduce_rank.shape()});

    std.log.info("YOut {f}", .{inputs.YOut.shape()});
    std.log.info("ALPHA {f}", .{inputs.out_alpha.shape()});

    const kernel_args = .{
        inputs.Y__base,
        inputs.Y__shape_0,
        inputs.Y__shape_1,
        inputs.Y__shape_2,
        inputs.Y__shape_3,
        inputs.Y__shape_4,
        inputs.Y__stride_0,
        inputs.Y__stride_1,
        inputs.Y__stride_2,
        inputs.Y__stride_3,
        inputs.Y__stride_4,
        inputs.YPtr,
        inputs.stride_y_k,
        inputs.stride_y_z,
        inputs.stride_y_m,
        inputs.stride_y_n,

        inputs.X__base,
        inputs.X__shape_0,
        inputs.X__shape_1,
        inputs.X__shape_2,
        inputs.X__shape_3,
        inputs.X__shape_4,
        inputs.X__stride_0,
        inputs.X__stride_1,
        inputs.X__stride_2,
        inputs.X__stride_3,
        inputs.X__stride_4,
        inputs.XPtr,
        inputs.stride_x_z,
        inputs.stride_x_m,
        inputs.stride_x_k,

        inputs.W__base,
        inputs.W__shape_0,
        inputs.W__shape_1,
        inputs.W__shape_2,
        inputs.W__stride_0,
        inputs.W__stride_1,
        inputs.W__stride_2,
        inputs.WPtr,
        inputs.stride_w_e,
        inputs.stride_w_k,
        inputs.stride_w_n,

        inputs.WMxScale__base,
        inputs.WMxScale__shape_0,
        inputs.WMxScale__shape_1,
        inputs.WMxScale__shape_2,
        inputs.WMxScale__stride_0,
        inputs.WMxScale__stride_1,
        inputs.WMxScale__stride_2,
        inputs.stride_w_mx_e,
        inputs.stride_w_mx_k,
        inputs.stride_w_mx_n,

        inputs.B,
        inputs.stride_b_e,

        inputs.M,
        inputs.N,
        inputs.K,
        inputs.K_W,

        inputs.XSliceSizes,
        inputs.XSliceOffs,
        inputs.XBlockOffs,
        inputs.XBlockSchedule,

        inputs.batch_size,
        inputs.grid_m,
        inputs.grid_n,
        inputs.out_alpha,
        inputs.reduce_rank,
        inputs.YOut,
    };

    const ops_ = ops.TritonOps{
        .name = "_p_matmul__tensorized_wrapper",
        .ir = ir_z,
        .grid = .{ NUM_SMS, 1, 1 },
        .num_warps = NUM_WARPS,
        .num_stages = NUM_STAGES,
        .debug = true,
        .output_operand_aliases = &.{kernel_args.len - 1},
    };

    const res = ops.triton(kernel_args, .{inputs.YOut.shape()}, ops_);
    return res[0];
}

fn deinitKernelInputsBuffers(buffers: *zml.mem.Bufferized(KernelInputs)) void {
    buffers.Y__base.deinit();
    buffers.Y__shape_0.deinit();
    buffers.Y__shape_1.deinit();
    buffers.Y__shape_2.deinit();
    buffers.Y__shape_3.deinit();
    buffers.Y__shape_4.deinit();
    buffers.Y__stride_0.deinit();
    buffers.Y__stride_1.deinit();
    buffers.Y__stride_2.deinit();
    buffers.Y__stride_3.deinit();
    buffers.Y__stride_4.deinit();
    buffers.YPtr.deinit();
    buffers.stride_y_k.deinit();
    buffers.stride_y_z.deinit();
    buffers.stride_y_m.deinit();
    buffers.stride_y_n.deinit();

    buffers.X__base.deinit();
    buffers.X__shape_0.deinit();
    buffers.X__shape_1.deinit();
    buffers.X__shape_2.deinit();
    buffers.X__shape_3.deinit();
    buffers.X__shape_4.deinit();
    buffers.X__stride_0.deinit();
    buffers.X__stride_1.deinit();
    buffers.X__stride_2.deinit();
    buffers.X__stride_3.deinit();
    buffers.X__stride_4.deinit();
    buffers.XPtr.deinit();
    buffers.stride_x_z.deinit();
    buffers.stride_x_m.deinit();
    buffers.stride_x_k.deinit();

    buffers.W__base.deinit();
    buffers.W__shape_0.deinit();
    buffers.W__shape_1.deinit();
    buffers.W__shape_2.deinit();
    buffers.W__stride_0.deinit();
    buffers.W__stride_1.deinit();
    buffers.W__stride_2.deinit();
    buffers.WPtr.deinit();
    buffers.stride_w_e.deinit();
    buffers.stride_w_k.deinit();
    buffers.stride_w_n.deinit();

    buffers.WMxScale__base.deinit();
    buffers.WMxScale__shape_0.deinit();
    buffers.WMxScale__shape_1.deinit();
    buffers.WMxScale__shape_2.deinit();
    buffers.WMxScale__stride_0.deinit();
    buffers.WMxScale__stride_1.deinit();
    buffers.WMxScale__stride_2.deinit();
    buffers.stride_w_mx_e.deinit();
    buffers.stride_w_mx_k.deinit();
    buffers.stride_w_mx_n.deinit();

    buffers.B.deinit();
    buffers.stride_b_e.deinit();

    buffers.M.deinit();
    buffers.N.deinit();
    buffers.K.deinit();
    buffers.K_W.deinit();

    buffers.XSliceSizes.deinit();
    buffers.XSliceOffs.deinit();
    buffers.XBlockOffs.deinit();
    buffers.XBlockSchedule.deinit();

    buffers.batch_size.deinit();
    buffers.grid_m.deinit();
    buffers.grid_n.deinit();
    buffers.out_alpha.deinit();
    buffers.reduce_rank.deinit();

    buffers.YOut.deinit();
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    var safetensors_path: []const u8 = "moe_kernel.safetensors";
    var it = std.process.args();
    defer it.deinit();
    while (it.next()) |arg| {
        if (std.mem.startsWith(u8, arg, "--safetensors=")) {
            safetensors_path = arg["--safetensors=".len..];
        }
    }

    var registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, safetensors_path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const inputs: KernelInputs = KernelInputs.init(store.view());

    var platform = try zml.Platform.auto(allocator, io, .{});
    defer platform.deinit(allocator);

    if (platform.target != .cuda) {
        log.warn("Platform is not CUDA, skipping execution. This test requires CUDA.", .{});
        return;
    }

    var exe = try platform.compileFn(allocator, io, runKernel, .{inputs});
    defer exe.deinit();

    var buffers = try zml.io.load(KernelInputs, &inputs, allocator, io, platform, .{
        .parallelism = 4,
        .store = &store,
        .dma_chunks = 4,
        .dma_chunk_size = 16 * zml.MiB,
        .progress = null,
    });
    defer deinitKernelInputsBuffers(&buffers);

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{buffers});

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    exe.callOpts(io, args, &results, .{ .wait = true });

    var y_out_buf: zml.Buffer = results.get(zml.Buffer);
    defer y_out_buf.deinit();

    const y_slice = try y_out_buf.toSliceAlloc(allocator, io);
    defer y_slice.free(allocator);

    var checksum: u64 = 0;
    for (y_slice.constData()) |b| {
        checksum +%= b;
    }

    log.info("Kernel completed. YOut shape={f} checksum(bytes)={d}", .{ y_slice.shape, checksum });
}
