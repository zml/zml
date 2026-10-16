//! Named benchmark seeds, Cartesian configuration sweeps, and preflight validation.
const std = @import("std");
const zml = @import("zml");

pub const Routing = enum { uniform, skewed, balanced };
pub const Data = enum { fixture, random };

pub const Args = struct {
    pub const help =
        \\MoE kernel tuning benchmark
        \\  --preset=smoke|deepseek-decode|deepseek-prefill|laguna-decode|gemma-prefill|many-experts|all
        \\  --backend=cute,triton --scheme=none,mxfp4 --dtype=bf16,f32
        \\  --activation=swiglu --quantize-input=false,true
        \\  --placement=before_down,after_down --bias=false,true
        \\  --tokens=1,17,512,4096 --topk=6 --width=5120
        \\  --intermediate=2304 --experts=192 --swiglu-limit=none,7,10
        \\  --global-experts=384 --active-experts=0
        \\  --routing=uniform,skewed,balanced --sigma=0.6
        \\  --input-scale=0,1,16 --routing-scale=1,2.5
        \\  --data=fixture|random --seed=1 --seeds=1,2,3
        \\  --warmup=3 --iters=20 --samples=7 --sample-ms=20
        \\  --platform=cuda --cpu-devices=2 --filter=<case substring> --list
        \\  --check --json=results.jsonl --dump=out.bin --compare=ref.bin
        \\  --dump-dir=baselines --compare-dir=baselines --compare-exact
        \\
        \\Presets are visible seeds, independent of the correctness-test matrix.
        \\Each supplied axis replaces that preset value; comma lists form a Cartesian product.
        \\Unsupported combinations remain visible as skipped results; --filter matches case IDs.
        \\Default preset: smoke (CuTe/Fly: deepseek-decode). --preset=all selects the catalog.
        \\Default scheme: none (CuTe/Fly: mxfp4). Other defaults come from the preset.
        \\--swiglu-limit applies to SwiGLU/step activations; none removes clipping.
        \\--experts is per rank; by default both unsharded and available EP cases run.
        \\global-experts / experts must equal the device mesh partition count.
        \\--list initializes the platform but does not compile or allocate inputs.
        \\StableHLO is the fallback when no accelerated backend is available.
        \\Runtime routing variants and seeds share compilation of the same graph.
        \\Samples measure synchronized public API wall time, including dispatch overhead.
        \\Dump/compare require exactly one runnable case. Results are append-only JSONL.
        \\
    ;
    preset: ?[]const u8 = null,
    backend: ?[]const u8 = null,
    scheme: ?[]const u8 = null,
    dtype: ?[]const u8 = null,
    activation: ?[]const u8 = null,
    quantize_input: ?[]const u8 = null,
    placement: ?[]const u8 = null,
    bias: ?[]const u8 = null,
    tokens: ?[]const u8 = null,
    topk: ?[]const u8 = null,
    width: ?[]const u8 = null,
    intermediate: ?[]const u8 = null,
    experts: ?[]const u8 = null,
    swiglu_limit: ?[]const u8 = null,
    routing: []const u8 = "uniform,skewed,balanced",
    global_experts: ?i64 = null,
    active_experts: i64 = 0,
    sigma: f32 = 0.6,
    input_scale: []const u8 = "1",
    routing_scale: []const u8 = "1",
    data: Data = .random,
    seed: u64 = 1,
    seeds: ?[]const u8 = null,
    warmup: u32 = 3,
    iters: u32 = 20,
    samples: u32 = 7,
    sample_ms: u32 = 20,
    platform: ?zml.Target = null,
    cpu_devices: u32 = 2,
    filter: ?[]const u8 = null,
    list: bool = false,
    check: bool = false,
    json: ?[]const u8 = null,
    dump: ?[]const u8 = null,
    compare: ?[]const u8 = null,
    dump_dir: ?[]const u8 = null,
    compare_dir: ?[]const u8 = null,
    compare_exact: bool = false,

    pub fn validate(a: Args) !void {
        if ((a.dump != null and a.dump_dir != null) or (a.compare != null and a.compare_dir != null)) return error.ConflictingOutputPaths;
        if (a.dump_dir) |path| if (a.compare_dir) |other| if (std.mem.eql(u8, path, other)) return error.ConflictingOutputPaths;
        if (a.compare_exact and a.compare == null and a.compare_dir == null) return error.MissingComparisonBaseline;
        if (a.iters == 0 or a.samples == 0 or a.sample_ms == 0) return error.InvalidIterationCount;
        if (a.cpu_devices == 0) return error.InvalidDeviceCount;
        if (a.active_experts < 0 or !std.math.isFinite(a.sigma) or a.sigma < 0) return error.InvalidRouting;
        inline for (.{ a.input_scale, a.routing_scale }) |list| {
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |value| {
                const scale = try std.fmt.parseFloat(f32, value);
                if (!std.math.isFinite(scale) or scale < 0) return error.InvalidDataScale;
            }
        }
        if (a.global_experts) |e| if (e <= 0 or e > std.math.maxInt(i32)) return error.InvalidExpertCount;
        try validateEnumList(zml.moe.Backend, a.backend);
        try validateEnumList(zml.DataType, a.dtype);
        try validateEnumList(zml.moe.ActivationKind, a.activation);
        try validateEnumList(zml.moe.RoutingWeightPlacement, a.placement);
        try validateEnumList(Routing, a.routing);
        inline for (.{ a.quantize_input, a.bias }) |list| if (list) |s| {
            var it = std.mem.splitScalar(u8, s, ',');
            while (it.next()) |v| if (!std.mem.eql(u8, v, "true") and !std.mem.eql(u8, v, "false")) return error.InvalidBoolean;
        };
        if (a.preset) |list| {
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |name| {
                if (std.mem.eql(u8, name, "all")) continue;
                var found = false;
                for (presets) |p| found = found or std.mem.eql(u8, name, p.name);
                if (!found) return error.InvalidPreset;
            }
        }
        if (a.scheme) |list| {
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |v| if (!std.mem.eql(u8, v, "none") and std.meta.stringToEnum(zml.Quantization.Scheme, v) == null) return error.InvalidScheme;
        }
        inline for (.{ a.tokens, a.topk, a.width, a.intermediate, a.experts }) |list| if (list) |s| {
            var it = std.mem.splitScalar(u8, s, ',');
            while (it.next()) |v| {
                const n = try std.fmt.parseInt(i64, v, 10);
                if (n <= 0 or n > std.math.maxInt(i32)) return error.InvalidDimension;
            }
        };
        if (a.swiglu_limit) |list| {
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |v| {
                if (std.mem.eql(u8, v, "none")) continue;
                const n = try std.fmt.parseFloat(f32, v);
                if (!std.math.isFinite(n) or n < 0) return error.InvalidActivationLimit;
            }
        }
        if (a.seeds) |list| {
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |v| _ = try std.fmt.parseInt(u64, v, 10);
        }
        const paths = [_]?[]const u8{ a.json, a.dump, a.compare };
        for (paths, 0..) |path, i| if (path) |p| {
            for (paths[0..i]) |other| if (other) |o| if (std.mem.eql(u8, p, o)) return error.ConflictingOutputPaths;
        };
        inline for (.{ a.dump, a.compare }) |baseline| if (baseline) |base| {
            for (paths) |path| if (path) |p| {
                if (p.len == base.len + 5 and std.mem.startsWith(u8, p, base) and std.mem.endsWith(u8, p, ".json")) return error.ConflictingOutputPaths;
            };
        };
    }
};

fn validateEnumList(comptime T: type, list: ?[]const u8) !void {
    if (list) |s| {
        var it = std.mem.splitScalar(u8, s, ',');
        while (it.next()) |v| if (std.meta.stringToEnum(T, v) == null) return error.InvalidEnum;
    }
}

fn matches(list: []const u8, value: []const u8) bool {
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |v| if (std.mem.eql(u8, v, value)) return true;
    return false;
}

pub const Case = struct {
    backend: zml.moe.Backend = .stablehlo,
    activation: zml.moe.Activation = .{ .swiglu = .{} },
    scheme: ?zml.Quantization.Scheme = null,
    activation_dtype: zml.DataType = .bf16,
    quantize_input: bool = false,
    placement: zml.moe.RoutingWeightPlacement = .after_down,
    bias: bool = false,
    tokens: i64 = 1,
    topk: i64 = 1,
    width: i64 = 128,
    intermediate: i64 = 128,
    experts: i64 = 8,

    pub fn options(c: Case) zml.moe.Options {
        return .{ .activation = c.activation, .quantize_input = c.quantize_input, .routing_weight_placement = c.placement };
    }

    pub fn referenceOptions(c: Case, target: zml.Target) zml.moe.stablehlo.Options {
        return .{
            .activation = c.activation,
            .quantize_input = c.quantize_input,
            .routing_weight_placement = c.placement,
            .input_quantization = switch (c.backend) {
                .metal => .none,
                .cute => .mxfp8_bf16,
                .triton => if (c.scheme == .mxfp4 and target == .cuda) .mxfp8 else .automatic,
                else => .automatic,
            },
        };
    }

    pub fn tolerance(c: Case) zml.testing.CompareOpts {
        return .{
            .absolute_tolerance = if (c.scheme == .nvfp4) 0.078125 else 0.015625,
            .relative_tolerance = 0.02,
            .minimum_close_fraction = 1,
        };
    }
};

pub const Preset = struct { name: []const u8, description: []const u8, config: Case };

/// Model-inspired geometry with bounded local expert counts; override any dimension.
/// Adding a row adds a named seed, without modifying selection or backend validation.
pub const presets = [_]Preset{
    .{
        .name = "smoke",
        .description = "Small public API smoke case",
        .config = .{ .tokens = 17, .topk = 2 },
    },
    .{
        .name = "deepseek-decode",
        .description = "DeepSeek expert geometry, decode",
        .config = .{
            .tokens = 1,
            .topk = 6,
            .width = 5120,
            .intermediate = 2304,
            .activation = .{ .swiglu = .{ .limit = 10 } },
            .placement = .before_down,
        },
    },
    .{
        .name = "deepseek-prefill",
        .description = "DeepSeek expert geometry, prefill",
        .config = .{
            .tokens = 512,
            .topk = 6,
            .width = 5120,
            .intermediate = 2304,
            .activation = .{ .swiglu = .{ .limit = 10 } },
            .placement = .before_down,
        },
    },
    .{
        .name = "laguna-decode",
        .description = "Laguna routed expert geometry",
        .config = .{ .tokens = 1, .topk = 8, .width = 2048, .intermediate = 512, .experts = 256 },
    },
    .{
        .name = "gemma-prefill",
        .description = "Gemma hidden width with a bounded expert set",
        .config = .{ .tokens = 17, .topk = 2, .width = 5376, .intermediate = 512, .activation = .geglu_tanh },
    },
    .{
        .name = "many-experts",
        .description = "Routing pressure with small expert matrices",
        .config = .{ .tokens = 512, .topk = 6, .experts = 384 },
    },
};

pub const Workload = struct {
    case: Case,
    preset: []const u8 = "custom",
    routing: Routing,
    global_experts: i64,
    active_experts: i64,
    sigma: f32,
    seed: u64,
    data: Data,
    input_scale: f32 = 1,
    routing_scale: f32 = 1,

    pub fn isSharded(w: Workload) bool {
        return w.global_experts != w.case.experts;
    }

    pub fn mechanism(w: Workload) []const u8 {
        return if (w.isSharded()) "forwardMoe_expert_parallel" else "forwardMoe_unsharded";
    }

    pub fn sameGraph(a: Workload, b: Workload) bool {
        return std.meta.eql(a.case, b.case) and a.global_experts == b.global_experts and a.data == b.data;
    }

    pub fn graphId(w: Workload, allocator: std.mem.Allocator) ![]const u8 {
        const c = w.case;
        const limit: ?f32 = switch (c.activation) {
            .swiglu => |a| a.limit,
            .swiglu_step => |a| a.limit,
            else => null,
        };
        const bias: ?f32 = if (c.activation == .swiglu) c.activation.swiglu.bias else null;
        const scale: ?f32 = if (c.activation == .swiglu) c.activation.swiglu.scale else null;
        return std.fmt.allocPrint(allocator, "{s}/scheme={s}/dtype={s}/activation={s}/limit={?d}/activation-bias={?d}/scale={?d}/quantize-input={}/placement={s}/bias={}/tokens={d}/topk={d}/width={d}/intermediate={d}/experts={d}/global-experts={d}/data={s}", .{ @tagName(c.backend), if (c.scheme) |s| @tagName(s) else "none", @tagName(c.activation_dtype), @tagName(c.activation), limit, bias, scale, c.quantize_input, @tagName(c.placement), c.bias, c.tokens, c.topk, c.width, c.intermediate, c.experts, w.global_experts, @tagName(w.data) });
    }

    /// IDs describe effective inputs, so duplicate overrides across presets collapse.
    pub fn id(w: Workload, allocator: std.mem.Allocator) ![]const u8 {
        const graph = try w.graphId(allocator);
        defer allocator.free(graph);
        const base = try std.fmt.allocPrint(allocator, "{s}/routing={s}/active-experts={d}/sigma={d}/seed={d}", .{ graph, @tagName(w.routing), w.active_experts, w.sigma, w.seed });
        // Preserve identities of existing baselines with unscaled operands.
        if (w.input_scale == 1 and w.routing_scale == 1) return base;
        defer allocator.free(base);
        return std.fmt.allocPrint(allocator, "{s}/input-scale={d}/routing-scale={d}", .{ base, w.input_scale, w.routing_scale });
    }

    pub fn unsupported(w: Workload, target: zml.Target) ?[]const u8 {
        const c = w.case;
        inline for (.{ c.tokens, c.topk, c.width, c.intermediate, c.experts }) |dim| {
            if (dim <= 0 or dim > std.math.maxInt(i32)) return "dimensions must be positive i32 values";
        }
        if (w.global_experts <= 0 or w.global_experts > std.math.maxInt(i32)) return "global expert count must be a positive i32 value";
        if (w.global_experts < c.experts or @mod(w.global_experts, c.experts) != 0) return "global-experts must be a multiple of local experts";
        if (w.active_experts < 0 or w.active_experts > c.experts) return "active-experts must be between zero and local experts";
        if (!std.math.isFinite(w.sigma) or w.sigma < 0) return "routing sigma must be finite and nonnegative";
        if (!std.math.isFinite(w.input_scale) or w.input_scale < 0 or !std.math.isFinite(w.routing_scale) or w.routing_scale < 0) return "data scales must be finite and nonnegative";
        const active = if (w.active_experts == 0) c.experts else w.active_experts;
        if (c.topk > @divTrunc(w.global_experts, c.experts) * active) return "top-k exceeds eligible global experts";
        if (c.tokens > @divTrunc(std.math.maxInt(i32), c.topk) or c.tokens * c.topk > @divTrunc(std.math.maxInt(i32), @max(c.width, 2 * c.intermediate))) return "route tensor exceeds kernel i32 indexing";
        if (w.global_experts > @divTrunc(std.math.maxInt(i64), c.width * 2 * c.intermediate)) return "expert tensor size overflows i64";
        switch (c.activation_dtype) {
            .bf16, .f16, .f32 => {},
            else => return "activation dtype must be bf16, f16, or f32",
        }
        const limit: ?f32 = switch (c.activation) {
            .swiglu => |a| a.limit,
            .swiglu_step => |a| a.limit,
            else => null,
        };
        if (limit) |l| if (!std.math.isFinite(l) or l < 0) return "activation limit must be finite and nonnegative";
        if (c.activation == .swiglu) {
            if (c.activation.swiglu.scale != null) return "SwiGLU scale is unsupported";
            if (c.activation.swiglu.bias) |b| if (!std.math.isFinite(b)) return "activation bias must be finite";
        }
        const gated = switch (c.activation) {
            .relu, .gelu, .silu => false,
            else => true,
        };
        switch (c.backend) {
            .stablehlo => {},
            .cute, .fly, .triton => {
                if (c.activation_dtype != .bf16) return "backend requires BF16 activations";
                if (c.scheme == .nvfp4) return "backend does not support NVFP4 weights";
                if (c.backend == .cute or c.backend == .fly) {
                    if (c.scheme != .mxfp4) return "backend requires MXFP4 weights";
                    if (c.width != 5120 or c.intermediate != 2304) return "backend requires width=5120 and intermediate=2304";
                    if (c.placement != .before_down) return "backend requires routing weights before down projection";
                    if (c.bias) return "backend does not support projection bias";
                    if (!gated) return "backend requires a gated activation";
                }
                if (c.backend == .fly and w.global_experts > 512) return "Fly supports at most 512 global experts";
                if (c.backend == .cute or (c.backend == .triton and target == .cuda and c.scheme == .mxfp4)) {
                    if (c.activation != .swiglu) return "backend requires clipped SwiGLU";
                    if (c.activation.swiglu.limit == null or c.activation.swiglu.bias != null) return "backend requires clipped SwiGLU without activation bias";
                    if (c.bias) return "backend does not support projection bias";
                    if (@mod(c.width, 128) != 0 or @mod(c.intermediate, 128) != 0) return "CUDA MXFP4 dimensions must be multiples of 128";
                }
                if (c.backend == .triton and (@mod(c.width, 16) != 0 or @mod(c.intermediate, 16) != 0)) return "Triton dimensions must be multiples of 16";
            },
            .flashinfer_cutlass, .mosaic_tpu, .metal => {
                if (c.quantize_input) return "backend does not support input quantization";
                if (c.placement != .after_down) return "backend requires routing weights after down projection";
                switch (c.backend) {
                    .flashinfer_cutlass => {
                        if (c.scheme != null and c.scheme != .nvfp4) return "FlashInfer supports unquantized or NVFP4 weights";
                        if (c.activation_dtype != .bf16) return "FlashInfer requires BF16 activations";
                        if (c.activation != .swiglu) return "FlashInfer requires unclipped SwiGLU";
                        if (c.activation.swiglu.limit != null or c.activation.swiglu.bias != null) return "FlashInfer requires unclipped, unbiased SwiGLU";
                        if (c.bias) return "FlashInfer does not support projection bias";
                        if (c.topk > c.experts) return "FlashInfer requires top-k <= local experts";
                    },
                    .mosaic_tpu => {
                        if (c.scheme != null) return "Mosaic TPU does not support quantized weights";
                        if (!gated) return "Mosaic TPU requires a gated activation";
                    },
                    .metal => {
                        if (c.scheme == .mxfp4) return "Metal does not support MXFP4 weights";
                        if (c.scheme != null and c.activation_dtype != .bf16) return "Metal quantized weights require BF16 activations";
                        if (c.bias) return "Metal does not support projection bias";
                    },
                    else => unreachable,
                }
            },
        }
        const alignment: i64 = if (c.scheme) |s| switch (s) {
            .mxfp4, .mxfp8, .fp8_block32 => 32,
            .nvfp4 => 16,
            .fp8_block128 => 128,
            .fp8_per_channel, .fp8_per_tensor => 1,
        } else 1;
        if (@mod(c.width, alignment) != 0 or @mod(c.intermediate, alignment) != 0) return "dimensions do not satisfy quantization/layout alignment";
        if (c.backend == .flashinfer_cutlass and c.scheme == .nvfp4 and (@mod(c.width, 128) != 0 or @mod(c.intermediate, 64) != 0)) return "FlashInfer scale layout requires width multiple of 128 and intermediate multiple of 64";
        return null;
    }
};

pub const Selection = struct { workload: Workload, id: []const u8 };

/// The allocator owns the returned list and identifiers (normally a run-scoped arena).
pub fn select(allocator: std.mem.Allocator, args: Args, target: zml.Target, expert_partitions: i64) ![]Selection {
    _ = target;
    try args.validate();
    if (expert_partitions <= 0) return error.InvalidDeviceCount;
    var result: std.ArrayList(Selection) = .empty;
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(allocator);
    for (std.meta.tags(zml.moe.Backend)) |backend| {
        if (args.backend) |requested| {
            if (!matches(requested, @tagName(backend))) continue;
        } else if (backend == .stablehlo) continue;
        const specialized = backend == .cute or backend == .fly;
        const requested = args.preset orelse if (specialized) "deepseek-decode" else "smoke";
        for (presets) |preset| {
            if (!matches(requested, "all") and !matches(requested, preset.name)) continue;
            var c = preset.config;
            c.backend = backend;
            if (specialized) c.scheme = .mxfp4;
            try expandAxes(0, allocator, args, preset.name, c, expert_partitions, &result, &seen);
        }
    }
    return result.toOwnedSlice(allocator);
}

// One list defines the sweep dimensions; values absent from the CLI retain the preset.
const axes = .{ "scheme", "dtype", "activation", "quantize_input", "placement", "bias", "tokens", "topk", "width", "intermediate", "experts", "swiglu_limit" };

fn expandAxes(comptime axis: usize, allocator: std.mem.Allocator, args: Args, preset: []const u8, c: Case, expert_partitions: i64, result: *std.ArrayList(Selection), seen: *std.StringHashMapUnmanaged(void)) !void {
    if (comptime axis == axes.len) return expandRouting(allocator, args, preset, c, expert_partitions, result, seen);
    const name = axes[axis];
    if (@field(args, name)) |list| {
        var it = std.mem.splitScalar(u8, list, ',');
        while (it.next()) |value| {
            var next = c;
            if (comptime std.mem.eql(u8, name, "scheme")) {
                next.scheme = if (std.mem.eql(u8, value, "none")) null else std.meta.stringToEnum(zml.Quantization.Scheme, value).?;
            } else if (comptime std.mem.eql(u8, name, "dtype")) {
                next.activation_dtype = std.meta.stringToEnum(zml.DataType, value).?;
            } else if (comptime std.mem.eql(u8, name, "activation")) {
                const kind = std.meta.stringToEnum(zml.moe.ActivationKind, value).?;
                if (kind != std.meta.activeTag(c.activation)) {
                    next.activation = switch (kind) {
                        .swiglu => .{ .swiglu = .{} },
                        .swiglu_step => .{ .swiglu_step = .{} },
                        .gelu => .gelu,
                        .relu => .relu,
                        .silu => .silu,
                        .geglu => .geglu,
                        .geglu_tanh => .geglu_tanh,
                    };
                }
            } else if (comptime std.mem.eql(u8, name, "placement")) {
                next.placement = std.meta.stringToEnum(zml.moe.RoutingWeightPlacement, value).?;
            } else if (comptime std.mem.eql(u8, name, "swiglu_limit")) {
                const limit = if (std.mem.eql(u8, value, "none")) null else try std.fmt.parseFloat(f32, value);
                switch (next.activation) {
                    .swiglu => |*a| a.limit = limit,
                    .swiglu_step => |*a| a.limit = limit,
                    else => {},
                }
            } else if (comptime std.mem.eql(u8, name, "quantize_input") or std.mem.eql(u8, name, "bias")) {
                @field(next, name) = std.mem.eql(u8, value, "true");
            } else {
                @field(next, name) = try std.fmt.parseInt(i64, value, 10);
            }
            try expandAxes(axis + 1, allocator, args, preset, next, expert_partitions, result, seen);
        }
    } else try expandAxes(axis + 1, allocator, args, preset, c, expert_partitions, result, seen);
}

fn expandRouting(allocator: std.mem.Allocator, args: Args, preset: []const u8, c: Case, expert_partitions: i64, result: *std.ArrayList(Selection), seen: *std.StringHashMapUnmanaged(void)) !void {
    const global_counts = [_]i64{ args.global_experts orelse c.experts, args.global_experts orelse try std.math.mul(i64, c.experts, expert_partitions) };
    for (global_counts) |global_experts| {
        var w: Workload = .{ .case = c, .preset = preset, .routing = .uniform, .global_experts = global_experts, .active_experts = args.active_experts, .sigma = args.sigma, .seed = args.seed, .data = args.data };
        if (args.seeds) |list| {
            var seeds = std.mem.splitScalar(u8, list, ',');
            while (seeds.next()) |seed| {
                w.seed = try std.fmt.parseInt(u64, seed, 10);
                try appendRoutings(allocator, args, w, result, seen);
            }
        } else try appendRoutings(allocator, args, w, result, seen);
    }
}

fn appendRoutings(allocator: std.mem.Allocator, args: Args, workload: Workload, result: *std.ArrayList(Selection), seen: *std.StringHashMapUnmanaged(void)) !void {
    var routings = std.mem.splitScalar(u8, args.routing, ',');
    while (routings.next()) |r| {
        var w = workload;
        w.routing = std.meta.stringToEnum(Routing, r).?;
        var input_scales = std.mem.splitScalar(u8, args.input_scale, ',');
        while (input_scales.next()) |input_scale| {
            w.input_scale = try std.fmt.parseFloat(f32, input_scale);
            var routing_scales = std.mem.splitScalar(u8, args.routing_scale, ',');
            while (routing_scales.next()) |routing_scale| {
                w.routing_scale = try std.fmt.parseFloat(f32, routing_scale);
                try append(allocator, args, w, result, seen);
            }
        }
    }
}

fn append(allocator: std.mem.Allocator, args: Args, w: Workload, result: *std.ArrayList(Selection), seen: *std.StringHashMapUnmanaged(void)) !void {
    const id = try w.id(allocator);
    if (args.filter) |filter| if (std.mem.indexOf(u8, id, filter) == null) {
        allocator.free(id);
        return;
    };
    const entry = try seen.getOrPut(allocator, id);
    if (entry.found_existing) {
        allocator.free(id);
        return;
    }
    try result.append(allocator, .{ .workload = w, .id = id });
}

test "explicit dtype and bias sweep axes retain every requested combination" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const selected = try select(arena.allocator(), .{ .backend = "stablehlo", .dtype = "bf16,f32", .bias = "false,true", .routing = "uniform" }, .cpu, 1);
    try std.testing.expectEqual(@as(usize, 4), selected.len);
    for (selected) |s| try std.testing.expect(s.workload.unsupported(.cpu) == null);
    const unsupported = try select(arena.allocator(), .{ .backend = "cute", .scheme = "none", .routing = "uniform" }, .cuda, 1);
    try std.testing.expectEqual(@as(usize, 1), unsupported.len);
    try std.testing.expect(unsupported[0].workload.unsupported(.cuda) != null);
}

test "shape overrides deduplicate while routing and seeds reuse graphs" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const selected = try select(arena.allocator(), .{ .backend = "cute", .preset = "deepseek-decode,deepseek-prefill", .tokens = "1,17,1", .seeds = "1,2", .routing = "uniform,balanced" }, .cuda, 1);
    try std.testing.expectEqual(@as(usize, 8), selected.len);
    var graphs: std.StringHashMapUnmanaged(void) = .empty;
    defer graphs.deinit(arena.allocator());
    for (selected) |s| {
        try std.testing.expect(s.workload.unsupported(.cuda) == null);
        try graphs.put(arena.allocator(), try s.workload.graphId(arena.allocator()), {});
    }
    try std.testing.expectEqual(@as(usize, 2), graphs.count());
    try std.testing.expect(selected[0].workload.sameGraph(selected[1].workload));
    try std.testing.expect(!std.mem.eql(u8, selected[0].id, selected[1].id));
}

test "operand scale sweeps reuse graphs and preserve baseline identities" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const selected = try select(arena.allocator(), .{
        .backend = "cute",
        .routing = "uniform",
        .input_scale = "0,1,16",
        .routing_scale = "1,2.5",
    }, .cuda, 1);
    try std.testing.expectEqual(@as(usize, 6), selected.len);
    for (selected) |s| try std.testing.expect(s.workload.sameGraph(selected[0].workload));
    try std.testing.expect(std.mem.indexOf(u8, selected[2].id, "input-scale") == null);
    try std.testing.expect(!std.mem.eql(u8, selected[0].id, selected[1].id));
    try std.testing.expectError(error.InvalidDataScale, (Args{ .input_scale = "nan" }).validate());
    try std.testing.expectError(error.InvalidDataScale, (Args{ .routing_scale = "-1" }).validate());
}

test "backend contracts reject unaligned and incompatible sweep combinations" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const selected = try select(arena.allocator(), .{ .backend = "triton", .scheme = "none,fp8_per_tensor,fp8_per_channel", .width = "128,129", .routing = "uniform" }, .cuda, 1);
    try std.testing.expectEqual(@as(usize, 6), selected.len);
    for (selected) |s| try std.testing.expectEqual(s.workload.case.width == 129, s.workload.unsupported(.cuda) != null);
    const invalid = try select(arena.allocator(), .{ .backend = "flashinfer_cutlass", .activation = "relu", .routing = "uniform" }, .cuda, 1);
    try std.testing.expect(invalid[0].workload.unsupported(.cuda) != null);
    const mxfp4 = try select(arena.allocator(), .{ .backend = "triton", .scheme = "mxfp4", .routing = "uniform" }, .cuda, 1);
    try std.testing.expect(mxfp4[0].workload.unsupported(.cuda) != null);
}

test "catalog selects each named preset and explicit expert parallelism" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const selected = try select(arena.allocator(), .{ .backend = "stablehlo", .preset = "all", .routing = "balanced" }, .cpu, 2);
    try std.testing.expectEqual(@as(usize, presets.len * 2), selected.len);
    for (presets) |preset| {
        var covered = false;
        for (selected) |s| covered = covered or std.mem.eql(u8, s.workload.preset, preset.name);
        try std.testing.expect(covered);
    }
    const explicit = try select(arena.allocator(), .{ .backend = "stablehlo", .experts = "8", .global_experts = 8, .routing = "uniform" }, .cpu, 2);
    try std.testing.expectEqual(@as(usize, 1), explicit.len);
    try std.testing.expect(!explicit[0].workload.isSharded());
    try std.testing.expectError(error.InvalidPreset, (Args{ .preset = "typo" }).validate());
    try std.testing.expectError(error.InvalidEnum, (Args{ .backend = "typo" }).validate());
    try std.testing.expectError(error.InvalidDimension, (Args{ .tokens = "0" }).validate());
    try std.testing.expectError(error.InvalidIterationCount, (Args{ .samples = 0 }).validate());
    try std.testing.expectError(error.ConflictingOutputPaths, (Args{ .dump = "same", .compare = "same" }).validate());
    try std.testing.expectError(error.ConflictingOutputPaths, (Args{ .dump = "output.bin", .json = "output.bin.json" }).validate());
    try std.testing.expectError(error.ConflictingOutputPaths, (Args{ .compare = "output.bin", .dump = "output.bin.json" }).validate());
    try std.testing.expectError(error.ConflictingOutputPaths, (Args{ .dump = "output.bin", .compare = "output.bin.json" }).validate());
    try std.testing.expectError(error.ConflictingOutputPaths, (Args{ .compare = "output.bin", .json = "output.bin.json" }).validate());
}
