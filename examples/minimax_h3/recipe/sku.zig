const std = @import("std");

const config = @import("config.zig");
const shard = @import("shard.zig");

const log = std.log.scoped(.minimax_h3);

// =============================================================================
// recipe/sku.zig — catalog, weight paths, CUDA / XLA process env
//
// H3_SKUS=5s,5s-hd compiles a subset. Unset compiles every row.
// =============================================================================

/// NVIDIA H3 Super Acceleration Stage 1 canvas (896×512, 24 fps).
pub const draft_width: u32 = 896;
pub const draft_height: u32 = 512;
/// NVIDIA Super Accel Stage 1: four DiT evaluations.
/// ZML `--steps` counts sigma points including terminal 0.
pub const denoise_evals: u32 = 4;
pub const schedule_points: u32 = denoise_evals + 1;
/// Official H3 FlowMatch shifts. Turbo 4-step is trained on these knots, not linear.
pub const turbo_video_shift: f32 = 12.0;
pub const turbo_audio_shift: f32 = 3.0;
pub const lora_strength: f32 = 1.0;
pub const ltx_refine_evals: u32 = 3;
pub const ltx_lora_strength: f32 = 0.8;
/// LTX-2.5 Stage 2 distilled sigmas. The sampler appends terminal 0.
pub const ltx_stage2_sigmas = [_]f32{ 0.909375, 0.725, 0.421875 };
pub const ltx_stage2_taus = [_]f32{ 1.0, 1.25, 1.5 };
pub const target_width: u32 = 1344;
pub const target_height: u32 = 768;
/// 32-aligned Full HD (1920×1080 is not a multiple of the VAE canvas).
pub const hd_width: u32 = 1920;
pub const hd_height: u32 = 1088;
pub const hd_draft_width: u32 = 1280;
pub const hd_draft_height: u32 = 704;

pub const Sku = struct {
    id: []const u8,
    duration_s: f32,
    draft_w: u32,
    draft_h: u32,
    target_w: u32,
    target_h: u32,
};

pub const skus = [_]Sku{
    .{ .id = "5s", .duration_s = 5, .draft_w = draft_width, .draft_h = draft_height, .target_w = target_width, .target_h = target_height },
    .{ .id = "10s", .duration_s = 10, .draft_w = draft_width, .draft_h = draft_height, .target_w = target_width, .target_h = target_height },
    .{ .id = "15s", .duration_s = 15, .draft_w = draft_width, .draft_h = draft_height, .target_w = target_width, .target_h = target_height },
    .{ .id = "5s-hd", .duration_s = 5, .draft_w = hd_draft_width, .draft_h = hd_draft_height, .target_w = hd_width, .target_h = hd_height },
    .{ .id = "10s-hd", .duration_s = 10, .draft_w = hd_draft_width, .draft_h = hd_draft_height, .target_w = hd_width, .target_h = hd_height },
    .{ .id = "15s-hd", .duration_s = 15, .draft_w = hd_draft_width, .draft_h = hd_draft_height, .target_w = hd_width, .target_h = hd_height },
};
pub const default_sku_id = "5s";

pub fn byId(id: []const u8) ?Sku {
    for (skus) |row| {
        if (std.mem.eql(u8, row.id, id)) return row;
    }
    return null;
}

/// `H3_SKUS=5s,5s-hd` compiles a subset. Unset means the full catalog.
pub fn enabled(id: []const u8) bool {
    const raw_c = std.c.getenv("H3_SKUS") orelse return true;
    const raw = std.mem.span(raw_c);
    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |part| {
        const p = std.mem.trim(u8, part, " ");
        if (p.len != 0 and std.mem.eql(u8, p, id)) return true;
    }
    return false;
}

pub fn seconds(row: Sku) u32 {
    return @intFromFloat(row.duration_s);
}

pub fn isHd(row: Sku) bool {
    return row.target_w >= hd_width;
}

pub fn isRequired(row: Sku) bool {
    return std.mem.eql(u8, row.id, default_sku_id);
}

pub fn familyLabel(row: Sku) []const u8 {
    return if (isHd(row)) "Full HD" else "Super";
}

/// Upsampled HD latent height after VAE /32 and the ×2 spatial upscaler.
pub fn hdUpsampledH() u32 {
    return hd_height / 32;
}

pub fn enabledCount() usize {
    var n: usize = 0;
    for (skus) |row| {
        if (enabled(row.id)) n += 1;
    }
    return n;
}

pub fn collectEnabled(buf: *[skus.len]Sku) []Sku {
    var n: usize = 0;
    for (skus) |row| {
        if (enabled(row.id)) {
            buf[n] = row;
            n += 1;
        }
    }
    return buf[0..n];
}

/// Shared H3 + Gemma compile cap. Demo prompts fit in ~80. 256 is LTX-core's
/// Gemma width; 1024 would 4× the H3 encoder on every request.
pub const prompt_tokens: u32 = 256;

pub const default_model = "hf://MiniMaxAI/MiniMax-H3";
pub const default_lora_path = "hf://larryvrh/MiniMax-H3-Turbo-Lora/minimax_h3_turbo_v4_step600_ema.safetensors";
pub const default_taeh3_path = "https://github.com/simsim9-stack/ComfyUI-MiniMaxH3-PreviewOverride/raw/main/minivae/taeh3_decoder.safetensors";
pub const hf_ltx_dit = "hf://Lightricks/LTX-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors";
pub const hf_ltx_lora = "hf://Lightricks/LTX-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors";
pub const hf_ltx_vae = "hf://Lightricks/LTX-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors";
pub const hf_ltx_up = "hf://Lightricks/LTX-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";
pub const hf_gemma = "hf://Lightricks/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors";
pub const hf_gemma_tokenizer = "hf://google/gemma-4-12B-it/tokenizer.json";
pub const http_taehv = "https://github.com/madebyollin/taehv/raw/2026_03_11_taeltx23_wide/safetensors/taeltx2_3_wide.safetensors";
/// Optional local fused overlay. Missing is fine — runtime LoRA merge uses `default_lora_path`.
pub const default_fused_dit = "/var/models/super-accel/h3-turbo-fused";
pub const taeh3_paths = [_][]const u8{
    "/var/models/super-accel/taeh3.safetensors",
    "output/taeh3.safetensors",
    default_taeh3_path,
};
/// XLA GPU autotune + CUDA JIT cache. Graph compile still runs each process.
pub const default_cache_dir = "output/xla-cache";

pub fn draftCanvas() config.Size {
    return .{ .w = draft_width, .h = draft_height };
}

/// Stage 2 encodes at half the target, then the x2 latent upscaler restores it.
pub fn refineEncodeSize(target_w: u32, target_h: u32) error{InvalidSize}!config.Size {
    if (target_w < 64 or target_h < 64) return error.InvalidSize;
    const w = (target_w / 2 / config.canvas_multiple) * config.canvas_multiple;
    const h = (target_h / 2 / config.canvas_multiple) * config.canvas_multiple;
    if (w == 0 or h == 0) return error.InvalidSize;
    return .{ .w = w, .h = h };
}

pub fn centerCrop(src_w: u32, src_h: u32, crop_w: u32, crop_h: u32) struct { x: u32, y: u32, w: u32, h: u32 } {
    const w = @min(crop_w, src_w);
    const h = @min(crop_h, src_h);
    return .{
        .x = (src_w - w) / 2,
        .y = (src_h - h) / 2,
        .w = w,
        .h = h,
    };
}

pub fn visibleDevices(n: u32) [128]u8 {
    var buf: [128]u8 = @splat(0);
    var i: usize = 0;
    var d: u32 = 0;
    while (d < n) : (d += 1) {
        if (i != 0) {
            if (i + 1 >= buf.len) break;
            buf[i] = ',';
            i += 1;
        }
        const wrote = std.fmt.bufPrint(buf[i..], "{d}", .{d}) catch break;
        i += wrote.len;
    }
    return buf;
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

var cuda_pin: [128]u8 = @splat(0);

/// Restrict CUDA_VISIBLE_DEVICES to the TP degree so leftover GPUs stay closed.
/// Returns that degree, or 0 when the visible set cannot be determined.
pub fn narrowVisible(io: std.Io, heads: shard.HeadCounts, cap: u32) usize {
    if (std.c.getenv("CUDA_VISIBLE_DEVICES")) |raw_c| {
        const n = writeCudaSubset(std.mem.span(raw_c), cap, heads);
        if (n != 0) log.info("cuda visible pinned to {d} GPU(s)", .{n});
        return n;
    }
    const host = countHostGpus(io);
    const available: usize = if (cap != 0)
        if (host) |h| @min(cap, h) else cap
    else
        host orelse return 0;
    const tp = shard.tensorParallelDegreeForAll(available, heads);
    if (tp == 0) return 0;
    applyCudaVisible(@intCast(tp));
    log.info("cuda visible pinned to {d} GPU(s) (host={d})", .{ tp, available });
    return tp;
}

fn countHostGpus(io: std.Io) ?usize {
    var dir = std.Io.Dir.openDirAbsolute(io, "/proc/driver/nvidia/gpus", .{ .iterate = true }) catch return null;
    defer dir.close(io);
    var n: usize = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |_| n += 1;
    return if (n == 0) null else n;
}

fn writeCudaSubset(raw: []const u8, cap: u32, heads: shard.HeadCounts) usize {
    var ids: [16][]const u8 = undefined;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, raw, ',');
    while (it.next()) |part| {
        const p = std.mem.trim(u8, part, " ");
        if (p.len == 0) continue;
        if (n >= ids.len) break;
        ids[n] = p;
        n += 1;
    }
    if (n == 0) return 0;
    if (cap != 0) n = @min(n, cap);
    const tp = shard.tensorParallelDegreeForAll(n, heads);
    if (tp == 0) return 0;
    var i: usize = 0;
    var k: usize = 0;
    while (k < tp) : (k += 1) {
        if (k != 0) {
            cuda_pin[i] = ',';
            i += 1;
        }
        const id = ids[k];
        if (i + id.len + 1 >= cuda_pin.len) break;
        @memcpy(cuda_pin[i..][0..id.len], id);
        i += id.len;
    }
    cuda_pin[i] = 0;
    _ = setenv("CUDA_VISIBLE_DEVICES", @as([*:0]const u8, @ptrCast(&cuda_pin)), 1);
    return tp;
}

pub fn applyCudaVisible(n: u32) void {
    if (n == 0) return;
    if (std.c.getenv("CUDA_VISIBLE_DEVICES") != null) return;
    var vis = visibleDevices(n);
    vis[vis.len - 1] = 0;
    _ = setenv("CUDA_VISIBLE_DEVICES", @as([*:0]const u8, @ptrCast(&vis)), 1);
}

/// Persist XLA autotune and CUDA JIT so later boots skip kernel retune.
/// Does not serialize PJRT executables; HLO compile still runs.
pub fn applyCompileCache(io: std.Io) void {
    const cuda_dir = default_cache_dir ++ "/cuda";
    std.Io.Dir.cwd().createDirPath(io, default_cache_dir) catch {};
    std.Io.Dir.cwd().createDirPath(io, cuda_dir) catch {};
    if (std.c.getenv("ZML_AUTOTUNE_CACHE_DIR") == null) {
        _ = setenv("ZML_AUTOTUNE_CACHE_DIR", default_cache_dir, 0);
    }
    if (std.c.getenv("CUDA_CACHE_PATH") == null) {
        _ = setenv("CUDA_CACHE_PATH", cuda_dir, 0);
    }
}

/// Native SM100 GEMM / CUDA-graph command buffers. Append, do not replace, any
/// existing `XLA_FLAGS` (the CUDA plugin adds `--xla_gpu_cuda_data_dir=`).
pub fn applyXlaAccelFlags() void {
    const extra = " --xla_gpu_enable_triton_gemm=true --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,CUDNN,CUSTOM_CALL,DYNAMIC_SLICE_FUSION";
    const prev = std.c.getenv("XLA_FLAGS") orelse "";
    var buf: [1024]u8 = undefined;
    const merged = std.fmt.bufPrintZ(&buf, "{s}{s}", .{ std.mem.span(prev), extra }) catch return;
    _ = setenv("XLA_FLAGS", merged, 1);
}

pub fn fusedDitPresent(io: std.Io, path: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, path, .{}) catch return false;
    defer dir.close(io);
    const fused = dir.openFile(io, "fused.safetensors", .{ .mode = .read_only }) catch return false;
    fused.close(io);
    const index = dir.openFile(io, "diffusion_pytorch_model.safetensors.index.json", .{ .mode = .read_only }) catch return false;
    index.close(io);
    return true;
}

/// Runtime LoRA merge is the fallback when no fused overlay is available.
/// `--lora` forces a merge. `--dit` without `--lora` assumes the overlay is already fused.
pub fn useRuntimeLora(dit_override: []const u8, lora_override: []const u8, fused_ready: bool) bool {
    if (lora_override.len != 0) return true;
    if (dit_override.len != 0) return false;
    return !fused_ready;
}

pub fn resolvedDit(dit_override: []const u8, fused_ready: bool) []const u8 {
    if (dit_override.len != 0) return dit_override;
    if (fused_ready) return default_fused_dit;
    return "";
}

pub fn validatePrompt(prompt: []const u8) !void {
    if (std.mem.trim(u8, prompt, " \t\r\n").len == 0) return error.IntentEmpty;
}

/// Official TAEHV H3 pixel frames from latent time: `(t - 2) // 5 * 17 + 5`.
pub fn h3OutFrames(latent_t: u32) u32 {
    if (latent_t < 2) return 1;
    return (latent_t - 2) / 5 * 17 + 5;
}
