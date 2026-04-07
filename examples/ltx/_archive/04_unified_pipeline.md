# Task: Unified End-to-End Pipeline Binary

## Context

The LTX-2.3 Zig pipeline currently has three separate binaries:

```
export (Python)
    │
    ▼
denoise_stage1 ──► video_latent.bin + audio_latent.bin     (raw bf16)
    │
    ▼
bridge         ──► stage2_inputs.safetensors               (12 tensors)
    │
    ▼
denoise_e2e    ──► video_latent.bin + audio_latent.bin     (raw bf16)
    │
    ▼
decode (Python) ──► output.mp4
```

Each binary writes intermediate files to disk, then the next binary reads them
back. This adds I/O latency and requires manual orchestration.

**Goal:** Create a single `inference.zig` binary that runs Stage 1 → bridge →
Stage 2 in one process, passing tensors in GPU memory between stages without
writing intermediate files to disk.

## Architecture Analysis

### What the three drivers share

All three drivers follow the same initialization pattern:

```zig
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);
    // ... load stores, compile, run
}
```

All three import `model.zig` and use `platform.compileFn()` to compile model
functions into XLA executables. They share the same `loadBuf`, `sigma1dBuffer`,
and `writeBuffer` helpers (duplicated in each file).

### Data flow through the pipeline

```
                 denoise_stage1
                 ══════════════
Inputs:          stage1_inputs.safetensors (14 tensors from Python export)
                 base checkpoint (ltx-2.3-22b-dev.safetensors)

Compile:         11 executables (preprocess, 4 block variants, 2 proj,
                 2 to_denoised, 2 denoise_step, 1 guider_combine)

Loop:            30 steps × 4 passes × 48 blocks

Outputs on GPU:  v_latent_buf: Buffer [1, T_v1, 128] bf16
                 a_latent_buf: Buffer [1, T_a,  128] bf16


                 bridge
                 ══════
Inputs on GPU:   v_latent_buf, a_latent_buf (from Stage 1)
                 v_context_pos, a_context_pos (kept on GPU from Stage 1)
From disk:       upsampler checkpoint, main checkpoint (per-channel stats),
                 stage2_noise.safetensors (pre-drawn random noise from Python),
                 pipeline_meta.json

**Data provenance note:**
- Text contexts (`v_context_pos`, `a_context_pos`) were loaded to GPU during
  Stage 1. In the unified binary, Stage 1 keeps them alive and passes them
  to the bridge (no re-read from disk).
- Stage 2 noise (`video_noise_s2`, `audio_noise_s2`) is pre-drawn by the
  Python export script with a deterministic seed. It is not a model weight
  and not a Stage 1 output — it is a pipeline input that must be loaded from
  `stage2_noise.safetensors`.

Compile:         5 executables (unpatchify, upsample, patchify, 2 noise_init)

Steps:           unpatchify → upsample → patchify → compute positions/masks →
                 load noise from disk → noise init

Outputs on GPU:  video_latent: Buffer [1, T_v2, 128] bf16  (noised)
                 audio_latent: Buffer [1, T_a,  128] bf16  (noised)
                 video/audio_positions, video/audio_denoise_mask: Buffers
                 v_context, a_context: Buffers (renamed from v/a_context_pos)


                 denoise_e2e
                 ═══════════
Inputs on GPU:   All bridge outputs (12 tensors)
Additional:      distilled checkpoint (ltx-2.3-22b-distilled.safetensors)

Compile:         7 executables (2 noise_init, preprocess, block, 2 proj,
                 2 denoise_step)

Loop:            3 steps × 1 pass × 48 blocks

Outputs on GPU:  v_latent_buf: Buffer [1, T_v2, 128] bf16
                 a_latent_buf: Buffer [1, T_a,  128] bf16
```

### Key observation: tensors stay on GPU

The critical insight is that between stages, the data is already on GPU as
`zml.Buffer` objects. The current pipeline just downloads them to host, writes
to disk, then the next binary re-uploads. In a unified binary, we skip all
that — the `Buffer` handles pass directly from one stage's code to the next.

## Design Decisions

### 1. Unified CLI arguments

The unified binary needs paths for:

| Argument | Source | Used by |
|----------|--------|---------|
| `--stage1-ckpt` | ltx-2.3-22b-dev.safetensors | Stage 1 |
| `--stage2-ckpt` | ltx-2.3-22b-distilled.safetensors | Stage 2 |
| `--upsampler-ckpt` | ltx-2.3-spatial-upscaler-x2-1.1.safetensors | Bridge |
| `--stage1-inputs` | stage1_inputs.safetensors | Stage 1 (contexts kept on GPU for Stage 2) |
| `--stage2-noise` | stage2_noise.safetensors | Bridge (pre-drawn noise from Python export) |
| `--meta` | pipeline_meta.json | Bridge |
| `--output-dir` | directory for final .bin outputs | Stage 2 output |
| `--bf16-attn-stage1` | flag | Stage 1 |
| `--bf16-attn-stage2` | flag | Stage 2 |

Note: `--stage2-ckpt` is also used by the bridge (for `per_channel_statistics`).
In the current bridge, this is `--main-ckpt`, but for the unified binary
the distilled checkpoint is the natural source (it contains the same VAE stats).

#### Example command

```bash
bazel run --config=release --@zml//platforms:cuda=true //examples/ltx:inference -- \
  --stage1-ckpt /root/models/ltx-2.3/ltx-2.3-22b-dev.safetensors \
  --stage2-ckpt /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
  --upsampler-ckpt /root/models/ltx-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --stage1-inputs $OUT/stage1_inputs.safetensors \
  --stage2-noise $OUT/stage2_noise.safetensors \
  --meta $OUT/pipeline_meta.json \
  --output-dir $OUT/unified \
  --bf16-attn-stage2
```

The `--bf16-attn-stage2` flag is recommended to avoid OOM on Stage 2
(larger token count after spatial upsample). Stage 1 defaults to f32 attention.
Add `--dump-intermediates` to also write `stage1_video_latent.bin` and
`stage1_audio_latent.bin` for debugging.

### 2. Memory management strategy

**Problem:** The three stages combined would hold all weights + intermediates in
GPU memory simultaneously, which may exceed VRAM.

**Solution — staged weight loading with explicit free:**

```
Phase 1: Stage 1
  - Load Stage 1 checkpoint weights → GPU
  - Load Stage 1 inputs (latents, contexts, positions, masks) → GPU
  - Run 30-step denoising loop
  - Free all Stage 1 weights (block weights, proj, adaln, etc.)
  - Free Stage 1 inputs no longer needed (negative contexts, Stage 1 positions/masks)
  - Keep: v_latent_buf, a_latent_buf, v_context_pos, a_context_pos

Phase 2: Bridge
  - Load upsampler weights + per_channel_stats → GPU
  - Run unpatchify → upsample → patchify
  - Free upsampler weights
  - Compute Stage 2 positions/masks on host → upload to GPU
  - Load Stage 2 noise from stage2_noise.safetensors → GPU
  - Reuse text contexts (already on GPU from Stage 1, no file I/O)
  - Run noise init
  - Free noise buffers (not needed after noising)
  - Keep: noised latents, positions, masks, contexts

Phase 3: Stage 2
  - Load Stage 2 (distilled) checkpoint weights → GPU
  - Run 3-step denoising loop
  - Free all Stage 2 weights
  - Keep: final v_latent_buf, a_latent_buf
  - Write .bin outputs to disk
```

This way, at any given time, only ONE set of model weights is on GPU.
The base model (~22B params) and distilled model (~22B params) never coexist.

### 3. Code organization

**Option A: Single `inference.zig` file with three phases**

Each phase is a function: `runStage1()`, `runBridge()`, `runStage2()`.
They take/return `Buffer` handles. Simple, direct.

**Option B: Refactor each driver into a library function, call from `inference.zig`**

Extract `denoise_stage1.main()` into `stage1.run(platform, ckpt, inputs) → Buffers`,
same for bridge and e2e. This preserves the existing drivers and adds a thin
orchestrator. More modular but requires refactoring existing code.

**Recommendation: Option A** — Write `inference.zig` as a new file, extracting the
core logic from each driver into functions. The existing per-stage drivers remain
untouched for standalone use and debugging. The new `inference.zig` will share
`model.zig` but not import code from the other drivers.

### 4. What about compilation time?

The unified binary compiles ~23 XLA executables total:
- Stage 1: 11 exes (but some are at different shapes than Stage 2)
- Bridge: 5 exes
- Stage 2: 7 exes

However, Stage 1 and Stage 2 `forwardPreprocess` / `forwardBlock` / `forwardOutputProjection`
are compiled at **different input shapes** (T_v1=6144 vs T_v2=24576), so they
can't share executables.

**Optimization: Compile lazily per-phase.** Only compile Stage 1 executables
before the Stage 1 loop, bridge executables before bridge, etc. This keeps peak
memory during compilation lower and avoids compiling Stage 2 exes while Stage 1
is still running.

### 5. Sigma schedule handling

- Stage 1: Dynamic — `model.computeSigmaSchedule(30, ...)` → 31 f32 values
- Stage 2: Loaded from `pipeline_meta.json` → `stage2.sigmas` array (e.g. 4
  values for 3-step distilled). The denoising loop count is `sigmas.len - 1`.

The bridge uses `sigmas[0]` as the noise-init sigma (formerly a separate
`sigma_0` field, now derived from the schedule). Both stages' sigma schedules
live in `PipelineMeta.stage1.sigmas` / `PipelineMeta.stage2.sigmas` as `[]const f32`.

Stage 1 currently still uses `model.computeSigmaSchedule()` (a Zig function)
rather than reading `stage1.sigmas` from the JSON. The computed and JSON values
should match.

### 6. Checkpoint stores

Current approach per-driver: open registry + store with `defer .deinit()`.

In the unified binary, stores should be opened and closed per-phase to avoid
holding all three checkpoints' metadata in memory:

```
// Phase 1
var s1_store = openStore(stage1_ckpt_path);
... run stage 1 ...
s1_store.deinit();  // explicit, not defer

// Phase 2
var up_store = openStore(upsampler_ckpt_path);
var main_store = openStore(stage2_ckpt_path);
... run bridge ...
up_store.deinit();
main_store.deinit();

// Phase 3
var s2_store = openStore(stage2_ckpt_path);
... run stage 2 ...
s2_store.deinit();
```

Note: `stage2_ckpt` is opened twice (bridge for per_channel_stats, then Stage 2
for transformer weights). This is fine — the file is mmapped, so the OS
deduplicates automatically.

### 7. Noise init: bridge vs. Stage 2

Currently, **both** bridge and denoise_e2e run `forwardNoiseInit`:
- Bridge: `noised = clean*(1-σ) + noise*σ` → writes to safetensors
- denoise_e2e: reads clean+noise+mask back from safetensors → runs noise init again

The result is identical. In the unified binary, **noise init runs only once**
(in the bridge phase). Stage 2 receives the already-noised `video_latent` and
`audio_latent` buffers directly.

This means Stage 2's denoising loop starts with the noised buffers instead of
calling noise_init_exe. Saves 2 compilations + 2 kernel launches.

## Implementation Plan

### Step 0: Preparation (refactor shared helpers)

Extract duplicated helpers into a shared utility (or just duplicate them in
inference.zig, matching the codebase pattern):
- `loadBuf(allocator, io, platform, store, name, sharding) → Buffer`
- `sigma1dBuffer(io, platform, sigma, sharding) → Buffer`
- `writeBuffer(allocator, io, buf, dir, filename)`

### Step 1: Create `inference.zig` skeleton + CLI

New file with:
- `CliArgs` struct with all unified arguments
- `parseArgs()` using named flags
- `pub fn main()` that parses args, loads `pipeline_meta.json`, initializes platform

**Validation:** Builds, runs, prints args and metadata.

### Step 2: Implement `runStage1()`

Extract Stage 1 logic into:
```zig
const Stage1Result = struct {
    v_latent: zml.Buffer,       // [1, T_v1, 128] bf16
    a_latent: zml.Buffer,       // [1, T_a,  128] bf16
    v_context_pos: zml.Buffer,  // [1, S, 4096] bf16 — kept for Stage 2
    a_context_pos: zml.Buffer,  // [1, S, 2048] bf16 — kept for Stage 2
};

fn runStage1(
    allocator: Allocator,
    io: anytype,
    platform: *zml.Platform,
    sharding: anytype,
    ckpt_path: []const u8,
    inputs_path: []const u8,
    use_bf16_attn: bool,
) !Stage1Result
```

This function:
1. Opens Stage 1 checkpoint store
2. Opens inputs store, loads input buffers
3. Compiles all 11 Stage 1 executables
4. Runs the 30-step denoising loop
5. Frees all executables, weights, and unused input buffers
6. Closes stores
7. Returns the final latent buffers **and** the positive text context buffers
   (which are already on GPU and will be reused by Stage 2)

**Validation:** Run unified binary, compare Stage 1 outputs against standalone
`denoise_stage1` outputs. Should be bitwise identical (same code path, same
inputs, same hardware).

Validation command:
```bash
# Run standalone Stage 1 (reference)
bazel run //examples/ltx:denoise_stage1 -- $S1_CKPT $OUT/stage1_inputs.safetensors $OUT/stage1_ref/

# Run unified (dump intermediates with --dump-stage1)
bazel run //examples/ltx:inference -- ... --dump-stage1 $OUT/stage1_test/

# Compare
diff <(xxd $OUT/stage1_ref/video_latent.bin) <(xxd $OUT/stage1_test/video_latent.bin)
diff <(xxd $OUT/stage1_ref/audio_latent.bin) <(xxd $OUT/stage1_test/audio_latent.bin)
```

### Step 3: Implement `runBridge()`

Extract bridge logic into:
```zig
const BridgeResult = struct {
    // Noised latents (Stage 2 starting point)
    v_latent: zml.Buffer,       // [1, T_v2, 128] bf16
    a_latent: zml.Buffer,       // [1, T_a,  128] bf16
    // Positions and masks (needed by Stage 2 preprocess)
    v_positions: zml.Buffer,    // [1, 3, T_v2, 2] bf16
    a_positions: zml.Buffer,    // [1, 1, T_a,  2] f32
    v_mask: zml.Buffer,         // [1, T_v2, 1] f32
    a_mask: zml.Buffer,         // [1, T_a,  1] f32
    // Text contexts (positive only, for distilled Stage 2)
    v_context: zml.Buffer,      // [1, S, 4096] bf16
    a_context: zml.Buffer,      // [1, S, 2048] bf16
};

fn runBridge(
    allocator: Allocator,
    io: anytype,
    platform: *zml.Platform,
    sharding: anytype,
    s1_video: zml.Buffer,       // ownership transferred in
    s1_audio: zml.Buffer,       // ownership transferred in
    v_context: zml.Buffer,      // from Stage 1 (already on GPU)
    a_context: zml.Buffer,      // from Stage 1 (already on GPU)
    upsampler_ckpt_path: []const u8,
    main_ckpt_path: []const u8,
    noise_path: []const u8,     // stage2_noise.safetensors (from Python export)
    meta: PipelineMeta,
) !BridgeResult
```

This function:
1. Runs unpatchify → upsample → patchify (s1_video consumed, freed internally)
2. Computes positions/masks on host
3. Loads Stage 2 noise from `stage2_noise.safetensors` (pre-drawn by Python export)
4. Reuses text context buffers (passed in, already on GPU — no file I/O)
5. Runs noise init
6. Frees upsampler weights and intermediates
7. Returns all buffers needed by Stage 2

**Validation:** Run unified bridge against standalone bridge output. Use the
existing `compareRef` logic or the Python `validate_mixed_pipeline.py` script.

### Step 4: Implement `runStage2()`

Extract Stage 2 logic into:
```zig
const Stage2Result = struct {
    v_latent: zml.Buffer,  // [1, T_v2, 128] bf16
    a_latent: zml.Buffer,  // [1, T_a,  128] bf16
};

fn runStage2(
    allocator: Allocator,
    io: anytype,
    platform: *zml.Platform,
    sharding: anytype,
    ckpt_path: []const u8,
    bridge: BridgeResult,    // ownership transferred in
    use_bf16_attn: bool,
) !Stage2Result
```

**Key difference from standalone denoise_e2e:** This function receives
already-noised latents (from bridge), so it does NOT call `forwardNoiseInit`.
It starts the Euler loop directly with `bridge.v_latent` / `bridge.a_latent`.

**Validation:** Run full unified pipeline, compare Stage 2 outputs against
the standalone pipeline (denoise_stage1 → bridge → denoise_e2e). Should be
bitwise identical.

### Step 5: Wire up `main()` and add optional intermediate dumps

```zig
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = try parseArgs(...);
    const meta = try loadPipelineMeta(...);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    const sharding = try zml.sharding.replicatedSharding(platform);

    // Phase 1: Stage 1
    std.log.info("=== Stage 1: 30-step guided denoising ===", .{});
    var s1 = try runStage1(allocator, io, platform, sharding,
        args.stage1_ckpt, args.stage1_inputs, args.bf16_attn);

    if (args.dump_intermediates) {
        try writeBuffer(allocator, io, s1.v_latent, args.output_dir, "stage1_video_latent.bin");
        try writeBuffer(allocator, io, s1.a_latent, args.output_dir, "stage1_audio_latent.bin");
    }

    // Phase 2: Bridge
    std.log.info("=== Bridge: upsample + prepare Stage 2 inputs ===", .{});
    var bridge = try runBridge(allocator, io, platform, sharding,
        s1.v_latent, s1.a_latent,
        s1.v_context_pos, s1.a_context_pos,  // stay on GPU, no file I/O
        args.upsampler_ckpt, args.stage2_ckpt,
        args.stage2_noise, meta);

    // Phase 3: Stage 2
    std.log.info("=== Stage 2: 3-step distilled denoising ===", .{});
    var s2 = try runStage2(allocator, io, platform, sharding,
        args.stage2_ckpt, bridge, args.bf16_attn);

    // Write final outputs
    try writeBuffer(allocator, io, s2.v_latent, args.output_dir, "video_latent.bin");
    try writeBuffer(allocator, io, s2.a_latent, args.output_dir, "audio_latent.bin");
    s2.v_latent.deinit();
    s2.a_latent.deinit();
    platform.deinit(allocator);
}
```

### Step 6: Update BUILD.bazel

```starlark
zig_binary(
    name = "inference",
    main = "inference.zig",
    srcs = ["model.zig"],
    deps = ["//zml"],
)
```

Add to the build test group.

### Step 7: End-to-end validation

Run the full unified pipeline and compare against the multi-binary pipeline:

```bash
# Multi-binary reference
denoise_stage1 → bridge → denoise_e2e → decode_latents.py → ref_output.mp4

# Unified binary
inference → decode_latents.py → test_output.mp4

# Compare
diff <(xxd $OUT/ref/video_latent.bin) <(xxd $OUT/test/video_latent.bin)
diff <(xxd $OUT/ref/audio_latent.bin) <(xxd $OUT/test/audio_latent.bin)
```

Outputs will **not** be bitwise identical due to GPU floating-point
non-determinism in CUDA attention kernels. Even running the standalone driver
twice produces different results (cosine sim ~0.97 after 30 steps). The
comparison validates that the unified pipeline divergence is within the same
range as standalone-vs-standalone divergence. See "Validation Results" below.

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU OOM from overlapping weights | Pipeline fails to run | Staged loading: free weights between phases |
| Compilation latency (23 exes total) | Slow first run | Compile per-phase; XLA caching helps subsequent runs |
| Non-determinism between unified and standalone | Validation requires statistical comparison | Confirmed: cosine sim ~0.97 matches standalone-vs-standalone baseline |
| Code duplication from existing drivers | Maintenance burden | Keep standalone drivers as-is; `inference.zig` is separate. Shared logic lives in `model.zig` |
| Refactoring risk to existing working code | Breaks existing drivers | **No changes** to `denoise_stage1.zig`, `bridge.zig`, or `denoise_e2e.zig` |

## Summary

The unified binary is a straightforward orchestration of three phases that
already work independently. The main engineering is:
1. **Buffer ownership**: pass `zml.Buffer` handles between phases instead of file I/O
2. **Staged weight management**: load and free checkpoint weights per-phase
3. **Skip redundant work**: noise_init in bridge replaces noise_init in Stage 2
4. **Unified CLI**: merge the three drivers' argument sets

No changes to `model.zig` or the existing drivers. The unified binary is a new
file that is validated by comparison against the existing multi-binary pipeline.

## Validation Results

### Observation: GPU floating-point non-determinism

The original plan expected **bitwise identical** outputs between the unified and
standalone pipelines. In practice, outputs differ due to **GPU non-determinism**
in CUDA attention kernels (non-deterministic reduction order in dot-product
accumulation). This is not a bug in the unified pipeline — it affects the
standalone driver equally.

### Stage 1 video latent comparison

**Unified vs standalone** (`stage1_video_latent.bin`, 786,432 bf16 values):
```
cosine sim:     0.97078294
max abs diff:   3.873047
mean abs diff:  0.088169
```

**Standalone run 1 vs standalone run 2** (same binary, same inputs):
```
cosine sim:     0.96666360
max abs diff:   3.916992
mean abs diff:  0.099800
```

The unified-vs-standalone divergence (~0.971 cosine) is **within the range** of
standalone-vs-standalone divergence (~0.967 cosine), confirming the unified
pipeline is numerically correct. The gap is purely GPU FP non-determinism
compounding over 30 steps × 4 passes × 48 blocks = 5,760 sequential kernel
invocations.

### Visual validation

The decoded video from the unified pipeline output is visually correct and
quasi-indistinguishable from the multi-binary pipeline output. Minor visual
variations (e.g. color shifts) can occur between any two runs due to the
accumulated GPU non-determinism, but the overall scene content is the same.
