# Loader simplification plan (sixteenth pass)

`CTX.md` is the authoritative description of the loader's design and measured
behaviour. This file is the execution checklist for the simplifications that
survived the 2026-09-10/11 review (ten scoped finders, 44 candidates, three
adversarial verifiers per candidate: correctness and callers, recorded
evidence in `CTX.md`, net simplicity; the candidate ids `Cnn` below are the
review's). After every task:

1. run the task's validation;
2. update the status here;
3. update `CTX.md` so it describes the code that now exists (the "Current
   design" bullets named in each task, plus a "Sixteenth pass" section that
   accumulates the results);
4. do not begin the next task until those updates are complete.

Rules that hold for every task:

- No behaviour change on TPU/neuron/metal (the buffered backend,
  `zml/io/buffered_loader.zig`); every task below states why it is provably
  untouched there.
- Line numbers refer to commit `65aca3e4` ("remove adaptive source width");
  re-grep after each landed task.
- The user commits with `jj`; leave the working copy uncommitted after a task.
- The throttle watch stays until group D lands. The review's item to delete
  it outright (C08) is withdrawn: the user requires rate limits to reduce
  traffic, and the replacement is the VFS-owned governor of group D (tasks
  23 to 28), which removes the watch in task 26. Group B tasks that touch
  the watch keep it working on the read gate alone if they land first.

## Validation

Run after every task, from `~/github/zml/zml`:

```
~/.cache/bazel/_bazel_brabier/ed6f309db0cabd683780d1f69fd3a15d/external/rules_zig++zig+zig_0.16.0_x86_64-linux/zig fmt --check <changed files>
bazel test //zml:test //vfs:test
bazel build //examples/io //examples/llm //examples/mnist
```

CPU playground, at the end of group A and after every group B task
(Qwen3.5-4B sharded over four CPU devices, two packs, read-back checks):

```
ZML_LOAD_PACKS=2 ZML_LOAD_PACK_WIDTH=16 ZML_LOAD_CHECK=64 bazel run --config=release //examples/io -- load /var/models/Qwen/Qwen3.5-4B sharded
```

Expect `live loader ready: ... source_width=16, lifecycle_credits=33,
workers=17`, `pregrown=264 MiB`, one admission retire, `pack check: ok`,
`load check: ok`, `Loaded weights` within 3.5 to 3.8 s (fifteenth pass:
3.64 s).

gb300-2, after every group B task: the pinned detached worktree
`~/github/zml/zml-directio` with an overlay tarball of the changed files
(procedure and scripts in CTX.md "Fifteenth pass", verification table), GPU
chosen from `nvidia-smi`, Llama-3.1-8B-Instruct, three `ZML_LOAD_PACKS=64
ZML_LOAD_PACK_WIDTH=16` runs and one with `ZML_LOAD_CHECK=16`. Compare the
loader `elapsed` (fifteenth pass: 0.27 to 0.33 s packs plus bulk, 0.27 to
0.28 s bulk only, pinned 400 MiB at width 16), not the `Loaded weights` wall,
which includes calibration. Tasks that touch admission add the
`ZML_GPU_MEMORY_FRACTION=0.08` run (expect retires in the debug log, no OOM,
`pack check: ok`, non-zero `min_room_seen`).

## Group A: surface trims (no device run needed between tasks)

## Group B: mechanism changes (CPU playground plus gb300-2 after each)

## Group C: sound, but needs a measurement or a decision first

- [ ] 19. Merge `ReadyTransfer` and `EventContext` into one plan-owned node
  (C15, about 50 lines). One `Submission` node per DMA piece in the plan:
  target, block, source and destination offsets, len, `pjrt_event` (null
  until submitted and once destroyed), `err`, `submitted_ns`, `next`; the
  worker fills and links it at the tail of the pump's intrusive ready list
  (`ready_head`/`ready_tail` under the pump mutex), the pump pops (nulling
  `next`), submits and registers the callback, the callback relinks the node
  into `retired`; `abortReady` reads `next` before `block.complete()`. Let the
  planner store `Job.event_start` beside `block_start` so `events_used`
  goes. Deletes `ReadyTransfer`, `ReadyQueue`, the deques, `ready_entries`,
  `abandonSubmissions`, the catch in `ReadRequest.run`, `Scratch.queue_counts`
  (a device mask suffices for kicks). Keep `err` destroyed by the pump or
  `retireBatch`, never inside the callback (CTX 439-447, 3380-3387); keep
  `lockAllPumps` in `enqueueBlocks`. Required validation before landing: one
  gb300-2 DeepSeek-replicated pair against the per-device-pump arm (CTX
  1595-1610) and a `ZML_LOAD_CHECK` read-back on a multi-device fixture;
  tests `:3706-3745`, `:3804-3850` rewritten on the node.

- [ ] 21. DMA block override instead of the screen (C31 amended, decision).
  A per-target block table is refuted (CUDA hosts disagree: 2 MiB on the
  RTX 5090 host, 16 MiB on gb300-2, CTX 878-884, 1041). What the record
  supports: `Loader.Options.dma.block_size: ?usize = null` that skips the
  screen when set (still warming allocators and pre-growing), giving llmd its
  0.3 to 1.1 s of startup back on a known host (half of the open item at CTX
  3362-3364), plus demoting the confirmation tunables
  (`confirmation_duration_ns`, `confirmation_minimum_transfers`,
  `confirmation_margin`, `dma_calibration.zig:52-56`) to private constants.
  Needs the user's decision on the knob; then a playground `ZML_DMA_BLOCK_MIB`.

- [ ] 22. Drop `LoadProfile.direct_io_alignment` for the VFS constant (C11).
  Blocked on the llmd migration adding `vfs.loadProfile` and on deciding
  whether a profile-less VFS caller gets `.auto`; as written it would open
  remote files on the submit task (`direct_loader.zig:977-984`).

## Group D: rate limits and retries move into the VFS

Decided with the user on 2026-09-11: the VFS owns rate-limit handling; the
scope is one governor per backend instance for now, keyed so that a
per-authority scope is a local change later; retries and the hold apply to
every HTTP request a backend makes, not only to range reads. Three designs
were written and each attacked by two critics that day; placement survived
unanimously, an AIMD permit ladder and additive recovery did not (unmeasured
policy of the class the fifteenth pass deleted; the ladder collapses to one
permit on a burst because permits are released per attempt). The
choices below that the review left open are taken here as defaults and can
be changed in `InitOpts` without touching the mechanism.

Mechanism, in one paragraph. Every request on a backend goes through one
loop: wait out the backend's hold, send, classify. A throttle (429
everywhere; 503 on S3 and GCS, whose `unavailable` is `.throttle`) arms or
extends a backend-wide hold and is not charged to the request's retry
budget; every other retryable failure (connect and receive errors, 408,
other 5xx) keeps today's per-request backoff and `max_retries`. Timeouts
never arm a hold. A hold is a deadline: nothing is narrowed permanently,
recovery is the deadline passing, and the loader's width stays an immutable
ceiling because the loop runs inside the loader's read gate. A backend that
is throttled continuously for longer than `throttle_budget` fails its
requests with `error.RateLimited`, which the loader treats as any other
sticky failure. Cancellation propagates through the wait.

Defaults taken (each an `InitOpts` field of the four backends unless noted):

- `max_retries = 5`, `retry_initial_delay = 500 ms`, `retry_max_delay = 30 s`
  as today; they bound the non-throttle backoff only.
- `max_hold = 2 min`: the longest hold, applied to a server-named delay as
  well (today a `Retry-After` is honoured uncapped by the one request that
  received it). Loader teardown does not cancel workers, so `destroy` can
  park for up to `max_hold` after a failure; today's worst case is five
  30 s sleeps. Revisit with `worker_group.cancel` once the pipeline's
  cancellation points are audited.
- `throttle_budget = 5 min`: an episode of throttling with no clean window
  longer than this fails with `error.RateLimited`.
- The hold floor is `retry_initial_delay`: `Retry-After: 0` and HF's
  `RateLimit; t=0` parse to zero (`range_read.zig:262-265`) and a zero hold
  is a hot loop once throttles stop charging retries.
- Waiters wake at the deadline plus a per-waiter jitter in
  `[0, retry_initial_delay)`, so the readers do not fire in one instant.
- Scope key: the URI authority (`host:port`) of the request. Today
  `Governor.holdFor(key)` ignores it and returns the single hold; per
  authority is a map inside that function. One generic `HTTP` instance
  serves every host of its scheme and one `S3` spans buckets, so the
  per-instance scope over-reaches on those two; recorded as the reason to
  flip the key later.
- The buffered backend (TPU, neuron, metal) reads through the same loop
  and gains the hold with zero lines changed in `buffered_loader.zig`; its
  happy path is one inert deadline check per attempt. This is a behaviour
  change under throttling on hosts that cannot be tested here and needs
  the user's sign-off before task 24 lands.

Validation for the group, after tasks 23 to 27: `zig fmt --check`,
`bazel test //vfs:test //zml:test`, the three example builds; one hf://
Qwen3.5-4B load on the CPU playground (`bazel run --config=release
//examples/io -- load hf://Qwen/Qwen3.5-4B sharded`) expecting
`source_holds=0`, `source_throttles=0` on the batch source line and parity
with the last hf:// recording in CTX.md "Fifteenth pass"; the CPU playground
and gb300-2 runs of the group B validation are unaffected (local profiles
have no governor traffic) and need no repeat.

- [ ] 27. One end-to-end test through the loader. Move `MockServer` to
  `vfs/mock_server.zig`, exported as `VFS.MockServer` behind
  `builtin.is_test`, so `zml/io/loader.zig`'s test section can serve the
  fixture's safetensors file over the `http` VFS backend: `HTTP.init(...,
  .http)`, `vfs.registerBackend("http", ...)`, the store built with
  `TensorRegistry.fromRepo` on a `Dir` opened through `vfs.io()` at
  `127.0.0.1:port/` (the mock answers 404 for the index and serves
  `model.safetensors`). Feasible as checked on 2026-09-11: `HTTP.dirOpenDir`
  exists (`http.zig:195`), `HTTP.dirRead` is a stub (`:264`) that nothing
  on this path calls, and `resolveModelEntrypoint`
  (`zml/safetensors.zig:1066-1080`) only probes the two names by `openFile`,
  which the path table answers; the HEAD at open goes through the governor
  too, so a throttled HEAD is covered by the same test.
  Direct backend on CPU, `read_parallelism = 4`, the window limiter at 2
  GETs per 100 ms: the load completes, the buffers match, `holds > 0` on
  the backend's stats, the loader's `request_gate.inUse == 0` after
  `awaitAll`, and pinned `high_water` equals the fifteenth-pass fixture
  value (the hold lengthens the time credits and blocks are held; it does
  not raise the count). If `fromRepo` cannot be driven through the HTTP
  backend, replace the store with the fixture's file served whole and
  keep the rest of the assertions.

- [ ] 28. Record: CTX.md "Current design" bullets at 154 (the side channel
  is observability, not control), 177 (no throttle watch), 218-240 (the
  governed loop, the two budgets, the hold), 404-432 (the width is
  immutable; a throttled source holds in the VFS), and the "Sixteenth
  pass" section (the design, the defaults, the tests that exercised the
  hold, the hf:// parity number, the open decisions: per-authority key,
  teardown cancellation, the TPU sign-off); `docs/learn/loader.md`
  implementation map (`request.zig` row); the memory file.

## Withdrawn or judged not worth it after verification

- C08 delete the throttle watch: withdrawn (see the rules above). The
  replacement, rate-limit handling inside the VFS (hold on the server-named
  delay, reduction, recovery, retry budget), is a separate design.
- C17 one job-length rule: net two lines, adds a same-named function with
  different semantics.
- C20 inline `Backend.initBuffered`: zero lines, adds a front-end import of
  the buffered backend.
- C32 `Submission` as a `Batch` union: 18 lines become 19.
- C05, C11 (as written), C12, C13, C22, C27 (without the A/B), C29, C31 (as
  written), C33 to C44: rejected; the decisive reasons are in CTX.md
  "Sixteenth pass" once recorded (batch diagnostics are consumed by CTX
  tables; `.auto` is used by `zml/testing.zig:465`; eager file opens move hf
  and s3 HEAD chains onto the submit task; `validateExecutableSharding`
  guards `device.id` indexing; single-binding `loadExecute` loses Laguna's
  per-layer pack coalescing; the read gate is the watch's actuator; direct
  I/O demotion protects the streaming fallback).

## What the llmd migration needs from this API (do not remove)

`Loader.init(allocator, io, platform, .{ .load_profile = try
vfs.loadProfile(model) })` (without the profile a remote source gets width
16, no direct I/O and no throttle watch); `load`, multi-binding `loadExecute`,
`awaitAll`, `bytesLoaded()`, `deinit`; several stores and sharding sets per
loader (dflash); the `delivered` set and `TransformedTensorNotDelivered`;
`Binding.transformed` on single-source bindings; the TensorStore view API;
defaults that work without knobs (`direct_io = .auto`, no host budget, no DMA
options); `calibration()` optional (playground only).
