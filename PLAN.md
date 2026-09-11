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

- [ ] 5. Dead TensorStore accessors (C03, 35 lines).
  Delete `getReaderById` (`zml/io/TensorStore.zig:40-45`), `View.parent`
  (`:72-82`), `View.getShapeOpts` (`:170-178`), `getPtrFromId` (`:262-266`).
  No caller in zml, examples, tests or llmd (llmd uses `withPrefix`,
  `withLayer`, `createTensor`, `maybeCreate*`, `hasKey`, `count`, `prefix`,
  `getShape`, `getSourcesById`, `getReader`). Keep `pub const limits` in
  `zml/io.zig` (cited by `loader.zig:89-90` and `loader.md:131`).

- [ ] 6. Log-only metrics and the `call_count` parameter (C18, 10 lines).
  Remove `Metrics.source_calls` and `Metrics.transfer_pieces`
  (`zml/io/direct_loader.zig:2404-2405`), the `fetchAdd` at `:2069`, the
  `physical_source_calls` and `tensor_transfer_pieces` fields of the summary
  lines (`:493-499`, `:513-514`). Then delete the
  `call_count: ?*std.atomic.Value(u64)` parameter of
  `safetensors.readFilePositionalAllV` (`zml/safetensors.zig:23-24` doc,
  `:35`, `:68`) and its arguments at `safetensors.zig:295` and
  `direct_loader.zig:1716`. Keep `read_operations`, `read_bytes`,
  `dma_submissions` and every timer and pump counter. CTX 235-236: physical
  request counts for remote sources come from the VFS `batch source` line;
  local profiles have none.

- [ ] 7. Derivable pipeline state (C06, 12 lines).
  `zml/io/direct_loader.zig`: remove `ReadRequest.completed` (`:1623`, the
  `idle` constant `:1635`, the store at `:1773`) and assert
  `request.pending.load(.acquire) == 0` at `:2004`; remove
  `EventContext.pipeline` (`:1875`, init `:2169`) and read
  `ctx_.block.pipeline` in the callback (`:2187`) and
  `self.block.pipeline.platform.pjrt_api` in `destroyEvent` (`:1889`,
  `:1891`); remove `DevicePump.ready_entries` (`:1837`, `:2067`, `:2118`,
  `:2241`) and assert `device_pump.queue.len == 0` at `:1951`. Tests: delete
  `:2595` (the pending check at `:2596` stays), `:2926` becomes `pending ==
  0`, delete `:3727`, `:3736` becomes `queue.len == 0`, `:3738` becomes
  `pending == 0`, drop `.pipeline = pipeline,` at `:3823`. CTX 356: drop
  "completed". If task 18 (C15) lands later, only the `completed` removal of
  this task remains relevant.

- [ ] 8. `allocatedBytesPerDevice` without the bool (C23, 6 lines).
  `zml/io/backend.zig:85-96` becomes a `void` method reading the direct
  payload: `for (out, self.direct.allocated_bytes) |*bytes, *counter| bytes.*
  = counter.load(.acquire);`, doc: only the direct backend counts and the
  front end never asks the buffered one. `zml/io/loader.zig:188` becomes
  `if (self.backend != .direct) return false;` (before any
  `device.memoryStats()` call, so CPU/TPU stats behaviour is unchanged);
  `:388` becomes the plain call; test `:1024` becomes a plain call before its
  two `expectEqual`s. The per-device atomic counters and their refresh in
  `readRoom` stay (fourteenth-pass room numbers).

- [ ] 9. One no-VFS load profile (C21, 12 lines).
  Delete `LoadProfile.default` (`vfs/vfs.zig:66-75`) and make
  `LoadProfile.local` (8 MiB, not high latency, null alignment, null stats)
  the `Loader.Options.load_profile` default (`zml/io/loader.zig:97`; reword
  the comment: the no-VFS local profile, prepare one with `VFS.loadProfile`
  for a VFS path). Leave `vfs.loadProfile` as it is (it already patches
  `.local`). `examples/llm/models/llama_tests.zig:54-56` and
  `lfm2_tests.zig:56-58` may drop their explicit `.load_profile = .local`.
  CTX 2545: name the one constant. Evidence: B70 local 8/16/32 MiB = 27.05 /
  24.21 / 21.33 GiB/s with 2x pinned high-water at 16 (CTX 546-549). TPU: the
  buffered backend reads `read_chunk_size` only when `high_latency` is true
  (`buffered_loader.zig:65`, `:157-162`, `:352-358`), so profile-less loads
  there are byte-identical.

- [ ] 10. `backend.Config` becomes the one `Options` (C14, 18 lines).
  Move the body of `Loader.Options` (five fields, defaults, `auto`, docs) to
  `zml/io/backend.zig` as `pub const Options`, add
  `pub fn readWidth(self: Options) usize { return self.read_parallelism
  orelse limits.defaultReadParallelism(self.load_profile.high_latency); }`,
  delete `Config` (`backend.zig:14-24`). In `loader.zig`: `pub const Options
  = backend.Options;`, `init` passes `opts` through (`:139-145`), the fixture
  at `:788` uses `opts.readWidth()`. `Backend.init`, `initBuffered` and
  `direct_loader.Loader.create` / `Sizing.init` take `backend.Options`;
  `direct_loader.zig:90` and `:324` call `opts.readWidth()`; the two
  direct_loader tests (`:2611-2616`, `:2657-2662`) drop `.dma = .{}` if it
  is now the default. `buffered_loader.Loader.create` keeps its
  `(allocator, io, platform, read_parallelism: usize, profile)` signature and
  receives `opts.readWidth(), opts.load_profile`: byte-identical values on
  TPU. Fix the stale `Loader.backendFor(target)` in `docs/learn/loader.md:105`.

- [ ] 11. Drop the `max_host_bytes` knob, keep the guard (C24, 18 lines).
  Delete `Loader.Options.max_host_bytes` (`loader.zig:101-102`, `:143`) and
  the backend option field; replace `Workspace.Options`
  (`zml/io/host_memory.zig:313-317`) with a constant 16 GiB ceiling on
  `Workspace` (keep the "safety guard, not an allocation target" comment on
  the field at `:322`) and make `Workspace.init(allocator, io, platform)` use
  it; delete `minimum_mapped_bytes` (`:16`) and its check (`:339-340`).
  `direct_loader.zig:296-298` calls the three-argument init; remove the
  `.max_host_bytes = 64 MiB` lines from the tests at `:2615` and `:2661`
  (56 MiB of pre-growth fits either ceiling; width, credits and workers are
  unchanged). Drop `ZML_DMA_BENCH_MAX_MAPPED_MIB` (`examples/io/main.zig:173`;
  never used in a recording). Retarget the test at `loader.zig:1030-1044`
  to an invalid profile (`.local` with `direct_io_alignment = 3`,
  `.direct_io = .on`, expect `error.InvalidLoadProfile`) so the post-Sizing
  errdefer path stays covered. `docs/learn/loader.md:115-117`: the direct
  backend caps its pinned arenas at a fixed 16 GiB. Task 16 (C04) later
  documents that a budget below the pre-grown set narrows the width.

- [ ] 12. Dead Workspace and BlockPool surface (C10, 27 lines).
  `zml/io/host_memory.zig`: delete `newly_mapped_bytes` and
  `unused_tail_bytes` (`:464-465`, increments `:605` and `:624`, assertions
  `:858`, `:1004`, `:1011-1012`, `:1039`, the "reported unused" clause in the
  `BlockPool.init` doc `:470-471`); delete `pub const Error = anyerror`
  (`:428`) and make `acquireMany` return `!void`; remove the
  `enumerated_bytes` accumulator and its `!= mapped_bytes` return in
  `BlockPool.init` (`:494-500`), keeping the `block_size == 0 /
  max_mapped_bytes` check and the zero-length arena check in `attachArena`.
  Keep `findArena` (two production callers: `dma_calibration.zig:169`,
  `:520`) and `initForTesting`. `Workspace.Options` went with task 11.

- [ ] 13. host_memory backend union (C09, 20 lines).
  `Backend = union(enum) { pjrt_host: PjrtHost, pages: Pages }`
  (`zml/io/host_memory.zig:25-28`); `HugePageAllocator.init` takes
  `?*const Platform` and `initPageable` (`:216-230`) goes; `Backend.init`
  maps cuda/oneapi to a platform and cpu to null; the four per-variant
  switches (`:113-139`, `:179-199`) fold; `place()` sets `self.numa_mask =
  0` after the warning and `leaveUnplaced` (`:275-277`) and its test
  (`:724-728`) go; `initForTesting` becomes `.pages` with a null platform.
  Keep the arena log's `kind=` string byte-identical ("dma_map" when
  `pages.allocator.platform != null`, else "pageable"; CTX 2105-2106) and
  keep the `.tpu, .neuron, .metal => error.DmaBenchmarkUnsupported` arm at
  `:109` (a recoverable error, not `unreachable`).

- [ ] 14. DispatchSpans in one pass (C16, 35 lines).
  `zml/io/DispatchSpans.zig`: append with `try` in one recursion; delete
  `placementSpanCount` (`:178-188`), `appendPlacementSpan` (`:129-135`), the
  count pre-pass (`:31-46`) and the count assert; the two
  `NonContiguousShardPlacement` returns (`:101`, `:126`) become
  `std.debug.assert` with a comment that `Planner.appendTransfers`
  (`direct_loader.zig:1287-1330`) already relies on the spans tiling
  `[0, byteSize)`, and that `Placement.init` tiles by `@divExact`
  (`zml/Sharding.zig:1811-1841`); delete the error and the gaps/overlaps test
  (`:230-248`). `init`'s error set is `OutOfMemory` plus `Sharding.Error`.
  Keep `deduplicateByRange`, the mirrored masks and the axis recursion
  (llmd's column-parallel and the 2x4 MI300X mesh need them).

- [ ] 15. Admission helpers without the boundary ceremony (C28 narrow, 30 lines).
  Keep `zml/io/execute_admission.zig` and the `Fit` enum (the `.unmeasured`
  path after init is real). Delete `DeviceStats` (`:13-17`): `room` takes
  `(limit: ?u64, in_use: u64, submitted: u64, allocated: u64, reserve: u64)
  ?u64` and its test's literals become positional; delete `roomPerDevice`
  and its test (`:38-44`, `:63-71`): `readRoom` in `loader.zig` computes the
  room per device inline from `device.memoryStats()` (replacing the copy
  loop at `:384-387` and the call at `:389`, returning false on a null
  limit); delete `scratch.stats` (field `:78`, alloc and errdefer `:159-160`,
  init `:174`, free `:311`); delete `Cost` (`:19-27`): `admits(rooms,
  pending, inputs, execution)` takes the slices positionally (calls at
  `:344`, `:379`). `min_room_seen` keeps updating only from devices with a
  limit. Validation adds the `ZML_GPU_MEMORY_FRACTION=0.08` gb300-2 run
  (CTX 3161-3165: expect retires, no OOM, `pack check: ok`).

## Group B: mechanism changes (CPU playground plus gb300-2 after each)

- [ ] 16. Lifecycle credits become a constant (C07, about 100 lines).
  After pre-growth, `retained = capacity / blocks_per_request >= width + 1 +
  floor(reserve / bpr) >= width + dma_stage`, so
  `RequestGateLimits.init` (`direct_loader.zig:2388-2399`) always yields
  `read = width`, `lifecycle = retained`, `workers = width + 1`; both
  recorded ready lines confirm it (CTX 3302: 33 = 17 + 16; CTX 3308: 25 =
  17 + 8) and the throttle test pins the lifecycle limit at 41 across
  halvings (`:3495`, `:3507`).
  In `create`: `width = @min(opts.readWidth(), retained - 1)` with `if
  (retained < 2) return error.DmaMappedBudgetExceeded` replacing the dead
  `feasible_width == 0` check (`:339`; `retained = pool.capacity /
  maximum_blocks_per_job`, already computed at `:347`); `read_gate =
  .init(width)`, `request_gate = .init(retained)`, `workers = width + 1`;
  `std.debug.assert(retained >= width + 1)`.
  Delete `RequestGateLimits` with `Config` and `at` (`:2373-2400`), the
  write-only `Loader.limits` (`:70`, `:91-95`, `:111`), `Sizing.feasible_width`
  and `dma_stage_requests` (`:286`, `:338-339`, `:348-353`), `dmaStageRequests`
  (`:2551-2557`) and its test (`:2714-2723`), `BlockPool.potentialRequestWidth`
  (`host_memory.zig:568-571`) and its three test lines (`:1040-1042`; keep the
  test's reserve-refusal and tail assertions), the lifecycle-gate test
  (`:3565-3583`).
  The throttle watch keeps `cursor`, `metrics`, `read_gate` and `width`;
  `tick` becomes `read_gate.setLimit(io, narrower)`; its `limits` and
  `request_gate` fields go; the "source throttled" log drops
  `lifecycle_credits`; the watch test (`:3469-3510`) checks the read gate and
  the width only. Rider (C26): spawn the watch into `worker_group` and delete
  `throttle_group` (`:58`, `:152`, `:437`), one await in `stopWorkers`.
  Keep the fourth-pass measurement (one credit beyond the width: 24.3 vs
  43.8 GiB/s, CTX 1050-1054) as the doc comment of the lifecycle gate field,
  with one sentence that the credits are the pre-grown capacity, which
  exceeds the width plus the calibrated DMA depth whenever the request size
  is a multiple of the block size. Ready line: print `lifecycle_credits` and
  drop `feasible_width`. `docs/learn/loader.md:132-133`: "clipped to what the
  pre-grown set holds". CTX "Current design" 396-414.
  Behaviour differs only in the clipped host-budget regime, unreachable with
  the 16 GiB ceiling; the gb300-2 numbers must reproduce (0.27 to 0.33 s,
  pinned 400 MiB). TPU: the buffered backend has no gates.

- [ ] 17. Fixed-size pinned pool (C04 amended, about 190 lines). After task 16.
  Size the pool once in `Sizing.init`: `fitted = @min(width, ((max_mapped -
  mapped) / block + usable - reserve) / bpr - 1)`, `error.DmaMappedBudgetExceeded`
  when it is 0 (no reserve-drop fallback, no "leave growth to the load");
  pre-grow as two arenas exactly as today (`growToBlocks` to the reserve,
  then to `(fitted + 1) * bpr + reserve`) so ROCm's per-node byte balance
  (`host_memory.zig:175-181`, CTX 1999-2005) is unchanged; the loader's width
  is `fitted` and the ready line logs it.
  Delete in `host_memory.zig`: `reserve`, `slab_blocks`, `default_slab_size`,
  `canEverAcquire`, `remainingBlockBudget`, `reservedGrowthBlocks`, `grow`,
  `allocateSlab` (`:461-466`, `:481-503` reserve check, `:517-532` grow loop,
  `:560-625`); merge `attachArena` and `attachArenaAssumeCapacity`;
  `acquireMany` = `RequestExceedsCapacity` if `output.len > capacity`, else
  wait on the condition until `free >= output.len` or closed (allocates
  nothing). Keep `Workspace.allocate`, `usableBlocks`, `growToBlocks`,
  `Lease`, `close`, `high_water`, the `DmaMappedBudgetExceeded` check in
  `allocate`. Delete `ensureLoadBlockReserve` and `ensureSourceWorkingSet`
  (`direct_loader.zig:2498-2549`) in favour of the two `growToBlocks` calls
  in `Sizing.init`; reject a request size that is not a multiple of the
  block size at init with `error.InvalidDmaLoadConfig` (task 16's constant
  credit relies on it; every shipped profile satisfies it).
  Tests: delete the metadata-retry (`:825-842`), free-list-capacity
  (`:883-900`), allocates-nothing (`:901-921`) and grows-on-demand
  (`:989-1016`) tests; rewrite the reblock test as reblock-only and the
  ownership-transfer test (`:798-824`) without the 4-block acquire; keep the
  never-fit (`:1045-1063`) and close (`:922-956`) tests; add "Sizing refuses
  a budget below the reserve plus two requests". Document that a ceiling
  below the pre-grown set narrows the width or refuses at init.
  Evidence: "nothing maps a slab inside a load" is the recorded win (CTX
  388-395, 419-425, 550-553, 955-957); every fifteenth-pass run shows
  `pinned_mapped == high_water` (CTX 3300-3311). Risk: the ROCm 41/59 split
  under sequential growth (commit `a2a0a9b8`) post-dates the last MI300X
  runs; record it and re-check one 8x MI300X Llama load when the host is
  available (`zml/io` CTX "mi300" rules: check the plugin first).

- [ ] 18. Scheduler as a FIFO of plans (C25, about 55 lines).
  `Scheduler.queue` holds `*Plan` (`direct_loader.zig:1434-1438`); `publish`
  appends the plan under the mutex, sets `plan.batch` there (not in
  `Plan.create`, which the planner and the allocation-failure test at
  `:2571-2600` call without a batch) and adds its completion units to the
  batch; `claim` hands out `queue[head]`'s next job and pops the plan when
  its cursor reaches `jobs.len`; `fail` retires each queued plan's unclaimed
  jobs through `plan.batch.finishJobs` as the last access to that batch;
  `seal` only stamps diagnostics (the sentinel already prevents completion
  before the seal); keep `std.debug.assert(batch.diagnostics.sealed_at ==
  null)` on publish. Derive `Claim.batch` and `ReadRequest.batch` (`:1617`,
  `:1631`, `:1769`) from `plan.batch`. Delete `Batch.plan_cursor`, `queued`,
  `sealed`, `claimJob`, `exhausted`, `retireUnclaimed`,
  `appendPlanAssumeCapacity` (`:648-660`, `:681-721`), the seal-time pop and
  the open-exhausted-head rule (`:1400-1403`, `:1489-1512`, `:1551-1556`).
  The ownership rule re-derives one level down: a plan is queued only with
  jobs, its units are added in the same critical section, it is popped on
  its last claim or cleared by `fail`, so a queued plan's batch cannot reach
  `done`. Ordering is unchanged because `submit` publishes every plan and
  seals before returning on one task (CTX 1136-1139). Tests `:3046-3100`,
  `:3168`, `:3323-3414` rewritten on `remaining` and `done`. Rewrite the
  Scheduler doc (`:1391-1403`) and CTX 331-350.

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

- [ ] 20. Drop `Planner.fairOrder` and the per-device charge queues (C27,
  about 225 lines). No recording justifies the byte-fair order either way:
  every multi-device number used axis-0 sharding where it is active, and the
  CPU pump is saturated (CTX 2185-2190). Measure first: force the planning
  order with a two-line local patch (take the `@memcpy(plan.jobs,
  planning_jobs)` branch unconditionally in `preparePlan`) and run
  interleaved warm A/Bs on the B70 four-CPU sharded Qwen3.5-4B and on
  four-B70 Llama (0.640 s anchor); gb300-2 and MI300X are not diagnostic.
  Delete only if the A/B is flat: `fairOrder` (`:1337-1403`), the queues and
  the per-job physical row in `preparePlan`, the charging loop and parameter
  of `appendTransfers`, `TensorPlan.device_indices`, the five fair-order
  tests and two helpers; `loader.md:87`.

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

- [ ] 23. The governed request loop (`vfs/request.zig`, new; `range_read.zig`
  keeps the Range specifics). Add `request.zig` to `VFS_SRCS` in
  `vfs/BUILD.bazel:4-10`.
  `RetryConfig` moves here and gains `max_hold` and `throttle_budget`
  (`fromOptions` reads the two new `InitOpts` fields; the four backends add
  them with the defaults above beside `max_retries` at `hf.zig:55-59`,
  `s3.zig:146-150`, `gcs.zig:207-209`, `http.zig:14-16`).
  `Governor` (one per backend, a field beside `read_stats`): `io` (the
  backend's inner io, for the clock and the sleep), `retry: RetryConfig`,
  `stats: *AtomicReadStats`, `mutex: std.Io.Mutex`, one `Hold {until:
  ?std.Io.Clock.Timestamp (.awake), length: Duration, consecutive: u32,
  episode_started: ?Timestamp, exhausted: bool, generation: u64}`.
  `pub const Key = []const u8` (the URI authority); `fn holdFor(self, key)
  *Hold` returns `&self.hold` with the comment that per-authority scope
  replaces this body with a map.
  `admit(self, key) error{Canceled, RateLimited}!void`: under the mutex run
  `refresh(now)` (below), fail with `RateLimited` when the episode is
  exhausted, read `until` and `generation`; unlock; if `until` is null or
  passed return; otherwise draw the jitter with `io.random`, `Clock.Timestamp
  .wait(until + jitter, io)` (cancelable, std `Io.zig:817`), add the wait
  to `stats.hold_wait_ns`, and loop, since the hold may have been re-armed
  while sleeping.
  `reportThrottle(self, key, named: ?Duration) error{RateLimited}!void`:
  under the mutex, `refresh(now)`; set `episode_started` if null; if `now -
  episode_started > throttle_budget` set `exhausted` and return
  `RateLimited` (do not extend the hold; the next clean window clears the
  episode); if not currently holding (`until` null or passed):
  `consecutive += 1`, `delay = clamp(max(named orelse 0, fullJitterDelay(
  initial, max_delay, consecutive)), initial_delay, max_hold)`, `until =
  now + delay`, `length = delay`, `generation += 1`, `stats.recordHold()`;
  if already holding and `named` is longer than what remains, extend
  `until` to `min(now + named, now + max_hold)` (a request that was in
  flight when the hold armed got its own 429 with a later reset).
  `refresh(now)`: when `until` is set and `now >= until + length` (a clean
  window as long as the last hold with no new throttle), clear `until`,
  `consecutive`, `episode_started` and `exhausted`. No success counter is
  needed: a throttle during the window would have re-armed `until`.
  `perform(comptime T, gov, spec: RequestSpec, ctx, comptime attempt: fn
  (@TypeOf(ctx), Attempt) anyerror!Outcome(T)) anyerror!T`, the one loop:
  `charged = 0`; loop { `try gov.admit(spec.key)`; `stats.recordAttempt()`;
  switch on `attempt(ctx, .{ .ordinal = charged, .previous_status })`:
  `.done => |v| return v`; `.retry => |f|` { `stats.recordFailure(f.failure)`;
  if `f.failure == .throttle` { `try gov.reportThrottle(spec.key, f.delay)`;
  continue } if `charged >= max_retries` return `error.RetriesExhausted`;
  `charged += 1`; `stats.recordRetry()`; delay = `f.delay orelse
  fullJitterDelay(initial, max_delay, charged)`; `stats.recordRetryDelay`;
  `try io.sleep(delay, .awake)` (propagate `Canceled`; today's `catch return
  error.RetriesExhausted` at `range_read.zig:133` goes) } }.
  `Outcome(T) = union(enum) { done: T, retry: Failure }` with `Failure
  {failure: ReadFailure, delay: ?Duration, status: ?std.http.Status}`.
  `exchange(comptime T, client, method, uri, options: ExchangeOptions,
  spec, ctx, comptime consume: fn (@TypeOf(ctx), *std.http.Client.Response)
  anyerror!T) anyerror!Outcome(T)`, the shared attempt body: `client
  .request`, `sendBodiless` (or `sendBodyComplete(payload)` when
  `options.payload` is set, for the OAuth POSTs), `receiveHead` into the
  caller's redirect buffer, mapping the connect, send and receive errors
  to `.retry` exactly as `performAttempt` does today (`range_read.zig:
  157-190`: `Timeout` is `.timeout`, connection and network errors are
  `.transient`, anything else is fatal); a status outside
  `options.accept` (a small set: `.success`, `.success_or_redirect`) goes
  through `classifyStatus(status, spec.unavailable)`: a classified status
  returns `.retry` with `serverRetryDelay(head)` and the status, an
  unclassified one logs and returns `error.RequestFailed` (the accepted
  4xx that callers map today, 404 and 401/403, are handled in `consume` or
  by the caller's accept set); an accepted response is handed to `consume`
  while the request is alive. `classifyStatus`, `serverRetryDelay`,
  `fullJitterDelay`, `RequestSpec` (gaining `key`), `Attempt` (gaining
  `previous_status: ?std.http.Status`), `PreparedRequest`, `prepareStatic`
  move to `request.zig`; `range_read.zig` keeps `ContentRange`,
  `parseContentRange`, `readSize`, `readResponse`, `readScatter` and a
  `performRangeRead(gov, client, spec, data, offset, size)` that is
  `perform` over `exchange` with `readResponse` as `consume` (the Range
  header appended per attempt as today, `:112-121`).
  `AtomicReadStats` and `ReadStats` (`vfs/base.zig:15-38`, `46-121`) gain
  `holds` and `hold_wait_ns` with `recordHold`, `recordHoldWait`, in
  `sub` and `snapshot`; the test at `:105` covers them.
  New error: `error.RateLimited` (in `request.zig`), distinct from
  `RetriesExhausted`.
  Unit tests in `request.zig`, no server: a throttle arms a hold of at least
  `initial_delay` even for a named zero; a longer named delay extends a
  running hold, a shorter one does not; two holds without a clean window
  double the jittered base (assert the bound, not the draw); a clean window
  resets `consecutive` and the episode; the budget marks the episode
  exhausted and `admit` fails with `RateLimited` until the window passes;
  `report` of `.timeout` and `.server_failure` leaves `until` null; `admit`
  returns at once with no hold; a cancelled waiter returns `Canceled`
  (spawn the waiter in an `Io.Group` and cancel it). Move the three
  `range_read.zig` tests at `:408-440` (classification, server delay, retry
  config) to `request.zig`.

- [ ] 24. Every backend request through the loop. Per site, the request is
  one `perform` over `exchange` with the site's `consume`; redirect chains
  stay in the backend as a loop of governed hops with `accept =
  .success_or_redirect` and `redirect_behavior = .unhandled`. Sites:
  HF (`vfs/hf.zig`): the tree GET `fetchTreeFromAPI` (`:322-369`; runs
  under `self.mutex` from `getOrFetchTree` `:305-318`, which is fine because
  the hold is a deadline, not a permit: other opens wait on the mutex for
  the hold they would wait on anyway); each hop of `resolveDownloadUrl`
  (`:371-415`); the data GET `performRead` (`:787-803`) with a
  `DataRequest.prepare` hook replacing `prepareStatic`: when
  `attempt.previous_status` is 401 or 403 it re-resolves the download URL
  through the governed HEAD chain and stores it on the handle under
  `self.mutex` (a signed CDN URL can expire during a long hold; today a 403
  is unclassified and sticky); the data GET's `RequestSpec` accepts one
  401/403 as a charged retry (`spec.retry_forbidden_once`) and every other
  site keeps them fatal. `unavailable` stays `.server_failure`.
  S3 (`vfs/s3.zig`): `listObjectsBody` (`:632-704`), `fetchSize` (`:746-796`,
  404 stays `FileNotFound` in `consume`), `performRead` (`:798-815`) with
  `SignedRequest.prepare` unchanged (it re-signs per attempt, `:823-838`).
  GCS (`vfs/gcs.zig`): `refreshMetadataServerToken` (`:296`),
  `refreshAuthorizedUserToken` (`:317`), `refreshServiceAccountToken`
  (`:340`) as governed POSTs with a payload (they run under `self.mutex`
  from `getOrRefreshToken` `:404-416`, the same argument as HF's tree);
  `listObjectsBody` (`:825-875`); `fetchSize` (`:926-958`, 404 and 401/403
  mapped in `consume` as today); `performRead` (`:960-968`) with
  `BearerRequest.prepare` unchanged.
  HTTP (`vfs/http.zig`): each hop of `fetchSize` (`:344-381`, switch to
  `.unhandled` and `.success_or_redirect`); `performRead` (`:383-395`).
  Keys: every site passes the request URI's authority (`uri.host` and
  port) as `spec.key`; unused today.
  Wrappers: `fileReadPositional` and the open, stat and read-dir entry
  points of the four backends (`hf.zig:755-761`, `http.zig:312-318`,
  `s3.zig:559-565`, `gcs.zig:721-727` and their `dirOpenFile`,
  `dirStatFile`, `dirRead` counterparts) pass `error.Canceled` through
  (every `std.Io` error set includes `Cancelable`) and keep mapping the
  rest to `Unexpected` with the existing log line, so a cancelled task no
  longer surfaces as `Unexpected`.
  Behaviour that must not change: the S3 and GCS `503 => .throttle` and the
  HF and HTTP `503 => .server_failure` classification (tests at
  `s3.zig:846`, `gcs.zig:988`); one exact scatter read per source job; the
  per-attempt re-signing. The buffered backend's read path is untouched
  in `buffered_loader.zig`.

- [ ] 25. Mock server and acceptance tests (`vfs/http_acceptance_test.zig`).
  `MockServer.Options` gains `throttle: struct { first_gets: usize = 0,
  status: std.http.Status = .too_many_requests, retry_after_s: ?u32 = null,
  rate_limit_reset_s: ?u32 = null, window: ?struct { gets: usize, per_ms:
  u64 } = null, first_heads: usize = 0 }`; the GET handler (`:105-150`)
  answers the first `first_gets` GETs (and HEADs) with `status` and the
  named headers, and under `window` answers with `status` once more than
  `gets` GETs arrived in the last `per_ms` ms; it records
  `first_throttle_at` and `first_get_after_throttle_at` (awake
  timestamps) and offers `resetPeak()` since `peak_gets` is a monotone
  `fetchMax` (`:110`). Serve a small path table instead of the single
  `/object` (`:95-101`), task 27 needs `/model.safetensors` and a 404 for
  `/model.safetensors.index.json`.
  Tests, each with the existing eight-reader harness of `:338-434`:
  one 429 with `Retry-After: 1` holds every reader (every GET after the
  throttle is at least 1 s later, `holds == 1`, bytes correct, `retries`
  unchanged); a window limiter (4 GETs per 200 ms) with sixteen readers
  over a 64-range object completes, no `RetriesExhausted`, `holds > 0`;
  `Retry-After: 0` produces a hold of at least `retry_initial_delay` and
  `get_requests` stays below readers times (1 + elapsed / initial_delay);
  a 503 with `unavailable = .throttle` holds and with `.server_failure`
  retries per request without a hold (`holds == 0`); a throttled HEAD at
  open waits and succeeds (`head_requests == 2`); a permanently throttling
  server with `throttle_budget = 300 ms` fails with `RateLimited` within
  about a second and the reader's `fileReadPositional` returns
  `Unexpected` with the log line; a reader cancelled during a hold returns
  `Canceled`. The four existing tests keep their assertions on the server
  counters. `Retry-After` parses whole seconds, so the sub-second cases use
  the window mode or the governor's unit tests.

- [ ] 26. The loader keeps the width and drops the watch
  (`zml/io/direct_loader.zig`). Delete `ThrottleWatch` (`:2253-2300`),
  `ReadStatsCursor` (`:2432-2452`), `RequestGate.setLimit` (`:2338-2344`),
  the `throttle` and `throttle_group` fields (`:53-58`), the watch
  construction and spawn (`:140-152`), the two `stopWorkers` lines
  (`:435`, `:437`), the tests at `:3429-3520` (`FakeStatsProvider` and the
  two watch tests) and `:3604-3630` (gate reduction). `Loader.width` is
  written once at create. Keep `Metrics.read_operations` (the summary
  uses it), `LoadProfile.stats`, `Diagnostics.source_stats` and the batch
  source line (`:476-489`), which gains `source_holds` and
  `hold_wait_ms` from the two new counters. Docs: `loader.zig:89-94` ("The
  direct backend halves it when the source throttles" becomes "fixed for
  the load; a throttled source holds the VFS, see `VFS.request`"),
  `docs/learn/loader.md:72` and `:130-138`, the header of
  `direct_loader.zig:1-4` and the comments at `:40`, `:55`, `:67-68`.
  Task 16 drops its throttle-watch rider (no `throttle_group`, no watch
  fields to trim) if this lands first; either order works.

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
