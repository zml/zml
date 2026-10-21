# Loader review, end to end (2026-10-21)

This reviews the weight loader from llmd's use of it (`monorepo/llmd/main.zig`, `models.zig`, `weights.zig`, the model files) through zml's front end (`zml/io/loader.zig`, `backend.zig`, `execute_admission.zig`), the Source algebra (`Source.zig`, `TensorStore.zig`, `ShardGrid.zig`), the direct backend (`direct_loader.zig`, `host_memory.zig`, `dma_calibration.zig`), the buffered backend, and the VFS data plane (`vfs/`). It is measured against one brief: code earns its place, fewer abstractions, logic local, simple enough to reason about locally. It also hunts for bugs and missing parts.

**Method.**
- Twelve independent reviewers: eight by area (consumer, front end, Source/store, direct types and lifecycle, planner, execution, memory/calibration/buffered, VFS) and four cross-cutting (the end-to-end story, concurrency and ownership, failure modes, tests). Two completeness rounds added five focused reviews: remote stalls, the PJRT boundary, zio, remote credentials, and llmd's quantized bindings.
- 185 findings. Each was checked by one to three verifiers told to refute it: three lenses (trace, guards, reachability) for high bugs, two for other bugs, one skeptic otherwise.
  - 184 survived. 57 were confirmed as stated. 123 survived only as "partial", with a corrected claim, severity or fix; the text below uses the corrected versions. 4 are unverified because their verifier's call failed (they are marked).
  - 1 was refuted: the CTX "Open work" note that `swapEngramHostPinnedBuffers`' errdefer frees owners it leaves in the list. It does not double-free, so that note can be dropped.
- The review read code only: nothing was built or run, since the GPUs are busy.
- I re-checked these against the code myself:
  - the Source harness running zero cases;
  - `//llmd:test` not compiling;
  - the compressed-tensors NVFP4 globals;
  - dense NVFP4 dropping `input_scale`;
  - `io_impl` never reaching the platform;
  - the borrowed `file_uri` keys;
  - the `auto` probe warming what it samples;
  - progress running out of `std.Progress`'s 127 nodes;
  - `Workspace.init`'s `unreachable` on mixed device kinds;
  - `Exe.call` panicking;
  - the BlockPool wait that cannot fire.

  All hold.

**Verdict.** The core that moves bytes is sound: Source lowering, ShardGrid, the per-file planner, FIFO claims, per-device pumps, credits and the sticky-error drain. Reviewers traced it on success and failure paths and found no wrong-byte path, leak or deadlock in it. What does not earn its place sits at the seams:
- one byte run has three types;
- parts are validated two or three times, by a 175-line overlap prover that re-proves what Source already guarantees;
- pinned memory is bounded twice, and the second bound can never act;
- `Pipeline` is a second handle on `Loader`;
- three dispatch layers only forward;
- calibration (740 lines) now only picks a block size;
- llmd's Packer hand-pairs `bindProgram` with recipe entries.

The real bugs are mostly outside the byte core: in llmd's quantized bindings, in the remote VFS backends, in error and cancellation paths, and in tests that do not run.

## Read this first

The issue numbers (#n) refer to Part 1; the move numbers refer to Part 2's simplification plan.

**Wrong results or broken builds today.**
1. **The Source op harness checks nothing.** `zml/zml/io/Source.zig:1253` iterates `std.meta.declarations(cases)`, which lists only `pub` declarations, and every case is private. The loop body even uses `decl.name`, which would not compile on Zig 0.17 if the list were non-empty. So "every Source op places the bytes the same Tensor ops compute" runs zero cases. Make the cases `pub`, adapt the loop, and fix `WidenBits`, which the reviewers found invalid.
2. **`//llmd:test` does not compile.** `monorepo/llmd/models/deepseek41.zig:3214` asserts `binding.transformed`, a field that no longer exists. The 28th pass ran only `//zml:test` and `//llmd:weights_test`.
3. **compressed-tensors NVFP4 MoE on the FlashInfer path inverts the global scales** (#3). `weights.zig:526-533` ignores `scale.operation`; the compressed-tensors globals are `.divide` (`weights/linear.zig:145`). Every MoE layer outputs garbage, silently.
4. **Dense NVFP4 Linears drop `input_scale`** (#10, `weights/linear.zig:298-325`).
5. **FP8 per-tensor experts with separate gate and up get `[E,2]` scales that Triton cannot reshape** (#11). These are three llmd-side quantization bugs in code with no test; see the test gaps in Part 1.
6. **llmd never passes `io_impl` to `Platform.auto`** (#4, `monorepo/llmd/main.zig:362`). The platform defaults to `.threaded`, so under llmd's default zio, `Compiler.compile` skips `zio.blockInPlace` and emits MLIR on an 8 MiB coroutine. This is the most likely cause of the Laguna compile stack overflow left open in CTX. The fix is one line.

**Hangs, crashes and silent misbehavior.**
7. **A stalled remote range GET hangs the load forever,** unlogged and uncancellable (#1). The head and body reads have no bound.
8. **HF re-resolve frees a URL other in-flight reads still use** (#2). The signed URL is also logged (#14).
9. **The remote backends map "absent" and "forbidden" inconsistently,** so a missing optional file aborts llmd on s3:// and https:// (#6). S3 lacks session tokens (#5) and signs unencoded keys (#12); https cannot follow redirects (#13); GCS key files never parse (#16).
10. **Error paths.**
    - A failed pack execution panics in `Exe.call`, and asynchronous host-to-device failures are never checked (#17).
    - `deinit` with pending submissions reads and transfers all of them before the error surfaces, while llmd has already freed the executables and the progress frames they borrow (#18).
    - `Workspace.init` hits `unreachable` on mixed GPU kinds (#9).
    - Progress freezes after about 125 destinations because each holds a `std.Progress` node until its batch retires (#19).
    - The direct backend keys `source_slots` by borrowed registry strings (#8). llmd's registry outlives the loader, so this is latent.

**Performance.**
11. **The `auto` O_DIRECT probe warms what it samples** (#7). An RWF_NOWAIT miss queues readahead (the comment at `vfs/direct_io.zig:107-108` says so), so the next load sees those pages cached and reads the file buffered. Sampling offsets should shift per probe. The effect on the GPU hosts is not measured.
12. **Other measured or estimated costs:**
    - The engram pinning stall (2.4 s of 11.9 s; known).
    - Device destinations created on the read path (up to about 0.8 s).
    - On remote sources, the buffered backend's open per part and the serial registry parse.

**Simplification: the moves with the best value per risk** (Part 2 has all ten).
- Move 2: let the lifecycle credit be the only bound on pinned memory. `BlockPool`'s wait can never fire, because credits are `capacity / maximum_blocks_per_job`. About 250-300 lines go.
- Move 3: validate once at the front door and delete the overlap prover. About 300 lines go.
- Move 1: give host-resident destinations one shared-allocation path. The replica-copy machinery runs only in tests.
- Move 4: fold `Pipeline` into `Loader`, and narrow the Scheduler and the diagnostics. About 200-250 lines go.
- Move 5 is a decision for you, because it reverses a recorded one: fix the DMA block size per target and delete `dma_calibration.zig` and the workspace growth API. About 850-900 lines go, and 0.7-1.25 s comes off each PCIe GPU init.

## Part 1: Bugs, missing parts, tests and docs

### Bugs and missing parts

#### High

**1. A stalled range GET hangs the load forever, with nothing logged and no way to cancel it**
`zml/vfs/request.zig:348-416`, `zml/vfs/range_read.zig:198-207`, `zml/zml/io/direct_loader.zig:306`, `zml/zml/io/direct_loader.zig:3967`, `zml/zml/io/loader.zig:167`, `zml/zml/io/loader.zig:300`
Waiting for the response head and reading the body have no bound; connect is bounded only by kernel SYN retries (~127 s per attempt). A peer that goes silent without RST or FIN parks one worker that holds a credit, a gate slot and leased blocks. Its batch never completes, `awaitAll` and `deinit` wait in `waitUncancelable`, and nothing logs above debug. A stalled HEAD inside a `SourceSlot` LazyOnce blocks every worker on that file. Fix it in `request.perform` with an inactivity watchdog: race the attempt against a sleep and cancel when a progress counter, bumped per ~1 MiB slice in `readScatter`, has not moved for ~60 s. Do not use a total `30 s + size/1 MiB/s` deadline: with 32 connections it retry-storms on any link below ~140 Mbit/s. On expiry, set `connection.closing = true` and classify from the timer arm. A cancel today surfaces as `ReadFailed` (logged as transient) or goes through `fatal`, so the "Cancellation propagates" comment at :157 is false. Keepalive and TCP_USER_TIMEOUT are a stopgap only: they miss live-but-stalled peers and the TLS handshake. The proposed stall watchdog in `awaitBatch` is unnecessary once attempts are bounded.

**2. HF re-resolve frees the download URL that other in-flight reads still use**
`zml/vfs/hf.zig:870-931`, `zml/vfs/hf.zig:308`
`reresolve` frees `handle.download_url` under `hf.mutex`. Other readers of the same handle hold `url`, `uri` (which slices into it), `spec.target` and `spec.key` without the lock, and concurrent re-resolvers free each other's fresh URLs. This runs exactly when a signed CDN URL expires or returns 401/403 while width-32 workers share one `SourceSlot`: the result is a crash or a garbage request, never wrong data, because Content-Range is checked. Fix: have each `DataRequest` copy the URL under the mutex at the start of `performRead` (or keep superseded URLs in a per-handle arena freed in `Handle.deinit`), and point `spec.target`/`key` at the stable `handle.uri`. Getting "one HEAD chain per expiry" needs a single-flight mutex held across the resolve; an equality check on entry does not give it, and it is optional. Separately, the unlocked `trees.getPtr` fast path at :308 races the insert done under the lock. Add a mock-server test that answers 403 once to concurrent range GETs.

**3. compressed-tensors NVFP4 experts get inverted global scales on the FlashInfer path**
`monorepo/llmd/weights.zig:526-533`, `monorepo/llmd/weights.zig:717-747`, `monorepo/llmd/weights.zig:1261-1270`, `monorepo/llmd/weights.zig:1629-1636`, `zml/zml/moe/cutlass_flashinfer.zig:771-776`
`loadFlashinferCutlassNvfp4ExpertWeight` drops `scale.operation`. Its two void recipes take the reciprocal of the max, and the product, of what compressed-tensors stores as divisors, then label the result `.multiply`. With igs≈100 and wgs≈5000, the activation global comes out 0.01 instead of ~100 and alpha is off by ~1e11, so every MoE layer on B200/GB300 outputs garbage with no error. The only test uses ModelOpt names. Minimal fix: give both recipes an `Operation` and convert inputs to multipliers before the max and product (~6 lines). Better fix: bind the per-expert scalars with their `.operation` through `bindExpert`, as `loadStackedExpertWeight` already does, and compute `1/max(input)` and alpha in cutlass_flashinfer.zig. That deletes both recipes, both programs, `stackExpertScalars`, the second `[P*E]` stacking convention and the `.multiply` mislabel, at the cost of a cross-shard max per forward (or one reduction at load time). The same function reads only projection 0's weight global; exporters unify w1 and w3, so say so in a comment at :738. Add a test that uses compressed-tensors names.

**4. llmd never sets `Platform.io_impl`, so under zio every compile runs on an 8 MiB coroutine**
`monorepo/llmd/main.zig:362`, `zml/zml/platform.zig:785`, `zml/zml/Compiler.zig:416-443`, `zml/zml/Compiler.zig:50`, `zml/pjrt/pjrt.zig:475-478`, `monorepo/llmd/weights.zig:1602`
`--io-impl` defaults to zio, but `CreateOptions.io_impl` defaults to `.threaded`, so `Compiler.compile` never takes its `zio.blockInPlace` branch. MLIR is emitted on an 8 MiB coroutine; this is the most likely cause of the Laguna segfault recorded at CTX:5331. The thread-local `_current` is safe only because traced code never parks. `pjrt.Client.compile` also wraps `PJRT_Client_Compile` in `io.async` followed by an immediate await, which pins an executor during pack compiles that overlap the load. Minimal fix: pass `.io_impl = args.io_impl` at main.zig:362. Better fix: call `zio.blockInPlace` unconditionally (it runs inline when no zio task is mounted, i.e. under Threaded), call `compileRaw` directly, and delete `Platform.IoImpl`, `CreateOptions.io_impl`, `Platform.io_impl` and the switch. Confirm with a Laguna run on default flags.

**5. S3 accepts only static environment keys**
`zml/vfs/s3.zig:14-18`, `zml/vfs/s3.zig:52-55`, `zml/vfs/s3.zig:224-267`, `zml/vfs/s3.zig:453-458`, `zml/vfs/s3.zig:830`, `monorepo/llmd/main.zig:334-336`
There is no `AWS_SESSION_TOKEN`/`x-amz-security-token`, and no web-identity, container, IMDS or profile credential source. On EC2, EKS, ECS or SSO the first HEAD returns 403, either unsigned or signed without the token. That becomes `RequestFailed`, then `Unexpected`, and llmd exits from `parseConfig`. Fixes, in order:
- Add the session token. First collapse the three copies of signer plus `headers: [2]` (Listing, SizeRequest, SignedRequest) into one `sign()` that returns the headers to send.
- Map a HEAD 403 to `PermissionDenied`.
- Print the "unauthenticated" warning on the first s3:// open; today it prints on every llmd start.

Add a credential chain only when an AWS deployment needs one (container credentials and IMDSv2 return the same JSON; STS returns XML). Do not make `S3.auto` fail on `AWS_PROFILE` and similar: llmd, zoo and the examples call it unconditionally, so hf:// and local loads would break. Report unsupported sources on the first s3:// request instead.

**6. A missing optional file aborts llmd on s3:// (without ListBucket) and on https://**
`monorepo/llmd/chat-template/jinja.zig:234-237`, `monorepo/llmd/chat-template/jinja.zig:250-253`, `zml/vfs/s3.zig:830`, `zml/vfs/s3.zig:840-843`, `zml/vfs/http.zig:254-260`, `zml/vfs/request.zig:401-404`
llmd probes `chat_template.jinja` and `.json` and tolerates only `FileNotFound`. Under a GetObject-only policy S3 answers 403 for a missing key, and HTTP does not accept 404; both become `Unexpected`, so model init fails even though every required file is readable. Fix:
- S3: accept `.forbidden` and map it to `PermissionDenied`, as gcs.zig:1046-1049 does, with pass-through arms in `dirOpenFile` and `dirStatFile`.
- HTTP: accept `.not_found` and map it to `FileNotFound`.
- llmd: widen the two jinja catches to `FileNotFound, PermissionDenied => null`, with a warn, because a template that exists but is denied changes precedence.

This also removes the spurious err logged by the swallowed index probe. An `openOptional` helper does not earn its place for two call sites in one file.

#### Medium

**7. The `auto` probe caches the pages it samples, so `auto` becomes `off` after the first load (performance only)**
`zml/vfs/direct_io.zig:104-128`, `zml/vfs/direct_io.zig:26-28`, `zml/zml/io/backend.zig:41-45`
On a miss, each of the 32 RWF_NOWAIT samples (at offsets that depend only on file size) pulls its page into the cache. The next process probes the same offsets, sees 32/32 and reads the whole cold file buffered (verified on Linux 7.1/ext4). The second load then takes ~35 s instead of 12.8 s on MI300 DeepSeek and 4.58 s instead of 3.73 s on GB300. Fix: apply one random page-aligned shift per probe (`offset = window*i + shift`, seeded from `io.random`). Correct the "cold stays cold" text at backend.zig:41-43, direct_io.zig:26-28 and :107-109, and in CTX. An end-to-end test needs a file of at least 64 MiB, a skip when DONTNEED does not evict (tmpfs), and a wait for the first probe's readahead: still-locked pages answer EAGAIN and make the test pass on the current code. A deterministic unit test ("two probes sample different offsets") is cheaper.

**8. Direct-backend `source_slots` keep borrowed `file_uri` strings for the loader's whole lifetime**
`zml/zml/io/direct_loader.zig:100`, `zml/zml/io/direct_loader.zig:337-345`, `zml/zml/io/direct_loader.zig:538`, `zml/zml/io/direct_loader.zig:595-602`, `zml/zml/io/direct_loader.zig:629`, `zml/zml/io/direct_loader.zig:1120-1136`, `zml/zml/io/loader.zig:47-50`, `zml/docs/learn/loader.md:57`
Map keys and `SourceSlot.uri` point into the TensorRegistry arena, but the documented borrow ends at retirement. After `awaitAll` plus `registry.deinit()`, a later submit reads freed memory in `get` (same path), in every `grow` rehash, and on a lazy open or log through a never-opened slot. Fix: `dupe` the URI in `sourceSlot` (with an errdefer free), use it as both key and `slot.uri`, and free it in `destroy`, about 4 lines. The regression test must free the registry, not only the store, and assert `source_slots.count() == 1` after reloading the same path: the testing allocator poisons freed memory rather than unmapping it, so a crash-based test would pass. Do not drop slots in `awaitBatch`: pending batches share them, so that creates a new use-after-free. Related: the slot's `LazyOnce` caches a planning-time open failure from `alignmentOf`, so a submit rejected as "recoverable" replays the same error forever. Resetting the LazyOnce is unsafe (it is shared, and its Event is set once); instead make `file` a plain `?File` opened at planning time, before publish. Document that a loader keeps one open fd per path until `deinit`.

**9. `Workspace.init` hits `unreachable` on mixed device kinds**
`zml/zml/io/host_memory.zig:319-329`, `zml/zml/io/direct_loader.zig:387`, `zml/zml/io/direct_loader.zig:3531`, `zml/zml/platform.zig:318-327`, `zml/docs/learn/loader.md:240`
Two GPU SKUs (for example 3090 + 3090 Ti, or an iGPU plus an Arc under oneAPI) make `Loader.init` panic in release_safe, after a log line naming `UnsupportedPlatform`, an error nothing returns. Nothing needs devices of one kind: calibration measures `devices[0]`, and its result is valid on every device. Delete the kind loop and the `len == 0` arm (Platform.init already returns MissingDevices). The `> 64` arm can be reached, because Platform.init only warns above MAX_NUM_DEVICES and keeps every device; its "won't be used" message is false. Keep one check, returning `error.UnsupportedPlatform` at the top of `direct_loader.Loader.create` before `Sizing.init`, tied to the u64 `writer_mask`, and fix the loader.md row.

**10. Dense NVFP4 Linears drop the checkpoint's input scale**
`monorepo/llmd/weights/linear.zig:271-277`, `monorepo/llmd/weights/linear.zig:298-325`, `monorepo/llmd/weights.zig:375-380`, `monorepo/llmd/weights.zig:472-479`
`detect` keeps `input_scale` for NVFP4, but neither `createWeight` nor `loadFusedWeight` sets it (a regression from #407). On Blackwell, dense activations are therefore quantized with per-block e4m3 scales clamped to [2^-6, 448]: blocks with amax below ~0.094 lose every value under ~0.004, and blocks above 2688 saturate. The same checkpoint's experts do use the scale. Fix:
- In `createWeight`, set `.input_scale = if (q.input_scale) |s| s.create(store) else null`.
- In `loadFusedWeight`, use the first member's scale as a scalar, not a broadcast.
- Have `assembleLinears` reject members whose input-scale presence or operation differ. If a max across members is ever wanted, take it over `asMultiplier` values.

Delete the `activation_scale` convention; its only output is an input scale that is discarded for every non-NVFP4 scheme.

**11. FP8 per-tensor experts with separate gate and up get `[E,2]` scales that Triton cannot reshape**
`monorepo/llmd/weights.zig:450-465`, `monorepo/llmd/weights.zig:503`, `monorepo/llmd/weights.zig:639-644`, `monorepo/llmd/weights.zig:686-700`, `zml/zml/moe/triton.zig:76-77`
For ModelOpt or llm-compressor FP8-static gate/up checkpoints (qwen3_5 per-expert, laguna, deepseek41, gemma4), `bindExpert`'s scalar branch builds `[expert, proj=2]` with the scheme still fp8_per_tensor. `triton.call` then reshapes 2E elements into `[E,1,1]` and compilation fails; one scale per expert could not cover both projections anyway. Fix only the weight-scale call in `loadStackedExpertWeight`; metal's NVFP4 globals depend on the `[E,2]` stacking. When the scheme is fp8_per_tensor and `projections.len > 1`, reshape each member to `[E]`, `Source.broad` it to `[E, rows_p, 1]` (stride 0, no program), concatenate on rows, and set the scheme to `fusedScheme(false)`. Add a CPU test next to weights.zig:2267.

**12. SigV4 signs the raw S3 key**
`zml/vfs/s3.zig:44-47`, `zml/vfs/s3.zig:76-81`, `zml/vfs/s3.zig:645-648`, `zml/vfs/s3.zig:785-787`, `zml/vfs/s3.zig:856-859`
`s3Url` concatenates the unencoded key, and `Uri.parse` keeps it as `.percent_encoded`, so the canonical URI is not UriEncode(key). Any of `=+!$()*,;:@` or non-ASCII gives SignatureDoesNotMatch (a bare 403 on HEAD), a space gives a 400, and `?` or `#` requests a different key. Hive-style `model=…/step=…` prefixes fail to open while `ls`, which does encode, works. Fix: encode with `Component.percentEncode(w, key, isS3PathChar)`, where `isS3PathChar(c) = s3EncodeIsValid(c) or c == '/'`; `.raw` with std's `isPathChar` is wrong because it leaves sub-delims unencoded. Once the key is encoded, the existing parse is safe. Also hoist `var host_buf` to function scope: it is currently used after its block ends. Test: `a=b/c d+e` must produce `a%3Db/c%20d%2Be`.

**13. https:// sources behind a redirect cannot be read**
`zml/vfs/http.zig:363-432`, `zml/vfs/http.zig:444-458`, `zml/vfs/request.zig:316`, `zml/vfs/hf.zig:401-472`
`fetchSize` follows the HEAD chain but discards the final URL, and `performRead` then GETs the original URL with `.forbid`, so the header read fails with TooManyHttpRedirects. The walker also has three defects:
- The first hop overwrites `full_url`, which relative Locations are resolved against.
- An error from `resolveInPlace` hits `catch unreachable`.
- There is no hop limit, and an empty `Location:` loops forever.

Simplest fix: let HTTP range GETs use `.redirects = .follow`. std resolves relative Locations, keeps the Range header, caps the chain at 3 hops and strips auth across hosts, and re-resolving on every read makes signed-URL expiry irrelevant. Fix the HEAD walker locally: a separate base buffer, `try`, and a hop counter. Resolving once and storing the URL would also need HF's 401/403 re-resolve; a shared `request.resolveRedirects` needs a consume hook and a credential-policy parameter.

**14. Signed HF CDN URLs are logged at warn and err**
`zml/vfs/hf.zig:416-420`, `zml/vfs/hf.zig:457`, `zml/vfs/hf.zig:716-719`, `zml/vfs/hf.zig:873-878`, `zml/vfs/range_read.zig:130-132`, `zml/vfs/request.zig:394-426`
`RequestSpec.target = handle.download_url`, so retry, hold and failure lines print the presigned URL, which works as a bearer credential for gated repos. After a re-resolve the target also dangles. Fix: use `.target = handle.uri`, as gcs.zig:1069 does, in `performRead` and `reresolve`, and a stable identity on resolve hops. `dirOpenFile` maps resolve failures to `OpenError.Unexpected` without logging the cause; log it before mapping.

**15. `exchange` returns a connection whose send failed to the pool, and treats a TLS handshake reset as fatal**
`zml/vfs/request.zig:352-373`, `zml/vfs/request.zig:413-416`
After a send failure the reader is still `.ready`, so `Request.deinit` releases the dead socket to the tail of `free`, which is where `findConnection` looks first. Behind an RST-on-idle middlebox, a serial reader burns all 6 attempts on that one socket. std reports every handshake I/O failure as `TlsInitializationFailed`, which is not in the transient list, so one reset during any of a long load's many handshakes fails the load. Fix: one send catch that sets `req.connection.?.closing = true` before `retryable`, and add `TlsInitializationFailed` to the transient connect errors (a permanent certificate error then costs at most ~31 s). For the follow-redirect connect, also add it to the receive-head list, and use `if (req.connection) |c|`, because `redirect` nulls the connection.

**16. GCS service-account key files never parse, and `GCS.auto` silently falls back to another identity**
`zml/vfs/gcs.zig:57-138`, `zml/vfs/gcs.zig:228-293`, `zml/vfs/gcs.zig:397-402`, `monorepo/llmd/main.zig:338-340`
`ignore_unknown_fields = false` plus a strict `ServiceAccount` struct rejects every key Google issues (UnknownField), and the required `account`/`universe_domain` fields reject older ADC files. `auto` logs "Invalid GCS credentials JSON" with no path or reason, then falls back to the metadata server (the wrong principal on GCE/GKE) or to anonymous. A key that did parse would still hit `error.Unimplemented` (std has no RS256 signing) and surface as `Unexpected`. Fix:
- Delete the service_account variant, `JwtAssertion`, `makeAssertion`, `refreshServiceAccountToken` and its commented-out body.
- Parse to `Value`, require `type == "authorized_user"` and that `type` is a string (the `.string` access is unchecked today), then `parseFromValueLeaky(AuthorizedUser, …, .{ .ignore_unknown_fields = true })` with only the fields actually read.
- When GOOGLE_APPLICATION_CREDENTIALS is set but unusable, fail on the first gs:// request with the path and reason, not in `auto`, which llmd runs unconditionally.

#### Low

**17. A failed pack execution panics in `Exe.call`, and asynchronous H2D failures are never checked**
`zml/zml/exe.zig:353-385`, `zml/zml/io/loader.zig:347-353`, `zml/zml/io/loader.zig:454-462`, `zml/zml/io/loader.zig:818-827`, `zml/zml/io/direct_loader.zig:294-300`, `zml/zml/io/direct_loader.zig:3801-3806`
`BoundExecutable.execute` can fail only on host allocation. Any PJRT Execute or completion-event error panics instead of becoming `failure`, against loader.md:74-75. That includes an output or temporaries OOM after "admitted anyway" or under co-tenant pressure, and a pack input whose DMA failed asynchronously, which the direct backend never checks. llmd exits either way, so what is lost is diagnostics and clean unwinding. Fix: make `Exe.call` itself return `pjrt.ApiError!void` and move the panic into its ~6 callers, rather than adding a `callChecked` twin. The fallible body must await and deinit every partition's event while keeping the first error, must destroy the PJRT outputs in `results` when an event fails, and the log text at loader.zig:459 needs updating. Alternatively, drop `.wait = true` and await the output, which turns both errored inputs and execution failures into `AwaitError`. Awaiting every bulk destination's shards in `awaitBatch` needs measuring first (~3 PJRT calls per shard, ~69k tensors on DeepSeek). Rewrite "the loader cannot see" at direct_loader.zig:296-298 as "does not check".

**18. Error paths cannot abort, and llmd lets borrowed state die while submissions are still pending**
`zml/zml/io/loader.zig:47-50`, `zml/zml/io/loader.zig:163-167`, `zml/zml/io/loader.zig:1264`, `zml/zml/io/backend.zig:611`, `zml/zml/io/direct_loader.zig:3586-3595`, `zml/zml/io/buffered_loader.zig:113`, `monorepo/llmd/weights.zig:1676-1685`, `monorepo/llmd/main.zig:471-475`, `monorepo/llmd/models.zig:235`
When `loadInto` fails after publishing pack groups (a compile error, or `TensorNotFound` in the bulk `load`), main's `defer loader.deinit()` reads and transfers every pending submission (bounded by admission) before the error surfaces. Two borrows are already dead by then:
- `deinitPrograms` has freed the `Exe`s those submissions borrow. That is safe only because `deinit` never executes; an `awaitAll` or an admission retire would run freed executables.
- Workers dereference `*std.Progress.Node`s whose frames have already returned.

Fix:
- Add `Backend.abort()` reusing `pipeline.recordError(error.Closed, …)` in the direct backend (`Closed` is already in `PipelineError` and logs at debug) and `first_error` in the buffered backend (which stops only tasks that have not yet reached `loadOne`).
- Add `Loader.cancel()` (abort, then retire without executing). `deinit` calls it when `pending.len != 0`, and `loadInto` adds `errdefer loader.cancel()` after `defer deinitPrograms`.
- Store `Progress.Node` by value in `Destination` and `Batch`; it is a value handle.
- Add `exe` to the borrow lists.

This changes the deinit contract (pending bulk outputs end up errored), so update loader.md:79, CTX:142 and the test at loader.zig:1264. An executing `errdefer awaitAll` would be wrong.

**19. Direct progress: one node per destination is held until the batch retires, exhausting std.Progress's 127 slots (display only)**
`zml/zml/io/direct_loader.zig:590-593`, `zml/zml/io/direct_loader.zig:1272`, `zml/zml/io/direct_loader.zig:1349`, `zml/zml/io/direct_loader.zig:1355-1365`, `zml/zml/io/direct_loader.zig:1405-1410`, `zml/zml/io/buffered_loader.zig:278-279`, `zml/docs/learn/loader.md:49-55`, `monorepo/llmd/models.zig:235`, `monorepo/llmd/models/dspark_drafter.zig:137`, `monorepo/llmd/models/dflash_drafter.zig:876`
After ~125 destinations, `start` returns `Node.none`, whose `end` counts nothing, and the parent moves only when a batch retires. "Loading weights" sits at 0 through the bulk load, then rises by at most ~125 of ~69k. Fix: delete `TensorTransfer.progress`, `startProgress` and the `end` in `deinit`. Move the read counter onto `Destination` and call `completeOne()` on the unique crossing (`before < covered and before + bytes >= covered`, using `fetchAdd`'s return), plus a debug assert in a successful `awaitBatch` that every destination reached `covered_bytes`. Not a per-batch byte node (each `end` adds a spurious item), and not ending child nodes at the crossing (a racy `setCompletedItems` into a reused slot, and slots still unbounded). For the buffered backend, start its node after the admission wait. In llmd, add `defer node.end()` at the three sites that never end their node.

**20. The safetensors header is trusted without validation**
`zml/zml/safetensors.zig:870-921`, `zml/zml/safetensors.zig:1214-1232`, `zml/docs/learn/loader.md:229-231`
- No cap on header length: a git-LFS pointer reads as ~2.4e18 and fails with OutOfMemory.
- `.?`, union access and `@intCast` panic on missing, mistyped or negative fields.
- `data_offsets[1]` is ignored and nothing is checked against the file length, so truncated shards fail mid-load, terminally, and without a tensor name.
- "F8_E5M2" is not mapped.
- Duplicate names across shards are silently overwritten.

Fix:
- Reject `len < 8` and `hdr_len > min(100 MB, len-8)`.
- Decode each non-`__metadata__` entry with `parseFromValueLeaky(struct { dtype: []const u8, shape: []const u64, data_offsets: [2]u64 }, …, .{ .ignore_unknown_fields = true })`.
- Require `start <= end`, `end - start == byteSize` (checked multiplication), and max end ≤ file length (one stat per shard).
- Use `getOrPut` for duplicates and map F8_E5M2.
- Drop the proposed overlap check.

Also verify whether the official FP4 name is "F4" rather than "F4_E2M1".

**21. `loadExecute`'s non-program branch double-loads and checks the wrong shape**
`zml/zml/io/loader.zig:253-256`, `zml/zml/io/loader.zig:271-272`, `zml/zml/io/loader.zig:587`, `zml/zml/io/loader.zig:825`, `zml/zml/io/loader.zig:876-885`, `zml/zml/io/Source.zig:511-516`, `zml/docs/learn/loader.md:205-206`
`prepareModelLoad` skips only `inputs != null`, so a plain tensor sent through `loadExecute` is loaded again by `load`. If `output` aliases the model buffer, the retire write and `TensorTransfer.init` race. `checkExecutableBinding` compares plain bindings against `tensors[0].shape` (the checkpoint's shape), so a relabeled `bindAs` source is rejected while the raw checkpoint shape is accepted. No production caller uses this branch. Fix: restrict `loadExecute` to program bindings. That deletes :253-256, the :271-272 fallback, :876-885 and the doc sentence, fixes both issues, and tests move to `maybeCreateBinding`/`bindProgram`. If the branch is kept, the shape fix is one line (`tensor.shape()`). Do not "skip every delivered id": it breaks the sizing assert at :547-551 and leaves model buffers unwritten.

**22. GCS tokens: no refresh margin, failed refreshes are not remembered, and a 401 is fatal**
`zml/vfs/gcs.zig:150-153`, `zml/vfs/gcs.zig:443-476`, `zml/vfs/gcs.zig:1040-1092`
A token is refreshed only at `now >= expires_at` in whole seconds, and `expires_at` is stamped after the response arrives, so the margin is 0-1 s. A read 401 is not retried, so one lost race at a token rollover fails the load. A failed refresh is not recorded: during a token-endpoint outage, up to ~31 readers rerun the full retry ladder one after another under the mutex, ~8-16 min before the error surfaces. Fix in `getOrRefreshToken`, keeping one mutex:
- Refresh at `now + min(60 s, expires_in/2)` on a monotonic clock in nanoseconds; this stays below the metadata server's ~5 min cache.
- Keep using a still-valid token when a refresh fails.
- Record `failed_at` so waiters fail fast.

A separate token mutex buys nothing, since every read needs the token. Optional: `retry_once = &.{.unauthorized}`, invalidating only if the failing header is still the current one.

**23. `resolveModelEntrypoint`/`resolveModelRepo` erase open errors and break relative single-file paths**
`zml/zml/safetensors.zig:755-770`, `zml/zml/safetensors.zig:1025-1075`, `zml/vfs/hf.zig:632`
Probe errors become FileNotFound, and openDir errors other than FileNotFound become InvalidPath; GCS 401/403 leave no trace at all, and `Canceled` is consumed. `fromPath` reopens a relative path relative to its own dirname, and a bare file name gives InvalidPath. Fix:
- Delete `ModelPathResolutionError` and both mapping switches.
- Propagate `Canceled` at once. Otherwise remember the index error, try `model.safetensors`, and return the remembered error if both fail. Returning the index error immediately would break S3 and HTTP single-file repos until #6 lands.
- `resolveModelRepo` becomes `openDir(cwd, if file then dirname orelse "." else path)`.
- Open the entrypoint by `basename` relative to `repo`.
- hf.zig:632 must pass `Canceled` through.

**24. S3 endpoint precedence is reversed and the region never picks the endpoint**
`zml/vfs/s3.zig:229-259`, `zml/vfs/request.zig:315`, `zml/bench_aws.sh:11`
`AWS_ENDPOINT_URL` wins over `AWS_ENDPOINT_URL_S3`, and `AWS_REGION` alone still targets the global endpoint path-style. A bucket outside us-east-1 then gets a 301, which becomes `Unexpected` on HEAD, or `TooManyHttpRedirects` with no headers on GET. Fix: `AWS_ENDPOINT_URL_S3 orelse AWS_ENDPOINT_URL orelse "https://s3.{region}.amazonaws.com"`, and map `Redirects.forbid` to std `.unhandled` so a 3xx on GET reaches the head-logging branch. Special handling of 301/400 on HEAD is redundant: request.zig:402 already logs the full head, including `x-amz-bucket-region`.

**25. Admission does not charge VMM's per-buffer rounding**
`zml/zml/io/execute_admission.zig:8-17`, `zml/zml/io/loader.zig:380`, `zml/zml/io/loader.zig:504`, `zml/zml/platform.zig:778`
The default `vmm` allocator rounds each buffer up to ~2 MiB, and `bytes_in_use` counts the rounded size, but pending and new submissions are charged their exact sizes. The 64 MiB reserve was sized for BFC, and its comment's "1e5 shards" is really 8,192. An OOM needs the rounding to exceed both the reserve and the formula's existing overcharge, so realistic llmd groups stay safe. Under VMM, retires never raise measured room either, so tight loads serialize. Fix: record the allocator kind on Platform and charge a per-kind constant per shard (2 MiB for vmm, 256 B for bfc) on both sides of the ledger. Do not measure the granule with a 1-byte warmup probe: reuse gives 0, concurrent init distorts it, and BFC chunk splitting is invisible to it. Correct the reserve comment and state the existing overcharge.

**26. The bulk `load` is never admitted**
`zml/zml/io/loader.zig:203`, `zml/zml/io/loader.zig:342`, `zml/zml/io/direct_loader.zig:2817`, `zml/docs/learn/loader.md:69`
Bulk destinations are allocated as workers claim their jobs, behind the last pack job in FIFO order. While the last pending packs drain, that run-ahead can take the room admission set aside for their outputs and temporaries. Gate on `.exceeds` only: gating on `.unmeasured` would retire every pack before the bulk on CPU and buffered backends and bring back the drain CTX:3060-3070 removed. Extract `makeRoom(inputs, execution)` from `admit`, with `execution` as a zeroed n-slice (`admits` iterates four slices in lockstep). Otherwise document the bound; either way, "load is never gated" has to change.

**27. `Binding.memory` is ignored for executable bindings**
`zml/zml/io/TensorStore.zig:19-20`, `zml/zml/io/TensorStore.zig:135-142`, `zml/zml/io/loader.zig:852`, `monorepo/llmd/weights.zig:781-792`
A host-pinned request on a tensor computed through `loadExecute` silently lands where the executable puts it. llmd's dtype-changing `concatenateRows` goes down this path, so an offloaded table would stay on the device and a later `createView` would be applied to a device buffer. Fix: add `if (binding.memory != .default) return error.UnsupportedMemoryKind;` in `checkExecutableBinding`; it covers program and plain bindings. Optionally also assert in `setMemory` for program bindings.

**28. Foreign mesh device ids: an unchecked index in the front end, and a planner check that runs after publication**
`zml/zml/io/loader.zig:504-510`, `zml/zml/io/loader.zig:938`, `zml/zml/io/direct_loader.zig:1067-1070`, `zml/zml/io/direct_loader.zig:1557`, `zml/zml/io/direct_loader.zig:1664-1690`
`addPlacementBytes` indexes `out[device.id]` without a check, so `load` of a device-memory tensor on another platform's mesh panics, or writes out of bounds in ReleaseFast. Host-memory specs reach the planner's per-(file, destination) check, which can fire after earlier files have been published and fails the loader stickily. Fix:
- Validate device ids once for every spec, before `addPlacementBytes`.
- Delete `Config.device_count`, the planner copy of the check, `plan_slot`/`unplanned` (the index is never read) and the `destinations` list. `appendFillJob` then selects `part == &part.destination.parts[0]` over `order`, as the gather selection already does.

Add `load`-path tests with a foreign mesh, in device and host memory.

**29. The sticky error is not uniform**
`zml/zml/io/direct_loader.zig:222`, `zml/zml/io/direct_loader.zig:232-237`, `zml/zml/io/direct_loader.zig:2575-2578`, `zml/zml/io/loader.zig:296-300`
If a worker fails between `submit`'s `checkOpen` and its first publish, `Scheduler.publish` returns `Closed`, and `submit` returns that instead of the stored error. After an execute failure in the front end (host OOM), `awaitAll` keeps executing later packs. Fix: `return self.pipeline.errorValue() orelse err;` in the branch before publication, and `retireOldest(self.failure == null)` in `awaitAll`, with its doc updated.

**30. Validation failures are logged only at debug**
`zml/zml/io/log.zig:1-4`, `zml/zml/io/loader.zig:583`, `zml/zml/io/loader.zig:684`, `zml/zml/io/loader.zig:843-939`, `zml/zml/io/direct_loader.zig:528`
llmd treats TensorNotFound, EmptyTensor, InvalidLayout, Executable*Mismatch and SourceSizeMismatch as fatal, but exits with only the error name. Raise the existing detection-site logs, which already carry tensor id, shape, source and file (~15 in loader.zig, ~5 in direct_loader.zig), to `warn` or `err`. Fold `checkExecutableBinding`'s detail lines into the one at :843. Update the log.zig header and loader.md:225-228. A diagnostics out-parameter does not earn its place.

**31. Valid NVFP4 expert variants panic instead of returning an error**
`monorepo/llmd/weights.zig:524-533`, `monorepo/llmd/weights.zig:706-710`, `monorepo/llmd/weights/linear.zig:327-334`, `zml/zml/moe/cutlass_flashinfer.zig:771-798`
An llm-compressor NVFP4A16 MoE on SM100/103/120 panics on `input_scale.?` at weights.zig:527. Inside `loadFlashinferCutlassNvfp4ExpertWeight`, use `orelse { log.err; return error.UnsupportedQuantization; }` for both scales. That deletes the two `else null` branches, makes the kernel's `.?` an invariant, and pairs naturally with #3. Reject native F4_E2M1 and non-u8 weights there; no producer emits them. `scalarScale` can become `createReplicatedTensor(key, null)`.

**32. Pre-stacked experts bind quantized tensors as plain weights**
`monorepo/llmd/weights.zig:70-86`, `monorepo/llmd/weights.zig:399-417`, `zml/zml/moe/moe.zig:89-91`
An FP8 `gate_up_proj` with a `_scale_inv` sibling loads without scales and fails far from the cause: a Triton `InvalidMlir`, a metal assert, or a TPU UnsupportedType. A pre-stacked bias is dropped, though no model has one. Fix: in that branch (shared with `expertsFormat`), return `UnsupportedQuantization` when the dtype is not bf16/f16/f32 or when `store.hasKey(proj ++ "_")`. Remove the dtype from `expertsFormat` and `Backend.auto` (`_ = dtype`). Correct the doc at :399-400: a missing tensor panics in `createTensor`.

**33. An fp8_block128 partial final block silently misaligns scales when fused, split, interleaved or row-sharded**
`zml/zml/quantization.zig:88-92`, `monorepo/llmd/weights/linear.zig:46-61`, `monorepo/llmd/weights/linear.zig:84-96`, `monorepo/llmd/weights.zig:248`, `monorepo/llmd/weights.zig:382-386`, `monorepo/llmd/weights.zig:573-578`
Classification accepts `divCeil` scale rows, but split, fusion, swizzle and sharding all infer granularity as rows/scale_rows. Fusing [200,K] with [256,K] gives most of the second member wrong scales, with no error. No in-tree model hits this. Verifiers split on the fix; the smaller one is a single invariant where the problem starts: require `scale_rows * 128 == rows` for any View that is split, fused or row-sharded, which leaves linear.zig:89-90 as the only place partial blocks are allowed. The alternative maps scheme to block size at three call sites. fp8_block32 needs no change.

**34. `swizzleBlockFp8Scales` panics on projections whose scales are not e8m0**
`monorepo/llmd/weights.zig:226-252`, `monorepo/llmd/weights.zig:1226-1258`
On sm_10x, unquantized, per-tensor, block128 or f32/bf16-scale projections panic. The official V4.1 layout never reaches this. Fix: return early unless the projection is quantized with `f8e8m0` scales, which only classify as block32 or mx; document that restriction.

**35. Teardown uses `dmaUnmap … catch unreachable`**
`zml/zml/io/host_memory.zig:267-273`
After a sticky device error, unregistering can fail and `Loader.deinit` panics, hiding the original error. Fix: log the failure, leak the arena, and return. Freeing pages whose registration may still be live risks reuse of that address range. `PinnedHostAllocation.destroy`'s `catch unreachable` states a real invariant; leave it.

**36. `warmupDeviceAllocators` leaks running group tasks when a spawn fails**
`zml/zml/platform.zig:466-468`, `zml/zml/io/loader.zig:123`
A spawn failure for device i>0 returns while earlier tasks still write into the dead frame. Preferred fix: `group.async`, which falls back to running inline and never fails, and drops ConcurrencyUnavailable from `Loader.init`. Minimal fix: `errdefer group.cancel(io)`.

**37. Init cancellation is swallowed in one place and logged as an error in another**
`zml/zml/io/host_memory.zig:630-635`, `zml/zml/io/direct_loader.zig:389`
`memoryNodeMask` consumes `Canceled`. A cancel during calibration, which llmd triggers whenever model init fails, logs a spurious err. Fix: use `load_log.failure(err, …)` at :389, and call `io.recancel()` in `memoryNodeMask`, keeping its signature.

**38. Buffered staging admission counts tensor bytes but hands out larger buffers**
`zml/zml/io/buffered_loader.zig:22-27`, `zml/zml/io/buffered_loader.zig:161-166`, `zml/zml/io/buffered_loader.zig:456-509`
`take` returns the best-fit larger retained buffer and counts its full length. Live staging is therefore bounded by the sum of the W largest tensors, not the documented 12× largest; this matters only for MoE checkpoints with many near-largest tensors. Fix: fold `StagingAdmission` into `StagingPool` under one mutex and condition, waiting on the size of the buffer actually chosen. Capping reuse at `2*size` does not restore the bound, and reserving after acquiring can deadlock.

**39. Ordered strided parts with gaps are refused above one read, and the comment says otherwise**
`zml/zml/io/direct_loader.zig:552-567`, `zml/zml/io/direct_loader.zig:2243-2262`, `zml/docs/learn/loader.md:166-170`
Column slices of large tensors are refused with UnsupportedLayout. With today's fixed `gather_len` that is the correct guard, because it keeps the failure before publication. The proposed average-ratio sizing is unsound: outer-dimension gaps make it fail mid-planning. No model needs this, so fix the comment and the docs, and add an adaptive cut in `appendGatherJobs` only when a model needs it.

**40. The event-slot count is a hand-kept counter at three planner sites**
`zml/zml/io/direct_loader.zig:895`, `zml/zml/io/direct_loader.zig:1975`, `zml/zml/io/direct_loader.zig:2232`, `zml/zml/io/direct_loader.zig:2361`, `zml/zml/io/direct_loader.zig:3774-3777`
The counter is correct today, and ReleaseSafe bounds-checks the slice. Derive the count in `preparePlan` as Σ `submissionCount` over DMA jobs (or pass the jobs to `Plan.create`, which also removes the memcpy) and delete `JobCut.dma_submissions`. A runtime `error.Internal` guard would be unsafe: it leaves uninitialized or leaked events and releases leases early.

**41. Direct reads are blocking `preadv` calls on zio executors**
`zml/vfs/direct_io.zig:141-166`, `zml/vfs/vfs.zig:684-696`
Under zio this caps in-flight direct reads at the executor count on CPU-limited hosts; the cancellation argument does not apply. Fix: check alignment in the VFS before dispatch and demote misaligned reads, route aligned reads through `backend.vtable.fileReadPositional`, and demote on `error.Unexpected`. STATX_DIOALIGN alone is not enough (absent on NFS, FUSE and kernels before 6.1). "Direct reads never target PJRT memory" is false on ROCm, so confirm that hipHostMalloc pool memory accepts O_DIRECT.

**42. The buffered backend misplaces destinations on a sub-mesh**
`zml/zml/buffer.zig:159-168`, `zml/zml/buffer.zig:250`, `zml/zml/buffer.zig:322`, `zml/zml/HostAccessible.zig:252`, `zml/zml/io/buffered_loader.zig:614-622`
`Buffer.from` walks `platform.physical_mesh` but slices by the shape's mesh. Production meshes are unaffected, but the loader.zig composite tests (1634, 1727, 1819) would fail on TPU or neuron hosts with more devices. Fix: iterate `placement.mesh().?.devicesInCanonicalOrder()` in `from`, `uninitialized`, `toSlice`, `toSliceAlloc` and `HostAccessible.init` (one rule), then run the buffered test on `devices.init` and delete `Devices.initCpu`.

**43. A 200 answer to a Range GET is accepted at any offset**
`zml/vfs/range_read.zig:175-198`, `zml/vfs/range_read.zig:245-252`, `monorepo/llmd/common.zig:143-149`
This makes transfer O(file²). Fix: `.ok => if (offset == 0) 0 else return error.RangeNotSupported` (already logged and not retried). This is a policy change: small files on servers without Range support stop working. Separately, `common.zig`'s 256-byte reader issues one GET per 256 bytes of config.json on remote backends.

**44. On single-node device hosts the pinned arenas still interleave across sockets (contested)**
`zml/zml/io/host_memory.zig:59-71`, `zml/zml/io/host_memory.zig:644-651`, `monorepo/llmd/main.zig:190-207`
llmd pins CPU affinity to the devices' node but sets no memory policy. The 18th-pass penalty (~21%) was measured only with the page cache on the remote socket; the 7th pass, with a node-local page cache, measured parity or interleave winning. Binding to the device node in `Backend.init` contradicts CTX:1876-1881. A per-thread policy set by llmd never reaches the pool thread that runs `Loader.init`. The candidate fix is to intersect the interleave mask with the nodes of the calling thread's allowed CPUs, but only after measuring that case; until then, document `numactl`. Verifiers disagree on medium versus low, and on whether llmd's default setup pays the penalty at all.

**45. A calibration failure fails `Loader.init` (no code change recommended)**
`zml/zml/io/direct_loader.zig:389`, `zml/zml/io/dma_calibration.zig:139`
The 496 MiB session is a lower bound on the load's own memory need, so a calibration failure predicts the load would fail too. A blanket fallback would also swallow `Canceled`. Keep failing fast; a single-entry `opts.dma.block_sizes` already skips the screen.

### Tests

**Gaps worth filling**
- **`//llmd:test` does not compile (high).** `monorepo/llmd/models/deepseek41.zig:3214` asserts `!binding.transformed`, and that field was removed from `TensorStore.Binding`. deepseek41.zig's 8 tests are compiled by //llmd:test (main.zig's `refAllDecls` reaches `models.Model`), so the whole target, ~264 tests, fails to build; the 28th pass ran only //zml:test and //llmd:weights_test. Fix: `expect(binding.inputs == null and binding.layout != null)`, or drop the line, then delete the dead `monorepo/llmd/deepseek41_test.zig` and run `bazel test --config=localzml //llmd:test`. The proposed `zig_test` target would not build.
- **The Source-vs-Tensor harness runs zero cases (high).** `zml/zml/io/Source.zig:1019-1258`: `std.meta.declarations` lists only `pub` declarations, and every case is private, so "every Source op places the bytes…" checks nothing. WidenBits is also invalid. Make the cases `pub` (adapt the loop to the 0.17 declaration API) and fix WidenBits. Then add `store.sourceOf(bound).?.rebind(bound, bound.shape())` to `expectCase`, which covers sourceOf, ofBinding, rebind and unbind, and assert `getSourcesById(bound.id) == null`.
- **`Packer.bf16Linear`, `dequantize_block_fp8` and `swapEngramHostPinnedBuffers` have no tests.** The first two run 40×/DeepSeek load. Add a weights_test case on 1 and 2 devices against host dequantization, for both block32 and block128 grids.
- **The quantized `experts()` branches are untested.** Add tests through `experts()` for per-tensor FP8 gate/up (#11), compressed-tensors NVFP4 globals against hand-computed multipliers (#3), and pre-stacked FP8 with `_scale_inv` expecting an error (#32). Land each with its fix; most checks are declaration-time and need no load. Add a `linear.create` NVFP4 test asserting the global and input scales (this fails today, see #10). Extract the per-expert source setup that existing tests copy (55-70 lines each).
- **Measured admission.** No test reaches `.exceeds`, the measured retire loop or the oversize warning. Cheap, ungated: in the test at loader.zig:1146, assert `pending_execution[d] == 2*output + compiledTempBytes` after `loadExecute` and zero after `awaitAll` (same in the test at :1282). GPU-gated: with two submissions pending, inflate both `pending.front().execution[d]` and `pending_execution[d]` by `room+1`, then assert one retire leaves `pending.len == 2`; inflating only `pending_execution` cannot tell retire-until-fits from retire-all. No production seam is needed. Fold execute_admission.zig into loader.zig: each function has one caller, and its tests restate a saturating subtraction.
- **Direct I/O through the VFS has never run in a loader test.** Add a debug assert in the worker read path that the offset and every iovec pointer and length are multiples of the plan's alignment; the existing alignment-64 tests then catch misaligned bounce, gather, arena and lease buffers on any filesystem. Add one wiring test through `vfs.io()` with `.direct_io = .on` over two files, one under 4 KiB, so a gather job mixes alignments. Do not rely on kernel rejection: tmpfs and btrfs serve misaligned direct reads, and a `useDirectIo` skip turns the test into a no-op. For a planner-level stub test, swap `alignmentOf` to check `slot.alignment` before `vfs`.
- **The buffered backend never runs through the front end in CI** (8eb14029 removed the dual-backend fixture; CTX:2136-2138 is stale). Its strided branch (buffered_loader.zig:192-197) and high-latency helper split (:236-257) have never run. Extend its composite test with a strided part and a `high_latency` profile with a small `read_chunk_size`. Add one front-end test that swaps `loader.backend` to buffered on an owned CPU platform.
- **Failure paths that span several actors.** Move the abandonment into `enqueueBlocks`' failure path, deleting `abandonSubmissions`, and test it on TestPipeline. Add one test-only countdown that fails the Nth `TensorTransfer.init` or view, as a case in the existing failure loop, with a 4-device replicated host_pinned spec to cover partial `initHost`. A CPU `transferData` failure test is feasible but needs a real platform, pool and gate, and must not call `setBufferError`. Drop the proposed CPU init-cancellation test, which cannot observe anything.
- **HTTP mock server.** It answers one request per connection with `Connection: close`, so pooled reuse never runs. Add a `receiveHead` loop, then truncate, stall, close-idle and plain-HTTP 302 options. 403 is already injectable via `throttle.status`; extend http_acceptance_test.zig:240 with `retry_once`. HF needs an injectable Hub endpoint before it can be tested at all.
- **S3 and GCS auth.** Add a SigV4 vector, with a `payload_hash` field defaulting to UNSIGNED_PAYLOAD or a botocore vector, including a key with `=` and a space. Test `GCS.init`, not `auto` (err logs fail the test runner), on a full Google key and an ADC file. Test `S3.auto` endpoint precedence, and S3 404/403 via a status mode in the mock server.
- **zio.** No loader or llmd load test runs on llmd's default Io. Extract 3-4 end-to-end bodies into `fn(io)` and also run them under `zio.Runtime` with `.executors = .exact(2)` on CPU and the cuda/rocm legs; callback-thread wakes and pump-lock contention need a GPU. Set `.io_impl` only if llmd does (#4).
- **`auto` probe.** Add a deterministic "two probes sample different offsets" test after #7.
- **examples/io "Loaded weights" timing includes `Loader.init`** (`zml/examples/io/main.zig:317-346`). Move the timestamp and its `defer` after init, log init as its own line, delete the now-redundant `warmupDeviceAllocators` at :309, and update CTX:2044 and :4346.

**Tests and fixtures that do not earn their place**
- **llmd's `weights.testing.pattern`/`expectShards`** (`monorepo/llmd/weights.zig:1757-1801`) duplicate zml's `io/testing.zig`. Export it from `zml/zml/io.zig` and drop the memory argument: it is a no-op on the CPU Fixture, and the `binding.memory` asserts cover routing. Keep Fixture.
- **Assertions that cannot fail:**
  - The HTTP loader test's `pool.high_water <= 4`, its lifecycle lines and the VFS stats (`zml/zml/io/loader.zig:1447-1458`). Keep `physical_bytes` and `throttles > 0`.
  - The deinit-without-executing and input-freeing tests (`zml/zml/io/loader.zig:1221`, `:1264`). Use a `._shards = .empty` sentinel checked after `deinit`.
  - The DMA "dip" case (`zml/zml/io/dma_calibration.zig:629`). Use rates {99,70,100} and expect block 2.
  - `seed += 4` (`zml/zml/io/direct_loader.zig:5582`, `:5687`) shares bytes across runs, and the 256-byte pattern period makes the `rows` shards identical. Use `seed += 5` and a pattern that mixes in `index >> 8`.
  - The n-gram test's `.host_pinned` pass (`monorepo/llmd/weights.zig:1829`).
- **Scheduler white-box tests** (`zml/zml/io/direct_loader.zig:4427-4740`). Merge the 8 into three (order, failure, wake/stop) that assert through `claim`, `waitForWork`, `remainingJobs` and `batch.done`. Keep 4427's regrouping coverage by putting the planner test's sources out of file order. In the planner test, drop the event, request and diagnostics counts and the historical comment, and replace the exact block ranges with a contiguity invariant. Delete `zml/zml/io/host_memory.zig:934-950` and `:952-971`. About 200-240 lines go.
- **Front-end tests that assume the direct backend.**
  - `zml/zml/io/loader.zig:1335` and the HTTP tail at :1458 read `backend.direct` unguarded, which aborts the whole binary on TPU or neuron: guard both.
  - Keep :1310-1339, the only check of the loadExecute ledger.
  - Delete :1341-1360, which duplicates direct_loader.zig:4031-4048.
  - Skip direct_loader.zig:5479 and :5538 on buffered targets instead of forcing `initCpu`.
  - Move the HTTP-throttle test, which asserts direct internals, next to direct_loader or the VFS.
- **Platform-init errors turned into skips** (13 tests: direct_loader.zig:4034, 4053, 4257, 4430, 5311, 5408; ShardGrid.zig:193; dma_calibration.zig:682). Gate on `comptime Target.cpu.isEnabled()` and then `try`. Plain `try` would break the cuda/rocm CI builds, which set cpu=false. Share one `cpuPlatform` helper with `Devices.initWith`'s CPU fallback.
- **The `fusedScheme` test** (`monorepo/llmd/weights/linear.zig:366-382`) restates a one-liner, and its null case cannot happen. Replace it with a fused NVFP4 assembly test (scales `.copy`, global `.broadcast`) and drop the optional return.

### Docs and comments

- `zml/zml/io/loader.zig:99-113`: the `Loader.Binding` doc says inputs are "one per source". They are one per program input; otherwise the tensor as its binding fills it. Reuse the wording at :860-862.
- `zml/zml/io/loader.zig:47-50`, `zml/docs/learn/loader.md:56-58`: the borrow list omits `exe`, and "failure is sticky" overstates the guarantee (#18, #29).
- `zml/zml/io/execute_admission.zig:14`, `:36`: the saturation guards a failed, partly published submit, not lazy allocation.
- Unmeasured admission retires only until no executable submission is pending, not "every pending submission": `zml/docs/learn/loader.md:66-69`, `zml/zml/io/loader.zig:76-79`, `zml/CTX.md:150-152`, `zml/CTX.md:3141-3142`. loader.md:69 "load is never gated" is also affected (#26).
- `zml/docs/learn/loader.md:136-141` omits the exact sorted-runs path for clusters of more than 8 parts with at most 2^21 runs.
- `zml/docs/learn/loader.md:240` lists `UnsupportedPlatform`, which nothing returns. `zml/zml/platform.zig:326-327` says "Some devices won't be used" and then uses them all.
- `zml/docs/learn/loader.md:49-55`, `zml/CTX.md:66`: "each loaded tensor completes one item" is false on the direct backend (#19).
- `zml/zml/io/TensorStore.zig:19`, `zml/docs/learn/loader.md:28-31`: `Binding.memory` applies only to tensors that `load` fills.
- `zml/zml/io/loader.zig:39-42`, `:196-197`: `LayoutError` describes a limitation that no longer exists. The front end has two UnsupportedLayout cases (strided parts into host memory, overlap it cannot prove disjoint); the read-limit case belongs in the backend docs.
- `zml/zml/io/backend.zig:109-111`: `max_part_dims` claims an invariant Source does not have. Define one `TensorStore.Part.max_dims` used by both.
- `zml/zml/io/backend.zig:61-63`: says the backend copies each `LoadSpec`; the direct backend borrows `dims` and `file_uri`.
- `zml/zml/io/direct_loader.zig:1590-1595`: "an io that is no VFS never reads directly". Without a VFS every file is widened; say to pass the VFS's io with a VFS profile.
- `zml/zml/io/direct_loader.zig:1864-1866`: cutRuns boundaries are not about safety. Any cut is correct; the boundaries only avoid splitting a part.
- `zml/zml/io/direct_loader.zig:2515-2517`, `:1541`, `zml/CTX.md:303`, `:368`: these claim byte-fair job order, but the planner uses file order (:1732). Delete "Fairness is decided here once".
- `zml/zml/io/direct_loader.zig:932-934`: `source_items` counts load parts, not sources, which changes how `coalescing_ratio` reads.
- `zml/zml/io/direct_loader.zig:1223-1227`: the `countsAgainstDevice` branch is always true after the `hostResident` return; unwrap it. Rename `TensorTransfer` to `Destination.State` (~15 sites).
- `zml/zml/io/direct_loader.zig:193`: the TMP `created_at` debug log is leftover; delete it, and the CTX notes at 4378-4379 and 5939-5940.
- `zml/zml/io/direct_loader.zig:2699-2702`: `host_iovecs` also serves the DMA `ScatterSink`. Merge it with `iovecs`, sized `@max(maximum_blocks, max_positional_iovecs)`; the uses never overlap. `zml/docs/learn/loader.md:360-361` should say "width + 1 requests plus the DMA reserve".
- Host memory docs:
  - `zml/zml/io/host_memory.zig:297-299`: arenas are pregrown once at init; loading does not grow them.
  - `:352`: drop "when no arena growth is running".
  - `zml/zml/io/dma_calibration.zig:115-118`: calibration does not warm the allocators; `Loader.init` does.
  - `zml/zml/io/backend.zig:24-27`: the width is clipped to the mapped ceiling minus the DMA reserve and one spare request.
  - `zml/docs/learn/loader.md:343-344`: interleave is limited to the task's memory-policy nodes.
- `zml/zml/io/backend.zig:41-43`, `zml/vfs/direct_io.zig:26-28`, `:107-109`, `zml/CTX.md:2662`: "cold stays cold" is false (#7).
- `zml/vfs/request.zig:157`: "Cancellation propagates" is false for the head and body stages.
- Two different `Binding`s meet in loadExecute. Rename `BoundExecutable.binding` to `sources` and `getSourcesById` to `bindingOf(tensor)`. A top-level `binding` would shadow locals at TensorStore.zig:117 and :139; `Loader.Program` collides with llmd's `Program`, and `Execution` with `pending_execution`.
- `monorepo/llmd/models/qwen3_5.zig:1637`: a stale borrowing comment. Remove it together with the redundant await at :1638, and the same await at `monorepo/llmd/models/qwen4.zig:320`. Keep models.zig:239.
- `monorepo/llmd/weights.zig:1184-1205`: the swizzle paragraph belongs on `swizzleBlockScales` and wrongly calls the `Tensor` path test-only (it is the default `.device` path). `layOutExperts` mentions a `Source` it never receives: take `Tensor`, or inline it into `packLayOutExperts`.
- `monorepo/llmd/weights.zig:1566-1567`: `MissingFusedComponent` reports a broken Packer invariant. Use `stdx.debug.panic` naming the tensor.
- `monorepo/llmd/weights.zig:749-753`: `concatenateRows`' dtype conversion has no production caller. Remove it, with `Recipe.ConcatenateRows`, `packConcatenateRows`, the compile arm, the test tail at :1856-1862 and the uncalled `convertRows`/`dequantizeBlocks`.
- `monorepo/llmd/models/qwen4/model.zig:595`: the TODO duplicates spec.zig:161; delete it.
- `zml/zml/io/log.zig:1-4`, `zml/docs/learn/loader.md:225-228`: the debug-only validation policy (#30).
- `zml/zml/io/loader.zig:163-165`, `zml/docs/learn/loader.md:79`: "deinit awaits" must change if #18 lands.
- `zml/vfs/range_read.zig:175-176`, `zml/CTX.md:231-232`, `:692-693`: update if #43 lands.

## Part 2: Simplification and performance

### How the loader reads today

The loader turns about 1e5 safetensors byte ranges into placed device or host buffers, and runs a few llmd pack programs for tensors that have to be computed rather than copied. The byte-moving core holds up: the Source algebra, ShardGrid, the per-file planner, FIFO claims, the per-device pumps and the sticky-error drain. Reviewers traced it on every success and failure path and found no wrong-byte path, no leak and no deadlock. The excess is at the seams:
- One byte run has three types and two encodings of "whole tensor".
- The same parts are validated two or three times, and the loader re-proves a disjointness that Source already guarantees.
- Pinned memory is bounded twice.
- `Pipeline` is a second handle on `Loader`.
- Three dispatch layers only forward.
- llmd hand-builds the tensor-to-program mapping that the store does not record, and with it a call-order rule.

Anchors are relative to `zml/zml/io/` unless they start with `monorepo/`. Steps marked **[forwards]** pass the call on and add nothing.

1. **Store.** `monorepo/llmd/main.zig:353` `TensorRegistry.fromRepo` parses every shard header, one after another. `main.zig:393` `TensorStore.fromRegistry` (`TensorStore.zig:84`) wraps the registry, the meshes and an id→Binding map.
2. **Loader.init, in parallel with model init.** `main.zig:422-423` → `loader.zig:115` → `Backend.init` (`backend.zig:615`) **[forwards: picks direct or buffered by target]** → `direct.Loader.create` (`direct_loader.zig:122`). `Sizing` (`:366`) calibrates the block size, pre-grows the pinned arena, builds the BlockPool, the read gate and the credits, and starts width+1 workers.
3. **Binding.** `Model.init` binds every tensor:
   - plain tensors through `View.createTensor` (`TensorStore.zig:209`), which is a second copy of `Source.bind`;
   - fused, padded, stacked and swizzled tensors as Source expressions, lowered by `Source.bind`/`bindAs` (`Source.zig:498-504`, `Lowering` `:584`) into Layout parts;
   - computed tensors through `Source.bindProgram` (`:566`), plus a `Packer.recipes` entry that llmd pairs with it by hand.
4. **Packs.** After the model compiles, `monorepo/llmd/models.zig:236` → `weights.loadInto` (`weights.zig:1665`) → `loadPacked` (`:1476`). It dedupes and compiles the programs (`program` `:1556`, `compileRecipe` `:1611`) and publishes groups of at most 8 GiB through `loader.loadExecute` (`:1535`).
5. **loadExecute** (`loader.zig:239`) checks the executable against the binding. `admit` (`:319`) retires and runs older packs until the room fits. Each input is lowered to a LoadSpec (`appendCompositeParts` `:674`: expand, sort, **checkParts #1**) and marked `delivered` (`:293`).
6. **Bulk.** `weights.zig:1684` → `loader.load` (`loader.zig:205`) → `prepareModelLoad` (`:527`). One model visit only counts parts (`loadPartCount` `:654`), a second builds the specs. Program tensors not yet delivered are refused (`:588`).
7. `Loader.submit` (`loader.zig:416`) adds placement bytes to the admission ledger → `Backend.submit` (`backend.zig:653`) **[forwards]** → `direct.submit` (`direct_loader.zig:221`).
8. `createDestinations` (`:501`) checks the parts again (**checkParts #2**), interns one SourceSlot per file and builds a ShardGrid per destination. `publishFiles` (`:1605`) sorts the parts by (uri, offset), decides O_DIRECT per file, and runs `preparePlan` (`:1657`) per file. Each plan is published as soon as it is ready (`Scheduler.publish` `:2575`), so workers read file 1 while the rest is still being planned. `startAllocations` (`:266`) creates host destinations on a side task.
9. `preparePlan`: `cutRuns` (`:1844`) coalesces touching ranges into jobs of at most one request, cut where no tensor is split. `placeDma` (`:2001`), `stage` (`:2033`), `appendGatherJobs` (`:2157`) and `appendFillJob` (`:2336`) write exact transfer records.
10. `workerMain` (`:604`): credit → `claim` (`:2661`) → `ReadRequest.run` (`:2790`).
    - `ensureState` (`:1079`) creates the PJRT managers on first touch (`TensorTransfer.init` `:1226`, under LazyOnce).
    - The worker leases blocks, a second bound on the memory its credit already reserves, then reads through `readGated` (`:3013`) or `landJob` (`:3071`). Host destinations go through `runHost` (`:2881`).
    - Request code reaches its state through both `loader` and `Pipeline` **[Pipeline forwards: 9 of its 13 fields point back into Loader]**.
11. `enqueueBlocks` (`:3643`) queues transfer×writer pieces on the per-device pumps. `pump` (`:3716`) submits within the byte and piece budget, and `submitTransfer` (`:3755`) flags the piece that completes each target.
12. **Completion.** The OnReady callback → `eventCompleted` (`:3822`) pumps more → `retireEvent` (`:3631`) → the lease is released → `ReadRequest.release` (`:2995`) returns the credit → `Batch.finishJobs` (`:998`). The last unit sets `done`.
13. `awaitAll` (`loader.zig:299`) → `retireOldest` (`:445`) → `Submission.await` (`backend.zig:679`) **[forwards]** → `awaitBatch` (`direct_loader.zig:305`): wait, mark unclosed targets as failed, `retireBatch` (`:3604`), `destroyBatch` (`:590`).
14. The front end runs the pack (`BoundExecutable.execute`, `loader.zig:818` → `Exe.call`), writes its output and frees its inputs.
15. llmd awaits again (`weights.zig:1685`, `models.zig:239`) and frees the programs. `main.zig:473` `loader.deinit` → `direct.destroy` (`:337`).

### Simplification plan

| # | Move | Lines out (approx.) | Runtime change | Risk |
|---|---|---|---|---|
| 1 | Host-resident destinations share one allocation; two memory kinds | 90-110 | none on shipped targets | very low |
| 2 | One bound on pinned memory: the credit | 250-300 | one lock per request instead of two; nothing per block | low |
| 3 | Validate once at the front door; Source owns disjointness | ~300 | buffered size errors arrive at `load` | low-medium |
| 4 | Direct loader: one root object, plans that say what they mean | 200-250 | none | low |
| 5 | Fixed DMA block size per target *(your decision)* | 850-900 | −0.7-1.25 s per GPU init | medium |
| 6 | llmd: the Packer carries only model decisions | 300-400 | none (fixes a latent use-after-free) | low |
| 7 | One byte-run representation, one lowering | 150-180 | ~25 MB arena at 1e5 tensors | medium (public API) |
| 8 | Buffered backend: one file per URI, a few workers per batch | ~100 net | remote TPU loads: minutes → seconds | medium (no CI coverage) |
| 9 | Admission ledger in the backend that allocates | 60-70 | none | low |
| 10 | VFS remote backends become stores | 850-950 + ~250 small | uniform error mapping | high (largest change) |

#### 1. Host-resident destinations share one allocation; only `.default` and `.host_pinned` exist

- **Changes.**
  - `initHost` always shares the allocation of a replicated region. It reads the regions off `grid.writers`: a region's first device is `@ctz(mask)`, and every region has `grid.replicas()` devices.
  - `HostShard` becomes a data pointer. A host transfer lands at `host[@ctz(writer_mask)] + destination_offset` and adds `len * @popCount(writer_mask)` to `host_written`.
  - `supportsMemory` accepts `.default` always and `.host_pinned` on direct targets. `countsAgainstDevice` becomes `kind == .default` and `hostResident` becomes `memory == .host_pinned`.
- **Disappears.**
  - The configuration: `share_host_allocations` and `sharesHostMemory` (`direct_loader.zig:117-120, 166, 1500-1505`). Its rationale (portable pinned memory, the oneAPI view check) moves into the `initHost` doc.
  - The write machinery: `HostShard.writer`/`covers`, `HostWriters`/`hostWriters`, `HostScatter.Copy`/`copies`/`copy_count`/`addCopy`, `Scratch.host_copies`, `hasRoom`'s argument, and the `while (writers.next())` loops in the fill and bounce branches.
  - The O(n²) `sameSlices` region scan (`:1303-1312`, `:1533-1539`) and HostResidentTest's `share` dimension.
  - The `.device` and `.host_unpinned` arms.
- **Costs and risks.** Nothing changes on the shipped targets: production already shares, and every host write is one memcpy. Two things do change and should be stated:
  - TPU `.host_unpinned` delivery through `Buffer.from` goes away. It was untested and accidental.
  - A future direct target that cannot share would get the copy path back from git history.
  - Update `loader.zig:2119` and `:2210`.
- **Conflicts resolved.** The tests verifier pointed out that `.host_unpinned` would reach the copy path if TPU moved to direct. Narrowing the memory kinds closes that, so sharing then holds by construction and only ever for pinned memory.
- **Absorbs:** direct-front "replica copies run only in tests", story "unshared host-allocation path", tests "dead production mechanism", gap1-1 "four memory kinds".

#### 2. One bound on pinned memory: the credit

- **Step 1: pure deletion, no runtime change.**
  - `acquireMany` becomes lock, assert that enough blocks are free, pop.
  - Write the invariant once, next to the credit computation (`direct_loader.zig:475`): a credit holder leases at most `maximum_blocks_per_job` blocks and returns them before the credit.
  - Delete the wait loop, the condition and its broadcast, `closed` and `close()`, `AcquireError`/`RequestExceedsCapacity` (also in the public `LoadError` and `loader.md:246, 252-253`), `pool.close` in `recordError`, and `block_wait_ns` with its log field, along with the close and can-never-fit tests.
  - Keep the `errorValue()` re-checks at `:2848`/`:2952`: they also guard the window while a tensor is being initialized.
- **Step 2: a credit is a slot of `maximum_blocks_per_job` blocks.**
  - Take the slots from the existing per-block carving, not as contiguous runs per arena. Per-arena runs would lose a slot per arena and break `retained_credits >= width + 1`.
  - The credit gate becomes a free stack of slot indices under its existing mutex.
  - `ReadRequest.pending = 1 + sum(queue_counts)`. `ReadyTransfer` and `EventContext` carry the request and a data slice. The final release returns the slot and the credit, then calls `finishJobs`. An enqueue failure becomes one `fetchSub`.
- **Disappears.**
  - BlockPool: free list, mutex, `in_use`, `acquireMany`/`releaseMany`/`release`, `detachBlock`.
  - `Lease`, the lease in `BlockContext`, `registerBlock`, `abandonSubmissions`, the `references` scratch and its counting loop, `plan.blocks`, `scratch.leased`.
  - One refcount layer: Lease → ReadRequest → Batch becomes ReadRequest → Batch.
  - In total about 250-300 lines, of which about 100 are tests.
- **Costs and risks.**
  - A broken invariant now trips an assert, and is UB in ReleaseFast, instead of blocking. That is acceptable because sizing fixes the invariant.
  - `RequestGate` is shared with the read gate, so the credit gate becomes its own small type (about 20-30 lines).
  - `pool.high_water` becomes the high-water mark of credits in use.
  - Performance improves slightly: one lock per request instead of gate plus pool, and no pool mutex or broadcast per block completion on PJRT callback threads.
- **Conflicts resolved.**
  - Rejected: execution's option B (drop the request gate and let the pool wait). Fills from the zero block would then enqueue without bound, request depth would go past anything measured (CTX's 24.3 vs 43.8 GiB/s shows depth matters), and the pool would need FIFO waiters.
  - Keep the read gate. It makes `read_parallelism` an exact count, which the docs promise and the buffered backend also enforces. Width+1 workers without a gate would allow N+1 reads, so that option is rejected. `workers = width` with no gate is acceptable only after an A/B run on a host_bounce load.
  - Meanwhile add a comment at `direct_loader.zig:142` saying the spare worker stages the next request and copies outside the gate. Also fix CTX.md:3446, PLAN.md:29/133/193 and loader.md:361.
- **Absorbs:** execution "credits and the pool both bound the blocks", memory "BlockPool waits never do anything", story "BlockPool's wait can never fire", concurrency "credits already bound pinned memory", execution "read gate no longer earns its place".

#### 3. Validate once at the front door; let Source own disjointness

- **Changes.**
  - Run one validation pass in `Loader.submit`'s existing spec loop (`loader.zig:422-425`), which both `load` and `loadExecute` go through. It covers:
    - EmptyTensor;
    - `checkParts` on every spec, plain ones included (a single O(1) part);
    - the refusal of strided parts into host memory;
    - `Sharding.Placement` for every memory kind, so IncompatibleSharding is logged once with the tensor's shape.
  - Backends trust the LoadSpec contract (`backend.zig:73-76`). The direct backend keeps only its read-limit check, because it needs `plan_config`. That check moves to a Planner function next to `appendExtent`, with `read_limit` computed once, and `appendExtent`'s own check becomes an assert.
  - `checkStridedParts` keeps only its first loop: dims, `dstOrdered`, spans in bounds, sorted starts, covered ≤ destination, SourceSizeMismatch.
  - Add a double-write check to `TestCheckpoint.assemble` (`Source.zig` ~988), so Source's byte-level tests catch a future overlapping piece.
- **Disappears.**
  - The prover: `checkCluster`, `overlaps`, `Footprint`, `OverlapSearch`, `max_listed_runs`, `Lattice.hasRunIn` (`backend.zig:396-572`, ~175 lines) and their tests (`:734-830`, ~95 lines).
  - `checkParts`' allocator parameter.
  - The duplicate checks: `loader.zig:691-695, 744-747`, `direct_loader.zig:518-530, 548-551`, `buffered_loader.zig:156-160`.
  - The backend branch in `loader.zig:1530-1549`, the caveat at `loader.md:232-233`, and the "unprovable layout" meaning of UnsupportedLayout.
  - About 300 lines in total. No more 32 MiB run sorts during planning.
- **Costs and risks.**
  - On the buffered backend, SourceSizeMismatch now fails at `load` and leaves the loader usable, as on direct. The test of buffered stickiness needs a different worker failure, for example a truncated file.
  - This reverses CTX:5024-5026, a decision made when the check was a linear scan.
  - The one real risk: the direct gather path's `zeroed = covered < end - start` (`direct_loader.zig:2229`) assumes disjoint parts. A hand-built overlapping Layout would leave stale bytes in gaps instead of failing with UnsupportedLayout. Nothing in production builds such a layout.
- **Conflicts resolved.**
  - "Make `putSourcesNoClobber` private" cannot work in Zig: Source.zig calls it at `:518` and `:578`. Keep it `pub` with a doc line ("a Layout must come from Source"). Delete only the two hand-built strided overlap cases in `loader.zig:1971-1985`.
  - Two other placements were proposed. Deleting the front-end copies and keeping the backend ones breaks buffered's promise to fail before publication. Deleting the backend copies while keeping the front-end ones leaves plain specs unchecked. A single site covering every spec is the chosen fix.
  - Never write `std.debug.assert(checkParts(...))`: the call allocates and is evaluated in release builds.
  - If a runtime guard against hand-built overlaps is wanted, replace the prover with an exact k-way merge over `dstOrdered` cursors (about 30 lines, no 2^21 cap, no UnsupportedLayout). Do not keep today's prover. This supersedes the planner's "merge only above the cap".
  - Keep `validateOptions`' `read_chunk_size` check: the buffered backend divides by it.
- **Absorbs:** source "overlap proof re-proves what Source guarantees", story "re-proves twice per spec", planner "quadratic cliff above 2^21 runs", frontend "validation two or three times", source "front end reruns the backends' validation", direct-front "spec validation in both layers", story "validation two to four times", tests "checkParts in three places".

#### 4. Direct loader: one root object, plans that say what they mean

- **Changes.**
  - **Error state on Loader.** `first_error`/`failed`/`errorValue`/`recordError`/`checkOpen` move onto `Loader`, next to `stopWorkers`, which closes the same gates.
  - **DMA stage stays in place.** It stays where it is in the file as a `Dma` section: the nested types plus `enqueueBlocks`/`pump`/`submitOne`/`submitTransfer`/`eventCompleted`/`retireEvent`/`abortReady`/`retireBatch` as functions over `*Loader`. Pumps, the DMA budget and the zero block are held without back-pointers.
  - **One handle per request.** `ReadRequest` holds `loader`, and `run(self, scratch)` takes no second handle. There is one `block_size`: `plan_config`'s.
  - **One comment in `recordError`.** It says why failing other batches from a callback thread is safe: claims are FIFO, retirement follows publish order, and `destroy` waits for the worker group.
  - **Scheduler.** `queue` becomes a `std.Deque`. `claim` returns a plain `{plan, index}`, and ReadRequest stores the index and gets `job()`/`transfers()`. `sealed_at` is stamped once, after `publishFiles` returns, and `published_at` keeps its stamp at the first publish.
  - **Diagnostics.** `logBatch` derives the plan, transfer, event and job-kind counts from `batch.plans` in one loop, behind a check that debug logging is enabled.
  - **Allocation task.** It walks the plans' host jobs in claim order, the same loop as `runHost`.
  - **Gather jobs.** Items and extents pair index for index, so `landJob` loops over them pairwise.
  - **Job length.** The scatter term goes. `maximumJobLen` keeps only the alignment rule, `read_limit = request_size`, and Sizing uses plain divisions.
- **Disappears.**
  - `Pipeline.init` (11 arguments) and its 9 forwarded fields, plus the dead `Loader.calibration`.
  - `Scheduler.seal`, `Claim` (5 accessors), `head`/`popHead`, `remainingJobs`, `finishScheduling`.
  - 4 JobCut counters, 4 Plan fields, about 8 Diagnostics fields and about 11 lines under the scheduler mutex.
  - `Batch.host_destinations`, `Destination.queued_ahead`, `hostDestinationsInOrder`.
  - `Job.data_offset`/`Extent.data_offset` and `stage`'s `data_start`; `Item.extent` and its cursor loop.
  - `limits.maximumCoalescedJobBlocks` with its test and the iovec plan case, and three dummy `Config`s.
  - About 200-250 lines in total.
- **Costs and risks.** No behavior or performance change. Tests churn: TestPipeline becomes a partly undefined Loader (about 7 tests), and the diagnostics tests count from the plans.
- **Conflicts resolved.**
  - Rejected: moving the ~470 lines of DMA code into Loader's body. It breaks the file's top-down order (CTX:2267).
  - Rejected: `@fieldParentPtr`.
  - Rejected: `claim` returning an initialized ReadRequest. The scheduler would then need to know the pipeline.
  - For the job bounds, the verifier's leaner version wins over precomputed Config fields, which would churn about 10 test configs.
- **Optional, unverified.** Make the job kind a tagged union (`dma_read`, `dma_staged`, `dma_gather`, `dma_fill`, `host_read`, `host_fill`), with `stage()` emitting its Extent. That would remove the `fill`/`gather`/`delivery` flags, `Plan.direct_reads`, landJob's staged branch and several debug asserts. No verifier checked it.
- **Absorbs:** the three Pipeline findings (direct-front, execution, concurrency), direct-front "diagnostics passed through four layers", execution "Scheduler surface wider than its job", direct-front "host_destinations restates the plans", execution "data_offset is write-only", execution "items are 1:1 with extents", planner "job-length bounds", planner "seven job kinds in five flags" (optional).

#### 5. Fixed DMA block size per target *(a decision for you: it reverses a recorded one)*

- **Changes.**
  - Per-target constants sit next to `max_dma_pieces_per_device`: ROCm 16 MiB, oneAPI 8 MiB, CPU 4 MiB, CUDA 8 MiB.
  - CUDA 8 MiB is the choice that loses least: it keeps today's 8 MiB local request size and is about 1.5% behind 16 MiB on GB300 DeepSeek. The RTX 5090 screen picked 2 MiB, and CTX:3402 considers a CUDA table refuted, but no load was measured at another size. Alternatively, measure one 5090 load at 16 MiB first.
  - Keep a nullable `block_size` override in `Options` for experiments.
  - Sizing becomes direct: `available = max_mapped_bytes / block_size`, one allocation for the reserve and one for the rest.
- **Disappears.**
  - `dma_calibration.zig` (740 lines, ~150 of them tests), unless it is kept as an examples/io tool. In that case most of those lines stay.
  - `Options.dma`, `Loader.calibration`/`Sizing.calibration`, `limits.max_dma_parallelism`.
  - The Workspace growth API (`findArena`, `usableBlocks`, `growToBlocks`, pregrow) and its 3 tests.
  - The homogeneous-device precondition, and with it `Workspace.init`'s `unreachable` on mixed-GPU hosts.
  - The `ZML_DMA_BENCH_*` plumbing. About 850-900 lines in total.
- **Gains.**
  - 0.7-1.25 s off every GPU `Loader.init` on PCIe hosts.
  - Request size, DMA budget and pinned footprint become deterministic per target. The 19th pass had to "compare arms at the same block size only".
  - No 496 MiB transient on device 0 during model init.
  - About 15-40 s off each GPU test run.
- **Costs and risks.** A host's optimum may differ by about 1-2%, the recorded effect of the screen's run-to-run swings. The first H2D setup moves into the first load: unmeasured, probably small.
- **If declined, do the minimum.**
  - Delete `findArena`: it always returns null in production. About 25 lines; the two pointer-identity asserts switch to `backend.arenaAt`.
  - Tests use a `load_testing.fixed_dma = .{ .block_sizes = &.{4 << 20} }` constant. A one-element list already skips the confirmation windows, so no production shortcut is needed.
  - Let `calibrate`'s CPU branch honor a one-element list, so CompositeTest's scenarios can run end to end. Do not skip measurement on accelerators: that breaks `dma-bench`.
- **Absorbs:** memory "calibration only picks a block size", tests "every accelerator test loader calibrates", tests "findArena always returns null", consumer "calibration under NCCL contention" (moot; reordering would still overlap the compile).

#### 6. llmd: the Packer carries only model decisions

- **Changes.**
  - `concatenateLinears`/`interleaveLinears`/`assembleLinears` become free functions, and `interleaveLinears` takes `[]const linear.View`. Port `Interleaving.explicit`'s block-boundary checks by calling the file-private `assembleLinears` from tests.
  - llama, gemma3_text, lfm2, ministral3, muse_glimmer and the dflash/dspark drafters drop their Packer, and `models.zig:236` calls `loader.load` directly.
  - One `Packer.bindProgram(inputs, output, recipe)` replaces the 8 hand-paired `bindProgram` + `recipes.put` sites.
  - **One owner for cleanup.** `defer model.deinit()` goes right after `model.init` in main.zig. Delete the 6 errdefers in `loadBuffers` (3 can never run; the other 3 become redundant). Delete the awaits at `qwen3_5.zig:1638` (with its stale comment) and `qwen4.zig:320`. Keep `loadInto`'s await (it protects the programs) and `models.zig:239` (the dispatch owns completion).
  - Add `errdefer loader.awaitAll() catch {};` after `defer deinitPrograms` in `loadInto`, plus a zml doc clause that submissions borrow their executables until retired. That closes the latent use-after-free.
  - `splitTensor` becomes `Source.split` + `bind` at its two call sites.
  - Start one "Loading weights" node in models.zig with `defer end` and pass it to every model.
- **Disappears.**
  - Dead or test-only code:
    - `convertRows` and `dequantizeBlocks` (no callers, not even tests);
    - the dtype-converting branch of `concatenateRows`;
    - `finalizeLoadedBuffers`;
    - `sameLoadShape` (use `Shape.eqlWithTagsAndPartitioning`);
    - `Program.compiled`/`exe` (use a pointer capture on `future.result`);
    - `loadInto`'s `io`/`meshes` parameters;
    - the `recipes.contains` check in `swizzleBlockFp8Scales`;
    - Recipe's duplicated `output_shape`.
  - Plumbing:
    - three of the eight Recipe variants;
    - the Packer field, init, errdefer, deinit and about 70 parameter threadings in 7 model files;
    - six per-model progress starts and laguna's root total.
  - About 300-400 lines.
- **Decision for you.** `ExpertLayout.loader` has a measured reason recorded in CTX (no staging memory, better on hosts with few cores) but no CLI or env switch. Either expose it, or delete it together with its `.interleaved`/`.swizzled` arms and half of two expert test matrices. Deleting it also lets `layOutExperts` and `swizzleBlockScales` drop `comptime T`.
- **Conflicts resolved.**
  - Keep `concatenateRows` (`qwen4/model.zig:627` calls it); only its conversion branch goes, which overrides story's "delete it".
  - Defer moving the program into the zml binding. The direction is right, but it adds compile, dedupe and executable ownership to zml. Deduping on raw argument bytes is unsound, because args containing a Shape have undefined BoundedArray tails. It also changes the "caller owns executables" contract. Revisit only if `loadExecute`'s public surface is to go.
  - Keep the `delivered` check. Moving it to `awaitAll` adds about as much code, loses fail-fast, and buys a bulk-first order that is unmeasured and works against admission: the first group would be admitted unconditionally, and a group that does not fit would wait for the whole bulk.
  - Starting pack compiles in `Model.init` is covered under Performance: measure W first.
- **Absorbs:** consumer "Recipe/Packer hand-build a closure" (llmd half), consumer "FusedLinear builders live on Packer", consumer and story "dead or test-only generality", consumer "load/cleanup spread over four layers", concurrency and tests "loadInto frees executables", consumer "progress" (llmd half), gap2-1 "splitTensor", frontend and story "`delivered` forces packs first" (resolved: keep).

#### 7. One byte-run representation, one lowering

- **Changes.**
  - `TensorStore.Part.source` becomes `*const safetensors.Tensor`, the arena copy that `View.source` already made. The backends take this type as their part.
  - Every non-program binding has a Layout. A plain tensor is one whole part whose shape is the bound shape; program bindings keep `inputs`.
  - `Lowering` becomes one loop over the pieces calling `lowerPiece`, and sorts the parts by `dst_offset` once.
  - `View.maybeCreateTensor` becomes `maybeSource(...).bind(...)`, and `createReplicatedTensor` becomes `source(...).bind(.replicated, ...)`. `Source.place` relies on `Shape.withPartitioning`'s dispatch.
  - In the front end, `zero_fill` is true for every spec, since a whole part has no gaps. The existing `layout.shape.byteSize() != shape.byteSize()` check (`loader.zig:683`) becomes the plain-tensor size rule and returns SourceSizeMismatch rather than InvalidLayout, so the public error keeps its meaning. After this move, `checkParts`' `zero_fill = false` branch has no front-end caller.
  - An empty tensor gets a one-line `layout.shape.byteSize() == 0 → EmptyTensor` check before the no-parts check. That also fixes binding an empty tensor through `bind`/`bindProgram` today.
  - The parts list grows as needed. Each spec records its first part index, and one pass sets the slices at the end (or each call uses an arena). The expansion loop uses `Lattice.at`.
- **Disappears.**
  - The duplicate types and conversions: `backend.LoadPart`, `LoadPart.whole`, `Layout.whole`, the index→pointer translation and its InvalidLayout case, about 7 null-layout branches, `ofBinding`'s plain branch, and `Part.chunks` (it duplicates `Lattice.runs`).
  - Lowering's index map, tensor list and second copy of every source tensor.
  - The View copy of `Source.bind`: `maybeCreateTensorInternal` and `placeShape`.
  - `createHostPinnedTensor`/`maybeCreateHostPinnedTensor`: public API with no production caller.
  - The counting pass: `loadPartCount`/`layoutPartCount`, the counting visit and both exact-capacity asserts. Today, if the count ever drifted from the fill, `appendAssumeCapacity` would overflow unchecked in ReleaseFast.
  - About 150-180 lines net.
- **Fixed on the way.**
  - `loadExecute` checks a relabeled plain binding against the checkpoint's shape (`loader.zig:879`).
  - `Layout.whole` indexes `tensors[part.source]` without a bounds check.
  - View panics on tags combined with a Partitioning.
  - Independent one-liner that can land first: move `admit` to after the specs are built in `loadExecute`.
- **Costs.**
  - About 23-35 MB more arena at 1e5 tensors.
  - llmd's FP8 scale loop (`weights.zig:196`) and `logNotDelivered` iterate part sources and may see a tensor twice; dedupe them locally.
  - The tests at `loader.zig:1089, 1174, 1517, 1526` and `docs/learn/loader.md:29` change.
- **Conflicts resolved.**
  - Keep the 64-run / 1 MiB expansion policy in the front end. Moving it into `createDestinations`, as the source and story findings proposed, regresses buffered: for runs of 1 MiB or more, the expansion reads straight into staging instead of a whole-span heap copy. Only the counting goes. Buffered's per-run opens are fixed in move 8.
  - Drop the `wholeTensor` fast path. It saves tens of ms and adds a second "whole" predicate.
  - `zero_fill = !isWhole` is unnecessary.
- **Absorbs:** source "Binding/Layout/Part restate Source and LoadPart", source "View's createTensor family reimplements Source.bind", frontend "lowering written twice with a counting pass", source "front end decides the planner's expansion policy" (counting half), story "three representations".

#### 8. Buffered backend: one file per URI, a few workers per batch

- **Changes.**
  - Hoist `SourceSlot` and `LazyOnce` into backend.zig and use them in both backends.
  - Buffered `submit`, which is single-threaded, resolves one slot per part and stores the pointers next to its copied parts. `loadOneStaged` reads through `TensorReader.initBorrowedPositional(io, source, try slot.ensure(io))`, and `destroy` closes the slots after the group finishes.
  - Each batch runs `W = min(width, specs.len)` workers through `concurrent`, pulling from a `fetchAdd` cursor, in a per-batch `std.Io.Group` that `awaitBatch` awaits with cancelation blocked.
  - Strided parts reserve `tensor_bytes + max strided span` and use the tail of the staging buffer as the source buffer.
- **Disappears.**
  - Per-part opens and closes.
  - `pending`/`done`/`finish` and the batch sentinel, plus the per-spec closures.
  - `stdx.Io.LimitedGroup` (`stdx/Io.zig:25-239`, about 100 lines and 5 tests; this is its only user).
  - The uncounted alloc/free per strided part.
  - About 100 lines net.
- **Gains.**
  - One HEAD (or one HF redirect chain) per shard file instead of per part; see Performance.
  - `submit` never loads tensors inline.
  - Thread count equals the width, not the CPU count.
  - A canceled admission can no longer skip a completion unit (a latent hang in `awaitBatch`).
  - Strided spans stay inside the staging budget.
- **Costs and risks.**
  - Two overlapping batches may run 2W tensors. Reads and bytes are still bounded by ReadPermits and StagingAdmission.
  - Sharing one HF handle across tensors widens the existing reresolve use-after-free (`vfs/hf.zig:869-930`). Fix it in the same change: keep superseded URLs until close, or copy the URL under the lock.
  - Nothing in CI runs the buffered backend end to end. Add a CPU test that builds it directly.
- **Rejected.**
  - A mutex-guarded map filled on first use: it serializes remote opens and races across batches.
  - "Read runs straight into staging when `srcTiled`": that means one positional read per run, which is exactly what the expansion thresholds avoid.
  - A VFS-level HEAD cache: it adds cached state to three backends.
- **Absorbs:** memory and story "buffered opens per part", memory "one LimitedGroup task per tensor", memory "uncounted strided span".

#### 9. Admission ledger in the backend that allocates

- **Changes.**
  - The direct backend keeps two monotonic per-device counters: `submitted`, counted once the batch is sealed, after `createDestinations` succeeds, and `allocated`, counted in `TensorTransfer.init`. It exposes `unlandedBytesPerDevice = submitted -| allocated`.
  - `pending_execution` (in Loader) and `execution` (in PendingSubmission) become `[MAX_NUM_DEVICES]u64`. `room`, `unlanded` and `inputs` become stack locals in `admit`/`measureFit`.
  - Temporaries are charged as one per-device high-water mark instead of a sum.
  - Inline `execute_admission.zig`'s `room`/`admits` and their tests into loader.zig.
  - `readRoom` reads `self.backend.direct` directly, and a single `backendFor(target)` serves both `Backend.init` and `supportsMemory`.
- **Disappears.**
  - The cross-layer invariant: the front end's `submitted_bytes` and its accounting loop in `submit` (placement validation now lives in move 3's loop), and the equality tests at `loader.zig:1216, 1338, 1716`.
  - The heap arrays: `Scratch` (four slices), three init allocations with their errdefers and frees, and the per-call `execution` allocation.
  - `execute_admission.zig`, `readRoom`'s dead buffered arm and the duplicated platform list.
  - About 60-70 lines, plus the rule that two files must compute the same expression to the byte.
- **Costs.** Multi-object `for` loops need `[0..n]`, and `out[device.id]` loses its bounds check above n (mesh ids are already validated in `preparePlan`). The separately reported VMM 2 MiB rounding fix becomes one expression in one file.
- **Rejected.**
  - An `Admission` struct inside execute_admission.zig: it adds an abstraction and contradicts the file's "pure arithmetic" purpose.
  - Subtracting the remainder in `destroyBatch`: the monotonic pair is simpler, and the skew after a failure does not matter because failure is sticky.
  - `Backend.supportsMemory` as a method: it threads the backend into `prepareModelLoad` and 3 tests.
- **Absorbs:** frontend "admission ledger split across layers", frontend "temporaries summed", frontend "which platforms use which backend". The admission tests are kept, because they are the only coverage on CPU runs.

#### 10. VFS remote backends become stores

- **Changes.**
  - Each remote backend implements `stat(path) → {kind, size}`, `read(path, size, data, offset)` (one governed range read) and an optional `list(prefix)`.
  - The VFS Handle gains `remote: ?{ path (owned), size, pos }` and answers stat, length, seek, realpath, close and directory reads itself.
  - Status-to-error mapping lives in one request.zig helper.
  - HF keeps its trees and a path-keyed URL cache that never frees a URL a read is still using.
- **Disappears.**
  - Four handle tables, their closed lists and four 17-entry std.Io vtables, plus the HTTP and S3 handle mutexes.
  - `Handle.type` and the duplicated `parseListObjectsResponse`/`EncodeIsValid`.
  - About 850-950 lines net.
- **Fixes on the way.**
  - Create and delete on s3:// or hf:// paths currently fall through to the local disk.
  - Streaming errors are mapped to EndOfStream.
  - `fileSeekBy` underflows in HTTP and HF.
  - Four divergent `dirOpenFile` mappings; this is the root of S3 and https reporting a missing file as `Unexpected`.
- **Costs and risks.** This is the largest change in the plan. `http_acceptance_test` must go through a VFS. VFSBase stays, because the VFS itself needs its forwarders. Do it in stages, standalone deletions first:
  - GCS dead code (about 115 lines): `EndpointUrl`, `JwtAssertion`/`makeAssertion`, the commented `refreshServiceAccountToken` body, `getTimestamp`, the commented `authHeaders`, the region fields. Reject service-account key files at init rather than implement them; std has no RSA. Fetch the metadata token as a plain request, so a 404 is logged instead of surfacing as a bare `Unexpected`.
  - S3: one `sign` helper with an embedded buffer, and `auth_service` as a constant (about 20-25 lines net).
  - `ReadStatsProvider` becomes `?*const AtomicReadStats`. The per-batch source line, which double-counts overlapping batches, becomes backend-wide totals in the summary line (about 20 lines net).
  - TensorReader's aligned stream mode: delete `InitOpts`, `padding_remaining`, `streamAligned` (which has a `catch unreachable` on a file read) and `discardAligned`, and build `readPositionalAll` on std's version. `readFilePositionalAllV`/`advanceScatter` move next to `readGated`, and `max_positional_iovecs` moves into limits.zig, which avoids a circular import (about 90 lines).
  - `retry_once`: have `exchange` map 401/403 to PermissionDenied, and have HF's `performRead` catch it once. That removes `retry_once`, `previous_status`, `ordinal` and the `Attempt` parameter from nine hooks. This beats keeping a flag in `perform`.
  - `LoadProfile.direct_io_alignment`: add one condition at `direct_loader.zig:135` (`VFS.fromIo(io) != null`) rather than deleting the field. Deleting it would open every remote file on the planner thread, at one HEAD each.
  - Optional, unverified: generate the VFSBase forwarders at comptime (about 590 lines).
- **Absorbs:** gap2-0 "four handle tables", gap2-0 "credential code that does not earn its place", vfs "VFSBase forwarders", vfs "ReadStatsProvider", vfs "TensorReader aligned mode", vfs "LoadProfile alignment", gap1-0 "retry_once is not once".

#### Remaining small simplifications

- **PJRT errors (unverified).** Delete `awaitBatch`'s SetBufferError pass (`direct_loader.zig:315-328`) and `Target.closed`. At the XLA pin 41370d1124, the transfer manager's destructor already errors every pending buffer and tolerates an event that already failed, and that pass can abort the process through `CHECK(definition_event)` when a load fails twice. First add the assertion that a failed batch's output ready event returns an error. About 30 lines.
- **`Event.onReady` returns void**, using `catch unreachable` like its siblings, or a panic that names the cause. `submitTransfer` then drops the `awaitRaw` branch (`:3814-3818`), which would block a PJRT callback thread if it ever ran.
- **Logging.** Log every failure once, at `err`, where it is detected (log.zig already demotes these under test). Planning rejects go through `load_log.failure`. Loader totals log at info; per-batch lines stay at debug. Delete the `TMP loader created_at` log (`:193`). This removes the two-tier policy and `loader.md:225-232`.
- **Progress.** The direct backend drops its per-destination child nodes and counts bytes on the caller's node. That removes `startProgress` and `TensorTransfer.progress`, and also the 127-slot cap that today stops progress after about 126 tensors per batch. Remove the callers' totals in the same change.
- **CompositeTest.** Once CPU honors a single block size, replace it with end-to-end loads plus a 15-20 line test that inspects the plan's shape. `landJob`/`landExtent`/`ScatterSink` then take the root pointer, and `FileReader`/`BytesReader` go (about 250 lines net). If CompositeTest stays, write its sources to tmp files and still drop the generic reader (about 25 lines net).
- **Test oracle.** Export one rank-N `expectedShard` + `expectShards` (residency check included, `packedShape` for sub-byte dtypes) via `zml.testing`. That replaces llmd's copies, the five hand-rolled shard loops, `Replay.expect` and the inline residency loops (about 140 lines net). Keep llmd's Fixture.
- **Tests.** Delete the pump, last-flag and coalesced-blocks tests. Keep the `read_chunk_size` InvalidOptions asserts. Make the limits.zig doc say "InvalidOptions" and drop the unreachable `selected > max`.
- **Parts sort.** Give `SourceSlot` a required `ordinal` (creation order) and compare it instead of the URI. `fileGroupEnd` and the assert compare slot pointers.
- **Dead code.**
  - `pub const limits` in io.zig, and loader.zig's unused `dma_calibration` import.
  - ShardGrid.init's IncompatibleSharding returns, its `mask == 0` check and the `max_boxes` debug assert. Writers can be sized by `MAX_NUM_DEVICES`.
  - `putSourcesNoClobber`'s dead errdefer and the error union every caller turns into a panic.
  - Source's ~8 `_sharding` resets, and `place()` setting it twice.
- **`nccl_preload`.** Add `try checkNccl(result)` after the init loop.
- **`startAllocations`.** Fix the "created ahead of their reads" doc rather than moving the call: starting it before publication is a use-after-free on the rejection path.
- **Optional.** `Submission` as a union of batch pointers plus `Backend.await` (about 5 lines).
- **Considered, not worth code.**
  - Locking only the touched pumps in `enqueueBlocks`: a few ms of CPU in total, no wall-clock effect.
  - Destroying managers early, per target: a 1-10 ms tail at real destination counts.
  - Lowering the O_DIRECT threshold: see Performance.

### Performance

Ranked by expected effect where each one applies.

1. **The buffered backend opens the file once per part** (confirmed). On TPU, neuron or metal reading from gs://, s3:// or hf://, each part costs one HEAD, or a two-hop chain on hf://, with at most 32 in flight.
   - 10,000 parts: 10,000 × 2 × 40 ms / 32 ≈ 25 s.
   - A DeepSeek-class checkpoint (69,187 tensors): about 140k governed requests, about 3 min at 40 ms. Every HEAD also counts against rate limiting: a 429 holds the backend, and after the 5 min throttle budget the load fails with `RateLimited`.
   - Locally: 69k extra open/close pairs.
   - Fix: move 8. Opens drop to the number of files (46).
2. **CDNA3 FP8 normalization reads every scale tensor serially at `Model.init`** (story version confirmed).
   - Qwen3.5-MoE: 60 × 512 × 3 = 92,160 scale tensors. Locally, at about 15 µs per open/pread/close, ≈ 1.4 s before compilation starts. Remote, at 1 HEAD + 1 GET each and 20-100 ms per request, ≈ 1-5 h.
   - Fix: group the reads by `file_uri`, keep a local file map for the call, and use sorted, coalesced positional reads. A layer's expert scales are contiguous, so that is one read per Linear and file.
   - Alternatively, convert unconditionally: the guard only matters for scales near 2^127. That is your call. A device-side check does not work, because the dtype is chosen at compile time.
3. **The remote registry parse is serial** (confirmed).
   - 46 shards × ~4 round trips × 50-100 ms = 9-18 s before the loader starts; 163 shards take 33-65 s. A 16-wide parse would take 1-4 s.
   - First step, almost no code: raise `header_buf` from 64 KiB to about 1 MiB, so the length prefix and the JSON arrive in one GET. That saves one round trip per shard.
   - Then parse each shard into its own arena and merge in index order, at a fixed width. LoadProfile has no width field.
4. **Pinned registration of the engram tables stalls the load** (partial: measured, not validated by a fix). On DeepSeek-V4.1 / 2×GB300 it costs 2.4 s of 11.9 s (20%).
   - Cause: `cuMemHostRegister` on four 46 GiB shards holds the driver lock, which stalls the pumps, credit release and device-destination creation. It is the lock hold, not the scheduling.
   - Fix, CUDA only: `initHost` creates placed pages without a driver call, the host jobs' own writes fault them in, and each destination is DmaMapped once `host_written == host_total`. Estimated cost is about 0.2-0.3 s per 46 GiB, in the tail of the load.
   - Two preconditions are unverified: device-kind views of DmaMapped memory, and `SharedHostAllocation`'s on_delete on CUDA.
   - Not ROCm: `hipHostRegister` measured 6.3 s/GiB there.
   - Rejected: publishing host destinations earlier, or a single creator. Neither removes the lock hold.
5. **DMA calibration runs on every `Loader.init`** (partial). It costs 0.7-1.25 s on PCIe hosts, 0.1-0.3 s on GB300 and 3.5-7 s on a degraded host. llmd hides it behind compilation; examples/io, mnist and tests pay it: 21-45 loaders × 0.27-1.1 s ≈ 15-40 s per GPU test run. Fix: move 5, or have tests pin one block size.
6. **Device destinations are created on the read path** (partial). `tensor_init_ms_per_read = 0.229` comes to ≈ 13.9 s of worker time, 6.9% of 17 workers × 11.9 s.
   - Upper bound on the wall-clock gain: about 0.8 s (14 s / 17 workers).
   - Most of that time is LazyOnce waits and driver-lock waits; the cuMem calls for 4,748 buffers account for a few seconds of CPU at most.
   - Fix: drop the `hostResident` filter so the allocation task creates every destination in claim order. One task makes about 1-3k destinations/s against about 200/s consumed by workers. LazyOnce stays as the fallback.
   - Measure that `tensor_init_ms_per_read` falls toward 0 while pump stops stay unchanged.
7. **Pack programs compile on the load path** (partial). The recoverable time is W, the wait for the slowest program before the first group: 0.2-2 s of a 12-13 s load (0.85 s for one program on GB300). The bulk publishing at +1.9 s comes from admission retires, not compiles. Measure W (time to the first `loadExecute`) before adding a `packer.startPrograms(...)`. That call also needs a stable copy of the meshes, because deepseek41's `parallelism` is a local.
8. **O_DIRECT is decided per file** (partial, known in CTX).
   - GB300 DeepSeek read 205 GiB from disk where per-range modes would read 156, so 44-49 GiB were re-read. The time cost is unmeasured and may be zero: on GB300, direct reads (3.74/3.63 s) beat warm buffered reads (3.83 s).
   - On mi300 the break-even is near 87% cached.
   - Next steps: on gb300-1, measure `ZML_DIRECT_IO=on` against `auto`; on mi300, try `cold_fraction` ≈ 0.85.
   - Do not lower the threshold: on mi300 cold buffered reads run at 4.7-4.9 GiB/s against 25.1 GiB/s direct, about 3× slower for a 40%-cached file.
9. **Sorting the parts compares URI strings** (confirmed). At 69k parts the sort takes 21-50 ms (modelled) on the path to the first read. Comparing a slot ordinal brings it to 6-21 ms; a key array to 4-6 ms, but adopt that only if a measurement shows it matters.
10. **Every composite is checked twice:** a run list and sort of up to 2^21 runs (32 MiB) per check. Unmeasured; DeepSeek's first publish at 0.17 s bounds today's cost. Fix: move 3.
11. **Admission sums temporaries.** It charges N·T where at most T is live. No recorded run is affected, because the measured GPU temporaries are 0. Fix: the high-water mark in move 9.
12. **Below noise, not worth code.**
    - The manager-destroy tail: about 4.7k managers at 2 devices and 19k at 8, so 1-10 ms.
    - Locking every pump in `enqueueBlocks`: a few ms of CPU in total.
    - Sharing zio executors with the caller: the B70 measured 9.9 s threaded against 10.0 s on zio. It only matters under a small cgroup CPU quota. The cheapest mitigations are O_DIRECT reads through the base Io, the allocation task on its own thread, and llmd passing `io_impl` to the Platform so compiles use `blockInPlace`.
    - Calibration under NCCL contention: unmeasured, and reordering would still overlap the model compile.

**Outside the loader (partial).** Mosaic's TPU MoE kernel transposes both expert tensors on every forward. For gemma4 bf16 that is gate_up [128, 1408, 2816] plus down [128, 2816, 704] = 1.52 GB per layer × 30 layers = 46 GB read and rewritten per step. If XLA materializes it (unmeasured), that is about 91 GB of HBM traffic per step, roughly 55 ms per decode step at 1.6 TB/s (my estimate). Fix it in the kernel with `transpose_rhs` (`kernels/mosaic_tpu/builder.zig:1599`), not with a loader transpose: 2-byte-run strided parts are slow on buffered and refused on direct.

### What earns its place

- **Source's algebra:** pieces as affine bit lattices. Every op was traced. One test checks every op byte for byte against the same Tensor program. It is what lets model code describe fused, padded, stacked and swizzled layouts in a few lines with no compute.
- **ShardGrid:** placement by arithmetic, with no lists. It is correct for replicated, folded, 2D and 3D layouts, and checked element by element.
- **The planner's core.** `cutRuns` coalesces at cuts where no tensor is split, and exact transfer records mean workers and pumps decide nothing. Staging is what keeps a column-sharded tensor from costing one submission per row. The gather path (about 700 lines) is still an open question: measure it against expansion plus staging, with the swizzles as programs, before removing it.
- **Per-file publication and FIFO claims:** workers read file 1 while the rest of the batch is still being planned. FIFO order is also what makes failing other batches from a callback thread safe.
- **The batch completion counter:** a publish sentinel, one unit per job, one for the allocation task. `retireBatch` takes every pump lock. The `retired` event stack exists so an event is never destroyed inside its own callback, and it bounds live PJRT events.
- **Per-device pumps and the `pumping` handoff:** one submitter per device, no lost wakeup across worker and callback threads, and a byte and piece budget per device.
- **Lifecycle credits:** the real bound on pinned memory. They are deadlock-free because DMA completion depends on no credit.
- **The read gate, for now:** it makes `read_parallelism` an exact count, as documented and as the buffered backend enforces. Revisit it only with an A/B run.
- **The failure machinery:** the sticky first error, `Scheduler.fail`, `abortReady`, LazyOnce's error replay, and validation before publication that leaves the loader usable. Each piece is what makes the drain exact.
- **LazyOnce for file opens and destinations:** creation happens once, whether a worker or the allocation task gets there first.
- **SharedHostAllocation:** one owner plus views, released by a reference count on on_delete. It serves replicated host tables, and has the shape that loader-owned pinned memory will need.
- **O_DIRECT mechanics:** reads widened to 4 KiB with room reserved in the job bound; a file whose filesystem refuses the flag stays buffered and its plan stays exact; a refused read demotes the file under the mutex; decisions always come before reads.
- **The 64-run / 1 MiB expansion thresholds:** they also help the buffered backend, which reads large runs straight into staging.
- **Admission plus FIFO retirement:** pack reads overlap with pack executions under a memory bound. `readRoom` reads `allocated` before `memoryStats`, so a race can only make the room look smaller, never larger.
- **The `delivered` check:** zml's only guard that a program-bound tensor gets written. Moving it buys an ordering nobody can use profitably today.
- **Two backends:** they stay while TPU, neuron and metal are unverified on the direct path.
- **HugePageAllocator's THP step:** measured at 12-17% faster loads with 62% fewer page walks on x86 CUDA (commits 0e19c755, ff2945ac). That record was dropped from CTX.md, so put the result back in the doc comment instead of removing the step. mbind with its cpuset fallback and the DmaMap/DmaUnmap pairing are also correct.
- **The VFS request governor, hold and throttle budget:** they absorb rate limiting without multiplying the loader's width.

## Appendix A: every surviving finding

Sorted by corrected severity, then category. Status: `confirmed` as stated, `partial` (verifiers corrected the claim, severity or fix; Parts 1 and 2 use the corrected version), `contested` (verifiers split), `unverified` (verifier call failed). Locations are relative to `~/github/zml`. The full records (claims, evidence, scenarios, proposals and verifier notes) are in this machine's workflow journal, `~/.claude/projects/-home-brabier-github-zml/59b09782-699a-4f2f-b5c4-0d5bf28f7efe/subagents/workflows/wf_fbfc17f7-6ca/journal.jsonl`.

| Sev. | Category | Status | Area | Finding | Location |
|---|---|---|---|---|---|
| high | bug | partial | gap1-2 | llmd never sets Platform.io_impl, so on its default zio every compile, pack programs included, runs on an 8 MiB coroutine | `monorepo/llmd/main.zig:130` |
| high | bug | confirmed | gap2-0 | A missing optional file on s3:// (no ListBucket) or https:// aborts llmd: backends map 'absent' three ways | `zml/vfs/s3.zig:830` |
| high | bug | partial | gap2-1 | FP8 per-tensor experts with separate gate/up get [E,2] scales the kernel cannot read; the dense fused path handles the same case | `monorepo/llmd/weights.zig:503` |
| high | bug | confirmed | gap2-1 | FlashInfer NVFP4 expert globals ignore compressed-tensors' `.divide` and come out inverted | `monorepo/llmd/weights.zig:526-533` |
| high | bug | partial | vfs | Re-resolving an HF download URL frees the URL that other in-flight reads (and the request's own spec) still use | `zml/vfs/hf.zig:870-890` |
| high | missing | confirmed | gap1-0 | One stalled range GET hangs the whole llmd load forever, with nothing logged and no way to cancel it | `zml/vfs/request.zig:348-352` |
| high | missing | partial | gap2-0 | S3 accepts only static env keys: session tokens and role credentials are dropped, then fail as Unexpected | `zml/vfs/s3.zig:224-267` |
| high | test | partial | consumer | DeepSeek V4.1 tests are never built, one no longer compiles, and bf16Linear has no test | `monorepo/llmd/deepseek41_test.zig:1-3` |
| high | test | partial | source | sourceOf/ofBinding/rebind/unbind have no zml test; the Source harness only uses u16 inputs | `zml/zml/io/Source.zig:85-115` |
| high | performance | unverified | consumer | CDNA3 FP8 normalization opens and reads every block-FP8 scale tensor serially at Model.init | `monorepo/llmd/weights.zig:165-204` |
| high | performance | confirmed | memory | Buffered backend opens the source file once per part, which is a HEAD round trip per tensor on gs://, s3:// and hf:// | `zml/zml/io/buffered_loader.zig:181-185` |
| high | performance | confirmed | story | Buffered backend opens the source file once per part per tensor, and each remote open costs at least one HEAD | `zml/zml/io/buffered_loader.zig:180-186` |
| high | simplification | confirmed | concurrency | Credits already bound pinned memory: BlockPool's blocking acquire and per-block leases are a second bound that can never act | `zml/zml/io/direct_loader.zig:475` |
| high | simplification | partial | gap2-0 | Four backends each keep a handle table, mutex and 17-entry std.Io vtable behind the VFS's own table | `zml/vfs/s3.zig:169-199` |
| high | simplification | confirmed | memory | BlockPool's waits, close and per-block leases never do anything: the lifecycle credits already reserve every block | `zml/zml/io/host_memory.zig:413-562` |
| high | simplification | confirmed | source | checkParts' strided overlap proof (~175 lines) re-proves a disjointness Source already guarantees | `zml/zml/io/backend.zig:396-410` |
| medium | bug | partial | concurrency | Loader-wide SourceSlot map borrows file_uri past retirement and keeps a planning-time open failure forever | `zml/zml/io/direct_loader.zig:100` |
| medium | bug | partial | concurrency | Progress nodes live until batch retirement: past ~127 tensors, completions are never counted | `zml/zml/io/direct_loader.zig:1272` |
| medium | bug | partial | direct-front | SourceSlot keys and URIs borrow the first store's strings but live as long as the loader | `zml/zml/io/direct_loader.zig:100` |
| medium | bug | confirmed | failures | Source slots borrow registry file URIs for the loader's whole life, beyond the documented retirement | `zml/zml/io/direct_loader.zig:595` |
| medium | bug | partial | failures | Workspace.init hits `unreachable` on mixed device kinds; the doc promises UnsupportedPlatform | `zml/zml/io/host_memory.zig:319` |
| medium | bug | partial | failures | loadExecute over a plain binding is not skipped by the bulk: double load and a race on output.* | `zml/zml/io/loader.zig:293` |
| medium | bug | partial | frontend | Direct backend keeps borrowed file_uri keys and open fds for the loader's lifetime, against the borrow-until-retired contract | `zml/zml/io/loader.zig:48` |
| medium | bug | confirmed | gap1-0 | Remote retry and failure logs print HF's signed CDN URL at warn and err, and the loader reports every remote failure as `Unexpected` | `zml/vfs/hf.zig:873-878` |
| medium | bug | partial | gap1-0 | exchange returns a connection whose write failed to the pool, and treats a reset during the TLS handshake as fatal | `zml/vfs/request.zig:352-361` |
| medium | bug | confirmed | gap1-0 | https:// sources behind a redirect cannot be read: the open follows the redirect, the reads forbid it, and a relative Location is resolved against overwritten bytes | `zml/vfs/http.zig:369-372` |
| medium | bug | partial | gap2-0 | GCS: Google service-account key files never parse; GCS.auto falls back to the VM identity or anonymous | `zml/vfs/gcs.zig:57-71` |
| medium | bug | confirmed | gap2-0 | SigV4 canonical URI is the raw key: '=', '+', space, non-ASCII, '?' and '#' in an S3 path break every signed request | `zml/vfs/s3.zig:645-648` |
| medium | bug | partial | story | Direct backend keeps one std.Progress node per tensor until the batch retires, so progress stops being tracked after about 120 tensors | `zml/zml/io/direct_loader.zig:1272` |
| medium | bug | confirmed | vfs | Loader source slots keep borrowed file_uri keys after a submission is retired, contradicting the documented borrow | `zml/zml/io/direct_loader.zig:595-602` |
| medium | bug | partial | vfs | The auto probe caches the 32 pages it samples, so a reloaded cold file looks 100% cached and is read buffered | `zml/vfs/direct_io.zig:104-128` |
| medium | missing | partial | gap2-0 | GCS treats a 401 as fatal: no refresh-and-retry, and expiry has no margin | `zml/vfs/gcs.zig:150-153` |
| medium | missing | confirmed | gap2-1 | Dense NVFP4 Linears drop the checkpoint's input scale; the third `conventions` entry has no effect | `monorepo/llmd/weights/linear.zig:141-148` |
| medium | missing | partial | memory | When every device is on one NUMA node, the arenas still interleave across sockets, the configuration CTX measured 27% slower | `zml/zml/io/host_memory.zig:59-71` |
| medium | missing | confirmed | memory | Workspace.init reaches `unreachable` on heterogeneous or more than 64 devices, crashing Loader.init instead of returning an error | `zml/zml/io/host_memory.zig:319-329` |
| medium | missing | partial | vfs | The safetensors header is not validated: malformed JSON panics, the data_offsets end is ignored, nothing is checked against the file size | `zml/zml/safetensors.zig:870-874` |
| medium | test | partial | gap1-0 | The HTTP test server closes every connection, so pooled reuse, truncated or stalled bodies, redirects and 401/403 re-resolve are never tested | `zml/vfs/mock_server.zig:33-40` |
| medium | test | partial | gap1-2 | No loader test, and no llmd load test, runs on zio, llmd's default Io | `monorepo/llmd/main.zig:130` |
| medium | test | partial | gap2-0 | No test covers SigV4 output, credential parsing, env resolution or the 401/403 paths of S3 and GCS | `zml/vfs/s3.zig:907-913` |
| medium | test | confirmed | gap2-1 | Untested: every quantized experts() branch, compressed-tensors scales, fused NVFP4 and quantized linear.create | `monorepo/llmd/weights.zig:411-417` |
| medium | test | partial | source | The llmd engram host-pinned composite test is never compiled and has rotted (binding.transformed) | `monorepo/llmd/models/deepseek41.zig:3179-3215` |
| medium | test | partial | tests | The default local path, real O_DIRECT through the VFS, is never exercised by a loader test | `zml/zml/io/direct_loader.zig:1579` |
| medium | performance | partial | direct-front | Host allocation starts at submit with no ordering against DMA, and two creators make it hard to schedule | `zml/zml/io/direct_loader.zig:257-289` |
| medium | performance | partial | execution | Device destinations are created on the workers' read path; measured ~14 s of worker time (~0.8 s wall) on DeepSeek | `zml/zml/io/direct_loader.zig:2814-2821` |
| medium | performance | confirmed | story | normalizeBlockFp8 reads every block-FP8 scale tensor serially on the init thread, opening the file once per tensor | `monorepo/llmd/weights.zig:183-204` |
| medium | performance | confirmed | vfs | The remote registry parse is serial (about 4 round trips per shard before anything loads), and the registry mutex is dead | `zml/zml/safetensors.zig:1000-1013` |
| medium | simplification | confirmed | concurrency | Pipeline is an 11-argument pointer view of the Loader; failure handling and request code reach state through both | `zml/zml/io/direct_loader.zig:3497` |
| medium | simplification | partial | consumer | Dead or test-only generality in weights.zig and models.zig | `monorepo/llmd/weights.zig:796-811` |
| medium | simplification | confirmed | consumer | FusedLinear builders are stateless but live on Packer, so every model threads a Packer it never needs | `monorepo/llmd/weights.zig:308-397` |
| medium | simplification | partial | consumer | Recipe/Packer.recipes/Program/loadPacked hand-build a closure that Source.bindProgram should carry | `monorepo/llmd/weights.zig:88-131` |
| medium | simplification | confirmed | direct-front | Per-device replica copies for host-resident destinations run only in tests | `zml/zml/io/backend.zig:632-650` |
| medium | simplification | confirmed | direct-front | Pipeline is a second handle on the Loader: 9 of its 13 fields are back-pointers or copies | `zml/zml/io/direct_loader.zig:3497-3550` |
| medium | simplification | partial | execution | Lifecycle credits and the BlockPool's wait both bound the same pinned blocks; the pool can never block | `zml/zml/io/direct_loader.zig:475` |
| medium | simplification | partial | frontend | Binding-to-LoadSpec lowering is written twice and needs a counting pass so spec slices stay valid | `zml/zml/io/loader.zig:545` |
| medium | simplification | partial | frontend | Part validation runs two or three times, and the plain-size error arrives at different times per backend | `zml/zml/io/loader.zig:744` |
| medium | simplification | partial | frontend | The admission ledger is split across layers and files, with seven per-device arrays and a must-match-to-the-byte invariant | `zml/zml/io/loader.zig:80` |
| medium | simplification | unverified | gap1-1 | awaitBatch's SetBufferError pass and Target.closed repeat the manager destructor, and abort where it would not | `zml/zml/io/direct_loader.zig:291-304` |
| medium | simplification | confirmed | gap2-0 | Credential code that does not earn its place: dead GCS service-account paths, a log-suppressing metadata probe, S3 signing written three times | `zml/vfs/gcs.zig:15` |
| medium | simplification | partial | memory | Buffered loader spawns one LimitedGroup task per tensor; std.Io.async runs the overflow inline, so submit runs the load itself, and a canceled admission hangs awaitBatch | `zml/zml/io/buffered_loader.zig:65-68` |
| medium | simplification | partial | memory | DMA calibration costs ~0.7-1 s per Loader.init and only picks a block size, which is noisy and moves loads by at most ~2% | `zml/zml/io/dma_calibration.zig:12-20` |
| medium | simplification | unverified | planner | Job packs seven kinds into five flags; staged jobs rebuild an Extent at run time | `zml/zml/io/direct_loader.zig:752-789` |
| medium | simplification | partial | source | Binding/Layout/Part restate Source and LoadPart: two Part types, a dedupe layer, and a nullable 'identity' layout | `zml/zml/io/TensorStore.zig:17-74` |
| medium | simplification | confirmed | source | View's createTensor family reimplements Source.bind; createHostPinnedTensor/maybeCreateHostPinnedTensor are unused | `zml/zml/io/TensorStore.zig:208-239` |
| medium | simplification | confirmed | story | BlockPool's blocking wait, close and RequestExceedsCapacity can never fire: the lifecycle credits already bound the blocks | `zml/zml/io/host_memory.zig:513-526` |
| medium | simplification | partial | story | The loader re-proves twice per spec a disjointness that Source guarantees, and refuses valid Source layouts | `zml/zml/io/backend.zig:381-575` |
| medium | simplification | partial | story | llmd keeps test-only recipes, helpers and options: concatenateRows/convertRows/dequantizeBlocks, Interleaving.explicit, ExpertLayout.loader | `monorepo/llmd/weights.zig:44-56` |
| medium | simplification | partial | tests | CompositeTest replays plans through a hand-written model of the worker; production keeps a generic reader for it | `zml/zml/io/direct_loader.zig:5088` |
| medium | simplification | partial | tests | Loader test helpers are duplicated in llmd and across zml, and expectShards is rank-2 only | `zml/zml/io/testing.zig:115` |
| medium | simplification | partial | tests | Tests keep a dead production mechanism alive: unshared host replicas and their copy path | `zml/zml/io/direct_loader.zig:117` |
| medium | api | confirmed | execution | Pipeline is a bag of back-pointers into its owner; recordError reaches up into the scheduler, pool and gates | `zml/zml/io/direct_loader.zig:3497-3550` |
| medium | api | partial | gap2-1 | ExpertsLayout is only partly honored: order and orientation are dropped, and a single fused projection is trusted | `zml/zml/moe/moe.zig:26-35` |
| medium | api | partial | story | Three representations of what fills a tensor and three Part types, converted back and forth; the front end makes the direct planner's expansion decision | `zml/zml/io/TensorStore.zig:17-29` |
| low | bug | partial | concurrency | Sticky failure is not uniformly sticky: awaitAll keeps executing later packs, and a publish race returns Closed | `zml/zml/io/loader.zig:299` |
| low | bug | confirmed | concurrency | warmupDeviceAllocators leaks running group tasks that point at its stack when a spawn fails | `zml/zml/platform.zig:466` |
| low | bug | partial | consumer | Fused/split fp8_block128 members with a partial final block get silently wrong scales | `monorepo/llmd/weights.zig:382-386` |
| low | bug | partial | consumer | loadInto frees pack executables while the loader may still hold submissions borrowing them | `monorepo/llmd/weights.zig:1676-1685` |
| low | bug | confirmed | direct-front | A submit that races a failure returns error.Closed instead of the loader's stored error | `zml/zml/io/direct_loader.zig:232-237` |
| low | bug | partial | direct-front | Per-destination progress nodes are held until batch retirement, exhausting std.Progress's 127 nodes | `zml/zml/io/direct_loader.zig:1353-1359` |
| low | bug | partial | execution | Event-slot count is a three-site planner counter consumed by an atomic index guarded only by a debug assert | `zml/zml/io/direct_loader.zig:1975` |
| low | bug | partial | failures | Direct backend progress: one node per tensor held until batch retirement; std.Progress has 127 slots | `zml/zml/io/direct_loader.zig:1355` |
| low | bug | partial | frontend | Admission undercounts device memory under the default VMM allocator, which rounds every buffer up to a granule | `zml/zml/io/execute_admission.zig:8` |
| low | bug | partial | gap1-0 | GCS refreshes its token over the network while holding the handle mutex that every read takes uncancelably | `zml/vfs/gcs.zig:462-476` |
| low | bug | partial | gap2-0 | resolveModelEntrypoint and resolveModelRepo erase every open error into FileNotFound or InvalidPath | `zml/zml/safetensors.zig:1025-1028` |
| low | bug | partial | gap2-1 | Pre-stacked experts silently load quantized tensors as plain weights and drop biases | `monorepo/llmd/weights.zig:399-417` |
| low | bug | partial | memory | Buffered staging admission counts tensor bytes but StagingPool hands out any larger retained buffer, so staging memory escapes the budget | `zml/zml/io/buffered_loader.zig:161-166` |
| low | bug | confirmed | source | loadExecute checks a plain binding against the checkpoint's shape, not the bound tensor's | `zml/zml/io/loader.zig:876-883` |
| low | bug | confirmed | story | source_slots keys borrow file_uri for the loader's whole life, longer than the documented borrow | `zml/zml/io/direct_loader.zig:595-602` |
| low | bug | partial | tests | Buffered backend misplaces sub-mesh destinations; its only test sidesteps this with an owned CPU platform | `zml/zml/io/buffered_loader.zig:614` |
| low | bug | partial | vfs | Direct reads are raw syscalls outside the Io: they block zio executors (llmd's default) and are not cancellation points | `zml/vfs/direct_io.zig:141-166` |
| low | bug | partial | vfs | TensorRegistry.fromPath opens a relative .safetensors path relative to its own directory | `zml/zml/safetensors.zig:755-770` |
| low | missing | partial | concurrency | Asynchronous DMA failures are never surfaced: pack programs run on errored inputs and Exe.call panics | `zml/zml/io/direct_loader.zig:1166` |
| low | missing | partial | concurrency | Bulk load is never admitted: its destinations take the room admission promised to pending packs, which can OOM a load where the final model fits | `zml/zml/io/loader.zig:203` |
| low | missing | partial | concurrency | Init-future cancellation is swallowed in Workspace.init and logged as an error in calibration | `zml/zml/io/host_memory.zig:630` |
| low | missing | partial | concurrency | No way to abort pending work: deinit and error paths read and transfer every pending submission to completion | `zml/zml/io/loader.zig:166` |
| low | missing | confirmed | consumer | NVFP4 FlashInfer weight global scale silently ignores w3's weight_scale_2 | `monorepo/llmd/weights.zig:731-747` |
| low | missing | partial | consumer | maybeSwizzleBlockFp8Scales panics on Blackwell for non-e8m0, unquantized, per-tensor or partial-block projections | `monorepo/llmd/weights.zig:226-252` |
| low | missing | partial | execution | No abort: a failing caller waits for every pending byte to be read and uploaded before its error surfaces | `zml/zml/io/direct_loader.zig:306` |
| low | missing | partial | failures | Checkpoint headers are trusted: data_offsets end ignored, malformed fields panic, truncation found mid-load | `zml/zml/safetensors.zig:870` |
| low | missing | partial | failures | DMA calibration failure aborts Loader.init instead of falling back to the default sizing | `zml/zml/io/direct_loader.zig:389` |
| low | missing | partial | failures | Pack execution failures panic in Exe.call instead of returning AwaitError | `zml/zml/exe.zig:353` |
| low | missing | partial | frontend | A failed pack execution panics the process instead of becoming the loader's sticky error | `zml/zml/io/loader.zig:454` |
| low | missing | partial | frontend | Error paths drain everything: deinit cannot abort, awaitAll keeps executing after a failure, and partial-publication errors are not sticky | `zml/zml/io/loader.zig:166` |
| low | missing | confirmed | frontend | Validation failures carry no tensor context at default log levels | `zml/zml/io/log.zig:1` |
| low | missing | partial | gap2-0 | S3 endpoint and region: wrong env precedence, region ignored for the endpoint, and region or auth failures reported as bare status lines | `zml/vfs/s3.zig:229-238` |
| low | missing | partial | gap2-1 | Valid NVFP4 expert variants panic instead of returning UnsupportedQuantization | `monorepo/llmd/weights.zig:524-533` |
| low | missing | partial | memory | Teardown unmaps with `catch unreachable`: after a sticky device error, Loader.deinit panics and hides the real failure | `zml/zml/io/host_memory.zig:267-273` |
| low | missing | partial | planner | Device-id check runs in the planner per file; plan_slot's index is never read | `zml/zml/io/direct_loader.zig:1664-1690` |
| low | missing | partial | planner | Strided parts whose source runs are ordered but not back to back are refused above one read | `zml/zml/io/direct_loader.zig:552-567` |
| low | missing | confirmed | source | Binding.memory is silently ignored for program-computed tensors | `zml/zml/io/TensorStore.zig:19-20` |
| low | missing | partial | story | A failed pack-program execution, including an asynchronous DMA failure on its input, aborts the process instead of returning from awaitAll | `zml/zml/io/loader.zig:347-353` |
| low | missing | partial | tests | Workspace.init reaches `unreachable` on heterogeneous devices; untested | `zml/zml/io/host_memory.zig:318` |
| low | missing | confirmed | vfs | A 200 answer to a Range GET is accepted at any offset by discarding the prefix, so transfer is quadratic | `zml/vfs/range_read.zig:187-198` |
| low | test | partial | concurrency | The failure paths of the extra actors have no test: host-destination allocation failure and init cancellation | `zml/zml/io/direct_loader.zig:275` |
| low | test | confirmed | consumer | llmd's load test helpers duplicate zml's io/testing.zig | `monorepo/llmd/weights.zig:1690-1802` |
| low | test | partial | execution | Submission and abandonment failure paths have no test | `zml/zml/io/direct_loader.zig:2867-2874` |
| low | test | partial | failures | The production read path (VFS local profile: 4 KiB alignment, direct_io=auto) has no loader-level test | `zml/zml/io/direct_loader.zig:4087` |
| low | test | partial | frontend | Measured admission (fits, exceeds, temporaries, pending_execution) has no test | `zml/zml/io/loader.zig:319` |
| low | test | partial | memory | The buffered backend has no end-to-end test on any CI platform; strided parts, helper-split reads and loadExecute are never run | `zml/zml/io/buffered_loader.zig:527-679` |
| low | test | partial | planner | Untested: gather jobs mixing direct and buffered files, and planner failure after publication | `zml/zml/io/direct_loader.zig:1711-1714` |
| low | test | partial | story | The measured admission path never runs in tests | `zml/zml/io/loader.zig:180-187` |
| low | test | partial | tests | Assertions that cannot fail | `zml/zml/io/loader.zig:1456` |
| low | test | partial | tests | Front-end tests crash, fail or take an unrepresentative path on a buffered platform | `zml/zml/io/loader.zig:1335` |
| low | test | partial | tests | Measured loadExecute admission (retire until fits) has no test on any platform | `zml/zml/io/loader.zig:180` |
| low | test | partial | tests | Overlapping white-box tests for Scheduler, Planner and BlockPool pin internals | `zml/zml/io/direct_loader.zig:4427` |
| low | test | partial | tests | Platform init failures become skips, hiding regressions | `zml/zml/io/direct_loader.zig:4034` |
| low | test | partial | vfs | examples/io 'Loaded weights' throughput includes Loader.init (DMA calibration and pinned pre-growth) | `zml/examples/io/main.zig:317-326` |
| low | performance | partial | concurrency | Host-destination allocation task starts only after the whole batch is planned and published | `zml/zml/io/direct_loader.zig:232` |
| low | performance | partial | consumer | Loader.init's DMA calibration is measured while NCCL preload holds the CUDA driver lock | `monorepo/llmd/main.zig:389-390` |
| low | performance | partial | consumer | known (CTX: twenty-eighth pass 'Left open'): pack programs compile on the load path and delay the bulk load | `monorepo/llmd/weights.zig:1509-1546` |
| low | performance | partial | execution | enqueueBlocks locks every device's pump even when the job feeds one device | `zml/zml/io/direct_loader.zig:3654-3655` |
| low | performance | partial | failures | known (CTX: Twenty-ninth pass): per-file O_DIRECT decision re-reads cached data under page-cache pressure | `zml/vfs/direct_io.zig:23` |
| low | performance | partial | frontend | Temporaries are summed across pending submissions although executions run one at a time | `zml/zml/io/loader.zig:331` |
| low | performance | partial | gap1-1 | Post-done retirement tail: D×N transfer-manager destroys run serially, outside the loader's own clock | `zml/zml/io/direct_loader.zig:305-307` |
| low | performance | partial | gap1-2 | Under zio the loader's blocking workers, allocation task and pack compiles share the caller's executors | `zml/zml/io/direct_loader.zig:142` |
| low | performance | partial | memory | Buffered strided parts allocate an uncounted heap copy of each part's whole source span | `zml/zml/io/buffered_loader.zig:192-197` |
| low | performance | partial | planner | Overlap check of strided parts jumps to the 20 s quadratic path just above 2^21 runs, and runs twice | `zml/zml/io/backend.zig:413-453` |
| low | performance | confirmed | planner | Sorting the parts compares URI strings through two pointer hops, before the first read | `zml/zml/io/direct_loader.zig:2399-2419` |
| low | performance | partial | source | The front end runs both backends' submit-time validation a second time (checkParts and the strided-into-host check) | `zml/zml/io/loader.zig:691-695` |
| low | performance | partial | tests | Every accelerator test loader runs the full DMA calibration, about 40 s per test run | `zml/zml/io/direct_loader.zig:389` |
| low | performance | partial | vfs | known (CTX: Open work, twenty-ninth pass): the per-file direct/buffered choice re-reads cached data from disk | `zml/vfs/direct_io.zig:22-24` |
| low | simplification | partial | consumer | Load/cleanup responsibilities are spread over four layers: duplicated loadBuffers, triple awaitAll, dead errdefers | `monorepo/llmd/models/llama.zig:643-646` |
| low | simplification | partial | consumer | known (CTX: twenty-first pass item 3): host-pinned tables viewed as device memory are re-implemented per model | `monorepo/llmd/models/deepseek41.zig:1544-1558` |
| low | simplification | partial | direct-front | Batch diagnostics are passed through four layers for counts that can be computed when the log line is written | `zml/zml/io/direct_loader.zig:1824-1827` |
| low | simplification | confirmed | direct-front | Batch.host_destinations, Destination.queued_ahead and hostDestinationsInOrder restate what the published plans already contain | `zml/zml/io/direct_loader.zig:971-973` |
| low | simplification | confirmed | execution | Gather items are always 1:1 with extents; Item.extent and the grouping loop are unused generality | `zml/zml/io/direct_loader.zig:801` |
| low | simplification | confirmed | execution | Job.data_offset and Extent.data_offset are write-only state; the Job doc describes a relation the worker does not use | `zml/zml/io/direct_loader.zig:772-788` |
| low | simplification | confirmed | execution | Scheduler surface is wider than its job: diagnostic-only seal, test-only accessors, a hand-rolled deque, Claim duplicated by ReadRequest | `zml/zml/io/direct_loader.zig:2611-2618` |
| low | simplification | partial | execution | The read gate no longer earns its place: workers = width+1, gate = width, and the throttle watch it served is gone | `zml/zml/io/direct_loader.zig:142` |
| low | simplification | confirmed | execution | landJob's reader is generic (`anytype`, FileReader/BytesReader) only so one test can replay jobs without files | `zml/zml/io/direct_loader.zig:3025-3062` |
| low | simplification | partial | frontend | Which platforms use which backend is decided in several places; Submission duplicates the loader pointer | `zml/zml/io/backend.zig:589` |
| low | simplification | partial | gap1-0 | retry_once is not 'once' in the loop that names it: a second 401/403 sleeps a backoff, logs 'retrying once' again, and then fails without its status | `zml/vfs/request.zig:110-118` |
| low | simplification | partial | gap1-1 | PJRT_Event_OnReady can only fail on a struct-size mismatch; both failure branches are dead, and the loader's would block a completion thread | `zml/pjrt/pjrt.zig:1409-1430` |
| low | simplification | partial | gap1-1 | The loader accepts four memory kinds, but only .default and .host_pinned are distinct, documented or used | `zml/zml/io/backend.zig:589-608` |
| low | simplification | confirmed | gap2-1 | splitTensor re-implements Source.split + bind, and both callers re-place its result | `monorepo/llmd/weights.zig:832-854` |
| low | simplification | partial | memory | THP alignment and MADV_HUGEPAGE are hard-coded to 2 MiB and produce no huge pages on any host where they were checked | `zml/zml/io/host_memory.zig:205-206` |
| low | simplification | partial | planner | Job-length bounds: the scatter limit never applies, and the bounds are recomputed at five sites | `zml/zml/io/direct_loader.zig:2384-2394` |
| low | simplification | partial | source | Front end decides the direct planner's expand-or-gather policy, which costs extra reads on the buffered backend and forces two-pass part counting | `zml/zml/io/loader.zig:633-666` |
| low | simplification | confirmed | story | The unshared host-allocation path (`share_host_allocations = false`) can only be reached from a test | `zml/zml/io/direct_loader.zig:115-120` |
| low | simplification | partial | story | Validation runs two to four times across the front end and the backends | `zml/zml/io/loader.zig:273-278` |
| low | simplification | partial | tests | Tests that restate one-line expressions, with names and docs that mislead | `zml/zml/io/direct_loader.zig:4742` |
| low | simplification | confirmed | tests | Workspace.findArena always returns null in production | `zml/zml/io/host_memory.zig:353` |
| low | simplification | confirmed | vfs | ReadStatsProvider type-erases a single implementation for one debug line whose per-batch delta double-counts | `zml/vfs/base.zig:115-122` |
| low | simplification | confirmed | vfs | TensorReader's aligned stream mode is dead code, and the loader's padded read loop lives in safetensors.zig | `zml/zml/safetensors.zig:171-173` |
| low | simplification | unverified | vfs | VFSBase is 650 lines of hand-written forwarders, reachable only through vtable() | `zml/vfs/base.zig:162-817` |
| low | api | confirmed | concurrency | llmd loadInto frees pack executables on error while the loader still holds submissions that borrow them | `monorepo/llmd/weights.zig:1676` |
| low | api | partial | consumer | Progress: the loader leaves totals to callers that never set them, and llmd leaks/mixes nodes | `zml/zml/io/loader.zig:201-203` |
| low | api | partial | direct-front | Spec validation runs in both the front end and createDestinations, including checkParts on every composite | `zml/zml/io/direct_loader.zig:519-530` |
| low | api | partial | failures | In llmd every loader diagnostic except terminal failures is compiled out; IncompatibleSharding is never logged | `monorepo/llmd/main.zig:27` |
| low | api | partial | frontend | The `delivered` check forces llmd to publish the bulk after every pack group, delaying it and hiding it from admission | `zml/zml/io/loader.zig:587` |
| low | api | partial | story | The `delivered` rule plus llmd-owned pack executables force packs-first ordering and make every model load synchronous | `zml/zml/io/loader.zig:62-69` |
| low | api | partial | tests | checkParts runs in three places with different failure timing, and a test pins the divergence | `zml/zml/io/loader.zig:744` |
| low | api | confirmed | tests | llmd loadInto frees pack executables still borrowed by pending submissions on error; untested | `monorepo/llmd/weights.zig:1665` |
| low | api | partial | vfs | LoadProfile.direct_io_alignment duplicates the VFS's per-file decision and allows silently wrong combinations | `zml/vfs/vfs.zig:87-90` |
| low | docs | confirmed | concurrency | Scheduler and Planner docs claim byte-fair job order; the planner deliberately uses file order | `zml/zml/io/direct_loader.zig:2516` |
| low | docs | partial | consumer | Docs and names that mislead around packing | `monorepo/llmd/weights.zig:1184-1205` |
| low | docs | confirmed | direct-front | Leftover TMP debug log in Loader.create | `zml/zml/io/direct_loader.zig:193` |
| low | docs | confirmed | direct-front | Stale fairness docs, wrong source_items doc, a dead branch, and a misleading TensorTransfer name | `zml/zml/io/direct_loader.zig:2515-2517` |
| low | docs | confirmed | execution | Docs: Scratch.host_iovecs is documented as the host read's ranges but also serves DMA scatter; loader.md says the set pre-grows `width` requests | `zml/zml/io/direct_loader.zig:2699-2702` |
| low | docs | confirmed | failures | Doc contradicts code on admission without measurement and on how strided overlaps are checked | `docs/learn/loader.md:68` |
| low | docs | confirmed | frontend | Docs that contradict the code: Binding inputs, executable borrowing, room rationale, admission wording | `zml/zml/io/loader.zig:105` |
| low | docs | partial | memory | Stale or misleading comments around host memory and calibration | `zml/zml/io/host_memory.zig:297-299` |
| low | docs | confirmed | planner | Comments that contradict the code: byte-fair claims, non-VFS widening, 'safe' boundaries, copied parts | `zml/zml/io/direct_loader.zig:2515-2516` |
| low | docs | confirmed | source | Stale or misleading names and comments around layouts and bindings | `zml/zml/io/loader.zig:39-42` |
| low | docs | confirmed | story | `Loader.Binding` and `TensorStore.Binding` are different concepts with the same name; stale comments about borrowing | `zml/zml/io/loader.zig:105-113` |

## Appendix B: refuted

- **`swapEngramHostPinnedBuffers`' errdefer frees owners it leaves in the list** (`monorepo/llmd/models/deepseek41.zig:2814-2831`; CTX "Open work"). The note is accurate as worded, but nothing goes wrong:
  - `Buffer.deinit` empties the buffer in place (`_shards = .empty`, `zml/zml/buffer.zig:96-101`), so the owners left in the list are never freed twice.
  - The caller's errdefer at deepseek41.zig:2806 frees the views and the unswapped owners.
  - `Model.deinit` never runs after a failed load (main.zig:479 registers it after the load).

  What remains is small: the errdefer at deepseek41.zig:2810 is the last statement of `loadBuffers` and can never run. The CTX item can be dropped.
