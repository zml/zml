# Adaptive Concurrency Control for Finite Batch Tensor Transfers

> Historical design research. The adaptive pageable-staging controller was
> removed in July 2026 when the loader moved to a static vectored DmaMap
> pipeline. See `CTX.md` and `zml/io.zig` for the current architecture.

## Executive recommendation

The strongest options for this problem are not the classic “increase until something breaks” schemes by themselves, but **delay-aware adaptive limiters that explicitly watch for queue formation**, combined with a **fast startup probe** so the controller reaches a near-saturating concurrency before too much of the finite batch has already finished. The best practical default is therefore a **hybrid controller**: start conservatively, increase multiplicatively at first, then switch to a mostly stable **Gradient2/Vegas-style delay controller** that optimizes **bytes committed to GPU memory per second** while backing off on rising queue or resource-wait time. That recommendation is most consistent with the logic behind TCP Vegas, BBR’s fast startup, Netflix’s adaptive concurrency limits, and Envoy’s gradient controller: throughput alone is not enough; the controller also needs an early signal that the bottleneck is starting to queue. citeturn25view0turn9view4turn15view0turn17view3turn30view3

The next-best option is a **straight Netflix/Envoy-style gradient controller**, adapted from request latency to **byte-weighted transfer latency or wait time**. It is simpler than a full BBR-style model, more noise-tolerant than raw Vegas because it smooths the baseline, and operationally lighter than hill climbing with full response-surface estimation. A reasonable third option is **throughput hill climbing with explicit delay guardrails**, but only if the system cannot measure a trustworthy no-load latency or queue-wait baseline. Pure AIMD, standalone Little’s Law, and generic PID control are all useful components or fallbacks, but they are weaker default choices for this finite-batch transfer problem. citeturn20view0turn9view4turn10view0turn22view0turn19view0turn8view5

## Which algorithms fit this problem best

**Additive increase and multiplicative decrease** uses a very simple signal: success versus congestion, timeout, throttling, or explicit failure. In the Netflix implementation, it increases additively when inflight demand is near the current limit and decreases multiplicatively on an error or timeout; the default backoff ratio is configurable in the range `[0.5, 1.0)`. That makes AIMD cheap to implement and robust when the only reliable signal is “things started failing.” But it is also late: it does not detect incipient saturation before queues build, so it usually backs off only after throughput or latency has already degraded. For a finite batch, that lag is costly, because a lot of the bytes may finish during the overshoot-and-recover phase. It is therefore best as a safety brake, not as the main optimizer. citeturn19view0turn6search0

**Throughput hill climbing** uses measured output directly: change concurrency, look at throughput, and keep moving in the direction that helped. That is attractive because your objective really is total batch completion time, so throughput is the right objective signal. The problem is that production experience with the .NET ThreadPool found this hard in noisy environments: small concurrency changes are difficult to attribute to throughput changes, the concurrency-throughput curve moves over time, and control actions have dead-time effects. Microsoft ended up using signal-processing techniques to distinguish the effect of concurrency changes from noise, and even then noted that adjustments happen relatively slowly because the controller must gather enough data to stabilize its estimate. That makes pure hill climbing weaker for finite batches with large transfers unless it is augmented with byte-weighting, minimum sample sizes, and a separate queue-delay guardrail. citeturn10view0turn10view1turn10view3turn22view0

**TCP Vegas-style delay control** is much closer to what this problem needs. Its key signal is the gap between a no-load baseline latency and current sampled latency. The Netflix concurrency-limits implementation expresses the practical queue estimate as `L * (1 - minRTT / sampleRtt)` and then nudges the limit up if the estimated queue is below a small threshold, or down if it exceeds a higher threshold. This is a strong fit because storage, pinned-memory staging, and H2D copy all show saturation by **extra waiting** before they show obvious loss of effective bandwidth. Vegas therefore detects the wrong side of the knee earlier than throughput-only methods. Its weaknesses are that raw minimum-latency baselines drift, and its additive step size can still be too slow if the batch is short and the optimal concurrency is far above the initial value. citeturn9view0turn9view1turn30view3

**BBR-style bandwidth and latency estimation** uses two signals at once: an estimate of bottleneck bandwidth and an estimate of no-load path latency. The BBR paper is explicit that highest throughput with lowest delay occurs when delivery rate matches bottleneck bandwidth and inflight data is near one bandwidth-delay product, and that neither signal alone is enough because one becomes hard to observe when the other dominates. BBR’s strengths for this problem are its very fast startup search and its ability to separate “how much bandwidth is there?” from “how much queue am I creating?” Its weaknesses here are practical: the full model is more complex, it assumes a notion similar to ACK-clocked delivery-rate sampling, and the paper itself notes that application-limited request/response traffic may never fill the pipe enough to observe bottleneck bandwidth, while stale latency estimates must be periodically refreshed. For a finite batch of tensor transfers, borrowing **BBR’s startup/probe pattern** makes sense, but a full BBR clone is usually overkill. citeturn15view0turn15view1turn15view3turn15view4

**Netflix-style adaptive concurrency limits** are the strongest production-style match. Netflix’s published design uses the ratio of best-case latency to sampled latency as a gradient, then updates the limit with `newLimit = currentLimit × gradient + queueSize`; Netflix reports using a square-root headroom term so growth is fast at small limits and more stable at large ones. The open-source `Gradient2` variant improves on raw minimum-latency control by comparing a short-term RTT measurement to a long-term exponentially smoothed baseline, clamping the gradient, and applying smoothing to the limit update. Envoy’s adaptive concurrency filter implements the same family of ideas, including periodic minRTT remeasurement under a very small concurrency cap. Relative to raw Vegas, this family is less fragile under burstiness and baseline drift, and relative to BBR, it is much simpler to implement. For this workload, it adapts well if RTT is reinterpreted as **byte-weighted end-to-end transfer time or queue/wait inflation**. citeturn25view0turn20view0turn21view1turn9view4turn9view3

**Little’s Law-based concurrency estimation** is indispensable as a sanity check, but insufficient as a standalone controller. Little’s Law says average concurrency equals average throughput times average time in system, and the 50th-anniversary review emphasizes that this relation is exact even over finite observation intervals under the stated conditions. Netflix’s background material uses the same idea to motivate why concurrency, not just RPS, is the right control variable. The catch is that the law is descriptive, not prescriptive: it tells you the concurrency implied by the current throughput and latency, but not whether the current latency is healthy or queue-inflated. To turn it into a controller, you still need either a target latency, a no-load baseline, or another signal that marks saturation. So Little’s Law belongs inside the controller’s bookkeeping, not at the top of the control loop. citeturn8view5turn9view0turn25view0

**PID control** can work if you can define a stable target, such as “keep queue delay at 10% of no-load latency” or “keep staging-buffer occupancy at a fixed setpoint.” But this problem is a time-varying, noisy queueing system with dead time between admissions and measured completions, and experience applying control theory to real software controllers found that variability, curve drift, and dead-time must all be handled explicitly. In practice, that means PID needs careful gain tuning and often an outer loop that adapts the setpoint or resets the integrator when the bottleneck shifts. That is harder to justify than using a delay-aware limiter whose update law already encodes “increase when no queue, decrease when queue grows.” PID is therefore viable for a well-instrumented specialized deployment, but not the best default. citeturn22view0turn10view3

**Startup probing followed by a mostly stable limit** is not a separate signal so much as the right control shape for a finite batch. BBR explicitly uses a fast startup search to discover bandwidth quickly, then spends most of its time in a comparatively stable probing regime. Netflix’s own description of adaptive limits shows exactly the same pattern: start low, probe higher while latency stays flat, and back off when latency increases. For finite tensor batches, this matters more than it does for indefinite services: if the controller spends the first quarter of the batch inching upward by one, it loses the race before convergence. So even if the steady-state controller is Gradient2 or Vegas-like, the **outer policy should be “fast probe, settle, recheck occasionally.”** citeturn15view3turn25view0

## What should be measured

The controller should optimize **end-to-end goodput measured in bytes committed to GPU memory per second**, not completed-transfer count and not raw storage-read throughput. NVIDIA’s guidance on effective bandwidth is explicit that bandwidth should be measured as bytes transferred divided by time, and the tensor-loading objective is exactly “how many useful bytes reached GPU memory.” Storage-read throughput is still useful diagnostically, but it is not the objective: it can rise while pinned-memory buffers back up or H2D copies stall, and the batch still finishes late. In modern ML input pipelines, end-to-end training time is strongly affected by whether data reaches the accelerator fast enough, and slow input processing can leave expensive GPUs or TPUs idle even when upstream work is active. citeturn24view0turn23view0turn23view1turn23view2

Throughput alone is not sufficient. Delay-based algorithms exist for a reason: throughput often stays flat around the knee while hidden queues grow. Netflix’s concurrency-limit design starts from the premise that latency measurements reveal when queuing happens; Vegas estimates queue directly from `minRTT` and `sampleRtt`; Envoy computes a gradient from `minRTT` and `sampleRTT`; and BBR’s model explicitly needs both bandwidth and no-load latency because either one alone hides part of the saturation picture. For this workload, the most useful protective signals are therefore **queueing or resource-wait time** and **latency inflation relative to a no-load baseline**, plus hard-failure signals such as timeout, retry, throttle, or storage 429/503-style backpressure. citeturn9view0turn9view4turn15view1turn25view0

The concrete measurement set should be small but layered. At the **end-to-end level**, track committed bytes per second, byte-weighted transfer latency, active transfers, inflight bytes, and the remaining batch bytes. At the **stage level**, track storage-read bytes per second and storage wait time; host-memory staging queue occupancy and bytes pinned; H2D DMA bytes per second, copy-engine or stream wait time, and DMA completion latency. CUDA documentation matters here because asynchronous host-to-device copies require pinned host memory; pinned memory delivers the highest transfer bandwidth but is scarce; and overlap depends on hardware copy engines and non-default streams. Those facts mean that staging-buffer pressure and H2D wait time are not incidental diagnostics; they are often the earliest visibility into the actual bottleneck. citeturn24view1turn24view2turn24view3

## How to detect real marginal improvement

Because tensor sizes vary widely, the controller should treat **bytes**, not operations, as the statistical weight of each sample. A simple and effective representation is a byte-weighted window:
\[
G_w=\frac{\sum_i b_i^{\text{gpu-committed}}}{\Delta t_w}
\]
for end-to-end goodput, and
\[
\bar d_w=\frac{\sum_i b_i\, d_i}{\sum_i b_i}
\]
for byte-weighted latency or queue-delay. This matches how effective bandwidth is defined in CUDA and avoids letting many tiny tensors outweigh a few very large ones. If stage-specific waits are available, the latency term should be decomposed into service time and queue/wait time, because that separates “hardware got slower” from “I created excess queue.” citeturn24view0turn9view4turn15view1

To decide whether an increase was real, do not compare single windows. Use both a **minimum sample floor** and **hysteresis**. Production adaptive limiters wrap their updates in windows for exactly this reason: Netflix’s `WindowedLimit` defaults to a one-second update cadence with a minimum of ten samples and allows windows as short as 100 ms, because under-sampled windows are too noisy. For transfer control, the right synthesis is to close a window only when both enough **time** and enough **bytes** have accumulated. In practice, a good starting rule is: update every `max(200 ms, enough time to collect 64–256 MiB of committed bytes)` with a cap around `500 ms`, and require at least several completed transfers or enough bytes before acting. That recommendation is an inference from the cited request-controller windowing practice plus the fact that the objective metric is byte bandwidth, not request count. citeturn29view0turn29view1turn29view4turn24view0

A practical marginal-improvement test is:

\[
\text{improve if } G^{\text{fast}} > (1+\epsilon*{\uparrow})G^{\text{ref}}
\]
and
\[
q^{\text{fast}} < q*{\text{hi}}
\]
for at least `h_up` consecutive windows, where `G^fast` is a fast EMA of goodput, `G^ref` is either the current settled level or the best recent EMA, and `q` is either normalized queue delay
\[
q=\max\!\left(0,\frac{\bar d-d_0}{d_0}\right)
\]
or a byte-weighted resource-wait ratio. Using a small positive threshold such as `ε_up = 0.03` avoids chasing measurement noise; using a higher decrease threshold such as `ε_down = 0.05` avoids oscillating between adjacent limits. This mirrors the experience from hill-climbing controllers that noisy throughput differences are easy to misread, and from delay-based controllers that queue growth is the early warning that should veto an apparent throughput tie. citeturn10view3turn10view4turn9view4turn25view0

## Preferred controller design

The preferred default is a **hybrid byte-weighted startup-plus-gradient controller**. Its steady-state update law should borrow from the Netflix/Envoy family:
\[
g=\operatorname{clamp}\!\left(g*{\min},\,1,\,\frac{d_0(1+\text{buffer_pct})}{\bar d}\right)
\]
\[
C*{\text{target}}=g\,C+\sqrt{C}
\]
\[
C \leftarrow \operatorname{clamp}\!\big((1-s)C+s\,C*{\text{target}},\,C*{\min},\,C\_{\max}\big)
\]
where `d0` is the no-load or lightly loaded byte-weighted latency baseline, `d̄` is the current byte-weighted latency, `sqrt(C)` is the headroom term, and `s` is a smoothing factor. This follows the structure published by Netflix and Envoy and the `Gradient2` implementation, while adapting the measurements from RPC latency to transfer latency and wait time. citeturn25view0turn9view3turn20view0turn21view4

For a finite batch, however, the controller should not begin in steady state. It should begin in **STARTUP**, because BBR’s central insight is correct for this workload too: if the search space is wide, slow additive exploration wastes too much wall time. The startup phase should therefore use a multiplicative probe such as `C ← ceil(1.8C)` (bounded by the hard maximum and memory budget) while goodput improves by at least `3%` and queue inflation stays below about `10–15%`. The first time the goodput gain falls below threshold, or queue inflation exceeds threshold, or errors/throttling appear, the controller should roll back to the best recent limit and enter **STEADY**. That is an intentional simplification of BBR startup plus delay-based settling. citeturn15view3turn25view0turn9view4

The controller state should contain the following fields: current concurrency `C`; hard count cap `Cmax`; minimum cap `Cmin`; staging-byte budget `Bstage_max`; inflight bytes; fast and slow EMAs of end-to-end goodput; fast and slow byte-weighted latency or wait; no-load baseline `d0`; recent best `(C, G)` pair; counters for consecutive good and bad windows; and a timer for periodic reprobes. The **admission rule** should be dual: admit a new tensor only if `inflight_count < C` **and** `inflight_staging_bytes + estimated_tensor_bytes <= Bstage_max`. That second condition is essential because pinned memory is scarce and tensor sizes vary; a count-only controller can still overfill the staging area with a few huge tensors. Reductions in `C` should affect only new admissions; they should never cancel already in-flight transfers. That is exactly how practical concurrency limiters are enforced: once the limit is hit, new work is blocked or rejected, while existing work drains naturally. citeturn24view1turn24view2turn17view3turn9view0

A concrete update policy is:

```text
state:
  mode ∈ {STARTUP, STEADY, BACKOFF}
  C, Cmin=1, Cmax
  Bstage_max
  goodput_fast, goodput_slow
  delay_fast, delay_slow, d0
  best_C, best_goodput
  consecutive_up, consecutive_down
  last_reprobe_time

on each update window w:
  measure:
    bytes_gpu = committed bytes to GPU during w
    bytes_read = storage bytes read during w
    inflight_count, inflight_stage_bytes
    delay = byte-weighted end-to-end latency during w
    wait = byte-weighted queue/resource-wait during w
    errors = retries + timeouts + throttles during w

  goodput = bytes_gpu / window_time
  q = max(wait / max(d0, tiny),
          (delay - d0) / max(d0, tiny),
          0)

  update EMAs for goodput and delay
  refresh d0 only from lightly loaded windows
    (e.g. inflight_count <= 2 or q <= 0.02)

  if errors > 0 or q > q_hard:
      C = max(Cmin, floor(beta_hard * C))
      mode = BACKOFF
      continue

  if mode == STARTUP:
      if goodput_fast > (1 + eps_up) * best_goodput and q < q_soft:
          best_goodput = goodput_fast
          best_C = C
          C = min(Cmax, startup_step(C))
      else:
          C = best_C
          mode = STEADY
      continue

  gradient = clamp(g_min, 1.0, (d0 * (1 + buffer_pct)) / delay_fast)
  C_target = gradient * C + sqrt(C)
  C_next = clamp(round((1 - smoothing) * C + smoothing * C_target), Cmin, Cmax)

  if goodput_fast > (1 + eps_up) * best_goodput and q < q_soft:
      consecutive_up += 1
      consecutive_down = 0
      if consecutive_up >= h_up:
          best_goodput = goodput_fast
          best_C = C_next
          C = max(C, C_next)
  elif goodput_fast < (1 - eps_down) * best_goodput or q > q_soft:
      consecutive_down += 1
      consecutive_up = 0
      if consecutive_down >= h_down:
          C = max(Cmin, min(C_next, floor(beta_soft * C)))
  else:
      consecutive_up = 0
      consecutive_down = 0
      C = C_next

  if periodic_reprobe_due() and remaining_batch_bytes is still large:
      C = min(Cmax, C + reprobe_step(C))
```

The initial parameters I would use are these. Set `startup_step(C) = max(C + 1, ceil(1.8C))`; `buffer_pct = 0.10`; `g_min = 0.5`; `smoothing = 0.2`; `β_soft = 0.85`; `β_hard = 0.7`; `ε_up = 0.03`; `ε_down = 0.05`; `q_soft = 0.10`; `q_hard = 0.20`; `h_up = 2`; `h_down = 1`; and periodic reprobe every `2–5 s` while the remaining batch is still large enough to benefit. Use update windows with a **minimum** around `200 ms`, a **maximum** around `500 ms`, and a **minimum byte floor** such as `64 MiB`; if the system is very fast, raise the byte floor rather than shrinking the time floor below `100 ms`. Those values are a synthesis of the cited production controllers’ one-second / hundred-millisecond window scales, Gradient2’s smoothing and clamped-gradient design, and Netflix/Envoy’s square-root growth headroom. citeturn29view0turn29view1turn20view0turn21view4turn25view0turn9view3

## Global control versus stagewise control

The implementable recommendation is to adapt **one global end-to-end concurrency limit on tensor admissions**, backed by **bounded intermediate queues** and **fixed per-stage semaphores or byte budgets**. This is the best default because the objective is end-to-end completion time, not local stage utilization, and because the bottleneck can move among storage, host preprocessing, pinned-memory staging, and H2D DMA over the life of the batch. ML input-system work shows that keeping accelerators fed is an end-to-end pipeline problem, and that CPU/RAM requirements and bottlenecks vary significantly across jobs; CUDA documentation adds that H2D overlap and bandwidth depend on pinned memory and available copy-engine resources. A single global controller sees the objective directly, while stage queues and waits reveal where backpressure is forming. citeturn23view0turn23view1turn23view2turn24view1turn24view2

I would not start with separate adaptive controllers for storage and GPU DMA. Independent controllers tend to fight through the intermediate queue: the storage controller can become “successful” by overfilling pinned buffers even while the DMA controller is already saturated, and the DMA controller can become underfed while the storage controller backs off on a transient upstream stall. If separate adaptation is ever needed, it should be reserved for cases where the stages are genuinely independent and separately shared with unrelated traffic. In most deployments, the safer design is: one end-to-end adaptive admission controller; one bounded storage→host queue; one bounded host→GPU queue; fixed semaphores for local non-adaptive resource constraints such as number of storage worker threads, maximum outstanding reads, or maximum number of H2D streams; and a hard byte budget for pinned memory. That structure turns stage congestion into explicit wait time that the global controller can see, instead of letting it become silent memory growth. citeturn8view5turn24view1turn24view2turn23view2

## Failure modes and how the controller avoids them

The first failure mode is **throughput-only false positives**. Around the saturation knee, goodput can look flat while queues quietly expand. The controller avoids this by letting goodput decide the objective, but letting **queue inflation veto increases** and trigger reductions before effective bandwidth collapses. That is precisely why Vegas, Gradient, Envoy’s filter, and BBR all incorporate latency or no-load delay rather than using delivered bytes alone. citeturn9view4turn15view1turn25view0

The second failure mode is **noise from highly variable tensor sizes**. Pure completion counts make tiny tensors dominate the statistics, and small windows cause random variance to look like signal. The controller avoids this by using byte-weighted goodput and byte-weighted latency, by imposing minimum-byte windows, and by using hysteresis with different thresholds for increasing and decreasing. That follows both the bandwidth-measurement guidance from CUDA and the real-world experience from hill-climbing controllers that throughput measurements are noisy and must be interpreted over windows. citeturn24view0turn10view3turn29view4

The third failure mode is **slow convergence on a finite batch**. Additive one-step exploration can spend too much of the batch below the optimal concurrency. The controller avoids that by using multiplicative startup probing, then settling into a mostly stable delay-aware regime. That borrows the right part of BBR without importing the full complexity of a bandwidth-delay-product model. citeturn15view3turn25view0

The fourth failure mode is **baseline drift**. Raw minimum latency can become unrealistically low or stale, which causes overreaction. The controller avoids that by refreshing `d0` only from lightly loaded windows, or by using a smoothed long-term baseline in the style of `Gradient2`, and by periodically reprobe-checking whether conditions changed. This is exactly the motivation behind Gradient2’s long-term RTT measurement and Envoy’s explicit minRTT recalculation. citeturn20view0turn21view1turn9view4

The fifth failure mode is **memory blow-up from large tensors**. A count limit alone does not bound pinned-memory staging if a few huge tensors arrive together. The controller avoids that by making admission subject to both a count cap and a pinned-byte budget, and by letting reduced limits stop new admissions without canceling work already in flight. CUDA’s own guidance that pinned memory is scarce and should not be overused makes this an essential part of the design, not an implementation detail. citeturn24view1turn24view3

The bottom line is straightforward. If you want the best default for minimizing finite-batch completion time, use **one global, byte-weighted, delay-aware admission controller with fast startup probing, square-root headroom, smoothed decreases, bounded staging bytes, and periodic lightweight reprobes**. If you need something off-the-shelf in spirit, adapt **Netflix Gradient2 / Envoy Gradient** to transfer latency and wait time. If you cannot measure queue delay reliably, fall back to **throughput hill climbing with hard AIMD-style backoff on error or throttle events**. Everything else is either too slow, too blind, or too complex for this specific job. citeturn20view0turn9view3turn19view0turn10view0turn15view3
