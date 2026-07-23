# Fully Adaptive Source and DMA Loader

## Summary

Remove the hidden HF/S3/GCS parallel_read concurrency layer. One controller will optimize a global source tuple—read concurrency and request size—plus an independent per-device DMA limit:
adaptive source tuple (concurrency, request size)
-> one VFS Range request per admitted read
-> DmaMapped blocks
-> adaptive DMA admission
Retries remain inside each VFS. They retain their source credit and publish typed telemetry out-of-band through ReadStatsProvider.
Defaults:
Source concurrency: adaptive, initial 12, absolute cap 128.
Remote no-response bootstrap: at most 32 until timing data arrives.
DMA concurrency: adaptive, initial 8, cap 32 per device.
Request size: adaptive from the VFS minimum through 128 MiB.
DMA block: fixed 2 MiB.
Pinned-memory hard limit: 2 GiB.

## Interfaces and Data Plane

Replace the current scalar controls with fixed/adaptive configurations:Read parallelism: .adaptive = { initial = 12, maximum = 128 } or .fixed.
DMA parallelism: .adaptive = { initial = 8, maximum = 32 } or .fixed.
Request size: .adaptive = { initial = source_minimum, maximum = 128 MiB } or .fixed.

Retain explicit benchmark overrides. Add environment controls for adaptive initial values, caps, and fixed read/DMA values; the existing fixed request-size variable continues disabling size adaptation.
Enforce absolute limits of 128 reads, 32 events/device, and 128 MiB/request. Automatic sizes are power-of-two MiB values no smaller than the VFS minimum or DMA block.
Use one global source profile. If several profiles unexpectedly appear, use the largest minimum, aggregate their telemetry, and log that the mixed load is using one conservative tuple.
Delete parallel_read.zig. HF/S3/GCS issue one physical Range GET for each positional VFS call and perform bounded retries serially within that call.
Replace read_pool options with flat VFS fields: minimum_request_size, max_retries, and initial/maximum retry delay. Remove worker, queue, and chunk-size settings.
Keep reusable stateless Range parsing and scatter filling. Strictly validate 206; correctly position a 200 response that ignored Range.
Replace the precomputed fixed-size job array with a mutex-protected round-robin scheduler. Each admission claims the next tensor range using the current request size, so size changes affect only unscheduled bytes and concurrent ranges of one tensor remain supported.
Bound the currently feasible read width by max_pinned_bytes / request_size. Dynamically resize the read and lifecycle gates; reductions drain naturally. Remote lifecycle slack remains at most eight requests but is clipped by the pinned-byte capacity.

## VFS Telemetry and Retry Feedback

Extend cumulative ReadStats with successful-request timing buckets for 2, 4, 8, 16, 32, 64, and 128 MiB:Attempts, successes, and successful bytes.
Time from attempt start to the first response body byte.
Time from the first body byte through the final requested byte.
Transient retries, timeouts, server failures, throttles, and retry delay.

Timestamp immediately before request creation, after reading the first body byte, and after filling the final destination slice. Only successful, non-retried attempts contribute to bandwidth/overhead estimates; all attempts contribute failure counters.
Publish retry/throttle counters before sleeping. Retry delays retain the admitted source credit and use jitter, preventing replacement work from increasing concurrency.
Treat feedback according to cause:Any throttle causes an immediate 30% source reduction and five-second performance-probe cooldown.
A single transient retry does not change concurrency.
A transient failure ratio above 10% for two reliable windows causes a 30% reduction.
Terminal errors use the existing load abort-and-drain path.

The provider remains an out-of-band ZML extension rather than part of std.Io. Size buckets separate old and candidate timing samples while controller epochs continue attributing final GPU commits.
Local files report total positional-read service time from the loader; TTFB is unsupported and does not participate in local decisions.

## Adaptive Controller

Retain one unified controller and permit only one active probe: source concurrency, source-size tuple, or DMA.
Start local sources at 12 reads and remote sources at 12, bootstrapping 12 → 24 → 32 only while no response exists. Do not exceed 32 until successful TTFB/body samples exist.
Estimate source characteristics from settled timing:Per-request service time = TTFB + body time.
Body bandwidth = bytes/body time.
Required concurrency = ceil(1.2 × target_bandwidth × service_time / request_size).
Clamp it by the manual cap and pinned-memory feasibility.

When DMA is starved and reads are saturated, probe the larger of the modeled concurrency and a 1.5× startup step. Keep it for a 3% committed-goodput gain or a material starvation reduction with goodput within 3%.
Maintain low-load TTFB and body-bandwidth baselines per request-size bucket. Two windows of 1.5× TTFB inflation or 20% body-rate loss veto further concurrency only when committed goodput is flat or regressing.
Probe request sizes by doubling toward 128 MiB. Compute the candidate concurrency from the timing model; if memory cannot preserve the old width, probe the modeled feasible (concurrency, size) tuple together.
Score source-size and concurrency probes using GPU-committed logical bytes from matching request epochs. TTFB/body measurements are explanatory signals and pressure guardrails, not the final objective.
Activate a source-size probe only after at least eight full-sized candidate responses and max(64 MiB, 4 × request_size) matching bytes. A clear ±10% result may finish after 50 ms; ambiguous probes require 100 ms. Keep a larger size only for at least 3% goodput gain, or when it removes throttling without exceeding a 3% regression.
Prefer the lower-memory tuple whenever results are within 3%. After changing request size, reset source timing baselines and re-evaluate concurrency.
Probe DMA only after the source tuple has settled and device queues are demonstrably fed. Preserve the current completion-latency pressure checks and 32-event/device maximum.
Suppress a probe when its estimated cost exceeds 25% of remaining wall time or fewer than four candidate-sized requests remain. Reprobe settled dimensions after two seconds when substantial work remains.
Log source tuple, feasible width, TTFB, body bandwidth, overhead fraction, retry classes, probe attribution, pinned high-water, DMA starvation, and final selected tuple.

## Performance Development and Acceptance

Record fresh baselines before replacement:Five warm-file runs for one B70, four-device sharding, and replication.
Three runs for S3Proxy at 10 ms/1000 MiB/s and 250 ms/1000 MiB/s, plus one 1000 ms/100 MiB/s run.
Three real-AWS runs using bench_aws.sh; record the current approximately 948 MiB/s result.

After removing parallel_read, first rerun the current fixed 16 MiB/32-read configuration to isolate data-plane regressions.
Establish static oracle tuples:File: 2–16 MiB requests and widths 8–32, then verify widths through 128 at the best size.
S3Proxy: 16/32/64/128 MiB against widths 16/32/48/64/96/128, clipped by pinned memory.
AWS: test only the S3Proxy-shortlisted tuples, then take three-run medians.
Sweep DMA around 4/8/12/16/24/32 only after selecting the source tuple.

Require the adaptive result to be within 3% of the best static median for every profile, selecting the lowest pinned footprint within that band.
Require no regression from fresh local baselines and no regression from 948 MiB/s on AWS; treat improvement above that as the primary remote result.
Verify one successful logical remote read produces one physical Range request, larger request sizes are not silently split, and observed physical concurrency never exceeds the controller’s admission.
Run controller/unit tests, mock-server retry/timing tests, core Zig tests, release CUDA/oneAPI builds, four-B70 correctness runs, and local perf profiles. Do not build or modify XLA.
Keep CTX.md authoritative, including convergence traces, AWS results, rejected tuples, and the final default parameters. Preserve user scripts and existing performance recordings.
