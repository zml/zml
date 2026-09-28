# Furiosa migration validation — 2026-09-28

ZML baseline: `3866ec6bce5866cbb31f61dc1817ae1c633cc8a3`, with the current
uncommitted migration. XLA: `6fc35380c8` (no source changes in this task).
Linux x86-64; pinned Bazel 9.1.1 for ZML and 8.7.0 for XLA.

## Build and packaging

- Exact XLA build passed: `bazel build --config=hermetic_linux_x86 --jobs=16 //xla/pjrt/furiosa:pjrt_c_api_furiosa_plugin`.
- Affected ZML targets and full `//...` Furiosa build passed using the override,
  `--jobs=16 --@zml//platforms:furiosa=true --@zml//platforms:cpu=false`.
- CPU-only `//...` build passed with no repository override and Furiosa disabled.
- Enabled Furiosa without an override fails with the intended setup diagnostic.
- Retired Bazel flags and explicit `llama_tests` platform names are rejected.
- Invalid explicit plugin path and missing manifest runfile fail visibly.
- Out-of-workspace launch from `/tmp` resolves the plugin using repository mapping.
- `GetPjrtApi` is exported, the loaded API reports 0.115, client identity is
  `furiosa`, and memory/layout extensions are available.
- Initial testing exposed `$ORIGIN` lookup failure for `libunwind.so.1` through
  the override symlink. The loader now resolves the real file before loading;
  runtime tests pass without `LD_LIBRARY_PATH` or a system installation.
- After the independent XLA rebuild, XLA output, override, and LLM runfile all
  have SHA-256 `0db5c9303df4fe5ef2008c5ee3830e9eb9d1f9c6bdcf1676bc4f03d4e3866ced`.
  The rebuild produced identical bytes. The runfile resolves directly to that
  output; no copied plugin needs refreshing.
- Changed Zig files pass pinned `zig fmt --check`; changed Bazel files use the
  repository's pinned buildifier. `git diff --check` passes.

## Runtime environment and generation

Compiler: `/tmp/furiosa-integrated-sdk/furiosa-tcc` (TCC 2026.3.0).
Runtime: `/tmp/furiosa-integrated-sdk/libdevice_runtime.so`, bridge/19 contract;
plugin executable format 7 and topology generation 1 are unchanged.
Cache: `/home/kevin/.local/state/xla-rngd/furiosa/tcl-ir-v7/zml`, populated by
these runs; subsequent checks may reuse compiled programs. Optional compiler
fusion flags were not enabled by these commands.

A four-PE, chip-0 BF16 projection (`1x32 @ 32x32^T`) passed elementwise checks
from `/tmp` with `XLA_FURIOSA_PJRT_LIBRARY` unset.

The exact override-based Llama command in [README.md](README.md) passed on
chip 0 with eight PEs, vanilla attention, BF16
`/var/models/meta-llama/Llama-3.1-8B-Instruct`, sequence length 128 and greedy
sampling. It generated the integers 1 through 20 correctly. Prefill compilation
took about 3m32s and decode compilation about 1m2s with this cache state. This is
an acceptance smoke run, not a benchmark or proof of general model quality.

## Regression suites

| Check | Result |
| --- | --- |
| CPU `bazel test --jobs=16 --config=debug --@zml//platforms:furiosa=false --@zml//platforms:cpu=true //...`, no override | All 24 targets pass. |
| Generic Furiosa `//zml:test`, four PEs, chip 0 | Times out at 300 seconds while compiling the first large attention case; full suite is not validated. |
| Focused four-PE tests | 12 pass, 3 fail, none skipped. The three failures are pinned allocations: buffered writer, `toMemory`, and bulk placement. |
| Focused StableHLO MLA attention and Triton availability | Pass; included in the focused count above. |
| Eight-PE pinned writer, chip 0 and chips 0,1 | Fails to allocate pinned memory in both processes. |
| Eight-PE sharded execution and input ownership, chip 0 and chips 0,1 | Pass in both processes, including repeated execution and unchanged input values. |
| `llama_attention_tests`, CPU and Furiosa enabled | Pass: F32/BF16 vanilla SDPA, including nonzero offsets. |
| 32-layer whole-forward CPU comparison, four PEs, position 0 | Pass: exact argmax, existing KV tolerances, untouched cache bits. |
| Same whole-forward comparison, eight PEs, position 0 | Pass with the same checks. |
| Eight-PE whole-forward with CPU-computed history, offset 7 | Fails existing KV tolerance; argmax agrees (1001). Maximum absolute error reported is 0.09375; no NaN/Inf. |
| Eight-PE teacher-forced decode, counting prompt, two continuation steps | Pass: 0/14 argmax mismatches, including 0/2 continuation mismatches; finite active caches and unchanged padding. This diagnostic does not replace the separate KV tolerance gate. |

The CPU suite initially exposed an existing test assumption: CPU aliases pinned
host memory to its device memory. The writer regression now compares against the
requested memory's PJRT kind and explicitly still requires `host_pinned` on
Furiosa. No numerical tolerances were changed and no failing tests were removed.

Pinned allocations return `ResourceExhausted` for even 32/128-byte requests.
Opening `/dev/dma_heap/system` as the current user directly returns
`EACCES` / `Permission denied`; the device is owned by `root:plugdev` and this
session is not in that group. Device permissions were not changed. The pinned
cases remain enabled and failing, rather than being converted to skips.

The nonzero-history mismatch is an unresolved numerical validation failure.
This task did not establish whether it predates the migration. It must not be
presented as a validated history/KV result despite matching output tokens.
No XLA compiler, runtime ABI, or model tolerance changes were made to hide it.

The dedicated `//zml:furiosa_8pe_test` runs separately from default four-PE tests.
Its pinning and execution checks are separate, so a permissions failure does
not prevent the deterministic one/two-chip execution gate from running.
Model checks used `--config=debug`, the repository override, and both CPU and
Furiosa enabled. Whole-forward arguments were `--platform=furiosa
--furiosa-pe-count=4` (then `8`) `--compare-cpu --forward-only --layers=32
--seqlen=1 --cache-seqlen=128`, plus the model path above. The history run added
`--token-offset=7 --prefill-history`; the decode run instead added
`--decode-comparison-steps=2
--decode-comparison-prompt='Count from 1 to 20, separated by commas.'`.

### Reproduce the focused four-PE checks

Use the SDK exports from [README.md](README.md). Include the discovery tests
in the filters; omitting them can yield a misleading zero-test run. The full
unfiltered suite remains unchanged.

```bash
furiosa_test_flags=()
for pattern in 'zml.test.zml' 'test_0' 'BufferedMemoryWriter' 'toMemory' \
  'bulk memory placement' 'Furiosa client options' 'Triton availability' \
  'execute stablehlo mla kernel'; do
  furiosa_test_flags+=(--@rules_zig//zig/settings:zigopt=--test-filter
    "--@rules_zig//zig/settings:zigopt=$pattern")
done
bazel test --override_repository=libzml_furiosa=/home/kevin/furiosa/xla-override \
  --jobs=16 --config=debug --@zml//platforms:furiosa=true --@zml//platforms:cpu=false \
  --test_env=XLA_FURIOSA_COMPILER --test_env=XLA_FURIOSA_RUNTIME_LIBRARY \
  --test_env=XLA_FURIOSA_COMPILER_CACHE --test_env=XLA_FURIOSA_VISIBLE_DEVICES=0 \
  "${furiosa_test_flags[@]}" //zml:test
```


The optional Python quality and FP8 converter tests could not import NumPy in
the system Python environment. They did not execute. Those utility sources
were retained unchanged; FP8 quality is not revalidated by this migration.

## Local logs

Build and test logs are under `/tmp/zml-*.log` for this session, with command
and exit-code records in `/tmp/zml-validation-results.json`,
`/tmp/zml-device-results.json`, and `/tmp/zml-final-device-results.json`.
The successful CPU rerun is `/tmp/zml-cpu-tests-fixed.log`; the initial batch
record retains its original failure. Final focused evidence is
`/tmp/zml-focused-final.log` and `/tmp/zml-8pe-final-{0,0-1}.log`. These temporary files are not required by the
build and are not committed test fixtures. Historical captures removed from
the model testdata directory remain in Git history.
