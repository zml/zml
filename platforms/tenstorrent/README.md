# Tenstorrent

Enable the experimental Tenstorrent PJRT backend with
`--@zml//platforms:tenstorrent=true`. The initial integration loads a locally
built plugin; it does not download a prebuilt SDK.

Build these targets in the XLA checkout on branch `steeve/tt`:

```sh
bazel build --config=hermetic_linux_x86 \
  //xla/pjrt/tenstorrent:pjrt_c_api_tenstorrent_plugin \
  //xla/pjrt/tenstorrent:hardware_test
```

Set `ZML_TENSTORRENT_PLUGIN_PATH` to the absolute path of
`bazel-bin/xla/pjrt/tenstorrent/libpjrt_c_api_tenstorrent_plugin.so` and
`TT_METAL_RUNTIME_ROOT` to the `+tenstorrent+tt_metal` directory inside
`bazel-bin/xla/pjrt/tenstorrent/hardware_test.runfiles`. That runfiles directory
combines device sources, SFPI, and generated firmware support objects. Preserve
the plugin's relative library layout for its libnuma and libdwarf dependencies.

The host needs Linux x86-64, the Tenstorrent driver, and configured hugepages.
The backend currently compiles single-device programs. ZML selects the first
addressable device by default; `CreateOptions.tenstorrent.device_index` selects
another device. Disable the CPU platform so a failed accelerator initialization
cannot fall back to CPU:

```sh
bazel run \
  --@zml//platforms:tenstorrent=true --@zml//platforms:cpu=false \
  //examples/mnist

bazel run \
  --@zml//platforms:tenstorrent=true --@zml//platforms:cpu=false \
  //examples/llm -- --model=/path/to/llama --prompt="Hello" \
  --backend=stablehlo --seqlen=64 --topk=1
```

Host transfers use buffered copies. Pinned host memory, external device pointers,
and GPU attention kernels are not used by this integration. Model execution is
subject to the operation and dtype coverage of the experimental XLA backend.

## Integration results

Tested with the native Bazel plugin on Blackhole hardware:

- `//examples/mnist` completed compilation, weight transfer, execution, and host
  readback with CPU disabled. The sampled image was recognized as digit 5.
- `//examples/llm` with `/var/models/meta-llama/Llama-3.1-8B-Instruct`, StableHLO
  attention, sequence length 64, and top-k 1 generated `Hello!` and stopped for
  `--prompt="Say hello in one short sentence."`, matching the CPU reference.
  Both prefill and decode execute on device 0, with the CPU platform disabled.
  The plugin includes the unsigned cache-index fix from XLA commit
  `1a3c9a68dc`; without it, decode corrupts the KV cache.
- With XLA commit `32711c00e6`, the counting prompt
  `Count from 1 to 20, separated by commas.` at sequence length 128 and top-k 1
  generated the correct sequence and stopped: 14.025 seconds of decode,
  5.2 tok/s, versus 179.331 seconds and 0.4 tok/s before optimization. This is
  a single-card measurement, including first-use work, with operation profiling
  disabled. TT-MLIR optimization level 1 is enabled; cache scatters preserve
  parallel rows instead of flattening the entire cache into one row.
  Repeating the run with allocation reuse enabled at XLA commit `647d6977c4`
  completed correctly in 14.036 seconds, also 5.2 tok/s.
- The plugin honors `reuseBuffer` donation: successful execution invalidates
  the input PJRT contents, and ZML destroys the old handle when installing the
  result. Non-donatable inputs remain readable. The plugin reuses matching,
  exclusively owned device allocations by copying completed graph results into
  them. It preserves separately live views. This still needs a temporary result
  and a final device copy; individual cache updates are not yet in place.
  HLO `must-alias` remains unsupported because reuse is not guaranteed.
- Sampling with top-k greater than 1 still requires unsupported `u64` RNG
  state. Greedy Llama omits that state.
- The CPU `//zml:test` suite passed: 262 passed, 14 skipped, 0 failed.

The focused execution suite also passes on Tenstorrent with CPU disabled: all
five existing execution tests, including raw donation, runner replacement with
and without waiting, unchanged donated outputs, and optional state. With the
plugin and runtime asset environment variables above exported, run:

```sh
bazel test --nocache_test_results \
  --@zml//platforms:tenstorrent=true --@zml//platforms:cpu=false \
  --test_env=HOME --test_env=ZML_TENSTORRENT_PLUGIN_PATH \
  --test_env=TT_METAL_RUNTIME_ROOT //zml:exe_test
```

The full core suite is not yet supported on Tenstorrent: it includes RNG and
pinned-host-memory paths that this backend does not implement.
