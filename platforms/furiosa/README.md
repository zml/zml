# Furiosa RNGD

Build with `--@zml//platforms:furiosa=true` and set
`XLA_FURIOSA_PJRT_LIBRARY` to the absolute path of the RNGD PJRT shared library.
Install the driver and native runtime/compiler required by that plugin. The loader
uses the existing PJRT C API and does not require Python bindings.

The current integration targets a single card and selects `.vanilla` attention
by default. The client uses all eight PEs; set `CreateOptions.furiosa.pe_count`
to 4 to use a four-PE client. Host-pinned memory and paged attention are not enabled. Device buffers
use dense row-major layouts. Full-model validation is in progress.

For the experimental XLA plugin in the companion checkout:

```sh
source /home/steeve/.local/share/xla-rngd-sdk/env.sh
export XLA_FURIOSA_PJRT_LIBRARY=/home/steeve/xla-private/bazel-bin/xla/pjrt/furiosa/libpjrt_c_api_furiosa_plugin.so
bazel run //examples/llm --@zml//platforms:furiosa=true -- \
  --model=/var/models/meta-llama/Llama-3.1-8B-Instruct \
  --backend=vanilla --seqlen=128 --topk=1 --prompt='What is the capital of France?'
```

Use the repository's Bazel version (9.1.1). `//examples/llm:llama_attention_tests`
compares the exact vanilla attention path against CPU for prefill, offset prefill,
and decode with grouped-query heads in F32/BF16. It requires both CPU and Furiosa
platforms enabled. All input data is deterministic and generated on the host;
only the CPU reference and final comparison run on the host.

Validation on 2026-09-24: `bazel test //... --jobs=16 --config=debug` passes all
24 targets on CPU, and the LLM/attention binaries build with Furiosa enabled.
The plugin loads and reports an eight-PE device. With the companion XLA fixes,
all six vanilla attention comparisons pass on eight PEs (F32/BF16, prefill,
offset prefill, decode). `--dtype=bf16` or `--dtype=f32` can select one dtype.
Full 8B execution remains in progress; the prefill layer currently exposes a
large scalar-broadcast tactic failure.
