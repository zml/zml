# Safetensor explorer

Browse tensor metadata without loading model weights, using the same Vaxis TUI
library as zml-smi:

```sh
bazel run //examples/io:safetensor-explorer -- --model=hf://Qwen/Qwen3-0.6B
```

The built executable accepts:

```sh
./safetensor-explorer --model=<MODEL_URI>
```

Like the LLM example, `--model` accepts model repositories through the VFS:
local paths, `file://`, `hf://` (including revisions), `s3://`, and `gs://`.
HTTP and HTTPS are also supported. Existing VFS credentials and proxy settings
apply. A repository resolves `model.safetensors.index.json` first, then
`model.safetensors`. You can also pass a safetensors file or shard index directly.
Relative paths under `bazel run` resolve from the directory where you ran Bazel.

| Key | Action |
| --- | --- |
| Up / Down | Select a visible row, wrapping at either end |
| Enter | Show tensor details; toggle a parent without a tensor |
| Left | Collapse a parent, or select the containing parent |
| Right | Expand a parent, or select its first child |
| Space | Toggle the selected parent |
| `/` | Edit the search query |
| Backspace / Ctrl-U | Delete a character / clear the query while editing |
| Esc | Leave search and clear the filter |
| Page Up / Page Down | Scroll the details panel |
| q / Ctrl-C | Quit (Ctrl-C also works while editing search) |

Search is a case-insensitive substring match on the full tensor name. Matching
ancestors are shown and expanded; clearing the query restores the original fold
state. Up/Down works while searching, and Enter leaves search and activates the
selected row. The right panel stays on the tensor opened with Enter as you browse.

Details include the full name, shape and dtype, byte size, source file URI, and
absolute file byte offset (including the safetensors header). The displayed end
offset is exclusive. Long names and URIs wrap in the details panel.

Run the focused checks with:

```sh
bazel build //examples/io:safetensor-explorer
bazel test //examples/io:tensor_tree_test //examples/io:safetensor_explorer_test
```
