# Native XET download for ZML (via xet-core C-API)

Downloads HuggingFace XET-backed files by driving the mature **xet-core** (Rust)
client through its C-API, instead of reimplementing the XET protocol in Zig.

## Why the C-API and not a Zig reimplementation

XET-backed files reconstruct client-side (fetch compressed chunks from the CAS,
LZ4/BG4 decompress, reassemble). Getting that fast needs adaptive concurrency,
bounded streaming, decompress-once scheduling, connection pooling and an on-disk
chunk cache. xet-core already has all of it; a young Zig reimplementation does
not. Benchmark on an m6i.2xlarge (us-east-1), cold download of
`model-00001-of-00004.safetensors` (4.98 GB) from a Llama-3.1-8B repo:

| Path | Throughput |
|------|-----------|
| **this binding: Zig -> C-API -> xet-core, `HF_XET_HIGH_PERFORMANCE=1`** | **~1070 MB/s** |
| hf-xet (Python, same Rust core), high perf | ~1030 MB/s |
| ZML `parallel_read.zig` (plain parallel HTTPS, 32 workers) | ~1000 MB/s |
| xet-core default concurrency | ~640 MB/s |
| a from-scratch Zig XET reimplementation (jedisct1/zig-xet) | ~386 MB/s |

The C-API matches ZML's own HTTP downloader at line rate **and** adds BLAKE3
verification + the on-disk chunk cache (cross-file / cross-revision dedup),
which plain HTTP has no equivalent of. Both were network-bound at ~1 GB/s here.

## Files

- `xet_hub.zig`  : HF Hub HTTPS/JSON (pure Zig) — CAS read-token exchange and
  XET file listing (path -> hash + size).
- `xet_capi.zig` : Zig binding over `hf_xet.h` (`@cImport`). `Session` wraps a
  xet-core download group; `downloadToPath` reconstructs a whole file.
- `xet.zig`      : high-level `downloadFile(...)` tying the two together.
  Exposed as the `xet` namespace in `index.zig` (a helper module, not yet a
  `std.Io` VFS provider like `hf.zig` — see Limitations).

The C-API is whole-file oriented (`xet_file_download_group_download_to_path`),
so the integration model is **download-to-cache**: materialize a repo file
locally, then open it through the normal filesystem, exactly like
`huggingface_hub` + `hf-xet`. Set `HF_XET_HIGH_PERFORMANCE=1` for peak
concurrency/buffers.

## Verification status

Built with Zig 0.16.0 (this repo's pin) + `libxet_capi` built from the xet-core
`assaf/c-api` branch.

- `xet.downloadFile` ran end-to-end (Zig -> C-API -> xet-core) against
  `huggingface.co`, reconstructing full XET-backed files to disk (exact size
  match): a 1.16 GB file locally and the 4.98 GB file on the EC2 box above.
- On that box it measured **~1070 MB/s** in high-performance mode, matching
  `hf-xet` (same Rust core) and ZML's own HTTP downloader: the FFI overhead is
  within noise.

## Limitations / follow-ups

This is a proof of concept focused on showing xet-core-via-C-API reaches full
throughput from Zig. Known gaps, in rough priority order:

- **Not a VFS provider.** `downloadFile` is a standalone helper, not a `std.Io`
  vtable like `hf.zig`, so `hf://` XET files do not download transparently. The
  natural home is `hf.zig` `dirOpenFile`/`performRead`: when a tree entry has
  `xetHash`, materialize the file to a cache dir via the C-API, then serve
  reads from the local file. The real tension to decide: the C-API is
  whole-file (eager), while ZML's read path is lazy/positional.
- **Per-file session.** Each `downloadFile` mints a token and a fresh
  `Session`/download-group. The C-API's `FileDownloadGroup` is meant to batch
  many files through one session (shared cache/dedup); a repo-scoped handle
  that lists+tokens once and downloads N files would be the right shape.
- **Whole-tree metadata fetch.** `listXetFiles` GETs the full recursive tree to
  find one file's hash+size. A single `HEAD` on the resolve URL returning
  `X-Xet-Hash` + `X-Linked-Size` is cheaper (note: CloudFront strips
  `X-Xet-Hash` on cache hits unless an hf-xet `User-Agent` + cache-bust is sent).
- **Blocking poll + coarse errors.** `downloadToPath` busy-polls the op with a
  1 ms `nanosleep` (the C-API exposes no blocking wait), and `capiError`
  collapses every `XetStatus` into one error, so no auth-refresh/retry logic is
  possible yet.

## Building / wiring the C-API lib in Bazel

`vfs/BUILD.bazel` depends on `:xet_capi_import` (a `cc_import`). Produce the lib:

```
# in a checkout of xet-core (assaf/c-api branch)
cargo build --release -p xet_capi
#   -> xet_capi/include/hf_xet.h
#   -> target/release/libxet_capi.a   (also .dylib)
```

Then expose it as an external repo `@xet_capi` (prebuilt per platform, or via
rules_rust building the crate), providing `:hf_xet_h` and `:libxet_capi.a`.
Extra link deps the Rust staticlib needs:

- Linux: `-lpthread -ldl -lm`
- macOS: `-framework Security -framework CoreFoundation -framework SystemConfiguration`

Note: `libxet_capi.a` is large (pulls tokio/reqwest); cross-compile it for each
ZML target. This Rust build dependency is the one real cost of this approach
(the reason a pure-Zig reimplementation is otherwise attractive despite being
slower).
