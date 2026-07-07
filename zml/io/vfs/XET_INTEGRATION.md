# Native XET download for ZML

Adds a VFS provider that reads HuggingFace files through the XET content-addressed
storage (CAS) protocol, instead of the `resolve/` byte-range path in `hf.zig`.

## Why

For XET-backed repos, `hf.zig` downloads the fully reconstructed object through
HF's `resolve/` compatibility bridge (server-side reconstruction, no client-side
dedup, re-fetched every run). Talking to the CAS directly enables chunk-level
deduplication and, with a shared chunk cache, reuse of overlapping content across
files and revisions (e.g. iterating on quantizations of the same base model).

## Files

- `xet_core.zig` - token exchange (`requestReadToken`), `Session` (CAS client +
  reconstructor) with `readRange` / `downloadToWriter`, and `listXetFiles`.
  Delegates the protocol (chunk reconstruction, LZ4/BG4, BLAKE3) to the
  `jedisct1/zig-xet` package.
- `xet.zig` - repo-scoped VFS provider (`Xet`), mirrors `http.zig` handle
  bookkeeping; `fileReadPositional` reconstructs only the requested byte range.
- Registered in `index.zig` (`Xet`, `xet_core`).

Everything is driven by `std.Io`, matching ZML's async substrate, so the parallel
fetcher shares the same concurrency model as the rest of `zml/io`.

## Verification status

Built with Zig 0.16.0 (the version this repo pins).

- `xet_core.zig`: compiled AND run end-to-end against `huggingface.co` -
  listed XET files, exchanged the token, and reconstructed a 1 MiB range of a
  4.68 GB XET-backed GGUF (magic bytes `GGUF` confirmed) in ~1.3 s.
- `xet.zig`: compiled against the real `base.zig` + `stdx`, so every VFS vtable
  override signature is validated against `std.Io.VTable`.

## Remaining: Bazel dependency wiring

`vfs/BUILD.bazel` now depends on `@xet//:xet`. That external repo still has to be
declared, matching how other vendored Zig deps (e.g. `libvaxis`) are wired via the
`non_module_deps` extension in `MODULE.bazel`:

- Declare `zig-xet` and its two zon deps as repos:
  - `xet`: https://github.com/jedisct1/zig-xet (release 0.2.5), import name `xet`
  - `lz4`: https://github.com/jedisct1/zig-lz4/archive/refs/tags/0.1.4.tar.gz
    (zon hash `lz4-0.1.4-fUqx0T2RAgCcxwjGQ-TAq2XC8ENcyrAWsQjgjcZfe75d`)
  - `ultracdc`: https://github.com/jedisct1/zig-ultracdc/archive/refs/tags/0.1.6.tar.gz
    (zon hash `ultracdc-0.1.6-vd07qWxWAAD8eep8mIHR-7p7IPeEdLhIQnoiwyHtKzsV`)
- Add their content hashes to `bazel/zig_index.json`.
- `use_repo(non_module_deps, ..., "xet")`.

Note: `zig-xet` targets `minimum_zig_version = 0.16.0-dev.2903`; its own
`build.zig` (example/test run steps) uses `run_cmd.addPassthruArgs()`, removed in
the 0.16.0 release. Building only the `xet` library module (as ZML does) does not
hit that path, but the upstream `build.zig` should be updated for 0.16.0 if its
examples are built.
