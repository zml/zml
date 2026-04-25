"""Slice XLA's per-pass IR dumps + opt LLIR/PTX into per-stage files.

XLA writes each `dump_via_xla` compile to several files:

    <program>.<kernel>.triton-to-llvm.txt        (one MLIR snapshot per pass)
    module_NNNN.<kernel>.0.ir-with-opt.ll        (post-opt LLVM IR for kernel)
    module_NNNN.<kernel>.0.ptx                   (PTX for kernel)

We pick:
  - last snapshot before `ConvertTritonToTritonGPU` → `<kernel>.ttir`
  - last snapshot before `ConvertTritonGPUToLLVM`   → `<kernel>.ttgir`
  - the `.0.ir-with-opt.ll` file                    → `<kernel>.llir`
  - the `.0.ptx` file                               → `<kernel>.ptx`

XLA's IR-printer hardcodes generic op-form + debug info, so each MLIR snapshot
is re-parsed through Triton's parser to convert to the assembly form
`compare_ir.py` understands. LLIR and PTX are copied verbatim.
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

_HEADER_RE = re.compile(r"^// -----// IR Dump Before (?P<pass>\w+)")

# Pretty-debug-info that XLA's `enableDebugInfo(/*pretty=*/true)` adds and that
# Triton's parser doesn't accept. We strip them all and let the round-trip
# through `module.str()` re-attach fresh `loc(...)` markers.
_ARG_NAME = re.compile(r'(\: [^",\)\n]+) "[^"]*"')
_ARG_INLINE_LOC = re.compile(r"(?:\s+(?:-:\d+:\d+|\[unknown\])|\(#loc\w+\))(?=[,\)])")
_TRAILING_LOC = re.compile(r'\s+"[\w\.\-/]+"(?:\([^()]*(?:\([^()]*\)[^()]*)*\))?\s*$', re.M)
_LOC_DEF_LINE = re.compile(r"^#loc[\w]*\s*=.*$\n?", flags=re.M)


def _strip_inline_locs(s: str) -> str:
    """Remove every `loc(...)` call with balanced parens, skipping string literals."""
    out, i, n = [], 0, len(s)
    while i < n:
        if s.startswith("loc(", i):
            depth, j = 1, i + 4
            while j < n and depth > 0:
                c = s[j]
                if c == "(": depth += 1
                elif c == ")": depth -= 1
                elif c == '"':
                    j += 1
                    while j < n and s[j] != '"':
                        if s[j] == "\\": j += 1
                        j += 1
                j += 1
            i = j
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _normalize(snapshot: str) -> str:
    s = _ARG_NAME.sub(r"\1", snapshot)
    s = _ARG_INLINE_LOC.sub("", s)
    s = s.replace(" [unknown]", "")
    s = re.sub(r"\s+-:\d+:\d+(?=$|\s)", "", s, flags=re.M)
    while True:
        new_s = _TRAILING_LOC.sub("", s)
        if new_s == s: break
        s = new_s
    s = _LOC_DEF_LINE.sub("", s)
    s = _strip_inline_locs(s)

    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as f:
        f.write(s)
        tmp = f.name
    try:
        from triton._C.libtriton import ir
        ctx = ir.context()
        ir.load_dialects(ctx)
        return ir.parse_mlir_module(tmp, ctx).str()
    finally:
        Path(tmp).unlink(missing_ok=True)


def _split_snapshots(text: str) -> list[tuple[str, str]]:
    out, current_pass, current_lines = [], None, []
    for line in text.splitlines(keepends=True):
        m = _HEADER_RE.match(line)
        if m:
            if current_pass is not None:
                out.append((current_pass, "".join(current_lines)))
            current_pass = m.group("pass")
            current_lines = []
        elif current_pass is not None:
            current_lines.append(line)
    if current_pass is not None:
        out.append((current_pass, "".join(current_lines)))
    return out


def _last_before(snapshots, pass_name):
    matches = [body for name, body in snapshots if name == pass_name]
    return matches[-1] if matches else None


_LL_RE = re.compile(r"^module_\d+\.(?P<kernel>.+)\.0\.ir-with-opt\.ll$")
_PTX_RE = re.compile(r"^module_\d+\.(?P<kernel>.+)\.0\.ptx$")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)

    in_dir = Path(args.in_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0

    # MLIR per-pass snapshots → .ttir / .ttgir
    for dump in sorted(in_dir.glob("*.triton-to-llvm.txt")):
        stem = dump.name[: -len(".triton-to-llvm.txt")]
        if "." not in stem:
            continue
        _, kernel = stem.split(".", 1)
        snapshots = _split_snapshots(dump.read_text())
        for pass_name, ext in (("ConvertTritonToTritonGPU", "ttir"),
                                ("ConvertTritonGPUToLLVM", "ttgir")):
            body = _last_before(snapshots, pass_name)
            if body is None:
                print(f"extract_xla_dump: no {ext} snapshot for {kernel}", file=sys.stderr)
                continue
            (out_dir / f"{kernel}.{ext}").write_text(_normalize(body))
            written += 1

    # Copy LLIR / PTX verbatim from XLA's per-module dump files.
    for f in sorted(in_dir.iterdir()):
        for rx, ext in ((_LL_RE, "llir"), (_PTX_RE, "ptx")):
            m = rx.match(f.name)
            if m:
                (out_dir / f"{m.group('kernel')}.{ext}").write_text(f.read_text())
                written += 1
                break

    print(f"extract_xla_dump: wrote {written} files under {out_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
