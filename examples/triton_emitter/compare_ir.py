"""Diff two IR files (or directories) after stripping debug locations.

Both `loc(...)` markers and `{operandSegmentSizes = ...}` attributes are
stripped, runs of `arith.constant` lines are sorted by RHS, and SSA names
are renumbered to a stable `%v<N>` form so trivial naming differences
between Python's frontend and the Zig DSL don't show up as diffs.

Default mode (no `--stage`) compares all four stages (ttir/ttgir/llir/ptx)
and prints a per-kernel summary grouped by stage. Pass `--show-diffs` to
print the full unified-diff bodies for non-matching cells.
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_LOC_INLINE = re.compile(r"\s*loc\(")
_LOC_DEF_LINE = re.compile(r"^#loc\w*\s*=\s*loc\(")
_OPERAND_SEG_RE = re.compile(
    r"\s*\{\s*operandSegmentSizes\s*=\s*dense<[^>]*>\s*:\s*tensor<[^>]*>\s*\}"
)
_SSA_NAME_RE = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*|\d+)\b")
_CONST_RE = re.compile(r"^(\s*)%[A-Za-z_0-9.\-]+\s*=\s*arith\.constant\s+(.*)$")

_DEFAULT_STAGES = ("ttir", "ttgir", "llir", "ptx")


# --- color helpers -----------------------------------------------------------

def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


_USE_COLOR = _supports_color()
_C = {
    "reset": "\033[0m" if _USE_COLOR else "",
    "bold": "\033[1m" if _USE_COLOR else "",
    "dim": "\033[2m" if _USE_COLOR else "",
    "green": "\033[32m" if _USE_COLOR else "",
    "red": "\033[31m" if _USE_COLOR else "",
    "yellow": "\033[33m" if _USE_COLOR else "",
    "cyan": "\033[36m" if _USE_COLOR else "",
    "gray": "\033[90m" if _USE_COLOR else "",
}


def _c(s: str, *colors: str) -> str:
    return "".join(_C[c] for c in colors) + s + _C["reset"]


# --- normalization -----------------------------------------------------------

def _balanced_paren_end(s: str, start: int) -> int:
    depth, i = 0, start
    while i < len(s):
        c = s[i]
        if c == "(": depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0: return i + 1
        elif c in ('"', "'"):
            quote = c
            i += 1
            while i < len(s) and s[i] != quote:
                if s[i] == "\\": i += 2; continue
                i += 1
        i += 1
    return len(s)


def _strip_inline_locs(line: str) -> str:
    out, i = [], 0
    while i < len(line):
        m = _LOC_INLINE.search(line, i)
        if m is None:
            out.append(line[i:])
            break
        out.append(line[i:m.start()])
        open_paren = line.index("(", m.end() - 1)
        i = _balanced_paren_end(line, open_paren)
    return "".join(out)


def _scoped_def_names(line: str) -> List[str]:
    """Return SSA names introduced as scoped block-args by `line`."""
    names: List[str] = []
    m = re.match(r"\s*\^bb\d+\((.*)\)\s*:", line)
    if m:
        for piece in m.group(1).split(","):
            mm = re.match(r"\s*%([A-Za-z_][A-Za-z0-9_]*|\d+)\b", piece)
            if mm: names.append(mm.group(1))
        return names
    m = re.search(r"\bscf\.for\s+%([A-Za-z_][A-Za-z0-9_]*|\d+)\b", line)
    if m:
        names.append(m.group(1))
        ia = re.search(r"iter_args\(([^)]*)\)", line)
        if ia:
            for piece in ia.group(1).split(","):
                mm = re.match(r"\s*%([A-Za-z_][A-Za-z0-9_]*|\d+)\b", piece)
                if mm: names.append(mm.group(1))
    for m in re.finditer(r"\bscf\.while\s*\(([^)]*)\)", line):
        for piece in m.group(1).split(","):
            mm = re.match(r"\s*%([A-Za-z_][A-Za-z0-9_]*|\d+)\b", piece)
            if mm: names.append(mm.group(1))
    return names


def _renumber_ssa(lines: List[str]) -> List[str]:
    counter = [0]
    stack: List[Dict[str, int]] = [dict()]

    def fresh() -> int:
        n = counter[0]; counter[0] += 1; return n

    def lookup(name: str) -> int:
        for scope in reversed(stack):
            if name in scope: return scope[name]
        idx = fresh()
        stack[0][name] = idx
        return idx

    def repl(m: re.Match) -> str:
        return f"%v{lookup(m.group(1))}"

    out: List[str] = []
    for raw in lines:
        defs = _scoped_def_names(raw)
        opens, closes = raw.count("{"), raw.count("}")
        if defs:
            stack.append({n: fresh() for n in defs})
        out.append(_SSA_NAME_RE.sub(repl, raw))
        net = opens - closes
        if defs: net -= 1
        if net > 0:
            for _ in range(net): stack.append({})
        elif net < 0:
            for _ in range(-net):
                if len(stack) > 1: stack.pop()
    return out


def _sort_constants(lines: List[str]) -> List[str]:
    out, i = [], 0
    while i < len(lines):
        m = _CONST_RE.match(lines[i])
        if m is None:
            out.append(lines[i]); i += 1
            continue
        keys: List[Tuple[str, str]] = []
        while i < len(lines):
            mm = _CONST_RE.match(lines[i])
            if mm is None: break
            keys.append((mm.group(2), lines[i])); i += 1
        keys.sort(key=lambda kv: kv[0])
        out.extend(line for _, line in keys)
    return out


def normalize(text: str, *, mlir: bool = True) -> List[str]:
    """Strip locations, sort constants, renumber SSA names. For LLIR/PTX
    (`mlir=False`), only renumber SSA-like names."""
    if not mlir:
        # LLIR / PTX: just normalize register / SSA-like names. Both formats
        # use `%N` register numbering that varies with op-emit order.
        out = [ln.rstrip() for ln in text.splitlines()]
        return _renumber_ssa(out)

    out: List[str] = []
    for raw in text.splitlines():
        if _LOC_DEF_LINE.match(raw):
            continue
        cleaned = _strip_inline_locs(raw)
        cleaned = cleaned.replace(" attributes {noinline = false}", "")
        cleaned = _OPERAND_SEG_RE.sub("", cleaned)
        cleaned = cleaned.rstrip()
        if cleaned == "" and out and out[-1] == "":
            continue
        out.append(cleaned)
    while out and out[-1] == "":
        out.pop()
    return _renumber_ssa(_sort_constants(out))


# --- comparison primitives ---------------------------------------------------

def _diff_lines(left_lines, right_lines, lname, rname) -> List[str]:
    return list(difflib.unified_diff(left_lines, right_lines,
                                     fromfile=lname, tofile=rname, lineterm=""))


def _count_changes(diff: List[str]) -> Tuple[int, int]:
    plus = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
    minus = sum(1 for ln in diff if ln.startswith("-") and not ln.startswith("---"))
    return plus, minus


def _is_mlir_stage(stage: str) -> bool:
    return stage in ("ttir", "ttgir")


def _format_diff(diff: List[str]) -> str:
    out = []
    for ln in diff:
        if ln.startswith("+++") or ln.startswith("---"):
            out.append(_c(ln, "bold"))
        elif ln.startswith("@@"):
            out.append(_c(ln, "cyan"))
        elif ln.startswith("+"):
            out.append(_c(ln, "green"))
        elif ln.startswith("-"):
            out.append(_c(ln, "red"))
        else:
            out.append(ln)
    return "\n".join(out)


# --- single-pair mode (file vs file) -----------------------------------------

def _file_pair(left: Path, right: Path) -> int:
    mlir = _is_mlir_stage(left.suffix.lstrip("."))
    diff = _diff_lines(normalize(left.read_text(), mlir=mlir),
                       normalize(right.read_text(), mlir=mlir),
                       str(left), str(right))
    if diff:
        sys.stdout.write(_format_diff(diff) + "\n")
    return len(diff)


# --- summary mode (cross-stage table) ----------------------------------------

class Cell:
    __slots__ = ("status", "added", "removed", "diff", "left_path", "right_path")

    def __init__(self, status: str, added: int = 0, removed: int = 0,
                 diff: List[str] | None = None, left_path: Path | None = None,
                 right_path: Path | None = None):
        self.status = status   # "match" | "diff" | "missing-left" | "missing-right" | "absent"
        self.added = added
        self.removed = removed
        self.diff = diff or []
        self.left_path = left_path
        self.right_path = right_path

    def label(self) -> str:
        if self.status == "match":
            return _c("✓", "green")
        if self.status == "diff":
            return _c(f"+{self.added}/-{self.removed}", "red")
        if self.status == "missing-left":
            return _c("missing-left", "yellow")
        if self.status == "missing-right":
            return _c("missing-right", "yellow")
        return _c("—", "gray")


def _compare_one(left: Path, right: Path, stage: str) -> Cell:
    if not left.exists() and not right.exists():
        return Cell("absent", left_path=left, right_path=right)
    if not left.exists():
        return Cell("missing-left", left_path=left, right_path=right)
    if not right.exists():
        return Cell("missing-right", left_path=left, right_path=right)
    mlir = _is_mlir_stage(stage)
    diff = _diff_lines(normalize(left.read_text(), mlir=mlir),
                       normalize(right.read_text(), mlir=mlir),
                       str(left), str(right))
    if not diff:
        return Cell("match", left_path=left, right_path=right)
    plus, minus = _count_changes(diff)
    return Cell("diff", added=plus, removed=minus, diff=diff,
                left_path=left, right_path=right)


def _all_kernels(left_dir: Path, right_dir: Path, stages: List[str]) -> List[str]:
    names = set()
    for d in (left_dir, right_dir):
        for stage in stages:
            for f in d.glob(f"*.{stage}"):
                names.add(f.stem)
    return sorted(names)


def _summary(left_dir: Path, right_dir: Path, stages: List[str],
             kernel_filter: str, mapping: Dict[str, str],
             show_diffs: bool) -> int:
    kernels = _all_kernels(left_dir, right_dir, stages)
    if kernel_filter:
        kernels = [k for k in kernels if k == kernel_filter]
    if not kernels:
        print(_c("compare: no kernels found", "yellow"), file=sys.stderr)
        return 1

    # Build the table.
    rows: Dict[str, Dict[str, Cell]] = {}
    for k in kernels:
        right_name = mapping.get(k, k)
        rows[k] = {
            stage: _compare_one(left_dir / f"{k}.{stage}",
                                right_dir / f"{right_name}.{stage}", stage)
            for stage in stages
        }

    # Print per-kernel grouped output.
    name_w = max(len(k) for k in kernels)
    stage_w = max(8, max(len(s) for s in stages))
    n_match = n_diff = n_missing = n_codegen = 0
    codegen_kernels: List[str] = []
    # "codegen-equivalent" = ttir/ttgir differ but llir+ptx match. We surface
    # these in yellow because the final compiled artifact is identical even
    # though the high-level IR isn't textually the same.
    _CODEGEN_STAGES = ("llir", "ptx")
    _SOURCE_STAGES = ("ttir", "ttgir")
    for k in kernels:
        cells = rows[k]
        # Worst-status drives the kernel-line color.
        statuses = [cells[s].status for s in stages]
        if all(s == "match" for s in statuses):
            header_color = "green"
            n_match += 1
        elif any(s in ("missing-left", "missing-right") for s in statuses):
            header_color = "yellow"
            n_missing += 1
        else:
            codegen_present = [s for s in _CODEGEN_STAGES if s in stages]
            source_present = [s for s in _SOURCE_STAGES if s in stages]
            codegen_match = (
                bool(codegen_present)
                and all(cells[s].status == "match" for s in codegen_present)
            )
            source_diff = any(
                cells[s].status == "diff" for s in source_present
            )
            if codegen_match and source_diff:
                header_color = "yellow"
                n_codegen += 1
                codegen_kernels.append(k)
            else:
                header_color = "red"
                n_diff += 1
        print(_c(f"{k:<{name_w}}", "bold", header_color))
        for stage in stages:
            cell = cells[stage]
            print(f"  {_c(stage.ljust(stage_w), 'dim')}  {cell.label()}")
        print()

    # Summary line.
    total = len(kernels)
    parts = []
    if n_match: parts.append(_c(f"{n_match} matching", "green"))
    if n_codegen: parts.append(_c(f"{n_codegen} codegen-equivalent", "yellow"))
    if n_diff: parts.append(_c(f"{n_diff} differing", "red"))
    if n_missing: parts.append(_c(f"{n_missing} missing", "yellow"))
    print(_c("─" * 60, "gray"))
    print(f"{total} kernel(s): " + ", ".join(parts))

    # Diff bodies, if requested.
    if show_diffs:
        for k in kernels:
            for stage in stages:
                cell = rows[k][stage]
                if cell.status != "diff":
                    continue
                print()
                print(_c(f"=== {k} ({stage}) — +{cell.added}/-{cell.removed} ===", "bold", "red"))
                print(_format_diff(cell.diff))

    return 0


# --- entry point -------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="*", help="Two file paths to diff.")
    p.add_argument("--left", help="Left directory (typically xla_py).")
    p.add_argument("--right", help="Right directory (typically xla_zig).")
    p.add_argument("--stage", default=None,
                   help="Single stage to diff (default: all four).")
    p.add_argument("--map", action="append", default=[],
                   help="Per-kernel rename: --map left_name=right_name.")
    p.add_argument("--kernel", default="", help="Only compare this kernel.")
    p.add_argument("--show-diffs", action="store_true",
                   help="Print the full unified-diff body for every non-match.")
    args = p.parse_args(argv)

    if len(args.paths) == 2:
        _file_pair(Path(args.paths[0]), Path(args.paths[1]))
        return 0
    if args.left and args.right:
        mapping = dict(item.split("=", 1) for item in args.map)
        stages = [args.stage] if args.stage else list(_DEFAULT_STAGES)
        return _summary(Path(args.left), Path(args.right), stages,
                        args.kernel, mapping, args.show_diffs)
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
