import argparse
import subprocess
import sys
from pathlib import Path

import torch


MODES = (
    "attn1",
    "attn2",
    "audio_attn1",
    "audio_attn2",
    "audio_to_video_attn",
    "video_to_audio_attn",
)

ACTIVATION_KEYS = {
    "attn1": "velocity_model.transformer_blocks.0.attn1",
    "attn2": "velocity_model.transformer_blocks.0.attn2",
    "audio_attn1": "velocity_model.transformer_blocks.0.audio_attn1",
    "audio_attn2": "velocity_model.transformer_blocks.0.audio_attn2",
    "audio_to_video_attn": "velocity_model.transformer_blocks.0.audio_to_video_attn",
    "video_to_audio_attn": "velocity_model.transformer_blocks.0.video_to_audio_attn",
}

_ACTIVATION_CACHE: dict[Path, set[str]] = {}


def resolve_trace_pt(trace_arg: Path) -> Path:
    trace_arg = trace_arg.expanduser()

    if trace_arg.exists() and trace_arg.is_file():
        return trace_arg.resolve()

    search_dir = trace_arg if trace_arg.is_dir() else trace_arg.parent
    if not search_dir.exists() or not search_dir.is_dir():
        raise FileNotFoundError(f"Trace path not found: {trace_arg}")

    candidates = sorted(search_dir.glob("acts_stage2_transformer_step_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No replay traces matching acts_stage2_transformer_step_*.pt found in {search_dir}"
        )

    def candidate_rank(path: Path) -> tuple[int, str]:
        name = path.name
        if name.endswith("_diag.pt"):
            return (0, name)
        if "sdpa_diag" in name:
            return (1, name)
        if "minimal" in name:
            return (2, name)
        return (3, name)

    return min(candidates, key=candidate_rank).resolve()


def list_trace_candidates(trace_arg: Path) -> list[Path]:
    trace_arg = trace_arg.expanduser()

    candidates: list[Path] = []
    if trace_arg.exists() and trace_arg.is_file():
        candidates.append(trace_arg.resolve())

    search_dir = trace_arg if trace_arg.is_dir() else trace_arg.parent
    if search_dir.exists() and search_dir.is_dir():
        for candidate in sorted(search_dir.glob("acts_stage2_transformer_step_*.pt")):
            resolved = candidate.resolve()
            if resolved not in candidates:
                candidates.append(resolved)

    return candidates


def activation_keys_for_trace(trace_path: Path) -> set[str]:
    cached = _ACTIVATION_CACHE.get(trace_path)
    if cached is not None:
        return cached

    obj = torch.load(trace_path, map_location="cpu", weights_only=False)
    acts = obj.get("activations", {})
    keys = set(acts.keys()) if isinstance(acts, dict) else set()
    _ACTIVATION_CACHE[trace_path] = keys
    return keys


def trace_supports_mode(trace_path: Path, mode: str) -> bool:
    prefix = ACTIVATION_KEYS[mode]
    keys = activation_keys_for_trace(trace_path)
    if prefix in keys:
        return True
    return any(key == prefix or key.startswith(prefix + ".") for key in keys)


def resolve_trace_for_mode(trace_arg: Path, mode: str, token_limit: int | None = None) -> Path | None:
    candidates = list_trace_candidates(trace_arg)
    if not candidates:
        return None

    supported = [candidate for candidate in candidates if trace_supports_mode(candidate, mode)]
    if not supported:
        return None

    def rank(path: Path) -> tuple[int, str]:
        name = path.name
        token_hint = 0
        if token_limit is not None:
            token_hint = 0 if f"_t{token_limit}.pt" in name else 1
        mode_hint = 0 if mode in name else 1
        if name.endswith("_diag.pt"):
            kind = 0
        elif "sdpa_diag" in name:
            kind = 1
        elif "minimal" in name:
            kind = 2
        else:
            kind = 3
        return (token_hint, mode_hint, kind, name)

    return min(supported, key=rank)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fixtures and run attention parity checks for all AttentionKind modes"
    )
    parser.add_argument("checkpoint", type=Path, help="Stage-2 checkpoint safetensors")
    parser.add_argument("trace_pt", type=Path, help="Replay .pt file containing captured activations")
    parser.add_argument("output_dir", type=Path, help="Directory for generated fixtures and logs")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to zml repo root",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run export_attention_fixture.py",
    )
    parser.add_argument(
        "--bazel",
        default="bazel",
        help="Bazel executable",
    )
    parser.add_argument(
        "--checker-target",
        default="//examples/ltx:attention_forward_check",
        help="Bazel target for the parity checker",
    )
    parser.add_argument(
        "--export-script",
        type=Path,
        default=None,
        help=(
            "Path to export_attention_fixture.py. "
            "Default: sibling of this suite script, else <repo-root>/examples/ltx/export_attention_fixture.py"
        ),
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=None,
        help="Optional token limit passed to the checker",
    )
    parser.add_argument(
        "--diagnostic-dir",
        type=Path,
        default=None,
        help="Optional directory containing <mode>_reference.safetensors files",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run bazel with --@zml//platforms:cuda=true",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path, log_path: Path) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(proc.stdout)
    return proc.returncode, proc.stdout


def print_step(mode: str, step: str, path: Path) -> None:
    print(f"[{mode}] {step}: {path}")


def main() -> int:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_script = Path(__file__).resolve()
    sibling_export = suite_script.with_name("export_attention_fixture.py")
    repo_export = repo_root / "examples" / "ltx" / "export_attention_fixture.py"
    if args.export_script is not None:
        export_script = args.export_script.expanduser().resolve()
    elif sibling_export.exists():
        export_script = sibling_export
    else:
        export_script = repo_export

    if not export_script.exists():
        raise FileNotFoundError(
            "Could not resolve exporter script. "
            f"Checked: {export_script}. "
            "Set --export-script explicitly."
        )

    print(f"Using exporter script: {export_script}")

    results: list[tuple[str, bool, Path]] = []

    for mode in MODES:
        fixture_path = output_dir / f"{mode}_fixture.safetensors"
        export_log = output_dir / f"{mode}_export.log"
        check_log = output_dir / f"{mode}_check.log"

        trace_pt = resolve_trace_for_mode(args.trace_pt, mode, args.token_limit)
        if trace_pt is None:
            available = list_trace_candidates(args.trace_pt)
            available_text = ", ".join(str(path) for path in available) if available else "<none>"
            message = (
                f"No replay trace containing activations for mode={mode} under {args.trace_pt}. "
                f"Available trace files: {available_text}\n"
                f"Expected activation prefix: {ACTIVATION_KEYS[mode]}"
            )
            export_log.write_text(message)
            print(message)
            results.append((mode, False, check_log))
            continue

        print(f"Using trace file for {mode}: {trace_pt}")

        export_cmd = [
            args.python,
            str(export_script),
            str(trace_pt),
            str(fixture_path),
            "--mode",
            mode,
        ]
        if args.token_limit is not None:
            export_cmd.extend(["--token-limit", str(args.token_limit)])
        print_step(mode, "export", fixture_path)
        export_code, export_out = run_command(export_cmd, repo_root, export_log)
        if export_code != 0:
            print(export_out, end="")
            results.append((mode, False, check_log))
            continue

        checker_cmd = [args.bazel, "run"]
        if args.cuda:
            checker_cmd.append("--@zml//platforms:cuda=true")
        checker_cmd.extend(
            [
                args.checker_target,
                "--",
                str(args.checkpoint),
                str(fixture_path),
                mode,
            ]
        )
        if args.token_limit is not None:
            checker_cmd.append(str(args.token_limit))
            checker_cmd.append("--token-limited-reference")

        diag_ref = None
        if args.diagnostic_dir is not None:
            candidate = args.diagnostic_dir / f"{mode}_reference.safetensors"
            if candidate.exists():
                diag_ref = candidate

        if diag_ref is not None:
            checker_cmd.append(str(diag_ref))

        print_step(mode, "check", check_log)
        check_code, check_out = run_command(checker_cmd, repo_root, check_log)
        passed = check_code == 0 and f"attention parity PASSED for mode={mode}" in check_out
        if not passed:
            print(check_out, end="")
        results.append((mode, passed, check_log))

    print("\nAttentionKind suite summary")
    failures = 0
    for mode, passed, log_path in results:
        status = "PASS" if passed else "FAIL"
        print(f"- {mode}: {status} ({log_path})")
        if not passed:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
