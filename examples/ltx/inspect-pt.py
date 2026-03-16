import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect .pt traces with compact activation summaries")
    parser.add_argument(
        "input_path",
        nargs="?",
        default="trace_run",
        help=".pt file or directory of .pt files (default: trace_run)",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=15,
        help="Max number of keys to print for dict/activation summaries",
    )
    return parser.parse_args()


def bytes_human(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            return f"{x:.2f}{unit}"
        x /= 1024.0
    return f"{n}B"


def tensor_nbytes(t: torch.Tensor) -> int:
    return t.nelement() * t.element_size()


def estimate_nbytes(obj) -> int:
    if torch.is_tensor(obj):
        return tensor_nbytes(obj)
    if isinstance(obj, dict):
        return sum(estimate_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(estimate_nbytes(v) for v in obj)
    return 0


def describe_value(v) -> str:
    if torch.is_tensor(v):
        return f"tensor shape={tuple(v.shape)} dtype={v.dtype} bytes={bytes_human(tensor_nbytes(v))}"
    if isinstance(v, dict):
        return f"dict len={len(v)} est_bytes={bytes_human(estimate_nbytes(v))}"
    if isinstance(v, list):
        return f"list len={len(v)} est_bytes={bytes_human(estimate_nbytes(v))}"
    if isinstance(v, tuple):
        return f"tuple len={len(v)} est_bytes={bytes_human(estimate_nbytes(v))}"
    return f"{type(v)}"


def print_activation_summary(acts, max_keys: int) -> None:
    if not isinstance(acts, dict):
        print(f"activations: {describe_value(acts)}")
        return

    keys = list(acts.keys())
    print(f"activations: key_count={len(keys)} est_bytes={bytes_human(estimate_nbytes(acts))}")

    for k in keys[:max_keys]:
        v = acts[k]
        if isinstance(v, tuple) and len(v) == 3:
            # ActivationCollector stores (name, inputs_or_none, output_or_none).
            _, inp, out = v
            print(f"  {k}")
            print(f"    input:  {describe_value(inp)}")
            print(f"    output: {describe_value(out)}")
        elif isinstance(v, dict) and ("input" in v or "output" in v):
            print(f"  {k}")
            print(f"    input:  {describe_value(v.get('input'))}")
            print(f"    output: {describe_value(v.get('output'))}")
        else:
            print(f"  {k}: {describe_value(v)}")

    if len(keys) > max_keys:
        print(f"  ... (+{len(keys) - max_keys} more keys)")


def inspect_file(path: Path, max_keys: int) -> None:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    print(f"\n== {path} ==")

    if torch.is_tensor(obj):
        print(describe_value(obj))
        return

    if isinstance(obj, dict):
        top_keys = list(obj.keys())
        print(f"dict keys={len(top_keys)} est_bytes={bytes_human(estimate_nbytes(obj))}")
        for k in top_keys[:max_keys]:
            print(f"  {k}: {describe_value(obj[k])}")
        if len(top_keys) > max_keys:
            print(f"  ... (+{len(top_keys) - max_keys} more keys)")

        if "activations" in obj:
            print_activation_summary(obj["activations"], max_keys)
        return

    if isinstance(obj, list):
        print(f"list len={len(obj)} est_bytes={bytes_human(estimate_nbytes(obj))}")
        if obj:
            print(f"first: {describe_value(obj[0])}")
        return

    print(type(obj))


def iter_pt_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob("*.pt"))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    files = iter_pt_files(input_path)

    if not files:
        print(f"No .pt files found at: {input_path}")
        return

    for path in files:
        inspect_file(path, max_keys=args.max_keys)


if __name__ == "__main__":
    main()
