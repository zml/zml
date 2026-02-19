#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jinja2  # pyright: ignore[reportMissingImports]
from transformers.utils.chat_template_utils import render_jinja_template  # pyright: ignore[reportMissingImports]


KNOWN_IGNORED_CASES: set[tuple[str, str]] = {
    # DeepSeek-R1 template concatenates tool.function.arguments as a raw string,
    # while our fixtures intentionally use OpenAI-like dict arguments.
    ("ag", "04"),
    ("ag", "05"),
    ("ag", "06"),
    ("ag", "13"),
    ("ag", "14"),
    # This template only supports a single tool call in one assistant turn.
    ("ae", "05"),
    # This template renders from/value instead of role/content, which breaks
    # continue_final_message compatibility.
    ("af", "10"),
}


def extract_id(path: Path) -> str:
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def render_basic_suite(input_root: Path, output_root: Path) -> int:
    basic_dir = input_root / "basic"
    output_basic_dir = output_root / "basic"
    output_basic_dir.mkdir(parents=True, exist_ok=True)
    env = jinja2.Environment()  # pyright: ignore[reportAttributeAccessIssue]
    rendered_count = 0

    for path in output_basic_dir.glob("*.txt"):
        path.unlink()

    for path in sorted(basic_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        template_src = payload["template"]
        data = payload.get("data", {})
        expected = env.from_string(template_src).render(**data)
        output_name = f"{path.stem}.txt"
        (output_basic_dir / output_name).write_text(expected, encoding="utf-8")
        rendered_count += 1

    return rendered_count


def validate_unique_prefix_ids(files: list[Path], label: str) -> None:
    ids = [extract_id(path) for path in files]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"Duplicate {label} ids found; ensure unique filename prefixes."
        )


def render_chat_like_suite(
    input_root: Path,
    output_root: Path,
    suite: str,
    render_jinja_template,
) -> tuple[int, list[tuple[str, str, str, str]]]:
    suite_dir = input_root / suite
    data_dir = suite_dir / "data"
    templates_dir = suite_dir / "templates"
    expected_dir = output_root / suite

    if not data_dir.exists() or not templates_dir.exists():
        return 0, []

    expected_dir.mkdir(parents=True, exist_ok=True)
    for path in expected_dir.glob("*.txt"):
        path.unlink()

    data_files = sorted(data_dir.glob("*.json"))
    template_files = sorted(templates_dir.glob("*.jinja"))
    validate_unique_prefix_ids(data_files, f"{suite}/data")
    validate_unique_prefix_ids(template_files, f"{suite}/templates")

    rendered_count = 0
    failures: list[tuple[str, str, str, str]] = []

    for template_path in template_files:
        template_id = extract_id(template_path)
        chat_template = template_path.read_text(encoding="utf-8")

        for data_path in data_files:
            data_id = extract_id(data_path)
            payload = json.loads(data_path.read_text(encoding="utf-8"))

            messages = payload.get("messages", [])
            tools = payload.get("tools")
            documents = payload.get("documents")
            add_generation_prompt = bool(payload.get("add_generation_prompt", False))
            continue_final_message = bool(payload.get("continue_final_message", False))

            kwargs = dict(payload)
            for key in (
                "messages",
                "tools",
                "documents",
                "add_generation_prompt",
                "continue_final_message",
            ):
                kwargs.pop(key, None)

            output_path = expected_dir / f"{template_id}_{data_id}.txt"
            try:
                rendered, _ = render_jinja_template(
                    conversations=[messages],
                    tools=tools,
                    documents=documents,
                    chat_template=chat_template,
                    continue_final_message=continue_final_message,
                    add_generation_prompt=add_generation_prompt,
                    **kwargs,
                )
                output_path.write_text(rendered[0], encoding="utf-8")
                rendered_count += 1
            except Exception as exc:  # noqa: BLE001
                if (template_id, data_id) in KNOWN_IGNORED_CASES:
                    continue
                failures.append(
                    (
                        f"{suite}/{template_path.name}",
                        f"{suite}/{data_path.name}",
                        type(exc).__name__,
                        str(exc),
                    )
                )

    return rendered_count, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--skip-basic",
        action="store_true",
    )
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root if args.output_root is not None else input_root

    basic_rendered = 0
    if not args.skip_basic:
        basic_rendered = render_basic_suite(input_root, output_root)

    chat_rendered, chat_failures = render_chat_like_suite(
        input_root,
        output_root,
        "chat",
        render_jinja_template,
    )
    multimodal_rendered, multimodal_failures = render_chat_like_suite(
        input_root,
        output_root,
        "multimodal",
        render_jinja_template,
    )

    failures = [*chat_failures, *multimodal_failures]

    print(f"basic_rendered={basic_rendered}")
    print(f"chat_rendered={chat_rendered}")
    print(f"multimodal_rendered={multimodal_rendered}")
    print(f"rendered={basic_rendered + chat_rendered + multimodal_rendered}")
    print(f"errors={len(failures)}")

    if failures:
        print("failure_report_start")
        for template_name, data_name, exc_type, message in failures:
            print(
                f"- template={template_name} data={data_name} error={exc_type}: {message}"
            )
        print("failure_report_end")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
