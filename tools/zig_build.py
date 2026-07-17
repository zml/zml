#!/usr/bin/env python3
"""
Use Bazel's Zig to run `zig build ...`
Bazel remains responsible for building LLVM, XLA, generated sources, and the native libraries.
Arguments after the first `--` are passed to `bazel` itself, including config/platform.
Arguments after a second `--` are passed to `zig build`.

Example:
```sh
bazel run //tools:zig_build -- //examples/llm:llm --config=debug --@zml//platforms:cuda=true

bazel run //tools:zig_build -- //examples/mnist:mnist --config=debug -- --watch -fincremental
```

This allows to access the features exclusive to `zig build` like incremental recompilation, or the webui.

```sh
bazel run //tools:zig_build -- \
    //examples/mnist:mnist --config=debug -- \
    -fincremental --webui=0.0.0.0:34001 --time-report
```

To leverage incremental compilation you need to pass `--config=debug` (implying `--strategy=ZigBuildLib=local`)
to Bazel so that Bazel uses the files from your local checkout instead of files in the sandbox.
This will give you feedback based on your latest changes to your files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def bazel(workspace: Path, arguments: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        ["bazel", *arguments],
        cwd=workspace,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def emitted_path(arguments: list[str]) -> str | None:
    prefix = "-femit-bin="
    return next((arg[len(prefix) :] for arg in arguments if arg.startswith(prefix)), None)


def parse_modules(arguments: list[str]) -> list[dict[str, object]]:
    modules: list[dict[str, object]] = []
    dependencies: list[dict[str, str]] = []
    index = 0
    while index < len(arguments):
        arg = arguments[index]
        if arg == "--dep":
            index += 1
            mapping = arguments[index]
            name, separator, module = mapping.partition("=")
            dependencies.append({"name": name, "module": module if separator else name})
        elif arg.startswith("-M"):
            name, separator, source = arg[2:].partition("=")
            if not separator:
                raise RuntimeError(f"unsupported Zig module argument: {arg}")
            modules.append({"name": name, "source": source, "deps": dependencies})
            dependencies = []
        index += 1
    if not modules:
        raise RuntimeError("the selected Zig action has no modules")
    return modules


def action_config(graph: dict[str, object], execroot: str, label: str) -> dict[str, object]:
    actions = graph["actions"]
    links = [action for action in actions if action.get("mnemonic") == "CppLink"]
    zig_actions = [action for action in actions if action.get("mnemonic") == "ZigBuildLib"]
    if not links or not zig_actions:
        raise RuntimeError(f"{label} is not a rules_zig binary using Bazel's C++ linker")

    selected: tuple[dict[str, object], dict[str, object], str] | None = None
    for link in links:
        link_arguments = link.get("arguments", [])
        for zig_action in zig_actions:
            archive = emitted_path(zig_action.get("arguments", []))
            if archive and archive in link_arguments:
                if selected is not None:
                    raise RuntimeError(f"{label} produced more than one candidate Zig/C++ link pair")
                selected = (zig_action, link, archive)
    if selected is None:
        raise RuntimeError(f"could not match {label}'s Zig archive to its C++ link action")

    zig_action, link, archive = selected
    zig_arguments = zig_action["arguments"]
    link_arguments = link["arguments"]
    target = zig_arguments[zig_arguments.index("-target") + 1]
    optimize = zig_arguments[zig_arguments.index("-O") + 1]
    output_index = link_arguments.index("-o") + 1
    output_path = link_arguments[output_index]
    rewritten_link_args = [
        "$ZIG_ARCHIVE" if arg == archive else "$OUTPUT" if index == output_index else arg
        for index, arg in enumerate(link_arguments[1:], start=1)
    ]
    return {
        "execroot": execroot,
        "name": Path(output_path).name,
        "target": target,
        "optimize": optimize,
        "modules": parse_modules(zig_arguments),
        "linker": link_arguments[0],
        "link_args": rewritten_link_args,
        "link_env": link.get("environmentVariables", []),
    }


def split_arguments(arguments: list[str]) -> tuple[list[str], list[str]]:
    try:
        separator = arguments.index("--")
    except ValueError:
        return arguments, []
    return arguments[:separator], arguments[separator + 1 :]


def parse_cli(argv: list[str]) -> tuple[str, list[str], list[str]]:
    parser = argparse.ArgumentParser(
        description="Build a rules_zig target with Bazel and export it to build.zig",
    )
    parser.add_argument("label", help="a zig_binary label, for example //examples/mnist")
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv)
        raise AssertionError("argparse should have exited")

    # Parse only the label. argparse consumes a leading `--` from a REMAINDER
    # positional, which loses the separator in `<label> -- <zig args>`.
    args = parser.parse_args(argv[:1])
    bazel_flags, zig_args = split_arguments(argv[1:])
    return args.label, bazel_flags, zig_args


def main() -> int:
    label, user_bazel_flags, zig_args = parse_cli(sys.argv[1:])

    workspace = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", Path.cwd())).resolve()
    bazel_flags = ["--config=debug", *user_bazel_flags]
    bazel(workspace, ["build", *bazel_flags, label])
    graph_text = bazel(
        workspace,
        ["aquery", *bazel_flags, "--include_commandline", "--output=jsonproto", label],
        capture=True,
    )
    execroot = bazel(workspace, ["info", *bazel_flags, "execution_root"], capture=True).strip()
    config = action_config(json.loads(graph_text), execroot, label)

    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
    config_path = Path("/tmp") / "zig-bazel" / safe_label / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    zig = next(
        action["arguments"][0]
        for action in json.loads(graph_text)["actions"]
        if action.get("mnemonic") == "ZigBuildLib" and emitted_path(action.get("arguments", []))
    )
    zig_path = Path(execroot, zig)
    command = [
        str(zig_path),
        "build",
        f"-Dbazel-config={config_path}",
        *zig_args,
    ]
    print("\nBazel dependencies and configuration are ready. Running:\n")
    print("  " + shlex.join(command), flush=True)
    subprocess.run(command, cwd=workspace, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
