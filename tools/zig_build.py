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


def append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def is_link_input(argument: str) -> bool:
    return argument.endswith((".o", ".a", ".dylib", ".so"))


def target_with_os_version(target: str, os_version_min: str | None) -> str:
    if not os_version_min:
        return target

    parts = target.split("-")
    if len(parts) < 2 or "." in parts[1]:
        return target

    parts[1] = f"{parts[1]}.{os_version_min}"
    return "-".join(parts)


def parse_zig_link(link_arguments: list[str], archive: str) -> dict[str, object]:
    objects: list[str] = []
    archive_objects: list[str] = []
    library_paths: list[str] = []
    system_libraries: list[str] = []
    frameworks: list[str] = []
    needed_frameworks: list[str] = []
    weak_frameworks: list[str] = []
    skipped_args: list[str] = []
    sysroot: str | None = None
    os_version_min: str | None = None
    headerpad_max_install_names = False
    dead_strip = False
    in_start_lib = False

    index = 1
    while index < len(link_arguments):
        arg = link_arguments[index]

        if arg == "-Wl,--start-lib":
            in_start_lib = True
            index += 1
            continue
        if arg == "-Wl,--end-lib":
            in_start_lib = False
            index += 1
            continue
        if arg == archive:
            index += 1
            continue
        if arg in ("-o", "-target"):
            index += 2
            continue
        if arg == "--sysroot":
            sysroot = link_arguments[index + 1]
            index += 2
            continue
        if arg.startswith("--sysroot="):
            sysroot = arg.removeprefix("--sysroot=")
            index += 1
            continue
        if arg.startswith("-mmacosx-version-min="):
            os_version_min = arg.removeprefix("-mmacosx-version-min=")
            index += 1
            continue
        if arg == "-framework":
            append_unique(frameworks, link_arguments[index + 1])
            index += 2
            continue
        if arg == "-needed_framework":
            append_unique(needed_frameworks, link_arguments[index + 1])
            index += 2
            continue
        if arg == "-weak_framework":
            append_unique(weak_frameworks, link_arguments[index + 1])
            index += 2
            continue
        if arg == "-L":
            append_unique(library_paths, link_arguments[index + 1])
            index += 2
            continue
        if arg.startswith("-L") and len(arg) > 2:
            library_path = arg[2:]
            if "libunwind_library_search_directory" not in library_path:
                append_unique(library_paths, library_path)
            index += 1
            continue
        if arg.startswith("-l") and len(arg) > 2:
            append_unique(system_libraries, arg[2:])
            index += 1
            continue
        if arg == "-pthread":
            append_unique(system_libraries, "pthread")
            index += 1
            continue
        if arg in ("-headerpad_max_install_names",):
            headerpad_max_install_names = True
            index += 1
            continue
        if arg in ("-dead_strip",):
            dead_strip = True
            index += 1
            continue
        if arg in (
            "-fuse-ld=lld",
            "-rtlib=compiler-rt",
            "-Wl,-no_warn_duplicate_libraries",
            "-Wl,-oso_prefix,.",
            "-Wl,--icf=safe",
        ):
            index += 1
            continue
        if arg == "-Wl,-dead_strip":
            dead_strip = True
            index += 1
            continue
        if arg.startswith("-Wl,-framework,"):
            append_unique(frameworks, arg.removeprefix("-Wl,-framework,"))
            index += 1
            continue
        if arg.startswith("-Wl,-needed_framework,"):
            append_unique(needed_frameworks, arg.removeprefix("-Wl,-needed_framework,"))
            index += 1
            continue
        if arg.startswith("-Wl,-weak_framework,"):
            append_unique(weak_frameworks, arg.removeprefix("-Wl,-weak_framework,"))
            index += 1
            continue
        if arg.startswith("-Wl,-force_load,"):
            append_unique(objects, arg.removeprefix("-Wl,-force_load,"))
            index += 1
            continue
        if arg.startswith("-Wl,-install_name,"):
            index += 1
            continue
        if is_link_input(arg):
            if "clang_rt.builtins.static" not in arg:
                append_unique(archive_objects if in_start_lib else objects, arg)
            index += 1
            continue

        skipped_args.append(arg)
        index += 1

    return {
        "target": target_with_os_version(
            link_arguments[link_arguments.index("-target") + 1].replace("apple-darwin", "macos-none")
            if "-target" in link_arguments
            else "",
            os_version_min,
        ),
        "sysroot": sysroot,
        "objects": objects,
        "archive_objects": archive_objects,
        "library_paths": library_paths,
        "framework_paths": [f"{sysroot}/System/Library/Frameworks"] if sysroot else [],
        "system_libraries": system_libraries,
        "frameworks": frameworks,
        "needed_frameworks": needed_frameworks,
        "weak_frameworks": weak_frameworks,
        "headerpad_max_install_names": headerpad_max_install_names,
        "dead_strip": dead_strip,
        "skipped_args": skipped_args,
    }


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
        "zig_link": parse_zig_link(link_arguments, archive),
        "runfiles_dir": f"{output_path}.runfiles",
        "runfiles_manifest": f"{output_path}.runfiles_manifest",
    }


def parse_run_script(script: str) -> dict[str, object]:
    tokens = shlex.split(script.replace("\\\n", " "))
    try:
        cwd = tokens[tokens.index("cd") + 1]
        env_index = tokens.index("env")
    except (ValueError, IndexError) as err:
        raise RuntimeError("could not parse Bazel run script") from err

    run_env: list[dict[str, str]] = []
    index = env_index + 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-u":
            index += 2
            continue
        name, separator, value = token.partition("=")
        if separator and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            run_env.append({"key": name, "value": value})
            index += 1
            continue
        break

    if index >= len(tokens):
        raise RuntimeError("Bazel run script has no executable")

    run_args = tokens[index + 1 :]
    if run_args and run_args[-1] == "$@":
        run_args.pop()

    return {
        "run_cwd": cwd,
        "run_env": run_env,
        "run_args": run_args,
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
    run_script_path = config_path.parent / "run.sh"
    bazel(workspace, ["run", f"--script_path={run_script_path}", *bazel_flags, label])
    config.update(parse_run_script(run_script_path.read_text()))
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
