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

On Linux, `-Dzig-linker=true` uses a ZML-specific development path: a Bazel LLD
action collapses the native XLA/PJRT/tokenizer graph into one shared image, then
Zig incrementally rebuilds its archive and performs a thin final link. Restart
this tool after adding a new Zig-to-native symbol reference or changing Bazel deps.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET


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


def is_dev_null_path(path: str) -> bool:
    return path == os.devnull or path.startswith(f"{os.devnull}{os.sep}")


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
            value = link_arguments[index + 1]
            sysroot = None if is_dev_null_path(value) else value
            index += 2
            continue
        if arg.startswith("--sysroot="):
            value = arg.removeprefix("--sysroot=")
            sysroot = None if is_dev_null_path(value) else value
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
    module_action = zig_action
    if not any(argument.startswith("-M") for argument in zig_arguments):
        module_actions = [
            action
            for action in actions
            if action.get("mnemonic") == "ZigBuildTest"
            and emitted_path(action.get("arguments", [])) in zig_arguments
        ]
        if len(module_actions) != 1:
            raise RuntimeError(
                f"could not match {label}'s Zig archive to one module compilation action"
            )
        module_action = module_actions[0]
    module_arguments = module_action["arguments"]
    link_arguments = link["arguments"]
    target = module_arguments[module_arguments.index("-target") + 1]
    optimize = module_arguments[module_arguments.index("-O") + 1]
    output_index = link_arguments.index("-o") + 1
    output_path = link_arguments[output_index]
    rewritten_link_args = [
        "$ZIG_ARCHIVE" if arg == archive else "$OUTPUT" if index == output_index else arg
        for index, arg in enumerate(link_arguments[1:], start=1)
    ]
    return {
        "execroot": execroot,
        "name": Path(output_path).name,
        "kind": "test" if module_action.get("mnemonic") == "ZigBuildTest" else "binary",
        "target": target,
        "optimize": optimize,
        "modules": parse_modules(module_arguments),
        "linker": link_arguments[0],
        "link_args": rewritten_link_args,
        "link_env": link.get("environmentVariables", []),
        "native_prelink": None,
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


def query_rule_args(workspace: Path, label: str) -> list[str]:
    xml_text = bazel(workspace, ["query", label, "--output=xml"], capture=True)
    root = ET.fromstring(xml_text)
    rule = root.find("rule")
    if rule is None:
        return []
    args = rule.find("list[@name='args']")
    if args is None:
        return []
    return [item.attrib["value"] for item in args.findall("string")]


def cquery_json(workspace: Path, bazel_flags: list[str], label: str, expression: str) -> object:
    return json.loads(
        bazel(
            workspace,
            ["cquery", *bazel_flags, label, "--output=starlark", f"--starlark:expr={expression}"],
            capture=True,
        )
    )


def run_info(workspace: Path, bazel_flags: list[str], label: str, script_dir: Path) -> dict[str, object]:
    formatter = script_dir / "run_info.cquery.bzl"
    formatter.write_text(
        """
def _file_info(file):
    return {
        "path": file.path,
        "short_path": file.short_path,
        "repo_name": file.owner.repo_name,
    }

def format(target):
    target_providers = providers(target)
    default = target_providers["DefaultInfo"]
    files_to_run = default.files_to_run
    run_environment = target_providers.get("RunEnvironmentInfo")
    environment = [] if run_environment == None else [
        {"key": key, "value": run_environment.environment[key]}
        for key in sorted(run_environment.environment.keys())
    ]
    inherited_environment = [] if run_environment == None else run_environment.inherited_environment
    executable = files_to_run.executable
    repo_mapping_manifest = files_to_run.repo_mapping_manifest
    runfiles_manifest = files_to_run.runfiles_manifest
    return json.encode({
        "executable": None if executable == None else executable.path,
        "repo_mapping_manifest": None if repo_mapping_manifest == None else repo_mapping_manifest.path,
        "runfiles_manifest": None if runfiles_manifest == None else runfiles_manifest.path,
        "runfiles": [_file_info(file) for file in default.default_runfiles.files.to_list()],
        "run_env": environment,
        "inherited_run_env": inherited_environment,
    })
""".lstrip()
    )
    return json.loads(
        bazel(
            workspace,
            ["cquery", *bazel_flags, label, "--output=starlark", f"--starlark:file={formatter}"],
            capture=True,
        )
    )


def label_workspace_and_package(label: str) -> tuple[str, str]:
    workspace = ""
    rest = label
    if label.startswith("@"):
        separator = label.index("//")
        workspace = label[:separator]
        rest = label[separator:]
    if not rest.startswith("//"):
        raise RuntimeError(f"unsupported Bazel label: {label}")
    package_and_name = rest[2:]
    package = package_and_name.partition(":")[0]
    return workspace, package


def resolve_label(label: str, base_label: str) -> str:
    if label.startswith("@") or label.startswith("//"):
        return label
    workspace, package = label_workspace_and_package(base_label)
    if label.startswith(":"):
        return f"{workspace}//{package}{label}"
    return f"{workspace}//{package}:{label}"


def label_files(
    workspace: Path,
    bazel_flags: list[str],
    label: str,
    cache: dict[str, list[dict[str, str]]],
) -> list[dict[str, str]]:
    if label not in cache:
        expression = (
            'json.encode([{"path": file.path, "short_path": file.short_path, '
            '"repo_name": file.owner.repo_name} '
            'for file in providers(target)["DefaultInfo"].files.to_list()])'
        )
        cache[label] = cquery_json(workspace, bazel_flags, label, expression)  # type: ignore[assignment]
    return cache[label]


def execroot_absolute(execroot: str, path: str) -> str:
    return path if os.path.isabs(path) else str(Path(execroot, path))


def location_path(kind: str, file: dict[str, str], execroot: str) -> str:
    if kind.startswith("execpath"):
        return execroot_absolute(execroot, file["path"])
    return file["short_path"]


LOCATION_PATTERN = re.compile(
    r"\$\((location|locations|execpath|execpaths|rootpath|rootpaths|rlocationpath|rlocationpaths)\s+([^)]+)\)"
)


def expand_run_args(
    workspace: Path,
    bazel_flags: list[str],
    base_label: str,
    execroot: str,
    args: list[str],
) -> list[str]:
    cache: dict[str, list[dict[str, str]]] = {}

    def replace(match: re.Match[str]) -> str:
        kind, raw_label = match.groups()
        label = resolve_label(raw_label.strip(), base_label)
        files = label_files(workspace, bazel_flags, label, cache)
        if not kind.endswith("s") and len(files) != 1:
            raise RuntimeError(f"{kind} expects one file for {label}, got {len(files)}")
        return " ".join(location_path(kind, file, execroot) for file in files)

    return [LOCATION_PATTERN.sub(replace, arg) for arg in args]


def module_name(workspace: Path) -> str:
    module_file = workspace / "MODULE.bazel"
    if not module_file.exists():
        return workspace.name
    match = re.search(r"\bmodule\s*\(\s*name\s*=\s*\"([^\"]+)\"", module_file.read_text())
    return match.group(1) if match else workspace.name


def runfiles_key(short_path: str) -> str:
    if short_path.startswith("../"):
        key = posixpath.normpath(short_path.removeprefix("../"))
    else:
        key = posixpath.normpath(posixpath.join("_main", short_path))
    if key.startswith("../") or key == "..":
        raise RuntimeError(f"unsupported runfiles path escaping root: {short_path}")
    return key


def apparent_repo_names(canonical: str) -> list[str]:
    names = [canonical]
    for prefix in ("+non_module_deps+", "+"):
        if canonical.startswith(prefix):
            names.append(canonical.removeprefix(prefix))
    return sorted(set(names))


def materialize_runfiles(
    *,
    runfiles_dir: Path,
    run_info: dict[str, object],
    execroot: str,
    main_repo_name: str,
) -> Path:
    if runfiles_dir.exists() or runfiles_dir.is_symlink():
        if runfiles_dir.is_symlink() or runfiles_dir.is_file():
            runfiles_dir.unlink()
        else:
            shutil.rmtree(runfiles_dir)
    runfiles_dir.mkdir(parents=True)
    (runfiles_dir / "_main").mkdir()

    executable = run_info.get("executable")
    manifest_lines: list[str] = []
    repo_names: set[str] = set()
    for file in run_info["runfiles"]:  # type: ignore[index]
        assert isinstance(file, dict)
        if file["path"] == executable:
            continue
        key = runfiles_key(file["short_path"])
        source = Path(execroot_absolute(execroot, file["path"]))
        if not source.exists():
            raise RuntimeError(f"Bazel did not materialize runfile input: {source}")
        destination = runfiles_dir / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source, destination, target_is_directory=source.is_dir())
        manifest_lines.append(f"{key} {source}\n")
        if file["repo_name"]:
            repo_names.add(file["repo_name"])

    repo_mapping_lines: set[str] = set()
    source_repos = ["", *sorted(repo_names)]
    apparent_main_names = sorted({main_repo_name, "_main", ""})
    for source_repo in source_repos:
        for apparent in apparent_main_names:
            repo_mapping_lines.add(f"{source_repo},{apparent},_main\n")
        for canonical in sorted(repo_names):
            for apparent in apparent_repo_names(canonical):
                repo_mapping_lines.add(f"{source_repo},{apparent},{canonical}\n")

    repo_mapping = runfiles_dir / "_repo_mapping"
    repo_mapping.write_text("".join(sorted(repo_mapping_lines)))
    manifest_lines.append(f"_repo_mapping {repo_mapping}\n")

    manifest = runfiles_dir / "MANIFEST"
    manifest.write_text("".join(sorted(manifest_lines)))
    return manifest


def split_arguments(arguments: list[str]) -> tuple[list[str], list[str]]:
    try:
        separator = arguments.index("--")
    except ValueError:
        return arguments, []
    return arguments[:separator], arguments[separator + 1 :]


def has_bazel_config(arguments: list[str], name: str) -> bool:
    for index, argument in enumerate(arguments):
        if argument == f"--config={name}":
            return True
        if argument == "--config" and index + 1 < len(arguments) and arguments[index + 1] == name:
            return True
    return False


def uses_llvm_static_runtime(config: dict[str, object]) -> bool:
    link_args = config.get("link_args")
    if not isinstance(link_args, list):
        return False

    llvm_runtime_archives = {
        "liblibcxx.static.a",
        "liblibcxxabi.static.a",
        "liblibunwind.static.a",
    }
    return any(
        isinstance(argument, str) and Path(argument).name in llvm_runtime_archives
        for argument in link_args
    )


def zig_bool_option(arguments: list[str], name: str) -> bool:
    prefix = f"-D{name}="
    value: str | None = None
    for argument in arguments:
        if argument == f"-D{name}":
            value = "true"
        elif argument.startswith(prefix):
            value = argument.removeprefix(prefix)
    return value == "true"


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
    bazel_flags = [*user_bazel_flags] if has_bazel_config(user_bazel_flags, "debug") else ["--config=debug", *user_bazel_flags]
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
    use_zig_linker = zig_bool_option(zig_args, "zig-linker")
    output_groups = ["zig_build_inputs"]
    if use_zig_linker:
        target = config["zig_link"]["target"]  # type: ignore[index]
        if not isinstance(target, str) or not target.endswith("-linux-gnu"):
            raise RuntimeError(f"ZML's native prelink currently supports Linux GNU targets, got {target}")
        output_groups.append("zig_native_prelink")

    if not use_zig_linker and uses_llvm_static_runtime(config):
        # HACK: These archives are implicit inputs of LLVM's CppLink action, so
        # rules_zig's zig_build_inputs output group does not materialize them.
        # This tool is ZML-specific and experimental, so build LLVM's runtime
        # filegroup explicitly instead of making the extraction generic for now.
        bazel(
            workspace,
            [
                "build",
                "--show_result=0",
                *bazel_flags,
                "@llvm//runtimes/cxxstdlib:static_runtime_lib",
            ],
        )
    if use_zig_linker:
        print("\nPrelinking ZML's native dependencies with LLD inside Bazel...\n", flush=True)
    bazel(
        workspace,
        ["build", "--show_result=0", *bazel_flags, f"--output_groups={','.join(output_groups)}", label],
    )

    if use_zig_linker:
        outputs = bazel(
            workspace,
            ["cquery", *bazel_flags, "--output=files", "--output_groups=zig_native_prelink", label],
            capture=True,
        ).splitlines()
        if len(outputs) != 1:
            raise RuntimeError(f"expected one zig_native_prelink output for {label}, got {outputs}")
        native_prelink = Path(execroot_absolute(execroot, outputs[0]))
        if not native_prelink.exists():
            raise RuntimeError(f"Bazel native prelink was not materialized: {native_prelink}")
        config["native_prelink"] = str(native_prelink)

    target_run_info = run_info(workspace, bazel_flags, label, config_path.parent)
    runfiles_dir = config_path.parent / "runfiles"
    runfiles_manifest = materialize_runfiles(
        runfiles_dir=runfiles_dir,
        run_info=target_run_info,
        execroot=execroot,
        main_repo_name=module_name(workspace),
    )
    run_env = [
        {"key": "BUILD_EXECROOT", "value": execroot},
        {"key": "BUILD_WORKING_DIRECTORY", "value": os.environ.get("BUILD_WORKING_DIRECTORY", str(workspace))},
        {"key": "BUILD_WORKSPACE_DIRECTORY", "value": str(workspace)},
    ]
    if "BUILD_ID" in os.environ:
        run_env.append({"key": "BUILD_ID", "value": os.environ["BUILD_ID"]})
    run_env.extend(target_run_info["run_env"])
    for name in target_run_info["inherited_run_env"]:
        if name in os.environ:
            run_env.append({"key": name, "value": os.environ[name]})
    config.update(
        {
            "run_cwd": str(runfiles_dir / "_main"),
            "run_env": run_env,
            "run_args": expand_run_args(
                workspace,
                bazel_flags,
                label,
                execroot,
                query_rule_args(workspace, label),
            ),
            "runfiles_dir": str(runfiles_dir),
            "runfiles_manifest": str(runfiles_manifest),
        }
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    zig = next(
        action["arguments"][0]
        for action in json.loads(graph_text)["actions"]
        if action.get("mnemonic") == "ZigBuildLib" and emitted_path(action.get("arguments", []))
    )
    zig_path = Path(execroot, zig)
    sysroot_args: list[str] = []
    zig_link = config.get("zig_link")
    if isinstance(zig_link, dict):
        sysroot = zig_link.get("sysroot")
        if isinstance(sysroot, str) and sysroot:
            sysroot_args = ["--sysroot", execroot_absolute(execroot, sysroot)]
    command = [
        str(zig_path),
        "build",
        *sysroot_args,
        f"-Dbazel-config={config_path}",
        *zig_args,
    ]
    print("\nBazel dependencies and configuration are ready. Running:\n")
    print("  " + shlex.join(command), flush=True)
    try:
        return subprocess.run(command, cwd=workspace).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
