#!/usr/bin/env bash
set -euo pipefail

# The Bazel ZLS launcher provides the SDK path to compiler subprocesses.
zig_lib_dir=${ZIG_LIB_DIR:-}
if [[ -n "$zig_lib_dir" && -x "$zig_lib_dir/../zig" ]]; then
    exec "$zig_lib_dir/../zig" "$@"
fi

zig_exe=$(
    cd "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
    zig_bin=$(bazel cquery --noshow_progress --output=starlark \
        --starlark:expr='providers(target)["TemplateVariableInfo"].variables["ZIG_BIN"]' \
        @rules_zig//zig:resolved_toolchain) || exit 1
    if [[ "$zig_bin" != /* ]]; then
        exec_root=$(bazel info execution_root) || exit 1
        zig_bin="$exec_root/$zig_bin"
    fi
    printf '%s\n' "$zig_bin"
)
exec "$zig_exe" "$@"
