#!/bin/bash
if git diff --quiet; then
    echo BUILD_SCM_REVISION "$(git rev-parse --short HEAD)"
else
    echo BUILD_SCM_REVISION "dev"
fi

# Keep the benchmark's source identity even for a dirty kernel tuning tree.
echo STABLE_MOE_BENCH_REVISION "$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git status --porcelain --untracked-files=normal 2>/dev/null)" ]; then
    echo STABLE_MOE_BENCH_DIRTY true
else
    echo STABLE_MOE_BENCH_DIRTY false
fi
# Hash tracked edits and new source files, excluding benchmark dumps and build
# artifacts. git hash-object is available wherever the revision can be read.
moe_bench_source_hash=$(
    {
        git rev-parse HEAD
        git diff --no-color --no-ext-diff --no-textconv --binary HEAD -- .
        while IFS= read -r -d '' source; do
            case "$source" in
                (*.zig|*.bzl|*.bazel|*.py|*.sh|*.c|*.cc|*.cpp|*.cu|*.cuh|*.h|*.hpp|*.mlir)
                    printf '%s\0' "$source"
                    git hash-object -- "$source"
                    ;;
            esac
        done < <(git ls-files --others --exclude-standard -z)
    } | git hash-object --stdin
)
echo STABLE_MOE_BENCH_SOURCE_HASH "$moe_bench_source_hash"

echo STABLE_ZML_SMI_VERSION "$(git describe --tags --dirty --match 'zml-smi-v*' 2>/dev/null || echo "dev")"
