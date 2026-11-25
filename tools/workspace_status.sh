#!/bin/bash
# Trees synced without .git (e.g. to remote build hosts) report "dev" quietly.
if git diff --quiet 2>/dev/null; then
    echo BUILD_SCM_REVISION "$(git rev-parse --short HEAD)"
else
    echo BUILD_SCM_REVISION "dev"
fi

echo STABLE_ZML_SMI_VERSION "$(git describe --tags --dirty --match 'zml-smi-v*' 2>/dev/null || echo "dev")"
