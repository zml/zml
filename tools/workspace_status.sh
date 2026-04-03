#!/bin/bash
if git diff --quiet; then
    echo BUILD_SCM_REVISION "$(git rev-parse --short HEAD)"
else
    echo BUILD_SCM_REVISION "dev"
fi
