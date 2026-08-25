#!/usr/bin/env bash
set -euo pipefail
bin="${1:-}"
if [[ -n "$bin" && ! -x "$bin" && -n "${TEST_SRCDIR:-}" ]]; then
    for prefix in "${TEST_WORKSPACE:-_main}" zml _main; do
        cand="$TEST_SRCDIR/$prefix/$bin"
        if [[ -x "$cand" ]]; then
            bin="$cand"
            break
        fi
    done
fi
exec "$bin"
