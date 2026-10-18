#!/usr/bin/env bash
set -euo pipefail
script_path="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "${BASH_SOURCE[0]}")"
exec python3 "$(dirname -- "${script_path}")/zls_launcher.py" "$@"
