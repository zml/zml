import os
import sys
from pathlib import Path

from xprof import profile_plugin
from xprof.cli.xprof_cli import main as xprof_main


def _configure_static_dir() -> None:
    static_dir = Path(profile_plugin.__file__).parent / "static"
    # Bazel runfiles contain a real directory whose files are symlinks into the
    # repository cache. Resolve a file so XProf's traversal check sees the same
    # base path as the assets it opens.
    resolved_static_dir = (static_dir / "index.html").resolve().parent
    os.environ.setdefault("XPROF_STATIC_DIR", str(resolved_static_dir))


def _argv() -> list[str]:
    user_args = sys.argv[1:]
    if user_args and not user_args[0].startswith("-"):
        logdir = user_args[0]
        trailing_args = user_args[1:]
    else:
        logdir = "."
        trailing_args = user_args
    return [
        sys.argv[0],
        "--logdir",
        logdir,
        "--max_concurrent_worker_requests",
        "32",
        "--port",
        "6006",
        "--hide_capture_profile_button",
        *trailing_args,
    ]


if __name__ == "__main__":
    _configure_static_dir()
    sys.argv = _argv()
    raise SystemExit(xprof_main())
