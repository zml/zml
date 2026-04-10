import sys

from xprof.server import main as xprof_main


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
    sys.argv = _argv()
    raise SystemExit(xprof_main())
