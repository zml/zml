import sys

try:
    from xprof.cli import xprof_cli
except ImportError:
    xprof_cli = None
    from xprof.server import main as xprof_server_main


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
    if xprof_cli is not None:
        raise SystemExit(xprof_cli.main(sys.argv))
    raise SystemExit(xprof_server_main())
