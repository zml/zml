#!/usr/bin/env python3
"""Serve the Source-to-HLO viewer and one compiler artifact directory."""

from __future__ import annotations

import argparse
import http.server
import mimetypes
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit
import webbrowser


ARTIFACT_NAMES = frozenset(
    {
        "source.zig",
        "stablehlo.mlir",
        "hlo.before_optimizations.txt",
        "mapping.json",
    }
)
STATIC_NAMES = frozenset({"index.html", "app.js", "styles.css"})


class ExplorerHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler restricted to known viewer and artifact files."""

    artifact_dir: Path
    viewer_dir: Path

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self._serve(send_body=False)

    def _serve(self, *, send_body: bool) -> None:
        request_path = unquote(urlsplit(self.path).path)
        if request_path in {"", "/", "/index.html"}:
            path = self.viewer_dir / "index.html"
        elif request_path.startswith("/artifact/"):
            name = request_path.removeprefix("/artifact/")
            if name not in ARTIFACT_NAMES:
                self.send_error(http.HTTPStatus.NOT_FOUND)
                return
            path = self.artifact_dir / name
        else:
            name = request_path.removeprefix("/")
            if name not in STATIC_NAMES:
                if name == "favicon.ico":
                    self.send_response(http.HTTPStatus.NO_CONTENT)
                    self.end_headers()
                else:
                    self.send_error(http.HTTPStatus.NOT_FOUND)
                return
            path = self.viewer_dir / name

        self._send_file(path, send_body=send_body)

    def _send_file(self, path: Path, *, send_body: bool) -> None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            self.send_error(http.HTTPStatus.NOT_FOUND)
            return
        except OSError as error:
            self.send_error(http.HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in {".zig", ".mlir", ".txt"}:
            content_type = "text/plain"
        self.send_response(http.HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'")
        self.end_headers()
        if send_body:
            self.wfile.write(data)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    fixture_dir = script_dir / "viewer" / "fixture"
    parser = argparse.ArgumentParser(
        description="Serve a ZML Source-to-HLO Explorer artifact bundle.",
    )
    parser.add_argument(
        "artifact_dir",
        nargs="?",
        type=Path,
        default=fixture_dir,
        help=f"artifact directory (default: bundled fixture at {fixture_dir})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="listen address")
    parser.add_argument("--port", type=int, default=8000, help="listen port; use 0 for any free port")
    parser.add_argument("--open", action="store_true", help="open the viewer in the default browser")
    return parser.parse_args()


def validate_artifacts(artifact_dir: Path) -> Path:
    resolved = artifact_dir.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(f"artifact directory does not exist: {resolved}")
    missing = sorted(name for name in ARTIFACT_NAMES if not (resolved / name).is_file())
    if missing:
        raise ValueError(f"artifact directory is missing: {', '.join(missing)}")
    return resolved


def main() -> int:
    args = parse_args()
    try:
        artifact_dir = validate_artifacts(args.artifact_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    viewer_dir = Path(__file__).resolve().parent / "viewer"
    handler = type(
        "ConfiguredExplorerHandler",
        (ExplorerHandler,),
        {"artifact_dir": artifact_dir, "viewer_dir": viewer_dir},
    )
    try:
        server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as error:
        print(f"error: could not listen on {args.host}:{args.port}: {error}", file=sys.stderr)
        return 2

    host, port = server.server_address[:2]
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{port}/"
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("warning: the explorer is exposed without authentication", file=sys.stderr, flush=True)
    print(f"Serving artifacts from {artifact_dir}", flush=True)
    print(f"Open {url}", flush=True)
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping explorer.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
