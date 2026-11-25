#!/usr/bin/env python3
"""Serve completed checkpoints and cfg.yaml over an SSH-forwarded HTTP port.

Default root: scripts/wandb, containing run-*/files/checkpoint_latest.pt.
Legacy outputs/<date>/<run>/wandb/latest-run/files is also supported.
Runs remain listed across midnight. /download/<run-id> serves a checkpoint;
/config/<run-id> serves its saved training configuration.
"""

import argparse
import html
import os
import shutil
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse


def latest_checkpoint(run_dir: Path) -> Path:
    for relative in ("files/checkpoint_latest.pt", "checkpoint_latest.pt",
                     "wandb/latest-run/files/checkpoint_latest.pt"):
        candidate = run_dir / relative
        if candidate.is_file():
            return candidate
    return run_dir / "files/checkpoint_latest.pt"


def get_runs(root: Path) -> list[tuple[Path, Path]]:
    candidates = set(root.glob("run-*"))
    candidates.update(root.glob("????-??-??/*"))
    result = [(path, latest_checkpoint(path)) for path in candidates if path.is_dir()]
    result = [(path, ckpt) for path, ckpt in result if ckpt.is_file()]
    return sorted(result, key=lambda item: item[1].stat().st_mtime, reverse=True)


class CheckpointHandler(BaseHTTPRequestHandler):
    @property
    def _root(self) -> Path:
        # Set by the custom HTTPServer subclass below
        return self.server.root  # type: ignore[attr-defined]

    def _resolve_run_path(self, run_id: str) -> Path | None:
        """Resolve URL-encoded run id to a path under root; return None if invalid."""
        raw = unquote(run_id, errors="strict")
        if not raw or ".." in raw or raw.startswith("/"):
            return None
        path = (self._root / raw).resolve()
        try:
            path.relative_to(self._root)
        except ValueError:
            return None
        return path if path.is_dir() else None

    def _send_plain(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _file_headers(self, path: Path, filename: str | None = None, *, stat=None) -> None:
        """Send 200 and headers for a checkpoint file (Last-Modified so clients can skip re-download)."""
        st = stat if stat is not None else path.stat()
        size = st.st_size
        last_modified = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(st.st_mtime))
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size))
        self.send_header("Last-Modified", last_modified)
        self.send_header("Cache-Control", "no-cache")
        if filename:
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{html.escape(filename)}"',
            )
        self.end_headers()

    def _send_file(self, path: Path, filename: str | None = None) -> None:
        try:
            file = path.open("rb")
        except FileNotFoundError:
            self._send_plain("Not found", 404)
            return
        # Open once: atomic training saves may replace the path during download.
        with file:
            self._file_headers(path, filename, stat=os.fstat(file.fileno()))
            shutil.copyfileobj(file, self.wfile)

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # Include older runs so a continuous run stays visible after midnight.
        if path == "/":
            runs = get_runs(self._root)
            rows: list[str] = []
            for run_dir, ckpt in runs:
                rel = run_dir.relative_to(self._root)
                run_id = str(rel).replace(os.sep, "/")
                display_name = ckpt.resolve().name
                encoded_id = quote(run_id, safe="/")
                rows.append(
                    f'<tr><td>{html.escape(run_dir.name)}</td>'
                    f'<td><a href="/download/{encoded_id}">'
                    f'{html.escape(display_name)}</a></td>'
                    f'<td><a href="/config/{encoded_id}">cfg.yaml</a></td></tr>'
                )
            if rows:
                table = (
                    '<table border="1"><tr><th>Run</th><th>Latest checkpoint</th><th>Config</th></tr>\n'
                    + "\n".join(rows)
                    + "\n</table>"
                )
            else:
                table = "<p>No completed checkpoints yet.</p>"

            body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Training checkpoints</title></head>
<body>
<h1>Training checkpoints</h1>
{table}
</body></html>"""
            self._send_html(body)
            return

        # Download latest checkpoint for a specific run
        if path.startswith(("/download/", "/config/")):
            prefix = "/config/" if path.startswith("/config/") else "/download/"
            run_id = path[len(prefix) :].lstrip("/")
            run_dir = self._resolve_run_path(run_id)
            if run_dir is None:
                self._send_plain("Invalid run", 400)
                return
            latest = latest_checkpoint(run_dir)
            if prefix == "/config/":
                self._send_file(latest.parent / "cfg.yaml", "cfg.yaml")
                return
            if not latest.exists():
                self._send_plain("No checkpoint found", 404)
                return
            filename = latest.resolve().name if latest.is_symlink() else latest.name
            self._send_file(latest, filename)
            return

        self._send_plain("Not found", 404)

    def do_HEAD(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        """HEAD for /download/<run_id>: same headers as GET so clients can check Last-Modified without downloading."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path.startswith("/download/"):
            run_id = path[len("/download/") :].lstrip("/")
            run_dir = self._resolve_run_path(run_id)
            if run_dir is None:
                self._send_plain("Invalid run", 400)
                return
            latest = latest_checkpoint(run_dir)
            if not latest.exists():
                self._send_plain("No checkpoint found", 404)
                return
            filename = latest.resolve().name if latest.is_symlink() else latest.name
            self._file_headers(latest, filename)
            return
        self._send_plain("Not found", 404)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        # Print to stdout; you can customize or silence logging here.
        print(format % args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve latest checkpoints from run directories for download."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "wandb",
        help="Checkpoint root (default: scripts/wandb; legacy outputs also supported)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)",
    )
    parser.add_argument(
        "--bind",
        type=str,
        default="127.0.0.1",
        help="Address to bind (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    class Server(HTTPServer):
        def __init__(self, server_address, RequestHandlerClass, root_path: Path):
            super().__init__(server_address, RequestHandlerClass)
            self.root = root_path

    server = Server((args.bind, args.port), CheckpointHandler, root)
    print(f"Serving checkpoints from {root} at http://{args.bind}:{args.port}/")
    print("Completed checkpoints from all runs are listed on the index page.")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

