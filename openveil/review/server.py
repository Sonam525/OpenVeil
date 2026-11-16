# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Local review server: one web page for checking and correcting person boxes, frame by frame.

    python -m openveil.review.server REVIEW_DIR [--port 8765] [--demo] [--reset] [--open]

REVIEW_DIR holds ``manifest.json`` (written by :mod:`openveil.review.build` or
:mod:`openveil.review.demo`). Every decision is appended to
``REVIEW_DIR/decisions.jsonl`` the moment it is saved, one JSON line per save. The
latest line of a frame wins and nothing is ever rewritten, so stopping the server
(Ctrl+C) loses nothing.

The frames usually show people before masking. The server is therefore built to keep
them on this computer:

* it listens on 127.0.0.1 only;
* it answers only requests whose ``Host`` is ``127.0.0.1:<port>`` or
  ``localhost:<port>`` (this blocks DNS rebinding);
* saving needs the custom header ``X-Review-Tool: 1``, which a page on another site
  cannot send without a CORS preflight that this server never grants;
* every response carries ``Cross-Origin-Resource-Policy: same-origin``,
  ``X-Frame-Options: DENY`` and a Content-Security-Policy with
  ``frame-ancestors 'none'``, so other sites can neither embed the images nor frame
  the page;
* images are served only by frame id, from the paths in the manifest, never from a
  path in the request, and the client never sees those paths.

Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import shutil
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
APP_HTML = os.path.join(HERE, "app.html")

MANIFEST = "manifest.json"
DECISIONS = "decisions.jsonl"
SEED_DECISIONS = "seed_decisions.jsonl"

STATUSES = ("confirmed", "unsure")
DEFAULT_PORT = 8765
DEMO_PORT = 8766
REAL_DATA_BANNER = "Unmasked data - keep on this computer"

MAX_BODY = 200_000          # bytes of one POST body
MAX_BOXES = 100             # person boxes in one decision
MAX_NOTE = 500              # characters of a note
MAX_QUEUE = 64              # characters of the queue id stored with a decision
MAX_MS = 600_000            # the time spent on a frame is clamped to 10 minutes
MIN_BOX_PX = 3              # saved boxes smaller than this (either side) are dropped

ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")
IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
NONCE_PLACEHOLDER = "__CSP_NONCE__"


class DemoGuardError(ValueError):
    """The review folder is not a self-contained demo, so ``--demo`` (and ``--reset``) refuse it."""


# ---------------------------------------------------------------------------------------------------------------------
# shared helpers (also used by build, export and demo)
# ---------------------------------------------------------------------------------------------------------------------
def write_json_atomic(path: str, obj: Any, indent: int | None = None) -> None:
    """Write JSON to a temporary file next to ``path``, flush it to disk, then rename it over ``path``.

    A crash or a full disk therefore never leaves a half-written file under the final name."""
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=indent)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_manifest(review_dir: str) -> dict:
    """Read ``REVIEW_DIR/manifest.json`` and check its structure (raises ``ValueError`` with every problem found)."""
    with open(os.path.join(review_dir, MANIFEST), encoding="utf-8-sig") as fh:
        manifest = json.load(fh)
    problems = validate_manifest(manifest)
    if problems:
        more = f" (and {len(problems) - 10} more)" if len(problems) > 10 else ""
        raise ValueError("invalid manifest: " + "; ".join(problems[:10]) + more)
    return manifest


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_box(b: Any, n: int = 4) -> bool:
    return (isinstance(b, list) and len(b) >= n
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in b[:n]))


def validate_manifest(manifest: Any) -> list[str]:
    """Return a list of problems with a manifest (empty when it is valid).

    Required: ``meta`` (dict), ``frames`` (dict fid -> record with ``id`` == fid, ``key``, ``kind``,
    positive ``width``/``height``, ``path``, ``persons`` [[x1, y1, x2, y2], ...] and ``cands``
    [[x1, y1, x2, y2, score, label], ...]) and ``queues`` (list of ``{id, title, desc, entries}``
    whose entries name existing frames). Frame and queue ids may use only letters, digits and ``_.-``
    (they appear in URLs)."""
    p: list[str] = []
    if not isinstance(manifest, dict):
        return ["the manifest is not a JSON object"]
    if not isinstance(manifest.get("meta"), dict):
        p.append("'meta' must be an object")
    frames = manifest.get("frames")
    if not isinstance(frames, dict):
        return p + ["'frames' must be an object"]
    keys = set()
    for fid, f in frames.items():
        if not ID_RE.match(str(fid)):
            p.append(f"frame id {fid!r} has characters other than letters, digits and _.-")
        if not isinstance(f, dict):
            p.append(f"frame {fid}: not an object")
            continue
        if f.get("id") != fid:
            p.append(f"frame {fid}: 'id' differs from its name")
        if not isinstance(f.get("key"), str) or not f["key"]:
            p.append(f"frame {fid}: 'key' must be a non-empty string")
        elif f["key"] in keys:
            p.append(f"frame {fid}: key {f['key']!r} is used twice")
        else:
            keys.add(f["key"])
        if not isinstance(f.get("kind"), str):
            p.append(f"frame {fid}: 'kind' must be a string")
        for k in ("width", "height"):
            v = f.get(k)
            if not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
                p.append(f"frame {fid}: '{k}' must be a positive integer")
        if not isinstance(f.get("path"), str) or not f["path"]:
            p.append(f"frame {fid}: 'path' must be a non-empty string")
        if not (isinstance(f.get("persons"), list) and all(_is_box(b) for b in f["persons"])):
            p.append(f"frame {fid}: 'persons' must be a list of [x1, y1, x2, y2]")
        if not (isinstance(f.get("cands", []), list) and all(_is_box(c, 5) for c in f.get("cands", []))):
            p.append(f"frame {fid}: 'cands' must be a list of [x1, y1, x2, y2, score, label]")
    queues = manifest.get("queues")
    if not isinstance(queues, list):
        return p + ["'queues' must be a list"]
    seen = set()
    for q in queues:
        if not isinstance(q, dict) or not isinstance(q.get("id"), str):
            p.append("a queue without an id")
            continue
        qid = q["id"]
        if not ID_RE.match(qid):
            p.append(f"queue id {qid!r} has characters other than letters, digits and _.-")
        if qid in seen:
            p.append(f"queue id {qid!r} is used twice")
        seen.add(qid)
        if not isinstance(q.get("entries"), list):
            p.append(f"queue {qid}: 'entries' must be a list")
            continue
        for e in q["entries"]:
            if not isinstance(e, dict) or e.get("fid") not in frames:
                p.append(f"queue {qid}: an entry names no frame of the manifest")
                break
    if not seen:
        p.append("the manifest has no queue")
    return p


def frame_path(review_dir: str, frame: dict) -> str:
    """Absolute, resolved path of a frame's image. A relative path is relative to the review folder."""
    path = frame["path"]
    if not os.path.isabs(path):
        path = os.path.join(review_dir, path)
    return os.path.realpath(path)


def _inside(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(path), os.path.normcase(folder)]) == os.path.normcase(folder)
    except ValueError:                                     # different drives on Windows
        return False


def check_demo(review_dir: str, manifest: dict) -> None:
    """Refuse (``DemoGuardError``) unless ``meta.demo`` is true and every frame image lies inside the review folder.

    This is the guard that keeps demo mode (which may reset decisions and is meant for screenshots)
    from being pointed at real, unmasked data."""
    if not (isinstance(manifest.get("meta"), dict) and manifest["meta"].get("demo") is True):
        raise DemoGuardError("--demo refuses this folder: its manifest is not a demo manifest (meta.demo is not true)")
    root = os.path.realpath(review_dir)
    outside = [fid for fid, f in manifest["frames"].items() if not _inside(frame_path(review_dir, f), root)]
    if outside:
        raise DemoGuardError(f"--demo refuses this folder: {len(outside)} frame image(s) lie outside it (e.g. {outside[0]})")


def _timestamp() -> str:
    """Local time ``YYYY-MM-DDTHH:MM:SS.mmm`` from one clock reading (so seconds and milliseconds agree)."""
    now = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"


def valid_decision(d: Any, frames: dict) -> bool:
    """True when ``d`` is a decision record that belongs to this manifest (same frame id and key)."""
    return (isinstance(d, dict) and isinstance(d.get("fid"), str) and d["fid"] in frames
            and d.get("key") == frames[d["fid"]]["key"] and d.get("status") in STATUSES
            and isinstance(d.get("persons"), list) and all(_is_box(b) for b in d["persons"]))


def read_decisions(review_dir: str, frames: dict, warn: bool = True) -> dict[str, dict]:
    """Latest decision of every frame (fid -> record), read from ``decisions.jsonl``.

    Lines are applied in file order and the latest line of a frame wins, whatever its status;
    callers that want only confirmed frames filter afterwards (``latest wins, then filter``).
    Unreadable lines (a write torn by a crash), lines of another manifest (key mismatch) and
    lines with a bad status are skipped with a warning."""
    path = os.path.join(review_dir, DECISIONS)
    latest: dict[str, dict] = {}
    if not os.path.exists(path):
        return latest
    bad = []
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                bad.append(n)
                continue
            if not valid_decision(d, frames):
                bad.append(n)
                continue
            latest[d["fid"]] = d
    if bad and warn:
        print(f"WARNING: {len(bad)} unreadable or foreign line(s) in {DECISIONS} were skipped "
              f"(line numbers {bad[:20]}); those frames show as not reviewed", file=sys.stderr)
    return latest


# ---------------------------------------------------------------------------------------------------------------------
# the store: manifest + decisions
# ---------------------------------------------------------------------------------------------------------------------
class Store:
    """The manifest of a review folder and the latest decision of every frame; appends new decisions durably."""

    def __init__(self, review_dir: str, demo: bool = False):
        self.dir = os.path.abspath(review_dir)
        self.manifest = load_manifest(self.dir)
        if demo:
            check_demo(self.dir, self.manifest)
        self.meta = dict(self.manifest["meta"])
        if not self.meta.get("demo") and not self.meta.get("banner"):
            self.meta["banner"] = REAL_DATA_BANNER
        self.frames: dict[str, dict] = self.manifest["frames"]
        self.queues = {q["id"]: q for q in self.manifest["queues"]}
        self.path = os.path.join(self.dir, DECISIONS)
        self.lock = threading.Lock()
        self.latest = read_decisions(self.dir, self.frames)
        self.images = {fid: frame_path(self.dir, f) for fid, f in self.frames.items()}
        self.roots = {os.path.dirname(p) for p in self.images.values()}

    def status(self, fid: str) -> str | None:
        d = self.latest.get(fid)
        return None if d is None else d["status"]

    def queue(self, qid: str) -> list[dict]:
        """The entries of a queue, each with the current status of its frame."""
        return [dict(e, status=self.status(e["fid"])) for e in self.queues[qid]["entries"]]

    def stats(self) -> dict[str, dict]:
        """Per queue: entries ``n``, frames with a decision ``done``, of which ``unsure``."""
        out = {}
        for qid, q in self.queues.items():
            st = [self.status(e["fid"]) for e in q["entries"]]
            out[qid] = dict(n=len(st), done=sum(s is not None for s in st), unsure=sum(s == "unsure" for s in st))
        return out

    def summary(self) -> dict:
        """What ``/api/manifest`` returns: meta, the queue list (without entries) and the stats."""
        return dict(meta=self.meta,
                    queues=[dict(id=q["id"], title=q.get("title", q["id"]), desc=q.get("desc", ""), n=len(q["entries"]))
                            for q in self.manifest["queues"]],
                    stats=self.stats())

    def frame(self, fid: str) -> dict:
        """A frame record for the page: without its path, with its latest decision (or None)."""
        f = self.frames[fid]
        return dict({k: v for k, v in f.items() if k != "path"}, decision=self.latest.get(fid))

    def image(self, fid: str) -> tuple[str, str] | None:
        """(path, content type) of a frame's image, or None when it is missing or of an unsupported type."""
        path = self.images[fid]
        ctype = IMAGE_TYPES.get(os.path.splitext(path)[1].lower())
        if ctype is None or os.path.dirname(path) not in self.roots or not os.path.isfile(path):
            return None
        return path, ctype

    def _ends_with_newline(self) -> bool:
        try:
            with open(self.path, "rb") as fh:
                if fh.seek(0, 2) == 0:
                    return True
                fh.seek(-1, 2)
                return fh.read(1) == b"\n"
        except FileNotFoundError:
            return True

    def save(self, rec: dict) -> dict | None:
        """Validate, clamp and append one decision.

        Returns the stored record, or None when the frame got a newer decision since the page
        loaded it (``rec['base']`` is not the ``ts`` of the current latest decision): another tab
        saved it, and the caller answers 409. Raises ``ValueError`` on a malformed record."""
        fid = rec["fid"]
        f = self.frames[fid]
        W, H = f["width"], f["height"]
        persons = []
        for b in rec["persons"]:
            # extra fields after the four coordinates (e.g. a score from a hand-made manifest) are ignored
            if not (isinstance(b, list) and len(b) >= 4 and all(_is_num(v) for v in b[:4])):
                raise ValueError("a box needs four numbers")
            x1, y1, x2, y2 = [float(v) for v in b[:4]]
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                raise ValueError("box coordinates must be finite")
            big = abs(x2 - x1) >= MIN_BOX_PX and abs(y2 - y1) >= MIN_BOX_PX
            x1, x2 = sorted((min(max(x1, 0.0), W), min(max(x2, 0.0), W)))
            y1, y2 = sorted((min(max(y1, 0.0), H), min(max(y2, 0.0), H)))
            if x2 - x1 >= MIN_BOX_PX and y2 - y1 >= MIN_BOX_PX:
                persons.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
            elif big:
                # a real box that lies outside the frame: never drop a person box silently
                raise ValueError(f"box {[round(v, 1) for v in b[:4]]} lies outside the {W} x {H} frame")
        status = rec["status"] if rec.get("status") in STATUSES else "confirmed"
        ms = float(rec.get("ms", 0) or 0)
        ms = int(min(max(ms, 0), MAX_MS)) if math.isfinite(ms) else 0
        d = dict(fid=fid, key=f["key"], kind=f["kind"], queue=str(rec.get("queue", ""))[:MAX_QUEUE], status=status,
                 persons=persons, note=str(rec.get("note", ""))[:MAX_NOTE], ms=ms, ts="")
        with self.lock:
            cur = self.latest.get(fid)
            if "base" in rec and rec["base"] != (cur["ts"] if cur else None):
                return None
            d["ts"] = _timestamp()
            while cur is not None and d["ts"] == cur["ts"]:   # the stale-save guard needs a new ts for every save of a frame
                time.sleep(0.001)
                d["ts"] = _timestamp()
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(("" if self._ends_with_newline() else "\n") + json.dumps(d) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.latest[fid] = d
        return d


# ---------------------------------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------------------------------
_COMMON_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Referrer-Policy": "no-referrer",
}


class Handler(BaseHTTPRequestHandler):
    """Request handler; the store and the port come from the server (``self.server``)."""

    server_version = "OpenVeilReview"
    sys_version = ""

    def log_message(self, fmt, *args):                     # no request log on the console
        pass

    # -- helpers --
    def _allowed_hosts(self) -> tuple[str, ...]:
        port = self.server.server_address[1]
        return (f"127.0.0.1:{port}", f"localhost:{port}")

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in self._allowed_hosts()

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in tuple(f"http://{h}" for h in self._allowed_hosts())

    def _send(self, code: int, body: Any, ctype: str = "application/json", csp: str | None = None) -> None:
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith(("application/json", "text/html")) else "private, max-age=3600")
        for k, v in _COMMON_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Security-Policy", csp or "default-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _page(self) -> None:
        nonce = secrets.token_urlsafe(16)
        with open(APP_HTML, encoding="utf-8") as fh:
            html = fh.read().replace(NONCE_PLACEHOLDER, nonce)
        csp = (f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; "
               "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
        self._send(200, html, "text/html; charset=utf-8", csp=csp)

    # -- GET --
    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"error": "host not allowed"})
        S: Store = self.server.store
        p = self.path.split("?", 1)[0]
        try:
            if p in ("/", "/index.html"):
                return self._page()
            if p == "/api/manifest":
                return self._send(200, S.summary())
            if p == "/api/stats":
                return self._send(200, S.stats())
            if p.startswith("/api/queue/"):
                qid = p[len("/api/queue/"):]
                if qid not in S.queues:
                    return self._send(404, {"error": "no such queue"})
                return self._send(200, S.queue(qid))
            if p.startswith("/api/frame/"):
                fid = p[len("/api/frame/"):]
                if fid not in S.frames:
                    return self._send(404, {"error": "no such frame"})
                return self._send(200, S.frame(fid))
            if p.startswith("/img/"):
                fid = p[len("/img/"):]
                if fid not in S.frames:
                    return self._send(404, {"error": "no such frame"})
                found = S.image(fid)
                if found is None:
                    return self._send(404, {"error": "image not available"})
                with open(found[0], "rb") as fh:
                    return self._send(200, fh.read(), found[1])
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    do_HEAD = do_GET

    # -- POST --
    def do_POST(self):
        if not self._host_ok() or not self._origin_ok() or self.headers.get("X-Review-Tool") != "1":
            return self._send(403, {"error": "forbidden"})
        S: Store = self.server.store
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_BODY:
                return self._send(400, {"error": "bad length"})
            rec = json.loads(self.rfile.read(n))
            if self.path != "/api/decision":
                return self._send(404, {"error": "not found"})
            if (not isinstance(rec, dict) or not isinstance(rec.get("fid"), str) or rec["fid"] not in S.frames
                    or not isinstance(rec.get("persons"), list) or len(rec["persons"]) > MAX_BOXES):
                return self._send(400, {"error": "bad record"})
            d = S.save(rec)
            if d is None:
                return self._send(409, {"error": "this frame was changed since you opened it (another tab?): reload the page"})
            return self._send(200, dict(ok=True, saved=d, stats=S.stats()))
        except (ValueError, KeyError, TypeError) as e:
            return self._send(400, {"error": f"bad request: {e}"})
        except OSError as e:                               # disk full, file locked ...
            return self._send(500, {"error": f"could not write the decision: {e}"})

    def do_OPTIONS(self):                                  # never grant a CORS preflight
        return self._send(403, {"error": "forbidden"})


class ReviewServer(ThreadingHTTPServer):
    """The HTTP server; ``store`` holds the review folder. Binds 127.0.0.1 only."""

    allow_reuse_address = False                            # on Windows SO_REUSEADDR would let a second copy bind the port silently
    daemon_threads = True

    def __init__(self, store: Store, port: int = DEFAULT_PORT):
        self.store = store
        super().__init__(("127.0.0.1", port), Handler)

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    @property
    def port(self) -> int:
        return self.server_address[1]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"


def reset_decisions(review_dir: str) -> None:
    """Demo only: replace ``decisions.jsonl`` by ``seed_decisions.jsonl`` (or remove it when there is no seed)."""
    manifest = load_manifest(review_dir)
    check_demo(review_dir, manifest)
    seed = os.path.join(review_dir, SEED_DECISIONS)
    dst = os.path.join(review_dir, DECISIONS)
    if os.path.exists(seed):
        shutil.copyfile(seed, dst)
    elif os.path.exists(dst):
        os.remove(dst)


def make_server(review_dir: str, port: int = DEFAULT_PORT, demo: bool = False, reset: bool = False) -> ReviewServer:
    """Load the review folder and bind the server (not started yet; call ``serve_forever``).

    ``port=0`` picks a free port (see ``server.port``). ``demo=True`` applies the demo guard;
    ``reset=True`` (demo only) restores the seed decisions first. Raises ``DemoGuardError``,
    ``ValueError`` (bad manifest) or ``OSError`` (port in use)."""
    if reset and not demo:
        raise DemoGuardError("--reset is allowed only together with --demo (it replaces the decisions)")
    if reset:
        reset_decisions(review_dir)
    store = Store(review_dir, demo=demo)
    return ReviewServer(store, port)


def serve(review_dir: str, port: int = DEFAULT_PORT, demo: bool = False, reset: bool = False, open_browser: bool = False) -> None:
    """Run the review server until Ctrl+C."""
    try:
        if reset and not demo:
            raise DemoGuardError("--reset is allowed only together with --demo (it replaces the decisions)")
        if reset:
            reset_decisions(review_dir)
        store = Store(review_dir, demo=demo)
    except DemoGuardError as e:
        raise SystemExit(str(e))
    except FileNotFoundError:
        raise SystemExit(f"cannot start: {review_dir} is not a review folder (no {MANIFEST}); build one with "
                         "'openveil review build' or try 'openveil review demo'")
    except (ValueError, OSError) as e:
        raise SystemExit(f"cannot start: the review folder {review_dir} cannot be read: {e}")
    try:
        srv = ReviewServer(store, port)
    except OSError as e:
        raise SystemExit(f"cannot start: port {port} is already in use ({e}). Close the other review window first, or use --port.")
    S = srv.store
    print(f"review tool on {srv.url}   (frames: {len(S.frames)}, reviewed so far: {len(S.latest)}; Ctrl+C to stop)", flush=True)
    print(f"banner: {S.meta.get('banner', '')}", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(srv.url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        srv.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="openveil review serve", description="Local review page for person boxes (127.0.0.1 only).")
    ap.add_argument("review_dir")
    ap.add_argument("--port", type=int, default=None,
                    help=f"default {DEFAULT_PORT} ({DEMO_PORT} with --demo); 0 picks a free port (the URL is printed)")
    ap.add_argument("--demo", action="store_true", help="refuse anything but a self-contained demo folder (meta.demo, images inside the folder)")
    ap.add_argument("--reset", action="store_true", help="demo only: restore seed_decisions.jsonl before starting")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    a = ap.parse_args(argv)
    port = a.port if a.port is not None else (DEMO_PORT if a.demo else DEFAULT_PORT)
    serve(a.review_dir, port, demo=a.demo, reset=a.reset, open_browser=a.open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
