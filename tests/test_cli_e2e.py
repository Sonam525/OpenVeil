# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""End-to-end tests of the ``openveil`` command line on synthetic data (no models, no people).

The video flow runs every step a user runs, through :func:`openveil.cli.main`:
detect (fake proposer/verifier) -> episodes -> review build -> a review through the HTTP API
-> review export -> apply video (with the reviewer's overrides) -> verify. The image flow does
review build -> review export -> apply images. Smoke tests cover ``config show``,
``review demo --no-serve``, ``--version`` and ``python -m openveil``.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request

import numpy as np

from openveil import cli

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    import av  # noqa: F401
    HAVE_AV = True
except ImportError:  # pragma: no cover
    HAVE_AV = False
try:
    from PIL import Image  # noqa: F401
    HAVE_PIL = True
except ImportError:  # pragma: no cover
    HAVE_PIL = False
try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except ImportError:  # pragma: no cover
    HAVE_CV2 = False
try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except ImportError:  # pragma: no cover
    HAVE_YAML = False

HAVE_X264 = HAVE_AV and "libx264" in __import__("av").codecs_available
HAVE_IMG = HAVE_PIL or HAVE_CV2


def run_cli(*argv: str) -> tuple[int, str]:
    """Run ``openveil argv...`` in this process; returns (exit code, stdout)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(list(argv))
        except SystemExit as e:                     # argparse errors and --help
            code = e.code if isinstance(e.code, int) else 1
    return code, out.getvalue() + err.getvalue()


def decode_rgb(path: str) -> list[tuple[float, np.ndarray]]:
    """All frames of a video as (seconds, RGB array), in presentation order."""
    import av

    out = []
    with av.open(path) as c:
        st = c.streams.video[0]
        for fr in c.decode(st):
            out.append((float(fr.pts * st.time_base), fr.to_ndarray(format="rgb24")))
    out.sort(key=lambda x: x[0])
    return out


def frame_at(frames, t: float) -> np.ndarray:
    return min(frames, key=lambda x: abs(x[0] - t))[1]


def make_camera_video(path: str, frame_fn, seconds: float = 12.0, fps: int = 10) -> str:
    """H.264 MP4 like most cameras write it (no B-frames, parameter sets in every keyframe, fixed
    GOP), so that the apply step can smart-cut it. ``frame_fn(i)`` gives frame ``i`` (RGB)."""
    import av
    from fractions import Fraction

    with av.open(path, "w", format="mp4") as oc:
        st = oc.add_stream("libx264", rate=fps)
        st.width, st.height = frame_fn(0).shape[1], frame_fn(0).shape[0]
        st.pix_fmt = "yuv420p"
        st.codec_context.options = {"preset": "veryfast", "crf": "12",
                                    "x264-params": f"keyint={fps}:min-keyint={fps}:scenecut=0:bframes=0:repeat-headers=1"}
        for i in range(int(round(seconds * fps))):
            fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_fn(i)), format="rgb24")
            fr.pts, fr.time_base = i, Fraction(1, fps)
            for pkt in st.encode(fr):
                oc.mux(pkt)
        for pkt in st.encode(None):
            oc.mux(pkt)
    return path


class ReviewClient:
    """Talks to a review server the way the page does."""

    def __init__(self, review_dir: str):
        from openveil.review.server import make_server

        self.srv = make_server(review_dir, port=0)
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        self.base = self.srv.url.rstrip("/")

    def get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def save(self, rec: dict) -> dict:
        req = urllib.request.Request(self.base + "/api/decision", data=json.dumps(rec).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "X-Review-Tool": "1"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(5)


# ---------------------------------------------------------------------------------------------
# video flow
# ---------------------------------------------------------------------------------------------
@unittest.skipUnless(HAVE_X264 and HAVE_IMG, "needs PyAV with libx264 and Pillow or OpenCV")
class VideoFlowTest(unittest.TestCase):
    """detect -> episodes -> review build -> review -> export -> apply video -> verify."""

    @classmethod
    def setUpClass(cls):
        import test_detect_fakes as fk

        cls.fk = fk
        cls.tmp = tempfile.mkdtemp(prefix="ov_cli_")
        cls.videos = os.path.join(cls.tmp, "videos")
        os.makedirs(cls.videos)
        cls.video = make_camera_video(os.path.join(cls.videos, "cam1.mp4"), fk.synth_frame, seconds=12.0)
        cls.cfg = os.path.join(cls.tmp, "cfg.json")
        with open(cls.cfg, "w", encoding="utf-8") as f:
            json.dump({"verifier": {"mode": "none"}, "run": {"device": "cpu"}}, f)
        cls.tables = os.path.join(cls.tmp, "tables")
        cls.review = os.path.join(cls.tmp, "review")
        cls.overrides = os.path.join(cls.tmp, "overrides.json")
        cls.masked = os.path.join(cls.tmp, "masked", "cam1.mp4")
        cls.masked_plain = os.path.join(cls.tmp, "masked", "cam1_no_overrides.mp4")
        cls.added_box = [128.0, 86.0, 158.0, 112.0]     # straddles the decoy's lower-right corner

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_flow(self):
        from openveil.table import load_table

        # 1. detect (folder input, fake models built by the factory, in this process)
        code, out = run_cli("detect", self.videos, "--out", self.tables, "--config", self.cfg, "--device", "cpu",
                            "--in-process", "--factory", "test_detect_fakes:fake_factory")
        self.assertEqual(code, 0, out)
        table_path = os.path.join(self.tables, "cam1.json.gz")
        t = load_table(table_path, validate=True)
        self.assertEqual(t.video, "cam1")
        det_times = [t.seconds(r[0]) for r in t.frames if r[1]]
        self.assertTrue(det_times, "the walker was not detected")
        self.assertGreaterEqual(min(det_times), self.fk.WALK_T0 - 0.5)
        self.assertLessEqual(max(det_times), self.fk.WALK_T1 + 0.5)

        # 2. episodes: the walker box is small, so the giant rule never fires
        js = os.path.join(self.tmp, "episodes.json")
        code, out = run_cli("episodes", self.tables, "--min-share", "0.55", "--json", js)
        self.assertEqual(code, 0, out)
        self.assertIn("0 giant-rule episode(s) in 1 table(s)", out)
        with open(js, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["episodes"], {"cam1": []})

        # 3. review build
        code, out = run_cli("review", "build", "--tables", self.tables, "--videos", self.videos, "--out", self.review,
                            "--per-queue", "4")
        self.assertEqual(code, 0, out)
        self.assertTrue(os.path.isfile(os.path.join(self.review, "manifest.json")))

        # 4. review through the HTTP API: add a missed "person" on one mosaic frame, mark a random frame unsure
        client = ReviewClient(self.review)
        try:
            man = client.get("/api/manifest")
            qids = [q["id"] for q in man["queues"]]
            self.assertIn("mosaic", qids)
            self.assertIn("random", qids)
            entry = client.get("/api/queue/mosaic")[0]
            fr = client.get(f"/api/frame/{entry['fid']}")
            self.assertEqual(fr["video"], "cam1")
            persons = [list(b[:4]) for b in fr.get("applied", fr["persons"])] + [self.added_box]
            r = client.save(dict(fid=fr["id"], queue="mosaic", status="confirmed", persons=persons, ms=1500))
            self.assertTrue(r["ok"])
            t_added = fr["t"]
            rnd = client.get("/api/queue/random")[0]
            fr2 = client.get(f"/api/frame/{rnd['fid']}")
            r = client.save(dict(fid=fr2["id"], queue="random", status="unsure", persons=fr2["persons"], note="dark corner"))
            self.assertTrue(r["ok"])
        finally:
            client.close()

        # 5. review export
        code, out = run_cli("review", "export", self.review, "--overrides", self.overrides)
        self.assertEqual(code, 0, out)
        with open(self.overrides, encoding="utf-8") as f:
            ov = json.load(f)
        self.assertIn("cam1", ov)
        self.assertEqual(len(ov["cam1"]["add"]), 1)
        self.assertEqual(ov["cam1"]["add"][0][2], self.added_box)
        self.assertEqual(ov["cam1"]["delete"], [])
        with open(os.path.join(self.review, "unsure_frames.json"), encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)), 1)

        # 6. apply video with and without the overrides
        res_path = os.path.join(self.tmp, "result.json")
        code, out = run_cli("apply", "video", self.video, "--table", table_path, "--out", self.masked,
                            "--overrides", self.overrides, "--result", res_path)
        self.assertEqual(code, 0, out)
        self.assertIn("overrides:", out)
        with open(res_path, encoding="utf-8") as f:
            res = json.load(f)
        self.assertTrue(res["ok"])
        self.assertEqual(res["mode"], "smart")
        code, out = run_cli("apply", "video", self.video, "--table", table_path, "--out", self.masked_plain)
        self.assertEqual(code, 0, out)

        # 7. verify both outputs
        for p in (self.masked, self.masked_plain):
            code, out = run_cli("verify", self.video, p)
            self.assertEqual(code, 0, out)
            self.assertIn("OK", out)

        # 8. pixels: the walker is mosaicked; the reviewer's box only with the overrides
        src, ovr, plain = decode_rgb(self.video), decode_rgb(self.masked), decode_rgb(self.masked_plain)
        self.assertEqual(len(src), len(ovr))
        t_mid = (self.fk.WALK_T0 + self.fk.WALK_T1) / 2
        x1, y1, x2, y2 = (int(v) for v in self.fk.walker_box(t_mid))
        walker_diff = np.abs(frame_at(src, t_mid)[y1:y2, x1:x2].astype(int) - frame_at(plain, t_mid)[y1:y2, x1:x2]).max()
        self.assertGreater(walker_diff, 40, "the walker was not mosaicked")
        x1, y1, x2, y2 = (int(v) for v in self.added_box)

        def region_diff(frames):
            a = frame_at(src, t_added)[y1:y2, x1:x2].astype(int)
            return np.abs(a - frame_at(frames, t_added)[y1:y2, x1:x2]).max()

        self.assertGreater(region_diff(ovr), 20, "the reviewer's added box was not mosaicked")
        self.assertLess(region_diff(plain), 12, "the decoy was mosaicked without the overrides")
        # far from every box nothing changes beyond coding noise
        self.assertLess(np.abs(frame_at(src, 11.0).astype(int) - frame_at(ovr, 11.0)).max(), 12)

    def test_verify_detects_a_wrong_file(self):
        other = os.path.join(self.tmp, "other.mp4")
        self.fk.make_video(other, seconds=6.0)
        code, out = run_cli("verify", self.video, other)
        self.assertEqual(code, 1, out)
        self.assertIn("FAILED", out)

    def test_apply_refuses_bad_overrides(self):
        bad = os.path.join(self.tmp, "bad_overrides.json")
        with open(bad, "w", encoding="utf-8") as f:
            json.dump({"cam1": {"add": [[1.0, 0.5, [0, 0, 10, 10]]]}}, f)      # t1 before t0
        tables = os.path.join(self.tmp, "tables_bad")
        os.makedirs(tables)
        _write_table(os.path.join(tables, "cam1.json.gz"), giant=False)
        code, out = run_cli("apply", "video", self.video, "--table", os.path.join(tables, "cam1.json.gz"),
                            "--out", os.path.join(self.tmp, "never.mp4"), "--overrides", bad)
        self.assertEqual(code, 1, out)
        self.assertIn("before", out)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "never.mp4")))


def _write_table(path: str, giant: bool = True) -> None:
    """A tiny 160x120 table at 10 fps: with ``giant``, frames 20-24 hold a box over most of the
    image with two people inside it (the giant rule drops it)."""
    from openveil.table import BoxTable, save_table

    frames = []
    for i in range(0, 40, 2):
        boxes = [[10.0, 30.0, 30.0, 90.0, "det", 0.9]]
        if giant and 20 <= i <= 24:
            boxes += [[50.0, 40.0, 70.0, 100.0, "det", 0.8], [0.0, 0.0, 150.0, 115.0, "det", 0.6]]
        frames.append([i, boxes])
    save_table(BoxTable(video="cam1", width=160, height=120, fps=10.0, kind="video", time_base=(1, 10), first_pts=0,
                        start_local=None, params={}, chunks=[[0.0, "full", 40, 10]], frames=frames), path)


class EpisodesTest(unittest.TestCase):
    def test_giant_episode_listed(self):
        with tempfile.TemporaryDirectory() as d:
            _write_table(os.path.join(d, "cam1.json.gz"))
            code, out = run_cli("episodes", d)
            self.assertEqual(code, 0, out)
            self.assertIn("1 giant-rule episode(s) in 1 table(s) at min_share 0.55", out)
            self.assertRegex(out, r"cam1\s+2\.00-\s+2\.40 s\s+frames\s+3")
            # a higher threshold: the box no longer counts as giant
            code, out = run_cli("episodes", os.path.join(d, "cam1.json.gz"), "--min-share", "0.95")
            self.assertEqual(code, 0, out)
            self.assertIn("0 giant-rule episode(s)", out)

    def test_missing_folder(self):
        code, out = run_cli("episodes", os.path.join(tempfile.gettempdir(), "openveil-no-such-folder"))
        self.assertEqual(code, 1)
        self.assertIn("no such table or folder", out)


# ---------------------------------------------------------------------------------------------
# image flow
# ---------------------------------------------------------------------------------------------
@unittest.skipUnless(HAVE_PIL, "needs Pillow")
class ImageFlowTest(unittest.TestCase):
    """review build --images -> review -> export -> apply images."""

    def test_flow(self):
        from PIL import Image

        from openveil.review.server import Store

        with tempfile.TemporaryDirectory() as d:
            imgs, rev, out = os.path.join(d, "frames"), os.path.join(d, "review"), os.path.join(d, "masked")
            os.makedirs(imgs)
            rng = np.random.default_rng(3)
            for i in range(3):
                a = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)       # texture: a mosaic changes it
                Image.fromarray(a).save(os.path.join(imgs, f"img{i}.png"))
            boxes = {"img0.png": [[10, 10, 40, 60, 0.9]], "img1.png": [[60, 20, 90, 80, 0.05]], "img2.png": []}
            bj = os.path.join(d, "boxes.json")
            with open(bj, "w", encoding="utf-8") as f:
                json.dump(boxes, f)
            code, o = run_cli("review", "build", "--images", imgs, "--boxes", bj, "--out", rev)
            self.assertEqual(code, 0, o)
            # the reviewer adopts the low-scored candidate of img1 and confirms img0 and img2
            st = Store(rev)
            for fid, f in st.frames.items():
                persons = [list(b[:4]) for b in f["persons"]]
                if f["image"] == "img1.png":
                    persons.append([60.0, 20.0, 90.0, 80.0])
                st.save(dict(fid=fid, queue="random", status="confirmed", persons=persons))
            code, o = run_cli("review", "export", rev)
            self.assertEqual(code, 0, o)
            reviewed = os.path.join(rev, "reviewed_boxes.json")
            with open(reviewed, encoding="utf-8") as f:
                rb = json.load(f)
            self.assertEqual(rb["img1.png"], [[60.0, 20.0, 90.0, 80.0, "manual"]])     # confirmed: kept by the giant rule
            self.assertEqual(rb["img2.png"], [])
            code, o = run_cli("apply", "images", imgs, "--boxes", reviewed, "--out", out)
            self.assertEqual(code, 0, o)
            for name, changed in (("img0.png", True), ("img1.png", True), ("img2.png", False)):
                a = np.asarray(Image.open(os.path.join(imgs, name)).convert("RGB"))
                b = np.asarray(Image.open(os.path.join(out, name)).convert("RGB"))
                self.assertEqual(not np.array_equal(a, b), changed, name)
            with open(os.path.join(imgs, "img2.png"), "rb") as f1, open(os.path.join(out, "img2.png"), "rb") as f2:
                self.assertEqual(f1.read(), f2.read())


# ---------------------------------------------------------------------------------------------
# smoke tests
# ---------------------------------------------------------------------------------------------
class SmokeTest(unittest.TestCase):
    def test_config_show_json(self):
        from openveil.config import get_preset, to_dict

        code, out = run_cli("config", "show", "--preset", "production", "--format", "json")
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(out), to_dict(get_preset("production")))

    @unittest.skipUnless(HAVE_YAML, "needs PyYAML")
    def test_config_show_yaml_and_files(self):
        import yaml

        from openveil.config import get_preset, load_config, to_dict

        code, out = run_cli("config", "show")
        self.assertEqual(code, 0, out)
        self.assertEqual(yaml.safe_load(out), to_dict(get_preset()))
        # configs/production.yaml is the production preset written out in full
        self.assertEqual(to_dict(load_config(os.path.join(ROOT, "configs", "production.yaml"))), to_dict(get_preset()))
        code, out = run_cli("config", "show", "--config", os.path.join(ROOT, "configs", "example_barn.yaml"))
        self.assertEqual(code, 0, out)
        d = yaml.safe_load(out)
        self.assertEqual(d["proposer"]["prompt"], "worker. person. cow. calf.")
        self.assertEqual(d["proposer"]["positive_phrases"], ["worker", "person"])
        self.assertEqual(d["proposer"]["competitor_phrases"], ["cow", "calf"])
        self.assertEqual(d["schedule"]["day_night"]["policy"], "clock")
        self.assertEqual(d["apply"]["min_share"], 0.55)

    def test_config_errors(self):
        code, out = run_cli("config", "show", "--preset", "no-such-preset")
        self.assertEqual(code, 1)
        self.assertIn("no preset or config file", out)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"apply": {"min_shares": 0.5}}, f)
            code, out = run_cli("config", "show", "--config", p)
            self.assertEqual(code, 1)
            self.assertIn("unknown key", out)
        code, _ = run_cli("config", "show", "--config", "a.yaml", "--preset", "production")
        self.assertEqual(code, 2)                           # mutually exclusive

    @unittest.skipUnless(HAVE_PIL, "needs Pillow")
    def test_review_demo_no_serve(self):
        from openveil.review.server import load_manifest, make_server

        with tempfile.TemporaryDirectory() as d:
            demo = os.path.join(d, "demo")
            code, out = run_cli("review", "demo", "--no-serve", "--out", demo)
            self.assertEqual(code, 0, out)
            man = load_manifest(demo)
            self.assertTrue(man["meta"]["demo"])
            srv = make_server(demo, port=0, demo=True, reset=True)      # the demo guard accepts it
            srv.server_close()

    def test_help_and_version(self):
        code, out = run_cli("--version")
        self.assertEqual(code, 0)
        self.assertIn("openveil 0.1.0", out)
        code, out = run_cli()
        self.assertEqual(code, 2)
        self.assertIn("detect", out)
        for argv in (["review"], ["apply"], ["config"]):
            code, out = run_cli(*argv)
            self.assertEqual(code, 2, argv)
            self.assertIn("usage", out)
        for argv in (["review", "build", "--help"], ["review", "serve", "--help"], ["review", "export", "--help"],
                     ["review", "demo", "--help"], ["apply", "video", "--help"], ["detect", "--help"]):
            code, out = run_cli(*argv)
            self.assertEqual(code, 0, argv)
            self.assertIn("usage: openveil", out)

    def test_detect_needs_videos(self):
        with tempfile.TemporaryDirectory() as d:
            code, out = run_cli("detect", d, "--out", os.path.join(d, "t"))
            self.assertEqual(code, 1)
            self.assertIn("no video files found", out)
            code, out = run_cli("detect", os.path.join(d, "missing.mp4"), "--out", os.path.join(d, "t"))
            self.assertEqual(code, 1)
            self.assertIn("no such video", out)

    def test_python_dash_m(self):
        r = subprocess.run([sys.executable, "-B", "-m", "openveil", "--version"], cwd=ROOT, capture_output=True,
                           text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("openveil 0.1.0", r.stdout)

    def test_import_is_light(self):
        code = ("import sys, openveil.cli; heavy = [m for m in ('av', 'torch', 'cv2', 'PIL', 'scipy', 'yaml', "
                "'transformers') if m in sys.modules]; print(heavy)")
        r = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "[]")


class ContractTest(unittest.TestCase):
    """Every ``openveil ...`` command in the README and docs parses with the real CLI."""

    def _commands(self):
        files = [os.path.join(ROOT, "README.md")] + [os.path.join(ROOT, "docs", n) for n in sorted(os.listdir(os.path.join(ROOT, "docs")))
                                                    if n.endswith(".md")]
        for p in files:
            with open(p, encoding="utf-8") as f:
                lines = f.read().replace("\\\n", " ").splitlines()      # join continued lines
            fence = None
            for line in lines:
                st = line.strip()
                if st.startswith("```"):
                    fence = None if fence is not None else (st[3:].strip() or "text")
                    continue
                if fence in ("bash", "sh", "console", "text"):
                    cmd = st.split("#", 1)[0].strip()
                    if cmd.startswith("openveil "):
                        yield os.path.basename(p), cmd.split()[1:]

    def test_docs_commands_parse(self):
        import importlib

        n = 0
        for where, argv in self._commands():
            n += 1
            with self.subTest(where=where, cmd=" ".join(argv)):
                if argv[0] == "review":
                    # the review sub-commands own their parsers: check each option is defined there
                    mod = importlib.import_module(cli.REVIEW_COMMANDS[argv[1]][0])
                    with open(mod.__file__, encoding="utf-8") as f:
                        src = f.read()
                    for tok in argv[2:]:
                        if tok.startswith("--"):
                            self.assertIn(f'"{tok}"', src, f"{tok} is not an option of 'review {argv[1]}'")
                    continue
                parser = cli.build_parser()
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    try:
                        parser.parse_args(argv)
                    except SystemExit:                       # pragma: no cover - shows the bad command
                        self.fail(f"does not parse: {' '.join(argv)}: {err.getvalue()}")
        self.assertGreater(n, 10)


if __name__ == "__main__":
    unittest.main()
