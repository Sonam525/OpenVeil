# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Review builder: from an image folder + boxes JSON, and from box tables + videos (synthetic data only)."""
import gzip
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

from openveil.review import build as rb
from openveil.review import server as rs

HAS_PIL = importlib.util.find_spec("PIL") is not None
HAS_AV = importlib.util.find_spec("av") is not None
HAS_NP = importlib.util.find_spec("numpy") is not None


class TestGeometry(unittest.TestCase):
    def test_union_area(self):
        self.assertEqual(rb.union_area([]), 0.0)
        self.assertEqual(rb.union_area([[0, 0, 10, 10]]), 100.0)
        self.assertEqual(rb.union_area([[0, 0, 10, 10], [5, 0, 15, 10]]), 150.0)
        self.assertEqual(rb.union_area([[0, 0, 10, 10], [2, 2, 4, 4]]), 100.0)
        self.assertEqual(rb.union_area([[0, 0, 10, 10], [20, 20, 30, 30]]), 200.0)

    def test_giant_rule(self):
        boxes = [[0, 0, 90, 90, "det"], [5, 5, 20, 20, "det"], [50, 50, 70, 70, "det"]]
        kept, dropped = rb.giant_split(boxes, 100, 100, 0.55)
        self.assertEqual(dropped, [boxes[0]])
        self.assertEqual(len(kept), 2)
        kept, dropped = rb.giant_split(boxes[:2], 100, 100, 0.55)          # one inner box: kept
        self.assertEqual(dropped, [])
        kept, dropped = rb.giant_split(boxes, 100, 100, 0.85)              # below min share
        self.assertEqual(dropped, [])

    def test_mosaic_share_and_score(self):
        self.assertAlmostEqual(rb.mosaic_share([[40, 40, 60, 60]], 100, 100), (28 * 28) / 1e4)
        self.assertEqual(rb.box_score([0, 0, 1, 1, "det", 0.4]), 0.4)
        self.assertEqual(rb.box_score([0, 0, 1, 1, 0.3]), 0.3)
        self.assertIsNone(rb.box_score([0, 0, 1, 1, "det"]))

    def test_read_boxes_layouts(self):
        d = tempfile.mkdtemp(prefix="ov_bx_")
        try:
            layouts = [
                {"a.jpg": [[0, 0, 5, 5]], "b.jpg": [{"box": [1, 1, 4, 4], "score": 0.2}]},
                {"images": {"a.jpg": {"boxes": [[0, 0, 5, 5, "det", 0.9]], "width": 10, "height": 8}}},
                [{"image": "a.jpg", "boxes": [[5, 5, 0, 0]]}, {"name": "b.jpg", "boxes": []}],
            ]
            for i, data in enumerate(layouts):
                p = os.path.join(d, f"{i}.json")
                with open(p, "w") as fh:
                    json.dump(data, fh)
                got = rb.read_boxes_json(p)
                self.assertEqual(got["a.jpg"]["boxes"][0][0], [0.0, 0.0, 5.0, 5.0])
            self.assertEqual(rb.read_boxes_json(os.path.join(d, "0.json"))["b.jpg"]["boxes"][0][1], 0.2)
            self.assertEqual(rb.read_boxes_json(os.path.join(d, "1.json"))["a.jpg"]["width"], 10)
            self.assertEqual(rb.read_boxes_json(os.path.join(d, "2.json"))["b.jpg"]["boxes"], [])
            # a malformed box is an error (it would otherwise vanish from the review and the export)
            p = os.path.join(d, "bad.json")
            with open(p, "w") as fh:
                json.dump({"b.jpg": [[0, 0, "x", 1]]}, fh)
            with self.assertRaises(ValueError):
                rb.read_boxes_json(p)
        finally:
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(HAS_PIL, "Pillow is not installed")
class TestBuildFromImages(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.root = tempfile.mkdtemp(prefix="ov_bimg_")
        self.img = os.path.join(self.root, "images")
        os.makedirs(os.path.join(self.img, "sub"))
        for i, name in enumerate(["a.jpg", "b.png", "c.jpg", "sub/d.jpg", "e.webp", "f.jpg"]):
            Image.new("RGB", (200, 100), (20 * i, 100, 150)).save(os.path.join(self.img, name))
        self.boxes = {
            "a.jpg": [[10, 10, 60, 90, "det", 0.9]],                                  # proposal
            "b.png": [[0, 0, 190, 95, "det", 0.8], [10, 10, 40, 60, "det", 0.9], [100, 10, 140, 60, "det", 0.7]],   # giant
            "c.jpg": [[50, 20, 70, 60, "det", 0.04]],                                  # below the threshold only
            "sub/d.jpg": [[5, 5, 30, 40]],                                             # no score: proposal
            "missing.jpg": [],                                                         # not in the folder, no boxes: skipped
        }
        self.bpath = os.path.join(self.root, "boxes.json")
        with open(self.bpath, "w") as fh:
            json.dump(self.boxes, fh)
        self.out = os.path.join(self.root, "review")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_build(self):
        m = rb.build_from_images(self.img, self.bpath, self.out, quiet=True)
        self.assertEqual(rs.validate_manifest(m), [])
        self.assertEqual(len(m["frames"]), 6)
        self.assertIs(m["meta"]["demo"], False)
        self.assertEqual(m["meta"]["banner"], rs.REAL_DATA_BANNER)
        q = {x["id"]: [e["fid"] for e in x["entries"]] for x in m["queues"]}
        self.assertEqual(set(q), {"mosaic", "giant", "below", "random"})
        by_name = {f["image"]: f for f in m["frames"].values()}
        self.assertEqual(q["giant"], [by_name["b.png"]["id"]])
        self.assertEqual(q["below"], [by_name["c.jpg"]["id"]])
        # largest mosaic first; measured after the giant rule, so the giant box of b.png does not count
        self.assertEqual(q["mosaic"][0], by_name["a.jpg"]["id"])
        self.assertEqual(set(q["mosaic"]), {by_name[n]["id"] for n in ("a.jpg", "b.png", "sub/d.jpg")})
        self.assertEqual(len(q["random"]), 6)
        self.assertEqual(by_name["c.jpg"]["persons"], [])
        self.assertEqual(by_name["c.jpg"]["cands"][0][4], 0.04)
        self.assertEqual(by_name["sub/d.jpg"]["persons"], [[5.0, 5.0, 30.0, 40.0]])
        self.assertEqual(len(by_name["b.png"]["applied"]), 2)
        self.assertIn("giant", by_name["b.png"])
        for f in m["frames"].values():
            self.assertFalse(os.path.isabs(f["path"]))                             # relative to the review folder
            self.assertEqual((f["width"], f["height"]), (200, 100))
        # the server serves them with the right content type
        s = rs.Store(self.out)
        self.assertEqual(s.image(by_name["b.png"]["id"])[1], "image/png")
        self.assertEqual(s.image(by_name["e.webp"]["id"])[1], "image/webp")

    def test_rebuild_keeps_ids_and_other_queues(self):
        m1 = rb.build_from_images(self.img, self.bpath, self.out, quiet=True)
        fid = next(f for f, r in m1["frames"].items() if r["image"] == "a.jpg")
        with open(os.path.join(self.out, "decisions.jsonl"), "w") as fh:
            fh.write(json.dumps(dict(fid=fid, key=m1["frames"][fid]["key"], kind="image", queue="random", status="confirmed",
                                     persons=[], note="", ms=1, ts="2026-01-01T00:00:00.000")) + "\n")
        # add a hand-made queue, then rebuild: it must survive, ids must not change
        m1["queues"].append(dict(id="extra", title="extra", desc="", entries=[dict(fid=fid, np=1, note="")]))
        rs.write_json_atomic(os.path.join(self.out, "manifest.json"), m1)
        from PIL import Image
        Image.new("RGB", (200, 100)).save(os.path.join(self.img, "g.jpg"))
        m2 = rb.build_from_images(self.img, self.bpath, self.out, quiet=True)
        self.assertEqual(len(m2["frames"]), 7)
        for f, r in m1["frames"].items():
            self.assertEqual(m2["frames"][f]["key"], r["key"])
        new = [f for f in m2["frames"] if f not in m1["frames"]]
        self.assertEqual(new, ["I0006"])
        self.assertIn("extra", [q["id"] for q in m2["queues"]])
        self.assertEqual(len(rs.read_decisions(self.out, m2["frames"])), 1)       # the decision is still valid

    def test_copy_mode(self):
        m = rb.build_from_images(self.img, self.bpath, self.out, copy=True, quiet=True)
        for f in m["frames"].values():
            self.assertTrue(f["path"].startswith("frames/"))
            self.assertTrue(os.path.isfile(os.path.join(self.out, f["path"])))

    def test_refuses_demo_folder(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, "manifest.json"), "w") as fh:
            json.dump(dict(meta=dict(demo=True), frames={}, queues=[]), fh)
        with self.assertRaises((SystemExit, ValueError)):
            rb.build_from_images(self.img, self.bpath, self.out, quiet=True)

    def test_cli(self):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(rb.main(["--images", self.img, "--boxes", self.bpath, "--out", self.out, "--per-queue", "3",
                                      "--title", "My review", "--banner", "Keep local"]), 0)
        self.assertIn("missing.jpg", out.getvalue())
        m = rs.load_manifest(self.out)
        self.assertEqual(m["meta"]["title"], "My review")
        self.assertEqual(m["meta"]["banner"], "Keep local")
        self.assertEqual(len(next(q for q in m["queues"] if q["id"] == "random")["entries"]), 3)


def make_video(path, n=150, fps=25, w=320, h=180):
    """A synthetic video whose frame i has brightness 1.5 * i in its top-left corner; returns the sorted pts and time base."""
    import av
    import numpy as np
    with av.open(path, "w") as c:
        s = c.add_stream("libx264" if "libx264" in av.codecs_available else "mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"g": "50"}
        for i in range(n):
            arr = np.full((h, w, 3), 90, np.uint8)
            arr[:40, :60] = int(1.5 * i)
            fr = av.VideoFrame.from_ndarray(arr, format="rgb24")
            fr.pts = i
            for p in s.encode(fr):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    with av.open(path) as c:
        st = c.streams.video[0]
        pts = sorted(f.pts for f in c.decode(st))
        tb = st.time_base
    return pts, [tb.numerator, tb.denominator]


@unittest.skipUnless(HAS_PIL and HAS_AV and HAS_NP, "needs Pillow, PyAV and numpy")
class TestBuildFromTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="ov_btab_")
        cls.videos = os.path.join(cls.root, "videos")
        cls.tables = os.path.join(cls.root, "tables")
        os.makedirs(cls.videos)
        os.makedirs(cls.tables)
        cls.pts, cls.tb = make_video(os.path.join(cls.videos, "cam1_clip.mp4"))
        P = cls.pts
        frames = []
        for i in range(10, 16):                                            # a large person box
            frames.append([P[i], [[40, 20, 200, 170, "det"]]])
        for i in range(60, 64):                                            # giant-rule episode
            frames.append([P[i], [[0, 0, 310, 175, "det"], [20, 20, 60, 120, "interp"], [200, 30, 250, 140, "det"]]])
        frames.append([P[120], [[280, 140, 300, 170, "det"]]])            # a small box
        table = dict(schema="openveil.boxes/1", video="cam1_clip.mp4", kind="video", fps=25.0, width=320, height=180,
                     time_base=cls.tb, first_pts=P[0], start_local="2024-01-02T10:00:00", params=dict(mask_thr=0.10),
                     chunks=[[0.0, "full", 150, 150]], frames=frames,
                     candidates=[[P[100], [[100, 50, 140, 120, 0.07]]], [P[130], [[10, 10, 30, 30, 0.02]]]])
        with gzip.open(os.path.join(cls.tables, "cam1_clip.json.gz"), "wt", encoding="utf-8") as fh:
            json.dump(table, fh)
        # a table without a video is skipped
        with gzip.open(os.path.join(cls.tables, "other.json.gz"), "wt", encoding="utf-8") as fh:
            json.dump(dict(table, video="nowhere.mp4"), fh)
        cls.out = os.path.join(cls.root, "review")
        cls.m = rb.build_from_tables(cls.tables, cls.videos, cls.out, per_queue=6, spacing_s=2.0, max_width=160, quiet=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_extract_frames(self):
        P = self.pts
        got = rb.extract_frames(os.path.join(self.videos, "cam1_clip.mp4"), [P[3], P[70] + 1, P[140], P[-1] + 10 ** 7])
        self.assertEqual(got[P[3]][0], P[3])
        self.assertEqual(got[P[70] + 1][0], P[71])                          # between frames: the next frame
        self.assertNotIn(P[-1] + 10 ** 7, got)                             # past the end
        self.assertAlmostEqual(float(got[P[140]][1][:40, :60].mean()), 1.5 * 140, delta=4)

    def test_queues_and_frames(self):
        m = self.m
        self.assertEqual(rs.validate_manifest(m), [])
        q = {x["id"]: x for x in m["queues"]}
        self.assertEqual(set(q), {"mosaic", "giant", "below", "random"})
        # mosaic: largest share first; the giant frames' mosaic (after the rule) is smaller than the big box
        first = m["frames"][q["mosaic"]["entries"][0]["fid"]]
        self.assertIn(first["pts"], self.pts[10:16])
        self.assertEqual(len(q["giant"]["entries"]), 1)                     # one episode
        g = m["frames"][q["giant"]["entries"][0]["fid"]]
        self.assertIn(g["pts"], self.pts[60:64])
        self.assertEqual(len(g["persons"]), 3)
        self.assertEqual(len(g["applied"]), 2)
        self.assertIn("share", g["giant"])
        b = m["frames"][q["below"]["entries"][0]["fid"]]
        self.assertEqual(b["pts"], self.pts[100])
        self.assertEqual(b["cands"][0][4], 0.07)
        self.assertEqual(len(q["random"]["entries"]), len({e["fid"] for e in q["random"]["entries"]}))
        for f in m["frames"].values():
            self.assertEqual((f["width"], f["height"], f["src_width"]), (160, 90, 320))
            self.assertTrue(f["path"].startswith("frames/"))
            self.assertIn("2024-01-02 10:00:", f["clock"])
            self.assertTrue(os.path.isfile(os.path.join(self.out, f["path"])))
        self.assertEqual(first["persons"], [[20.0, 10.0, 100.0, 85.0]])       # scaled by 1/2

    def test_frames_are_the_right_frames(self):
        from PIL import Image
        import numpy as np
        for f in self.m["frames"].values():
            i = self.pts.index(f["pts"])
            with Image.open(os.path.join(self.out, f["path"])) as im:
                corner = np.asarray(im.convert("RGB"))[:18, :28].mean()
            self.assertAlmostEqual(float(corner), 1.5 * i, delta=6, msg=f["id"])

    def test_rebuild_is_stable_and_export_maps_back(self):
        from openveil.review.export import export_review
        out = os.path.join(self.root, "review2")
        shutil.copytree(self.out, out)
        m2 = rb.build_from_tables(self.tables, self.videos, out, per_queue=6, spacing_s=2.0, max_width=160, quiet=True)
        self.assertEqual({k: v["key"] for k, v in m2["frames"].items()}, {k: v["key"] for k, v in self.m["frames"].items()})
        fid = next(e["fid"] for e in next(q for q in m2["queues"] if q["id"] == "mosaic")["entries"])
        f = m2["frames"][fid]
        with open(os.path.join(out, "decisions.jsonl"), "w") as fh:
            fh.write(json.dumps(dict(fid=fid, key=f["key"], kind=f["kind"], queue="mosaic", status="confirmed",
                                     persons=[[130, 60, 150, 85]], note="", ms=1, ts="2026-01-01T00:00:00.000")) + "\n")
        res = export_review(out, quiet=True)
        ov = res["overrides"]["cam1_clip.mp4"]
        self.assertEqual(ov["delete"], [[f["table_t"], [40.0, 20.0, 200.0, 170.0]]])   # source pixels
        self.assertEqual(ov["add"], [[f["t"], f["t"], [260.0, 120.0, 300.0, 170.0]]])


if __name__ == "__main__":
    unittest.main()
