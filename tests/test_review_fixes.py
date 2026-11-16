# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Review cases that must not uncover a person the reviewer confirmed (synthetic data only):

* keeping a giant box covers the whole giant-rule episode, not only +-2 s;
* moving or resizing a box never shortens the masked time;
* rebuilding a review folder keeps the frames that already have a decision;
* the image review matches names like ``apply images`` and refuses boxes that match no image;
* the server refuses (does not silently drop) a box outside the frame, accepts extra fields
  after the four coordinates and reports a missing review folder as such;
* a review folder ignores itself in git; detect's ``summary.json`` is not read as a table;
* a table already filtered by an older ``box_rules`` record without ``keep_kinds`` is accepted.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from openveil.apply.plan import MaskPlan
from openveil.review import build as rb
from openveil.review.export import build_overrides
from openveil.table import BoxTable, apply_overrides

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:  # pragma: no cover
    HAVE_PIL = False


def _covered(plan, t, x, y):
    return any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in plan.boxes_at(t))


class TestOverrides(unittest.TestCase):
    def test_kept_giant_box_covers_the_episode(self):
        W, H = 1000, 500
        G, A, B = [0, 0, 1000, 300], [100, 100, 200, 280], [700, 100, 800, 280]
        T = BoxTable(video="v.mp4", width=W, height=H, fps=25.0, time_base=(1, 1000), first_pts=0,
                     frames=[[t * 1000, [G + ["det"], A + ["det"], B + ["det"]]] for t in range(10, 31)])
        d = T.to_dict()
        rows, _ = rb.table_rows(d)
        ep = rb.firing_episodes(d, rows)[0]
        f = dict(id="G0000", kind="giant", key="v.mp4:10000", video="v.mp4", width=W, height=H, persons=[G, A, B],
                 applied=[A, B], t=10.0, table_t=10.0, giant=rb._giant_info(ep))
        ov = build_overrides({"frames": {"G0000": f}}, {"G0000": dict(fid="G0000", status="confirmed", persons=[G, A, B])})
        self.assertEqual(ov["v.mp4"]["keep"], [[10.0, 30.0]])
        plan = MaskPlan(apply_overrides(T, ov)[0])
        for t in (10.0, 12.5, 20.0, 25.0, 30.0):
            self.assertTrue(_covered(plan, t, 500, 20), t)
        # deleting the giant box instead gives no keep range
        ov = build_overrides({"frames": {"G0000": f}}, {"G0000": dict(fid="G0000", status="confirmed", persons=[A, B])})
        self.assertNotIn("v.mp4", ov)

    def test_resized_box_never_shortens_the_masked_time(self):
        box = [100, 100, 200, 300]
        T = BoxTable(video="v", width=640, height=480, fps=10.0, time_base=(1, 10), first_pts=0,
                     frames=[[int(t * 10), [box + ["det"]]] for t in (18, 19, 20, 21, 22)])
        f = dict(id="A0000", kind="random", key="v:200", video="v", width=640, height=480, persons=[box], applied=[box],
                 t=20.0, table_t=20.0)
        bigger = [100, 100, 200, 370]                                    # the feet added: IoU 0.74
        ov = build_overrides({"frames": {"A0000": f}}, {"A0000": dict(fid="A0000", status="confirmed", persons=[bigger])})
        self.assertEqual(ov["v"]["delete"], [])
        self.assertEqual(len(ov["v"]["add"]), 1)
        before, after = MaskPlan(T), MaskPlan(apply_overrides(T, ov)[0])
        for i in range(140, 260):
            t = i / 10
            if _covered(before, t, 150, 200):
                self.assertTrue(_covered(after, t, 150, 200), t)


class TestBoxRulesRecord(unittest.TestCase):
    def test_table_filtered_by_an_older_record(self):
        from openveil.rules import apply_rules_to_table
        t = dict(video="v", width=1280, height=720, fps=25, time_base=[1, 25], first_pts=0,
                 frames=[[0, [[0, 0, 1280, 720, "det"], [0, 0, 10, 10, "det"], [20, 20, 30, 30, "det"]]]],
                 params={"box_rules": dict(rule="giant", min_share=0.55, min_inner=2, inner_frac=0.8)})
        new, _ = apply_rules_to_table(t, 0.55)
        self.assertEqual(len(new["frames"][0][1]), 2)
        MaskPlan(t)                                                      # no error
        with self.assertRaises(ValueError):                              # other parameters are still refused
            apply_rules_to_table(t, 0.40)
        t["frames"][0][1].append([0, 0, 5, 5, "manual"])                 # now keep_kinds matters: refused
        with self.assertRaises(ValueError):
            apply_rules_to_table(t, 0.55)


@unittest.skipUnless(HAVE_PIL, "Pillow needed")
class TestImageReview(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="ov_rfix_")
        self.img = os.path.join(self.root, "images")
        os.makedirs(os.path.join(self.img, "sub"))
        self.out = os.path.join(self.root, "review")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _images(self, names):
        for i, n in enumerate(names):
            Image.new("RGB", (200, 100), (10 * i, 90, 140)).save(os.path.join(self.img, n))

    def _boxes(self, data):
        p = os.path.join(self.root, f"b{len(os.listdir(self.root))}.json")
        with open(p, "w") as fh:
            json.dump(data, fh)
        return p

    def test_file_name_match_and_unmatched_boxes(self):
        self._images(["sub/a.png"])
        m = rb.build_from_images(self.img, self._boxes({"a.png": [[50, 20, 120, 90]]}), self.out, quiet=True)
        f = next(iter(m["frames"].values()))
        self.assertEqual((f["image"], f["persons"]), ("sub/a.png", [[50.0, 20.0, 120.0, 90.0]]))
        with open(os.path.join(self.out, ".gitignore")) as fh:
            self.assertIn("*", fh.read().split())
        with self.assertRaises(ValueError):
            rb.build_from_images(self.img, self._boxes({"other/b.png": [[1, 1, 9, 9]]}), self.out + "2", quiet=True)

    def test_rebuild_keeps_reviewed_frames(self):
        from openveil.review.export import export_review
        from openveil.review.server import Store
        self._images([f"i{k}.png" for k in range(6)])
        boxes = self._boxes({f"i{k}.png": [] for k in range(6)})
        m1 = rb.build_from_images(self.img, boxes, self.out, per_queue=2, quiet=True)
        reviewed = [e["fid"] for q in m1["queues"] if q["id"] == "random" for e in q["entries"]]
        st = Store(self.out)
        for fid in reviewed:
            st.save(dict(fid=fid, queue="random", status="confirmed", persons=[[10, 10, 60, 60]]))
        self._images([f"j{k}.png" for k in range(12)])
        boxes2 = self._boxes({})
        m2 = rb.build_from_images(self.img, boxes2, self.out, per_queue=2, seed=3, quiet=True)
        redrawn = {e["fid"] for q in m2["queues"] if q["id"] == "random" for e in q["entries"]}
        earlier = {e["fid"] for q in m2["queues"] if q["id"] == "earlier" for e in q["entries"]}
        self.assertEqual(set(reviewed) - redrawn, earlier)
        self.assertTrue(earlier)                                         # the test does exercise the case
        res = export_review(self.out, quiet=True)
        names = {m1["frames"][fid]["image"] for fid in reviewed}
        for n in names:
            self.assertEqual(res["reviewed_boxes"][n], [[10.0, 10.0, 60.0, 60.0, "manual"]], n)

    def test_server_box_checks(self):
        from openveil.review.server import Store, serve
        self._images(["sub/a.png"])
        m = rb.build_from_images(self.img, self._boxes({"sub/a.png": [[50, 20, 120, 90]]}), self.out, quiet=True)
        fid = next(iter(m["frames"]))
        st = Store(self.out)
        rec = st.save(dict(fid=fid, queue="random", status="confirmed", persons=[[50, 20, 120, 90, 0.9]]))
        self.assertEqual(rec["persons"], [[50.0, 20.0, 120.0, 90.0]])   # extra fields are ignored
        with self.assertRaises(ValueError):                              # outside the 200 x 100 frame
            st.save(dict(fid=fid, queue="random", status="confirmed", persons=[[300, 150, 400, 190]], base=rec["ts"]))
        with self.assertRaises(SystemExit) as cm:
            serve(os.path.join(self.root, "nothing_here"), 0)
        self.assertIn("not a review folder", str(cm.exception))


class TestTablesFolder(unittest.TestCase):
    def test_summary_json_is_not_read_as_a_table(self):
        d = tempfile.mkdtemp(prefix="ov_tf_")
        try:
            for n in ("summary.json", "overrides.json"):
                with open(os.path.join(d, n), "w") as fh:
                    json.dump({"done": 1}, fh)
            logs = []
            with self.assertRaises(SystemExit):                          # no table at all, but nothing "skipped"
                _build_logged(d, logs)
            self.assertFalse(any("summary.json" in m for m in logs), logs)
        finally:
            shutil.rmtree(d, ignore_errors=True)


def _build_logged(d, logs):
    import builtins
    real = builtins.print
    builtins.print = lambda *a, **k: logs.append(" ".join(map(str, a)))
    try:
        rb.build_from_tables(d, d, os.path.join(d, "rev"))
    finally:
        builtins.print = real


if __name__ == "__main__":
    unittest.main()
