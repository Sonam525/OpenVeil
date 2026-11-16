# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Review export: latest decision wins then unsure is filtered, kept/removed/added/moved counts, overrides, reviewed boxes."""
import json
import os
import shutil
import tempfile
import unittest

from openveil.review import export as ex


def frame(fid, persons, **kw):
    return dict(dict(id=fid, kind="mosaic", key=f"vid:{fid}", width=100, height=50, path=f"frames/{fid}.jpg", persons=persons, cands=[]), **kw)


def decision(fid, persons, status="confirmed", ts="2026-01-01T00:00:00.000", note=""):
    return dict(fid=fid, key=f"vid:{fid}", kind="mosaic", queue="q", status=status, persons=persons, note=note, ms=2000, ts=ts)


class TestCompare(unittest.TestCase):
    def test_compare_counts(self):
        before = [[0, 0, 10, 10], [20, 0, 30, 10], [50, 0, 60, 10]]
        after = [[0, 0, 10, 10], [21, 0, 31.5, 10], [80, 0, 90, 10]]   # kept, moved, (removed + added)
        self.assertEqual(ex.compare(before, after), (2, 1, 1, 1))
        self.assertEqual(ex.compare([], [[0, 0, 1, 1]]), (0, 0, 1, 0))
        self.assertEqual(ex.compare([[0, 0, 1, 1]], []), (0, 1, 0, 0))

    def test_greedy_order(self):
        # the first proposal box takes its best match even if the second would match it better: same as production
        before = [[0, 0, 10, 10], [2, 0, 12, 10]]
        after = [[1, 0, 11, 10]]
        pairs, removed, added = ex.match(before, after)
        self.assertEqual([p[0] for p in pairs], [0])
        self.assertEqual(removed, [1])
        self.assertEqual(added, [])


class TestExport(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ov_exp_")
        frames = {
            # video frame, review image downscaled by 2 from a 200 x 100 source
            "M0000": frame("M0000", [[0, 0, 10, 10], [20, 0, 30, 10]], video="v1", t=4.0, table_t=4.0, src_width=200, src_height=100),
            # giant-rule frame: the apply step would keep only the inner boxes
            "G0000": frame("G0000", [[0, 0, 90, 45], [10, 10, 20, 20], [30, 10, 40, 20]], kind="giant", video="v1", t=9.5,
                           table_t=9.5, applied=[[10, 10, 20, 20], [30, 10, 40, 20]]),
            # random frame whose proposal came from a row 0.2 s away
            "A0000": frame("A0000", [[40, 10, 50, 30]], kind="random", video="v2", t=1.2, table_t=1.0),
            # confirmed, then marked unsure: unsure wins
            "A0001": frame("A0001", [[0, 0, 10, 10]], kind="random", video="v2", t=3.0, table_t=3.0),
            "I0000": frame("I0000", [[1, 1, 9, 9]], kind="image", image="x.jpg"),
            "I0001": frame("I0001", [[1, 1, 9, 9]], kind="image", image="y.jpg"),
            "I0002": frame("I0002", [[1, 1, 9, 9]], kind="image", image="z.jpg"),
        }
        queues = [dict(id="q", title="all", desc="", entries=[dict(fid=f, np=len(frames[f]["persons"]), note="") for f in frames]),
                  dict(id="empty", title="none", desc="", entries=[])]
        self.m = dict(meta=dict(title="t"), frames=frames, queues=queues)
        with open(os.path.join(self.d, "manifest.json"), "w") as fh:
            json.dump(self.m, fh)
        lines = [
            decision("M0000", [[0, 0, 10, 10], [60, 20, 70, 40]]),                     # one kept, one removed, one added
            decision("G0000", [[10, 10, 20, 20], [30, 10, 40, 20]]),                   # giant deleted: nothing to override
            decision("A0000", [], ts="2026-01-01T00:00:01.000"),
            decision("A0000", [[40, 10, 50, 30]], ts="2026-01-01T00:00:02.000"),        # latest wins: kept
            decision("A0001", [], ts="2026-01-01T00:00:01.000"),
            decision("A0001", [[0, 0, 10, 10], [70, 0, 80, 10]], status="unsure", ts="2026-01-01T00:00:02.000"),
            decision("I0000", []),
            decision("I0001", [[50, 20, 60, 30]], status="unsure"),
        ]
        with open(os.path.join(self.d, "decisions.jsonl"), "w") as fh:
            fh.write("".join(json.dumps(x) + "\n" for x in lines))
        self.res = ex.export_review(self.d, quiet=True)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_latest_wins_then_unsure_filtered(self):
        labels = self.res["labels"]
        self.assertEqual(labels["vid:A0000"]["persons"], [[40, 10, 50, 30]])
        self.assertNotIn("vid:A0001", labels)                               # confirmed first, unsure last
        self.assertNotIn("vid:I0001", labels)
        self.assertEqual({u["fid"] for u in self.res["unsure"]}, {"A0001", "I0001"})
        self.assertEqual(set(ex.latest_decisions(self.d)), {"M0000", "G0000", "A0000", "I0000"})
        self.assertEqual(len(ex.latest_decisions(self.d, include_unsure=True)), 6)
        with open(os.path.join(self.d, "human_labels.json")) as fh:
            self.assertEqual(json.load(fh), labels)

    def test_summary_counts(self):
        s = self.res["summary"]["q"]
        self.assertEqual(s["frames"], 6)
        self.assertEqual(s["unsure"], 2)
        # M0000: kept 1 removed 1 added 1; G0000: kept 2 removed 1; A0000: kept 1; A0001: kept 1 added 1;
        # I0000: removed 1; I0001: removed 1 added 1
        self.assertEqual((s["kept"], s["removed"], s["added"], s["moved"]), (5, 4, 3, 0))
        self.assertEqual(s["people_to_none"], 1)
        self.assertEqual(s["frames_changed"], 5)
        self.assertEqual(s["median_s"], 2.0)
        self.assertEqual(self.res["summary"]["empty"]["frames"], 0)

    def test_overrides(self):
        ov = self.res["overrides"]
        # M0000: removed box back in source pixels (x2), added box too
        self.assertEqual(ov["v1"]["delete"], [[4.0, [40.0, 0.0, 60.0, 20.0]]])
        self.assertEqual(ov["v1"]["add"], [[4.0, 4.0, [120.0, 40.0, 140.0, 80.0]]])
        # A0001 is unsure: it only adds, never deletes
        self.assertEqual(ov["v2"], {"add": [[3.0, 3.0, [70.0, 0.0, 80.0, 10.0]]], "delete": []})
        with open(os.path.join(self.d, "overrides.json")) as fh:
            self.assertEqual(json.load(fh), ov)

    def test_giant_kept_becomes_add(self):
        with open(os.path.join(self.d, "decisions.jsonl"), "a") as fh:
            fh.write(json.dumps(decision("G0000", [[0, 0, 90, 45], [10, 10, 20, 20], [30, 10, 40, 20]], ts="2026-01-01T00:00:09.000")) + "\n")
        res = ex.export_review(self.d, overrides_path=os.path.join(self.d, "o.json"), quiet=True)
        self.assertIn([9.5, 9.5, [0.0, 0.0, 90.0, 45.0]], res["overrides"]["v1"]["add"])
        self.assertTrue(os.path.exists(os.path.join(self.d, "o.json")))

    def test_delete_uses_table_row_time(self):
        with open(os.path.join(self.d, "decisions.jsonl"), "a") as fh:
            fh.write(json.dumps(decision("A0000", [], ts="2026-01-01T00:00:09.000")) + "\n")
        res = ex.export_review(self.d, quiet=True)
        self.assertEqual(res["overrides"]["v2"]["delete"], [[1.0, [40.0, 10.0, 50.0, 30.0]]])

    def test_reviewed_boxes_for_images(self):
        rb = self.res["reviewed_boxes"]
        self.assertEqual(rb["x.jpg"], [])                                    # confirmed: nobody
        self.assertEqual(rb["y.jpg"], [[1, 1, 9, 9], [50, 20, 60, 30, "manual"]])     # unsure: proposal + added (manual)
        self.assertEqual(rb["z.jpg"], [[1, 1, 9, 9]])                        # not reviewed: proposal
        self.assertNotIn("I0000", json.dumps(self.res["overrides"]))         # images never go into the video overrides


if __name__ == "__main__":
    unittest.main()
