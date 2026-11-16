# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Synthetic review demo: build, validate, demo guard refusals, reset, and a save + export round trip."""
import http.client
import importlib.util
import json
import os
import shutil
import tempfile
import threading
import unittest

from openveil.review import server as rs

HAS_PIL = importlib.util.find_spec("PIL") is not None


@unittest.skipUnless(HAS_PIL, "Pillow is not installed")
class TestDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from openveil.review import demo
        cls.demo = demo
        cls.root = tempfile.mkdtemp(prefix="ov_demo_")
        cls.d = os.path.join(cls.root, "demo")
        demo.make_demo(cls.d)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def manifest(self):
        with open(os.path.join(self.d, "manifest.json")) as fh:
            return json.load(fh)

    def test_manifest_is_valid_and_self_contained(self):
        m = self.manifest()
        self.assertEqual(rs.validate_manifest(m), [])
        self.assertIs(m["meta"]["demo"], True)
        self.assertEqual(m["meta"]["banner"], "Demo data: synthetic frames")
        self.assertEqual({q["id"] for q in m["queues"]}, {"false_alarms", "missed_person", "loose_box", "random_audit", "all"})
        for f in m["frames"].values():
            self.assertFalse(os.path.isabs(f["path"]))
            self.assertTrue(f["path"].endswith(".png"))
            self.assertIn(f["kind"], m["meta"]["kinds"])
            with open(os.path.join(self.d, f["path"]), "rb") as fh:
                self.assertEqual(fh.read(8), b"\x89PNG\r\n\x1a\n")
        rs.check_demo(self.d, m)                                          # does not raise
        for name, st in m["meta"]["states"].items():
            self.assertIn(st["fid"], m["frames"], name)

    def test_designed_cases(self):
        m = self.manifest()
        with open(os.path.join(self.d, "demo_truth.json")) as fh:
            truth = json.load(fh)
        f = m["frames"]
        # missed person: empty proposal, a 0.31 candidate on the figure, a small edge figure without a candidate
        self.assertEqual(f["M001"]["persons"], [])
        self.assertIn(0.31, [c[4] for c in f["M001"]["cands"]])
        self.assertEqual(len(truth["M001"]), 2)
        self.assertGreaterEqual(truth["M001"][1][2], 950)                  # touches the right edge
        # giant box around two people: the giant rule (0.55) drops it
        from openveil.review.build import giant_split
        kept, dropped = giant_split(f["F002"]["persons"], 960, 540)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(len(kept), 2)
        # loose box: about 40% larger than the truth
        t, b = truth["L001"][0], f["L001"]["persons"][0]
        ratio = ((b[2] - b[0]) * (b[3] - b[1])) / ((t[2] - t[0]) * (t[3] - t[1]))
        self.assertTrue(1.3 < ratio < 2.1, ratio)
        # random audit: mostly person-free
        self.assertGreaterEqual(sum(1 for fid in ("A001", "A002", "A003", "A004") if not truth[fid]), 4)

    def test_seed_progress(self):
        m = self.manifest()
        with open(os.path.join(self.d, "seed_decisions.jsonl")) as fh:
            seeds = [json.loads(x) for x in fh if x.strip()]
        self.assertTrue(seeds)
        for s in seeds:
            self.assertTrue(rs.valid_decision(s, m["frames"]))
        store = rs.make_server(self.d, port=0, demo=True, reset=True)
        try:
            st = store.store.stats()
            self.assertGreater(st["false_alarms"]["done"], 0)
            self.assertLess(st["false_alarms"]["done"], st["false_alarms"]["n"])
            self.assertEqual(st["random_audit"]["unsure"], 1)
        finally:
            store.server_close()

    def test_rebuild_into_same_folder(self):
        self.demo.make_demo(self.d)                                       # a demo folder may be rebuilt

    def test_make_demo_refuses_foreign_folders(self):
        other = os.path.join(self.root, "real")
        os.makedirs(other)
        with open(os.path.join(other, "notes.txt"), "w") as fh:
            fh.write("x")
        with self.assertRaises(rs.DemoGuardError):
            self.demo.make_demo(other)
        real = os.path.join(self.root, "real2")
        os.makedirs(real)
        with open(os.path.join(real, "manifest.json"), "w") as fh:
            json.dump(dict(meta={}, frames={}, queues=[]), fh)
        with self.assertRaises(rs.DemoGuardError):
            self.demo.make_demo(real)

    def test_demo_guard_refusals(self):
        bad = os.path.join(self.root, "guard")
        shutil.copytree(self.d, bad)
        m = self.manifest()
        # 1) not a demo manifest
        m1 = json.loads(json.dumps(m))
        m1["meta"]["demo"] = False
        with open(os.path.join(bad, "manifest.json"), "w") as fh:
            json.dump(m1, fh)
        with self.assertRaises(rs.DemoGuardError):
            rs.make_server(bad, port=0, demo=True)
        with self.assertRaises(rs.DemoGuardError):
            rs.make_server(bad, port=0, demo=True, reset=True)
        # 2) a frame outside the folder (relative escape and absolute)
        for path in ("../demo/frames/F001.png", os.path.join(self.d, "frames", "F001.png")):
            m2 = json.loads(json.dumps(m))
            m2["frames"]["F001"]["path"] = path
            with open(os.path.join(bad, "manifest.json"), "w") as fh:
                json.dump(m2, fh)
            with self.assertRaises(rs.DemoGuardError):
                rs.make_server(bad, port=0, demo=True)
            srv = rs.make_server(bad, port=0)                              # normal mode serves it
            srv.server_close()
        # 3) --reset without --demo
        with self.assertRaises(rs.DemoGuardError):
            rs.make_server(self.d, port=0, reset=True)
        # 4) the CLI exits with the message
        with self.assertRaises(SystemExit):
            rs.main([bad, "--demo", "--port", "0"])

    def test_reset_restores_seed(self):
        d = os.path.join(self.root, "reset")
        shutil.copytree(self.d, d)
        with open(os.path.join(d, "decisions.jsonl"), "w") as fh:
            fh.write("garbage\n")
        rs.reset_decisions(d)
        with open(os.path.join(d, "decisions.jsonl")) as fh, open(os.path.join(d, "seed_decisions.jsonl")) as gh:
            self.assertEqual(fh.read(), gh.read())

    def test_save_and_export_round_trip(self):
        from openveil.review.export import export_review
        d = os.path.join(self.root, "roundtrip")
        shutil.copytree(self.d, d)
        srv = rs.make_server(d, port=0, demo=True, reset=True)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            with open(os.path.join(d, "demo_truth.json")) as fh:
                truth = json.load(fh)

            def post(rec):
                c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
                c.request("POST", "/api/decision", body=json.dumps(rec), headers={"X-Review-Tool": "1", "Content-Type": "application/json"})
                r = c.getresponse()
                out = r.status, json.loads(r.read())
                c.close()
                return out
            # fix three frames to the truth: delete the giant box, adopt the missed people, fix the loose box
            for fid, q in (("F002", "false_alarms"), ("M001", "missed_person"), ("L001", "loose_box")):
                st, r = post(dict(fid=fid, persons=truth[fid], status="confirmed", queue=q, base=None))
                self.assertEqual(st, 200, r)
            c = http.client.HTTPConnection("127.0.0.1", srv.port, timeout=10)
            c.request("GET", "/img/F002")
            r = c.getresponse()
            self.assertEqual(r.getheader("Content-Type"), "image/png")
            r.read()
            c.close()
        finally:
            srv.shutdown()
            srv.server_close()
        res = export_review(d, quiet=True)
        s = res["summary"]
        self.assertEqual(s["false_alarms"]["frames"], 3)                   # F001, F004 seeded + F002
        f2 = s["false_alarms"]
        self.assertEqual(f2["removed"], 2)                                 # calf box (F001 seed) + giant box (F002)
        self.assertEqual(s["missed_person"]["added"], 2)
        self.assertEqual(s["missed_person"]["free_to_people"], 1)
        self.assertEqual(s["loose_box"]["moved"], 1)
        self.assertEqual(s["random_audit"]["unsure"], 1)
        self.assertIn("demo:M001", res["labels"])
        self.assertNotIn("demo:A003", res["labels"])                      # unsure: left out
        self.assertEqual(res["unsure"][0]["key"], "demo:A003")
        ov = res["overrides"]
        self.assertEqual(len(ov["demo_pen_a"]["add"]), 3)                   # 2 missed people + the corrected loose box
        self.assertTrue(os.path.exists(os.path.join(d, "overrides.json")))


if __name__ == "__main__":
    unittest.main()
