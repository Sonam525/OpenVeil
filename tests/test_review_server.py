# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Review server: endpoints, security headers, Host/header checks, the stale-save guard (standard library only)."""
import http.client
import json
import os
import shutil
import tempfile
import threading
import unittest

from openveil.review import server as rs

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 24


def write_review(d, meta=None, outside=None):
    """A small review folder: three frames (png, jpg, webp), one gif, one missing image, two queues."""
    os.makedirs(os.path.join(d, "frames"), exist_ok=True)
    for name, data in (("a.png", PNG), ("b.jpg", JPG), ("c.webp", WEBP), ("d.gif", b"GIF89a")):
        with open(os.path.join(d, "frames", name), "wb") as fh:
            fh.write(data)

    def fr(fid, path, persons):
        return dict(id=fid, kind="image", key=f"k:{fid}", width=200, height=100, path=path, persons=persons,
                    cands=[[10, 10, 50, 60, 0.3, -1]], image=f"{fid}.png")
    frames = {
        "F1": fr("F1", "frames/a.png", [[10, 10, 50, 60]]),
        "F2": fr("F2", "frames/b.jpg", []),
        "F3": fr("F3", "frames/c.webp", []),
        "F4": fr("F4", "frames/d.gif", []),
        "F5": fr("F5", "frames/missing.png", []),
    }
    if outside:
        frames["F6"] = fr("F6", outside, [])
    queues = [dict(id="q1", title="<b>Queue</b> one", desc="first", entries=[dict(fid=f, np=len(frames[f]["persons"]), note="")
                                                                            for f in frames]),
              dict(id="q2", title="two", desc="", entries=[dict(fid="F2", np=0, note="")])]
    m = dict(meta=meta if meta is not None else dict(title="Test"), frames=frames, queues=queues)
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        json.dump(m, fh)
    return m


class RunningServer:
    def __init__(self, d, **kw):
        self.srv = rs.make_server(d, port=0, **kw)
        self.port = self.srv.port
        self.th = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.th.start()

    def request(self, method, path, body=None, headers=None, host=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            if host is None:
                c.request(method, path, body=body, headers=headers or {})
            else:
                c.putrequest(method, path, skip_host=True)
                c.putheader("Host", host)
                for k, v in (headers or {}).items():
                    c.putheader(k, v)
                if body is not None:
                    c.putheader("Content-Length", str(len(body)))
                c.endheaders(body)
            r = c.getresponse()
            return r.status, dict(r.getheaders()), r.read()
        finally:
            c.close()

    def get_json(self, path):
        status, _, body = self.request("GET", path)
        return status, json.loads(body)

    def post(self, rec, headers=None):
        h = {"Content-Type": "application/json", "X-Review-Tool": "1"}
        h.update(headers or {})
        status, _, body = self.request("POST", "/api/decision", body=json.dumps(rec).encode(), headers=h)
        return status, json.loads(body)

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


class TestServer(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ov_rev_")
        write_review(self.d)
        self.s = RunningServer(self.d)

    def tearDown(self):
        self.s.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_page_and_security_headers(self):
        status, h, body = self.s.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertTrue(h["Content-Type"].startswith("text/html"))
        self.assertEqual(h["X-Frame-Options"], "DENY")
        self.assertEqual(h["Cross-Origin-Resource-Policy"], "same-origin")
        self.assertIn("frame-ancestors 'none'", h["Content-Security-Policy"])
        html = body.decode()
        self.assertNotIn(rs.NONCE_PLACEHOLDER, html)
        nonce = h["Content-Security-Policy"].split("'nonce-")[1].split("'")[0]
        self.assertIn(f'nonce="{nonce}"', html)

    def test_manifest_queue_frame(self):
        status, m = self.s.get_json("/api/manifest")
        self.assertEqual(status, 200)
        self.assertEqual(m["meta"]["banner"], rs.REAL_DATA_BANNER)      # default banner for real data
        self.assertEqual([q["id"] for q in m["queues"]], ["q1", "q2"])
        self.assertEqual(m["stats"]["q1"], dict(n=5, done=0, unsure=0))
        status, q = self.s.get_json("/api/queue/q1")
        self.assertEqual(status, 200)
        self.assertEqual(len(q), 5)
        self.assertIsNone(q[0]["status"])
        status, f = self.s.get_json("/api/frame/F1")
        self.assertEqual(status, 200)
        self.assertNotIn("path", f)
        self.assertIsNone(f["decision"])
        self.assertEqual(self.s.get_json("/api/queue/nope")[0], 404)
        self.assertEqual(self.s.get_json("/api/frame/nope")[0], 404)
        self.assertEqual(self.s.request("GET", "/../manifest.json")[0], 404)

    def test_image_content_types(self):
        for fid, ctype, data in (("F1", "image/png", PNG), ("F2", "image/jpeg", JPG), ("F3", "image/webp", WEBP)):
            status, h, body = self.s.request("GET", f"/img/{fid}")
            self.assertEqual(status, 200, fid)
            self.assertEqual(h["Content-Type"], ctype)
            self.assertEqual(h["Cross-Origin-Resource-Policy"], "same-origin")
            self.assertEqual(body, data)
        self.assertEqual(self.s.request("GET", "/img/F4")[0], 404)        # unsupported type
        self.assertEqual(self.s.request("GET", "/img/F5")[0], 404)        # missing file
        self.assertEqual(self.s.request("GET", "/img/zzz")[0], 404)

    def test_host_check(self):
        self.assertEqual(self.s.request("GET", "/api/manifest", host=f"evil.example:{self.s.port}")[0], 403)
        self.assertEqual(self.s.request("GET", "/api/manifest", host="127.0.0.1:1")[0], 403)
        self.assertEqual(self.s.request("GET", "/api/manifest", host=f"localhost:{self.s.port}")[0], 200)
        body = json.dumps(dict(fid="F1", persons=[])).encode()
        st = self.s.request("POST", "/api/decision", body=body, host=f"evil.example:{self.s.port}",
                            headers={"X-Review-Tool": "1", "Content-Type": "application/json"})[0]
        self.assertEqual(st, 403)

    def test_post_header_and_origin_checks(self):
        rec = dict(fid="F1", persons=[], status="confirmed")
        self.assertEqual(self.s.post(rec, headers={"X-Review-Tool": "0"})[0], 403)
        self.assertEqual(self.s.post(rec, headers={"Origin": "http://evil.example"})[0], 403)
        self.assertEqual(self.s.request("OPTIONS", "/api/decision")[0], 403)
        self.assertEqual(self.s.post(rec, headers={"Origin": f"http://127.0.0.1:{self.s.port}"})[0], 200)

    def test_decision_save_clamp_and_stale_guard(self):
        st, r = self.s.post(dict(fid="F1", persons=[[-5, 10, 120, 300], [1, 1, 2, 2]], status="confirmed", note="x" * 900,
                                 ms=1e12, queue="q1", base=None))
        self.assertEqual(st, 200)
        saved = r["saved"]
        self.assertEqual(saved["persons"], [[0.0, 10.0, 120.0, 100.0]])  # clamped; the 1-px box dropped
        self.assertEqual(len(saved["note"]), rs.MAX_NOTE)
        self.assertEqual(saved["ms"], rs.MAX_MS)
        self.assertEqual(r["stats"]["q1"]["done"], 1)
        # a second tab that loaded the frame before the save sends the old base -> 409
        st, r2 = self.s.post(dict(fid="F1", persons=[], status="confirmed", base=None))
        self.assertEqual(st, 409)
        st, r3 = self.s.post(dict(fid="F1", persons=[], status="unsure", base=saved["ts"]))
        self.assertEqual(st, 200)
        self.assertNotEqual(r3["saved"]["ts"], saved["ts"])
        self.assertEqual(r3["stats"]["q1"]["unsure"], 1)
        st, f = self.s.get_json("/api/frame/F1")
        self.assertEqual(f["decision"]["status"], "unsure")
        with open(os.path.join(self.d, "decisions.jsonl")) as fh:
            lines = [json.loads(x) for x in fh if x.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[-1]["key"], "k:F1")

    def test_bad_records(self):
        self.assertEqual(self.s.post(dict(fid="nope", persons=[]))[0], 400)
        self.assertEqual(self.s.post(dict(fid="F1", persons="x"))[0], 400)
        self.assertEqual(self.s.post(dict(fid="F1", persons=[[0, 0, 10, 10]] * 101))[0], 400)
        self.assertEqual(self.s.post(dict(fid="F1", persons=[[0, 0, 10]]))[0], 400)
        self.assertEqual(self.s.post(dict(fid="F1", persons=[[0, 0, "nan", 10]]))[0], 400)
        self.assertEqual(self.s.post(["F1"])[0], 400)
        st = self.s.request("POST", "/api/decision", body=b"{not json", headers={"X-Review-Tool": "1"})[0]
        self.assertEqual(st, 400)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="ov_rev_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_torn_and_foreign_lines_are_skipped(self):
        write_review(self.d)
        good = dict(fid="F2", key="k:F2", kind="image", queue="q1", status="confirmed", persons=[], note="", ms=5, ts="2026-01-01T00:00:00.000")
        with open(os.path.join(self.d, "decisions.jsonl"), "w") as fh:
            fh.write(json.dumps(good) + "\n")
            fh.write(json.dumps(dict(good, key="other")) + "\n")         # another manifest
            fh.write(json.dumps(dict(good, status="maybe")) + "\n")
            fh.write('{"fid": "F1", "key": "k:F1", "sta')                 # torn last line, no newline
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            s = rs.Store(self.d)
        self.assertEqual(set(s.latest), {"F2"})
        d = s.save(dict(fid="F1", persons=[], status="confirmed"))
        self.assertIsNotNone(d)
        with open(os.path.join(self.d, "decisions.jsonl")) as fh:
            last = fh.read().splitlines()[-1]
        self.assertEqual(json.loads(last)["fid"], "F1")                  # the torn line got its own line ending

    def test_validate_manifest(self):
        m = write_review(self.d)
        self.assertEqual(rs.validate_manifest(m), [])
        bad = json.loads(json.dumps(m))
        bad["frames"]["F1"]["width"] = 0
        bad["queues"][0]["id"] = "a/b"
        bad["queues"][1]["entries"].append(dict(fid="nope"))
        problems = rs.validate_manifest(bad)
        self.assertEqual(len(problems), 3, problems)
        with open(os.path.join(self.d, "manifest.json"), "w") as fh:
            json.dump(bad, fh)
        with self.assertRaises(ValueError):
            rs.Store(self.d)

    def test_absolute_path_outside_folder_is_served_in_normal_mode(self):
        other = tempfile.mkdtemp(prefix="ov_rev_other_")
        try:
            p = os.path.join(other, "x.png")
            with open(p, "wb") as fh:
                fh.write(PNG)
            write_review(self.d, outside=p)
            s = rs.Store(self.d)
            self.assertEqual(s.image("F6")[1], "image/png")
            with self.assertRaises(rs.DemoGuardError):                    # but never in demo mode
                rs.Store(self.d, demo=True)
        finally:
            shutil.rmtree(other, ignore_errors=True)

    def test_banner_from_meta(self):
        write_review(self.d, meta=dict(banner="Custom banner"))
        self.assertEqual(rs.Store(self.d).meta["banner"], "Custom banner")

    def test_atomic_write_leaves_no_temp(self):
        p = os.path.join(self.d, "x.json")
        rs.write_json_atomic(p, {"a": 1})
        rs.write_json_atomic(p, {"a": 2})
        with open(p) as fh:
            self.assertEqual(json.load(fh), {"a": 2})
        self.assertEqual(os.listdir(self.d), ["x.json"])


class TestPage(unittest.TestCase):
    def test_manifest_strings_are_escaped(self):
        with open(rs.APP_HTML, encoding="utf-8") as fh:
            html = fh.read()
        self.assertIn("const esc =", html)
        self.assertIn(rs.NONCE_PLACEHOLDER, html)
        self.assertNotIn("Internal data", html)
        for line in html.splitlines():
            if "innerHTML =" in line and "${" in line:
                # only numbers, esc()-ed text or chip() (which escapes) may be interpolated
                for expr in line.split("${")[1:]:
                    e = expr.split("}")[0]
                    ok = e.startswith(("esc(", "i", "Math.round", "scoreColor", "num(", "all.indexOf", "i + 1", "i === S.sel"))
                    self.assertTrue(ok, f"unescaped interpolation in innerHTML: {e}")


if __name__ == "__main__":
    unittest.main()
