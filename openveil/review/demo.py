# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""A self-contained demo of the review page, drawn from scratch (no photographs, nobody real).

    python -m openveil.review.demo [--out DIR] [--port 8766] [--no-browser] [--build-only]

Draws synthetic pen and field scenes with Pillow (fixed seed): straw or grass, fences, calf-like
ellipses and people as abstract grey silhouettes, some already pixelated so they read as "already
masked". Because the script draws every figure itself it knows the exact box of each one
(``demo_truth.json``). Four queues show the typical review cases:

* **Demo: false alarms** - a green box on a calf, a giant box around two people standing apart, a
  box on a fence post at night, and one frame that is already right;
* **Demo: missed person** - an empty proposal with a dashed candidate (score 0.31) on a figure and a
  small edge figure without any candidate, a person half behind a fence, a second person missed;
* **Demo: loose box** - a box 40% too large, a shifted box, a box that cuts off the legs;
* **Demo: random audit** - mostly person-free frames.

``seed_decisions.jsonl`` holds a few decisions, so the queue selector shows partial progress and one
frame shows the "saved" chip; the server's ``--demo --reset`` restores them.

``meta.states`` names page states for scripted screenshots, opened as ``/?state=<name>``:
``queues``, ``adopt``, ``adopted``, ``giant``, ``magnifier``, ``unsure``, ``saved``, ``loose``.

The demo folder holds only the files written here. ``make_demo`` refuses to write into a folder that
holds anything else (for example a real review folder).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile

from .server import DEMO_PORT, MANIFEST, SEED_DECISIONS, DemoGuardError, serve, write_json_atomic

W, H = 960, 540
TRUTH = "demo_truth.json"
DEMO_FILES = {"frames", MANIFEST, SEED_DECISIONS, "decisions.jsonl", TRUTH, "human_labels.json", "unsure_frames.json",
              "review_summary.json", "overrides.json", "reviewed_boxes.json"}
BLOCK = 15                                                 # mosaic cell of the "already masked" figures (= max(16, H/36) scaled)

KINDS = {
    "false_alarm": dict(label="demo: false alarm", warn=True,
                        hint="A green box sits on something that is not a person. Select it and press X; keep the boxes on people."),
    "missed": dict(label="demo: missed person",
                   hint="Somebody has no green box. Click the dashed candidate to adopt it, or drag on the image to draw a box."),
    "loose": dict(label="demo: loose box",
                  hint="A box is too large, shifted or too small. Select it, then drag it or its handles until it fits the person."),
    "audit": dict(label="demo: random audit",
                  hint="A random frame. Usually nobody is there: press Enter. Look closely for small or distant figures first."),
}

# --- scenes ---------------------------------------------------------------------------------------------------------
# people: (x centre, y of the feet, height, style); style 'grey' = silhouette, 'mosaic' = already pixelated
# refs in persons / cands: ("truth", i), ("calf", j), ("loose", i, f), ("shift", i, fx, fy), ("tight", i, f), ("box", [..])
SCENES = [
    dict(fid="F001", queues=["false_alarms"], kind="false_alarm", bg="straw", fence=True, video="demo_pen_a", t=12.0,
         calves=[(300, 395, 170, "brown"), (720, 410, 150, "pied")], people=[(520, 470, 260, "grey")],
         persons=[("truth", 0), ("calf", 0)], cands=[(("truth", 0), 0.86), (("calf", 0), 0.41), (("calf", 1), 0.09)],
         note="One green box is on a calf. Delete it (click it, press X), keep the box on the person, press Enter."),
    dict(fid="F002", queues=["false_alarms"], kind="false_alarm", bg="straw", fence=False, video="demo_pen_a", t=48.5,
         calves=[(480, 430, 120, "brown")], people=[(220, 485, 300, "grey"), (740, 475, 290, "grey")],
         persons=[("truth", 0), ("truth", 1), ("box", [50, 30, 905, 525])],
         cands=[(("truth", 0), 0.91), (("truth", 1), 0.88), (("box", [50, 30, 905, 525]), 0.34)],
         note="A giant box surrounds two people standing apart (the giant rule would drop it). Delete the giant box, keep the two person boxes."),
    dict(fid="F003", queues=["false_alarms"], kind="false_alarm", bg="straw", fence=True, ir=True, video="demo_pen_a", t=7300.0,
         calves=[(360, 420, 160, "pied")], people=[],
         persons=[("box", [604, 128, 637, 362])], cands=[(("box", [604, 128, 637, 362]), 0.15)],
         note="Night (infrared). The green box is on a fence post. Press N (nobody), then Enter."),
    dict(fid="F004", queues=["false_alarms"], kind="false_alarm", bg="field", fence=True, video="demo_field_b", t=95.0,
         calves=[(250, 420, 150, "brown")], people=[(640, 455, 240, "grey")],
         persons=[("truth", 0)], cands=[(("truth", 0), 0.95)],
         note="This frame is already right: one person, one box. Press Enter."),
    dict(fid="M001", queues=["missed_person"], kind="missed", bg="straw", fence=True, video="demo_pen_a", t=130.0,
         calves=[(220, 420, 160, "pied")], people=[(450, 455, 230, "grey"), (952, 300, 115, "grey")],
         persons=[], cands=[(("loose", 0, 0.04), 0.31), (("calf", 0), 0.07)],
         note="Nothing is green, but the dashed candidate (0.31) is a person: click it. A second, small person stands at the right edge "
              "without any candidate: drag a box around it (hold Z for the magnifier)."),
    dict(fid="M002", queues=["missed_person"], kind="missed", bg="straw", fence=True, fence_front=True, video="demo_pen_a", t=181.0,
         calves=[(700, 430, 170, "brown")], people=[(330, 360, 200, "grey")],
         persons=[], cands=[(("truth", 0), 0.18), (("calf", 0), 0.07)],
         note="A person half hidden behind the fence. Lower the minimum score with [ if the candidate is hidden, then adopt it."),
    dict(fid="M003", queues=["missed_person"], kind="missed", bg="field", fence=False, video="demo_field_b", t=240.0,
         calves=[(760, 440, 150, "pied")], people=[(260, 470, 250, "grey"), (520, 430, 150, "mosaic")],
         persons=[("truth", 0)], cands=[(("truth", 0), 0.77), (("truth", 1), 0.22)],
         note="One person has a box, the second (already pixelated) has only a dashed candidate. Adopt it."),
    dict(fid="L001", queues=["loose_box"], kind="loose", bg="straw", fence=True, video="demo_pen_a", t=305.0,
         calves=[(760, 420, 150, "brown")], people=[(420, 460, 240, "grey")],
         persons=[("loose", 0, 0.40)], cands=[(("loose", 0, 0.40), 0.64)],
         note="The box is 40% too large. Select it and drag its handles to the person."),
    dict(fid="L002", queues=["loose_box"], kind="loose", bg="field", fence=True, video="demo_field_b", t=362.0,
         calves=[(230, 430, 140, "pied")], people=[(600, 455, 220, "mosaic")],
         persons=[("shift", 0, 0.45, 0.10)], cands=[(("shift", 0, 0.45, 0.10), 0.58)],
         note="The box is shifted off the person. Select it and drag it onto the person."),
    dict(fid="L003", queues=["loose_box"], kind="loose", bg="straw", fence=False, video="demo_pen_a", t=410.0,
         calves=[(250, 420, 160, "brown"), (760, 440, 140, "pied")], people=[(520, 470, 250, "grey")],
         persons=[("tight", 0, 0.60)], cands=[(("tight", 0, 0.60), 0.71)],
         note="The box cuts off the legs. Drag the bottom handle down to the feet."),
    dict(fid="A001", queues=["random_audit"], kind="audit", bg="straw", fence=True, video="demo_pen_a", t=530.0,
         calves=[(200, 410, 150, "brown"), (520, 440, 170, "pied"), (800, 400, 130, "brown")], people=[],
         persons=[], cands=[], note=""),
    dict(fid="A002", queues=["random_audit"], kind="audit", bg="field", fence=True, video="demo_field_b", t=610.0,
         calves=[(380, 430, 170, "pied"), (700, 415, 150, "brown")], people=[],
         persons=[], cands=[(("calf", 0), 0.06)], note=""),
    dict(fid="A003", queues=["random_audit"], kind="audit", bg="straw", fence=True, ir=True, video="demo_pen_a", t=9100.0,
         calves=[(300, 430, 170, "brown"), (650, 420, 150, "pied")], people=[],
         persons=[], cands=[], note=""),
    dict(fid="A004", queues=["random_audit"], kind="audit", bg="straw", fence=True, video="demo_pen_a", t=760.0,
         calves=[(600, 420, 180, "brown")], people=[],
         persons=[], cands=[], note=""),
    dict(fid="A005", queues=["random_audit"], kind="audit", bg="field", fence=False, video="demo_field_b", t=845.0,
         calves=[(560, 440, 170, "pied")], people=[(150, 230, 75, "mosaic")],
         persons=[], cands=[], note=""),
]

QUEUES = [
    ("false_alarms", "Demo: false alarms", "Green boxes on things that are not people, and a giant box around two people."),
    ("missed_person", "Demo: missed person", "People without a green box: adopt a dashed candidate or draw a box."),
    ("loose_box", "Demo: loose box", "Boxes that are too large, shifted or too small."),
    ("random_audit", "Demo: random audit", "Random frames, mostly person-free: the only way to find people the pipeline missed."),
    ("all", "Demo: all frames", "Every demo frame in order."),
]

# decisions already "made", so the selector shows partial progress (frame ids -> (status, persons from truth?, note))
SEED = [("F004", "confirmed", "truth", ""), ("F001", "confirmed", "truth", ""), ("A001", "confirmed", "none", ""),
        ("A002", "confirmed", "none", ""), ("A003", "unsure", "none", "dark corner at the back: a person or a shadow?")]

STATES = {
    "queues": dict(queue="false_alarms", fid="F001", filter="all"),
    "adopt": dict(queue="missed_person", fid="M001", filter="all"),
    "adopted": dict(queue="missed_person", fid="M001", filter="all", adopt="0"),
    "giant": dict(queue="false_alarms", fid="F002", filter="all", sel="2"),
    "magnifier": dict(queue="missed_person", fid="M001", filter="all", lens="925,245"),
    "unsure": dict(queue="missed_person", fid="M002", filter="all", unsure="1", note="Behind the fence: a person or a post?"),
    "saved": dict(queue="false_alarms", fid="F004", filter="all", skip="0"),
    "loose": dict(queue="loose_box", fid="L001", filter="all", sel="0"),
}


# --- drawing ----------------------------------------------------------------------------------------------------------
def _jit(rng: random.Random, c, d):
    return tuple(max(0, min(255, v + rng.randint(-d, d))) for v in c)


def _background(img, draw, rng, kind):
    if kind == "field":
        draw.rectangle([0, 0, W, int(H * 0.28)], fill=(176, 200, 222))
        draw.rectangle([0, int(H * 0.28), W, H], fill=(96, 140, 70))
        for _ in range(1400):
            x, y = rng.uniform(0, W), rng.uniform(H * 0.28, H)
            ln = 4 + 10 * (y - H * 0.28) / (H * 0.72)
            draw.line([x, y, x + rng.uniform(-2, 2), y - ln], fill=_jit(rng, (80, 128, 58), 18), width=1)
    else:
        draw.rectangle([0, 0, W, int(H * 0.30)], fill=(156, 152, 144))
        for x in range(0, W, 96):
            draw.line([x, 0, x, int(H * 0.30)], fill=(140, 136, 128), width=2)
        draw.rectangle([0, int(H * 0.30), W, H], fill=(196, 170, 112))
        for _ in range(1800):
            x, y = rng.uniform(0, W), rng.uniform(H * 0.30, H)
            dx, dy = rng.uniform(-14, 14), rng.uniform(-4, 4)
            draw.line([x, y, x + dx, y + dy], fill=_jit(rng, (214, 186, 120), 28), width=1)


def _fence(draw, rng):
    top, bottom = int(H * 0.26), int(H * 0.62)
    for y in (top + 20, (top + bottom) // 2, bottom - 18):
        draw.rectangle([0, y - 5, W, y + 5], fill=(128, 128, 132), outline=(92, 92, 96))
    for x in range(40, W, 145):
        draw.rectangle([x - 9, top - 10, x + 9, bottom + 30], fill=(112, 112, 118), outline=(84, 84, 88))


def _calf(draw, cx, cy, s, colour, rng):
    """A calf-like shape (body and head ellipses, legs); returns its box."""
    main = (112, 70, 40) if colour == "brown" else (34, 34, 36)
    bw, bh = s, 0.52 * s
    leg_w, leg_h = 0.09 * s, 0.38 * s
    for lx in (cx - 0.36 * s, cx - 0.2 * s, cx + 0.18 * s, cx + 0.34 * s):
        draw.rectangle([lx - leg_w / 2, cy, lx + leg_w / 2, cy + leg_h], fill=main)
    draw.ellipse([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], fill=main)
    if colour == "pied":
        for _ in range(3):
            px, py = cx + rng.uniform(-0.3, 0.3) * s, cy + rng.uniform(-0.12, 0.12) * s
            r = rng.uniform(0.08, 0.14) * s
            draw.ellipse([px - r, py - 0.7 * r, px + r, py + 0.7 * r], fill=(236, 234, 228))
    hx, hy = cx + 0.55 * s, cy - 0.28 * s
    draw.ellipse([hx - 0.17 * s, hy - 0.13 * s, hx + 0.17 * s, hy + 0.13 * s], fill=main)
    draw.ellipse([hx - 0.2 * s, hy - 0.19 * s, hx - 0.08 * s, hy - 0.1 * s], fill=main)
    return [cx - bw / 2, hy - 0.19 * s, hx + 0.17 * s, cy + leg_h]


def _figure(img, x, feet, h, base):
    """Draw an abstract standing figure (head, torso, arms, legs) in flat greys; returns its exact box."""
    from PIL import Image, ImageDraw
    m = Image.new("L", (W, H), 0)
    dm, di = ImageDraw.Draw(m), ImageDraw.Draw(img)
    head, body, legs = (base + 38,) * 3, (base, base, base + 4), (base - 30,) * 3
    r = 0.075 * h
    top = feet - h
    tw, aw = 0.26 * h, 0.065 * h
    for side in (-1, 1):
        lx = x + side * 0.065 * h
        shape = [lx - 0.055 * h, feet - 0.47 * h, lx + 0.055 * h, feet]
        di.rectangle(shape, fill=legs)
        dm.rectangle(shape, fill=255)
    shape = [x - tw / 2, top + 2.2 * r, x + tw / 2, feet - 0.45 * h]
    di.rounded_rectangle(shape, radius=0.06 * h, fill=body)
    dm.rounded_rectangle(shape, radius=0.06 * h, fill=255)
    for side in (-1, 1):
        ax = x + side * (tw / 2 + aw / 2 + 0.01 * h)
        shape = [ax - aw / 2, top + 2.4 * r, ax + aw / 2, feet - 0.5 * h]
        di.rounded_rectangle(shape, radius=aw / 2, fill=(base - 14,) * 3)
        dm.rounded_rectangle(shape, radius=aw / 2, fill=255)
    shape = [x - r, top, x + r, top + 2 * r]
    di.ellipse(shape, fill=head)
    dm.ellipse(shape, fill=255)
    return [float(v) for v in m.getbbox()]


def _pixelate(img, box):
    """Mosaic a region on a grid anchored at (0, 0) with cells of BLOCK px (cell mean), like the apply step."""
    from PIL import Image
    x1, y1 = int(box[0] // BLOCK * BLOCK), int(box[1] // BLOCK * BLOCK)
    x2, y2 = min(W, int(-(-box[2] // BLOCK) * BLOCK)), min(H, int(-(-box[3] // BLOCK) * BLOCK))
    if x2 <= x1 or y2 <= y1:
        return
    region = img.crop((x1, y1, x2, y2))
    small = region.resize((max(1, -(-(x2 - x1) // BLOCK)), max(1, -(-(y2 - y1) // BLOCK))), Image.BOX)
    img.paste(small.resize((x2 - x1, y2 - y1), Image.NEAREST), (x1, y1))


def _resolve(ref, truth, calves):
    kind = ref[0]
    if kind == "box":
        return [float(v) for v in ref[1]]
    if kind == "calf":
        b = calves[ref[1]]
    else:
        b = truth[ref[1]]
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    if kind in ("truth", "calf"):
        out = [x1, y1, x2, y2]
    elif kind == "loose":                                   # every side pushed out: area grows by about (1 + f)
        f = ref[2] / 2
        out = [x1 - f * w, y1 - f * h, x2 + f * w, y2 + f * h]
    elif kind == "shift":
        out = [x1 + ref[2] * w, y1 + ref[3] * h, x2 + ref[2] * w, y2 + ref[3] * h]
    elif kind == "tight":
        out = [x1, y1, x2, y1 + ref[2] * h]
    else:
        raise ValueError(f"unknown reference {ref!r}")
    return [round(min(max(out[0], 0), W), 1), round(min(max(out[1], 0), H), 1),
            round(min(max(out[2], 0), W), 1), round(min(max(out[3], 0), H), 1)]


def draw_scene(scene: dict, seed: int):
    """Draw one scene. Returns (PIL image, truth boxes of the figures, calf boxes)."""
    from PIL import Image, ImageDraw, ImageOps
    rng = random.Random(f"{seed}:{scene['fid']}")
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    _background(img, draw, rng, scene["bg"])
    if scene.get("fence") and not scene.get("fence_front"):
        _fence(draw, rng)
    calves = [_calf(draw, cx, cy, s, c, rng) for cx, cy, s, c in scene["calves"]]
    truth = []
    for x, feet, h, style in scene["people"]:
        box = _figure(img, x, feet, h, 118 + rng.randint(-10, 10))
        if style == "mosaic":
            _pixelate(img, box)
        truth.append(box)
    if scene.get("fence") and scene.get("fence_front"):
        _fence(draw, rng)
    if scene.get("ir"):                                     # night: grey, low contrast, like an infrared camera
        img = ImageOps.grayscale(img).point(lambda v: int(24 + 0.62 * v)).convert("RGB")
    return img, truth, calves


# --- the demo folder --------------------------------------------------------------------------------------------------
def _check_target(out_dir: str) -> None:
    """Refuse to write into a folder that holds anything but demo files, or a manifest that is not a demo."""
    if not os.path.exists(out_dir):
        return
    if not os.path.isdir(out_dir):
        raise DemoGuardError(f"{out_dir} is not a folder")
    extra = sorted(set(os.listdir(out_dir)) - DEMO_FILES)
    if extra:
        raise DemoGuardError(f"refusing to write the demo into {out_dir}: it holds other files ({', '.join(extra[:5])})")
    mpath = os.path.join(out_dir, MANIFEST)
    if os.path.exists(mpath):
        try:
            with open(mpath, encoding="utf-8") as fh:
                is_demo = json.load(fh).get("meta", {}).get("demo") is True
        except (OSError, ValueError, AttributeError):
            is_demo = False
        if not is_demo:
            raise DemoGuardError(f"refusing to overwrite {mpath}: it is not a demo manifest")
    frames = os.path.join(out_dir, "frames")
    if os.path.isdir(frames):
        names = {f"{s['fid']}.png" for s in SCENES}
        extra = sorted(set(os.listdir(frames)) - names)
        if extra:
            raise DemoGuardError(f"refusing to write the demo into {out_dir}: frames/ holds other files ({', '.join(extra[:5])})")


def make_demo(out_dir: str, seed: int = 7, seed_progress: bool = True) -> str:
    """Write the demo folder (frames/*.png, manifest.json, demo_truth.json, seed_decisions.jsonl). Returns the manifest path."""
    _check_target(out_dir)
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)
    frames, truth_all = {}, {}
    for sc in SCENES:
        img, truth, calves = draw_scene(sc, seed)
        fid = sc["fid"]
        rel = f"frames/{fid}.png"
        img.save(os.path.join(out_dir, rel), "PNG", optimize=True)
        persons = [_resolve(r, truth, calves) for r in sc["persons"]]
        cands = sorted((_resolve(r, truth, calves) + [round(s, 4), -1] for r, s in sc["cands"]), key=lambda c: -c[4])
        minute = int(sc["t"] // 60)
        frames[fid] = dict(id=fid, kind=sc["kind"], key=f"demo:{fid}", video=sc["video"], t=sc["t"], table_t=sc["t"],
                           clock=f"demo clock 10:{minute % 60:02d}" if not sc.get("ir") else f"demo clock 02:{minute % 60:02d}",
                           night=bool(sc.get("ir")), ir=bool(sc.get("ir")), width=W, height=H, src_width=W, src_height=H,
                           path=rel, source="synthetic", persons=persons, cands=cands)
        truth_all[fid] = [[round(v, 1) for v in b] for b in truth]
    queues = []
    for qid, title, desc in QUEUES:
        members = [sc for sc in SCENES if qid == "all" or qid in sc["queues"]]
        queues.append(dict(id=qid, title=title, desc=desc,
                           entries=[dict(fid=sc["fid"], np=len(frames[sc["fid"]]["persons"]), note=sc["note"]) for sc in members]))
    meta = dict(demo=True, title="OpenVeil review demo", banner="Demo data: synthetic frames", note="", kinds=KINDS,
                t_op=0.12, t_low=0.06, states=STATES, seed=seed)
    manifest = dict(meta=meta, frames=frames, queues=queues)
    write_json_atomic(os.path.join(out_dir, TRUTH), truth_all, indent=1)
    lines = []
    if seed_progress:
        for n, (fid, status, which, note) in enumerate(SEED):
            persons = truth_all[fid] if which == "truth" else []
            lines.append(json.dumps(dict(fid=fid, key=f"demo:{fid}", kind=frames[fid]["kind"], queue=next(sc for sc in SCENES if sc["fid"] == fid)["queues"][0],
                                         status=status, persons=persons, note=note, ms=1200 + 700 * n,
                                         ts=f"2026-01-05T10:00:{n:02d}.000")))
    with open(os.path.join(out_dir, SEED_DECISIONS), "w", encoding="utf-8") as fh:
        fh.write("".join(line + "\n" for line in lines))
    path = os.path.join(out_dir, MANIFEST)
    write_json_atomic(path, manifest)
    return path


def default_demo_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "openveil_review_demo")


def run_demo(out_dir: str | None = None, port: int = DEMO_PORT, open_browser: bool = True, seed: int = 7) -> None:
    """Build the demo (if needed), restore the seed decisions and serve it with the demo guard on."""
    out_dir = out_dir or default_demo_dir()
    make_demo(out_dir, seed=seed)
    print(f"demo folder: {out_dir}", flush=True)
    serve(out_dir, port, demo=True, reset=True, open_browser=open_browser)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="openveil review demo", description="Build and serve the synthetic review demo.")
    ap.add_argument("--out", default=None, help="demo folder (default: a folder in the temporary directory)")
    ap.add_argument("--port", type=int, default=DEMO_PORT, help=f"default {DEMO_PORT}; 0 picks a free port")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--build-only", "--no-serve", dest="build_only", action="store_true", help="write the demo folder and exit")
    a = ap.parse_args(argv)
    try:
        if a.build_only:
            out = a.out or default_demo_dir()
            print(make_demo(out, seed=a.seed))
            print(f"serve it with:  openveil review serve {out} --demo", flush=True)
        else:
            run_demo(a.out, a.port, open_browser=not a.no_browser, seed=a.seed)
    except DemoGuardError as e:
        raise SystemExit(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())
