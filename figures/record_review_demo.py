# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Record a short screen recording of the review tool, on the synthetic demo only.

    python figures/record_review_demo.py [--port 8791] [--out DIR] [--keep]

Writes docs/img/review_demo.mp4 (H.264), docs/img/review_demo.gif (for the README)
and docs/img/review_demo_poster.png. Everything else goes to a temporary folder that
is removed at the end, also when a step fails (``--keep`` leaves it and prints its path).

Needs Python 3.10 or newer with Pillow and PyAV (with libx264; made with Pillow 10
and PyAV 18), and Chrome, Chromium or Edge (set the environment variable CHROME to
its path if it is not found). The run takes about a minute.

What it does:

1. draws the demo frames with ``openveil.review.demo`` (no photographs, nobody real),
   without the seed decisions so every queue starts at 0, and serves them with the
   demo guard on. It refuses to continue unless the served manifest says ``demo``
   and every frame is ``synthetic``;
2. starts a headless browser with a fresh temporary profile and drives the real page
   through the Chrome DevTools Protocol with real mouse and key input. After each
   step it checks on the page that the step had its effect, and at the end that the
   server stored the decisions the captions describe;
3. takes one screenshot per video frame. A headless screenshot has no mouse pointer,
   so the pointer, a ripple on each click, a key cap for each key and the caption bar
   above the page are drawn onto the frames with Pillow, and a ring around the queue
   selector while the queue is changed. The caption bar is at the top because a video
   player lays its controls over the bottom of the picture when the film is paused. The drop-down list of a headless browser takes
   no mouse input, so the list is opened with a click and the next queue is chosen
   with the Down and Enter keys; the recording shows both as key caps.
"""
import argparse
import base64
import io
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "docs", "img")
sys.path.insert(0, REPO)

VW, VH, BAR = 1280, 720, 80          # page viewport and the caption bar above it (video: 1280 x 800)
FPS = 15                             # video frames per second; the GIF keeps every second frame
GIF_W, GIF_H = 864, 540
PRIVATE_PORTS = (8765, 8766)         # default ports of the review server: they may serve real, unmasked data

KEYS = {"Enter": (13, "Enter", "\r"), "ArrowDown": (40, "ArrowDown", None),
        "x": (88, "KeyX", "x"), "u": (85, "KeyU", "u"), "z": (90, "KeyZ", "z")}


# --- Chrome DevTools Protocol over a minimal WebSocket client (standard library only) -----------
class CDP:
    def __init__(self, ws_url):
        u = urllib.parse.urlparse(ws_url)
        self.sock = socket.create_connection((u.hostname, u.port))
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        self.rd = self.sock.makefile("rb")
        if b" 101 " not in self.rd.readline():
            raise ConnectionError("the browser refused the DevTools connection")
        while self.rd.readline().strip():
            pass
        self.n = 0

    def _send(self, text):
        data, mask = text.encode(), os.urandom(4)
        n = len(data)
        size = bytes([0x80 | n]) if n < 126 else b"\xfe" + struct.pack(">H", n) if n < 65536 else b"\xff" + struct.pack(">Q", n)
        self.sock.sendall(b"\x81" + size + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def _recv(self):
        out = b""
        while True:
            b0, b1 = self.rd.read(2)
            n = b1 & 0x7F
            if n >= 126:
                n = struct.unpack(">H" if n == 126 else ">Q", self.rd.read(2 if n == 126 else 8))[0]
            data = self.rd.read(n)
            if b0 & 0x0F == 8:
                raise ConnectionError("the browser closed the DevTools connection")
            if b0 & 0x0F in (0, 1):                      # text frame or its continuation
                out += data
                if b0 & 0x80:
                    return out.decode()

    def call(self, method, **params):
        self.n += 1
        self._send(json.dumps(dict(id=self.n, method=method, params=params)))
        while True:
            msg = json.loads(self._recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]


def find_browser():
    names = ["google-chrome", "chromium", "chromium-browser", "chrome", "msedge", "microsoft-edge"]
    win = [os.path.join(os.environ.get(var, ""), tail) for var in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")
           for tail in ("Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe")]
    paths = ([os.environ.get("CHROME", "")] + win + ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
             + [shutil.which(n) or "" for n in names])
    for p in paths:
        if p and os.path.isfile(p):
            return p
    raise SystemExit("no Chrome, Chromium or Edge found: set the environment variable CHROME to its path")


def start_browser(profile):
    """Headless browser with a fresh profile and a DevTools port chosen by the browser; returns (process, CDP)."""
    proc = subprocess.Popen([find_browser(), "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={profile}",
                             f"--window-size={VW},{VH}", "--force-device-scale-factor=1", "--hide-scrollbars", "--no-first-run",
                             "--no-default-browser-check", "--disable-extensions", "--disable-sync", "--mute-audio",
                             "--disable-background-networking", "--disable-component-update", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        port = None
        for _ in range(300):                             # the browser writes its DevTools port into the profile
            try:
                with open(os.path.join(profile, "DevToolsActivePort")) as fh:
                    port = int(fh.read().split()[0])
                break
            except (OSError, ValueError, IndexError):
                time.sleep(0.1)
        if port is None:
            raise SystemExit("the browser did not start")
        pages = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
        cdp = CDP(next(p["webSocketDebuggerUrl"] for p in pages if p["type"] == "page"))
        cdp.call("Emulation.setDeviceMetricsOverride", width=VW, height=VH, deviceScaleFactor=1, mobile=False)
        cdp.call("Emulation.setEmulatedMedia", features=[dict(name="prefers-color-scheme", value="light")])
    except BaseException:
        proc.kill()
        raise
    return proc, cdp


# --- recording: real input to the page, one screenshot per video frame --------------------------
class Recorder:
    def __init__(self, cdp):
        self.cdp, self.frames = cdp, []
        self.x, self.y = VW * 0.37, VH * 0.6
        self.text, self.down, self.ring = "", False, None
        self.ripple = self.keycap = None                 # [x, y, age in frames] / [label, frames left, held]
        self.next_t = time.perf_counter()

    def js(self, expr):
        r = self.cdp.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"{expr}: {r['exceptionDetails'].get('text')}")
        return r["result"].get("value")

    def expect(self, expr, want):
        """Fail unless the page shows what the step should have produced."""
        got = self.js(expr)
        if got != want:
            raise SystemExit(f"step failed: {expr} is {got!r}, expected {want!r}")

    def shot(self):
        """Capture one video frame, paced to real time so that the page's own timers look natural."""
        time.sleep(max(0.0, self.next_t - time.perf_counter()))
        self.next_t = max(self.next_t, time.perf_counter() - 1 / FPS) + 1 / FPS
        png = base64.b64decode(self.cdp.call("Page.captureScreenshot", format="png")["data"])
        self.frames.append(dict(png=png, x=self.x, y=self.y, text=self.text, down=self.down, ring=self.ring,
                                ripple=self.ripple and tuple(self.ripple), key=self.keycap and self.keycap[0]))
        if self.ripple:
            self.ripple[2] += 1
            if self.ripple[2] > 7:
                self.ripple = None
        if self.keycap and not self.keycap[2]:
            self.keycap[1] -= 1
            if self.keycap[1] <= 0:
                self.keycap = None

    def caption(self, text):
        if FONT.getlength(text) > VW - 190:              # leave room for the key cap
            raise SystemExit(f"caption too long: {text}")
        self.text = text

    def hold(self, seconds):
        for _ in range(round(seconds * FPS)):
            self.shot()

    def _mouse(self, kind):
        self.cdp.call("Input.dispatchMouseEvent", type=kind, x=round(self.x), y=round(self.y), button="left",
                      buttons=1 if self.down else 0, clickCount=0 if kind == "mouseMoved" else 1)

    def move(self, xy, seconds=None):
        x0, y0, (x1, y1) = self.x, self.y, xy
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        n = max(2, round((seconds or min(1.1, max(0.45, dist / 800))) * FPS))
        for i in range(1, n + 1):
            t = i / n
            t = t * t * (3 - 2 * t)                      # ease in and out
            self.x, self.y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            self._mouse("mouseMoved")
            self.shot()

    def press(self):
        self.down = True
        self._mouse("mousePressed")
        self.ripple = [self.x, self.y, 0]
        self.shot()

    def release(self):
        self._mouse("mouseReleased")
        self.down = False
        self.shot()

    def click(self, xy):
        self.move(xy)
        self.hold(0.15)
        self.press()
        self.release()

    def drag(self, start, end, seconds=1.0):
        self.move(start)
        self.hold(0.2)
        self.press()
        self.move(end, seconds)
        self.hold(0.2)
        self.release()

    def _key(self, kind, name):
        code, phys, text = KEYS[name]
        extra = dict(text=text) if text and kind == "keyDown" else {}
        self.cdp.call("Input.dispatchKeyEvent", type=kind, key=name, code=phys, windowsVirtualKeyCode=code, **extra)

    def key_down(self, name, label):
        self._key("keyDown", name)
        self.keycap = [label, 9, True]                   # shown while held and for 9 frames after

    def key_up(self, name):
        self._key("keyUp", name)
        if self.keycap:
            self.keycap[2] = False

    def key(self, name, label=None):
        self.key_down(name, label or (name if len(name) > 1 else name.upper()))
        self.hold(0.14)
        self.key_up(name)

    def type(self, text):
        for i, ch in enumerate(text):
            self.cdp.call("Input.dispatchKeyEvent", type="keyDown", key=ch, text=ch)
            self.cdp.call("Input.dispatchKeyEvent", type="keyUp", key=ch)
            if i % 2:
                self.shot()

    # where things are on the page, in viewport pixels
    def at(self, selector):
        return tuple(self.js(f"(r => [r.left + r.width / 2, r.top + r.height / 2])(document.querySelector({selector!r}).getBoundingClientRect())"))

    def image(self, ix, iy):
        """Viewport position of a point given in pixels of the frame that is shown."""
        left, top, s = self.js("(r => [r.left, r.top, r.width / F.width])(document.getElementById('cv').getBoundingClientRect())")
        return left + ix * s, top + iy * s

    def choose_next_queue(self, holds=(1.1, 0.7, 1.1)):
        """Open the queue selector with a click and choose the next queue with the Down and Enter keys.

        ``holds``: seconds to wait with the list open, after Down, and after Enter."""
        left, top, right, bottom = self.js("(r => [r.left, r.top, r.right, r.bottom])(document.getElementById('queueSel').getBoundingClientRect())")
        self.ring = [round(left) - 5, round(top) - 5, round(right) + 5, round(bottom) + 5]
        self.click((right - 12, (top + bottom) / 2))
        self.hold(holds[0])
        self.key("ArrowDown", "↓")
        self.hold(holds[1])
        self.key("Enter")
        self.hold(holds[2])
        self.ring = None


POS = "document.getElementById('pos').textContent"
QUEUE = "document.getElementById('queueSel').selectedOptions[0].text"
STATE = "document.getElementById('state').textContent"
BOXES = "document.querySelectorAll('#persons .row').length"
NOTE = "Behind the fence: a person or a post?"


def centre(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def storyboard(r, frame, truth):
    """The recorded session. ``frame(fid)`` is the served frame record, ``truth`` the drawn figure boxes.

    Each caption is on screen for a moment before the action it explains, so that it can be read first."""
    r.caption("OpenVeil review tool, synthetic demo: green boxes would be mosaicked")
    r.hold(2.8)

    # 1. the proposal is right
    r.expect(POS, "4 / 4  ·  F004")
    r.caption("The proposal is right: Enter saves it and opens the next frame")
    r.hold(1.3)
    r.key("Enter")
    r.hold(1.0)
    r.expect(POS, "1 / 4  ·  F001")
    r.expect(QUEUE, "Demo: false alarms (1/4)")

    # 2. a false alarm on a calf
    r.caption("A false alarm on a calf: click the box, press X, then Enter")
    r.expect(BOXES, 2)
    r.hold(0.8)
    r.click(r.image(*centre(frame("F001")["persons"][1])))
    r.expect("document.querySelectorAll('#persons .row.sel').length", 1)
    r.hold(0.8)
    r.key("x")
    r.hold(1.4)
    r.expect(BOXES, 1)
    r.key("Enter")
    r.hold(1.0)
    r.expect(QUEUE, "Demo: false alarms (2/4)")

    # 3. the queue selector
    r.caption("Each queue shows its progress (done / total): choose the next queue")
    r.hold(0.4)
    r.choose_next_queue()
    r.expect(QUEUE, "Demo: missed person (0/3)")
    r.expect(POS, "1 / 3  ·  M001")

    # 4. a missed person: adopt the dashed candidate, draw a box for the figure without one
    r.caption("A missed person: one click on the dashed candidate makes it a person box")
    r.expect(BOXES, 0)
    r.hold(0.7)
    r.click(r.image(*centre(frame("M001")["cands"][0])))
    r.hold(0.9)
    r.expect(BOXES, 1)
    r.caption("No candidate on the small figure at the edge: drag to draw a box")
    t = truth["M001"][1]
    r.drag(r.image(t[0] - 5, t[1] - 5), r.image(t[2] + 8, t[3] + 5), 1.1)
    r.hold(0.8)
    r.expect(BOXES, 2)
    r.key("Enter")
    r.hold(1.0)
    r.expect(QUEUE, "Demo: missed person (1/3)")
    r.expect(POS, "2 / 3  ·  M002")

    # 5. the magnifier over a figure half hidden behind a fence, then unsure with a note
    r.caption("Hold Z for a 4x magnifier: a figure half hidden behind the fence")
    t = truth["M002"][0]
    cx = (t[0] + t[2]) / 2
    r.move(r.image(cx, t[1] + 20))
    r.hold(0.3)
    r.key_down("z", "hold Z")
    r.hold(0.5)
    r.expect("lens && mouse.in", True)
    r.move(r.image(cx + 4, t[3] - 25), 1.0)
    r.move(r.image(cx - 2, t[1] + 45), 0.8)
    r.hold(0.4)
    r.key_up("z")
    r.hold(0.5)
    r.expect("lens", False)
    r.caption("Not sure? Press U and add a note: unsure frames are listed separately")
    r.hold(1.0)
    r.key("u")
    r.hold(0.6)
    r.expect(STATE, "edited, not saved yet. same as the proposal. Marked unsure.")
    r.click(r.at("#note"))
    r.type(NOTE)
    r.hold(0.4)
    r.expect("document.getElementById('note').value", NOTE)
    r.click(r.at("#bConfirm"))
    r.hold(1.0)
    r.expect(QUEUE, "Demo: missed person (2/3)")

    # 6. a loose box
    r.caption("The next queue: boxes that are too loose")
    r.choose_next_queue((0.6, 0.4, 0.7))                 # shorter: the selector has been shown before
    r.expect(QUEUE, "Demo: loose box (0/3)")
    r.expect(POS, "1 / 3  ·  L001")
    r.caption("A loose box: click it, then drag its handles until it fits the person")
    b, t = frame("L001")["persons"][0], truth["L001"][0]
    r.hold(0.3)
    r.click(r.image(*centre(b)))
    r.expect("document.querySelectorAll('#persons .row.sel').length", 1)
    r.hold(0.4)
    r.drag(r.image(b[2], b[3]), r.image(t[2] + 3, t[3] + 3), 0.9)
    r.drag(r.image(b[0], b[1]), r.image(t[0] - 3, t[1] - 3), 0.9)
    r.hold(0.7)
    r.expect(STATE, "edited, not saved yet. differs from the proposal.")
    r.key("Enter")
    r.hold(1.0)
    r.expect(QUEUE, "Demo: loose box (1/3)")

    r.caption("Every decision is saved at once. Try it: openveil review demo")
    r.move(r.image(480, 300))
    r.hold(2.0)


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    i = max(w, 0) * max(h, 0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def check_decisions(demo_dir, truth):
    """The server must have stored exactly what the captions say was done."""
    with open(os.path.join(demo_dir, "decisions.jsonl"), encoding="utf-8") as fh:
        d = [json.loads(line) for line in fh if line.strip()]
    got = [(x["fid"], x["status"], len(x["persons"])) for x in d]
    want = [("F004", "confirmed", 1), ("F001", "confirmed", 1), ("M001", "confirmed", 2), ("M002", "unsure", 0), ("L001", "confirmed", 1)]
    if got != want or d[3]["note"] != NOTE:
        raise SystemExit(f"the stored decisions are not the recorded ones: {got}")
    for x in d:                                          # every confirmed box sits on a drawn figure
        if x["status"] == "confirmed" and any(max(iou(p, t) for t in truth[x["fid"]]) < 0.75 for p in x["persons"]):
            raise SystemExit(f"{x['fid']}: a saved box does not fit a figure")


# --- drawing the pointer, the captions and the key caps; writing the three files ----------------
def font(size):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def pointer():
    """An arrow pointer (black with a white edge), drawn at 4x and reduced for smooth edges."""
    k, pts = 4 * 1.7, [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
    big = Image.new("RGBA", (int(16 * k), int(24 * k)), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    pts = [(8 + x * k, 8 + y * k) for x, y in pts]
    d.polygon(pts, fill=(20, 20, 20, 255))
    d.line(pts + pts[:2], fill=(255, 255, 255, 255), width=8, joint="curve")
    return big.resize((big.width // 4, big.height // 4), Image.LANCZOS)


FONT, KEYFONT, POINTER = font(30), font(26), pointer()


def compose(fr):
    """One video frame: the caption bar, and under it the page screenshot with the pointer and its marks."""
    im = Image.new("RGB", (VW, VH + BAR), (27, 30, 35))
    im.paste(Image.open(io.BytesIO(fr["png"])).convert("RGB").crop((0, 0, VW, VH)), (0, BAR))
    d = ImageDraw.Draw(im, "RGBA")
    x, y = fr["x"], fr["y"] + BAR                        # page positions are BAR pixels lower in the video frame
    if fr["ring"]:
        left, top, right, bottom = fr["ring"]
        d.rounded_rectangle([left, top + BAR, right, bottom + BAR], radius=10, outline=(240, 120, 20, 255), width=4)
    if fr["ripple"]:
        rx, ry, age = fr["ripple"]
        rad = 10 + 5 * age
        d.ellipse([rx - rad, ry + BAR - rad, rx + rad, ry + BAR + rad], outline=(240, 120, 20, max(0, 255 - 32 * age)), width=5)
    if fr["down"]:
        d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=(240, 120, 20, 150))
    im.paste(POINTER, (round(x) - 2, round(y) - 2), POINTER)
    d.rectangle([0, 0, VW, BAR], fill=(27, 30, 35, 255))  # the bar stays clean when a mark reaches the top of the page
    d.text((26, BAR // 2), fr["text"], font=FONT, fill=(255, 255, 255, 255), anchor="lm")
    if fr["key"]:
        w = d.textlength(fr["key"], font=KEYFONT) + 36
        box = [VW - 24 - w, 16, VW - 24, BAR - 16]
        d.rounded_rectangle(box, radius=9, fill=(47, 208, 107, 255), outline=(255, 255, 255, 255), width=2)
        d.text(((box[0] + box[2]) / 2, BAR // 2), fr["key"], font=KEYFONT, fill=(4, 33, 15, 255), anchor="mm")
    return im


# H.264 High at Level 4.0 with 4 reference frames and 3 B-frames (phones and older hardware decoders
# guarantee no more), and a keyframe at least every 10 seconds so that seeking in the player stays quick.
# The thread count is fixed so that the settings x264 writes into the stream do not depend on the machine.
X264 = {"crf": "18", "preset": "veryslow", "tune": "animation", "level": "4.0",
        "x264-params": f"ref=4:bframes=3:keyint={10 * FPS}:min-keyint={FPS}:threads=4:lookahead-threads=1"}


def write_mp4(frames, path):
    import av
    from av.video.reformatter import ColorRange, Colorspace
    with av.open(path, "w", options={"movflags": "+faststart", "fflags": "+bitexact"}) as out:
        st = out.add_stream("libx264", rate=FPS, options=X264)
        st.width, st.height, st.pix_fmt = VW, VH + BAR, "yuv420p"
        cc = st.codec_context
        cc.colorspace, cc.color_primaries, cc.color_trc, cc.color_range = 1, 1, 1, 1      # BT.709, limited range
        for fr in frames:
            vf = av.VideoFrame.from_image(compose(fr)).reformat(format="yuv420p", dst_colorspace=Colorspace.ITU709,
                                                                 dst_color_range=ColorRange.MPEG)
            out.mux(st.encode(vf))
        out.mux(st.encode(None))


def write_gif(frames, path):
    """Every second frame at 864 x 540, the same palette for every frame, no dithering (so still areas cost nothing)."""
    small = [compose(fr).resize((GIF_W, GIF_H), Image.LANCZOS) for fr in frames[::2]]
    sample = small[::max(1, len(small) // 16)][:16]
    sheet = Image.new("RGB", (GIF_W, GIF_H * len(sample)))
    for i, im in enumerate(sample):
        sheet.paste(im, (0, GIF_H * i))
    palette = sheet.quantize(255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in small]
    durations = [(130, 130, 140)[i % 3] for i in range(len(pal))]                         # 2 / 15 s on average
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=False, disposal=1)


def record(tmp, port):
    """Draw and serve the demo from ``tmp`` and drive the page through the storyboard; returns (frames, seconds)."""
    from openveil.review.demo import make_demo
    from openveil.review.server import make_server

    demo_dir, profile = os.path.join(tmp, "demo"), os.path.join(tmp, "profile")
    make_demo(demo_dir, seed_progress=False)
    with open(os.path.join(demo_dir, "demo_truth.json"), encoding="utf-8") as fh:
        truth = json.load(fh)
    server = make_server(demo_dir, port, demo=True, reset=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    browser = None
    try:
        def get(path):
            return json.load(urllib.request.urlopen(server.url + path))
        meta = get("api/manifest")["meta"]
        if meta.get("demo") is not True or any(get("api/frame/" + fid).get("source") != "synthetic" for fid in truth):
            raise SystemExit("the server is not serving the synthetic demo: nothing recorded")
        browser, cdp = start_browser(profile)
        rec = Recorder(cdp)
        cdp.call("Page.navigate", url=server.url + "?queue=false_alarms&fid=F004")
        for _ in range(100):
            time.sleep(0.1)
            if rec.js("document.documentElement.dataset.ready") == "1":
                break
        else:
            raise SystemExit("the review page did not load")
        rec.move((rec.x, rec.y), 0.2)                    # the page learns where the pointer is
        rec.frames.clear()
        started = time.perf_counter()
        storyboard(rec, lambda fid: get("api/frame/" + fid), truth)
        took = time.perf_counter() - started
        check_decisions(demo_dir, truth)
        cdp.call("Browser.close")
    finally:
        if browser:
            try:
                browser.wait(10)
            except subprocess.TimeoutExpired:
                browser.kill()
                browser.wait(10)
        server.shutdown()
        server.server_close()
    return rec.frames, took


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=8791, help="port for the demo server (default 8791)")
    ap.add_argument("--out", default=OUT, help="folder for the three files (default docs/img; created if missing)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary folder (demo data, browser profile)")
    a = ap.parse_args()
    if a.port in PRIVATE_PORTS:
        raise SystemExit(f"port {a.port} is a default port of the review server and may serve real data: choose another")
    os.makedirs(a.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="openveil_record_")
    try:
        frames, took = record(tmp, a.port)
    finally:                                             # also when a step check fails
        if a.keep:
            print(f"temporary folder kept: {tmp}")
        else:
            for _ in range(20):                          # the browser may hold its profile for a moment
                shutil.rmtree(tmp, ignore_errors=True)
                if not os.path.exists(tmp):
                    break
                time.sleep(0.25)

    write_mp4(frames, os.path.join(a.out, "review_demo.mp4"))
    write_gif(frames, os.path.join(a.out, "review_demo.gif"))
    poster = next(i for i, fr in enumerate(frames) if fr["text"].startswith("A false alarm") and fr["ripple"]) + 9
    compose(frames[poster]).quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        os.path.join(a.out, "review_demo_poster.png"), optimize=True)
    print(f"{len(frames)} frames, {len(frames) / FPS:.1f} s at {FPS} frames per second (captured in {took:.1f} s)")
    for i, fr in enumerate(frames):                      # the storyboard as recorded
        if i == 0 or fr["text"] != frames[i - 1]["text"]:
            print(f"{i / FPS:5.1f} s  {fr['text']}")


if __name__ == "__main__":
    main()
