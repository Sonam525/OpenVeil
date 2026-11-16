# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Draw the repository figures as plain SVG. Standard library only.

    python figures/make_figures.py

Writes figures/banner.svg, figures/pipeline.svg and figures/pipeline_narrow.svg,
then copies them to docs/assets/ for the project page. Every figure is drawn:
there is no footage and no real person in any of them.

The banner is a small scene graph. The people in it are never written to the SVG:
they exist only as shapes that are sampled when the mosaic is computed. As in
OpenVeil, each person box is enlarged by 20% per side, and every cell of a fixed
grid that touches it is filled with the mean colour of the scene under that cell.
The pipeline figures follow the viewer's light or dark setting.
"""
import os
import shutil
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


# --------------------------------------------------------------------------------------
# Banner: a drawn calf barn where the people are veiled by a mosaic
# --------------------------------------------------------------------------------------

W, H = 1280, 420
CELL = 16                 # OpenVeil: max(16, round(H / 36)) px, i.e. 16 px at this height
PAD = 0.20                # OpenVeil: each box is enlarged by 20% per side


def _rgb(hexcol):
    h = hexcol.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


class VGrad:
    """A vertical linear gradient in user space, usable for drawing and for sampling."""

    def __init__(self, gid, y0, y1, stops):
        self.gid, self.y0, self.y1, self.stops = gid, y0, y1, [(o, _rgb(c)) for o, c in stops]

    def svg(self):
        stops = "".join('<stop offset="%g" stop-color="%s"/>' % (o, _hex(c)) for o, c in self.stops)
        return ('<linearGradient id="%s" gradientUnits="userSpaceOnUse" x1="0" y1="%g" x2="0" y2="%g">%s'
                '</linearGradient>' % (self.gid, self.y0, self.y1, stops))

    def at(self, y):
        t = min(1.0, max(0.0, (y - self.y0) / float(self.y1 - self.y0)))
        for (o0, c0), (o1, c1) in zip(self.stops, self.stops[1:]):
            if t <= o1:
                u = 0.0 if o1 == o0 else (t - o0) / (o1 - o0)
                return tuple(a + (b - a) * u for a, b in zip(c0, c1))
        return self.stops[-1][1]


class Scene:
    """Shapes in drawing order. Each shape can be drawn, sampled, or both."""

    def __init__(self):
        self.items = []      # (kind, geometry, fill, opacity, draw, sample)
        self.grads = []

    def grad(self, gid, y0, y1, stops):
        g = VGrad(gid, y0, y1, stops)
        self.grads.append(g)
        return g

    def rect(self, x, y, w, h, fill, rx=0, op=1.0, draw=True, sample=True):
        self.items.append(("rect", (x, y, w, h, rx), fill, op, draw, sample))

    def ellipse(self, cx, cy, rx, ry, fill, op=1.0, draw=True, sample=True):
        self.items.append(("ellipse", (cx, cy, rx, ry), fill, op, draw, sample))

    def poly(self, pts, fill, op=1.0, draw=True, sample=True):
        self.items.append(("poly", tuple(pts), fill, op, draw, sample))

    def raw(self, svg):
        """Drawn only; never sampled (curves, details). Must not lie under a mosaic."""
        self.items.append(("raw", svg, None, 1.0, True, False))

    # ---- sampling --------------------------------------------------------------------
    @staticmethod
    def _inside(kind, g, x, y):
        if kind == "rect":
            return g[0] <= x < g[0] + g[2] and g[1] <= y < g[1] + g[3]
        if kind == "ellipse":
            return ((x - g[0]) / g[2]) ** 2 + ((y - g[1]) / g[3]) ** 2 <= 1.0
        if kind == "poly":
            inside, n = False, len(g)
            for i in range(n):
                (x0, y0), (x1, y1) = g[i], g[(i + 1) % n]
                if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                    inside = not inside
            return inside
        return False

    def colour_at(self, x, y):
        c = (255.0, 255.0, 255.0)
        for kind, g, fill, op, _draw, sample in self.items:
            if not sample or not self._inside(kind, g, x, y):
                continue
            f = fill.at(y) if isinstance(fill, VGrad) else _rgb(fill)
            c = tuple(a * (1 - op) + b * op for a, b in zip(c, f))
        return c

    def mosaic(self, box, n=5):
        """Cells of the fixed grid that touch the enlarged box, each with its mean colour."""
        x0, y0, x1, y1 = box
        px, py = (x1 - x0) * PAD, (y1 - y0) * PAD
        x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)
        cells = []
        for gy in range(int(y0 // CELL), int(-(-y1 // CELL))):
            for gx in range(int(x0 // CELL), int(-(-x1 // CELL))):
                cx, cy = gx * CELL, gy * CELL
                ch = min(CELL, H - cy)
                acc = [0.0, 0.0, 0.0]
                for i in range(n):
                    for j in range(n):
                        col = self.colour_at(cx + (i + .5) * CELL / n, cy + (j + .5) * ch / n)
                        acc = [a + b for a, b in zip(acc, col)]
                cells.append((cx, cy, CELL, ch, _hex([a / (n * n) for a in acc])))
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        bounds = (min(xs), min(ys), max(xs) + CELL, max(c[1] + c[3] for c in cells))
        return cells, bounds

    # ---- drawing ---------------------------------------------------------------------
    def svg_items(self):
        out = []
        for kind, g, fill, op, draw, _sample in self.items:
            if not draw:
                continue
            if kind == "raw":
                out.append(g)
                continue
            f = "url(#%s)" % fill.gid if isinstance(fill, VGrad) else fill
            o = "" if op >= 1 else ' fill-opacity="%g"' % op
            if kind == "rect":
                r = ' rx="%g"' % g[4] if g[4] else ""
                out.append('<rect x="%g" y="%g" width="%g" height="%g"%s fill="%s"%s/>' % (g[:4] + (r, f, o)))
            elif kind == "ellipse":
                out.append('<ellipse cx="%g" cy="%g" rx="%g" ry="%g" fill="%s"%s/>' % (g + (f, o)))
            else:
                pts = " ".join("%g,%g" % p for p in g)
                out.append('<polygon points="%s" fill="%s"%s/>' % (pts, f, o))
        return out


CALF_BODY = """<rect x="-44" y="10" width="13" height="50" rx="5" fill="#d9d5cc"/>
<rect x="40" y="10" width="13" height="50" rx="5" fill="#d9d5cc"/>
<path d="M-66,-22 C-84,-14 -82,14 -78,30" stroke="#2b2b2e" stroke-width="4" fill="none" stroke-linecap="round"/><ellipse cx="-77" cy="35" rx="5" ry="9" fill="#2b2b2e"/>
<path d="M-68,-30 C-60,-44 44,-46 62,-34 C76,-24 74,16 58,24 C30,34 -40,34 -60,24 C-74,16 -76,-14 -68,-30Z" fill="#f7f4ee"/>
%s
<rect x="-58" y="12" width="14" height="50" rx="5" fill="#f7f4ee"/><rect x="-58" y="52" width="14" height="10" rx="3" fill="#3a3430"/>
<rect x="26" y="12" width="14" height="50" rx="5" fill="#f7f4ee"/><rect x="26" y="52" width="14" height="10" rx="3" fill="#3a3430"/>
<rect x="-44" y="52" width="13" height="8" rx="3" fill="#3a3430"/>
<rect x="40" y="52" width="13" height="8" rx="3" fill="#3a3430"/>
<path d="M54,-32 C70,-44 84,-46 92,-40 L98,-14 C86,-6 72,0 60,4Z" fill="#f7f4ee"/>
<path d="M86,-54 C104,-60 122,-52 126,-34 C130,-18 122,-6 110,-4 C96,-2 86,-14 84,-30 C83,-40 83,-50 86,-54Z" fill="#2b2b2e"/>
<path d="M104,-54 C110,-44 110,-30 106,-20 L100,-20 C102,-32 100,-44 96,-55Z" fill="#f7f4ee"/>
<ellipse cx="116" cy="-11" rx="17" ry="12" fill="#f3c6bf"/><circle cx="111" cy="-12" r="2" fill="#6d4a46"/><circle cx="122" cy="-12" r="2" fill="#6d4a46"/>
<circle cx="113" cy="-38" r="3.4" fill="#fff"/><circle cx="114" cy="-38" r="1.9" fill="#151515"/>
<path d="M86,-52 C78,-60 68,-58 64,-50 C72,-46 80,-46 87,-46Z" fill="#2b2b2e"/>
<path d="M95,-58 C95,-68 101,-72 105,-70 C105,-64 103,-59 101,-56Z" fill="#e9dfc9"/>
<path d="M112,-55 C114,-64 120,-68 124,-65 C123,-60 120,-55 116,-52Z" fill="#e9dfc9"/>"""

SPOTS = [
    '<path d="M-66,-20 C-56,-40 -30,-40 -24,-22 C-18,-6 -30,14 -48,12 C-64,10 -72,-6 -66,-20Z" fill="#2b2b2e"/>'
    '<path d="M-6,-42 C14,-44 30,-34 26,-18 C22,-6 -2,-4 -10,-16 C-16,-26 -16,-38 -6,-42Z" fill="#2b2b2e"/>'
    '<path d="M34,2 C46,-2 60,2 58,14 C54,24 34,26 30,16 C28,10 30,4 34,2Z" fill="#2b2b2e"/>',
    '<path d="M-30,-41 C-8,-44 6,-30 -4,-12 C-12,2 -40,-2 -44,-18 C-46,-28 -42,-38 -30,-41Z" fill="#2b2b2e"/>'
    '<path d="M28,-38 C46,-38 60,-26 56,-8 C52,6 28,6 22,-8 C18,-20 18,-34 28,-38Z" fill="#2b2b2e"/>',
    '<path d="M-58,-36 C-36,-44 -16,-36 -16,-16 C-16,2 -44,6 -56,-6 C-66,-16 -66,-30 -58,-36Z" fill="#2b2b2e"/>'
    '<path d="M14,-40 C34,-42 52,-34 50,-16 C48,-2 22,0 12,-12 C4,-22 4,-36 14,-40Z" fill="#2b2b2e"/>'
    '<path d="M-22,8 C-8,4 8,8 6,20 C4,30 -24,30 -28,20 C-30,14 -28,10 -22,8Z" fill="#2b2b2e"/>',
]


def calf(x, y, s, spots, facing=1):
    """A drawn calf (head towards +x when facing=1). Returns its SVG and its bounding box."""
    svg = '<g transform="translate(%g,%g) scale(%g,%g)">\n%s\n</g>' % (x, y, s * facing, s, CALF_BODY % SPOTS[spots])
    lo, hi = (-84 * s, 130 * s) if facing > 0 else (-130 * s, 84 * s)
    return svg, (x + lo, y - 72 * s, x + hi, y + 62 * s)


def brackets(x0, y0, x1, y1, colour, arm=22, r=8, width=4.5):
    """Corner brackets around a box, as in a tracking overlay."""
    s = []
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        s.append('<path d="M%g,%g L%g,%g Q%g,%g %g,%g L%g,%g" fill="none" stroke="%s" stroke-width="%g" '
                 'stroke-linecap="round"/>' % (cx + dx * arm, cy, cx + dx * r, cy, cx, cy, cx, cy + dy * r,
                                              cx, cy + dy * arm, colour, width))
    return "\n".join(s)


def tick_badge(cx, cy, colour):
    return ('<circle cx="%g" cy="%g" r="10" fill="%s" stroke="#ffffff" stroke-width="2.5"/>'
            '<path d="M%g,%g l3.6,3.6 l6.4,-7.2" fill="none" stroke="#ffffff" stroke-width="2.6" '
            'stroke-linecap="round" stroke-linejoin="round"/>' % (cx, cy, colour, cx - 5, cy + 0.4))


def cross_badge(cx, cy, colour):
    return ('<circle cx="%g" cy="%g" r="9" fill="%s" stroke="#ffffff" stroke-width="2.5"/>'
            '<path d="M%g,%g l7,7 M%g,%g l-7,7" stroke="#ffffff" stroke-width="2.4" stroke-linecap="round"/>'
            % (cx, cy, colour, cx - 3.5, cy - 3.5, cx + 3.5, cy - 3.5))


def person(sc, parts):
    """Add a person as sample-only shapes (never drawn). Returns the person's box."""
    xs, ys = [], []
    for kind, args in parts:
        if kind == "rect":
            x, y, w, h, fill = args
            sc.rect(x, y, w, h, fill, draw=False)
            xs += [x, x + w]
            ys += [y, y + h]
        elif kind == "ellipse":
            cx, cy, rx, ry, fill = args
            sc.ellipse(cx, cy, rx, ry, fill, draw=False)
            xs += [cx - rx, cx + rx]
            ys += [cy - ry, cy + ry]
        else:
            pts, fill = args
            sc.poly(pts, fill, draw=False)
            xs += [p[0] for p in pts]
            ys += [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def banner():
    sc = Scene()
    wall = sc.grad("wall", 60, 300, [(0, "#d8b07a"), (1, "#c49159")])
    ceiling = sc.grad("ceil", 0, 70, [(0, "#6e4c31"), (1, "#8a6442")])
    sky = sc.grad("sky", 104, 236, [(0, "#86c6e8"), (.6, "#cfe9ef"), (1, "#fbe6bb")])
    field = sc.grad("field", 226, 300, [(0, "#a9cf6f"), (1, "#86b85c")])
    straw = sc.grad("straw", 292, 420, [(0, "#ecd391"), (1, "#d8b76a")])
    panel = sc.grad("panel", 236, 300, [(0, "#f3f4ef"), (1, "#d9dcd3")])

    # back wall, planks and ceiling
    sc.rect(0, 0, W, 300, wall)
    for x in range(22, W, 44):
        sc.rect(x, 64, 3, 236, "#a97a45", op=.45)
    sc.rect(0, 0, W, 66, ceiling)
    for x0 in range(-120, W + 120, 240):          # rafters
        sc.poly([(x0, 66), (x0 + 120, 4), (x0 + 134, 4), (x0 + 14, 66)], "#5f412a", op=.85)
        sc.poly([(x0 + 240, 66), (x0 + 120, 4), (x0 + 106, 4), (x0 + 226, 66)], "#5f412a", op=.85)
    for x in (430, 850):                          # skylights
        sc.rect(x, 14, 130, 30, "#5a3f29", rx=3)
        sc.rect(x + 5, 18, 120, 22, "#d9eef3", rx=2)
        sc.rect(x + 63, 18, 4, 22, "#5a3f29")
    sc.rect(0, 58, W, 14, "#5a3d27")              # tie beam

    # open door on the left, with the field outside
    sc.rect(58, 90, 254, 214, "#6b4a2e", rx=4)
    sc.rect(72, 104, 226, 200, sky)
    sc.poly([(72, 236), (130, 214), (200, 222), (260, 210), (298, 218), (298, 304), (72, 304)], "#9cc77b")
    sc.rect(72, 232, 226, 72, field)
    sc.raw('<g transform="translate(112,232)"><rect x="-3" y="-4" width="6" height="18" fill="#7a5a3c"/>'
           '<ellipse cx="0" cy="-16" rx="18" ry="21" fill="#5f9e5a"/><ellipse cx="-7" cy="-22" rx="10" ry="11" '
           'fill="#74b56b"/></g>')
    sc.raw('<g fill="#ffffff" opacity=".85"><ellipse cx="160" cy="132" rx="30" ry="9"/>'
           '<ellipse cx="146" cy="126" rx="14" ry="8"/><ellipse cx="172" cy="124" rx="17" ry="10"/></g>')
    # the sliding door, pushed to the left
    sc.rect(0, 94, 66, 210, "#b5463a")
    sc.rect(0, 94, 66, 8, "#f2ece0")
    sc.rect(0, 296, 66, 8, "#f2ece0")
    sc.rect(58, 94, 8, 210, "#f2ece0")
    sc.poly([(0, 104), (6, 102), (58, 288), (52, 292)], "#f2ece0")
    sc.poly([(52, 102), (58, 104), (6, 292), (0, 288)], "#f2ece0")
    sc.rect(0, 84, 330, 8, "#4b3322", rx=3)       # door rail

    # a small, far-away person in the doorway (a hard case)
    far = person(sc, [
        ("ellipse", (247, 247, 4.5, 5, "#d6a57f")),
        ("rect", (241.5, 252, 11, 17, "#365a8c")),
        ("rect", (242, 268, 4.5, 15, "#2a3240")),
        ("rect", (247.5, 268, 4.5, 15, "#2a3240")),
    ])

    # window and lamp on the right
    sc.rect(1078, 104, 124, 86, "#6b4a2e", rx=4)
    sc.rect(1086, 112, 108, 70, "#d9eef3")
    sc.rect(1138, 112, 4, 70, "#6b4a2e")
    sc.rect(1086, 145, 108, 4, "#6b4a2e")

    # a person behind the pen wall (only the upper body shows)
    behind = person(sc, [
        ("ellipse", (1000, 178, 15, 9, "#3b2f2a")),
        ("ellipse", (1000, 188, 13.5, 16, "#c98f6a")),
        ("rect", (975, 203, 50, 40, "#9c3b2b")),
        ("rect", (962, 214, 15, 22, "#8a3222")),
        ("rect", (1023, 214, 15, 22, "#8a3222")),
        ("rect", (956, 228, 88, 10, "#9c3b2b")),
    ])
    behind[3] = 240                                # the wall hides the rest

    # pens: rails, posts and low walls
    sc.rect(330, 204, W - 330, 6, "#9fb0b5", rx=3)
    sc.rect(330, 220, W - 330, 5, "#9fb0b5", rx=2.5)
    for x in range(336, W, 186):
        sc.rect(x, 196, 9, 104, "#8a9ca2", rx=3)
    sc.rect(330, 236, W - 330, 64, panel)
    for x in range(424, W, 186):
        sc.rect(x, 236, 2, 64, "#c9cdc4")
    sc.rect(330, 236, W - 330, 4, "#ffffff", op=.7)

    # straw floor
    sc.rect(0, 294, W, 126, straw)
    sc.rect(0, 294, W, 6, "#c9a75d", op=.6)
    for x, y, w in ((96, 330, 26), (300, 396, 30), (360, 330, 22), (560, 404, 30), (610, 318, 20),
                    (740, 400, 28), (980, 404, 26), (1010, 320, 22), (1230, 396, 30), (1220, 330, 20),
                    (40, 404, 24), (690, 350, 18)):
        sc.raw('<path d="M%g,%g q%g,-5 %g,0" stroke="#c2a058" stroke-width="3" fill="none" '
               'stroke-linecap="round"/>' % (x, y, w / 2.0, w))
    # hay bale and a bucket at the right
    sc.raw('<g transform="translate(1214,350)"><rect x="-58" y="-34" width="116" height="62" rx="10" fill="#e3c26b"/>'
           '<rect x="-58" y="-34" width="116" height="14" rx="7" fill="#efd58c"/>'
           '<path d="M-28,-34 v62 M28,-34 v62" stroke="#b78f3a" stroke-width="4"/>'
           '<path d="M-46,-8 q10,-4 20,0 M8,6 q10,-4 20,0 M-40,14 q8,-4 16,0" stroke="#c9a14b" stroke-width="2.5" '
           'fill="none" stroke-linecap="round"/></g>')

    # the worker with a bucket, in front of the pens
    worker = person(sc, [
        ("ellipse", (470, 189, 15, 17, "#d9a27c")),
        ("ellipse", (470, 177, 17, 9, "#2f5d4a")),
        ("rect", (470, 176, 24, 5, "#2f5d4a")),
        ("rect", (464, 203, 12, 6, "#c88f6c")),
        ("rect", (444, 207, 52, 76, "#4a78b0")),
        ("rect", (452, 226, 36, 56, "#2f4f7a")),
        ("rect", (432, 211, 13, 62, "#4a78b0")),
        ("ellipse", (438.5, 278, 7, 8, "#d9a27c")),
        ("rect", (495, 211, 13, 58, "#4a78b0")),
        ("ellipse", (502, 273, 7, 7, "#d9a27c")),
        ("poly", ([(488, 281), (522, 281), (517, 318), (493, 318)], "#9aa7ad")),
        ("rect", (450, 281, 19, 96, "#2f4f7a")),
        ("rect", (471, 281, 19, 96, "#2f4f7a")),
        ("rect", (446, 362, 25, 42, "#2d5a3d")),
        ("rect", (469, 362, 25, 42, "#2d5a3d")),
    ])

    # calves: drawn on top, never under a mosaic
    calves = [calf(180, 374, .9, 0), calf(636, 354, .8, 1), calf(906, 356, .86, 2, facing=-1),
              calf(1086, 372, .62, 1, facing=-1)]
    for svg, _box in calves:
        sc.raw(svg)

    veils = [sc.mosaic(b) for b in (far, behind, worker)]
    for _cells, (x0, y0, x1, y1) in veils:
        for _svg, (a0, b0, a1, b1) in calves:
            assert a1 <= x0 or a0 >= x1 or b1 <= y0 or b0 >= y1, "a calf overlaps a mosaic"

    defs = "".join(g.svg() for g in sc.grads)
    defs += ('<linearGradient id="beam" x1="1" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff8e0" '
             'stop-opacity=".38"/><stop offset="1" stop-color="#fff8e0" stop-opacity="0"/></linearGradient>'
             '<clipPath id="frame"><rect width="%d" height="%d" rx="16"/></clipPath>' % (W, H))
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
           'aria-labelledby="t d">' % (W, H, W, H),
           '<title id="t">OpenVeil: people veiled, calves visible</title>',
           '<desc id="d">A drawn calf barn watched by a camera. Three people, one far away in the doorway, one '
           'behind a pen wall and one carrying a bucket, are each covered by a pixel mosaic with a green tick. '
           'The calves stay visible; a dashed box with a cross on one calf marks a candidate the verifier '
           'rejected. No real people or footage.</desc>',
           '<defs>%s</defs>' % defs, '<g clip-path="url(#frame)">']
    out += sc.svg_items()
    green, amber = "#1f9d6a", "#d99a00"
    for cells, (x0, y0, x1, y1) in veils:
        out.append('<g shape-rendering="crispEdges">')
        out += ['<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/>' % c for c in cells]
        out.append('</g>')
    # light from the camera, then the overlays
    out.append('<path d="M1150,96 L520,420 L1040,420Z" fill="url(#beam)"/>')
    for i, (_cells, (x0, y0, x1, y1)) in enumerate(veils):
        m = 3 if i == 0 else 4
        out.append(brackets(x0 - m, y0 - m, x1 + m, min(y1 + m, H - 5), green,
                            arm=12 if i == 0 else 22, r=5 if i == 0 else 8, width=3.5 if i == 0 else 4.5))
        out.append(tick_badge((x0 + x1) / 2.0, max(y0 - m - 14, 76), green))
    _svg, (a0, b0, a1, b1) = calves[2]
    out.append('<rect x="%g" y="%g" width="%g" height="%g" rx="14" fill="none" stroke="%s" stroke-width="2.5" '
               'stroke-dasharray="7 6" opacity=".95"/>' % (a0 - 4, b0 - 6, a1 - a0 + 8, b1 - b0 + 10, amber))
    out.append(cross_badge(a1 + 4, b0 - 6, amber))
    # the camera on the tie beam
    out.append('<g><rect x="1176" y="66" width="10" height="22" fill="#4b5a62"/>'
               '<g transform="rotate(-24 1170 96)">'
               '<rect x="1130" y="80" width="80" height="34" rx="9" fill="#f2f5f6"/>'
               '<rect x="1130" y="80" width="80" height="11" rx="5.5" fill="#d6dee2"/>'
               '<rect x="1114" y="85" width="20" height="24" rx="5" fill="#2e3b43"/>'
               '<circle cx="1114" cy="97" r="9.5" fill="#14323f"/><circle cx="1114" cy="97" r="5.2" fill="#37b6c9"/>'
               '<circle cx="1111" cy="94" r="1.7" fill="#ffffff"/><circle cx="1196" cy="97" r="3.8" fill="#ff4d4d"/>'
               '</g></g>')
    out.append('</g></svg>')
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# Pipeline diagrams
# --------------------------------------------------------------------------------------

STYLE = """
<style>
  text { font-family: %s; fill: #24292f; }
  .t { font-size: 14px; font-weight: 600; }
  .d { font-size: 12px; fill: #57606a; }
  .h { font-size: 12px; font-weight: 600; letter-spacing: .04em; fill: #57606a; }
  .box { fill: #ffffff; stroke: #d0d7de; stroke-width: 1; }
  .box.key { fill: #e8f3ee; stroke: #2f7d5b; }
  .box.human { fill: #fff8e8; stroke: #c98500; }
  .box.out { fill: #e9f1f8; stroke: #2a6f9f; }
  .band { fill: #f6f8fa; stroke: none; }
  .edge { stroke: #57606a; stroke-width: 1.5; fill: none; }
  .edge.dash { stroke-dasharray: 4 3; }
  .arrow { fill: #57606a; }
  @media (prefers-color-scheme: dark) {
    text { fill: #e6edf3; }
    .d, .h { fill: #9da7b3; }
    .box { fill: #161b22; stroke: #3d444d; }
    .box.key { fill: #12261e; stroke: #4aa37c; }
    .box.human { fill: #2a2110; stroke: #c98500; }
    .box.out { fill: #10202e; stroke: #4b8fc4; }
    .band { fill: #11161c; }
    .edge { stroke: #9da7b3; } .arrow { fill: #9da7b3; }
  }
</style>""" % FONT

NODES = {
    "video": ("1  Videos", ["Any number, any length.", "Decoded in 60 s chunks"], ""),
    "sched": ("2  Scan schedule", ["A check every 1 s; every 0.2 s", "after a hit. 0.0508 calls/frame", "(case study, sparse at night)"], ""),
    "prop": ("3  Proposer", ["Grounding DINO, text prompt.", "Keeps boxes scoring >= 0.05:", "recall first"], ""),
    "verif": ("4  Verifier", ["DINOv3 ViT-L/16 + LoRA crop", "classifier. Keeps a box if", "p(person) >= 0.10"], "key"),
    "table": ("5  Box table", ["Box rules, gap interpolation.", "One small .json.gz per video;", "rules re-apply without a GPU"], ""),
    "review": ("6  Review tool", ["A person checks the hard", "frames: 0.6 s median per", "frame. Writes overrides"], "human"),
    "mosaic": ("7  Mosaic + smart cut", ["Box +20% per side, held ±2 s.", "Only GOPs with a mosaic are", "re-encoded: 12.0% of packets"], ""),
    "release": ("8  Verified release", ["Each file checked against", "its source: all 1,023 masked", "videos passed"], "out"),
}


def node(x, y, w, h, key):
    title, lines, cls = NODES[key]
    s = ['<rect class="box %s" x="%s" y="%s" width="%s" height="%s" rx="8"/>' % (cls, x, y, w, h),
         '<text class="t" x="%s" y="%s">%s</text>' % (x + 12, y + 23, escape(title))]
    for i, line in enumerate(lines):
        line = line.replace(">=", "≥")
        s.append('<text class="d" x="%s" y="%s">%s</text>' % (x + 12, y + 43 + i * 16, escape(line)))
    return "\n".join(s)


def arrow(pts, dash=False):
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    if x1 > x0:
        tri, end = "%s,%s %s,%s %s,%s" % (x1, y1, x1 - 8, y1 - 4, x1 - 8, y1 + 4), (x1 - 8, y1)
    elif x1 < x0:
        tri, end = "%s,%s %s,%s %s,%s" % (x1, y1, x1 + 8, y1 - 4, x1 + 8, y1 + 4), (x1 + 8, y1)
    elif y1 > y0:
        tri, end = "%s,%s %s,%s %s,%s" % (x1, y1, x1 - 4, y1 - 8, x1 + 4, y1 - 8), (x1, y1 - 8)
    else:
        tri, end = "%s,%s %s,%s %s,%s" % (x1, y1, x1 - 4, y1 + 8, x1 + 4, y1 + 8), (x1, y1 + 8)
    d = "M" + " L".join("%s %s" % p for p in pts[:-1] + [end])
    return '<path class="edge%s" d="%s"/><polygon class="arrow" points="%s"/>' % (" dash" if dash else "", d, tri)


PIPE_TITLE = "OpenVeil pipeline"
PIPE_DESC = ("Detection on a GPU: videos, a scan schedule, the Grounding DINO proposer and the DINOv3 crop "
             "verifier write one box table per video. Then, without a GPU: a person checks the hard frames in "
             "the review tool, the boxes are enlarged, held and mosaicked with a smart cut, and every masked "
             "file is checked against its source before release.")


def _head(w, h, uid):
    return ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
            'aria-labelledby="%s-t %s-d">' % (w, h, w, h, uid, uid),
            '<title id="%s-t">%s</title>' % (uid, PIPE_TITLE),
            '<desc id="%s-d">%s</desc>' % (uid, escape(PIPE_DESC)), STYLE]


def pipeline():
    w, bw, bh, gap = 1000, 222, 104, 28
    xs = [20 + i * (bw + gap) for i in range(4)]
    y1, y2 = 56, 228
    s = _head(w, 352, "pw")
    s += ['<rect class="band" x="4" y="8" width="%d" height="164" rx="12"/>' % (w - 8),
          '<rect class="band" x="4" y="180" width="%d" height="164" rx="12"/>' % (w - 8),
          '<text class="h" x="20" y="36">DETECT, ON A GPU, ONCE PER VIDEO</text>',
          # starts right of the connector from box 4, which comes down at x = 131
          '<text class="h" x="146" y="208">REVIEW AND APPLY, NO GPU NEEDED</text>']
    for x, k in zip(xs, ("video", "sched", "prop", "verif")):
        s.append(node(x, y1, bw, bh, k))
    for x, k in zip(xs, ("table", "review", "mosaic", "release")):
        s.append(node(x, y2, bw, bh, k))
    for i in range(3):
        s.append(arrow([(xs[i] + bw, y1 + bh // 2), (xs[i + 1], y1 + bh // 2)]))
        s.append(arrow([(xs[i] + bw, y2 + bh // 2), (xs[i + 1], y2 + bh // 2)]))
    s.append(arrow([(xs[3] + bw // 2, y1 + bh), (xs[3] + bw // 2, 176), (xs[0] + bw // 2, 176),
                    (xs[0] + bw // 2, y2)]))
    s.append("</svg>")
    return "\n".join(s)


def pipeline_narrow():
    w, bw, bh, gap = 340, 300, 100, 24
    keys = ("video", "sched", "prop", "verif", "table", "review", "mosaic", "release")
    x, y0 = 20, 40
    h = y0 + len(keys) * (bh + gap)
    s = _head(w, h, "pn")
    s.append('<text class="h" x="20" y="24">ON A GPU (1-4), THEN NO GPU (5-8)</text>')
    for i, k in enumerate(keys):
        y = y0 + i * (bh + gap)
        s.append(node(x, y, bw, bh, k))
        if i:
            s.append(arrow([(x + bw // 2, y - gap), (x + bw // 2, y)]))
    s.append("</svg>")
    return "\n".join(s)


if __name__ == "__main__":
    out = {"banner.svg": banner(), "pipeline.svg": pipeline(), "pipeline_narrow.svg": pipeline_narrow()}
    assets = os.path.join(HERE, "..", "docs", "assets")
    os.makedirs(assets, exist_ok=True)
    for name, text in out.items():
        with open(os.path.join(HERE, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
        shutil.copy(os.path.join(HERE, name), assets)
        print("wrote figures/%s and docs/assets/%s" % (name, name))
