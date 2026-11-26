# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Haiyu Yang and Miel Hostens
"""Draw the repository figures as plain SVG. Standard library only.

    python figures/make_figures.py

Writes figures/banner.svg, figures/pipeline.svg and figures/pipeline_narrow.svg,
then copies them to docs/assets/ for the project page. Every figure is drawn:
there is no footage and no real person in any of them.

The banner shows four scenes side by side, each watched by a camera: a calf barn,
a factory line, a laboratory bench and a meadow with a deer. Each scene is a small
scene graph. The people in it are never written to the SVG: they exist only as
shapes that are sampled when the mosaic is computed. As in OpenVeil, each person
box is enlarged by 20% per side, and every cell of a fixed grid that touches it is
filled with the mean colour of the scene under that cell. What the footage is about
(calves, a robot arm, instruments, a deer) is drawn on top and is checked to stay
clear of every mosaic. A dashed box with a cross, on one calf and on the deer, stands
for a candidate that the proposer suggested and the verifier rejected.

The pipeline figures follow the viewer's light or dark setting. In both, nodes 3 and
4 are marked as a pair: a detector that proposes broadly, then a classifier that
verifies each candidate.
"""
import math
import os
import shutil
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


# --------------------------------------------------------------------------------------
# Banner: four drawn scenes, each watched by a camera, where the people are veiled
# --------------------------------------------------------------------------------------

W, H = 1280, 420
TW, TH = 298, 380         # one scene tile
MARGIN, GUTTER = 20, 16   # 20 + 4 * 298 + 3 * 16 + 20 = 1280
CELL = 16                 # OpenVeil: max(16, round(height / 36)) px, i.e. 16 px for a 380 px tile
PAD = 0.20                # OpenVeil: each box is enlarged by 20% per side
EDGE = 12                 # every overlay (brackets, dashed box) keeps at least this far from its tile's edge
GREEN, SLATE = "#1f9d6a", "#34495c"   # green: a person, masked; slate: a candidate the verifier left alone


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
    """One tile, in its own coordinates (0..TW, 0..TH). Shapes are kept in drawing order;
    each can be drawn, sampled, or both."""

    def __init__(self, uid, label, label_width):
        self.uid, self.label, self.label_width = uid, label, label_width
        self.items = []      # (kind, geometry, fill, opacity, draw, sample)
        self.grads = []
        self.people = []     # boxes of the people; the people themselves are sample-only
        self.subjects = []   # boxes of what the footage is about; must stay clear of every mosaic
        self.rejected = []   # (box, side of its badge): candidates the verifier rejected
        self.camera = (256, 44)

    def grad(self, name, y0, y1, stops):
        g = VGrad("%s-%s" % (self.uid, name), y0, y1, stops)
        self.grads.append(g)
        return g

    def rect(self, x, y, w, h, fill, rx=0, op=1.0, draw=True, sample=True):
        self.items.append(("rect", (x, y, w, h, rx), fill, op, draw, sample))

    def ellipse(self, cx, cy, rx, ry, fill, op=1.0, draw=True, sample=True):
        self.items.append(("ellipse", (cx, cy, rx, ry), fill, op, draw, sample))

    def poly(self, pts, fill, op=1.0, draw=True, sample=True):
        self.items.append(("poly", tuple(pts), fill, op, draw, sample))

    def bar(self, x0, y0, x1, y1, width, fill, op=1.0):
        """A straight bar from one point to another, as a four-point polygon."""
        n = math.hypot(x1 - x0, y1 - y0)
        px, py = -(y1 - y0) / n * width / 2.0, (x1 - x0) / n * width / 2.0
        self.poly([(x0 + px, y0 + py), (x1 + px, y1 + py), (x1 - px, y1 - py), (x0 - px, y0 - py)], fill, op=op)

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
        """Cells of the tile's fixed grid that touch the enlarged box, each with its mean colour."""
        x0, y0, x1, y1 = box
        px, py = (x1 - x0) * PAD, (y1 - y0) * PAD
        x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(TW, x1 + px), min(TH, y1 + py)
        cells = []
        for gy in range(int(y0 // CELL), int(-(-y1 // CELL))):
            for gx in range(int(x0 // CELL), int(-(-x1 // CELL))):
                cx, cy = gx * CELL, gy * CELL
                cw, ch = min(CELL, TW - cx), min(CELL, TH - cy)
                acc = [0.0, 0.0, 0.0]
                for i in range(n):
                    for j in range(n):
                        col = self.colour_at(cx + (i + .5) * cw / n, cy + (j + .5) * ch / n)
                        acc = [a + b for a, b in zip(acc, col)]
                cells.append((cx, cy, cw, ch, _hex([a / (n * n) for a in acc])))
        bounds = (min(c[0] for c in cells), min(c[1] for c in cells),
                  max(c[0] + c[2] for c in cells), max(c[1] + c[3] for c in cells))
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
                pts = " ".join("%g,%g" % (round(p[0], 2), round(p[1], 2)) for p in g)
                out.append('<polygon points="%s" fill="%s"%s/>' % (pts, f, o))
        return out


def person(sc, cx, feet, s, skin, hat, top, mid, legs, shoes, extra=(), cut=None):
    """Add a standing person as sample-only shapes: they are never drawn, and reach the SVG only as
    the mean colours of mosaic cells. (cx, feet) is the point between the feet, s the scale; `extra`
    holds more parts in the same units (x to the right of cx, y above the feet is negative). `cut` is
    the y below which something in front hides the person. Records and returns the person's box."""
    parts = [
        ("ellipse", (0, -215, 15, 17), skin), ("ellipse", (0, -227, 17, 9), hat), ("rect", (0, -228, 24, 5), hat),
        ("rect", (-6, -201, 12, 6), skin), ("rect", (-26, -197, 52, 76), top), ("rect", (-18, -178, 36, 56), mid),
        ("rect", (-38, -193, 13, 62), top), ("ellipse", (-31.5, -126, 7, 8), skin),
        ("rect", (25, -193, 13, 58), top), ("ellipse", (32, -131, 7, 7), skin),
        ("rect", (-20, -123, 19, 96), legs), ("rect", (1, -123, 19, 96), legs),
        ("rect", (-24, -42, 25, 42), shoes), ("rect", (-1, -42, 25, 42), shoes),
    ] + list(extra)
    xs, ys = [], []
    for kind, g, fill in parts:
        if kind == "rect":
            x, y, w, h = cx + g[0] * s, feet + g[1] * s, g[2] * s, g[3] * s
            sc.rect(x, y, w, h, fill, draw=False)
            xs += [x, x + w]
            ys += [y, y + h]
        elif kind == "ellipse":
            x, y, rx, ry = cx + g[0] * s, feet + g[1] * s, g[2] * s, g[3] * s
            sc.ellipse(x, y, rx, ry, fill, draw=False)
            xs += [x - rx, x + rx]
            ys += [y - ry, y + ry]
        else:
            pts = [(cx + px * s, feet + py * s) for px, py in g]
            sc.poly(pts, fill, draw=False)
            xs += [p[0] for p in pts]
            ys += [p[1] for p in pts]
    box = [min(xs), min(ys), max(xs), max(ys) if cut is None else cut]
    sc.people.append(box)
    return box


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
]

DEER_BODY = """<g fill="none" stroke="#8a5a36" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round">
<path d="M-28,10 L-35,36 L-31,62"/><path d="M27,12 L31,36 L28,62"/></g>
<ellipse cx="-50" cy="-10" rx="6" ry="9" fill="#fbf6ea"/>
<path d="M-48,-16 C-42,-30 18,-32 38,-26 C52,-20 52,8 40,15 C18,24 -26,24 -42,14 C-54,6 -54,-6 -48,-16Z" fill="#b9793f"/>
<path d="M-36,15 C-16,23 18,23 38,15 C22,27 -20,27 -36,15Z" fill="#e3b987"/>
<g fill="none" stroke="#b9793f" stroke-width="7" stroke-linecap="round" stroke-linejoin="round">
<path d="M-38,8 C-45,24 -39,36 -41,62"/><path d="M34,10 L35,62"/></g>
<g fill="none" stroke="#3a2a1e" stroke-width="7" stroke-linecap="round"><path d="M-41,59 v3"/><path d="M35,59 v3"/>
<path d="M-31,60 v2" stroke-width="6.5"/><path d="M28,60 v2" stroke-width="6.5"/></g>
<path d="M26,-24 C34,-42 38,-54 44,-64 L60,-58 C57,-40 54,-18 48,2Z" fill="#b9793f"/>
<path d="M46,-76 C39,-86 41,-94 48,-90 C52,-84 53,-78 52,-72Z" fill="#9a6232"/>
<path d="M44,-70 C50,-78 62,-78 70,-72 L80,-64 C83,-60 80,-55 75,-55 L58,-54 C48,-54 40,-62 44,-70Z" fill="#b9793f"/>
<ellipse cx="78" cy="-60" rx="3.6" ry="3.2" fill="#2b2320"/><circle cx="62" cy="-67" r="2.2" fill="#2b2320"/>
<g fill="none" stroke="#efe6d2" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round">
<path d="M53,-76 C50,-90 54,-100 62,-108"/><path d="M52,-88 L44,-96"/><path d="M56,-99 L50,-108"/>
<path d="M58,-76 C62,-88 70,-95 79,-98"/><path d="M65,-89 L73,-86"/></g>"""


def calf(x, y, s, spots, facing=1):
    """A drawn calf (head towards +x when facing=1). Returns its SVG and its bounding box."""
    svg = '<g transform="translate(%g,%g) scale(%g,%g)">\n%s\n</g>' % (x, y, s * facing, s, CALF_BODY % SPOTS[spots])
    lo, hi = (-84 * s, 130 * s) if facing > 0 else (-130 * s, 84 * s)
    return svg, (x + lo, y - 72 * s, x + hi, y + 62 * s)


def deer(x, y, s, facing=1):
    """A drawn deer (head towards +x when facing=1). Returns its SVG and its bounding box."""
    svg = '<g transform="translate(%g,%g) scale(%g,%g)">\n%s\n</g>' % (x, y, s * facing, s, DEER_BODY)
    lo, hi = (-58 * s, 84 * s) if facing > 0 else (-84 * s, 58 * s)
    return svg, (x + lo, y - 111 * s, x + hi, y + 66 * s)


def brackets(x0, y0, x1, y1, colour, arm=18, r=7, width=4):
    """Corner brackets around a box, as in a tracking overlay, on a white edge so they show on any ground."""
    d = []
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.append("M%g,%g L%g,%g Q%g,%g %g,%g L%g,%g" % (cx + dx * arm, cy, cx + dx * r, cy, cx, cy, cx, cy + dy * r,
                                                       cx, cy + dy * arm))
    d = " ".join(d)
    return ('<path d="%s" fill="none" stroke="#ffffff" stroke-width="%g" stroke-linecap="round"/>'
            '<path d="%s" fill="none" stroke="%s" stroke-width="%g" stroke-linecap="round"/>'
            % (d, width + 3, d, colour, width))


def tick_badge(cx, cy, colour):
    return ('<circle cx="%g" cy="%g" r="10" fill="%s" stroke="#ffffff" stroke-width="2.5"/>'
            '<path d="M%g,%g l3.6,3.6 l6.4,-7.2" fill="none" stroke="#ffffff" stroke-width="2.6" '
            'stroke-linecap="round" stroke-linejoin="round"/>' % (cx, cy, colour, cx - 5, cy + 0.4))


def cross_badge(cx, cy, colour):
    return ('<circle cx="%g" cy="%g" r="9" fill="%s" stroke="#ffffff" stroke-width="2.5"/>'
            '<path d="M%g,%g l7,7 M%g,%g l-7,7" stroke="#ffffff" stroke-width="2.4" stroke-linecap="round"/>'
            % (cx, cy, colour, cx - 3.5, cy - 3.5, cx + 3.5, cy - 3.5))


def camera(x, y, s=.62):
    """The camera, hanging from the point (x, y). Returns its SVG and the position of its lens.
    The housing has a dark outline and a shaded underside, so it shows on a pale wall too."""
    svg = ('<g transform="translate(%g,%g) scale(%g)"><rect x="-5" y="-4" width="10" height="20" fill="#3d4c56"/>'
           '<g transform="translate(-6,26) rotate(-24)">'
           '<rect x="-42.5" y="-19.5" width="85" height="39" rx="11.5" fill="#2c4050" fill-opacity=".8"/>'
           '<rect x="-40" y="-17" width="80" height="34" rx="9" fill="#f6f8f9"/>'
           '<path d="M-40,5 h80 v3 a9,9 0 0 1 -9,9 h-62 a9,9 0 0 1 -9,-9Z" fill="#b9c6cd"/>'
           '<rect x="-56" y="-12" width="20" height="24" rx="5" fill="#2e3b43"/>'
           '<circle cx="-56" cy="0" r="9.5" fill="#14323f"/><circle cx="-56" cy="0" r="5.2" fill="#37b6c9"/>'
           '<circle cx="-59" cy="-3" r="1.7" fill="#ffffff"/><circle cx="26" cy="0" r="3.8" fill="#ff4d4d"/>'
           '</g></g>' % (x, y, s))
    a = math.radians(-24)
    return svg, (x + s * (-6 - 56 * math.cos(a)), y + s * (26 - 56 * math.sin(a)))


# ---- the four scenes ---------------------------------------------------------------------

def barn():
    sc = Scene("barn", "CALF BARN", 71)
    wall = sc.grad("wall", 50, 252, [(0, "#d8b07a"), (1, "#c49159")])
    ceiling = sc.grad("ceil", 0, 50, [(0, "#6e4c31"), (1, "#8a6442")])
    sky = sc.grad("sky", 90, 200, [(0, "#86c6e8"), (.6, "#cfe9ef"), (1, "#fbe6bb")])
    field = sc.grad("field", 196, 256, [(0, "#a9cf6f"), (1, "#86b85c")])
    straw = sc.grad("straw", 248, 380, [(0, "#ecd391"), (1, "#d8b76a")])
    panel = sc.grad("panel", 200, 254, [(0, "#f3f4ef"), (1, "#d9dcd3")])

    # back wall, planks, ceiling and rafters
    sc.rect(0, 0, TW, 254, wall)
    for x in range(16, TW, 36):
        sc.rect(x, 50, 3, 204, "#a97a45", op=.45)
    sc.rect(0, 0, TW, 50, ceiling)
    for x0 in (-46, 144):
        sc.poly([(x0, 50), (x0 + 95, 4), (x0 + 107, 4), (x0 + 12, 50)], "#5f412a", op=.85)
        sc.poly([(x0 + 190, 50), (x0 + 95, 4), (x0 + 83, 4), (x0 + 178, 50)], "#5f412a", op=.85)
    sc.rect(0, 44, TW, 12, "#5a3d27")              # tie beam

    # open door on the left, with the field outside
    sc.rect(12, 80, 112, 176, "#6b4a2e", rx=4)
    sc.rect(22, 90, 92, 166, sky)
    sc.poly([(22, 200), (48, 186), (78, 192), (102, 182), (114, 188), (114, 256), (22, 256)], "#9cc77b")
    sc.rect(22, 198, 92, 58, field)
    sc.rect(40, 188, 5, 16, "#7a5a3c")
    sc.ellipse(42.5, 178, 13, 16, "#5f9e5a")
    sc.ellipse(38, 173, 7, 8, "#74b56b")
    for cx, cy, rx, ry in ((80, 116, 22, 7), (70, 111, 10, 6), (89, 110, 12, 7)):
        sc.ellipse(cx, cy, rx, ry, "#ffffff", op=.85)
    sc.rect(4, 72, 130, 7, "#4b3322", rx=3)        # door rail

    # pens: rails, posts and a low wall
    sc.rect(124, 172, TW - 124, 5, "#9fb0b5", rx=2.5)
    sc.rect(124, 186, TW - 124, 4, "#9fb0b5", rx=2)
    for x in (130, 208, 286):
        sc.rect(x, 164, 8, 90, "#8a9ca2", rx=3)
    sc.rect(124, 200, TW - 124, 54, panel)
    for x in (168, 246):
        sc.rect(x, 200, 2, 54, "#c9cdc4")
    sc.rect(124, 200, TW - 124, 4, "#ffffff", op=.7)

    # straw floor
    sc.rect(0, 248, TW, 132, straw)
    sc.rect(0, 248, TW, 5, "#c9a75d", op=.6)
    for x, y, w in ((150, 318, 20), (14, 270, 18), (154, 364, 24), (60, 284, 16)):
        sc.raw('<path d="M%g,%g q%g,-4 %g,0" stroke="#c2a058" stroke-width="2.5" fill="none" '
               'stroke-linecap="round"/>' % (x, y, w / 2.0, w))

    # the worker with a bucket, in front of the pens
    person(sc, 228, 322, .60, "#d9a27c", "#2f5d4a", "#4a78b0", "#2f4f7a", "#2f4f7a", "#2d5a3d",
           extra=[("poly", [(18, -123), (52, -123), (47, -86), (23, -86)], "#9aa7ad")])

    # calves: drawn on top, never under a mosaic
    small = calf(72, 244, .42, 1, facing=-1)
    big = calf(66, 326, .58, 0)
    for svg, box in (small, big):
        sc.raw(svg)
        sc.subjects.append(box)
    sc.rejected.append((big[1], 1))
    return sc


def factory():
    sc = Scene("factory", "FACTORY", 57)
    wall = sc.grad("wall", 46, 264, [(0, "#e3ebee"), (1, "#c6d3d9")])
    glass = sc.grad("glass", 68, 146, [(0, "#d9f0f4"), (1, "#a6d2dc")])
    floor = sc.grad("floor", 262, 380, [(0, "#bcc6cb"), (1, "#9eaab0")])

    # wall, two windows (the camera hangs on plain wall), roof truss
    sc.rect(0, 0, TW, 264, wall)
    for x in (16, 106):
        sc.rect(x, 64, 74, 86, "#51677a", rx=3)
        sc.rect(x + 5, 69, 64, 76, glass)
        sc.rect(x + 35, 69, 4, 76, "#51677a")
        sc.rect(x + 5, 105, 64, 4, "#51677a")
    sc.rect(0, 0, TW, 40, "#2c4050")
    for x0 in range(-30, TW, 80):
        sc.poly([(x0, 40), (x0 + 36, 0), (x0 + 44, 0), (x0 + 8, 40)], "#3d566b")
        sc.poly([(x0 + 80, 40), (x0 + 44, 0), (x0 + 36, 0), (x0 + 72, 40)], "#3d566b")
    sc.rect(0, 38, TW, 10, "#1f303d")
    # a pipe along the wall, and a dado
    sc.rect(0, 172, TW, 9, "#2a6f7f")
    sc.rect(0, 172, TW, 3, "#4c94a3")
    for x in (40, 146, 252):
        sc.rect(x, 168, 6, 17, "#1f5562", rx=2)
    sc.rect(0, 224, TW, 40, "#b3c1c8")
    sc.rect(0, 224, TW, 3, "#9eadb5")

    # floor with a safety line
    sc.rect(0, 262, TW, 118, floor)
    sc.rect(0, 360, TW, 7, "#e2b93b")

    # the operator stands behind the line: only the upper body shows
    person(sc, 78, 372, .88, "#8d5a3b", "#f2c230", "#d5df2a", "#f08a24", "#2c4050", "#1f303d", cut=268)

    # the robot's pedestal stands on the floor, behind the conveyor
    sc.rect(226, 226, 40, 128, "#33434e")
    sc.rect(226, 226, 6, 128, "#44565f")
    sc.rect(218, 351, 56, 7, "#26323b", rx=2)
    sc.rect(220, 216, 52, 12, "#2c3a44", rx=3)

    # conveyor: belt, frame, rollers, legs
    sc.rect(-10, 268, TW + 20, 12, "#33434e")
    sc.rect(-10, 268, TW + 20, 3, "#4a5c68")
    sc.rect(-10, 280, TW + 20, 16, "#2a6f7f")
    sc.rect(-10, 280, TW + 20, 3, "#4c94a3")
    for x in range(14, TW, 30):
        sc.ellipse(x, 289, 4, 4, "#d6dee2")
    for x in (30, 160):      # the line runs on out of the picture
        sc.rect(x, 296, 10, 58, "#4a5c68")
        sc.rect(x - 6, 352, 22, 6, "#2a3741", rx=2)
    sc.rect(30, 322, TW - 20, 5, "#4a5c68")

    # boxes: one on the belt, one leaving, one in the gripper
    def box(x, y, w, h):
        sc.rect(x, y, w, h, "#d9a868")
        sc.rect(x, y, w, 5, "#e6bd84")
        sc.rect(x + w / 2.0 - 4, y, 8, h, "#c08a4a")
    box(202, 234, 42, 34)
    box(272, 240, 36, 28)
    box(153, 208, 26, 26)

    # robot arm: shoulder, elbow, wrist, gripper
    (sx, sy), (ex, ey), (wx, wy) = (246, 208), (204, 130), (166, 176)
    sc.bar(sx, sy, ex, ey, 21, "#f28c28")
    sc.bar(sx, sy, ex, ey, 5, "#f7ad5a")
    sc.bar(ex, ey, wx, wy, 15, "#f28c28")
    sc.bar(ex, ey, wx, wy, 4, "#f7ad5a")
    for (jx, jy), r in (((sx, sy), 15), ((ex, ey), 12), ((wx, wy), 9)):
        sc.ellipse(jx, jy, r, r, "#2c3a44")
        sc.ellipse(jx, jy, r * .42, r * .42, "#d6dee2")
    sc.rect(wx - 4, wy + 7, 8, 14, "#3b4b56")
    sc.rect(wx - 18, wy + 20, 36, 6, "#2c3a44", rx=2)
    sc.rect(wx - 18, wy + 24, 5, 18, "#2c3a44", rx=1.5)
    sc.rect(wx + 13, wy + 24, 5, 18, "#2c3a44", rx=1.5)
    sc.subjects.append((148, 118, 272, 268))
    return sc


def laboratory():
    sc = Scene("lab", "LABORATORY", 84)
    wall = sc.grad("wall", 34, 252, [(0, "#eef4f4"), (1, "#d9e5e7")])
    glass = sc.grad("glass", 66, 152, [(0, "#d6eef2"), (1, "#b9dde6")])
    front = sc.grad("front", 262, 380, [(0, "#3b8795"), (1, "#2a6f7f")])

    # wall, ceiling, a window with blinds
    sc.rect(0, 0, TW, 252, wall)
    sc.rect(0, 0, TW, 34, "#c9d6da")
    sc.rect(0, 32, TW, 4, "#aebdc3")
    sc.rect(18, 60, 136, 98, "#aebdc3", rx=3)
    sc.rect(24, 66, 124, 86, glass)
    for y in range(68, 150, 11):
        sc.rect(24, y, 124, 4, "#f6f8f7", op=.9)
    sc.rect(84, 66, 4, 86, "#aebdc3")

    # a shelf with bottles
    sc.rect(172, 142, 116, 6, "#8a9ca2", rx=2)
    for x in (176, 276):
        sc.rect(x, 148, 5, 12, "#8a9ca2")
    for x, w, h, body, cap in ((180, 18, 30, "#c98500", "#2c4050"), (204, 16, 24, "#2a6f7f", "#f2f5f6"),
                               (226, 20, 34, "#f2f5f6", "#2a6f9f"), (252, 16, 26, "#58b78f", "#2c4050"),
                               (272, 12, 20, "#16324a", "#f2f5f6")):
        sc.rect(x, 142 - h, w, h, body, rx=3)
        sc.rect(x + w / 2.0 - 4, 142 - h - 6, 8, 7, cap, rx=1.5)
        sc.rect(x + 2, 142 - h * .62, w - 4, h * .3, "#ffffff", op=.55)

    # the researcher stands behind the bench: only the upper body shows
    person(sc, 102, 366, .92, "#f0c4a0", "#3b2f2a", "#5b93cf", "#eef3f4", "#2c4050", "#1f303d", cut=250)

    # bench: worktop and cupboards
    sc.rect(0, 262, TW, 118, front)
    for x in (4, 102, 200):
        sc.rect(x, 270, 94, 36, "#4c9aa8", rx=3)
        sc.rect(x, 312, 94, 50, "#4c9aa8", rx=3)
        sc.rect(x + 35, 284, 24, 5, "#e3ebee", rx=2.5)
        if x != 200:     # the stool stands in front of this drawer
            sc.rect(x + 35, 322, 24, 5, "#e3ebee", rx=2.5)
    sc.rect(0, 366, TW, 14, "#1f5562")
    sc.rect(0, 250, TW, 13, "#16324a")
    sc.rect(0, 250, TW, 3, "#2f5373")

    # on the bench: a beaker, a flask, a rack of tubes and a microscope
    sc.raw('<g transform="translate(18,250)"><path d="M-11,-30 L-11,-3 Q-11,0 -8,0 L8,0 Q11,0 11,-3 L11,-30" '
           'fill="#eaf5f6" fill-opacity=".75" stroke="#7fa9b3" stroke-width="1.6"/>'
           '<path d="M-10,-16 L10,-16 L10,-3 Q10,-1 8,-1 L-8,-1 Q-10,-1 -10,-3Z" fill="#58b78f"/>'
           '<path d="M-13,-30 h26" stroke="#7fa9b3" stroke-width="2.2" stroke-linecap="round"/></g>')
    sc.raw('<g transform="translate(182,250)"><path d="M-5,-46 L5,-46 L5,-30 L17,-3 Q18,0 15,0 L-15,0 Q-18,0 -17,-3 '
           'L-5,-30Z" fill="#eaf5f6" fill-opacity=".75" stroke="#7fa9b3" stroke-width="1.6" stroke-linejoin="round"/>'
           '<path d="M-10.5,-17 L10.5,-17 L16,-3 Q16.5,-1 14.5,-1 L-14.5,-1 Q-16.5,-1 -16,-3Z" fill="#37b6c9"/>'
           '<rect x="-7.5" y="-49" width="15" height="4.5" rx="2" fill="#7fa9b3"/></g>')
    tubes = "".join('<rect x="%g" y="-36" width="7" height="33" rx="3.5" fill="#eaf5f6" fill-opacity=".8" '
                    'stroke="#7fa9b3" stroke-width="1.3"/><rect x="%g" y="%d" width="5" height="%d" rx="2.5" '
                    'fill="%s"/>' % (x, x + 1, -4 - h, h, c)
                    for x, h, c in ((-15, 17, "#e2b93b"), (-3.5, 22, "#37b6c9"), (8, 13, "#d9534f")))
    sc.raw('<g transform="translate(222,250)">%s'
           '<rect x="-19" y="-24" width="38" height="4.5" rx="1.5" fill="#51677a"/>'
           '<rect x="-21" y="-4" width="42" height="4" rx="1.5" fill="#51677a"/></g>' % tubes)
    sc.raw('<g transform="translate(268,250) scale(1.08)">'
           '<rect x="-24" y="-9" width="48" height="9" rx="3" fill="#16324a"/>'
           '<path d="M13,-8 L13,-50 C13,-66 3,-72 -8,-68" fill="none" stroke="#16324a" stroke-width="9" '
           'stroke-linecap="round"/>'
           '<rect x="-24" y="-35" width="32" height="5" rx="2" fill="#51677a"/>'
           '<rect x="-16" y="-37" width="14" height="2.5" rx="1" fill="#9ed6de"/>'
           '<g transform="translate(-12,-60) rotate(-25)"><rect x="-4" y="12" width="8" height="10" fill="#8a9ca2"/>'
           '<rect x="-6.5" y="-24" width="13" height="38" rx="3" fill="#f2f5f6"/>'
           '<rect x="-7.5" y="-31" width="15" height="9" rx="2" fill="#16324a"/></g>'
           '<circle cx="13" cy="-28" r="5.5" fill="#8a9ca2"/><circle cx="13" cy="-28" r="2" fill="#51677a"/></g>')
    # a stool in front of the bench
    sc.raw('<g transform="translate(236,380)"><path d="M-15,-58 L-21,0 M15,-58 L21,0" stroke="#16324a" '
           'stroke-width="5" stroke-linecap="round"/><path d="M-18,-22 h36" stroke="#16324a" stroke-width="4"/>'
           '<rect x="-25" y="-68" width="50" height="11" rx="5.5" fill="#f2f5f6"/>'
           '<rect x="-25" y="-61" width="50" height="4" rx="2" fill="#c9d6da"/></g>')
    sc.subjects += [(162, 198, 204, 250), (200, 212, 244, 250), (236, 150, 296, 250)]
    return sc


def field():
    sc = Scene("field", "WILDLIFE", 59)
    sky = sc.grad("sky", 0, 214, [(0, "#7fc0e6"), (.6, "#cfe9ef"), (1, "#f8ecc8")])
    meadow = sc.grad("meadow", 208, 380, [(0, "#a9cf6f"), (1, "#74a850")])
    sc.camera = (254, 46)

    # sky, hills, a line of firs
    sc.rect(0, 0, TW, 216, sky)
    for cx, cy, rx, ry in ((70, 78, 26, 8), (56, 72, 12, 7), (82, 70, 15, 9), (176, 112, 20, 6), (166, 107, 10, 6)):
        sc.ellipse(cx, cy, rx, ry, "#ffffff", op=.85)
    sc.poly([(0, 176), (40, 150), (86, 168), (140, 138), (196, 160), (240, 144), (TW, 166), (TW, 216), (0, 216)],
            "#9cc9b4")
    sc.poly([(0, 196), (50, 178), (110, 192), (170, 174), (230, 190), (TW, 180), (TW, 216), (0, 216)], "#6fae8e")
    for i, x in enumerate((6, 28, 48, 150, 172, 196, 222, 246)):
        h = (44, 56, 40, 50, 62, 46, 54, 42)[i]
        col = "#2f6b4f" if i % 2 else "#3b7d5c"
        sc.rect(x - 2, 204, 4, 12, "#5a4630")
        sc.poly([(x - 13, 208), (x, 208 - h), (x + 13, 208)], col)
        sc.poly([(x - 10, 208 - h * .38), (x, 208 - h * 1.12), (x + 10, 208 - h * .38)], col)

    # meadow with a trail
    sc.rect(0, 208, TW, 172, meadow)
    sc.poly([(16, 380), (114, 380), (90, 318), (106, 268), (94, 222), (84, 222), (80, 268), (52, 318)], "#e4d3a0")
    sc.poly([(114, 380), (90, 318), (106, 268), (94, 222), (98, 268), (86, 318)], "#cdb97f", op=.6)
    for cx, cy, rx, ry, col in ((18, 300, 24, 15, "#4f9460"), (36, 308, 18, 12, "#5fa66c"),
                                (168, 372, 30, 14, "#4f9460")):
        sc.ellipse(cx, cy, rx, ry, col)
    for x, y in ((150, 300), (176, 336), (24, 352), (206, 372), (236, 262), (200, 236), (14, 250)):
        sc.raw('<path d="M%g,%g l-3,-9 M%g,%g l0,-11 M%g,%g l3,-9" stroke="#5c9648" stroke-width="2" '
               'stroke-linecap="round"/>' % (x - 3, y, x, y, x + 3, y))

    # a walker on the trail, some way off
    person(sc, 84, 284, .5, "#b97a55", "#e9dfc9", "#d9482b", "#d9482b", "#6b5b45", "#3a2a1e",
           extra=[("rect", (-42, -196, 18, 62), "#2a6f7f")])

    # the camera mast, with a small solar panel
    sc.rect(280, 26, 9, 354, "#6b4a2e")
    sc.rect(280, 26, 3, 354, "#8a6442")
    sc.rect(236, 38, 50, 7, "#6b4a2e", rx=2)
    sc.poly([(266, 30), (290, 18.7), (292, 24.7), (268, 36)], "#16324a")
    sc.poly([(267, 31), (290, 20.2), (290.6, 22.1), (267.6, 33)], "#3d6f9a")

    # birds, and the deer the camera is there for
    sc.raw('<g fill="none" stroke="#3b5566" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
           '<path d="M118,58 q6,-7 11,0 q5,-7 11,0"/><path d="M150,82 q4,-5 8,0 q4,-5 8,0"/></g>')
    stag = deer(188, 296, .86)
    sc.raw(stag[0])
    sc.subjects.append(stag[1])
    sc.rejected.append((stag[1], 1))
    return sc


def banner():
    scenes = [barn(), factory(), laboratory(), field()]
    defs = ['<linearGradient id="ground" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1b3d59"/>'
            '<stop offset="1" stop-color="#0f2638"/></linearGradient>',
            '<linearGradient id="beam" x1="1" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff8e0" '
            'stop-opacity=".34"/><stop offset="1" stop-color="#fff8e0" stop-opacity="0"/></linearGradient>',
            '<clipPath id="frame"><rect width="%d" height="%d" rx="16"/></clipPath>' % (W, H),
            '<clipPath id="tile"><rect width="%d" height="%d" rx="14"/></clipPath>' % (TW, TH)]
    body = []
    for i, sc in enumerate(scenes):
        veils = [sc.mosaic(b) for b in sc.people]
        assert veils, "every scene has at least one person"
        # the dashed box of a rejected candidate is a little larger than the object
        dashed = [((a0 - 3, b0 - 5, a1 + 3, b1 + 4), side) for (a0, b0, a1, b1), side in sc.rejected]
        for _cells, (x0, y0, x1, y1) in veils:
            # the brackets sit 4 px outside the mosaic
            assert min(x0, y0) - 4 >= EDGE and x1 + 4 <= TW - EDGE and y1 + 4 <= TH - EDGE, \
                "a mosaic is too close to the edge of its tile"
            for a0, b0, a1, b1 in sc.subjects:
                assert a1 <= x0 or a0 >= x1 or b1 <= y0 or b0 >= y1, "a subject overlaps a mosaic"
            for (a0, b0, a1, b1), _side in dashed:
                assert a1 + 16 <= x0 or a0 - 16 >= x1 or b1 + 16 <= y0 or b0 - 16 >= y1, \
                    "a rejected candidate is too close to a mosaic"
        for (a0, b0, a1, b1), _side in dashed:
            assert min(a0, b0) >= EDGE and a1 <= TW - EDGE and b1 <= TH - EDGE, \
                "a rejected candidate is too close to the edge of its tile"
        defs += [g.svg() for g in sc.grads]
        ox = MARGIN + i * (TW + GUTTER)
        body.append('<rect x="%d" y="%d" width="%d" height="%d" rx="14" fill="#07131c" fill-opacity=".55"/>'
                    % (ox, MARGIN + 5, TW, TH))
        body.append('<g transform="translate(%d,%d)"><g clip-path="url(#tile)">' % (ox, MARGIN))
        body += sc.svg_items()
        for cells, _bounds in veils:
            body.append('<g shape-rendering="crispEdges">')
            body += ['<rect x="%d" y="%d" width="%d" height="%d" fill="%s"/>' % c for c in cells]
            body.append('</g>')
        # light from the camera, then the overlays
        cam, (lx, ly) = camera(*sc.camera)
        body.append('<path d="M%g,%g L%g,%d L%g,%dZ" fill="url(#beam)"/>' % (lx, ly, lx - 168, TH, lx - 56, TH))
        for _cells, (x0, y0, x1, y1) in veils:
            body.append(brackets(x0 - 4, y0 - 4, x1 + 4, y1 + 4, GREEN))
            body.append(tick_badge((x0 + x1) / 2.0, y0 - 18, GREEN))
        for (a0, b0, a1, b1), side in dashed:
            dash = ('<rect x="%g" y="%g" width="%g" height="%g" rx="12" fill="none" stroke="%%s" stroke-width="%%g" '
                    'stroke-dasharray="7 6"/>' % tuple(round(v, 2) for v in (a0, b0, a1 - a0, b1 - b0)))
            body.append(dash % ("#ffffff", 4.5) + dash % (SLATE, 2.5))
            body.append(cross_badge(round(a1 if side > 0 else a0, 2), round(b0, 2), SLATE))
        body.append(cam)
        # caption, as a camera's on-screen label; textLength keeps it inside its pill in any font
        body.append('<rect x="10" y="10" width="%d" height="24" rx="12" fill="#0b1c28" fill-opacity=".8"/>'
                    '<circle cx="23" cy="22" r="4" fill="#ff4d4d"/>'
                    '<text x="33" y="26.2" font-family="%s" font-size="11.5" font-weight="700" letter-spacing="1" '
                    'textLength="%d" lengthAdjust="spacing" fill="#ffffff">%s</text>'
                    % (sc.label_width + 34, FONT, sc.label_width, escape(sc.label)))
        body.append('</g><rect x=".75" y=".75" width="%g" height="%g" rx="13.25" fill="none" stroke="#ffffff" '
                    'stroke-opacity=".28" stroke-width="1.5"/></g>' % (TW - 1.5, TH - 1.5))

    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
           'aria-labelledby="t d">' % (W, H, W, H),
           '<title id="t">OpenVeil: people veiled, the subject of the footage visible</title>',
           '<desc id="d">Four drawn scenes, each watched by a camera: a calf barn, a factory line with a robot arm, '
           'a laboratory bench and a meadow with a deer. In every scene the one person present is covered by a pixel '
           'mosaic with a green tick, while the calves, the machine, the instruments and the deer stay visible. '
           'A dashed box with a cross on one calf and on the deer marks a candidate the verifier rejected. '
           'No real people or footage.</desc>',
           '<defs>%s</defs>' % "".join(defs),
           '<g clip-path="url(#frame)"><rect width="%d" height="%d" fill="url(#ground)"/>' % (W, H)]
    out += body
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
  .k { font-weight: 600; fill: #24292f; }
  .h { font-size: 12px; font-weight: 600; letter-spacing: .04em; fill: #57606a; }
  .p { font-size: 12px; font-weight: 600; fill: #2a7354; }
  .tie { stroke: #2f7d5b; stroke-width: 1.5; fill: none; stroke-linejoin: round; stroke-linecap: round; }
  .box { fill: #ffffff; stroke: #d0d7de; stroke-width: 1; }
  .box.pair { stroke: #2f7d5b; stroke-width: 1.5; }
  .box.key { fill: #e8f3ee; stroke: #2f7d5b; stroke-width: 1.5; }
  .box.human { fill: #fff8e8; stroke: #c98500; }
  .box.out { fill: #e9f1f8; stroke: #2a6f9f; }
  .band { fill: #f6f8fa; stroke: none; }
  .edge { stroke: #57606a; stroke-width: 1.5; fill: none; }
  .edge.dash { stroke-dasharray: 4 3; }
  .arrow { fill: #57606a; }
  @media (prefers-color-scheme: dark) {
    text { fill: #e6edf3; }
    .d, .h { fill: #9da7b3; }
    .k { fill: #e6edf3; }
    .box { fill: #161b22; stroke: #3d444d; }
    .p { fill: #5fb891; }
    .tie { stroke: #4aa37c; }
    .box.pair { stroke: #4aa37c; }
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
    # Nodes 3 and 4 are the design: one model proposes broadly, a second one judges every candidate.
    # A line may be (lead, rest): the lead is set in bold, to say what the node is there for.
    "prop": ("3  Proposer", [("Proposes broadly:", "every box"), "Grounding DINO scores >= 0.05",
                             "(a low threshold) is a candidate"], "pair"),
    "verif": ("4  Verifier", [("Judges each crop:", "keeps a box"), "only if p(person) >= 0.10",
                              "(DINOv3 ViT-L/16 + LoRA)"], "key"),
    "table": ("5  Box table", ["Box rules, gap interpolation.", "One small .json.gz per video;", "rules re-apply without a GPU"], ""),
    "review": ("6  Review tool", ["A person checks the hard", "frames: 0.6 s median per", "frame. Writes overrides"], "human"),
    "mosaic": ("7  Mosaic + smart cut", ["Box +20% per side, held ±2 s.", "Only GOPs with a mosaic are", "re-encoded: 12.0% of packets"], ""),
    "release": ("8  Verified release", ["Each file checked against", "its source: all 1,023 masked", "videos passed"], "out"),
}

# The same texts, wrapped for the 300 px boxes of the narrow figure.
NARROW = {
    "video": ["Any number, any length.", "Decoded in 60 s chunks"],
    "sched": ["A check every 1 s; every 0.2 s after a hit.", "0.0508 calls/frame (case study, sparse at night)"],
    "prop": [("Proposes broadly:", "every box Grounding DINO"), "scores >= 0.05 (a low threshold) is a candidate"],
    "verif": [("Judges each crop:", "keeps a box only if"), "p(person) >= 0.10 (DINOv3 ViT-L/16 + LoRA)"],
    "table": ["Box rules, gap interpolation. One small .json.gz", "per video; rules re-apply without a GPU"],
    "review": ["A person checks the hard frames: 0.6 s median", "per frame. Writes overrides"],
    "mosaic": ["Box +20% per side, held ±2 s. Only GOPs with", "a mosaic are re-encoded: 12.0% of packets"],
    "release": ["Each file checked against its source:", "all 1,023 masked videos passed"],
}
PAIR = "A detector proposes broadly, then a classifier verifies each candidate"
PAIR_NARROW = ("propose broadly,", "then verify each crop")
HEAD_GPU, HEAD_CPU = "DETECT, ON A GPU, ONCE PER VIDEO", "REVIEW AND APPLY, NO GPU"


def node(x, y, w, h, key, lines=None):
    title, wide, cls = NODES[key]
    lines = wide if lines is None else lines
    s = ['<rect class="box %s" x="%s" y="%s" width="%s" height="%s" rx="8"/>' % (cls, x, y, w, h),
         '<text class="t" x="%s" y="%s">%s</text>' % (x + 12, y + 23, escape(title))]
    for i, line in enumerate(lines):
        if isinstance(line, tuple):
            line = '<tspan class="k">%s</tspan> %s' % (escape(line[0]), escape(line[1].replace(">=", "≥")))
        else:
            line = escape(line.replace(">=", "≥"))
        s.append('<text class="d" x="%s" y="%s">%s</text>' % (x + 12, y + 43 + i * 16, line))
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
PIPE_DESC = ("Detection on a GPU: videos, a scan schedule, the Grounding DINO proposer, which proposes person "
             "boxes broadly at a low threshold, and the DINOv3 crop verifier, which judges each crop and keeps "
             "the people, write one box table per video. Then, without a GPU: a person checks the hard frames in "
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
          '<text class="h" x="20" y="36">%s</text>' % HEAD_GPU,
          '<text class="h" x="20" y="208">%s</text>' % HEAD_CPU,
          # nodes 3 and 4 work as a pair: a bracket over both, with what the pair does
          '<path class="tie" d="M%d,%d v-5 h%d v5"/>' % (xs[2] + 1, y1 - 5, xs[3] + bw - xs[2] - 2),
          '<text class="p" x="%d" y="36" text-anchor="middle">%s</text>' % ((xs[2] + xs[3] + bw) // 2, PAIR)]
    for x, k in zip(xs, ("video", "sched", "prop", "verif")):
        s.append(node(x, y1, bw, bh, k))
    for x, k in zip(xs, ("table", "review", "mosaic", "release")):
        s.append(node(x, y2, bw, bh, k))
    for i in range(3):
        s.append(arrow([(xs[i] + bw, y1 + bh // 2), (xs[i + 1], y1 + bh // 2)]))
        s.append(arrow([(xs[i] + bw, y2 + bh // 2), (xs[i + 1], y2 + bh // 2)]))
    # from box 4 down to box 5; it comes down right of the second heading, which ends at about x = 192
    s.append(arrow([(xs[3] + bw // 2, y1 + bh), (xs[3] + bw // 2, 176), (xs[0] + bw - 28, 176),
                    (xs[0] + bw - 28, y2)]))
    s.append("</svg>")
    return "\n".join(s)


def pipeline_narrow():
    w, bw, bh, gap = 340, 300, 72, 24
    x, cx = 20, 170
    gpu, cpu = ("video", "sched", "prop", "verif"), ("table", "review", "mosaic", "release")
    top1, top2 = 4, 4 + 36 + 4 * bh + 3 * gap + 12 + 8     # tops of the two bands
    h = top2 + 36 + 4 * bh + 3 * gap + 12 + 4
    s = _head(w, h, "pn")
    ys = {}
    for top, keys, head in ((top1, gpu, HEAD_GPU), (top2, cpu, HEAD_CPU)):
        s += ['<rect class="band" x="4" y="%d" width="%d" height="%d" rx="12"/>'
              % (top, w - 8, 36 + 4 * bh + 3 * gap + 12),
              '<text class="h" x="20" y="%d">%s</text>' % (top + 24, head)]
        for i, k in enumerate(keys):
            ys[k] = y = top + 36 + i * (bh + gap)
            s.append(node(x, y, bw, bh, k, NARROW[k]))
            if i:
                s.append(arrow([(cx, y - gap), (cx, y)]))
    # nodes 3 and 4 work as a pair: a bracket beside both, and what the pair does at the arrow between them
    s += ['<path class="tie" d="M%d,%d h-5 v%d h5"/>' % (x - 5, ys["prop"] + 1, ys["verif"] + bh - ys["prop"] - 2),
          '<text class="p" x="%d" y="%d" text-anchor="end">%s</text>' % (cx - 12, ys["verif"] - 8, PAIR_NARROW[0]),
          '<text class="p" x="%d" y="%d">%s</text>' % (cx + 12, ys["verif"] - 8, PAIR_NARROW[1])]
    # from box 4 down to box 5, right of the second heading
    s.append(arrow([(x + bw - 28, ys["verif"] + bh), (x + bw - 28, ys["table"])]))
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
