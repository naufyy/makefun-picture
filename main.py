"""Molten chocolate cake (fondant au chocolat), close up.

One ingredient: chocolate. Writes /output/entry.png, 1024x1024.
Python 3.13 standard library only, no network, fixed seed: two runs write
byte-identical files. Everything is drawn at 2048x2048 and box-filtered
down to 1024x1024 for antialiasing.
"""

import math
import os
import random
import struct
import zlib

S = 2                      # supersampling factor
OUT = 1024
W = H = OUT * S
SEED = 20261122

# Scene geometry, in 1024-pixel units.
CX = 512.0                 # cake centre x
TOPY = 445.0               # centre y of the cake's top ellipse
BOTY = 685.0               # centre y of the cake's bottom ellipse
RX = 320.0                 # cake radius (about 62% of the width)
RY = 78.0                  # perspective squash of the cake ellipses

PLATE = (512.0, 737.0, 492.0, 170.0)
PLATE_IN = (512.0, 741.0, 425.0, 140.0)
POOL = (520.0, 727.0, 402.0, 106.0)

# Light comes from the upper left, towards the viewer.
LX, LY, LZ = -0.55, -0.35, 0.76

# Chocolate palette.
SAUCE_D = (22, 9, 4)
SAUCE_M = (74, 34, 15)
SAUCE_L = (138, 74, 38)
CRUST = (112, 66, 41)
SPONGE = (86, 47, 28)
CRACK = (30, 15, 8)
GOLD = (255, 224, 168)
WHITE = (255, 246, 232)

rng = random.Random(SEED)
NT = 256
LAT = [rng.random() for _ in range(NT * NT)]
GRAIN = [rng.random() - 0.5 for _ in range(512 * 512)]


def vnoise(x, y):
    """Smooth value noise in [0, 1] for non-negative x, y."""
    xi = int(x)
    yi = int(y)
    fx = x - xi
    fy = y - yi
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    x0 = xi & 255
    y0 = (yi & 255) * NT
    x1 = (x0 + 1) & 255
    y1 = ((yi + 1) & 255) * NT
    a = LAT[y0 + x0]
    b = LAT[y0 + x1]
    c = LAT[y1 + x0]
    d = LAT[y1 + x1]
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy


def grain(x, y):
    return GRAIN[((y & 511) << 9) | (x & 511)]


def lerp3(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def clamp(v):
    if v <= 0:
        return 0
    if v >= 255:
        return 255
    return int(v + 0.5)


def smooth(e0, e1, x):
    t = (x - e0) / (e1 - e0)
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    return t * t * (3 - 2 * t)


# ================================================================ CANVAS
class Canvas:
    def __init__(self):
        self.buf = bytearray(W * H * 3)

    def paint(self, spans, shade):
        """Run shade(x, y) -> (r, g, b, a) or None over each span."""
        buf = self.buf
        for y, x0, x1 in spans:
            base = y * W
            for x in range(x0, x1):
                c = shade(x, y)
                if c is None:
                    continue
                r, g, b, a = c
                if a <= 0.004:
                    continue
                i = (base + x) * 3
                if a >= 1.0:
                    buf[i] = clamp(r)
                    buf[i + 1] = clamp(g)
                    buf[i + 2] = clamp(b)
                else:
                    ia = 1.0 - a
                    buf[i] = clamp(buf[i] * ia + r * a)
                    buf[i + 1] = clamp(buf[i + 1] * ia + g * a)
                    buf[i + 2] = clamp(buf[i + 2] * ia + b * a)

    def fill(self, spans, col, a=1.0):
        r, g, b = col
        self.paint(spans, lambda x, y: (r, g, b, a))

    def line(self, x0, y0, x1, y1, t, col, a=1.0):
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / ln * t / 2, dx / ln * t / 2
        self.fill(poly_spans([(x0 + nx, y0 + ny), (x1 + nx, y1 + ny),
                              (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)]),
                  col, a)
        self.fill(poly_spans(ell_pts(x1, y1, t / 2, t / 2, n=16)), col, a)

    def soft_ellipse(self, cx, cy, rx, ry, rot, col, amax, power=2.0):
        """Feathered ellipse: alpha amax at the centre, 0 at the edge."""
        ca, sa = math.cos(rot), math.sin(rot)
        r, g, b = col

        def shade(x, y):
            u = x / S - cx
            v = y / S - cy
            lx = (u * ca + v * sa) / rx
            ly = (-u * sa + v * ca) / ry
            d2 = lx * lx + ly * ly
            if d2 >= 1.0:
                return None
            return (r, g, b, amax * (1.0 - d2) ** power)

        self.paint(poly_spans(ell_pts(cx, cy, rx, ry, rot)), shade)

    def downsample(self):
        """Box-filter to 1024, apply the vignette, return raw RGB rows."""
        src = self.buf
        out = bytearray()
        row2 = W * 3
        vx, vy, vr = 512.0, 540.0, 760.0
        for oy in range(OUT):
            out.append(0)
            r0 = 2 * oy * row2
            r1 = r0 + row2
            dy2 = ((oy + 0.5 - vy) / vr) ** 2
            for ox in range(OUT):
                i = r0 + ox * 6
                j = r1 + ox * 6
                d = math.sqrt(((ox + 0.5 - vx) / vr) ** 2 + dy2)
                k = 1.0 - 0.62 * smooth(0.42, 1.05, d)
                k *= 0.25
                out.append(clamp((src[i] + src[i + 3] + src[j] + src[j + 3]) * k))
                out.append(clamp((src[i + 1] + src[i + 4] + src[j + 1]
                                  + src[j + 4]) * k))
                out.append(clamp((src[i + 2] + src[i + 5] + src[j + 2]
                                  + src[j + 5]) * k))
        return out


def poly_spans(pts):
    """Even-odd scanline spans of a polygon given in 1024 units."""
    P = [(x * S, y * S) for x, y in pts]
    n = len(P)
    edges = []
    for i in range(n):
        xa, ya = P[i]
        xb, yb = P[(i + 1) % n]
        if ya != yb:
            edges.append((xa, ya, xb, yb, (xb - xa) / (yb - ya)))
    ys = [p[1] for p in P]
    y0 = max(0, int(math.floor(min(ys))))
    y1 = min(H - 1, int(math.ceil(max(ys))))
    out = []
    for y in range(y0, y1 + 1):
        sy = y + 0.5
        xs = [xa + (sy - ya) * k for xa, ya, xb, yb, k in edges
              if (ya <= sy < yb) or (yb <= sy < ya)]
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            a = max(0, int(math.ceil(xs[k] - 0.5)))
            b = min(W, int(math.ceil(xs[k + 1] - 0.5)))
            if b > a:
                out.append((y, a, b))
    return out


def ell_pts(cx, cy, rx, ry, rot=0.0, n=160, wob=None):
    ca, sa = math.cos(rot), math.sin(rot)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        m = wob(t) if wob else 1.0
        ex, ey = rx * m * math.cos(t), ry * m * math.sin(t)
        pts.append((cx + ex * ca - ey * sa, cy + ex * sa + ey * ca))
    return pts


def rim_y(u, cy):
    nx = (u - CX) / RX
    return cy + RY * math.sqrt(max(0.0, 1.0 - nx * nx))


# ================================================================ SCENE
def draw_background(cv):
    buf = cv.buf
    gx, gy = 400.0, 330.0
    for oy in range(OUT):
        rows = (2 * oy * W * 3, (2 * oy + 1) * W * 3)
        dy2 = (oy - gy) ** 2
        line = bytearray()
        for ox in range(OUT):
            d = math.sqrt((ox - gx) ** 2 + dy2) / 820.0
            t = 1.0 - smooth(0.0, 1.0, d)
            t = t * t
            r = clamp(7 + 58 * t)
            g = clamp(4 + 31 * t)
            b = clamp(3 + 18 * t)
            line += bytes((r, g, b, r, g, b))
        for r0 in rows:
            buf[r0:r0 + W * 3] = line


def draw_plate(cv):
    px, py, prx, pry = PLATE
    ix, iy, irx, iry = PLATE_IN
    # Plate edge (thickness) just below its top.
    cv.fill(poly_spans(ell_pts(px, py + 12, prx, pry)), (20, 20, 23))

    def shade(x, y):
        u, v = x / S, y / S
        dx = (u - px) / prx
        dy = (v - py) / pry
        ex = (u - ix) / irx
        ey = (v - iy) / iry
        e2 = ex * ex + ey * ey
        if e2 < 1.0:
            # Well of the plate: low sheen, a touch darker towards the back.
            k = 0.62 - 0.18 * ex - 0.10 * ey
            base = 52.0
            if e2 > 0.9:          # inner lip catches light at the front
                k += 0.35 * smooth(0.9, 1.0, e2) * max(0.0, ey)
        else:
            k = 0.85 - 0.35 * dx - 0.30 * dy
            base = 74.0
            d = math.sqrt(dx * dx + dy * dy)
            k *= 0.8 + 0.25 * smooth(0.95, 1.0, 1.0 - abs(d - 0.93) * 4)
        k += grain(x, y) * 0.03
        return (base * k, base * k * 0.99, base * k * 1.05, 1.0)

    cv.paint(poly_spans(ell_pts(px, py, prx, pry)), shade)
    # Polished highlight on the upper-left rim.
    cv.soft_ellipse(215, 665, 120, 13, -0.42, (235, 228, 220), 0.22)
    cv.soft_ellipse(820, 615, 90, 8, 0.25, (220, 214, 208), 0.10)


def draw_shadow(cv):
    # Cake's cast shadow, pushed down-right away from the key light.
    cv.soft_ellipse(CX + 55, BOTY + 18, RX + 40, RY + 40, 0.0,
                    (5, 3, 2), 0.75, 1.2)


def sauce_dome(x, y, cx, cy, rx, ry, front_glow):
    u, v = x / S, y / S
    dx = (u - cx) / rx
    dy = (v - cy) / ry
    d2 = dx * dx + dy * dy
    dome = max(0.0, 1.0 - d2)
    t = 0.18 + 0.55 * dome ** 0.6 - 0.25 * dx - 0.12 * dy
    t += front_glow * max(0.0, dy) * 0.5
    t += grain(x, y) * 0.02
    t = 0.0 if t < 0 else t
    if t < 1.0:
        c = lerp3(SAUCE_D, SAUCE_M, t)
    else:
        c = lerp3(SAUCE_M, SAUCE_L, min(1.0, t - 1.0))
    return c


def draw_pool(cv):
    cx, cy, rx, ry = POOL

    def wob(t):
        return (1.0 + 0.035 * math.sin(5 * t + 0.7)
                + 0.025 * math.sin(11 * t + 2.1) + 0.02 * math.sin(3 * t))

    def shade(x, y):
        r, g, b = sauce_dome(x, y, cx, cy, rx, ry, 0.9)
        return (r, g, b, 1.0)

    pts = ell_pts(cx, cy, rx, ry, wob=wob, n=240)
    # Thin dark rim where the sauce meets the plate.
    cv.fill(poly_spans(ell_pts(cx, cy + 2, rx + 4, ry + 4, wob=wob, n=240)),
            (12, 7, 5))
    cv.paint(poly_spans(pts), shade)


def pool_highlights(cv):
    cx, cy = POOL[0], POOL[1]
    cv.soft_ellipse(cx - 230, cy + 62, 80, 10, 0.16, GOLD, 0.45)
    cv.soft_ellipse(cx - 236, cy + 61, 42, 4, 0.16, WHITE, 0.85)
    cv.soft_ellipse(cx - 70, cy + 95, 60, 6, 0.02, GOLD, 0.35)
    cv.soft_ellipse(cx - 74, cy + 95, 26, 2.6, 0.02, WHITE, 0.7)
    cv.soft_ellipse(cx + 170, cy + 82, 50, 5, -0.14, GOLD, 0.25)
    cv.soft_ellipse(cx + 300, cy + 40, 36, 4, -0.5, GOLD, 0.18)


def draw_body(cv):
    pts = [(CX - RX, TOPY), (CX + RX, TOPY)]
    for i in range(121):
        t = math.pi * i / 120
        pts.append((CX + RX * math.cos(t), BOTY + RY * math.sin(t)))
    pts.append((CX - RX, TOPY))
    height = BOTY - TOPY

    def shade(x, y):
        u, v = x / S, y / S
        nx = (u - CX) / RX
        if nx < -1.0:
            nx = -1.0
        elif nx > 1.0:
            nx = 1.0
        nz = math.sqrt(1.0 - nx * nx)
        dif = -0.58 * nx + 0.81 * nz
        if dif < 0:
            dif = 0.0
        t = (v - (TOPY + RY * nz)) / height
        occ = 1.0 - 0.45 * t * t
        n = vnoise(u * 0.09, v * 0.09) * 0.6 + vnoise(u * 0.31, v * 0.31) * 0.4
        k = (0.16 + 0.95 * dif) * occ * (0.82 + 0.36 * n)
        k += grain(x, y) * 0.10
        # Crumb pores: small dark pits in the baked sponge.
        p = vnoise(u * 0.9 + 37.0, v * 0.9 + 11.0)
        if p < 0.12:
            k *= 0.55 + p * 3.0
        # Crisp baked rim just under the top edge.
        if t < 0.035:
            k *= 1.12
        # Warm rim light on the shadow side so the silhouette separates
        # from the dark backdrop, even at thumbnail size.
        k += 0.55 * smooth(0.88, 0.995, nx) * (1.0 - 0.6 * t)
        return (SPONGE[0] * k, SPONGE[1] * k, SPONGE[2] * k, 1.0)

    cv.paint(poly_spans(pts), shade)
    # Warm sheen on the lit side.
    cv.soft_ellipse(CX - 235, TOPY + 150, 30, 120, 0.05, (190, 120, 76), 0.13)


def draw_top(cv):
    def shade(x, y):
        u, v = x / S, y / S
        dx = (u - CX) / RX
        dy = (v - TOPY) / RY
        d = math.sqrt(dx * dx + dy * dy)
        k = 0.82 - 0.26 * dx - 0.16 * dy
        # Rounded edge: the back-left lip catches the light.
        e = smooth(0.84, 1.0, d)
        k += e * (-0.3 * dx - 0.35 * dy)
        n = vnoise(u * 0.12 + 5.0, v * 0.12 * 3.2) * 0.6 + \
            vnoise(u * 0.45, v * 0.45 * 3.2) * 0.4
        k *= 0.8 + 0.4 * n
        k += grain(x, y) * 0.12
        return (CRUST[0] * k, CRUST[1] * k, CRUST[2] * k, 1.0)

    cv.paint(poly_spans(ell_pts(CX, TOPY, RX, RY, n=200)), shade)
    cv.soft_ellipse(CX - 175, TOPY - 30, 110, 26, -0.1, (205, 146, 98), 0.2)


def draw_cracks(cv):
    r = random.Random(SEED + 7)
    angles = [0.25, 0.95, 1.75, 2.45, 2.95, 3.55, 4.1, 4.7, 5.25, 5.85]
    for a0 in angles:
        a = a0 + r.uniform(-0.12, 0.12)
        rad = 0.70 + r.uniform(-0.03, 0.03)
        end = r.uniform(0.93, 0.99)
        pts = []
        while rad < end:
            pts.append((CX + RX * rad * math.cos(a), TOPY + 8 + RY * rad
                        * math.sin(a)))
            rad += r.uniform(0.06, 0.085)
            a += r.uniform(-0.05, 0.05)
        pts.append((CX + RX * end * math.cos(a), TOPY + 8 + RY * end
                    * math.sin(a)))
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            f = 1.0 - i / max(1, len(pts) - 1)
            w = 1.3 + 4.2 * f
            # Lit lower lip of the fissure, then the dark split itself.
            cv.line(x0 + 0.9, y0 + 1.3, x1 + 0.9, y1 + 1.3, w * 0.85,
                    (184, 124, 82), 0.55)
            cv.line(x0, y0, x1, y1, w, CRACK)


CAP = (CX - 12, TOPY + 14, 228.0, 60.0)


def cap_wob(t):
    s = math.sin(t)
    front = max(0.0, s) ** 2
    return (1.0 + 0.06 * math.sin(3 * t + 1.0) + 0.04 * math.sin(7 * t + 2.0)
            + 0.025 * math.sin(13 * t) + 0.24 * front)


def draw_drips(cv):
    # (offset from centre, width, length); length None = runs into the pool.
    drips = [(-176, 24, 78), (-128, 30, 158), (-80, 24, 96), (-30, 40, None),
             (24, 28, 128), (76, 36, None), (124, 24, 90), (166, 30, None)]
    for off, w, length in drips:
        x = CX + off
        top = rim_y(x, TOPY) - 16
        hw0 = w / 2
        to_pool = length is None
        if to_pool:
            # Run down the side and spread into the sauce pool at the base.
            bot = rim_y(x, BOTY) + 10
            length = bot - top
            bead = 0.0
            bc = bot
        else:
            bot = top + length
            bead = hw0 * 1.0           # teardrop bead hanging at the tip
            bc = bot - bead

        def hw(v, top=top, length=length, hw0=hw0, to_pool=to_pool,
               bead=bead, bc=bc):
            t = (v - top) / length
            # Stem thins as the sauce runs down, with a gentle wobble.
            h = hw0 * (1.0 - 0.42 * t) * (1.0 + 0.08 * math.sin(v * 0.045 + hw0)
                                          + 0.04 * math.sin(v * 0.09 + 1.3))
            if t < 0.1:                # fillet where it leaves the rim
                h += hw0 * 0.9 * (1.0 - t / 0.1) ** 2
            if to_pool:
                # Foot flares out where it meets the pool.
                h += hw0 * 1.6 * smooth(0.82, 1.0, t) ** 2
            else:
                dv = v - bc
                if 0.0 <= dv < bead:
                    h = math.sqrt(bead * bead - dv * dv)   # round underside
                elif dv * dv < bead * bead:
                    c = math.sqrt(bead * bead - dv * dv)
                    # Smooth max of stem and bead circle: no hard step.
                    k = 4.0
                    h = (h + c + math.sqrt((h - c) ** 2 + k * k)) * 0.5 \
                        - k * 0.25
                elif v > bc:
                    h = 0.0
            return max(h, 0.0)

        pts = []
        steps = 90
        for i in range(steps + 1):
            v = top + (bot - top) * i / steps
            pts.append((x - hw(v), v))
        if to_pool:
            for i in range(0, 13):     # shallow rounded foot
                t = math.pi - math.pi * i / 12
                pts.append((x + hw(bot) * math.cos(t),
                            bot + 6.0 * math.sin(t)))
        for i in range(steps, -1, -1):
            v = top + (bot - top) * i / steps
            pts.append((x + hw(v), v))

        def shade(px, py, x=x, top=top, hw=hw, length=length):
            u, v = px / S, py / S
            h = hw(v)
            if h < 0.5:
                h = 0.5
            s = (u - x) / h
            if s < -1.0:
                s = -1.0
            elif s > 1.0:
                s = 1.0
            nz = math.sqrt(1.0 - s * s)
            dif = max(0.0, -0.6 * s + 0.8 * nz)
            c = lerp3(SAUCE_D, SAUCE_M, 0.12 + 0.8 * dif)
            t = (v - top) / length
            fade = 1.0 - 0.45 * t
            spec = math.exp(-((s + 0.45) / 0.10) ** 2) * 0.62 * fade
            rim = math.exp(-((s - 0.74) / 0.12) ** 2) * 0.12
            c = lerp3(c, SAUCE_L, rim)
            c = lerp3(c, GOLD if spec < 0.35 else WHITE, spec)
            return (c[0], c[1], c[2], 1.0)

        cv.paint(poly_spans(pts), shade)
        if to_pool:
            # Glint where the drip merges into the pool.
            cv.soft_ellipse(x - hw(bot) * 0.3, bot - 2, hw(bot) * 0.45, 3.0,
                            0.0, GOLD, 0.5)
        else:
            # Gleam on the bead at the end of the drip.
            cv.soft_ellipse(x - bead * 0.38, bc - bead * 0.3, bead * 0.34,
                            bead * 0.22, -0.6, WHITE, 0.85)


def draw_cap(cv):
    cx, cy, rx, ry = CAP
    # Shadowed broken crust edge around the molten centre.
    cv.fill(poly_spans(ell_pts(cx + 2, cy - 2, rx * 1.05, ry * 1.12,
                               wob=cap_wob, n=240)), (40, 20, 11))

    def shade(x, y):
        r, g, b = sauce_dome(x, y, cx, cy, rx, ry, 0.2)
        return (r, g, b, 1.0)

    cv.paint(poly_spans(ell_pts(cx, cy, rx, ry, wob=cap_wob, n=240)), shade)
    # Deep molten well where the cake split open.
    cv.soft_ellipse(cx + 18, cy - 2, 120, 26, 0.0, (10, 4, 2), 0.55, 1.5)
    # Layered speculars: wide warm glow, gold streak, white core.
    cv.soft_ellipse(cx - 95, cy - 20, 105, 20, -0.08, (180, 110, 60), 0.3)
    cv.soft_ellipse(cx - 100, cy - 24, 62, 7, -0.1, GOLD, 0.6)
    cv.soft_ellipse(cx - 106, cy - 25, 30, 3.2, -0.1, WHITE, 0.95)
    cv.soft_ellipse(cx + 60, cy - 32, 44, 4, 0.05, GOLD, 0.35)
    cv.soft_ellipse(cx - 20, cy + 30, 50, 5, 0.04, GOLD, 0.35)
    cv.soft_ellipse(cx - 24, cy + 30, 20, 2.2, 0.04, WHITE, 0.8)
    cv.soft_ellipse(cx + 150, cy + 20, 30, 4, 0.35, GOLD, 0.25)


def draw_bean(cv, bx, by, rot, rx=44.0, ry=25.0):
    cv.soft_ellipse(bx + 8, by + 12, rx * 1.1, ry * 0.9, rot, (3, 2, 1), 0.75)
    ca, sa = math.cos(rot), math.sin(rot)
    base = (146, 84, 58)

    def shade(x, y):
        u = x / S - bx
        v = y / S - by
        lx = (u * ca + v * sa) / rx
        ly = (-u * sa + v * ca) / ry
        d2 = lx * lx + ly * ly
        if d2 >= 1.0:
            return None
        nz = math.sqrt(1.0 - d2)
        # Screen-space normal of the local dome.
        sx = lx * ca - ly * sa
        sy = lx * sa + ly * ca
        dif = max(0.0, LX * sx + LY * sy + LZ * nz)
        k = 0.22 + 1.0 * dif
        k *= 0.9 + 0.2 * vnoise(u * 0.5 + 90, v * 0.5 + 40)
        # Groove along the long axis, gently curved.
        gy = ly - 0.10 * (1.0 - lx * lx)
        g = abs(gy)
        if g < 0.07 * (1.0 - lx * lx) + 0.02:
            k *= 0.45
        elif -0.2 < gy < -0.07:
            k *= 1.12
        k += grain(x, y) * 0.05
        return (base[0] * k, base[1] * k, base[2] * k, 1.0)

    cv.paint(poly_spans(ell_pts(bx, by, rx, ry, rot, n=80)), shade)
    hx = bx - 0.38 * rx * ca + 0.45 * ry * sa
    hy = by - 0.38 * rx * sa - 0.45 * ry * ca
    cv.soft_ellipse(hx, hy, rx * 0.35, ry * 0.15, rot, GOLD, 0.45)


def draw_steam(cv):
    r = random.Random(SEED + 99)
    wisps = [(452.0, 0.9, 1.2), (515.0, -0.6, 2.4), (575.0, 0.7, 4.0),
             (628.0, -0.8, 5.1)]
    for x0, drift, phase in wisps:
        y0 = TOPY - 6 + r.uniform(-4, 4)
        height = r.uniform(260, 320)
        amp = r.uniform(16, 26)
        n = 130
        for i in range(n):
            t = i / (n - 1)
            x = x0 + amp * t * math.sin(t * 7.5 + phase) + drift * 40 * t
            y = y0 - height * t
            rad = 5.0 + 17.0 * t
            a = 0.055 * (1.0 - t) ** 1.4 * min(1.0, t * 6.0)
            if a <= 0.002:
                continue
            cv.soft_ellipse(x, y, rad, rad * 1.15, 0.0, (240, 232, 226), a,
                            1.5)


def png_bytes(raw):
    def chunk(tag, data):
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", OUT, OUT, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b""))


def render():
    cv = Canvas()
    draw_background(cv)
    draw_plate(cv)
    draw_shadow(cv)
    draw_pool(cv)
    pool_highlights(cv)
    draw_body(cv)
    draw_top(cv)
    draw_cracks(cv)
    draw_drips(cv)
    draw_cap(cv)
    draw_bean(cv, 188, 815, -0.35)
    draw_bean(cv, 286, 860, 0.12)
    draw_bean(cv, 834, 832, 0.4)
    draw_steam(cv)
    return png_bytes(cv.downsample())


def main():
    os.makedirs("/output", exist_ok=True)
    data = render()
    with open("/output/entry.png", "wb") as f:
        f.write(data)


if __name__ == "__main__":
    main()
