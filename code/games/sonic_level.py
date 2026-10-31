"""Sonic levels: the traced Green Hill acts (levels/sonic/*.yaml) built into collision data.

A hill over a lower path is hollow with a solid top, as in the original: a mask holds 1 for solid on
every side and 2 for the hollow part, which a body on the ground reads as solid and a body in the air
only when landing on it, so a jump or a spring goes up through it. Steep faces keep a skin of 1s.

A level file holds polygons and object positions, not tiles. This module rasterises them into the
pixel masks and per-tile surface angles the original's sensors read. Two planes exist because a loop
is solid on one side at a time (the player's plane flips at the top, as on the Mega Drive):

  plane 0  the loop's right half is solid, the left half is open (the way in)
  plane 1  the left half is solid, the right half is open (the way out)

Angles use the console's units: 0-255 is a full turn, 0 floor, 0x40 left wall, 0x80 ceiling, 0xC0
right wall. Odd values mean "no real surface here": the reader snaps to the nearest quadrant.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pygame
import yaml

LEVEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "levels", "sonic")
TILE = 16
NO_ANGLE = 0xFF
WALK_CLEARANCE = 128      # how much of a wall is open above a walkway drawn in front of it
SHELL = 24                # solid kept under a surface above a carved walkway
LUMP = 40                 # a "surface" this close over the path is a plant the tracing took for ground
COS_CORNER = math.cos(math.radians(40))     # a turn sharper than this in the trace is a real corner
GRASS = 10                # how far grass tips stand above the surface they grow on
THIN_WALL = 32            # dirt this narrow with a room on both sides does not stop a body
SLIVER = 2                # solid thinner than 2 * this + 1 px is a carving leftover
SKIN = 8                  # how thick a steep face stays solid to a body in the air
TUBE_WALL = 48            # how far around a tunnel the dirt stays solid on every side
FIT_MIN = 8               # surface pixels a tile needs for an angle of its own
TUBE_GROW = 4             # the traced band is the dark floor of the tunnel: the bore is this much wider
FLOOR, RIGHT, CEILING, LEFT = 0, 1, 2, 3    # sensor directions, in the angle tables' order


def delta(text: str) -> list[tuple[int, int]]:
    """'10,20 1,2 3,4' -> [(10, 20), (11, 22), (14, 26)]"""
    x = y = 0
    out = []
    for pair in text.split():
        dx, dy = map(int, pair.split(","))
        x, y = x + dx, y + dy
        out.append((x, y))
    return out


def curve(pts: list, closed: bool = True) -> list:
    """A smooth line through traced points: the trace keeps one point per several tiles, so straight
    segments between them would turn a half-pipe into facets. Corners sharper than CORNER stay corners."""
    n = len(pts)

    def seg(i: int) -> tuple:
        a, b = pts[i % n], pts[(i + 1) % n]
        return b[0] - a[0], b[1] - a[1], math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0

    def tangent(i: int) -> tuple:
        """Direction through point i, as a unit vector, or None at a corner or a free end."""
        if not closed and (i == 0 or i == n - 1):
            return None
        ax, ay, al = seg(i - 1)
        bx, by, bl = seg(i)
        if (ax * bx + ay * by) / (al * bl) < COS_CORNER:
            return None
        tx, ty = ax * bl + bx * al, ay * bl + by * al       # the shorter neighbour weighs more
        k = math.hypot(tx, ty) or 1.0
        return tx / k, ty / k

    out = []
    for i in range(n if closed else n - 1):
        x0, y0 = pts[i]
        dx, dy, length = seg(i)
        t0, t1 = tangent(i) or (dx / length, dy / length), tangent(i + 1) or (dx / length, dy / length)
        steps = max(int(length // 6), 1)
        for k in range(steps):
            u = k / steps                                   # cubic Hermite between the two points
            h1, h2, h3 = 3 * u * u - 2 * u ** 3, u ** 3 - 2 * u * u + u, u ** 3 - u * u
            out.append((round(x0 + dx * h1 + length * (t0[0] * h2 + t1[0] * h3)),
                        round(y0 + dy * h1 + length * (t0[1] * h2 + t1[1] * h3))))
    return out if closed else out + [pts[-1]]


def points(text: str | None) -> list[list]:
    """'10,20 30,40,rings' -> [[10, 20], [30, 40, 'rings']]"""
    out = []
    for pair in (text or "").split():
        out.append([int(v) if v.lstrip("-").isdigit() else v for v in pair.split(",")])
    return out


def _fill(size: tuple[int, int], polys: list) -> np.ndarray:
    surf = pygame.Surface(size, depth=8)
    surf.set_palette([(0, 0, 0)] + [(255, 255, 255)] * 255)
    for p in polys:
        pygame.draw.polygon(surf, 1, p)
    return pygame.surfarray.array2d(surf).T.astype(bool)


def _box(m: np.ndarray, r: int) -> tuple:
    """Solid pixels in the (2r+1) square around each pixel, and that square's area at the edges."""
    H, W = m.shape
    I = np.zeros((H + 1, W + 1), dtype=np.int32)
    I[1:, 1:] = np.cumsum(np.cumsum(m, axis=0, dtype=np.int32), axis=1, dtype=np.int32)
    y0, y1 = np.clip(np.arange(H) - r, 0, H), np.clip(np.arange(H) + r + 1, 0, H)
    x0, x1 = np.clip(np.arange(W) - r, 0, W), np.clip(np.arange(W) + r + 1, 0, W)
    total = I[y1[:, None], x1[None, :]] - I[y0[:, None], x1[None, :]] - I[y1[:, None], x0[None, :]] + I[y0[:, None], x0[None, :]]
    return total, (y1 - y0)[:, None] * (x1 - x0)[None, :]


def _skin(mask: np.ndarray) -> np.ndarray:
    """Solid just inside a steep face: air close beside it, none close above it. Walls stay walls."""
    air = ~mask
    side, over = air.copy(), air.copy()
    for k in range(1, SKIN + 1):
        side[:, k:] |= air[:, :-k]
        side[:, :-k] |= air[:, k:]
    for k in range(1, 2 * SKIN + 1):
        over[k:, :] |= air[:-k, :]
    return mask & side & ~over


def _bridge_rows(m: np.ndarray, r: int) -> np.ndarray:
    """m with every gap of up to 2r px along a row filled (a closing along x only)."""
    def around(a: np.ndarray) -> np.ndarray:
        c = np.zeros((a.shape[0], a.shape[1] + 1), dtype=np.int32)
        c[:, 1:] = np.cumsum(a, axis=1, dtype=np.int32)
        x0, x1 = np.clip(np.arange(a.shape[1]) - r, 0, a.shape[1]), np.clip(np.arange(a.shape[1]) + r + 1, 0, a.shape[1])
        return c[:, x1] - c[:, x0]
    return around(~(around(m) > 0)) == 0


def _open(m: np.ndarray, r: int) -> np.ndarray:
    """Drop anything narrower than a (2r+1) square, keeping everything else as it was."""
    total, area = _box(m, r)
    return _box(total == area, r)[0] > 0


def _smooth(m: np.ndarray, r: int) -> np.ndarray:
    """Round off stair steps: a pixel keeps the majority of the square around it."""
    total, area = _box(m, r)
    return total * 2 >= area


def _grow(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    for k in range(1, r + 1):
        out[k:, :] |= m[:-k, :]
        out[:-k, :] |= m[k:, :]
        out[:, k:] |= m[:, :-k]
        out[:, :-k] |= m[:, k:]
    return out


def _join(lines: list) -> list:
    """Tracing breaks a walkway at every flower and post: mend gaps of a few columns at the same level."""
    out = []
    for line in sorted(lines, key=lambda l: (l[0][0], l[0][1])):
        for prev in out:
            if 0 <= line[0][0] - prev[-1][0] <= 48 and abs(line[0][1] - prev[-1][1]) <= 16:
                prev.extend(line)
                break
        else:
            out.append(list(line))
    return out


def _edge(mask: np.ndarray, x: int, y: int, step: int) -> int:
    """How far a walkway's end must run on to reach open air just above its floor (12 px if none is near)."""
    H, W = mask.shape
    for k in range(1, 65):
        xx = x + k * step
        if 0 <= xx < W and not mask[max(y - 8, 0), xx]:
            return xx
    return x + 12 * step


def _carve_walk(mask: np.ndarray, one_way: np.ndarray, line: list) -> None:
    """Open the wall above a walkway: a grass strip with dirt over it is a path in front of a wall."""
    H, W = mask.shape
    # tracing stops short of the wall's edge: run on past both ends so no sliver is left standing
    line = [(_edge(mask, *line[0], -1), line[0][1])] + line + [(_edge(mask, *line[-1], 1), line[-1][1])]
    for (x0, y0), (x1, y1) in zip(line, line[1:]):
        for x in range(max(x0, 0), min(x1, W - 1) + 1):
            y = y0 + (y1 - y0) * (x - x0) // max(x1 - x0, 1)
            top = max(y - WALK_CLEARANCE, 0)
            above = max(top - SHELL, 0)
            air = np.nonzero(~mask[above:y, x])[0]
            air = air[air < (y - above) // 2]   # open sky over the wall, not a pocket at the path's own level
            if air.size:                        # another surface sits close above: keep a roof under it...
                roof = above + int(air[-1]) + 1
                top = roof if y - roof <= LUMP else min(max(top, roof + SHELL), y)   # ...unless it is a fern on the path
            mask[top:y, x] = False
            air = np.nonzero(~mask[:top, x])[0]
            if air.size:                        # up to the next open air the dirt is scenery, passable from below
                one_way[air[-1] + 1:top, x] = True


def _angle_table(mask: np.ndarray, side: int) -> bytes:
    """Per tile, the angle of the surface exposed on `side`: a line fit through its boundary pixels."""
    H, W = mask.shape
    if side == FLOOR:
        edge = mask.copy()
        edge[1:, :] &= ~mask[:-1, :]
    elif side == CEILING:
        edge = mask.copy()
        edge[:-1, :] &= ~mask[1:, :]
    elif side == RIGHT:         # a wall on the body's right: solid with air to its left
        edge = mask.copy()
        edge[:, 1:] &= ~mask[:, :-1]
    else:
        edge = mask.copy()
        edge[:, :-1] &= ~mask[:, 1:]
    ys, xs = np.nonzero(edge)
    tw, th = W // TILE + 1, H // TILE + 1
    tid = (ys // TILE) * tw + xs // TILE
    fx, fy = xs.astype(np.float64), ys.astype(np.float64)
    if side in (RIGHT, LEFT):
        fx, fy = fy, fx                         # fit x = m y + c on walls
    sums = [np.bincount(tid, w, tw * th).astype(np.float64) for w in (None, fx, fy, fx * fx, fx * fy)]

    def around(a: np.ndarray) -> np.ndarray:
        """Each tile's total with its eight neighbours."""
        g, out = a.reshape(th, tw), np.zeros((th, tw))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                out[max(dy, 0):th + min(dy, 0), max(dx, 0):tw + min(dx, 0)] += \
                    g[max(-dy, 0):th + min(-dy, 0), max(-dx, 0):tw + min(-dx, 0)]
        return out.ravel()

    def slope(n, sx, sy, sxx, sxy) -> np.ndarray:
        den = n * sxx - sx * sx
        return np.divide(n * sxy - sx * sy, den, out=np.zeros_like(n), where=den > 1e-6)

    # a surface that only clips a tile's corner leaves too few pixels to fit: read those from the tiles around
    n = sums[0]
    m = np.where(n >= FIT_MIN, slope(*sums), slope(*[around(a) for a in sums]))
    ok = n > 0
    if side == FLOOR:
        rad = np.arctan2(m, 1.0)
    elif side == CEILING:
        rad = np.arctan2(-m, -1.0)
    elif side == RIGHT:
        rad = np.arctan2(-1.0, -m)
    else:
        rad = np.arctan2(1.0, m)
    ang = (np.rint(rad * 128.0 / math.pi).astype(np.int64) & 0xFF) & ~1     # even: a real surface
    # near-level ground reads exactly level, as hand-authored tiles do
    if side == FLOOR:
        ang[(ang <= 4) | (ang >= 252)] = 0
    elif side == CEILING:
        ang[np.abs(ang - 0x80) <= 4] = 0x80
    elif side == RIGHT:
        ang[np.abs(ang - 0xC0) <= 4] = 0xC0
    else:
        ang[np.abs(ang - 0x40) <= 4] = 0x40
    ang[~ok] = NO_ANGLE
    return ang.astype(np.uint8).tobytes()


class Level:
    """One act: collision planes, angle tables and the object lists, read by SonicCore."""

    def __init__(self, name: str, path: str | None = None):
        with open(path or os.path.join(LEVEL_DIR, f"{name}.yaml"), encoding="utf-8") as f:
            d = yaml.safe_load(f)
        with open(os.path.join(LEVEL_DIR, "tube.yaml"), encoding="utf-8") as f:
            tube = delta(yaml.safe_load(f)["tube"])
        self.name, self.data = name, d
        self.act = int(d["act"])
        self.start = tuple(d["start"])
        self.camera_right = int(d["camera_right"])
        self.camera_bottom_max = 1536 - 224
        self.width = self.camera_right + 320                        # the level's playable right edge
        self.polys = polys = [curve(delta(s)) for s in d["solid"]]
        H = max(y for p in polys for _, y in p)
        W = max(max(x for p in polys for x, _ in p), self.width + 64)
        self.px_w, self.px_h = W, H
        base = _fill((W, H), polys)
        traced = base.copy()
        one_way = np.zeros_like(base)
        for line in _join([delta(s) for s in d.get("walk") or []]):
            _carve_walk(base, one_way, curve(line, closed=False))
        # an object standing in front of a wall sits on a walkway, even where no grass shows between its legs
        for x0, x1, y in points(d.get("spikes")):
            self._walk_under(base, one_way, x0, x1, y + 16)
        for key, drop in (("monitors", 16), ("spring_up", 8), ("spring_up_red", 8), ("rock", 16)):
            for x, y, *_ in points(d.get(key)):
                self._walk_under(base, one_way, x, x, y + drop)
        for x0, y0, x1, y1 in points(d.get("open")) + points(d.get("pocket")):    # a passage no trace shows; off_route's mutant
            base[y0:y1, x0:x1] = False
        # a thin strip between two rooms is wall art, not a wall: the trace saw dirt, the original runs through it
        base &= ~_bridge_rows(traced & ~base, THIN_WALL // 2)
        self.tubes = [(x, y, x + 256, y + 288) for x, y in d.get("tubes") or []]
        if self.tubes:
            bore = _fill((W, H), [[(x + tx, y + ty) for tx, ty in tube] for x, y in d["tubes"]])
            base &= ~_grow(_smooth(bore, 9), TUBE_GROW)
            one_way &= _box(bore, TUBE_WALL)[0] == 0        # a tunnel needs its walls to hold the body in
        base = _open(base, SLIVER)                          # carving leaves hairlines: invisible, yet you can stand on them
        one_way &= ~_skin(base)
        for x0, y0, x1, y1 in points(d.get("block")):       # seal_route's mutant
            base[y0:y1, x0:x1], one_way[y0:y1, x0:x1] = True, False
        for x0, x1, y in (d.get("bridges") or []) + (d.get("gaps") or []):     # the logs are objects
            base[max(y - 32, 0):y + 24, x0 + 24:x1 - 23] = False
        self.loops = []
        planes = [base, base.copy()]
        for x0, y0, x1, y1 in d.get("loops") or []:
            # the traced hole ends at the grass tips: the running surface is GRASS px lower
            y1 += GRASS
            cx, cy, rx, ry = (x0 + x1) // 2, (y0 + y1) // 2, (x1 - x0) // 2, (y1 - y0) // 2
            self.loops.append((cx, cy, ry, y1))
            yy, xx = np.ogrid[0:H, 0:W]
            hole = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
            for p in planes:
                p &= ~hole
            one_way[y0 - GRASS:y1, cx - 128:cx + 128] = False   # a loop's ring holds the body too
            planes[0][y0:y1, cx - 128:cx - 16] = False      # the way in: run through the left leg
            planes[1][y0:y1, cx + 16:cx + 128] = False      # the way out: run through the right leg
        self.masks_np = planes
        self.grid = planes[0][16::32, 16::32].astype(int).tolist()      # 32 px cells for the debug panel's agent view
        self.rows, self.cols = len(self.grid), len(self.grid[0])
        self.masks = [(p.astype(np.uint8) * (1 + one_way.astype(np.uint8))).tobytes() for p in planes]
        self.angles = [[_angle_table(p, s) for s in (FLOOR, RIGHT, CEILING, LEFT)] for p in planes]
        self.tw = W // TILE + 1
        self.bridges = list(d.get("bridges") or [])
        self.waterfalls = list(d.get("waterfalls") or [])
        self.sign = tuple(d["sign"]) if d.get("sign") else None
        self.capsule = tuple(d["capsule"]) if d.get("capsule") else None
        gx, gy = self.sign or self.capsule
        self.goals = [SimpleNamespace(x=gx, y=gy, gObj=SimpleNamespace(x=gx, y=gy, active=True))]
        self.coins = points(d.get("rings"))
        self.objects = {k: points(d.get(k)) for k in (
            "rings", "spikes", "spring_up", "spring_up_red", "spring_side", "rock", "monitors", "lamp",
            "buzz", "crab", "moto", "newtron", "chopper", "swing", "platforms", "ledges", "totem", "palm",
            "flower", "sunflower")}

    @staticmethod
    def _walk_under(mask: np.ndarray, one_way: np.ndarray, x0: int, x1: int, y: int) -> None:
        if mask[max(y - 56, 0), x0:x1 + 1].any():           # dirt right over it: open sky means it stands in a pit
            _carve_walk(mask, one_way, [(x0 - 24, y), (x1 + 24, y)])

    def solid(self, x: int, y: int, plane: int = 0) -> bool:
        return 0 <= x < self.px_w and 0 <= y < self.px_h and bool(self.masks[plane][y * self.px_w + x])


@lru_cache(maxsize=4)
def load(name: str, path: str | None = None) -> Level:
    return Level(name, path)
