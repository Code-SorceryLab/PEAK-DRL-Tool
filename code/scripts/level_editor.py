#!/usr/bin/env python3
"""
PEAK Level Editor
-----------------
Standalone pygame level editor for PEAK platformer levels.
Paint tiles, place entities, and export to the ASCII .txt format
that LevelLoader reads directly.

Controls:
  Left click            — paint selected tile
  Right click           — erase (set to air)
  Middle drag           — pan camera
  Scroll wheel          — zoom in/out
  Ctrl+S                — save
  Ctrl+O                — open
  Ctrl+N                — new level
  Ctrl+Z / Ctrl+Shift+Z — undo / redo
  G                     — toggle grid
  F                     — fill tool toggle
  Home                  — reset camera to 0,0 and zoom to 1.0
  1–0                   — select tile by number

Usage:
  python level_editor.py                  # new blank level
  python level_editor.py stage_7.txt      # open existing level
"""

import os
import sys
import pygame
import pygame.freetype
from pathlib import Path
from tkinter import filedialog, simpledialog
import tkinter as tk

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_W, WINDOW_H = 1400, 850
TILE_SIZE  = 32
MIN_ZOOM   = 0.25
MAX_ZOOM   = 4.0
UNDO_LIMIT = 200

PANEL_W   = 220
TOOLBAR_H = 48
STATUS_H  = 28

DEFAULT_ROWS = 34
DEFAULT_COLS = 75

# ─────────────────────────────────────────────────────────────────────────────
# Game object imports (optional — falls back to colored rects)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from code.games.modules.Objects.Tile import Tile, create_tile
    from code.games.modules.Objects.Coin import Coin
    from code.games.modules.Objects.Enemy import Enemy
    from code.games.modules.Objects.Goal import Goal
    from code.games.modules.Objects.QuestionBlock import QuestionBlock
    from code.games.modules.Objects.GameObject import GameObject
    from code.games.modules.Parameters.Map_parameters import (
        TILE_GROUND, TILE_PLATFORM, TILE_SPIKE,
        COLOR_GROUND, COLOR_PLATFORM, COLOR_SPIKE, COLOR_GOAL
    )
    from code.games.modules.System.EntityType import EntityType
    USE_REAL_OBJECTS = True
except ImportError:
    USE_REAL_OBJECTS = False

# ─────────────────────────────────────────────────────────────────────────────
# Tile definitions
# ─────────────────────────────────────────────────────────────────────────────
TILES = [
    (' ',  'Air',            ( 30,  35,  50), ( 80,  90, 120)),
    ('#',  'Ground',         ( 80,  60,  40), (220, 200, 160)),
    ('=',  'Platform',       ( 60,  90,  60), (180, 230, 140)),
    ('^',  'Spike',          (200,  60,  60), (255, 200, 180)),
    ('?',  'QBlock (coin)',  (220, 180,  40), ( 80,  60,   0)),
    ('>',  'QBlock (star)',  (255, 220,  80), ( 80,  60,   0)),
    ('<',  'QBlock (mush)',  (200,  80,  80), (255, 240, 240)),
    ('C',  'Coin',           (255, 215,   0), ( 80,  60,   0)),
    ('E',  'Enemy',          (180,  60, 180), (255, 220, 255)),
    ('G',  'Goal',           ( 60, 200, 100), (  0,  60,  20)),
    ('P',  'Player Start',   ( 60, 160, 255), (  0,  30, 120)),
]

TILE_BY_CHAR = {t[0]: t for t in TILES}
SOLID_CHARS  = {'#', '=', '?', '>', '<'}

# ─────────────────────────────────────────────────────────────────────────────
# Moving Platform definitions (editor-side)
# ─────────────────────────────────────────────────────────────────────────────
PLAT_DEFAULT_W   = TILE_SIZE * 3   # 96 px
PLAT_DEFAULT_H   = TILE_SIZE // 2  # 16 px
PLAT_DEFAULT_SPD = 80.0            # px/s

PLAT_BODY_COL  = (205, 133,  63)   # tan body  (matches static platform)
PLAT_HIGH_COL  = (230, 165,  90)   # top highlight
PLAT_PATH_COL  = (255, 200,  80)   # travel-path line
PLAT_SEL_COL   = (255, 255, 100)   # selected outline
HANDLE_START   = ( 80, 220,  80)   # green  — start handle
HANDLE_END     = (220,  80,  80)   # red    — end handle
HANDLE_R       = 9                 # handle circle radius (pixels)
HANDLE_HIT_R   = 14               # grab-zone radius


class PlatformDef:
    """Lightweight editor-side record for a moving platform."""
    __slots__ = ('start', 'end', 'speed', 'width', 'height')

    def __init__(self, start, end,
                 speed=PLAT_DEFAULT_SPD,
                 width=PLAT_DEFAULT_W,
                 height=PLAT_DEFAULT_H):
        self.start  = list(start)   # [world_px_x, world_px_y]
        self.end    = list(end)     # [world_px_x, world_px_y]
        self.speed  = float(speed)
        self.width  = int(width)
        self.height = int(height)

    def to_dict(self):
        return {
            'start':  [int(self.start[0]), int(self.start[1])],
            'end':    [int(self.end[0]),   int(self.end[1])],
            'speed':  self.speed,
            'width':  self.width,
            'height': self.height,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d['start'], d['end'],
                   d.get('speed', PLAT_DEFAULT_SPD),
                   d.get('width', PLAT_DEFAULT_W),
                   d.get('height', PLAT_DEFAULT_H))

# ─────────────────────────────────────────────────────────────────────────────
# UI Colors
# ─────────────────────────────────────────────────────────────────────────────
UI_BG      = ( 18,  22,  32)
UI_PANEL   = ( 26,  30,  44)
UI_BORDER  = ( 50,  58,  80)
UI_SELECT  = ( 60, 120, 220)
UI_TEXT    = (200, 210, 230)
UI_SUBTEXT = (100, 115, 145)
UI_TOOLBAR = ( 22,  26,  38)
UI_STATUS  = ( 16,  20,  28)
GRID_COLOR = ( 40,  50,  70)
GRID_BOLD  = ( 60,  75, 100)


# ─────────────────────────────────────────────────────────────────────────────
# Real-object renderer
# ─────────────────────────────────────────────────────────────────────────────
def render_tile_object(surf, char, sx, sy, ts):
    """
    Render using real game object render methods.
    Objects instantiated fresh each call — no update, no state.
    """
    ti = int(ts)

    if char == '#':
        t = create_tile(TILE_GROUND, 0, 0, True, COLOR_GROUND)
        t.gObj.width = t.gObj.height = ti
        t.gObj.x = sx; t.gObj.y = sy
        t.render(surf, 0, 0)

    elif char == '=':
        t = create_tile(TILE_PLATFORM, 0, 0, True, COLOR_PLATFORM)
        t.gObj.width = t.gObj.height = ti
        t.gObj.x = sx; t.gObj.y = sy
        t.render(surf, 0, 0)

    elif char == '^':
        t = create_tile(TILE_SPIKE, 0, 0, False, COLOR_SPIKE)
        t.gObj.width = t.gObj.height = ti
        t.gObj.x = sx; t.gObj.y = sy
        t.render(surf, 0, 0)

    elif char == 'G':
        g = Goal(gObj=GameObject(float(sx), float(sy), ti, ti))
        g.render(surf, sx, sy)

    elif char == 'C':
        half = ti // 2
        c = Coin(gObj=GameObject(float(sx + half), float(sy + half), ti, ti))
        c.radius = max(4, half - 2)
        c.render(surf, sx + half, sy + half)

    elif char == 'E':
        e = Enemy(gObj=GameObject(float(sx), float(sy), ti, ti))
        e.render(surf, sx, sy)

    elif char in ('?', '>', '<'):
        contains = {'?': 'coin', '>': 'star', '<': 'mushroom'}[char]
        qb = QuestionBlock(
            gObj=GameObject(float(sx), float(sy), ti, ti), contains=contains
        )
        qb.render(surf, sx, sy)

    elif char == 'P':
        # Player has asset dependencies — colored rect fallback
        pygame.draw.rect(surf, (60, 160, 255), (sx, sy, ti, ti))
        pygame.draw.circle(
            surf, (255, 255, 255),
            (sx + ti * 3 // 4, sy + ti // 3),
            max(2, ti // 8)
        )

    else:
        # Air or unknown
        pygame.draw.rect(surf, UI_BG, (sx, sy, ti, ti))


# ─────────────────────────────────────────────────────────────────────────────
# Fallback colored-rect renderer
# ─────────────────────────────────────────────────────────────────────────────
def draw_tile_rect(surf, char, x, y, w, h, alpha=255):
    tile  = TILE_BY_CHAR.get(char, TILE_BY_CHAR[' '])
    color = tile[2]
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((*color, alpha))
    if char in SOLID_CHARS and w > 4:
        highlight = tuple(min(255, c + 40) for c in color)
        pygame.draw.line(s, (*highlight, alpha), (0, 0), (w - 1, 0), max(1, h // 12))
    surf.blit(s, (x, y))


def draw_tile_symbol(surf, font, char, x, y, w, h):
    if char == ' ':
        return
    tile    = TILE_BY_CHAR.get(char, TILE_BY_CHAR[' '])
    display = {
        '#': '█', '=': '▬', '^': '▲', '?': '?', '>': '★',
        '<': '♦', 'C': '●', 'E': '☻', 'G': '⚑', 'P': '▶',
    }.get(char, char)
    size       = max(8, min(w - 4, h - 4, 20))
    text_color = tile[3]
    try:
        bounds = font.get_rect(display, size=size)
        tx = x + (w - bounds.width)  // 2
        ty = y + (h - bounds.height) // 2
        font.render_to(surf, (tx, ty), display, fgcolor=text_color, size=size)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Level
# ─────────────────────────────────────────────────────────────────────────────
class Level:
    def __init__(self, rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
        self.rows      = rows
        self.cols      = cols
        self.grid      = [[' '] * cols for _ in range(rows)]
        self.platforms = []   # List[PlatformDef]
        self.filename  = None

    def get(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c]
        return None

    def set(self, r, c, char):
        """Set a tile, auto-expanding the canvas if r/c is outside current bounds."""
        if r < 0 or c < 0:
            return

        # Expand rows
        while r >= self.rows:
            self.grid.append([' '] * self.cols)
            self.rows += 1

        # Expand cols
        if c >= self.cols:
            extra = c - self.cols + 1
            for row in self.grid:
                row.extend([' '] * extra)
            self.cols = c + 1

        self.grid[r][c] = char

    def bounding_box(self):
        """
        Returns (min_row, min_col, max_row, max_col) of non-air content.
        Returns None if the level is completely empty.
        """
        min_r = self.rows;  max_r = -1
        min_c = self.cols;  max_c = -1
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] != ' ':
                    min_r = min(min_r, r);  max_r = max(max_r, r)
                    min_c = min(min_c, c);  max_c = max(max_c, c)
        if max_r == -1:
            return None
        return min_r, min_c, max_r, max_c

    def to_ascii(self, trim=True, padding=1):
        """
        Export to ASCII string.
        trim=True  — crop to content bounding box + padding.
        padding    — empty tile border around content (default 1).
        """
        if not trim:
            return '\n'.join(''.join(row) for row in self.grid)

        bb = self.bounding_box()
        if bb is None:
            return '\n'.join([' ' * DEFAULT_COLS] * DEFAULT_ROWS)

        min_r, min_c, max_r, max_c = bb
        r0 = max(0, min_r - padding)
        c0 = max(0, min_c - padding)
        r1 = min(self.rows - 1, max_r + padding)
        c1 = min(self.cols - 1, max_c + padding)

        width = c1 - c0 + 1
        lines = []
        for r in range(r0, r1 + 1):
            row_chars = self.grid[r][c0:c1 + 1]
            while len(row_chars) < width:
                row_chars.append(' ')
            lines.append(''.join(row_chars))
        return '\n'.join(lines)

    def save(self, path, trim=True):
        """Save .txt ASCII map and a sibling .yaml sidecar for dynamic objects."""
        with open(path, 'w') as f:
            f.write(self.to_ascii(trim=trim))
        self.filename = path
        # Always write the sidecar (even if empty) so LevelLoader knows the file is clean
        self._save_yaml(path)

    def _save_yaml(self, txt_path):
        """Write the [level].yaml sidecar alongside the .txt file."""
        import yaml
        yaml_path = str(txt_path).rsplit('.', 1)[0] + '.yaml'
        dynamics = {}
        if self.platforms:
            dynamics['moving_platforms'] = [p.to_dict() for p in self.platforms]
        data = {'dynamics': dynamics} if dynamics else {'dynamics': {}}
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def _load_yaml(self, txt_path):
        """Read [level].yaml sidecar if it exists; populate self.platforms."""
        import yaml
        yaml_path = str(txt_path).rsplit('.', 1)[0] + '.yaml'
        if not Path(yaml_path).exists():
            return
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            dynamics = data.get('dynamics', {}) or {}
            for pd in dynamics.get('moving_platforms', []):
                self.platforms.append(PlatformDef.from_dict(pd))
        except Exception as e:
            print(f"[Editor] YAML sidecar load error: {e}")

    def clone(self):
        new           = Level(self.rows, self.cols)
        new.grid      = [row[:] for row in self.grid]
        new.filename  = self.filename
        new.platforms = [PlatformDef(p.start[:], p.end[:], p.speed, p.width, p.height)
                         for p in self.platforms]
        return new

    @classmethod
    def from_ascii(cls, text):
        lines = text.split('\n')
        rows  = len(lines)
        cols  = max(len(ln) for ln in lines) if rows else DEFAULT_COLS
        level = cls(rows, cols)
        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                level.grid[r][c] = ch
        return level

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            text = f.read()
        level          = cls.from_ascii(text)
        level.filename = path
        level._load_yaml(path)   # pull in moving platforms from sidecar
        return level


# ─────────────────────────────────────────────────────────────────────────────
# Editor state
# ─────────────────────────────────────────────────────────────────────────────
class Editor:
    def __init__(self, level: Level):
        self.level      = level
        self.undo_stack = []
        self.redo_stack = []

        self.selected_tile = '#'
        self.show_grid     = True
        self.fill_mode     = False

        self.cam_x = 0.0
        self.cam_y = 0.0
        self.zoom  = 1.0

        self.painting    = False
        self.paint_char  = ' '
        self.panning     = False
        self.pan_start   = (0, 0)
        self.pan_cam     = (0.0, 0.0)
        self.last_cell   = None
        self.hover_cell  = None
        self.dirty       = False

        # ── Moving-platform tool state ─────────────────────────────────────
        # When selected_tile == 'M' the editor is in platform-placement mode.
        self.plat_placing     = False          # True between click-1 and click-2
        self.plat_start_world = None           # (px, py) world coords of click-1
        self.plat_ghost_end   = None           # (px, py) world coords of mouse hover
        self.plat_default_spd = PLAT_DEFAULT_SPD

        # Selection / handle-drag
        self.sel_plat_idx   = None             # index into level.platforms
        self.drag_handle    = None             # 'start' or 'end'
        self.drag_offset    = (0.0, 0.0)      # offset from handle centre to click

        self._center_camera()

    def _center_camera(self):
        vp_w    = WINDOW_W - PANEL_W
        vp_h    = WINDOW_H - TOOLBAR_H - STATUS_H
        level_w = self.level.cols * TILE_SIZE * self.zoom
        level_h = self.level.rows * TILE_SIZE * self.zoom
        self.cam_x = (level_w - vp_w) / 2
        self.cam_y = (level_h - vp_h) / 2

    def reset_view(self):
        """Home key — snap camera to origin and reset zoom to 1.0."""
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.zoom  = 1.0

    # ── Undo / Redo ───────────────────────────────────────────────────────
    def push_undo(self):
        self.undo_stack.append(self.level.clone())
        if len(self.undo_stack) > UNDO_LIMIT:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.level.clone())
            self.level = self.undo_stack.pop()
            self.dirty = True

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.level.clone())
            self.level = self.redo_stack.pop()
            self.dirty = True

    # ── Coordinate transforms ─────────────────────────────────────────────
    def world_to_screen(self, wx, wy):
        ts = TILE_SIZE * self.zoom
        return (
            int(wx * ts - self.cam_x + PANEL_W),
            int(wy * ts - self.cam_y + TOOLBAR_H),
        )

    def world_pixel_to_screen(self, px, py):
        """Convert world-space pixels (not tile units) to screen coords."""
        return (
            int(px * self.zoom - self.cam_x + PANEL_W),
            int(py * self.zoom - self.cam_y + TOOLBAR_H),
        )

    def screen_to_world_pixel(self, sx, sy, snap=True):
        """
        Convert screen coords to world-space pixels.
        If snap=True, snap to nearest tile-grid boundary.
        """
        ts = TILE_SIZE * self.zoom
        wx = (sx - PANEL_W   + self.cam_x) / self.zoom
        wy = (sy - TOOLBAR_H + self.cam_y) / self.zoom
        if snap:
            wx = round(wx / TILE_SIZE) * TILE_SIZE
            wy = round(wy / TILE_SIZE) * TILE_SIZE
        return wx, wy

    def screen_to_cell(self, sx, sy):
        ts = TILE_SIZE * self.zoom
        wx = (sx - PANEL_W   + self.cam_x) / ts
        wy = (sy - TOOLBAR_H + self.cam_y) / ts
        return int(wy), int(wx)   # no clamp — Level.set expands canvas

    # ── Paint ─────────────────────────────────────────────────────────────
    def paint_cell(self, r, c, char):
        if (r, c) == self.last_cell:
            return
        old = self.level.get(r, c)   # None = outside current bounds (air)
        if old != char:
            self.level.set(r, c, char)
            self.last_cell = (r, c)
            self.dirty     = True

    def flood_fill(self, r, c, char):
        target = self.level.get(r, c)
        if target is None or target == char:
            return
        self.push_undo()
        stack   = [(r, c)]
        visited = set()
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            if self.level.get(cr, cc) != target:
                continue
            visited.add((cr, cc))
            self.level.set(cr, cc, char)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if (nr, nc) not in visited and self.level.get(nr, nc) == target:
                    stack.append((nr, nc))
        self.dirty = True

    def hit_test_platform_handle(self, sx, sy):
        """
        Returns (plat_idx, 'start'|'end') if screen pos (sx,sy) is within
        HANDLE_HIT_R pixels of any platform handle, else (None, None).
        """
        import math
        for i, p in enumerate(self.level.platforms):
            for handle in ('start', 'end'):
                wp = p.start if handle == 'start' else p.end
                hsx, hsy = self.world_pixel_to_screen(wp[0] + p.width / 2,
                                                       wp[1] + p.height / 2)
                if math.hypot(sx - hsx, sy - hsy) <= HANDLE_HIT_R:
                    return i, handle
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Draw functions
# ─────────────────────────────────────────────────────────────────────────────
def draw_platforms(surf, editor: 'Editor', vp_rect):
    """
    Render all placed moving platforms plus the ghost being drawn.
    Layers per platform:
      1. Dashed travel-path line (start-centre → end-centre)
      2. Platform body at start  (solid rect)
      3. Platform body at end    (ghost rect, 50% alpha)
      4. Handle circles at both ends
      5. Speed label
    """
    import math

    def plat_body_screen(p, wp):
        sx, sy = editor.world_pixel_to_screen(wp[0], wp[1])
        sw = int(p.width  * editor.zoom)
        sh = int(p.height * editor.zoom)
        return sx, sy, sw, sh

    def draw_handle(scx, scy, col, label):
        pygame.draw.circle(surf, col,         (scx, scy), HANDLE_R)
        pygame.draw.circle(surf, (255,255,255),(scx, scy), HANDLE_R, 2)

    def draw_dashed_line(surf, col, p1, p2, dash=8, gap=5):
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length < 1: return
        ux, uy = dx / length, dy / length
        pos = 0.0
        drawing = True
        while pos < length:
            seg = dash if drawing else gap
            end = min(pos + seg, length)
            if drawing:
                x1 = int(p1[0] + ux * pos);  y1 = int(p1[1] + uy * pos)
                x2 = int(p1[0] + ux * end);  y2 = int(p1[1] + uy * end)
                pygame.draw.line(surf, col, (x1, y1), (x2, y2), 1)
            pos += seg
            drawing = not drawing

    # ── Placed platforms ────────────────────────────────────────────────────
    for i, p in enumerate(editor.level.platforms):
        selected = (i == editor.sel_plat_idx)

        # Body at start (solid)
        bx, by, bw, bh = plat_body_screen(p, p.start)
        body_rect = pygame.Rect(bx, by, bw, bh)
        if body_rect.colliderect(vp_rect):
            pygame.draw.rect(surf, PLAT_BODY_COL, body_rect)
            hi_h = max(2, bh // 4)
            pygame.draw.rect(surf, PLAT_HIGH_COL, (bx, by, bw, hi_h))
            outline = PLAT_SEL_COL if selected else (160, 90, 30)
            pygame.draw.rect(surf, outline, body_rect, 2)

        # Ghost body at end (50% alpha)
        ex, ey, ew, eh = plat_body_screen(p, p.end)
        ghost = pygame.Surface((ew, eh), pygame.SRCALPHA)
        ghost.fill((*PLAT_BODY_COL, 100))
        pygame.draw.rect(ghost, (*PLAT_HIGH_COL, 100), (0, 0, ew, max(2, eh // 4)))
        outline_col = (*PLAT_SEL_COL, 220) if selected else (160, 90, 30, 120)
        pygame.draw.rect(ghost, outline_col, (0, 0, ew, eh), 2)
        surf.blit(ghost, (ex, ey))

        # Dashed travel-path line (centre to centre)
        sc  = (bx + bw // 2, by + bh // 2)
        ec  = (ex + ew // 2, ey + eh // 2)
        draw_dashed_line(surf, PLAT_PATH_COL, sc, ec)

        # Handles
        draw_handle(bx + bw // 2, by + bh // 2, HANDLE_START, 'S')
        draw_handle(ex + ew // 2, ey + eh // 2, HANDLE_END,   'E')

        # Speed label near start handle
        if editor.zoom >= 0.5:
            try:
                _f = pygame.font.Font(None, max(12, int(14 * editor.zoom)))
                lbl = _f.render(f"{p.speed:.0f}px/s", True, (255, 255, 200))
                surf.blit(lbl, (bx + bw // 2 + HANDLE_R + 2,
                                by + bh // 2 - lbl.get_height() // 2))
            except Exception:
                pass

    # ── Ghost being drawn (first click placed, waiting for second) ──────────
    if editor.selected_tile == 'M' and editor.plat_placing and editor.plat_ghost_end:
        sw_px, sh_px = PLAT_DEFAULT_W, PLAT_DEFAULT_H
        sx0, sy0 = editor.world_pixel_to_screen(*editor.plat_start_world)
        sw  = int(sw_px * editor.zoom); sh = int(sh_px * editor.zoom)
        ex0, ey0 = editor.world_pixel_to_screen(*editor.plat_ghost_end)

        ghost_s = pygame.Surface((sw, sh), pygame.SRCALPHA)
        ghost_s.fill((*PLAT_BODY_COL, 160))
        surf.blit(ghost_s, (sx0, sy0))
        ghost_e = pygame.Surface((sw, sh), pygame.SRCALPHA)
        ghost_e.fill((*PLAT_BODY_COL, 80))
        surf.blit(ghost_e, (ex0, ey0))

        sc = (sx0 + sw // 2, sy0 + sh // 2)
        ec = (ex0 + sw // 2, ey0 + sh // 2)
        draw_dashed_line(surf, PLAT_PATH_COL, sc, ec)
        pygame.draw.circle(surf, HANDLE_START, sc, HANDLE_R)
        pygame.draw.circle(surf, HANDLE_END,   ec, HANDLE_R)


def draw_editor(surf, editor: Editor, font, small_font, anim_t: float = 0.0):
    vp_x = PANEL_W
    vp_y = TOOLBAR_H
    vp_w = WINDOW_W - PANEL_W
    vp_h = WINDOW_H - TOOLBAR_H - STATUS_H

    pygame.draw.rect(surf, UI_BG, (vp_x, vp_y, vp_w, vp_h))

    ts      = TILE_SIZE * editor.zoom
    level   = editor.level
    vp_rect = pygame.Rect(vp_x, vp_y, vp_w, vp_h)

    col_start = max(0, int(editor.cam_x / ts))
    row_start = max(0, int(editor.cam_y / ts))
    col_end   = min(level.cols, col_start + int(vp_w / ts) + 2)
    row_end   = min(level.rows, row_start + int(vp_h / ts) + 2)

    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            sx, sy    = editor.world_to_screen(c, r)
            char      = level.get(r, c) or ' '
            tile_rect = pygame.Rect(sx, sy, int(ts), int(ts))
            if not tile_rect.colliderect(vp_rect):
                continue
            if USE_REAL_OBJECTS:
                render_tile_object(surf, char, sx, sy, int(ts))
            else:
                draw_tile_rect(surf, char, sx, sy, int(ts), int(ts))
                if ts >= 20:
                    draw_tile_symbol(surf, font, char, sx, sy, int(ts), int(ts))

    # Level border
    bx, by = editor.world_to_screen(0, 0)
    pygame.draw.rect(surf, (80, 100, 160),
                     (bx, by, int(level.cols * ts), int(level.rows * ts)), 2)

    # Moving platforms
    draw_platforms(surf, editor, vp_rect)

    # Grid lines
    if editor.show_grid and ts >= 6:
        for c in range(col_start, col_end + 1):
            sx   = int(c * ts - editor.cam_x + vp_x)
            bold = c % 10 == 0
            pygame.draw.line(surf, GRID_BOLD if bold else GRID_COLOR,
                             (sx, vp_y), (sx, vp_y + vp_h), 2 if bold else 1)
        for r in range(row_start, row_end + 1):
            sy   = int(r * ts - editor.cam_y + vp_y)
            bold = r % 10 == 0
            pygame.draw.line(surf, GRID_BOLD if bold else GRID_COLOR,
                             (vp_x, sy), (vp_x + vp_w, sy), 2 if bold else 1)

    # Hover highlight + ghost
    if editor.hover_cell:
        hr, hc = editor.hover_cell
        sx, sy = editor.world_to_screen(hc, hr)
        pygame.draw.rect(surf, (255, 255, 255), (sx, sy, int(ts), int(ts)), 2)
        ghost = pygame.Surface((int(ts), int(ts)), pygame.SRCALPHA)
        draw_tile_rect(ghost, editor.selected_tile, 0, 0, int(ts), int(ts), alpha=100)
        surf.blit(ghost, (sx, sy))


def draw_palette(surf, editor: Editor, font, small_font):
    pygame.draw.rect(surf, UI_PANEL,
                     (0, TOOLBAR_H, PANEL_W, WINDOW_H - TOOLBAR_H - STATUS_H))
    pygame.draw.line(surf, UI_BORDER,
                     (PANEL_W, TOOLBAR_H), (PANEL_W, WINDOW_H - STATUS_H), 1)

    y      = TOOLBAR_H + 12
    tile_h = 36
    pad    = 8

    small_font.render_to(surf, (12, y), "TILES", fgcolor=UI_SUBTEXT, size=11)
    y += 20

    for tile in TILES:
        char, label, color, text_col = tile
        rect     = pygame.Rect(pad, y, PANEL_W - pad * 2, tile_h)
        selected = char == editor.selected_tile

        pygame.draw.rect(surf, UI_SELECT if selected else UI_PANEL,
                         rect, border_radius=4)

        swatch = pygame.Rect(pad + 4, y + 4, tile_h - 8, tile_h - 8)
        pygame.draw.rect(surf, color, swatch, border_radius=3)

        display = {
            '#': '█', '=': '▬', '^': '▲', '?': '?', '>': '★',
            '<': '♦', 'C': '●', 'E': '☻', 'G': '⚑', 'P': '▶', ' ': '·'
        }.get(char, char)
        try:
            b  = small_font.get_rect(display, size=13)
            sx = swatch.x + (swatch.width  - b.width)  // 2
            sy = swatch.y + (swatch.height - b.height) // 2
            small_font.render_to(surf, (sx, sy), display, fgcolor=text_col, size=13)
        except Exception:
            pass

        text_color = UI_TEXT if selected else UI_SUBTEXT
        try:
            small_font.render_to(surf,
                                  (pad + tile_h + 4, y + tile_h // 2 - 6),
                                  label, fgcolor=text_color, size=12)
            small_font.render_to(surf,
                                  (PANEL_W - pad - 16, y + tile_h // 2 - 6),
                                  char if char != ' ' else '·',
                                  fgcolor=(70, 85, 110), size=11)
        except Exception:
            pass

        if selected:
            pygame.draw.rect(surf, UI_SELECT, rect, 2, border_radius=4)

        y += tile_h + 4

    # Tool toggles
    y += 8
    pygame.draw.line(surf, UI_BORDER, (pad, y), (PANEL_W - pad, y), 1)
    y += 10
    small_font.render_to(surf, (12, y), "TOOLS", fgcolor=UI_SUBTEXT, size=11)
    y += 18

    fill_rect = pygame.Rect(pad, y, PANEL_W - pad * 2, 30)
    pygame.draw.rect(surf, UI_SELECT if editor.fill_mode else UI_PANEL,
                     fill_rect, border_radius=4)
    small_font.render_to(surf, (pad + 8, y + 9), "F  Fill Mode",
                          fgcolor=UI_TEXT if editor.fill_mode else UI_SUBTEXT, size=12)
    y += 36

    grid_rect = pygame.Rect(pad, y, PANEL_W - pad * 2, 30)
    pygame.draw.rect(surf, UI_SELECT if editor.show_grid else UI_PANEL,
                     grid_rect, border_radius=4)
    small_font.render_to(surf, (pad + 8, y + 9), "G  Grid",
                          fgcolor=UI_TEXT if editor.show_grid else UI_SUBTEXT, size=12)
    y += 42

    # ── Moving Platform tool ────────────────────────────────────────────────
    pygame.draw.line(surf, UI_BORDER, (pad, y), (PANEL_W - pad, y), 1)
    y += 10
    small_font.render_to(surf, (12, y), "MOVING PLATFORMS", fgcolor=UI_SUBTEXT, size=11)
    y += 18

    plat_active = editor.selected_tile == 'M'
    plat_rect   = pygame.Rect(pad, y, PANEL_W - pad * 2, 30)
    pygame.draw.rect(surf, UI_SELECT if plat_active else UI_PANEL,
                     plat_rect, border_radius=4)
    # Mini platform icon
    ico = pygame.Rect(pad + 4, y + 9, 22, 6)
    pygame.draw.rect(surf, PLAT_BODY_COL, ico, border_radius=2)
    pygame.draw.rect(surf, PLAT_HIGH_COL, (ico.x, ico.y, ico.w, 2), border_radius=2)
    pygame.draw.circle(surf, HANDLE_START, (ico.x + 4,       ico.centery), 4)
    pygame.draw.circle(surf, HANDLE_END,   (ico.x + ico.w - 4, ico.centery), 4)
    small_font.render_to(surf, (pad + 30, y + 9), "M  Mov. Platform",
                          fgcolor=UI_TEXT if plat_active else UI_SUBTEXT, size=12)
    y += 36

    # Speed slider display
    speed_rect = pygame.Rect(pad, y, PANEL_W - pad * 2, 26)
    pygame.draw.rect(surf, UI_PANEL, speed_rect, border_radius=4)
    pygame.draw.rect(surf, UI_BORDER, speed_rect, 1, border_radius=4)
    lbl = f"Speed: {editor.plat_default_spd:.0f} px/s"
    small_font.render_to(surf, (pad + 8, y + 7), lbl, fgcolor=UI_TEXT, size=11)
    # – and + buttons
    btn_w = 18
    minus_r = pygame.Rect(PANEL_W - pad - btn_w * 2 - 4, y + 4, btn_w, 18)
    plus_r  = pygame.Rect(PANEL_W - pad - btn_w,         y + 4, btn_w, 18)
    for btn, sym in ((minus_r, '-'), (plus_r, '+')):
        pygame.draw.rect(surf, UI_BORDER, btn, border_radius=3)
        try:
            b = small_font.get_rect(sym, size=13)
            small_font.render_to(surf,
                                  (btn.x + (btn.w - b.width) // 2,
                                   btn.y + (btn.h - b.height) // 2),
                                  sym, fgcolor=UI_TEXT, size=13)
        except Exception:
            pass

    # Store button rects on editor for hit-testing in event loop
    editor._spd_minus_r = minus_r
    editor._spd_plus_r  = plus_r

    # Hint when placing
    y += 32
    if plat_active:
        if editor.plat_placing:
            hint = "Click: set END point"
            hcol = (255, 220, 80)
        else:
            hint = "Click: set START point"
            hcol = (180, 220, 180)
        small_font.render_to(surf, (pad + 4, y), hint, fgcolor=hcol, size=11)
        y += 16
        small_font.render_to(surf, (pad + 4, y), "Del: remove selected",
                              fgcolor=UI_SUBTEXT, size=11)


def draw_toolbar(surf, editor: Editor, small_font):
    pygame.draw.rect(surf, UI_TOOLBAR, (0, 0, WINDOW_W, TOOLBAR_H))
    pygame.draw.line(surf, UI_BORDER,
                     (0, TOOLBAR_H - 1), (WINDOW_W, TOOLBAR_H - 1), 1)

    title = "PEAK Level Editor"
    if editor.level.filename:
        title = f"PEAK Level Editor  —  {Path(editor.level.filename).name}"
    if editor.dirty:
        title += "  ●"
    small_font.render_to(surf, (PANEL_W + 12, 14), title, fgcolor=UI_TEXT, size=14)

    hints = ("Ctrl+S Save  |  Ctrl+O Open  |  Ctrl+N New  |  "
             "Ctrl+Z Undo  |  Home Reset View  |  Scroll Zoom  |  MMB Pan  |  "
             "M Platforms  |  Del Remove Plat  |  Esc Cancel")
    try:
        b = small_font.get_rect(hints, size=11)
        small_font.render_to(surf, (WINDOW_W - b.width - 12, 17),
                              hints, fgcolor=UI_SUBTEXT, size=11)
    except Exception:
        pass


def draw_status(surf, editor: Editor, small_font):
    y = WINDOW_H - STATUS_H
    pygame.draw.rect(surf, UI_STATUS, (0, y, WINDOW_W, STATUS_H))
    pygame.draw.line(surf, UI_BORDER, (0, y), (WINDOW_W, y), 1)

    level      = editor.level
    tile_label = TILE_BY_CHAR.get(editor.selected_tile, TILES[0])[1] \
                 if editor.selected_tile != 'M' else 'Moving Platform'

    info = (f"  {level.rows}r x {level.cols}c"
            f"  |  Zoom {editor.zoom:.2f}x"
            f"  |  Brush: {tile_label}"
            f"  |  Platforms: {len(level.platforms)}")

    if editor.hover_cell and editor.selected_tile != 'M':
        hr, hc   = editor.hover_cell
        char     = level.get(hr, hc)
        ch_label = TILE_BY_CHAR.get(char, TILES[0])[1] if char else 'Air'
        info    += f"  |  [{hr}, {hc}]: {ch_label}"

    if editor.selected_tile == 'M' and editor.sel_plat_idx is not None:
        idx = editor.sel_plat_idx
        if 0 <= idx < len(level.platforms):
            p = level.platforms[idx]
            info += (f"  |  Plat[{idx}] "
                     f"start=({p.start[0]:.0f},{p.start[1]:.0f}) "
                     f"end=({p.end[0]:.0f},{p.end[1]:.0f}) "
                     f"spd={p.speed:.0f}")

    # Live export size
    bb = level.bounding_box()
    if bb:
        min_r, min_c, max_r, max_c = bb
        exp_r = max_r - min_r + 3   # +2 padding +1 inclusive
        exp_c = max_c - min_c + 3
        info += f"  |  Export: {exp_r}r x {exp_c}c"
    else:
        info += "  |  Export: empty"

    undo_info = (f"  Undo: {len(editor.undo_stack)}"
                 f"  |  Redo: {len(editor.redo_stack)}")
    try:
        small_font.render_to(surf, (PANEL_W, y + 7),
                              info, fgcolor=UI_SUBTEXT, size=11)
        b = small_font.get_rect(undo_info, size=11)
        small_font.render_to(surf, (WINDOW_W - b.width - 12, y + 7),
                              undo_info, fgcolor=UI_SUBTEXT, size=11)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# File dialogs
# ─────────────────────────────────────────────────────────────────────────────
def dialog_open():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Open Level",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()
    return path or None


def dialog_save(default_name="level.txt"):
    root = tk.Tk(); root.withdraw()
    path = filedialog.asksaveasfilename(
        title="Save Level",
        defaultextension=".txt",
        initialfile=default_name,
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    root.destroy()
    return path or None


def dialog_new_size():
    root = tk.Tk(); root.withdraw()
    rows = simpledialog.askinteger("New Level", "Rows:",
                                   initialvalue=DEFAULT_ROWS,
                                   minvalue=5, maxvalue=200, parent=root)
    cols = simpledialog.askinteger("New Level", "Cols:",
                                   initialvalue=DEFAULT_COLS,
                                   minvalue=10, maxvalue=400, parent=root)
    root.destroy()
    return (rows, cols) if rows and cols else (None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global WINDOW_W, WINDOW_H

    pygame.init()
    pygame.freetype.init()

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
    pygame.display.set_caption("PEAK Level Editor")
    clock = pygame.time.Clock()

    font_path  = pygame.font.match_font("dejavusans,segoeui,arial")
    font       = pygame.freetype.Font(font_path)
    small_font = pygame.freetype.Font(font_path)

    # Load level from arg or create blank with border walls
    level = None
    if len(sys.argv) > 1:
        try:
            level = Level.load(sys.argv[1])
            print(f"[Editor] Loaded: {sys.argv[1]}")
        except Exception as e:
            print(f"[Editor] Failed to load {sys.argv[1]}: {e}")

    if level is None:
        level = Level()
        for r in range(level.rows):
            level.set(r, 0, '#')
            level.set(r, level.cols - 1, '#')
        for c in range(level.cols):
            level.set(0, c, '#')
            level.set(level.rows - 1, c, '#')
        level.set(level.rows - 2, 2, 'P')

    editor           = Editor(level)
    painting_started = False

    running = True
    while running:
        clock.tick(60)
        WINDOW_W, WINDOW_H = screen.get_size()

        mx, my      = pygame.mouse.get_pos()
        in_viewport = (PANEL_W <= mx < WINDOW_W and
                       TOOLBAR_H <= my < WINDOW_H - STATUS_H)

        # Hover — allow one cell outside bounds so canvas auto-expands on paint
        if in_viewport:
            r, c = editor.screen_to_cell(mx, my)
            if r >= 0 and c >= 0 and r <= editor.level.rows and c <= editor.level.cols:
                editor.hover_cell = (r, c)
            else:
                editor.hover_cell = None
        else:
            editor.hover_cell = None

        # ── Events ───────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

            elif event.type == pygame.MOUSEWHEEL and in_viewport:
                old_zoom    = editor.zoom
                factor      = 1.15 if event.y > 0 else (1 / 1.15)
                editor.zoom = max(MIN_ZOOM, min(MAX_ZOOM, editor.zoom * factor))
                world_mx    = (mx - PANEL_W   + editor.cam_x) / (TILE_SIZE * old_zoom)
                world_my    = (my - TOOLBAR_H + editor.cam_y) / (TILE_SIZE * old_zoom)
                editor.cam_x = world_mx * TILE_SIZE * editor.zoom - (mx - PANEL_W)
                editor.cam_y = world_my * TILE_SIZE * editor.zoom - (my - TOOLBAR_H)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    px, py = event.pos
                    if px < PANEL_W:
                        # ── Speed +/- buttons ─────────────────────────────
                        if hasattr(editor, '_spd_minus_r') and editor._spd_minus_r.collidepoint(px, py):
                            editor.plat_default_spd = max(20.0, editor.plat_default_spd - 20.0)
                        elif hasattr(editor, '_spd_plus_r') and editor._spd_plus_r.collidepoint(px, py):
                            editor.plat_default_spd = min(600.0, editor.plat_default_spd + 20.0)
                        else:
                            tile_h = 36; pad = 8
                            ty0    = TOOLBAR_H + 12 + 20
                            idx    = (py - ty0) // (tile_h + 4)
                            if 0 <= idx < len(TILES):
                                editor.selected_tile = TILES[idx][0]
                                # Switching away from platform tool resets placement
                                editor.plat_placing = False

                        # Check click on "M  Mov. Platform" button
                        # The button is drawn after the tile list + tools section.
                        # We stored the rect implicitly; detect by checking y range.
                        # Re-derive its y: TOOLBAR_H+12+20 + len(TILES)*(36+4) + tool-sections
                        tile_section_h = len(TILES) * (36 + 4) + 20
                        tool_section_h = 36 + 36 + 10 + 18  # fill + grid + separator + label
                        plat_btn_y     = TOOLBAR_H + 12 + tile_section_h + tool_section_h
                        plat_btn_rect  = pygame.Rect(8, plat_btn_y, PANEL_W - 16, 30)
                        if plat_btn_rect.collidepoint(px, py):
                            editor.selected_tile = 'M'
                            editor.plat_placing  = False

                    elif in_viewport:
                        if editor.selected_tile == 'M':
                            # ── Platform placement ────────────────────────
                            # First check if clicking a handle to drag
                            pidx, handle = editor.hit_test_platform_handle(px, py)
                            if pidx is not None:
                                editor.sel_plat_idx = pidx
                                editor.drag_handle  = handle
                                p = editor.level.platforms[pidx]
                                wp = p.start if handle == 'start' else p.end
                                hsx, hsy = editor.world_pixel_to_screen(
                                    wp[0] + p.width / 2, wp[1] + p.height / 2)
                                editor.drag_offset = (hsx - px, hsy - py)
                            elif not editor.plat_placing:
                                # Click-1: set start
                                wx, wy = editor.screen_to_world_pixel(px, py, snap=True)
                                editor.plat_start_world = (wx, wy)
                                editor.plat_ghost_end   = (wx, wy)
                                editor.plat_placing     = True
                                editor.sel_plat_idx     = None
                            else:
                                # Click-2: finalise platform
                                wx, wy = editor.screen_to_world_pixel(px, py, snap=True)
                                editor.push_undo()
                                p = PlatformDef(
                                    list(editor.plat_start_world),
                                    [wx, wy],
                                    speed  = editor.plat_default_spd,
                                    width  = PLAT_DEFAULT_W,
                                    height = PLAT_DEFAULT_H,
                                )
                                editor.level.platforms.append(p)
                                editor.sel_plat_idx = len(editor.level.platforms) - 1
                                editor.plat_placing = False
                                editor.dirty        = True

                        elif editor.hover_cell:
                            r, c = editor.hover_cell
                            if editor.fill_mode:
                                editor.flood_fill(r, c, editor.selected_tile)
                            else:
                                if not painting_started:
                                    editor.push_undo()
                                    painting_started = True
                                editor.paint_char = editor.selected_tile
                                editor.painting   = True
                                editor.last_cell  = None
                                editor.paint_cell(r, c, editor.paint_char)

                elif event.button == 3 and in_viewport:
                    if editor.selected_tile == 'M':
                        # Right-click deselects / cancels placement
                        editor.plat_placing = False
                        editor.sel_plat_idx = None
                    elif editor.hover_cell:
                        if not painting_started:
                            editor.push_undo()
                            painting_started = True
                        r, c              = editor.hover_cell
                        editor.paint_char = ' '
                        editor.painting   = True
                        editor.last_cell  = None
                        editor.paint_cell(r, c, ' ')

                elif event.button == 2:
                    editor.panning   = True
                    editor.pan_start = event.pos
                    editor.pan_cam   = (editor.cam_x, editor.cam_y)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 3):
                    editor.painting  = False
                    editor.last_cell = None
                    painting_started = False
                    editor.drag_handle = None  # release platform handle drag
                elif event.button == 2:
                    editor.panning = False

            elif event.type == pygame.MOUSEMOTION:
                if editor.selected_tile == 'M':
                    mx2, my2 = event.pos
                    if editor.drag_handle is not None and editor.sel_plat_idx is not None:
                        # Drag a handle
                        actual_sx = mx2 + editor.drag_offset[0]
                        actual_sy = my2 + editor.drag_offset[1]
                        # Convert back to world pixel (centre of handle → top-left of body)
                        wx, wy = editor.screen_to_world_pixel(actual_sx, actual_sy, snap=True)
                        p = editor.level.platforms[editor.sel_plat_idx]
                        wp = [wx - p.width / 2, wy - p.height / 2]
                        if editor.drag_handle == 'start':
                            p.start = wp
                        else:
                            p.end = wp
                        editor.dirty = True
                    elif editor.plat_placing and in_viewport:
                        wx, wy = editor.screen_to_world_pixel(mx2, my2, snap=True)
                        editor.plat_ghost_end = (wx, wy)

                if editor.painting and in_viewport and editor.hover_cell:
                    r, c = editor.hover_cell
                    editor.paint_cell(r, c, editor.paint_char)
                if editor.panning:
                    dx = event.pos[0] - editor.pan_start[0]
                    dy = event.pos[1] - editor.pan_start[1]
                    editor.cam_x = editor.pan_cam[0] - dx
                    editor.cam_y = editor.pan_cam[1] - dy

            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                ctrl = mods & pygame.KMOD_CTRL

                if ctrl and event.key == pygame.K_s:
                    path = editor.level.filename or dialog_save()
                    if path:
                        editor.level.save(path)
                        editor.dirty = False
                        print(f"[Editor] Saved: {path}")

                elif ctrl and event.key == pygame.K_o:
                    path = dialog_open()
                    if path:
                        try:
                            editor.level = Level.load(path)
                            editor.undo_stack.clear()
                            editor.redo_stack.clear()
                            editor.dirty = False
                            editor._center_camera()
                        except Exception as e:
                            print(f"[Editor] Load failed: {e}")

                elif ctrl and event.key == pygame.K_z:
                    if mods & pygame.KMOD_SHIFT:
                        editor.redo()
                    else:
                        editor.undo()

                elif ctrl and event.key == pygame.K_n:
                    rows, cols = dialog_new_size()
                    if rows and cols:
                        editor.level = Level(rows, cols)
                        for r in range(rows):
                            editor.level.set(r, 0, '#')
                            editor.level.set(r, cols - 1, '#')
                        for c in range(cols):
                            editor.level.set(0, c, '#')
                            editor.level.set(rows - 1, c, '#')
                        editor.level.set(rows - 2, 2, 'P')
                        editor.undo_stack.clear()
                        editor.redo_stack.clear()
                        editor.dirty = False
                        editor._center_camera()

                elif event.key == pygame.K_HOME:
                    editor.reset_view()

                elif event.key == pygame.K_g:
                    editor.show_grid = not editor.show_grid

                elif event.key == pygame.K_f:
                    editor.fill_mode = not editor.fill_mode

                elif event.key == pygame.K_m:
                    editor.selected_tile = 'M'
                    editor.plat_placing  = False

                elif event.key == pygame.K_DELETE:
                    if editor.selected_tile == 'M' and editor.sel_plat_idx is not None:
                        idx = editor.sel_plat_idx
                        if 0 <= idx < len(editor.level.platforms):
                            editor.push_undo()
                            editor.level.platforms.pop(idx)
                            editor.sel_plat_idx = None
                            editor.dirty = True

                elif event.key == pygame.K_ESCAPE:
                    # Cancel platform placement
                    if editor.plat_placing:
                        editor.plat_placing = False

                # Number row tile shortcuts (1–0)
                num_keys = [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
                            pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8,
                            pygame.K_9, pygame.K_0]
                for i, k in enumerate(num_keys):
                    if event.key == k and i < len(TILES):
                        editor.selected_tile = TILES[i][0]

        # ── Draw ─────────────────────────────────────────────────────────
        screen.fill(UI_BG)
        draw_toolbar(screen, editor, small_font)
        draw_editor(screen, editor, font, small_font)
        draw_palette(screen, editor, font, small_font)
        draw_status(screen, editor, small_font)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()