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

DEFAULT_ROWS = 100
DEFAULT_COLS = 100

# ─────────────────────────────────────────────────────────────────────────────
# Game object imports (optional — falls back to colored rects)
# ─────────────────────────────────────────────────────────────────────────────
# Inline rendering — no external game object imports needed
USE_REAL_OBJECTS = True

# ─────────────────────────────────────────────────────────────────────────────
# Tile definitions
# ─────────────────────────────────────────────────────────────────────────────
TILES = [
    (' ',  'Air',            ( 30,  35,  50), (150, 160, 180)),
    ('#',  'Ground',         (139,  69,  19), (200, 170, 140)),
    ('=',  'Platform',       (205, 133,  63), (255, 200, 130)),
    ('^',  'Spike',          ( 50,  50,  50), (180, 180, 180)),
    ('?',  'QBlock (coin)',  (255, 165,   0), (255, 255, 255)),
    ('>',  'QBlock (star)',  (255, 165,   0), (255, 255, 255)),
    ('<',  'QBlock (mush)',  (255, 165,   0), (255, 255, 255)),
    ('C',  'Coin',           (255, 215,   0), (  0,   0,   0)),
    ('E',  'Enemy',          (139,  69,  19), (255, 255, 255)),
    ('G',  'Goal',           (255, 215,   0), (200, 160,   0)),
    ('P',  'Player Start',   ( 60, 160, 255), (255, 255, 255)),
]

TILE_BY_CHAR = {t[0]: t for t in TILES}
SOLID_CHARS  = {'#', '=', '?', '>', '<'}

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
    Render each tile using the same draw calls as the real game objects,
    scaled to the current tile size ts (handles zoom).
    Mirrors exact rendering logic in Tile, Enemy, Coin, Goal, QuestionBlock.
    """
    ti = int(ts)

    if char == ' ':
        return  # air - transparent, editor background shows through

    elif char == '#':
        # Tile.render with COLOR_GROUND = (139, 69, 19)
        pygame.draw.rect(surf, (139, 69, 19), (sx, sy, ti, ti))

    elif char == '=':
        # Tile.render with COLOR_PLATFORM = (205, 133, 63)
        pygame.draw.rect(surf, (205, 133, 63), (sx, sy, ti, ti))
        hi_h = max(2, ti // 8)
        pygame.draw.rect(surf, (230, 165, 90), (sx, sy, ti, hi_h))

    elif char == '^':
        # Tile.render with COLOR_SPIKE = (50, 50, 50) + triangle overlay
        pygame.draw.rect(surf, (50, 50, 50), (sx, sy, ti, ti))
        points = [(sx + ti // 2, sy + 1), (sx + 1, sy + ti - 1), (sx + ti - 1, sy + ti - 1)]
        pygame.draw.polygon(surf, (120, 120, 120), points)

    elif char in ('?', '>', '<'):
        # QuestionBlock.render - orange rect + black border + symbol
        pygame.draw.rect(surf, (255, 165, 0), (sx, sy, ti, ti))
        pygame.draw.rect(surf, (0, 0, 0), (sx, sy, ti, ti), max(1, ti // 20))
        label = {'?': '?', '>': '\u2605', '<': '\u2665'}[char]
        fsize = max(8, int(ti * 0.65))
        try:
            _f = pygame.font.Font(None, fsize)
            txt = _f.render(label, True, (255, 255, 255))
            surf.blit(txt, txt.get_rect(center=(sx + ti // 2, sy + ti // 2)))
        except Exception:
            pass

    elif char == 'C':
        # Coin.render - gold circle + black outline
        r  = max(3, ti // 2 - max(1, ti // 10))
        cx = sx + ti // 2
        cy = sy + ti // 2
        pygame.draw.circle(surf, (255, 215, 0), (cx, cy), r)
        pygame.draw.circle(surf, (0, 0, 0), (cx, cy), r, max(1, ti // 16))

    elif char == 'E':
        # Enemy.render - brown rect + white eye (facing left)
        pygame.draw.rect(surf, (139, 69, 19), (sx, sy, ti, ti))
        eye_w = max(2, ti // 8)
        eye_h = max(3, ti // 4)
        eye_x = sx + ti // 6
        eye_y = sy + ti // 5
        pygame.draw.rect(surf, (255, 255, 255), (eye_x, eye_y, eye_w, eye_h))

    elif char == 'G':
        # Goal - gold rect + pole + flag
        pygame.draw.rect(surf, (255, 215, 0), (sx, sy, ti, ti))
        pole_w = max(2, ti // 10)
        pygame.draw.rect(surf, (200, 160, 0), (sx + ti // 3, sy, pole_w, ti))
        flag_h = max(4, ti // 3)
        pygame.draw.polygon(surf, (255, 80, 80), [
            (sx + ti // 3 + pole_w, sy + 2),
            (sx + ti * 3 // 4,      sy + flag_h // 2),
            (sx + ti // 3 + pole_w, sy + flag_h),
        ])

    elif char == 'P':
        # Player Start - blue rect + white dot
        pygame.draw.rect(surf, (60, 160, 255), (sx, sy, ti, ti))
        pygame.draw.circle(
            surf, (255, 255, 255),
            (sx + ti * 3 // 4, sy + ti // 3),
            max(2, ti // 8)
        )

    else:
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
        self.rows     = rows
        self.cols     = cols
        self.grid     = [[' '] * cols for _ in range(rows)]
        self.filename = None

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
        with open(path, 'w') as f:
            f.write(self.to_ascii(trim=trim))
        self.filename = path

    def clone(self):
        new          = Level(self.rows, self.cols)
        new.grid     = [row[:] for row in self.grid]
        new.filename = self.filename
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


# ─────────────────────────────────────────────────────────────────────────────
# Draw functions
# ─────────────────────────────────────────────────────────────────────────────
def draw_editor(surf, editor: Editor, font, small_font):
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
             "Ctrl+Z Undo  |  Home Reset View  |  Scroll Zoom  |  MMB Pan")
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
    tile_label = TILE_BY_CHAR.get(editor.selected_tile, TILES[0])[1]

    info = (f"  {level.rows}r x {level.cols}c"
            f"  |  Zoom {editor.zoom:.2f}x"
            f"  |  Brush: {tile_label}")

    if editor.hover_cell:
        hr, hc   = editor.hover_cell
        char     = level.get(hr, hc)
        ch_label = TILE_BY_CHAR.get(char, TILES[0])[1] if char else 'Air'
        info    += f"  |  [{hr}, {hc}]: {ch_label}"

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
                        tile_h = 36; pad = 8
                        ty0    = TOOLBAR_H + 12 + 20
                        idx    = (py - ty0) // (tile_h + 4)
                        if 0 <= idx < len(TILES):
                            editor.selected_tile = TILES[idx][0]
                    elif in_viewport and editor.hover_cell:
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

                elif event.button == 3 and in_viewport and editor.hover_cell:
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
                elif event.button == 2:
                    editor.panning = False

            elif event.type == pygame.MOUSEMOTION:
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