#!/usr/bin/env python3
"""
PEAK Level Editor  v2.1
───────────────────────
Standalone pygame level editor for PEAK platformer levels.

Controls:
  Left click            — paint selected tile / erase (eraser mode)
  Right click           — erase (set to air)
  Middle drag           — pan camera
  Scroll wheel          — zoom in/out
  Ctrl+S / Ctrl+O       — save / open
  Ctrl+N                — new level
  Ctrl+Z / Ctrl+Shift+Z — undo / redo
  Ctrl+X                — clear all
  G / F / X             — grid / fill / eraser toggle
  Home                  — reset view
  1–0                   — select tile by number
"""

import os, sys, math, copy, re, subprocess
import pygame, pygame.freetype
from pathlib import Path
from tkinter import filedialog, simpledialog
import tkinter as tk
try:
    import yaml
except ImportError:
    yaml = None

# ─── Constants ────────────────────────────────────────────────────
WINDOW_W, WINDOW_H = 1500, 900
TILE_SIZE = 32
MIN_ZOOM = 0.25; MAX_ZOOM = 4.0
UNDO_LIMIT = 200
LEFT_PANEL_W = 210; RIGHT_PANEL_W = 270
TOOLBAR_H = 54; STATUS_H = 28; HSB_H = 12  # HSB_H = viewport horizontal scrollbar height
DEFAULT_ROWS = 34; DEFAULT_COLS = 75
SB_W = 10  # right-panel vertical scrollbar width

# ─── Game object imports (optional) ──────────────────────────────
try:
    from code.games.modules.Objects.Tile import Tile, create_tile
    from code.games.modules.Objects.Coin import Coin
    from code.games.modules.Objects.Enemy import Enemy
    from code.games.modules.Objects.Goal import Goal
    from code.games.modules.Objects.QuestionBlock import QuestionBlock
    from code.games.modules.Objects.GameObject import GameObject
    from code.games.modules.Parameters.Map_parameters import (
        TILE_GROUND, TILE_PLATFORM, TILE_SPIKE,
        COLOR_GROUND, COLOR_PLATFORM, COLOR_SPIKE, COLOR_GOAL)
    USE_REAL_OBJECTS = True
except ImportError:
    USE_REAL_OBJECTS = False

# ─── Tile definitions ────────────────────────────────────────────
TILES = [
    (' ', 'Air',           (30,35,50),   (80,90,120)),
    ('#', 'Ground',        (80,60,40),   (220,200,160)),
    ('=', 'Platform',      (60,90,60),   (180,230,140)),
    ('^', 'Spike',         (200,60,60),  (255,200,180)),
    ('?', 'QBlock (coin)', (220,180,40), (80,60,0)),
    ('>', 'QBlock (star)', (255,220,80), (80,60,0)),
    ('<', 'QBlock (mush)', (200,80,80),  (255,240,240)),
    ('C', 'Coin',          (255,215,0),  (80,60,0)),
    ('E', 'Enemy',         (180,60,180), (255,220,255)),
    ('G', 'Goal',          (60,200,100), (0,60,20)),
    ('P', 'Player Start',  (60,160,255), (0,30,120)),
]
TILE_BY_CHAR = {t[0]: t for t in TILES}
SOLID_CHARS = {'#','=','?','>','<'}

# ─── Moving Platform ─────────────────────────────────────────────
PLAT_DEFAULT_W=TILE_SIZE*3; PLAT_DEFAULT_H=TILE_SIZE//2; PLAT_DEFAULT_SPD=80.0
PLAT_BODY_COL=(205,133,63); PLAT_HIGH_COL=(230,165,90)
PLAT_PATH_COL=(255,200,80); PLAT_SEL_COL=(255,255,100)
HANDLE_START=(80,220,80); HANDLE_END=(220,80,80)
HANDLE_R=9; HANDLE_HIT_R=14

class PlatformDef:
    __slots__=('start','end','speed','width','height')
    def __init__(s,start,end,speed=PLAT_DEFAULT_SPD,width=PLAT_DEFAULT_W,height=PLAT_DEFAULT_H):
        s.start=list(start); s.end=list(end); s.speed=float(speed)
        s.width=int(width); s.height=int(height)
    def to_dict(s):
        return {'start':[int(s.start[0]),int(s.start[1])],
                'end':[int(s.end[0]),int(s.end[1])],
                'speed':s.speed,'width':s.width,'height':s.height}
    @classmethod
    def from_dict(cls,d):
        return cls(d['start'],d['end'],d.get('speed',PLAT_DEFAULT_SPD),
                   d.get('width',PLAT_DEFAULT_W),d.get('height',PLAT_DEFAULT_H))

# ─── UI Colors (PEAK logo palette) ──────────────────────────────
# Dark maroon background, crimson + peach + pale-blue accents,
# neon-pink / cyan highlights — mirrors the triangular logo.
UI_BG       = (18, 8, 12)          # very dark maroon canvas
UI_PANEL    = (26, 12, 18)         # left/right panel background
UI_PANEL2   = (34, 16, 24)         # inset / nested panel
UI_BORDER   = (72, 30, 44)         # panel dividers
UI_SELECT   = (110, 20, 35)        # selected item (crimson)
UI_SELECT_DIM=(65, 12, 22)         # dimmed selection
UI_TEXT     = (232, 210, 195)      # warm off-white (peach tint)
UI_SUBTEXT  = (130, 88, 100)       # muted rose-grey
UI_TOOLBAR  = (14,  6, 10)         # toolbar strip
UI_STATUS   = (12,  5,  8)         # status bar
UI_BTN      = (40, 18, 26)         # normal button
UI_BTN_HOVER= (58, 26, 38)         # hovered button
UI_BTN_ACT  = (100, 22, 36)        # active / pressed (deep crimson)
UI_ACCENT   = (255, 90, 140)       # neon pink  (logo outline)
UI_ACCENT2  = (90, 210, 220)       # neon cyan  (logo outline)
UI_WARN     = (210, 145, 65)       # amber warning
UI_DANGER   = (210, 48, 58)        # bright red
GRID_COLOR  = (36, 16, 24)         # dark maroon grid lines
GRID_BOLD   = (60, 26, 38)         # bold grid lines

# ─── GameConfig — reads game_config.yaml ─────────────────────────
class GameConfig:
    """Reads game_config.yaml for level browser + physics defaults.
    Also supports toggling levels on/off and writing back."""

    def __init__(self):
        self.yaml_data={}; self.levels={}; self.level_ids=[]
        self.physics={}; self.levels_dir=None; self.config_path=None
        self._load()

    def _load(self):
        if yaml is None: return
        # Search for game_config.yaml in many locations
        cwd = Path.cwd()
        candidates = [
            cwd / "game_config.yaml",
            cwd / "code" / "games" / "platformer" / "game_config.yaml",
            cwd / "code" / "games" / "game_config.yaml",
        ]
        # Also walk up from __file__ if we're inside the project
        script_dir = Path(__file__).resolve().parent
        candidates += [
            script_dir / "game_config.yaml",
            script_dir.parent / "game_config.yaml",
            script_dir.parent.parent / "game_config.yaml",
        ]
        for p in candidates:
            if p.exists():
                self.config_path = p; break
        if self.config_path is None: return

        try:
            with open(self.config_path,'r') as f:
                self.yaml_data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[GameConfig] {e}"); return

        # Find levels directory
        cfg_dir = self.config_path.parent
        level_dir_candidates = [
            cfg_dir / "levels",
            cfg_dir / "code" / "games" / "platformer" / "levels",
            cwd / "code" / "games" / "platformer" / "levels",
            cwd / "levels",
            cfg_dir,  # .txt files might be alongside game_config.yaml
        ]
        for p in level_dir_candidates:
            if p.exists() and p.is_dir():
                # Check if it actually has .txt files
                if list(p.glob("*.txt")):
                    self.levels_dir = p; break

        # Parse levels
        raw = self.yaml_data.get('levels',{}) or {}
        for lid, lcfg in raw.items():
            if isinstance(lcfg, dict):
                self.levels[str(lid)] = lcfg
                self.level_ids.append(str(lid))

        # Physics defaults
        defaults = self.yaml_data.get('defaults',{}) or {}
        self.physics = copy.deepcopy(defaults.get('physics',{}) or {})

    def get_level_file_path(self, level_id):
        lcfg = self.levels.get(level_id)
        if not lcfg: return None
        fn = lcfg.get('file','')
        if not fn: return None
        # Try levels_dir
        if self.levels_dir:
            p = self.levels_dir / fn
            if p.exists(): return str(p)
        # Try alongside config
        if self.config_path:
            p = self.config_path.parent / fn
            if p.exists(): return str(p)
        # Try cwd
        if Path(fn).exists(): return str(Path(fn).resolve())
        return None

    def get_all_level_files(self, extra_path=None):
        """Return all .txt files from levels dir + config dir + any extra path."""
        files = set()
        # Resolve all dirs to avoid symlink / cwd mismatch false-negatives
        seen_dirs = set()
        dirs_to_check = []

        def _add_dir(p):
            try:
                rp = Path(p).resolve()
            except Exception:
                rp = Path(p)
            if rp not in seen_dirs and rp.exists() and rp.is_dir():
                seen_dirs.add(rp)
                dirs_to_check.append(rp)

        if self.levels_dir:
            _add_dir(self.levels_dir)
        if self.config_path:
            _add_dir(self.config_path.parent)
        # Always include the directory of any currently-loaded or recently-saved file
        if extra_path:
            _add_dir(Path(extra_path).parent)
        # Also refresh levels_dir if it was never found (e.g. project had no .txt at startup)
        if self.levels_dir is None and extra_path:
            self.levels_dir = Path(extra_path).parent

        for d in dirs_to_check:
            for f in d.glob("*.txt"):
                files.add(f)
        return sorted(files, key=lambda p: p.name)

    def get_commented_levels(self):
        """Parse game_config.yaml raw text to find commented-out level entries."""
        if not self.config_path: return {}
        try:
            text = self.config_path.read_text()
        except: return {}
        # Match lines like:  # "1-6":
        commented = {}
        pattern = re.compile(r'^\s*#\s*"(\d+-\d+)":\s*$', re.MULTILINE)
        for m in pattern.finditer(text):
            lid = m.group(1)
            if lid not in self.levels:
                commented[lid] = True
        return commented

    def _get_commented_file(self, level_id):
        """Return the stored filename from a commented-out level block (if present)."""
        if not self.config_path: return None
        try:
            text = self.config_path.read_text(encoding='utf-8')
        except: return None
        # Find the commented key, then look for the file: line within the block
        pat = re.compile(
            r'#\s*"' + re.escape(level_id) + r'":\s*\n((?:\s*#[^\n]*\n)*)',
            re.MULTILINE)
        m = pat.search(text)
        if not m: return None
        block = m.group(1)
        fm = re.search(r'#\s*file:\s*"([^"]+)"', block)
        return fm.group(1) if fm else None

    def toggle_level_in_config(self, level_id, enable):
        """Enable or disable a level in game_config.yaml by commenting/uncommenting."""
        if not self.config_path or yaml is None: return False
        try:
            text = self.config_path.read_text()
        except: return False

        lines = text.split('\n')
        new_lines = []
        in_target_block = False
        block_indent = 0

        if enable:
            # Uncomment: remove leading '# ' from the level block
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith(f'# "{level_id}":') or stripped.startswith(f'#  "{level_id}":'):
                    in_target_block = True
                    block_indent = len(line) - len(stripped)
                    # Uncomment this line
                    new_lines.append(line.replace('# ', '', 1) if '# ' in line else line.replace('#', '', 1))
                    continue
                if in_target_block:
                    if stripped.startswith('#') and (stripped[1:].startswith('  ') or stripped[1:].startswith(' ')):
                        new_lines.append(line.replace('# ', '', 1) if '# ' in line else line.replace('#', '', 1))
                    else:
                        in_target_block = False
                        new_lines.append(line)
                else:
                    new_lines.append(line)
        else:
            # Comment out: add '# ' prefix to the level block
            for line in lines:
                stripped = line.lstrip()
                indent = len(line) - len(stripped)
                if stripped.startswith(f'"{level_id}":'):
                    in_target_block = True
                    block_indent = indent
                    new_lines.append(' ' * indent + '# ' + stripped)
                    continue
                if in_target_block:
                    if stripped and not stripped.startswith('"') and indent > block_indent:
                        new_lines.append(' ' * indent + '# ' + stripped)
                    else:
                        in_target_block = False
                        new_lines.append(line)
                else:
                    new_lines.append(line)

        try:
            self.config_path.write_text('\n'.join(new_lines))
            # Reload
            self.yaml_data={}; self.levels={}; self.level_ids=[]
            self._load()
            return True
        except Exception as e:
            print(f"[GameConfig] Write failed: {e}"); return False

    def assign_stage_to_level(self, filename, level_id):
        """Append (or update) a level entry in game_config.yaml using text manipulation."""
        if not self.config_path: return False
        try:
            text = self.config_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"[GameConfig] Read failed: {e}"); return False

        # If the level_id already exists, just update its file line
        existing_pat = re.compile(
            r'([ \t]*"' + re.escape(level_id) + r'":\s*\n[ \t]+file:\s*)".+?"',
            re.MULTILINE)
        if existing_pat.search(text):
            text = existing_pat.sub(
                lambda m: m.group(1) + f'"{filename}"', text)
        else:
            # Find the end of the levels: block and append
            levels_header = re.search(r'^levels:\s*$', text, re.MULTILINE)
            if not levels_header:
                print("[GameConfig] No 'levels:' section found"); return False
            new_entry = (
                f'\n  "{level_id}":\n'
                f'    file: "{filename}"\n'
                f'    time_limit: 300\n'
                f'    background_color: [0, 0, 0]\n'
            )
            text = text.rstrip() + new_entry

        try:
            self.config_path.write_text(text, encoding='utf-8')
            self.yaml_data={}; self.levels={}; self.level_ids=[]
            self._load()
            return True
        except Exception as e:
            print(f"[GameConfig] Write failed: {e}"); return False

    def rename_level_id(self, old_id, new_id):
        """Rename a level ID in game_config.yaml using text manipulation."""
        if not self.config_path: return False
        try:
            text = self.config_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"[GameConfig] Read failed: {e}"); return False

        pat = re.compile(r'(?m)^(\s*)"' + re.escape(old_id) + r'"(\s*:)')
        if not pat.search(text):
            print(f"[GameConfig] Level '{old_id}' not found in config"); return False
        text = pat.sub(lambda m: f'{m.group(1)}"{new_id}"{m.group(2)}', text)

        try:
            self.config_path.write_text(text, encoding='utf-8')
            self.yaml_data={}; self.levels={}; self.level_ids=[]
            self._load()
            return True
        except Exception as e:
            print(f"[GameConfig] Write failed: {e}"); return False

    def delete_level_in_config(self, level_id):
        """Remove a level entry (enabled or disabled/commented) from game_config.yaml entirely."""
        if not self.config_path: return False
        try:
            text = self.config_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"[GameConfig] Read failed: {e}"); return False

        lines = text.split('\n')
        new_lines = []
        skip = False
        block_indent = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # Match enabled:  "1-1":
            # or disabled: # "1-1":  (with possible extra spaces)
            is_key = (stripped == f'"{level_id}":' or
                      re.match(r'^#\s*"' + re.escape(level_id) + r'":', stripped))
            if is_key:
                skip = True
                block_indent = indent
                i += 1
                continue

            if skip:
                # Keep skipping lines that are part of this block:
                # — blank lines are kept as single separator (stop skipping after one)
                if stripped == '' or stripped == '#':
                    skip = False  # end of block; don't include this blank
                    new_lines.append(line)
                    i += 1
                    continue
                # Indented content (including commented indented lines) = still in block
                real_stripped = stripped.lstrip('#').lstrip()
                if indent > block_indent or (stripped.startswith('#') and indent >= block_indent):
                    i += 1  # skip this line
                    continue
                else:
                    skip = False  # new key at same/lesser indent — stop skipping

            new_lines.append(line)
            i += 1

        try:
            self.config_path.write_text('\n'.join(new_lines), encoding='utf-8')
            self.yaml_data={}; self.levels={}; self.level_ids=[]
            self._load()
            return True
        except Exception as e:
            print(f"[GameConfig] Delete failed: {e}"); return False

# ─── Renderers ───────────────────────────────────────────────────
def render_tile_object(surf, char, sx, sy, ts):
    ti=int(ts)
    if char=='#':
        t=create_tile(TILE_GROUND,0,0,True,COLOR_GROUND)
        t.gObj.width=t.gObj.height=ti; t.gObj.x=sx; t.gObj.y=sy; t.render(surf,0,0)
    elif char=='=':
        t=create_tile(TILE_PLATFORM,0,0,True,COLOR_PLATFORM)
        t.gObj.width=t.gObj.height=ti; t.gObj.x=sx; t.gObj.y=sy; t.render(surf,0,0)
    elif char=='^':
        t=create_tile(TILE_SPIKE,0,0,False,COLOR_SPIKE)
        t.gObj.width=t.gObj.height=ti; t.gObj.x=sx; t.gObj.y=sy; t.render(surf,0,0)
    elif char=='G':
        Goal(gObj=GameObject(float(sx),float(sy),ti,ti)).render(surf,sx,sy)
    elif char=='C':
        half=ti//2; c=Coin(gObj=GameObject(float(sx+half),float(sy+half),ti,ti))
        c.radius=max(4,half-2); c.render(surf,sx+half,sy+half)
    elif char=='E':
        Enemy(gObj=GameObject(float(sx),float(sy),ti,ti)).render(surf,sx,sy)
    elif char in ('?','>','<'):
        cn={'?':'coin','>':'star','<':'mushroom'}[char]
        QuestionBlock(gObj=GameObject(float(sx),float(sy),ti,ti),contains=cn).render(surf,sx,sy)
    elif char=='P':
        pygame.draw.rect(surf,(60,160,255),(sx,sy,ti,ti))
        pygame.draw.circle(surf,(255,255,255),(sx+ti*3//4,sy+ti//3),max(2,ti//8))
    else:
        pygame.draw.rect(surf,UI_BG,(sx,sy,ti,ti))

def draw_tile_rect(surf, char, x, y, w, h, alpha=255):
    tile=TILE_BY_CHAR.get(char,TILE_BY_CHAR[' ']); color=tile[2]
    s=pygame.Surface((w,h),pygame.SRCALPHA); s.fill((*color,alpha))
    if char in SOLID_CHARS and w>4:
        hi=tuple(min(255,c+40) for c in color)
        pygame.draw.line(s,(*hi,alpha),(0,0),(w-1,0),max(1,h//12))
    surf.blit(s,(x,y))

def draw_tile_symbol(surf, font, char, x, y, w, h):
    if char==' ': return
    tile=TILE_BY_CHAR.get(char,TILE_BY_CHAR[' '])
    disp={'#':'█','=':'▬','^':'▲','?':'?','>':'★','<':'♦','C':'●','E':'☻','G':'⚑','P':'▶'}.get(char,char)
    sz=max(8,min(w-4,h-4,20))
    try:
        b=font.get_rect(disp,size=sz)
        font.render_to(surf,(x+(w-b.width)//2,y+(h-b.height)//2),disp,fgcolor=tile[3],size=sz)
    except: pass

# ─── Palette icons ───────────────────────────────────────────────
def draw_palette_icon(surf, char, rect):
    x,y,w,h = rect.x,rect.y,rect.width,rect.height
    cx,cy = x+w//2, y+h//2
    col  = TILE_BY_CHAR.get(char,TILE_BY_CHAR[' '])[2]
    hi   = tuple(min(255,c+70) for c in col)
    dk   = tuple(max(0,c-40)   for c in col)

    if char==' ':
        # Dotted circle = empty / air
        for angle in range(0,360,45):
            import math as _m
            ax=cx+int(4*_m.cos(_m.radians(angle))); ay=cy+int(4*_m.sin(_m.radians(angle)))
            pygame.draw.circle(surf,(55,30,40),(ax,ay),1)

    elif char=='#':
        # Solid brick block
        pygame.draw.rect(surf,col,(x+2,y+2,w-4,h-4),border_radius=2)
        pygame.draw.rect(surf,hi,(x+2,y+2,w-4,h-4),1,border_radius=2)
        # Brick mortar lines
        midy=y+h//2
        pygame.draw.line(surf,dk,(x+3,midy),(x+w-4,midy),1)
        pygame.draw.line(surf,dk,(cx,y+3),(cx,midy-1),1)
        pygame.draw.line(surf,dk,(x+5,midy+1),(x+5,y+h-4),1)
        pygame.draw.line(surf,dk,(x+w-5,midy+1),(x+w-5,y+h-4),1)

    elif char=='=':
        # Platform — wide flat bar with rounded ends
        bh=max(5,h//3); by2=cy-bh//2
        pygame.draw.rect(surf,col,(x+1,by2,w-2,bh),border_radius=3)
        pygame.draw.rect(surf,hi,(x+1,by2,w-2,bh),1,border_radius=3)
        pygame.draw.line(surf,hi,(x+3,by2+1),(x+w-4,by2+1),1)

    elif char=='^':
        # Spike triangle — bright red tip
        pts=[(cx,y+2),(x+2,y+h-2),(x+w-2,y+h-2)]
        pygame.draw.polygon(surf,col,pts)
        pygame.draw.polygon(surf,(255,100,100),pts,1)
        pygame.draw.line(surf,(255,200,200),(cx,y+3),(cx+2,y+h//2),1)

    elif char in ('?','>','<'):
        # Question block — gold square with symbol
        qcol={'?':(210,170,30),'>': (240,200,50),'<':(195,80,70)}[char]
        pygame.draw.rect(surf,qcol,(x+2,y+2,w-4,h-4),border_radius=3)
        pygame.draw.rect(surf,tuple(min(255,c+60) for c in qcol),(x+2,y+2,w-4,h-4),1,border_radius=3)
        sym={'?':'?','>':'★','<':'♦'}[char]
        try:
            import pygame.freetype as _ft
            # Use a simple draw rather than font since sfont not available here
            pass
        except: pass
        # Draw the symbol via pixel art
        if char=='?':
            # Simple ? mark
            pygame.draw.circle(surf,(40,30,0),(cx,cy-1),max(2,w//6))
            pygame.draw.line(surf,(40,30,0),(cx,cy+1),(cx,cy+2),max(1,h//8))
        elif char=='>':
            pts2=[(cx,y+4),(x+w-3,cy),(cx,y+h-4)]
            pygame.draw.polygon(surf,(40,30,0),pts2)
        elif char=='<':
            pygame.draw.rect(surf,(40,30,0),(cx-2,cy-2,4,4),border_radius=1)

    elif char=='C':
        # Coin — gold circle with shine
        r2=min(w,h)//2-2
        pygame.draw.circle(surf,(215,175,0),(cx,cy),r2)
        pygame.draw.circle(surf,(255,240,100),(cx,cy),r2,1)
        pygame.draw.circle(surf,(255,250,160),(cx-r2//3,cy-r2//3),max(1,r2//3))

    elif char=='E':
        # Enemy — magenta circle with white eyes
        r2=min(w,h)//2-2
        pygame.draw.circle(surf,col,(cx,cy),r2)
        pygame.draw.circle(surf,tuple(min(255,c+60) for c in col),(cx,cy),r2,1)
        ey=cy-r2//4
        for ox in (-r2//3,r2//3):
            pygame.draw.circle(surf,(255,255,255),(cx+ox,ey),max(1,r2//3))
            pygame.draw.circle(surf,(20,0,20),(cx+ox,ey),max(1,r2//5))

    elif char=='G':
        # Goal — green flag on pole
        pole_x=cx-1
        pygame.draw.line(surf,(200,200,200),(pole_x,y+2),(pole_x,y+h-2),2)
        pts3=[(pole_x+1,y+3),(x+w-2,y+3+(h//4)),(pole_x+1,y+3+(h//2))]
        pygame.draw.polygon(surf,col,pts3)
        pygame.draw.polygon(surf,hi,pts3,1)

    elif char=='P':
        # Player — blue body + head
        head_r=max(3,h//5)
        pygame.draw.circle(surf,(80,180,255),(cx,y+h//3),head_r)
        pygame.draw.rect(surf,(50,140,220),(cx-w//5,y+h//3+head_r-1,w*2//5,h//3),border_radius=2)
        # Eyes
        pygame.draw.circle(surf,(255,255,255),(cx+2,y+h//3-1),max(1,head_r//3))

# ─── Level ───────────────────────────────────────────────────────
class Level:
    def __init__(self, rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
        self.rows=rows; self.cols=cols
        self.grid=[[' ']*cols for _ in range(rows)]
        self.platforms=[]; self.filename=None; self.level_id=None

    def get(self,r,c):
        return self.grid[r][c] if 0<=r<self.rows and 0<=c<self.cols else None
    def set(self,r,c,char):
        if r<0 or c<0: return
        while r>=self.rows: self.grid.append([' ']*self.cols); self.rows+=1
        if c>=self.cols:
            for row in self.grid: row.extend([' ']*(c-self.cols+1))
            self.cols=c+1
        self.grid[r][c]=char
    def clear_all(self):
        for r in range(self.rows):
            for c in range(self.cols): self.grid[r][c]=' '
    def bounding_box(self):
        mr=self.rows; xr=-1; mc=self.cols; xc=-1
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c]!=' ':
                    mr=min(mr,r); xr=max(xr,r); mc=min(mc,c); xc=max(xc,c)
        return (mr,mc,xr,xc) if xr!=-1 else None
    def to_ascii(self, trim=True, padding=1):
        if not trim: return '\n'.join(''.join(r) for r in self.grid)
        bb=self.bounding_box()
        if bb is None: return '\n'.join([' '*DEFAULT_COLS]*DEFAULT_ROWS)
        r0=max(0,bb[0]-padding); c0=max(0,bb[1]-padding)
        r1=min(self.rows-1,bb[2]+padding); c1=min(self.cols-1,bb[3]+padding)
        w=c1-c0+1; lines=[]
        for r in range(r0,r1+1):
            rc=self.grid[r][c0:c1+1]
            while len(rc)<w: rc.append(' ')
            lines.append(''.join(rc))
        return '\n'.join(lines)
    def save(self, path, trim=True):
        with open(path,'w') as f: f.write(self.to_ascii(trim=trim))
        self.filename=str(path); self._save_yaml(path)
    def _save_yaml(self, txt_path):
        if yaml is None: return
        yp=str(txt_path).rsplit('.',1)[0]+'.yaml'
        d={'dynamics':{'moving_platforms':[p.to_dict() for p in self.platforms]} if self.platforms else {}}
        with open(yp,'w') as f: yaml.dump(d,f,default_flow_style=False,sort_keys=False)
    def _load_yaml(self, txt_path):
        if yaml is None: return
        yp=str(txt_path).rsplit('.',1)[0]+'.yaml'
        if not Path(yp).exists(): return
        try:
            with open(yp,'r') as f: data=yaml.safe_load(f) or {}
            for pd in (data.get('dynamics',{}) or {}).get('moving_platforms',[]):
                self.platforms.append(PlatformDef.from_dict(pd))
        except Exception as e: print(f"[Editor] YAML sidecar error: {e}")
    def clone(self):
        n=Level(self.rows,self.cols); n.grid=[r[:] for r in self.grid]
        n.filename=self.filename; n.level_id=self.level_id
        n.platforms=[PlatformDef(p.start[:],p.end[:],p.speed,p.width,p.height) for p in self.platforms]
        return n
    @classmethod
    def from_ascii(cls, text):
        lines=text.split('\n'); rows=len(lines)
        cols=max((len(l) for l in lines),default=DEFAULT_COLS)
        lv=cls(rows,cols)
        for r,line in enumerate(lines):
            for c,ch in enumerate(line): lv.grid[r][c]=ch
        return lv
    @classmethod
    def load(cls, path):
        with open(path,'r') as f: text=f.read()
        lv=cls.from_ascii(text); lv.filename=str(path); lv._load_yaml(path); return lv

# ─── Physics Slider / Toolbar Button ────────────────────────────
class PhysicsSlider:
    __slots__=('label','key','value','min_val','max_val','step','rect','dragging')
    def __init__(s,label,key,value,mn,mx,step=10.0):
        s.label=label; s.key=key; s.value=float(value)
        s.min_val=float(mn); s.max_val=float(mx); s.step=float(step)
        s.rect=pygame.Rect(0,0,0,0); s.dragging=False
    def fraction(s):
        rng=s.max_val-s.min_val; return (s.value-s.min_val)/rng if rng>0 else 0.0
    def set_from_fraction(s,frac):
        frac=max(0.0,min(1.0,frac)); rng=s.max_val-s.min_val
        s.value=round((s.min_val+frac*rng)/s.step)*s.step
        s.value=max(s.min_val,min(s.max_val,s.value))

class ToolbarButton:
    __slots__=('label','shortcut','action','rect','toggle','get_active')
    def __init__(s,label,shortcut,action,toggle=False,get_active=None):
        s.label=label; s.shortcut=shortcut; s.action=action
        s.rect=pygame.Rect(0,0,0,0); s.toggle=toggle; s.get_active=get_active

# ─── Editor ──────────────────────────────────────────────────────
class Editor:
    def __init__(self, level, game_config):
        self.level=level; self.game_config=game_config
        self.undo_stack=[]; self.redo_stack=[]
        self.selected_tile='#'; self.show_grid=True
        self.fill_mode=False; self.eraser_mode=False
        self.cam_x=0.0; self.cam_y=0.0; self.zoom=1.0
        self.painting=False; self.paint_char=' '; self.panning=False
        self.pan_start=(0,0); self.pan_cam=(0.0,0.0)
        self.last_cell=None; self.hover_cell=None; self.dirty=False
        # Platform tool
        self.plat_placing=False; self.plat_start_world=None
        self.plat_ghost_end=None; self.plat_default_spd=PLAT_DEFAULT_SPD
        self.sel_plat_idx=None; self.drag_handle=None; self.drag_offset=(0.0,0.0)
        # Right panel vertical scrollbar
        self.right_scroll=0; self.right_max_scroll=0
        self.browser_level_id=None
        self._scrollbar_thumb=pygame.Rect(0,0,0,0)
        self._scrollbar_dragging=False
        self._scrollbar_drag_start_y=0
        self._scrollbar_drag_scroll_start=0
        # Viewport horizontal scrollbar
        self._hscroll_thumb=pygame.Rect(0,0,0,0)
        self._hscroll_dragging=False
        self._hscroll_drag_start_x=0
        self._hscroll_drag_cam_start=0.0
        # Hit-test rects stored each frame by draw functions
        self._spd_minus_r=pygame.Rect(0,0,0,0); self._spd_plus_r=pygame.Rect(0,0,0,0)
        self._tile_rects={}      # char -> Rect
        self._tool_rects={}      # 'fill'|'grid'|'eraser'|'platform' -> Rect
        self._level_btn_rects={} # level_id -> Rect
        self._unlisted_btn_rects={} # filepath -> Rect
        self._unlisted_del_rects={} # filepath -> Rect (delete file from disk)
        self._toggle_btn_rects={}   # level_id -> Rect (enable/disable checkbox)
        self._rename_btn_rects={}   # level_id -> Rect (rename level ID)
        self._delete_btn_rects={}   # level_id -> Rect (delete level from config)
        self._assign_btn_rect=pygame.Rect(0,0,0,0)  # assign unlisted -> level ID
        # Physics sliders
        ph=game_config.physics or {}; fric=ph.get('friction',{}) or {}
        self.physics_sliders=[
            PhysicsSlider("Gravity","gravity",ph.get('gravity',1300),200,3000,50),
            PhysicsSlider("Fast Fall","fast_fall",ph.get('fast_fall_gravity',2500),500,5000,50),
            PhysicsSlider("Ground Fric","ground_fric",fric.get('ground',1300),200,3000,50),
            PhysicsSlider("Air Fric","air_fric",fric.get('air',250),50,1000,25),
            PhysicsSlider("Max Run Spd","max_run",150,50,500,10),
            PhysicsSlider("Jump Vel","jump_vel",800,200,1500,25),
            PhysicsSlider("Max Fall Spd","max_fall",550,200,1200,25),
        ]
        self._center_camera()

    def _center_camera(self):
        vw=WINDOW_W-LEFT_PANEL_W-RIGHT_PANEL_W; vh=WINDOW_H-TOOLBAR_H-STATUS_H-HSB_H
        self.cam_x=(self.level.cols*TILE_SIZE*self.zoom-vw)/2
        self.cam_y=(self.level.rows*TILE_SIZE*self.zoom-vh)/2
    def reset_view(self): self.cam_x=0.0; self.cam_y=0.0; self.zoom=1.0
    def push_undo(self):
        self.undo_stack.append(self.level.clone())
        if len(self.undo_stack)>UNDO_LIMIT: self.undo_stack.pop(0)
        self.redo_stack.clear()
    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.level.clone()); self.level=self.undo_stack.pop(); self.dirty=True
    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.level.clone()); self.level=self.redo_stack.pop(); self.dirty=True
    def clear_all(self):
        self.push_undo(); self.level.clear_all(); self.level.platforms.clear(); self.dirty=True
    def world_to_screen(self,wx,wy):
        ts=TILE_SIZE*self.zoom
        return (int(wx*ts-self.cam_x+LEFT_PANEL_W), int(wy*ts-self.cam_y+TOOLBAR_H))
    def world_pixel_to_screen(self,px,py):
        return (int(px*self.zoom-self.cam_x+LEFT_PANEL_W), int(py*self.zoom-self.cam_y+TOOLBAR_H))
    def screen_to_world_pixel(self,sx,sy,snap=True):
        wx=(sx-LEFT_PANEL_W+self.cam_x)/self.zoom; wy=(sy-TOOLBAR_H+self.cam_y)/self.zoom
        if snap: wx=round(wx/TILE_SIZE)*TILE_SIZE; wy=round(wy/TILE_SIZE)*TILE_SIZE
        return wx,wy
    def screen_to_cell(self,sx,sy):
        ts=TILE_SIZE*self.zoom
        return int((sy-TOOLBAR_H+self.cam_y)/ts), int((sx-LEFT_PANEL_W+self.cam_x)/ts)
    def paint_cell(self,r,c,char):
        if (r,c)==self.last_cell: return
        if self.level.get(r,c)!=char:
            self.level.set(r,c,char); self.last_cell=(r,c); self.dirty=True
    def flood_fill(self,r,c,char):
        target=self.level.get(r,c)
        if target is None or target==char: return
        self.push_undo(); stack=[(r,c)]; vis=set()
        while stack:
            cr,cc=stack.pop()
            if (cr,cc) in vis: continue
            if self.level.get(cr,cc)!=target: continue
            vis.add((cr,cc)); self.level.set(cr,cc,char)
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr,nc=cr+dr,cc+dc
                if (nr,nc) not in vis and self.level.get(nr,nc)==target: stack.append((nr,nc))
        self.dirty=True
    def hit_test_platform_handle(self,sx,sy):
        for i,p in enumerate(self.level.platforms):
            for handle in ('start','end'):
                wp=p.start if handle=='start' else p.end
                hsx,hsy=self.world_pixel_to_screen(wp[0]+p.width/2,wp[1]+p.height/2)
                if math.hypot(sx-hsx,sy-hsy)<=HANDLE_HIT_R: return i,handle
        return None,None
    def load_level_by_id(self, level_id):
        path=self.game_config.get_level_file_path(level_id)
        if not path: print(f"[Editor] No file for '{level_id}'"); return False
        try:
            self.level=Level.load(path); self.level.level_id=level_id
            self.undo_stack.clear(); self.redo_stack.clear()
            self.dirty=False; self._center_camera(); self.browser_level_id=level_id
            print(f"[Editor] Loaded '{level_id}' from {path}"); return True
        except Exception as e: print(f"[Editor] Load failed: {e}"); return False
    def get_active_brush(self):
        """Return the char that will be painted (respects eraser mode)."""
        if self.eraser_mode: return ' '
        return self.selected_tile

# ─── Draw: Platforms ─────────────────────────────────────────────
def draw_platforms(surf, ed, vp_rect):
    def bscrn(p,wp):
        sx,sy=ed.world_pixel_to_screen(wp[0],wp[1])
        return sx,sy,int(p.width*ed.zoom),int(p.height*ed.zoom)
    def dashed(s,col,p1,p2,dash=8,gap=5):
        dx=p2[0]-p1[0]; dy=p2[1]-p1[1]; L=math.hypot(dx,dy)
        if L<1: return
        ux,uy=dx/L,dy/L; pos=0.0; on=True
        while pos<L:
            seg=dash if on else gap; end=min(pos+seg,L)
            if on:
                pygame.draw.line(s,col,(int(p1[0]+ux*pos),int(p1[1]+uy*pos)),
                                 (int(p1[0]+ux*end),int(p1[1]+uy*end)),1)
            pos+=seg; on=not on
    for i,p in enumerate(ed.level.platforms):
        sel=(i==ed.sel_plat_idx)
        bx,by,bw,bh=bscrn(p,p.start)
        br=pygame.Rect(bx,by,bw,bh)
        if br.colliderect(vp_rect):
            pygame.draw.rect(surf,PLAT_BODY_COL,br)
            pygame.draw.rect(surf,PLAT_HIGH_COL,(bx,by,bw,max(2,bh//4)))
            pygame.draw.rect(surf,PLAT_SEL_COL if sel else (160,90,30),br,2)
        ex,ey,ew,eh=bscrn(p,p.end)
        gs=pygame.Surface((max(1,ew),max(1,eh)),pygame.SRCALPHA); gs.fill((*PLAT_BODY_COL,100))
        pygame.draw.rect(gs,(*PLAT_SEL_COL,220) if sel else (160,90,30,120),(0,0,ew,eh),2)
        surf.blit(gs,(ex,ey))
        sc=(bx+bw//2,by+bh//2); ec=(ex+ew//2,ey+eh//2)
        dashed(surf,PLAT_PATH_COL,sc,ec)
        for cx2,cy2,cl in [(bx+bw//2,by+bh//2,HANDLE_START),(ex+ew//2,ey+eh//2,HANDLE_END)]:
            pygame.draw.circle(surf,cl,(cx2,cy2),HANDLE_R)
            pygame.draw.circle(surf,(255,255,255),(cx2,cy2),HANDLE_R,2)
    if ed.selected_tile=='M' and ed.plat_placing and ed.plat_ghost_end:
        sx0,sy0=ed.world_pixel_to_screen(*ed.plat_start_world)
        sw=int(PLAT_DEFAULT_W*ed.zoom); sh=int(PLAT_DEFAULT_H*ed.zoom)
        ex0,ey0=ed.world_pixel_to_screen(*ed.plat_ghost_end)
        g1=pygame.Surface((max(1,sw),max(1,sh)),pygame.SRCALPHA); g1.fill((*PLAT_BODY_COL,160)); surf.blit(g1,(sx0,sy0))
        g2=pygame.Surface((max(1,sw),max(1,sh)),pygame.SRCALPHA); g2.fill((*PLAT_BODY_COL,80)); surf.blit(g2,(ex0,ey0))
        dashed(surf,PLAT_PATH_COL,(sx0+sw//2,sy0+sh//2),(ex0+sw//2,ey0+sh//2))
        pygame.draw.circle(surf,HANDLE_START,(sx0+sw//2,sy0+sh//2),HANDLE_R)
        pygame.draw.circle(surf,HANDLE_END,(ex0+sw//2,ey0+sh//2),HANDLE_R)

# ─── Draw: Viewport (with clipping!) ────────────────────────────
def draw_viewport(surf, ed, font, sfont):
    vx=LEFT_PANEL_W; vy=TOOLBAR_H
    vw=WINDOW_W-LEFT_PANEL_W-RIGHT_PANEL_W
    vh=WINDOW_H-TOOLBAR_H-STATUS_H-HSB_H   # reserve bottom strip for h-scrollbar
    vp_rect=pygame.Rect(vx,vy,vw,vh)

    # CRITICAL: clip to viewport so tiles never overlap toolbar/panels
    surf.set_clip(vp_rect)

    pygame.draw.rect(surf,UI_BG,vp_rect)
    ts=TILE_SIZE*ed.zoom; lv=ed.level
    cs=max(0,int(ed.cam_x/ts)); rs=max(0,int(ed.cam_y/ts))
    ce=min(lv.cols,cs+int(vw/ts)+2); re=min(lv.rows,rs+int(vh/ts)+2)
    for r in range(rs,re):
        for c in range(cs,ce):
            sx,sy=ed.world_to_screen(c,r); ch=lv.get(r,c) or ' '
            tr=pygame.Rect(sx,sy,int(ts),int(ts))
            if not tr.colliderect(vp_rect): continue
            if USE_REAL_OBJECTS: render_tile_object(surf,ch,sx,sy,int(ts))
            else:
                draw_tile_rect(surf,ch,sx,sy,int(ts),int(ts))
                if ts>=20: draw_tile_symbol(surf,font,ch,sx,sy,int(ts),int(ts))
    bx,by=ed.world_to_screen(0,0)
    pygame.draw.rect(surf,(70,88,140),(bx,by,int(lv.cols*ts),int(lv.rows*ts)),2)
    draw_platforms(surf,ed,vp_rect)
    if ed.show_grid and ts>=6:
        for c in range(cs,ce+1):
            sx=int(c*ts-ed.cam_x+vx); bold=c%10==0
            pygame.draw.line(surf,GRID_BOLD if bold else GRID_COLOR,(sx,vy),(sx,vy+vh),2 if bold else 1)
        for r in range(rs,re+1):
            sy=int(r*ts-ed.cam_y+vy); bold=r%10==0
            pygame.draw.line(surf,GRID_BOLD if bold else GRID_COLOR,(vx,sy),(vx+vw,sy),2 if bold else 1)
    if ed.hover_cell:
        hr,hc=ed.hover_cell; sx,sy=ed.world_to_screen(hc,hr)
        if ed.eraser_mode:
            pygame.draw.rect(surf,(255,80,80),(sx,sy,int(ts),int(ts)),2)
            pygame.draw.line(surf,(255,80,80),(sx+2,sy+2),(sx+int(ts)-2,sy+int(ts)-2),2)
            pygame.draw.line(surf,(255,80,80),(sx+int(ts)-2,sy+2),(sx+2,sy+int(ts)-2),2)
        else:
            pygame.draw.rect(surf,(255,255,255),(sx,sy,int(ts),int(ts)),2)
            gh=pygame.Surface((int(ts),int(ts)),pygame.SRCALPHA)
            draw_tile_rect(gh,ed.selected_tile,0,0,int(ts),int(ts),alpha=100)
            surf.blit(gh,(sx,sy))

    surf.set_clip(None)

    # ── Horizontal scrollbar strip ────────────────────────────────
    hsb_y = vy + vh
    hsb_rect = pygame.Rect(vx, hsb_y, vw, HSB_H)
    pygame.draw.rect(surf, UI_PANEL2, hsb_rect)
    pygame.draw.line(surf, UI_BORDER, (vx, hsb_y), (vx+vw, hsb_y), 1)

    level_w = lv.cols * ts
    if level_w > vw:
        ratio = vw / level_w
        thumb_w = max(32, int(vw * ratio))
        scroll_range_px = max(1, vw - thumb_w)
        max_cam_x = level_w - vw
        frac = max(0.0, min(1.0, ed.cam_x / max_cam_x))
        thumb_x = vx + int(frac * scroll_range_px)
        mpos = pygame.mouse.get_pos()
        ht = pygame.Rect(thumb_x, hsb_y+2, thumb_w, HSB_H-4)
        hov_h = ht.collidepoint(mpos) or ed._hscroll_dragging
        pygame.draw.rect(surf, UI_ACCENT if hov_h else UI_BORDER, ht, border_radius=3)
        ed._hscroll_thumb = ht
    else:
        ed._hscroll_thumb = pygame.Rect(0,0,0,0)
        # Small thumb showing current position even when fully in view
        if level_w > 0:
            thumb_w2 = max(24, int(vw * vw / max(1, level_w)))
            pygame.draw.rect(surf, UI_SELECT_DIM,
                             pygame.Rect(vx+2, hsb_y+2, min(thumb_w2, vw-4), HSB_H-4), border_radius=3)

# ─── Draw: Left Panel (stores rects for click detection!) ────────
def draw_left_panel(surf, ed, font, sfont):
    px=LEFT_PANEL_W; pad=6; tile_h=32; icon_s=tile_h-4
    pygame.draw.rect(surf,UI_PANEL,(0,TOOLBAR_H,px,WINDOW_H-TOOLBAR_H-STATUS_H))
    pygame.draw.line(surf,UI_BORDER,(px-1,TOOLBAR_H),(px-1,WINDOW_H-STATUS_H),1)
    y=TOOLBAR_H+8
    sfont.render_to(surf,(pad+2,y),"TILES",fgcolor=UI_SUBTEXT,size=10); y+=16

    ed._tile_rects.clear()
    for tile in TILES:
        char,label,color,tcol=tile
        rect=pygame.Rect(pad,y,px-pad*2,tile_h)
        ed._tile_rects[char]=rect  # STORE for click detection
        sel = (char==ed.selected_tile and not ed.eraser_mode)
        pygame.draw.rect(surf,UI_SELECT if sel else UI_PANEL2,rect,border_radius=4)
        draw_palette_icon(surf,char,pygame.Rect(pad+3,y+2,icon_s,icon_s))
        tc=UI_TEXT if sel else UI_SUBTEXT
        try:
            sfont.render_to(surf,(pad+icon_s+8,y+tile_h//2-5),label,fgcolor=tc,size=11)
            sfont.render_to(surf,(px-pad-14,y+tile_h//2-5),char if char!=' ' else '`',fgcolor=(55,65,85),size=10)
        except: pass
        if sel: pygame.draw.rect(surf,UI_ACCENT,rect,2,border_radius=4)
        y+=tile_h+2

    # ── Tools section ────────────────────────────────────────────
    y+=6; pygame.draw.line(surf,UI_BORDER,(pad,y),(px-pad,y),1); y+=8
    sfont.render_to(surf,(pad+2,y),"TOOLS",fgcolor=UI_SUBTEXT,size=10); y+=16

    ed._tool_rects.clear()
    tool_items = [
        ("eraser", "X  Eraser",    ed.eraser_mode),
        ("fill",   "F  Fill Mode", ed.fill_mode),
        ("grid",   "G  Grid",      ed.show_grid),
    ]
    for key, label, active in tool_items:
        br=pygame.Rect(pad,y,px-pad*2,26)
        ed._tool_rects[key]=br  # STORE for click detection
        if key=="eraser" and active:
            pygame.draw.rect(surf,UI_DANGER,br,border_radius=4)
        elif active:
            pygame.draw.rect(surf,UI_BTN_ACT,br,border_radius=4)
        else:
            pygame.draw.rect(surf,UI_BTN,br,border_radius=4)
        try:
            fc=UI_TEXT if active else UI_SUBTEXT
            sfont.render_to(surf,(pad+8,y+7),label,fgcolor=fc,size=11)
        except: pass
        y+=30

    # ── Platform section ─────────────────────────────────────────
    y+=6; pygame.draw.line(surf,UI_BORDER,(pad,y),(px-pad,y),1); y+=8
    sfont.render_to(surf,(pad+2,y),"PLATFORMS",fgcolor=UI_SUBTEXT,size=10); y+=16
    pa=ed.selected_tile=='M'
    pr=pygame.Rect(pad,y,px-pad*2,26)
    ed._tool_rects['platform']=pr  # STORE
    pygame.draw.rect(surf,UI_BTN_ACT if pa else UI_BTN,pr,border_radius=4)
    ico=pygame.Rect(pad+6,y+10,18,5)
    pygame.draw.rect(surf,PLAT_BODY_COL,ico,border_radius=1)
    pygame.draw.circle(surf,HANDLE_START,(ico.x+3,ico.centery),3)
    pygame.draw.circle(surf,HANDLE_END,(ico.x+ico.w-3,ico.centery),3)
    try: sfont.render_to(surf,(pad+28,y+7),"M  Platform",fgcolor=UI_TEXT if pa else UI_SUBTEXT,size=11)
    except: pass
    y+=30
    # Speed control
    sr=pygame.Rect(pad,y,px-pad*2,24)
    pygame.draw.rect(surf,UI_PANEL2,sr,border_radius=3)
    pygame.draw.rect(surf,UI_BORDER,sr,1,border_radius=3)
    try: sfont.render_to(surf,(pad+6,y+6),f"Spd: {ed.plat_default_spd:.0f}",fgcolor=UI_TEXT,size=10)
    except: pass
    bw=18
    mr=pygame.Rect(px-pad-bw*2-4,y+3,bw,18); pr2=pygame.Rect(px-pad-bw,y+3,bw,18)
    for btn,sym in [(mr,'-'),(pr2,'+')]:
        pygame.draw.rect(surf,UI_BORDER,btn,border_radius=3)
        try:
            b=sfont.get_rect(sym,size=12)
            sfont.render_to(surf,(btn.x+(btn.w-b.width)//2,btn.y+(btn.h-b.height)//2),sym,fgcolor=UI_TEXT,size=12)
        except: pass
    ed._spd_minus_r=mr; ed._spd_plus_r=pr2
    y+=28
    # Hint
    if pa:
        hint = "Click END" if ed.plat_placing else "Click START"
        hcol = UI_WARN if ed.plat_placing else UI_ACCENT2
        try: sfont.render_to(surf,(pad+4,y),hint,fgcolor=hcol,size=10)
        except: pass
        y+=16

    # ── Controls reference ────────────────────────────────────────
    y+=6; pygame.draw.line(surf,UI_BORDER,(pad,y),(px-pad,y),1); y+=8
    try: sfont.render_to(surf,(pad+2,y),"CONTROLS",fgcolor=UI_SUBTEXT,size=10)
    except: pass
    y+=14

    CONTROLS=[
        ("LClick",  "Paint tile"),
        ("RClick",  "Erase cell"),
        ("MDrag",   "Pan camera"),
        ("Scroll",  "Zoom in/out"),
        ("Ctrl+S",  "Save"),
        ("Ctrl+O",  "Open"),
        ("Ctrl+N",  "New level"),
        ("Ctrl+P",  "Play level"),
        ("Ctrl+Z",  "Undo"),
        ("Ctrl+Y",  "Redo"),
        ("Ctrl+X",  "Clear all"),
        ("G",       "Toggle grid"),
        ("F",       "Fill mode"),
        ("X",       "Eraser mode"),
        ("M",       "Platform tool"),
        ("Del",     "Delete platform"),
        ("Home",    "Reset view"),
        ("1-0",     "Select tile"),
        ("Esc",     "Cancel action"),
    ]
    key_col  = UI_ACCENT2
    desc_col = (100, 72, 82)
    max_y    = WINDOW_H - STATUS_H - 4
    for key_txt, desc_txt in CONTROLS:
        if y+12 > max_y: break
        try:
            sfont.render_to(surf,(pad+2,   y), key_txt,  fgcolor=key_col,  size=9)
            sfont.render_to(surf,(pad+52,  y), desc_txt, fgcolor=desc_col, size=9)
        except: pass
        y+=12

# ─── Draw: Right Panel ───────────────────────────────────────────

def draw_right_panel(surf, ed, sfont, mpos):
    rx=WINDOW_W-RIGHT_PANEL_W; ry=TOOLBAR_H; rw=RIGHT_PANEL_W
    rh=WINDOW_H-TOOLBAR_H-STATUS_H
    cw=rw-SB_W          # content width (leaves room for scrollbar)
    sb_x=rx+cw          # scrollbar x

    # Panel background
    pygame.draw.rect(surf,UI_PANEL,(rx,ry,rw,rh))
    pygame.draw.line(surf,UI_BORDER,(rx,ry),(rx,ry+rh),1)

    # ── Content (clipped) ────────────────────────────────────────
    clip=pygame.Rect(rx,ry,cw,rh)
    surf.set_clip(clip)

    pad=8; y=ry+8-ed.right_scroll
    ICON_W=22; ICON_GAP=2; icons_total=(ICON_W+ICON_GAP)*2

    ed._level_btn_rects.clear(); ed._unlisted_btn_rects.clear()
    ed._unlisted_del_rects.clear(); ed._toggle_btn_rects.clear()
    ed._rename_btn_rects.clear();  ed._delete_btn_rects.clear()
    gc=ed.game_config

    # Helper: only register a rect for hit-testing if it's visible in the panel
    def reg(d, key, rect):
        if clip.colliderect(rect): d[key]=rect

    # ── LEVELS ───────────────────────────────────────────────────
    try: sfont.render_to(surf,(rx+pad,y),"LEVELS",fgcolor=UI_ACCENT,size=11)
    except: pass
    y+=18

    if gc.level_ids:
        for lid in gc.level_ids:
            lcfg=gc.levels.get(lid,{}); fn=lcfg.get('file','???')
            cur=lid==ed.browser_level_id; row_h=34
            # Icon buttons (right edge of content area)
            ren_r=pygame.Rect(rx+cw-icons_total+ICON_GAP,      y+6,ICON_W,ICON_W)
            del_r=pygame.Rect(rx+cw-ICON_W,                    y+6,ICON_W,ICON_W)
            reg(ed._rename_btn_rects,lid,ren_r)
            reg(ed._delete_btn_rects,lid,del_r)
            # Load button
            btn_r=pygame.Rect(rx+pad+20,y,cw-pad*2-20-icons_total-4,row_h)
            reg(ed._level_btn_rects,lid,btn_r)
            hov=btn_r.collidepoint(mpos)
            bg=UI_SELECT if cur else (UI_BTN_HOVER if hov else UI_BTN)
            pygame.draw.rect(surf,bg,btn_r,border_radius=4)
            try:
                sfont.render_to(surf,(btn_r.x+8,y+5), lid,fgcolor=UI_TEXT,   size=12)
                sfont.render_to(surf,(btn_r.x+8,y+20),fn, fgcolor=UI_SUBTEXT,size=9)
            except: pass
            if cur: pygame.draw.rect(surf,UI_ACCENT,btn_r,2,border_radius=4)
            # Toggle checkbox ✓ (green=enabled)
            cb=pygame.Rect(rx+pad,y+9,16,16)
            reg(ed._toggle_btn_rects,lid,cb)
            pygame.draw.rect(surf,UI_ACCENT2,cb,border_radius=3)
            pygame.draw.line(surf,(10,30,10),(cb.x+3,cb.y+8),(cb.x+6,cb.y+12),2)
            pygame.draw.line(surf,(10,30,10),(cb.x+6,cb.y+12),(cb.x+13,cb.y+3),2)
            # ✎ Rename
            hov_r=ren_r.collidepoint(mpos)
            pygame.draw.rect(surf,UI_BTN_HOVER if hov_r else UI_BTN,ren_r,border_radius=3)
            try: sfont.render_to(surf,(ren_r.x+4,ren_r.y+4),"✎",fgcolor=UI_ACCENT if hov_r else UI_SUBTEXT,size=11)
            except: pass
            # ✕ Delete
            hov_d=del_r.collidepoint(mpos)
            pygame.draw.rect(surf,UI_DANGER if hov_d else UI_BTN,del_r,border_radius=3)
            try: sfont.render_to(surf,(del_r.x+5,del_r.y+4),"✕",fgcolor=UI_TEXT if hov_d else UI_SUBTEXT,size=11)
            except: pass
            y+=row_h+2
    elif gc.config_path is None:
        try: sfont.render_to(surf,(rx+pad+4,y),"No game_config found",fgcolor=UI_SUBTEXT,size=10)
        except: pass
        y+=18

    # ── DISABLED (commented-out) — shown even when all levels are disabled ──
    commented=gc.get_commented_levels()
    if commented:
        y+=4
        try: sfont.render_to(surf,(rx+pad,y),"DISABLED",fgcolor=UI_DANGER,size=10)
        except: pass
        y+=16
        for lid in sorted(commented.keys()):
            row_h2=32
            del_r2=pygame.Rect(rx+cw-ICON_W,y+6,ICON_W,ICON_W)
            reg(ed._delete_btn_rects,f"off:{lid}",del_r2)
            btn_r=pygame.Rect(rx+pad+20,y,cw-pad*2-20-ICON_W-ICON_GAP,row_h2)
            reg(ed._toggle_btn_rects,f"off:{lid}",pygame.Rect(rx+pad,y+8,16,16))
            hov=btn_r.collidepoint(mpos)
            pygame.draw.rect(surf,UI_BTN_HOVER if hov else UI_BTN,btn_r,border_radius=3)
            try:
                sfont.render_to(surf,(btn_r.x+8,y+5), lid,  fgcolor=UI_SUBTEXT,size=11)
                stored_fn=gc._get_commented_file(lid)
                if stored_fn:
                    sfont.render_to(surf,(btn_r.x+8,y+19),stored_fn,fgcolor=(75,50,60),size=9)
            except: pass
            # Red empty checkbox (click to re-enable)
            cb2=pygame.Rect(rx+pad,y+8,16,16)
            pygame.draw.rect(surf,UI_DANGER,cb2,1,border_radius=3)
            hov_d2=del_r2.collidepoint(mpos)
            pygame.draw.rect(surf,UI_DANGER if hov_d2 else UI_BTN,del_r2,border_radius=3)
            try: sfont.render_to(surf,(del_r2.x+5,del_r2.y+4),"✕",fgcolor=UI_TEXT if hov_d2 else UI_SUBTEXT,size=11)
            except: pass
            y+=row_h2+2


    # ── UNLISTED FILES ───────────────────────────────────────────
    all_f=gc.get_all_level_files(extra_path=ed.level.filename)
    listed_files={gc.levels.get(lid,{}).get('file','') for lid in gc.level_ids}
    unlisted=[f for f in all_f if f.name not in listed_files]
    if unlisted:
        y+=8
        try: sfont.render_to(surf,(rx+pad,y),"UNLISTED FILES",fgcolor=UI_SUBTEXT,size=10)
        except: pass
        y+=16
        for fp in unlisted:
            del_uw=22
            del_ur=pygame.Rect(rx+cw-del_uw,y+3,del_uw,20)
            btn_r=pygame.Rect(rx+pad,y,cw-pad*2-del_uw-4,26)
            hov=btn_r.collidepoint(mpos); cur=str(fp)==ed.level.filename
            pygame.draw.rect(surf,UI_SELECT_DIM if cur else (UI_BTN_HOVER if hov else UI_BTN),btn_r,border_radius=3)
            try: sfont.render_to(surf,(btn_r.x+6,y+7),fp.name,fgcolor=UI_TEXT if cur else UI_SUBTEXT,size=10)
            except: pass
            reg(ed._unlisted_btn_rects,str(fp),btn_r)
            # Delete file button
            hov_du=del_ur.collidepoint(mpos)
            pygame.draw.rect(surf,UI_DANGER if hov_du else UI_BTN,del_ur,border_radius=3)
            try: sfont.render_to(surf,(del_ur.x+5,del_ur.y+3),"✕",fgcolor=UI_TEXT if hov_du else UI_SUBTEXT,size=10)
            except: pass
            reg(ed._unlisted_del_rects,str(fp),del_ur)
            y+=28

        # Assign button (only when an unlisted file is currently loaded)
        if ed.level.filename and any(str(fp)==ed.level.filename for fp in unlisted):
            y+=4
            ab=pygame.Rect(rx+pad,y,cw-pad*2,26)
            hov_a=ab.collidepoint(mpos)
            pygame.draw.rect(surf,UI_ACCENT if hov_a else UI_BTN_ACT,ab,border_radius=4)
            try: sfont.render_to(surf,(rx+pad+8,y+7),"⊕  Assign to Level ID",
                                  fgcolor=UI_BG if hov_a else UI_TEXT,size=10)
            except: pass
            ed._assign_btn_rect=ab; y+=30
        else:
            ed._assign_btn_rect=pygame.Rect(0,0,0,0)

    # ── PHYSICS sliders ──────────────────────────────────────────
    y+=10
    pygame.draw.line(surf,UI_BORDER,(rx+pad,y),(rx+cw-pad,y),1); y+=8
    try: sfont.render_to(surf,(rx+pad,y),"PHYSICS",fgcolor=UI_ACCENT,size=11)
    except: pass
    y+=18
    sw=cw-pad*2-4
    for sl in ed.physics_sliders:
        try:
            sfont.render_to(surf,(rx+pad+2,y),sl.label,fgcolor=UI_SUBTEXT,size=10)
            vs=f"{sl.value:.0f}"; vb=sfont.get_rect(vs,size=10)
            sfont.render_to(surf,(rx+cw-pad-vb.width-2,y),vs,fgcolor=UI_TEXT,size=10)
        except: pass
        y+=14
        tr=pygame.Rect(rx+pad+2,y,sw,10)
        pygame.draw.rect(surf,UI_PANEL2,tr,border_radius=5)
        pygame.draw.rect(surf,UI_BORDER,tr,1,border_radius=5)
        frac=sl.fraction(); fw=int(frac*(sw-4))
        if fw>0: pygame.draw.rect(surf,UI_ACCENT,(tr.x+2,tr.y+2,fw,6),border_radius=3)
        tx=tr.x+2+int(frac*(sw-4))
        pygame.draw.rect(surf,UI_TEXT,(tx-5,tr.y-2,10,14),border_radius=3)
        sl.rect=tr; y+=18

    # ── LEVEL INFO ───────────────────────────────────────────────
    y+=8; pygame.draw.line(surf,UI_BORDER,(rx+pad,y),(rx+cw-pad,y),1); y+=8
    try: sfont.render_to(surf,(rx+pad,y),"LEVEL INFO",fgcolor=UI_ACCENT,size=11)
    except: pass
    y+=16
    lv=ed.level
    info=[f"Size: {lv.rows}r x {lv.cols}c",f"Platforms: {len(lv.platforms)}"]
    if lv.filename: info.append(f"File: {Path(lv.filename).name}")
    if lv.level_id: info.append(f"ID: {lv.level_id}")
    bb=lv.bounding_box()
    if bb: info.append(f"Export: {bb[2]-bb[0]+3}r x {bb[3]-bb[1]+3}c")
    for line in info:
        try: sfont.render_to(surf,(rx+pad+4,y),line,fgcolor=UI_SUBTEXT,size=10)
        except: pass
        y+=14

    # ── Compute scroll range ──────────────────────────────────────
    content_h=(y+ed.right_scroll)-ry
    ed.right_max_scroll=max(0,content_h-rh+16)
    ed.right_scroll=min(ed.right_scroll,ed.right_max_scroll)

    surf.set_clip(None)

    # ── Scrollbar (drawn OUTSIDE clip) ───────────────────────────
    pygame.draw.rect(surf,UI_PANEL2,(sb_x,ry,SB_W,rh))
    pygame.draw.line(surf,UI_BORDER,(sb_x,ry),(sb_x,ry+rh),1)
    if ed.right_max_scroll>0:
        ratio=rh/max(1,content_h)
        thumb_h=max(28,int(rh*ratio))
        scroll_range=max(1,rh-thumb_h)
        thumb_y=ry+int(ed.right_scroll/max(1,ed.right_max_scroll)*scroll_range)
        thumb_r=pygame.Rect(sb_x+2,thumb_y,SB_W-4,thumb_h)
        hov_sb=thumb_r.collidepoint(mpos) or ed._scrollbar_dragging
        pygame.draw.rect(surf,UI_ACCENT if hov_sb else UI_BORDER,thumb_r,border_radius=4)
        ed._scrollbar_thumb=thumb_r
    else:
        ed._scrollbar_thumb=pygame.Rect(0,0,0,0)


# ─── Toolbar + Status ────────────────────────────────────────────
def build_toolbar_buttons():
    return [
        ToolbarButton("New","Ctrl+N",lambda e:"NEW"),
        ToolbarButton("Open","Ctrl+O",lambda e:"OPEN"),
        ToolbarButton("Save","Ctrl+S",lambda e:"SAVE"),
        ToolbarButton("|","",None),
        ToolbarButton("Undo","Ctrl+Z",lambda e:e.undo()),
        ToolbarButton("Redo","Ctrl+Y",lambda e:e.redo()),
        ToolbarButton("|","",None),
        ToolbarButton("Clear","Ctrl+X",lambda e:e.clear_all()),
        ToolbarButton("|","",None),
        ToolbarButton("Grid","G",None,True,lambda e:e.show_grid),
        ToolbarButton("Fill","F",None,True,lambda e:e.fill_mode),
        ToolbarButton("Eraser","X",None,True,lambda e:e.eraser_mode),
        ToolbarButton("|","",None),
        ToolbarButton("Home","Home",lambda e:e.reset_view()),
        ToolbarButton("|","",None),
        ToolbarButton("\u25b6 Play","Ctrl+P",lambda e:"PLAY"),
    ]

def draw_toolbar(surf, ed, sfont, btns, mpos):
    pygame.draw.rect(surf,UI_TOOLBAR,(0,0,WINDOW_W,TOOLBAR_H))
    pygame.draw.line(surf,UI_BORDER,(0,TOOLBAR_H-1),(WINDOW_W,TOOLBAR_H-1),1)
    try:
        sfont.render_to(surf,(10,8),"PEAK",fgcolor=UI_ACCENT,size=18)
        sfont.render_to(surf,(10,30),"Level Editor",fgcolor=UI_SUBTEXT,size=10)
    except: pass
    title=""
    if ed.level.filename: title=Path(ed.level.filename).name
    if ed.level.level_id: title=f"[{ed.level.level_id}]  {title}"
    if ed.dirty: title+="  \u25cf"
    if title:
        try: sfont.render_to(surf,(110,18),title,fgcolor=UI_TEXT,size=13)
        except: pass
    bx=LEFT_PANEL_W+12; by=8; bh=34
    for btn in btns:
        if btn.label=="|":
            pygame.draw.line(surf,UI_BORDER,(bx+4,by+4),(bx+4,by+bh-4),1); bx+=12; continue
        try: tw=sfont.get_rect(btn.label,size=11).width
        except: tw=30
        bw2=max(44,tw+20); btn.rect=pygame.Rect(bx,by,bw2,bh)
        hov=btn.rect.collidepoint(mpos)
        act=btn.get_active(ed) if btn.get_active else False
        is_play = btn.label.startswith("\u25b6")
        if is_play:
            bg=(30,90,50) if hov else (20,65,35)
        elif btn.label=="Eraser" and act:
            bg=UI_DANGER
        elif act:
            bg=UI_BTN_ACT
        elif hov:
            bg=UI_BTN_HOVER
        else:
            bg=UI_BTN
        pygame.draw.rect(surf,bg,btn.rect,border_radius=5)
        if is_play:
            pygame.draw.rect(surf,(60,200,100),btn.rect,2,border_radius=5)
        elif act: pygame.draw.rect(surf,UI_ACCENT if btn.label!="Eraser" else UI_DANGER,btn.rect,2,border_radius=5)
        try:
            lb=sfont.get_rect(btn.label,size=11)
            fc=(150,255,170) if is_play else (UI_TEXT if (act or hov) else UI_SUBTEXT)
            sfont.render_to(surf,(bx+(bw2-lb.width)//2,by+(bh-lb.height)//2),
                            btn.label,fgcolor=fc,size=11)
        except: pass
        if btn.shortcut and hov:
            try:
                sb=sfont.get_rect(btn.shortcut,size=8)
                sfont.render_to(surf,(bx+(bw2-sb.width)//2,by+bh-10),btn.shortcut,fgcolor=(80,90,110),size=8)
            except: pass
        bx+=bw2+4

def draw_status(surf, ed, sfont):
    y=WINDOW_H-STATUS_H
    pygame.draw.rect(surf,UI_STATUS,(0,y,WINDOW_W,STATUS_H))
    pygame.draw.line(surf,UI_BORDER,(0,y),(WINDOW_W,y),1)
    lv=ed.level
    if ed.eraser_mode:
        tl="Eraser"
    elif ed.selected_tile!='M':
        tl=TILE_BY_CHAR.get(ed.selected_tile,TILES[0])[1]
    else:
        tl='Platform'
    info=f"  {lv.rows}r x {lv.cols}c  |  Zoom {ed.zoom:.2f}x  |  Brush: {tl}  |  Plats: {len(lv.platforms)}"
    if ed.hover_cell and ed.selected_tile!='M':
        hr,hc=ed.hover_cell; ch=lv.get(hr,hc)
        info+=f"  |  [{hr},{hc}]: {TILE_BY_CHAR.get(ch,TILES[0])[1] if ch else 'Air'}"
    bb=lv.bounding_box()
    if bb: info+=f"  |  Export: {bb[2]-bb[0]+3}r x {bb[3]-bb[1]+3}c"
    ui=f"  Undo: {len(ed.undo_stack)}  |  Redo: {len(ed.redo_stack)}"
    try:
        sfont.render_to(surf,(LEFT_PANEL_W,y+7),info,fgcolor=UI_SUBTEXT,size=10)
        b=sfont.get_rect(ui,size=10)
        sfont.render_to(surf,(WINDOW_W-b.width-12,y+7),ui,fgcolor=UI_SUBTEXT,size=10)
    except: pass

# ─── File dialogs ────────────────────────────────────────────────
def launch_playtest(ed):
    """Save current level then spawn manual_play.py in a subprocess."""
    lv = ed.level
    # Auto-save if we have a filename
    if lv.filename:
        lv.save(lv.filename); ed.dirty=False
        print(f"[Editor] Auto-saved: {lv.filename}")
    else:
        p = dialog_save()
        if not p: return
        lv.save(p); ed.dirty=False; print(f"[Editor] Saved: {p}")

    level_id = lv.level_id or ed.browser_level_id
    # If the loaded level has no ID, check if its filename matches a known level in config
    if not level_id and lv.filename and ed.game_config.level_ids:
        loaded_name = Path(lv.filename).name
        for lid2 in ed.game_config.level_ids:
            if ed.game_config.levels.get(lid2,{}).get('file','') == loaded_name:
                level_id = lid2; break

    # ── Find project root (the dir that CONTAINS the 'code' package) ──
    # Walk up from this script until we find a directory that has a
    # 'code' subdirectory with an __init__.py (i.e. the package root).
    def find_project_root(start: Path) -> Path:
        current = start.resolve()
        for _ in range(10):  # max 10 levels up
            if (current / 'code' / '__init__.py').exists():
                return current
            if (current / 'code').is_dir():
                return current
            parent = current.parent
            if parent == current:
                break
            current = parent
        return start.resolve()  # fallback

    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)

    # ── Locate manual_play.py ──────────────────────────────────────
    candidates = [
        script_dir / 'manual_play.py',
        project_root / 'code' / 'scripts' / 'manual_play.py',
        project_root / 'manual_play.py',
    ]
    play_script = next((str(p) for p in candidates if Path(p).exists()), None)
    if not play_script:
        print("[Editor] Could not find manual_play.py"); return

    # ── Build environment: inherit current env + add PYTHONPATH ───
    env = os.environ.copy()
    existing_pp = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = (str(project_root) + os.pathsep + existing_pp).rstrip(os.pathsep)
    if level_id:
        env['PEAK_PLAY_LEVEL'] = str(level_id)

    args_cmd = [sys.executable, play_script, '--game', 'platformer']
    if level_id:
        args_cmd += ['--level', str(level_id)]
    elif lv.filename:
        # No level_id — pass the raw file path; manual_play will inject it
        args_cmd += ['--file', str(Path(lv.filename).resolve())]
    print(f"[Editor] Project root : {project_root}")
    print(f"[Editor] Launching    : {' '.join(args_cmd)}")
    subprocess.Popen(args_cmd, env=env, cwd=str(project_root))

def dialog_open():
    r=tk.Tk(); r.withdraw()
    p=filedialog.askopenfilename(title="Open Level",filetypes=[("Text","*.txt"),("All","*.*")])
    r.destroy(); return p or None
def dialog_save(default="level.txt"):
    r=tk.Tk(); r.withdraw()
    p=filedialog.asksaveasfilename(title="Save Level",defaultextension=".txt",
                                   initialfile=default,filetypes=[("Text","*.txt"),("All","*.*")])
    r.destroy(); return p or None
def dialog_new_size():
    r=tk.Tk(); r.withdraw()
    rows=simpledialog.askinteger("New Level","Rows:",initialvalue=DEFAULT_ROWS,minvalue=5,maxvalue=200,parent=r)
    cols=simpledialog.askinteger("New Level","Cols:",initialvalue=DEFAULT_COLS,minvalue=10,maxvalue=400,parent=r)
    r.destroy(); return (rows,cols) if rows and cols else (None,None)
def dialog_assign_level_id(filename, prompt=None):
    """Ask the user for a level ID (e.g. '1-5') to assign to the given filename."""
    r=tk.Tk(); r.withdraw()
    msg = prompt or f"Enter Level ID for  '{filename}'\n(e.g. 1-5, 2-3 …):"
    lid=simpledialog.askstring("Assign / Rename Level ID", msg, parent=r)
    r.destroy(); return lid.strip() if lid and lid.strip() else None

# ─── Main ────────────────────────────────────────────────────────
def main():
    global WINDOW_W, WINDOW_H
    pygame.init(); pygame.freetype.init()
    screen=pygame.display.set_mode((WINDOW_W,WINDOW_H),pygame.RESIZABLE)
    pygame.display.set_caption("PEAK Level Editor v2")
    clock=pygame.time.Clock()
    fp=pygame.font.match_font("dejavusans,segoeui,arial")
    font=pygame.freetype.Font(fp); sfont=pygame.freetype.Font(fp)
    gc=GameConfig()
    if gc.config_path:
        print(f"[Editor] Found game_config: {gc.config_path}")
        print(f"[Editor] Levels dir: {gc.levels_dir}")
        print(f"[Editor] {len(gc.level_ids)} levels configured")
    level=None
    if len(sys.argv)>1:
        try: level=Level.load(sys.argv[1]); print(f"[Editor] Loaded: {sys.argv[1]}")
        except Exception as e: print(f"[Editor] Failed: {e}")
    if level is None:
        level=Level()
        for r in range(level.rows): level.set(r,0,'#'); level.set(r,level.cols-1,'#')
        for c in range(level.cols): level.set(0,c,'#'); level.set(level.rows-1,c,'#')
        level.set(level.rows-2,2,'P')
    ed=Editor(level,gc); btns=build_toolbar_buttons(); pstarted=False
    running=True
    while running:
        clock.tick(60); WINDOW_W,WINDOW_H=screen.get_size()
        mx,my=pygame.mouse.get_pos()
        vl=LEFT_PANEL_W; vr2=WINDOW_W-RIGHT_PANEL_W
        in_vp=vl<=mx<vr2 and TOOLBAR_H<=my<WINDOW_H-STATUS_H-HSB_H
        if in_vp:
            r,c=ed.screen_to_cell(mx,my)
            ed.hover_cell=(r,c) if r>=0 and c>=0 and r<=ed.level.rows and c<=ed.level.cols else None
        else: ed.hover_cell=None

        for event in pygame.event.get():
            if event.type==pygame.QUIT: running=False
            elif event.type==pygame.VIDEORESIZE:
                screen=pygame.display.set_mode(event.size,pygame.RESIZABLE)
            elif event.type==pygame.MOUSEWHEEL:
                if mx>=WINDOW_W-RIGHT_PANEL_W:
                    ed.right_scroll=max(0,min(ed.right_max_scroll,ed.right_scroll-event.y*20))
                elif in_vp:
                    oz=ed.zoom; f=1.15 if event.y>0 else 1/1.15
                    ed.zoom=max(MIN_ZOOM,min(MAX_ZOOM,ed.zoom*f))
                    wmx=(mx-LEFT_PANEL_W+ed.cam_x)/(TILE_SIZE*oz)
                    wmy=(my-TOOLBAR_H+ed.cam_y)/(TILE_SIZE*oz)
                    ed.cam_x=wmx*TILE_SIZE*ed.zoom-(mx-LEFT_PANEL_W)
                    ed.cam_y=wmy*TILE_SIZE*ed.zoom-(my-TOOLBAR_H)

            elif event.type==pygame.MOUSEBUTTONDOWN:
                epx,epy=event.pos
                if event.button==1:
                    # ── H-scrollbar (viewport bottom strip) ──────────
                    hsb_y2=TOOLBAR_H+(WINDOW_H-TOOLBAR_H-STATUS_H-HSB_H)
                    in_hsb=(LEFT_PANEL_W<=epx<WINDOW_W-RIGHT_PANEL_W and
                            hsb_y2<=epy<hsb_y2+HSB_H)
                    if in_hsb:
                        if ed._hscroll_thumb.collidepoint(epx,epy):
                            ed._hscroll_dragging=True
                            ed._hscroll_drag_start_x=epx
                            ed._hscroll_drag_cam_start=ed.cam_x
                        else:
                            # Track click — jump to position
                            vw2=WINDOW_W-LEFT_PANEL_W-RIGHT_PANEL_W
                            frac2=max(0.0,min(1.0,(epx-LEFT_PANEL_W)/max(1,vw2)))
                            max_cx=max(0,ed.level.cols*TILE_SIZE*ed.zoom-vw2)
                            ed.cam_x=frac2*max_cx
                        continue  # don't process other clicks
                    # ── Toolbar ───────────────────────────────────
                    if epy<TOOLBAR_H:
                        for btn in btns:
                            if btn.label=="|": continue
                            if not btn.rect.collidepoint(epx,epy): continue
                            if btn.toggle:
                                if btn.label=="Grid": ed.show_grid=not ed.show_grid
                                elif btn.label=="Fill": ed.fill_mode=not ed.fill_mode
                                elif btn.label=="Eraser": ed.eraser_mode=not ed.eraser_mode
                            elif btn.action:
                                res=btn.action(ed)
                                if res=="NEW":
                                    rows,cols=dialog_new_size()
                                    if rows and cols:
                                        ed.level=Level(rows,cols)
                                        for r3 in range(rows): ed.level.set(r3,0,'#'); ed.level.set(r3,cols-1,'#')
                                        for c3 in range(cols): ed.level.set(0,c3,'#'); ed.level.set(rows-1,c3,'#')
                                        ed.level.set(rows-2,2,'P')
                                        ed.undo_stack.clear(); ed.redo_stack.clear(); ed.dirty=False; ed._center_camera()
                                elif res=="OPEN":
                                    p=dialog_open()
                                    if p:
                                        try: ed.level=Level.load(p); ed.undo_stack.clear(); ed.redo_stack.clear(); ed.dirty=False; ed._center_camera()
                                        except Exception as e2: print(f"[Editor] {e2}")
                                elif res=="SAVE":
                                    p=ed.level.filename or dialog_save()
                                    if p: ed.level.save(p); ed.dirty=False; print(f"[Editor] Saved: {p}")
                                elif res=="PLAY":
                                    launch_playtest(ed)
                            break

                    # ── Right panel ───────────────────────────────
                    elif epx>=WINDOW_W-RIGHT_PANEL_W:
                        # Scrollbar thumb drag
                        if ed._scrollbar_thumb.collidepoint(epx,epy):
                            ed._scrollbar_dragging=True
                            ed._scrollbar_drag_start_y=epy
                            ed._scrollbar_drag_scroll_start=ed.right_scroll
                            continue  # don't process content clicks
                        # Scrollbar track click (jump to position)
                        sb_x2=WINDOW_W-RIGHT_PANEL_W+(RIGHT_PANEL_W-SB_W)
                        if epx>=sb_x2:
                            rh2=WINDOW_H-TOOLBAR_H-STATUS_H
                            frac=max(0.0,min(1.0,(epy-TOOLBAR_H)/max(1,rh2)))
                            ed.right_scroll=int(frac*ed.right_max_scroll)
                            continue

                        clicked_something=False
                        # Level load buttons
                        for lid,rect in ed._level_btn_rects.items():
                            if rect.collidepoint(epx,epy):
                                ed.load_level_by_id(lid); clicked_something=True; break
                        # Rename buttons (pencil)
                        if not clicked_something:
                            for lid,rect in ed._rename_btn_rects.items():
                                if rect.collidepoint(epx,epy):
                                    new_lid=dialog_assign_level_id(
                                        gc.levels.get(lid,{}).get('file','???'),
                                        prompt=f"Rename level '{lid}' to:")
                                    if new_lid and new_lid!=lid:
                                        ok=gc.rename_level_id(lid,new_lid)
                                        if ok:
                                            if ed.browser_level_id==lid: ed.browser_level_id=new_lid
                                            if ed.level.level_id==lid:   ed.level.level_id=new_lid
                                            print(f"[Editor] Renamed '{lid}' → '{new_lid}'")
                                        else: print(f"[Editor] Rename failed")
                                    clicked_something=True; break
                        # Toggle checkboxes (enable/disable)
                        if not clicked_something:
                            for key,rect in ed._toggle_btn_rects.items():
                                if rect.collidepoint(epx,epy):
                                    if key.startswith("off:"):
                                        lid2=key[4:]
                                        gc.toggle_level_in_config(lid2, enable=True)
                                        ed.browser_level_id=lid2
                                        print(f"[Editor] Enabled level '{lid2}'")
                                    else:
                                        gc.toggle_level_in_config(key, enable=False)
                                        print(f"[Editor] Disabled level '{key}'")
                                    clicked_something=True; break
                        # Delete config entries (✕)
                        if not clicked_something:
                            for key,rect in ed._delete_btn_rects.items():
                                if rect.collidepoint(epx,epy):
                                    real_id = key[4:] if key.startswith("off:") else key
                                    import tkinter.messagebox as mb
                                    root2=tk.Tk(); root2.withdraw()
                                    confirm=mb.askyesno("Delete Level",
                                        f"Remove '{real_id}' from game_config.yaml?\n"
                                        f"(The .txt file will NOT be deleted)",parent=root2)
                                    root2.destroy()
                                    if confirm:
                                        gc.delete_level_in_config(real_id)
                                        if ed.browser_level_id==real_id: ed.browser_level_id=None
                                        if ed.level.level_id==real_id:   ed.level.level_id=None
                                        print(f"[Editor] Deleted '{real_id}' from config")
                                    clicked_something=True; break
                        # Unlisted file load
                        if not clicked_something:
                            for fpath,rect in ed._unlisted_btn_rects.items():
                                if rect.collidepoint(epx,epy):
                                    try:
                                        ed.level=Level.load(fpath); ed.undo_stack.clear(); ed.redo_stack.clear()
                                        ed.dirty=False; ed.browser_level_id=None; ed._center_camera()
                                    except Exception as e2: print(f"[Editor] {e2}")
                                    clicked_something=True; break
                        # Unlisted file delete (✕ — deletes the .txt from disk)
                        if not clicked_something:
                            for fpath,rect in ed._unlisted_del_rects.items():
                                if rect.collidepoint(epx,epy):
                                    import tkinter.messagebox as mb
                                    root3=tk.Tk(); root3.withdraw()
                                    fname3=Path(fpath).name
                                    confirm=mb.askyesno("Delete File",
                                        f"Permanently delete '{fname3}' from disk?\nThis cannot be undone.",parent=root3)
                                    root3.destroy()
                                    if confirm:
                                        try:
                                            Path(fpath).unlink()
                                            if ed.level.filename==fpath: ed.level.filename=None
                                            print(f"[Editor] Deleted file: {fpath}")
                                        except Exception as e3: print(f"[Editor] Delete failed: {e3}")
                                    clicked_something=True; break
                        # Assign to Level ID button
                        if not clicked_something and hasattr(ed,'_assign_btn_rect'):
                            if ed._assign_btn_rect.collidepoint(epx,epy):
                                fname=Path(ed.level.filename).name if ed.level.filename else None
                                if fname:
                                    new_lid=dialog_assign_level_id(fname)
                                    if new_lid:
                                        ok=gc.assign_stage_to_level(fname,new_lid)
                                        if ok:
                                            ed.level.level_id=new_lid
                                            ed.browser_level_id=new_lid
                                            print(f"[Editor] Assigned '{fname}' → level '{new_lid}'")
                                        else:
                                            print("[Editor] Assign failed — no game_config.yaml found")
                                clicked_something=True
                        # Physics sliders
                        if not clicked_something:
                            for sl in ed.physics_sliders:
                                if sl.rect.collidepoint(epx,epy):
                                    sl.dragging=True
                                    sl.set_from_fraction((epx-sl.rect.x)/max(1,sl.rect.width))
                                    break

                    # ── Left panel (clickable!) ───────────────────
                    elif epx<LEFT_PANEL_W:
                        clicked_something=False
                        # Speed buttons
                        if ed._spd_minus_r.collidepoint(epx,epy):
                            ed.plat_default_spd=max(20,ed.plat_default_spd-20); clicked_something=True
                        elif ed._spd_plus_r.collidepoint(epx,epy):
                            ed.plat_default_spd=min(600,ed.plat_default_spd+20); clicked_something=True
                        # Tool buttons (eraser, fill, grid)
                        if not clicked_something:
                            for key,rect in ed._tool_rects.items():
                                if rect.collidepoint(epx,epy):
                                    if key=="eraser": ed.eraser_mode=not ed.eraser_mode
                                    elif key=="fill": ed.fill_mode=not ed.fill_mode
                                    elif key=="grid": ed.show_grid=not ed.show_grid
                                    elif key=="platform":
                                        ed.selected_tile='M'; ed.plat_placing=False; ed.eraser_mode=False
                                    clicked_something=True; break
                        # Tile palette buttons
                        if not clicked_something:
                            for char,rect in ed._tile_rects.items():
                                if rect.collidepoint(epx,epy):
                                    ed.selected_tile=char; ed.plat_placing=False; ed.eraser_mode=False
                                    clicked_something=True; break

                    # ── Viewport ──────────────────────────────────
                    elif in_vp:
                        if ed.selected_tile=='M' and not ed.eraser_mode:
                            hi,hh=ed.hit_test_platform_handle(epx,epy)
                            if hi is not None:
                                ed.sel_plat_idx=hi; ed.drag_handle=hh
                                p=ed.level.platforms[hi]; wp=p.start if hh=='start' else p.end
                                hsx,hsy=ed.world_pixel_to_screen(wp[0]+p.width/2,wp[1]+p.height/2)
                                ed.drag_offset=(hsx-epx,hsy-epy)
                            elif not ed.plat_placing:
                                wx,wy=ed.screen_to_world_pixel(epx,epy,snap=True)
                                ed.plat_start_world=(wx,wy); ed.plat_ghost_end=(wx,wy)
                                ed.plat_placing=True; ed.sel_plat_idx=None
                            else:
                                wx,wy=ed.screen_to_world_pixel(epx,epy,snap=True); ed.push_undo()
                                ed.level.platforms.append(PlatformDef(list(ed.plat_start_world),[wx,wy],
                                    speed=ed.plat_default_spd,width=PLAT_DEFAULT_W,height=PLAT_DEFAULT_H))
                                ed.sel_plat_idx=len(ed.level.platforms)-1; ed.plat_placing=False; ed.dirty=True
                        elif ed.hover_cell:
                            r,c=ed.hover_cell
                            brush = ed.get_active_brush()
                            if ed.fill_mode:
                                ed.flood_fill(r,c,brush)
                            else:
                                if not pstarted: ed.push_undo(); pstarted=True
                                ed.paint_char=brush; ed.painting=True
                                ed.last_cell=None; ed.paint_cell(r,c,ed.paint_char)

                elif event.button==3 and in_vp:
                    if ed.selected_tile=='M': ed.plat_placing=False; ed.sel_plat_idx=None
                    elif ed.hover_cell:
                        if not pstarted: ed.push_undo(); pstarted=True
                        r,c=ed.hover_cell; ed.paint_char=' '; ed.painting=True
                        ed.last_cell=None; ed.paint_cell(r,c,' ')
                elif event.button==2:
                    ed.panning=True; ed.pan_start=event.pos; ed.pan_cam=(ed.cam_x,ed.cam_y)

            elif event.type==pygame.MOUSEBUTTONUP:
                if event.button in (1,3):
                    ed.painting=False; ed.last_cell=None; pstarted=False; ed.drag_handle=None
                    ed._scrollbar_dragging=False
                    ed._hscroll_dragging=False
                    for sl in ed.physics_sliders: sl.dragging=False
                elif event.button==2: ed.panning=False

            elif event.type==pygame.MOUSEMOTION:
                for sl in ed.physics_sliders:
                    if sl.dragging: sl.set_from_fraction((event.pos[0]-sl.rect.x)/max(1,sl.rect.width))
                # Vertical right-panel scrollbar drag
                if ed._scrollbar_dragging:
                    dy=event.pos[1]-ed._scrollbar_drag_start_y
                    rh2=WINDOW_H-TOOLBAR_H-STATUS_H
                    thumb_h=ed._scrollbar_thumb.height if ed._scrollbar_thumb.height>0 else 40
                    scale=ed.right_max_scroll/max(1,rh2-thumb_h)
                    ed.right_scroll=max(0,min(ed.right_max_scroll,ed._scrollbar_drag_scroll_start+int(dy*scale)))
                # Horizontal viewport scrollbar drag
                if ed._hscroll_dragging:
                    dx=event.pos[0]-ed._hscroll_drag_start_x
                    vw2=WINDOW_W-LEFT_PANEL_W-RIGHT_PANEL_W
                    tw=ed._hscroll_thumb.width if ed._hscroll_thumb.width>0 else 40
                    max_cx=max(0,ed.level.cols*TILE_SIZE*ed.zoom-vw2)
                    scale_h=max_cx/max(1,vw2-tw)
                    ed.cam_x=max(0.0,min(float(max_cx),ed._hscroll_drag_cam_start+dx*scale_h))
                if ed.selected_tile=='M':
                    mx2,my2=event.pos
                    if ed.drag_handle is not None and ed.sel_plat_idx is not None:
                        wx,wy=ed.screen_to_world_pixel(mx2+ed.drag_offset[0],my2+ed.drag_offset[1],snap=True)
                        p=ed.level.platforms[ed.sel_plat_idx]
                        wp=[wx-p.width/2,wy-p.height/2]
                        if ed.drag_handle=='start': p.start=wp
                        else: p.end=wp
                        ed.dirty=True
                    elif ed.plat_placing and in_vp:
                        ed.plat_ghost_end=ed.screen_to_world_pixel(mx2,my2,snap=True)
                if ed.painting and in_vp and ed.hover_cell:
                    ed.paint_cell(ed.hover_cell[0],ed.hover_cell[1],ed.paint_char)
                if ed.panning:
                    ed.cam_x=ed.pan_cam[0]-(event.pos[0]-ed.pan_start[0])
                    ed.cam_y=ed.pan_cam[1]-(event.pos[1]-ed.pan_start[1])

            elif event.type==pygame.KEYDOWN:
                mods=pygame.key.get_mods(); ctrl=mods&pygame.KMOD_CTRL
                if ctrl and event.key==pygame.K_s:
                    p=ed.level.filename or dialog_save()
                    if p: ed.level.save(p); ed.dirty=False; print(f"[Editor] Saved: {p}")
                elif ctrl and event.key==pygame.K_p:
                    launch_playtest(ed)
                elif ctrl and event.key==pygame.K_o:
                    p=dialog_open()
                    if p:
                        try: ed.level=Level.load(p); ed.undo_stack.clear(); ed.redo_stack.clear(); ed.dirty=False; ed._center_camera()
                        except Exception as e2: print(f"[Editor] {e2}")
                elif ctrl and event.key==pygame.K_z:
                    if mods&pygame.KMOD_SHIFT: ed.redo()
                    else: ed.undo()
                elif ctrl and event.key==pygame.K_n:
                    rows,cols=dialog_new_size()
                    if rows and cols:
                        ed.level=Level(rows,cols)
                        for r3 in range(rows): ed.level.set(r3,0,'#'); ed.level.set(r3,cols-1,'#')
                        for c3 in range(cols): ed.level.set(0,c3,'#'); ed.level.set(rows-1,c3,'#')
                        ed.level.set(rows-2,2,'P'); ed.undo_stack.clear(); ed.redo_stack.clear()
                        ed.dirty=False; ed._center_camera()
                elif ctrl and event.key==pygame.K_x: ed.clear_all()
                elif event.key==pygame.K_HOME: ed.reset_view()
                elif event.key==pygame.K_g: ed.show_grid=not ed.show_grid
                elif event.key==pygame.K_f: ed.fill_mode=not ed.fill_mode
                elif event.key==pygame.K_x and not ctrl: ed.eraser_mode=not ed.eraser_mode
                elif event.key==pygame.K_m: ed.selected_tile='M'; ed.plat_placing=False; ed.eraser_mode=False
                elif event.key==pygame.K_DELETE:
                    if ed.selected_tile=='M' and ed.sel_plat_idx is not None:
                        if 0<=ed.sel_plat_idx<len(ed.level.platforms):
                            ed.push_undo(); ed.level.platforms.pop(ed.sel_plat_idx)
                            ed.sel_plat_idx=None; ed.dirty=True
                elif event.key==pygame.K_ESCAPE:
                    if ed.plat_placing: ed.plat_placing=False
                    elif ed.eraser_mode: ed.eraser_mode=False
                else:
                    for i,k in enumerate([pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,pygame.K_5,
                                          pygame.K_6,pygame.K_7,pygame.K_8,pygame.K_9,pygame.K_0]):
                        if event.key==k and i<len(TILES):
                            ed.selected_tile=TILES[i][0]; ed.eraser_mode=False

        # ── Draw ─────────────────────────────────────────────────
        mpos=pygame.mouse.get_pos(); screen.fill(UI_BG)
        draw_viewport(screen,ed,font,sfont)       # draws clipped
        draw_toolbar(screen,ed,sfont,btns,mpos)    # over viewport
        draw_left_panel(screen,ed,font,sfont)      # over viewport
        draw_right_panel(screen,ed,sfont,mpos)     # over viewport
        draw_status(screen,ed,sfont)               # over viewport
        pygame.display.flip()
    pygame.quit()

if __name__=="__main__":
    main()