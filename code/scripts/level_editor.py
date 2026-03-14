#!/usr/bin/env python3
"""
PEAK Level Editor  v3.1  —  All v3 features + bug fixes
────────────────────────────────────────────────────────
Fixes from v3.0:
  • Toolbar buttons now respond to mouse clicks (were draw-only before)
  • Ctrl+Y added as redo shortcut
  • Grid resize via Ctrl+R / F5
  • Unsaved-changes confirmation on Quit / New / Open / Load
  • Keyboard shortcuts routed through shared action dispatcher
  • Improved fallback tile rendering when game objects unavailable
"""

import argparse
import fnmatch
import os, sys, math, copy, re, subprocess, string
import pygame, pygame.freetype
from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None

# ─── Constants ────────────────────────────────────────────────────
WINDOW_W, WINDOW_H = 1600, 920
TILE_SIZE   = 32
MIN_ZOOM    = 0.25; MAX_ZOOM = 4.0
UNDO_LIMIT  = 200
LEFT_PANEL_W  = 230; RIGHT_PANEL_W = 300
TOOLBAR_H = 52; STATUS_H = 28; HSB_H = 12; SB_W = 10
DEFAULT_ROWS = 34; DEFAULT_COLS = 75
DEFAULT_GAME = "platformer"
MOVING_PLATFORM_BRUSH = "__moving_platform__"
DISABLED_LEVELS_KEY = "disabled_levels"
LEGACY_DISABLED_LEVELS_KEY = "editor_disabled_levels"

# ─── Game object imports (optional) ──────────────────────────────
try:
    from code.games.modules.Objects.Tile import Tile, create_tile
    from code.games.modules.Objects.Coin import Coin
    from code.games.modules.Objects.Enemy import Enemy
    from code.games.modules.Objects.Goal import Goal
    from code.games.modules.Objects.Ladder import Ladder
    from code.games.modules.Objects.QuestionBlock import QuestionBlock
    from code.games.modules.Objects.GameObject import GameObject
    from code.games.modules.Parameters.Map_parameters import (
        TILE_GROUND, TILE_PLATFORM, TILE_SPIKE,
        COLOR_GROUND, COLOR_PLATFORM, COLOR_SPIKE, COLOR_GOAL)
    USE_REAL_OBJECTS = True
except ImportError:
    USE_REAL_OBJECTS = False

# ─── Tile definitions ─────────────────────────────────────────────
GAME_LABELS = {
    "platformer": "Mario / Platformer",
    "megaman": "Mega Man",
}

GAME_FILE_GLOBS = {
    "platformer": ["world*.txt", "stage*.txt", "platform*.txt", "level*.txt", "testlevel*.txt"],
    "megaman": ["mm_*.txt", "mega*.txt"],
}

TILE_LIBRARY = [
    (' ', 'Air',            (30, 35, 50),    (80, 90, 120)),
    ('#', 'Ground',         (80, 60, 40),    (220, 200, 160)),
    ('=', 'Platform',       (60, 90, 60),    (180, 230, 140)),
    ('^', 'Spike',          (200, 60, 60),   (255, 200, 180)),
    ('O', 'Pit',            (15, 8, 12),     (200, 60, 60)),
    ('?', 'QBlock (coin)',  (220, 180, 40),  (80, 60, 0)),
    ('>', 'QBlock (star)',  (255, 220, 80),  (80, 60, 0)),
    ('<', 'QBlock (mush)',  (200, 80, 80),   (255, 240, 240)),
    ('F', 'QBlock (fire)',  (255, 120, 30),  (80, 30, 0)),
    ('L', 'QBlock (life)',  (90, 220, 120),  (0, 60, 20)),
    ('C', 'Coin',           (255, 215, 0),   (80, 60, 0)),
    ('E', 'Enemy',          (180, 60, 180),  (255, 220, 255)),
    ('H', 'Ladder',         (120, 215, 255), (10, 60, 90)),
    ('M', 'Met Enemy',      (80, 140, 255),  (235, 230, 120)),
    ('B', 'Bat Enemy',      (190, 120, 255), (255, 220, 255)),
    ('G', 'Goal',           (60, 200, 100),  (0, 60, 20)),
    ('D', 'Boss Door',      (255, 90, 140),  (60, 10, 30)),
    ('P', 'Player Start',   (60, 160, 255),  (0, 30, 120)),
]
TILE_BY_CHAR = {t[0]: t for t in TILE_LIBRARY}
TILE_BY_CHAR['.'] = TILE_BY_CHAR[' ']
GAME_TILE_ORDER = {
    "platformer": [' ', '#', '=', '^', 'O', '?', '>', '<', 'F', 'L', 'C', 'E', 'G', 'P'],
    "megaman": [' ', '#', '=', '^', 'H', 'M', 'B', 'G', 'D', 'P'],
}
SOLID_CHARS  = {'#', '=', '?', '>', '<', 'F', 'L'}

def tile_entries_for(game):
    return [TILE_BY_CHAR[ch] for ch in GAME_TILE_ORDER.get(game, GAME_TILE_ORDER[DEFAULT_GAME])]

# ─── Moving Platform ─────────────────────────────────────────────
PLAT_DEFAULT_W = TILE_SIZE * 3; PLAT_DEFAULT_H = TILE_SIZE // 2; PLAT_DEFAULT_SPD = 80.0
PLAT_BODY_COL = (205,133,63); PLAT_HIGH_COL = (230,165,90)
PLAT_PATH_COL = (255,200,80); PLAT_SEL_COL  = (255,255,100)
HANDLE_START = (80,220,80); HANDLE_END = (220,80,80)
HANDLE_R = 9; HANDLE_HIT_R = 14

class PlatformDef:
    __slots__ = ('start','end','speed','width','height')
    def __init__(s, start, end, speed=PLAT_DEFAULT_SPD, width=PLAT_DEFAULT_W, height=PLAT_DEFAULT_H):
        s.start=list(start); s.end=list(end); s.speed=float(speed)
        s.width=int(width); s.height=int(height)
    def to_dict(s):
        return {'start':[int(s.start[0]),int(s.start[1])],
                'end':[int(s.end[0]),int(s.end[1])],
                'speed':s.speed,'width':s.width,'height':s.height}
    @classmethod
    def from_dict(cls, d):
        return cls(d['start'],d['end'],d.get('speed',PLAT_DEFAULT_SPD),
                   d.get('width',PLAT_DEFAULT_W),d.get('height',PLAT_DEFAULT_H))

# ─── UI Colors ───────────────────────────────────────────────────
UI_BG=(18,8,12); UI_PANEL=(26,12,18); UI_PANEL2=(36,18,26)
UI_BORDER=(80,36,52); UI_SELECT=(120,22,40); UI_SELECT_DIM=(70,14,26)
UI_TEXT=(245,225,210); UI_SUBTEXT=(180,140,155)
UI_TOOLBAR=(14,6,10); UI_STATUS=(12,5,8)
UI_BTN=(44,22,32); UI_BTN_HOVER=(64,32,46); UI_BTN_ACT=(110,24,42)
UI_ACCENT=(255,90,140); UI_ACCENT2=(90,215,225)
UI_WARN=(215,150,60); UI_DANGER=(215,52,62); UI_PIT_DIAG=(170,44,60)
GRID_COLOR=(36,16,24); GRID_BOLD=(62,28,42)

# ═══════════════════════════════════════════════════════════════════
# DIALOG SYSTEM (native pygame — no tkinter)
# ═══════════════════════════════════════════════════════════════════
class _DlgBase:
    W=540; H=320; PAD=18
    def __init__(self, screen, sf):
        self.screen=screen; self.sf=sf; self.done=False; self.result=None
        sw,sh=screen.get_size()
        self.rect=pygame.Rect((sw-self.W)//2,(sh-self.H)//2,self.W,self.H)
    def _ov(self):
        o=pygame.Surface(self.screen.get_size(),pygame.SRCALPHA); o.fill((0,0,0,170)); self.screen.blit(o,(0,0))
    def _card(self, title):
        r=self.rect
        pygame.draw.rect(self.screen,(22,10,16),r,border_radius=8)
        pygame.draw.rect(self.screen,UI_BORDER,r,2,border_radius=8)
        pygame.draw.rect(self.screen,UI_TOOLBAR,pygame.Rect(r.x,r.y,r.w,34),border_radius=8)
        pygame.draw.line(self.screen,UI_BORDER,(r.x,r.y+34),(r.right,r.y+34),1)
        try: self.sf.render_to(self.screen,(r.x+self.PAD,r.y+9),title,fgcolor=UI_ACCENT,size=13)
        except: pass
    def _btn(self, label, rect, mp, accent=False):
        h=rect.collidepoint(mp)
        bg=(UI_ACCENT if h else UI_BTN_ACT) if accent else (UI_BTN_HOVER if h else UI_BTN)
        fg=UI_BG if (accent and h) else UI_TEXT
        pygame.draw.rect(self.screen,bg,rect,border_radius=5)
        pygame.draw.rect(self.screen,UI_BORDER,rect,1,border_radius=5)
        try:
            b=self.sf.get_rect(label,size=11)
            self.sf.render_to(self.screen,(rect.x+(rect.w-b.width)//2,rect.y+(rect.h-b.height)//2),label,fgcolor=fg,size=11)
        except: pass
    def _txt(self, text, x, y, c=None, sz=11):
        try: self.sf.render_to(self.screen,(x,y),text,fgcolor=c or UI_TEXT,size=sz)
        except: pass
    def run(self):
        clk=pygame.time.Clock()
        while not self.done:
            mp=pygame.mouse.get_pos()
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT: self.done=True
                else: self.handle(ev,mp)
            self._ov(); self.draw(mp); pygame.display.flip(); clk.tick(60)
        return self.result
    def handle(self,ev,mp): pass
    def draw(self,mp): pass

class ConfirmDialog(_DlgBase):
    H=170
    def __init__(self, screen, sf, title, msg):
        super().__init__(screen,sf); self.title=title; self.msg=msg
        r=self.rect; bw=110; bh=32; by=r.y+self.H-50
        self.y_btn=pygame.Rect(r.right-bw*3-self.PAD*3,by,bw,bh)
        self.n_btn=pygame.Rect(r.right-bw*2-self.PAD*2,by,bw,bh)
        self.c_btn=pygame.Rect(r.right-bw-self.PAD,by,bw,bh)
    def handle(self,ev,mp):
        if ev.type==pygame.KEYDOWN:
            if ev.key in (pygame.K_RETURN,pygame.K_y): self.result="yes"; self.done=True
            elif ev.key==pygame.K_n: self.result="no"; self.done=True
            elif ev.key==pygame.K_ESCAPE: self.result="cancel"; self.done=True
        elif ev.type==pygame.MOUSEBUTTONDOWN:
            if self.y_btn.collidepoint(mp): self.result="yes"; self.done=True
            elif self.n_btn.collidepoint(mp): self.result="no"; self.done=True
            elif self.c_btn.collidepoint(mp): self.result="cancel"; self.done=True
    def draw(self,mp):
        self._card(self.title); self._txt(self.msg,self.rect.x+self.PAD,self.rect.y+55,UI_TEXT,12)
        self._btn("Yes",self.y_btn,mp,True); self._btn("No",self.n_btn,mp); self._btn("Cancel",self.c_btn,mp)

class TextInputDialog(_DlgBase):
    H=200
    def __init__(self, screen, sf, title, prompt, initial=""):
        super().__init__(screen,sf); self.title=title; self.prompt=prompt
        self.text=initial; self.cur=len(initial); self.blink=0
        r=self.rect; bw=110; bh=32; by=r.y+self.H-50
        self.ok=pygame.Rect(r.right-bw*2-self.PAD*2,by,bw,bh)
        self.cancel=pygame.Rect(r.right-bw-self.PAD,by,bw,bh)
        self.ir=pygame.Rect(r.x+self.PAD,r.y+90,r.w-self.PAD*2,34)
    def handle(self,ev,mp):
        if ev.type==pygame.KEYDOWN:
            if ev.key==pygame.K_RETURN: self.result=self.text.strip() or None; self.done=True
            elif ev.key==pygame.K_ESCAPE: self.done=True
            elif ev.key==pygame.K_BACKSPACE:
                if self.cur>0: self.text=self.text[:self.cur-1]+self.text[self.cur:]; self.cur-=1
            elif ev.key==pygame.K_DELETE: self.text=self.text[:self.cur]+self.text[self.cur+1:]
            elif ev.key==pygame.K_LEFT: self.cur=max(0,self.cur-1)
            elif ev.key==pygame.K_RIGHT: self.cur=min(len(self.text),self.cur+1)
            elif ev.key==pygame.K_HOME: self.cur=0
            elif ev.key==pygame.K_END: self.cur=len(self.text)
            elif ev.unicode and ev.unicode in string.printable:
                self.text=self.text[:self.cur]+ev.unicode+self.text[self.cur:]; self.cur+=1
        elif ev.type==pygame.MOUSEBUTTONDOWN:
            if self.ok.collidepoint(mp): self.result=self.text.strip() or None; self.done=True
            elif self.cancel.collidepoint(mp): self.done=True
    def draw(self,mp):
        self._card(self.title); self._txt(self.prompt,self.rect.x+self.PAD,self.rect.y+50,UI_SUBTEXT,11)
        ir=self.ir; pygame.draw.rect(self.screen,UI_PANEL2,ir,border_radius=4)
        pygame.draw.rect(self.screen,UI_ACCENT,ir,2,border_radius=4)
        try:
            self.sf.render_to(self.screen,(ir.x+8,ir.y+8),self.text,fgcolor=UI_TEXT,size=12)
            self.blink=(self.blink+1)%60
            if self.blink<30:
                cx=ir.x+8
                if self.cur>0: cx+=self.sf.get_rect(self.text[:self.cur],size=12).width
                pygame.draw.line(self.screen,UI_ACCENT,(cx,ir.y+6),(cx,ir.y+ir.h-6),2)
        except: pass
        self._btn("OK",self.ok,mp,True); self._btn("Cancel",self.cancel,mp)

class IntInputDialog(_DlgBase):
    H=210
    def __init__(self, screen, sf, title, prompt, initial=10, mn=1, mx=500):
        super().__init__(screen,sf); self.title=title; self.prompt=prompt
        self.val=int(initial); self.mn=int(mn); self.mx=int(mx)
        r=self.rect; cy=r.y+115; bh=36; bw=110
        self.minus=pygame.Rect(r.x+self.PAD,cy,42,bh)
        self.plus=pygame.Rect(r.x+self.PAD+42+80,cy,42,bh)
        self.vr=pygame.Rect(r.x+self.PAD+46,cy,76,bh)
        by=r.y+self.H-50
        self.ok=pygame.Rect(r.right-bw*2-self.PAD*2,by,bw,bh-4)
        self.cancel=pygame.Rect(r.right-bw-self.PAD,by,bw,bh-4)
    def handle(self,ev,mp):
        if ev.type==pygame.MOUSEBUTTONDOWN:
            if self.minus.collidepoint(mp): self.val=max(self.mn,self.val-1)
            elif self.plus.collidepoint(mp): self.val=min(self.mx,self.val+1)
            elif self.ok.collidepoint(mp): self.result=self.val; self.done=True
            elif self.cancel.collidepoint(mp): self.done=True
        elif ev.type==pygame.MOUSEWHEEL: self.val=max(self.mn,min(self.mx,self.val+ev.y))
        elif ev.type==pygame.KEYDOWN:
            if ev.key==pygame.K_RETURN: self.result=self.val; self.done=True
            elif ev.key==pygame.K_ESCAPE: self.done=True
            elif ev.key==pygame.K_UP: self.val=min(self.mx,self.val+1)
            elif ev.key==pygame.K_DOWN: self.val=max(self.mn,self.val-1)
    def draw(self,mp):
        self._card(self.title); self._txt(self.prompt,self.rect.x+self.PAD,self.rect.y+50,UI_SUBTEXT,11)
        self._btn("-",self.minus,mp)
        pygame.draw.rect(self.screen,UI_PANEL2,self.vr,border_radius=4)
        pygame.draw.rect(self.screen,UI_BORDER,self.vr,1,border_radius=4)
        try:
            vs=str(self.val); b=self.sf.get_rect(vs,size=14)
            self.sf.render_to(self.screen,(self.vr.x+(self.vr.w-b.width)//2,self.vr.y+(self.vr.h-b.height)//2),vs,fgcolor=UI_TEXT,size=14)
        except: pass
        self._btn("+",self.plus,mp); self._btn("OK",self.ok,mp,True); self._btn("Cancel",self.cancel,mp)

class FileBrowserDialog(_DlgBase):
    W=600; H=460
    def __init__(self, screen, sf, title, mode="open", start_dir=None, default_name="level.txt"):
        super().__init__(screen,sf); self.title=title; self.mode=mode
        self.current_dir=Path(start_dir or Path.cwd()).resolve()
        self.files=[]; self.sel=-1; self.scroll=0
        self.sn=default_name; self.sc=len(default_name); self.blink=0
        r=self.rect; bw=110; bh=32; by=r.y+self.H-50
        self.ok=pygame.Rect(r.right-bw*2-self.PAD*2,by,bw,bh)
        self.cancel=pygame.Rect(r.right-bw-self.PAD,by,bw,bh)
        self.lr=pygame.Rect(r.x+self.PAD,r.y+80,r.w-self.PAD*2,self.H-80-(70 if mode=="save" else 55))
        self.nr=pygame.Rect(r.x+self.PAD,r.y+self.H-100,r.w-self.PAD*2,30)
        self._refresh()
    def _refresh(self):
        self.files=[]
        try:
            entries=sorted(self.current_dir.iterdir(),key=lambda p:(p.is_file(),p.name.lower()))
            for e in entries:
                if e.is_dir(): self.files.append(('dir',e))
                elif e.suffix.lower() in ('.txt',''): self.files.append(('file',e))
            self.files.insert(0,('up',self.current_dir.parent))
        except: pass
        self.scroll=0; self.sel=-1
    def handle(self,ev,mp):
        if ev.type==pygame.MOUSEWHEEL: self.scroll=max(0,min(len(self.files)-1,self.scroll-ev.y))
        elif ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1:
            if self.lr.collidepoint(mp):
                idx=self.scroll+(mp[1]-self.lr.y)//26
                if 0<=idx<len(self.files):
                    k,p=self.files[idx]
                    if k in ('dir','up'): self.current_dir=p.resolve(); self._refresh()
                    else: self.sel=idx; self.sn=p.name; self.sc=len(self.sn)
            elif self.ok.collidepoint(mp): self._confirm()
            elif self.cancel.collidepoint(mp): self.done=True
        elif ev.type==pygame.KEYDOWN:
            if self.mode=="save": self._sk(ev)
            elif ev.key==pygame.K_RETURN: self._confirm()
            elif ev.key==pygame.K_ESCAPE: self.done=True
    def _sk(self,ev):
        if ev.key==pygame.K_RETURN: self._confirm()
        elif ev.key==pygame.K_ESCAPE: self.done=True
        elif ev.key==pygame.K_BACKSPACE:
            if self.sc>0: self.sn=self.sn[:self.sc-1]+self.sn[self.sc:]; self.sc-=1
        elif ev.key==pygame.K_DELETE: self.sn=self.sn[:self.sc]+self.sn[self.sc+1:]
        elif ev.key==pygame.K_LEFT: self.sc=max(0,self.sc-1)
        elif ev.key==pygame.K_RIGHT: self.sc=min(len(self.sn),self.sc+1)
        elif ev.unicode and ev.unicode in string.printable:
            self.sn=self.sn[:self.sc]+ev.unicode+self.sn[self.sc:]; self.sc+=1
    def _confirm(self):
        if self.mode=="open":
            if 0<=self.sel<len(self.files) and self.files[self.sel][0]=='file':
                self.result=str(self.files[self.sel][1]); self.done=True
        else:
            n=self.sn.strip()
            if n:
                if not n.endswith('.txt'): n+='.txt'
                self.result=str(self.current_dir/n); self.done=True
    def draw(self,mp):
        self._card(self.title); r=self.rect
        ps=str(self.current_dir)
        if len(ps)>60: ps='...'+ps[-57:]
        self._txt(ps,r.x+self.PAD,r.y+44,UI_SUBTEXT,9)
        lr=self.lr; pygame.draw.rect(self.screen,UI_PANEL2,lr,border_radius=4)
        pygame.draw.rect(self.screen,UI_BORDER,lr,1,border_radius=4)
        self.screen.set_clip(lr); vis=lr.h//26
        for i,(k,p) in enumerate(self.files[self.scroll:self.scroll+vis]):
            ri=i+self.scroll; ry=lr.y+i*26; is_sel=ri==self.sel
            rr=pygame.Rect(lr.x+2,ry+1,lr.w-4,24)
            if is_sel: pygame.draw.rect(self.screen,UI_SELECT,rr,border_radius=3)
            elif rr.collidepoint(mp): pygame.draw.rect(self.screen,UI_BTN_HOVER,rr,border_radius=3)
            nm='..' if k=='up' else p.name
            fg=UI_SUBTEXT if k in ('dir','up') else (UI_TEXT if is_sel else UI_SUBTEXT)
            try: self.sf.render_to(self.screen,(lr.x+8,ry+6),nm,fgcolor=fg,size=10)
            except: pass
        self.screen.set_clip(None)
        if self.mode=="save":
            self._txt("File name:",r.x+self.PAD,self.nr.y-16,UI_SUBTEXT,10)
            pygame.draw.rect(self.screen,UI_PANEL2,self.nr,border_radius=4)
            pygame.draw.rect(self.screen,UI_ACCENT,self.nr,1,border_radius=4)
            self.blink=(self.blink+1)%60
            try:
                self.sf.render_to(self.screen,(self.nr.x+6,self.nr.y+8),self.sn,fgcolor=UI_TEXT,size=11)
                if self.blink<30:
                    cx=self.nr.x+6
                    if self.sc>0: cx+=self.sf.get_rect(self.sn[:self.sc],size=11).width
                    pygame.draw.line(self.screen,UI_ACCENT,(cx,self.nr.y+4),(cx,self.nr.y+self.nr.h-4),2)
            except: pass
        self._btn("OK" if self.mode=="save" else "Open",self.ok,mp,True)
        self._btn("Cancel",self.cancel,mp)

# ─── Dialog helpers ───────────────────────────────────────────────
def dlg_open(scr,sf,sd=None): return FileBrowserDialog(scr,sf,"Open Level","open",sd).run()
def dlg_save(scr,sf,dn="level.txt",sd=None): return FileBrowserDialog(scr,sf,"Save Level","save",sd,dn).run()
def dlg_new(scr,sf):
    r=IntInputDialog(scr,sf,"New Level","Rows:",DEFAULT_ROWS,5,200).run()
    if r is None: return None,None
    c=IntInputDialog(scr,sf,"New Level","Columns:",DEFAULT_COLS,10,400).run()
    return (r,c) if r and c else (None,None)
def dlg_resize(scr,sf,cr,cc):
    r=IntInputDialog(scr,sf,"Resize Grid",f"Rows (now {cr}):",cr,5,200).run()
    if r is None: return None,None
    c=IntInputDialog(scr,sf,"Resize Grid",f"Cols (now {cc}):",cc,10,400).run()
    return (r,c) if r and c else (None,None)
def dlg_assign(scr,sf,fn,prompt=None):
    msg=prompt or f"Level ID for '{Path(fn).name}'  (e.g. 1-5, World1):"
    return TextInputDialog(scr,sf,"Assign / Rename Level ID",msg).run()
def dlg_unsaved(scr,sf):
    return ConfirmDialog(scr,sf,"Unsaved Changes","Save before continuing?").run()

# ═══════════════════════════════════════════════════════════════════
# GAME CONFIG (reads/writes game_config.yaml)
# ═══════════════════════════════════════════════════════════════════
class GameConfig:
    def __init__(self, game=DEFAULT_GAME):
        self.game = game if game in GAME_LABELS else DEFAULT_GAME
        self.game_label = GAME_LABELS.get(self.game, GAME_LABELS[DEFAULT_GAME])
        self.yaml_data = {}
        self.levels = {}
        self.level_ids = []
        self.disabled_levels = {}
        self.physics = {}
        self.levels_dir = None
        self.config_path = None
        self._load()

    def _root_cfg(self, data, create=False):
        if self.game == "platformer":
            return data
        if create and self.game not in data:
            data[self.game] = {}
        return data.get(self.game, {})

    def _default_level_entry(self, filename):
        if self.game == "megaman":
            return {
                "file": filename,
                "background_color": [120, 160, 255],
            }
        return {
            "file": filename,
            "time_limit": 300,
            "background_color": [0, 0, 0],
        }

    def _write_yaml(self, data):
        if not self.config_path or yaml is None:
            return False
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            return True
        except Exception:
            return False

    def _load(self):
        if yaml is None:
            return
        cwd = Path.cwd()
        sd = Path(__file__).resolve().parent
        for p in [
            cwd / "game_config.yaml",
            cwd / "code" / "games" / "platformer" / "game_config.yaml",
            cwd / "code" / "games" / "game_config.yaml",
            sd / "game_config.yaml",
            sd.parent / "game_config.yaml",
            sd.parent.parent / "game_config.yaml",
        ]:
            if p.exists():
                self.config_path = p
                break
        if not self.config_path:
            return
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.yaml_data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[GameConfig] {e}")
            return

        cd = self.config_path.parent
        for d in [cd / "levels", cd, cwd / "levels", cwd]:
            if d.exists() and list(d.glob("*.txt")):
                self.levels_dir = d
                break

        root_cfg = self._root_cfg(self.yaml_data)
        raw = root_cfg.get('levels', {}) or {}
        for lid, lcfg in raw.items():
            if isinstance(lcfg, dict):
                self.levels[str(lid)] = lcfg
                self.level_ids.append(str(lid))

        disabled = root_cfg.get(DISABLED_LEVELS_KEY)
        if not isinstance(disabled, dict):
            disabled = root_cfg.get(LEGACY_DISABLED_LEVELS_KEY, {}) or {}
        if isinstance(disabled, dict):
            self.disabled_levels = {str(lid): cfg for lid, cfg in disabled.items() if isinstance(cfg, dict)}

        self.physics = copy.deepcopy(root_cfg.get('physics', {}) or {})

    def _levels_dict(self, data, create=False):
        root_cfg = self._root_cfg(data, create=create)
        levels = root_cfg.get('levels')
        if not isinstance(levels, dict):
            if create:
                root_cfg['levels'] = {}
                return root_cfg['levels']
            return {}
        return levels

    def _disabled_levels_dict(self, data, create=False):
        root_cfg = self._root_cfg(data, create=create)
        levels = root_cfg.get(DISABLED_LEVELS_KEY)
        if not isinstance(levels, dict):
            if create:
                root_cfg[DISABLED_LEVELS_KEY] = {}
                return root_cfg[DISABLED_LEVELS_KEY]
            legacy = root_cfg.get(LEGACY_DISABLED_LEVELS_KEY, {})
            return legacy if isinstance(legacy, dict) else {}
        return levels

    def is_file_for_game(self, path):
        name = Path(path).name.lower()
        known = {Path(cfg.get('file', '')).name.lower() for cfg in list(self.levels.values()) + list(self.disabled_levels.values())}
        if name in known:
            return True
        for pattern in GAME_FILE_GLOBS.get(self.game, []):
            if fnmatch.fnmatch(name, pattern.lower()):
                return True
        return False

    def get_level_file_path(self, lid):
        lc = self.levels.get(lid) or self.disabled_levels.get(lid)
        fn = lc.get('file', '') if lc else ''
        if not fn:
            return None
        for b in ([self.levels_dir] if self.levels_dir else []) + ([self.config_path.parent] if self.config_path else []):
            p = Path(b) / fn
            if p.exists():
                return str(p)
        return str(Path(fn).resolve()) if Path(fn).exists() else None

    def get_all_level_files(self, extra_path=None):
        fs=set(); seen=set()
        def _a(p):
            try: rp=Path(p).resolve()
            except: rp=Path(p)
            if rp not in seen and rp.exists() and rp.is_dir():
                seen.add(rp)
                for f in rp.glob("*.txt"):
                    if self.is_file_for_game(f):
                        fs.add(f)
        if self.levels_dir: _a(self.levels_dir)
        if self.config_path: _a(self.config_path.parent)
        if extra_path:
            extra = Path(extra_path)
            _a(extra.parent)
            if extra.exists():
                fs.add(extra.resolve())
        return sorted(fs,key=lambda p:p.name)
    # ── Key-matching helpers ─────────────────────────────────────
    # YAML keys can appear unquoted, single-quoted, or double-quoted:
    #   Long Level:    "1-2":    'Mario1-2':
    # All text-manipulation methods must handle every form.
    @staticmethod
    def _is_key(stripped, key):
        """Does this lstripped line start with a YAML mapping key `key:`?"""
        for pat in [f'"{key}":', f"'{key}':", f'{key}:']:
            if stripped.startswith(pat): return True
        return False
    @staticmethod
    def _is_commented_key(stripped, key):
        """Does this lstripped line start with a commented-out key `# key:`?"""
        if not stripped.startswith('#'): return False
        inner = stripped.lstrip('#').lstrip()
        return GameConfig._is_key(inner, key)
    @staticmethod
    def _key_re(key):
        """Regex fragment matching any quoting of a YAML key."""
        ek = re.escape(key)
        return rf'(?:"{ek}"|' + rf"'{ek}'|" + rf'{ek})'

    # Known YAML property keys that are sub-fields of a level entry, not level IDs
    _LEVEL_SUBKEYS = {
        'file', 'time_limit', 'background_color', 'time', 'physics',
        'gravity', 'friction', 'max_fall_speed', 'max_run_speed',
        'dynamics', 'enemies', 'coins', 'powerups', 'moving_platforms',
        'player', 'render', 'reward', 'observation', 'seed',
    }

    def get_commented_levels(self):
        return dict(self.disabled_levels)

    def _get_commented_file(self, lid):
        if not self.config_path: return None
        try: text = self.config_path.read_text(encoding='utf-8')
        except: return None
        kp = self._key_re(lid)
        pat = re.compile(r'#\s*' + kp + r':\s*\n((?:\s*#[^\n]*\n)*)', re.MULTILINE)
        m = pat.search(text)
        if not m: return None
        # file value can be quoted or unquoted
        fm = re.search(r'#\s*file:\s*["\']?([^"\'\s]+)["\']?', m.group(1))
        return fm.group(1) if fm else None

    def toggle_level_in_config(self, lid, enable):
        if not self.config_path or yaml is None:
            return False
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return False
        levels = self._levels_dict(data, create=True)
        disabled = self._disabled_levels_dict(data, create=True)
        if enable:
            if lid in disabled:
                levels[lid] = disabled.pop(lid)
        else:
            if lid in levels:
                disabled[lid] = levels.pop(lid)
        if not disabled:
            self._root_cfg(data, create=True).pop(DISABLED_LEVELS_KEY, None)
        if self._write_yaml(data):
            self.yaml_data = {}; self.levels = {}; self.level_ids = []; self.disabled_levels = {}; self._load(); return True
        return False

    def reorder_levels(self, new_order):
        if not self.config_path or yaml is None: return False
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f: data = yaml.safe_load(f) or {}
        except: return False
        ol = self._levels_dict(data, create=True); nv = {}
        for lid in new_order:
            if lid in ol: nv[lid] = ol[lid]
        for lid, v in ol.items():
            if lid not in nv: nv[lid] = v
        self._root_cfg(data, create=True)['levels'] = nv
        if self._write_yaml(data):
            self.yaml_data = {}; self.levels = {}; self.level_ids = []; self.disabled_levels = {}; self._load(); return True
        return False

    def assign_stage_to_level(self, filename, lid):
        if not self.config_path or yaml is None:
            return False
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return False
        levels = self._levels_dict(data, create=True)
        entry = dict(levels.get(lid, self._default_level_entry(filename)))
        entry['file'] = filename
        levels[lid] = entry
        disabled = self._disabled_levels_dict(data, create=True)
        disabled.pop(lid, None)
        if not disabled:
            self._root_cfg(data, create=True).pop(DISABLED_LEVELS_KEY, None)
        if self._write_yaml(data):
            self.yaml_data = {}; self.levels = {}; self.level_ids = []; self.disabled_levels = {}; self._load(); return True
        return False

    def delete_level_from_config(self, lid):
        if not self.config_path or yaml is None:
            return False
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return False
        levels = self._levels_dict(data, create=True)
        disabled = self._disabled_levels_dict(data, create=True)
        levels.pop(lid, None)
        disabled.pop(lid, None)
        if not disabled:
            self._root_cfg(data, create=True).pop(DISABLED_LEVELS_KEY, None)
        if self._write_yaml(data):
            self.yaml_data = {}; self.levels = {}; self.level_ids = []; self.disabled_levels = {}; self._load(); return True
        return False

# ═══════════════════════════════════════════════════════════════════
# RENDERERS
# ═══════════════════════════════════════════════════════════════════
def render_tile_object(surf, char, sx, sy, ts):
    ti=int(ts)
    if USE_REAL_OBJECTS:
        if char=='#':
            t=create_tile(TILE_GROUND,0,0,True,COLOR_GROUND); t.gObj.width=t.gObj.height=ti; t.gObj.x=sx; t.gObj.y=sy; t.render(surf,0,0); return
        elif char=='=':
            t=create_tile(TILE_PLATFORM,0,0,True,COLOR_PLATFORM); t.gObj.width=t.gObj.height=ti; t.gObj.x=sx; t.gObj.y=sy; t.render(surf,0,0); return
        elif char=='^':
            t=create_tile(TILE_SPIKE,0,0,False,COLOR_SPIKE); t.gObj.width=t.gObj.height=ti; t.gObj.x=sx; t.gObj.y=sy; t.render(surf,0,0); return
        elif char=='H':
            Ladder.from_tile(sx, sy, ti).render(surf, sx, sy); return
        elif char=='G':
            Goal(gObj=GameObject(float(sx),float(sy),ti,ti)).render(surf,sx,sy); return
        elif char=='C':
            h2=ti//2; c=Coin(gObj=GameObject(float(sx+h2),float(sy+h2),ti,ti)); c.radius=max(4,h2-2); c.render(surf,sx+h2,sy+h2); return
        elif char=='E':
            Enemy(gObj=GameObject(float(sx),float(sy),ti,ti)).render(surf,sx,sy); return
        elif char in ('?','>','<','F','L'):
            cn={'?':'coin','>':'star','<':'mushroom','F':'flower','L':'life'}[char]
            QuestionBlock(gObj=GameObject(float(sx),float(sy),ti,ti),contains=cn).render(surf,sx,sy); return
    # Fallback rendering
    if char in (' ', '.'): pygame.draw.rect(surf,UI_BG,(sx,sy,ti,ti)); return
    if char in ('#','='):
        col=(139,69,19) if char=='#' else (205,133,63)
        pygame.draw.rect(surf,col,(sx,sy,ti,ti))
        hi=tuple(min(255,c+25) for c in col); lo=tuple(max(0,c-35) for c in col)
        pygame.draw.line(surf,hi,(sx,sy),(sx+ti-1,sy),1); pygame.draw.line(surf,hi,(sx,sy),(sx,sy+ti-1),1)
        pygame.draw.line(surf,lo,(sx+ti-1,sy),(sx+ti-1,sy+ti-1),1); pygame.draw.line(surf,lo,(sx,sy+ti-1),(sx+ti-1,sy+ti-1),1)
    elif char=='^':
        pygame.draw.rect(surf,(12,12,20),(sx,sy,ti,ti))
        pts=[(sx+ti//2,sy+int(ti*0.15)),(sx+int(ti*0.15),sy+int(ti*0.85)),(sx+int(ti*0.85),sy+int(ti*0.85))]
        pygame.draw.polygon(surf,(80,80,80),pts); pygame.draw.polygon(surf,(255,68,68),pts,max(1,ti//16))
    elif char=='O':
        cell=max(4,ti//4)
        for iy in range(0,ti,cell):
            for ix in range(0,ti,cell):
                if (ix//cell+iy//cell)%2==0:
                    s=pygame.Surface((min(cell,ti-ix),min(cell,ti-iy)),pygame.SRCALPHA); s.fill((180,30,50,120)); surf.blit(s,(sx+ix,sy+iy))
        pygame.draw.line(surf,(*UI_PIT_DIAG,200),(sx+2,sy+2),(sx+ti-2,sy+ti-2),2)
        pygame.draw.line(surf,(*UI_PIT_DIAG,200),(sx+ti-2,sy+2),(sx+2,sy+ti-2),2)
    elif char in ('?','>','<','F','L'):
        cols={'?':(255,165,0),'>':(255,215,0),'<':(255,68,68),'F':(255,102,0),'L':(90,220,120)}; col=cols.get(char,(255,165,0))
        pygame.draw.rect(surf,col,(sx,sy,ti,ti)); pygame.draw.rect(surf,tuple(max(0,c-40) for c in col),(sx,sy,ti,ti),max(1,ti//16))
        fsz=max(10,int(ti*0.55)); fnt=pygame.font.SysFont("Consolas",fsz,bold=True)
        tcol=(80,60,0) if char in ('?','>') else (255,240,240) if char=='<' else (0,60,20) if char=='L' else (80,30,0)
        marker='1' if char=='L' else char
        t=fnt.render(marker,True,tcol); surf.blit(t,t.get_rect(center=(sx+ti//2,sy+ti//2)))
    elif char=='C':
        pygame.draw.rect(surf,(12,12,20),(sx,sy,ti,ti)); cx2,cy2=sx+ti//2,sy+ti//2; r=max(3,ti//4)
        pygame.draw.circle(surf,(255,215,0),(cx2,cy2),r); pygame.draw.circle(surf,(184,134,11),(cx2,cy2),r,max(1,ti//16))
    elif char=='E':
        pygame.draw.rect(surf,(12,12,20),(sx,sy,ti,ti)); cx2,cy2=sx+ti//2,sy+ti//2; br=max(3,ti//4)
        pygame.draw.ellipse(surf,(139,69,19),(cx2-br,cy2-br//2,br*2,br)); er=max(1,ti//12)
        pygame.draw.circle(surf,(255,255,255),(cx2-er*2,cy2-er),er); pygame.draw.circle(surf,(255,255,255),(cx2+er*2,cy2-er),er)
    elif char=='H':
        rail=(120,215,255); rung=(240,240,200)
        pygame.draw.line(surf,rail,(sx+ti*0.32,sy+2),(sx+ti*0.32,sy+ti-2),3)
        pygame.draw.line(surf,rail,(sx+ti*0.68,sy+2),(sx+ti*0.68,sy+ti-2),3)
        for off in (0.22, 0.44, 0.66, 0.84):
            y2=int(sy+ti*off)
            pygame.draw.line(surf,rung,(sx+ti*0.32,y2),(sx+ti*0.68,y2),2)
    elif char=='M':
        pygame.draw.rect(surf,(38,48,80),(sx,sy,ti,ti))
        pygame.draw.rect(surf,(70,120,255),(sx+ti*0.12,sy+ti*0.45,ti*0.76,ti*0.38),border_radius=max(2,ti//10))
        pygame.draw.rect(surf,(235,230,120),(sx+ti*0.2,sy+ti*0.18,ti*0.6,ti*0.32),border_radius=max(2,ti//10))
        pygame.draw.circle(surf,(25,25,30),(int(sx+ti*0.35),int(sy+ti*0.58)),max(1,ti//14))
        pygame.draw.circle(surf,(25,25,30),(int(sx+ti*0.65),int(sy+ti*0.58)),max(1,ti//14))
    elif char=='B':
        pygame.draw.rect(surf,(18,12,34),(sx,sy,ti,ti))
        pygame.draw.polygon(surf,(190,120,255),[(sx+ti*0.18,sy+ti*0.48),(sx+ti*0.5,sy+ti*0.18),(sx+ti*0.42,sy+ti*0.56)])
        pygame.draw.polygon(surf,(190,120,255),[(sx+ti*0.82,sy+ti*0.48),(sx+ti*0.5,sy+ti*0.18),(sx+ti*0.58,sy+ti*0.56)])
        pygame.draw.ellipse(surf,(255,120,180),(sx+ti*0.34,sy+ti*0.42,ti*0.32,ti*0.28))
    elif char=='G':
        pygame.draw.rect(surf,(12,12,20),(sx,sy,ti,ti)); cx2=sx+ti//2; px=cx2-ti//8
        pygame.draw.line(surf,(136,136,136),(px,sy+ti*2//3),(px,sy+ti//6),2)
        pygame.draw.polygon(surf,(255,34,34),[(px+2,sy+ti//6),(px+ti//3,sy+ti//4),(px+2,sy+ti//3)])
    elif char=='D':
        pygame.draw.rect(surf,(40,16,24),(sx,sy,ti,ti))
        pygame.draw.rect(surf,(255,90,140),(sx+4,sy+2,ti-8,ti-4),border_radius=max(2,ti//12))
        pygame.draw.rect(surf,(60,10,30),(sx+8,sy+6,ti-16,ti-12),border_radius=max(2,ti//14))
    elif char=='P':
        pygame.draw.rect(surf,(12,12,20),(sx,sy,ti,ti)); cx2,cy2=sx+ti//2,sy+ti//2; hr=max(2,ti//6)
        pygame.draw.circle(surf,(255,204,170),(cx2,cy2-hr),hr)
        pygame.draw.rect(surf,(68,136,255),(cx2-ti//5,cy2,ti*2//5,ti//3))
    else:
        pygame.draw.rect(surf,UI_BG,(sx,sy,ti,ti))

def draw_tile_rect(surf, char, x, y, w, h, alpha=255):
    tile=TILE_BY_CHAR.get(char,TILE_BY_CHAR[' ']); color=tile[2]
    if char=='O':
        s=pygame.Surface((w,h),pygame.SRCALPHA); s.fill((0,0,0,0))
        cell=max(4,min(w,h)//4)
        for iy in range(0,h,cell):
            for ix in range(0,w,cell):
                if (ix//cell+iy//cell)%2==0:
                    s2=pygame.Surface((min(cell,w-ix),min(cell,h-iy)),pygame.SRCALPHA); s2.fill((180,30,50,120)); s.blit(s2,(ix,iy))
        surf.blit(s,(x,y)); return
    s=pygame.Surface((w,h),pygame.SRCALPHA); s.fill((*color,alpha))
    if char in SOLID_CHARS and w>4:
        hi=tuple(min(255,c+40) for c in color)
        pygame.draw.line(s,(*hi,alpha),(0,0),(w-1,0),max(1,h//12))
    surf.blit(s,(x,y))

# ─── Palette icons (same as v3) ──────────────────────────────────
def draw_palette_icon(surf, char, rect):
    x,y,w,h=rect.x,rect.y,rect.width,rect.height; cx,cy=x+w//2,y+h//2
    col=TILE_BY_CHAR.get(char,TILE_BY_CHAR[' '])[2]
    hi=tuple(min(255,c+70) for c in col); dk=tuple(max(0,c-40) for c in col)
    if char in (' ', '.'):
        for a in range(0,360,45):
            pygame.draw.circle(surf,(55,30,40),(cx+int(4*math.cos(math.radians(a))),cy+int(4*math.sin(math.radians(a)))),1)
    elif char=='O':
        cell=max(3,w//4)
        for iy in range(0,h,cell):
            for ix in range(0,w,cell):
                if (ix//cell+iy//cell)%2==0:
                    s=pygame.Surface((min(cell,w-ix),min(cell,h-iy)),pygame.SRCALPHA); s.fill((180,30,50,140)); surf.blit(s,(x+ix,y+iy))
        pygame.draw.line(surf,UI_PIT_DIAG,(x+2,y+2),(x+w-2,y+h-2),2); pygame.draw.line(surf,UI_PIT_DIAG,(x+w-2,y+2),(x+2,y+h-2),2)
    elif char=='#':
        pygame.draw.rect(surf,col,(x+2,y+2,w-4,h-4),border_radius=2); pygame.draw.rect(surf,hi,(x+2,y+2,w-4,h-4),1,border_radius=2)
        m2=y+h//2; pygame.draw.line(surf,dk,(x+3,m2),(x+w-4,m2),1); pygame.draw.line(surf,dk,(cx,y+3),(cx,m2-1),1)
    elif char=='=':
        bh2=max(5,h//3); by2=cy-bh2//2
        pygame.draw.rect(surf,col,(x+1,by2,w-2,bh2),border_radius=3); pygame.draw.rect(surf,hi,(x+1,by2,w-2,bh2),1,border_radius=3)
    elif char=='^':
        pts=[(cx,y+2),(x+2,y+h-2),(x+w-2,y+h-2)]; pygame.draw.polygon(surf,col,pts); pygame.draw.polygon(surf,(255,100,100),pts,1)
    elif char in ('?','>','<','F','L'):
        qc={'?':(210,170,30),'>':(240,200,50),'<':(195,80,70),'F':(240,110,20),'L':(90,220,120)}[char]
        pygame.draw.rect(surf,qc,(x+2,y+2,w-4,h-4),border_radius=3)
        pygame.draw.rect(surf,tuple(min(255,c+60) for c in qc),(x+2,y+2,w-4,h-4),1,border_radius=3)
        if char=='?': pygame.draw.circle(surf,(40,30,0),(cx,cy-1),max(2,w//6))
        elif char=='>': pygame.draw.polygon(surf,(40,30,0),[(cx,y+4),(x+w-3,cy),(cx,y+h-4)])
        elif char=='<': pygame.draw.rect(surf,(40,30,0),(cx-2,cy-2,4,4),border_radius=1)
        elif char=='L':
            pygame.draw.line(surf,(0,60,20),(cx,y+5),(cx,y+h-5),2)
            pygame.draw.line(surf,(0,60,20),(cx,y+h-5),(x+w-4,y+h-5),2)
        elif char=='F':
            for a in range(0,360,72):
                pygame.draw.circle(surf,(255,80,20),(cx+int((w//4)*math.cos(math.radians(a))),cy+int((h//4)*math.sin(math.radians(a)))),max(2,w//7))
            pygame.draw.circle(surf,(255,220,0),(cx,cy),max(2,w//6))
    elif char=='C':
        r2=min(w,h)//2-2; pygame.draw.circle(surf,(215,175,0),(cx,cy),r2); pygame.draw.circle(surf,(255,240,100),(cx,cy),r2,1)
    elif char=='E':
        r2=min(w,h)//2-2; pygame.draw.circle(surf,col,(cx,cy),r2)
        ey=cy-r2//4
        for ox in (-r2//3,r2//3): pygame.draw.circle(surf,(255,255,255),(cx+ox,ey),max(1,r2//3)); pygame.draw.circle(surf,(20,0,20),(cx+ox,ey),max(1,r2//5))
    elif char=='H':
        pygame.draw.line(surf,col,(x+w//3,y+3),(x+w//3,y+h-3),2)
        pygame.draw.line(surf,col,(x+w*2//3,y+3),(x+w*2//3,y+h-3),2)
        for yy in range(y+6, y+h-3, max(5, h//4)):
            pygame.draw.line(surf,hi,(x+w//3,yy),(x+w*2//3,yy),2)
    elif char=='M':
        pygame.draw.rect(surf,(70,120,255),(x+3,y+h//2,w-6,h//3),border_radius=3)
        pygame.draw.rect(surf,(235,230,120),(x+6,y+4,w-12,h//3),border_radius=3)
    elif char=='B':
        pygame.draw.polygon(surf,col,[(x+3,cy),(cx,y+3),(cx-2,cy+2)])
        pygame.draw.polygon(surf,col,[(x+w-3,cy),(cx,y+3),(cx+2,cy+2)])
        pygame.draw.circle(surf,(255,120,180),(cx,cy+2),max(2,w//7))
    elif char=='G':
        px2=cx-1; pygame.draw.line(surf,(200,200,200),(px2,y+2),(px2,y+h-2),2)
        pygame.draw.polygon(surf,col,[(px2+1,y+3),(x+w-2,y+3+h//4),(px2+1,y+3+h//2)])
    elif char=='D':
        pygame.draw.rect(surf,col,(x+3,y+2,w-6,h-4),border_radius=3)
        pygame.draw.rect(surf,(60,10,30),(x+7,y+6,w-14,h-12),border_radius=3)
    elif char=='P':
        hr=max(3,h//5); pygame.draw.circle(surf,(80,180,255),(cx,y+h//3),hr)
        pygame.draw.rect(surf,(50,140,220),(cx-w//5,y+h//3+hr-1,w*2//5,h//3),border_radius=2)

# ═══════════════════════════════════════════════════════════════════
# LEVEL
# ═══════════════════════════════════════════════════════════════════
class Level:
    def __init__(self, rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
        self.rows=rows; self.cols=cols; self.grid=[[' ']*cols for _ in range(rows)]
        self.platforms=[]; self.filename=None; self.level_id=None
        self.physics_overrides={}; self.enemy_physics={}
    def get(self,r,c): return self.grid[r][c] if 0<=r<self.rows and 0<=c<self.cols else None
    def set(self,r,c,ch):
        if r<0 or c<0: return
        while r>=self.rows: self.grid.append([' ']*self.cols); self.rows+=1
        if c>=self.cols:
            for row in self.grid: row.extend([' ']*(c-self.cols+1))
            self.cols=c+1
        self.grid[r][c]=ch
    def clear_all(self):
        for r in range(self.rows):
            for c in range(self.cols): self.grid[r][c]=' '
    def resize(self, nr, nc):
        ng=[[' ']*nc for _ in range(nr)]
        for r in range(min(self.rows,nr)):
            for c in range(min(self.cols,nc)): ng[r][c]=self.grid[r][c]
        self.grid=ng; self.rows=nr; self.cols=nc
    def bounding_box(self):
        mr=self.rows; xr=-1; mc=self.cols; xc=-1
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c]!=' ': mr=min(mr,r); xr=max(xr,r); mc=min(mc,c); xc=max(xc,c)
        return (mr,mc,xr,xc) if xr!=-1 else None
    def to_ascii(self, trim=True, pad=1):
        if not trim: return '\n'.join(''.join(r) for r in self.grid)
        bb=self.bounding_box()
        if bb is None: return '\n'.join([' '*DEFAULT_COLS]*DEFAULT_ROWS)
        r0=max(0,bb[0]-pad); c0=max(0,bb[1]-pad); r1=min(self.rows-1,bb[2]+pad); c1=min(self.cols-1,bb[3]+pad)
        w=c1-c0+1
        lines=[]
        for r in range(r0,r1+1):
            rc=self.grid[r][c0:c1+1]
            while len(rc)<w: rc.append(' ')
            lines.append(''.join(rc))
        return '\n'.join(lines)
    def save(self, path, trim=True):
        with open(path,'w') as f: f.write(self.to_ascii(trim=trim))
        self.filename=str(path); self._save_yaml(path)
    def _save_yaml(self, tp):
        if yaml is None: return
        yp=str(tp).rsplit('.',1)[0]+'.yaml'
        d={'dynamics':{'moving_platforms':[p.to_dict() for p in self.platforms]} if self.platforms else {}}
        if self.physics_overrides:
            d['physics']=self.physics_overrides
        if self.enemy_physics:
            d.setdefault('physics',{})['enemies']=self.enemy_physics
        with open(yp,'w') as f: yaml.dump(d,f,default_flow_style=False,sort_keys=False)
    def _load_yaml(self, tp):
        if yaml is None: return
        yp=str(tp).rsplit('.',1)[0]+'.yaml'
        if not Path(yp).exists(): return
        try:
            with open(yp,'r') as f: data=yaml.safe_load(f) or {}
            for pd in (data.get('dynamics',{}) or {}).get('moving_platforms',[]):
                self.platforms.append(PlatformDef.from_dict(pd))
            ph=data.get('physics',{}) or {}
            if ph:
                self.physics_overrides={k:v for k,v in ph.items() if k!='enemies'}
                self.enemy_physics=ph.get('enemies',{}) or {}
        except Exception as e: print(f"[Editor] YAML sidecar error: {e}")
    def clone(self):
        n=Level(self.rows,self.cols); n.grid=[r[:] for r in self.grid]
        n.filename=self.filename; n.level_id=self.level_id
        n.platforms=[PlatformDef(p.start[:],p.end[:],p.speed,p.width,p.height) for p in self.platforms]
        n.physics_overrides=dict(self.physics_overrides); n.enemy_physics=dict(self.enemy_physics)
        return n
    @classmethod
    def from_ascii(cls, text):
        lines=text.split('\n'); rows=len(lines); cols=max((len(l) for l in lines),default=DEFAULT_COLS)
        lv=cls(rows,cols)
        for r,line in enumerate(lines):
            for c,ch in enumerate(line): lv.grid[r][c]=ch
        return lv
    @classmethod
    def load(cls, path):
        with open(path,'r') as f: text=f.read()
        lv=cls.from_ascii(text); lv.filename=str(path); lv._load_yaml(path); return lv

# ═══════════════════════════════════════════════════════════════════
# PHYSICS SLIDER / TOOLBAR BUTTON
# ═══════════════════════════════════════════════════════════════════
class PhysicsSlider:
    __slots__=('label','key','value','min_val','max_val','step','rect','dragging')
    def __init__(s,label,key,value,mn,mx,step=10.0):
        s.label=label;s.key=key;s.value=float(value);s.min_val=float(mn);s.max_val=float(mx);s.step=float(step)
        s.rect=pygame.Rect(0,0,0,0);s.dragging=False
    def fraction(s):
        rng=s.max_val-s.min_val; return (s.value-s.min_val)/rng if rng>0 else 0.0
    def set_from_fraction(s,f):
        f=max(0.0,min(1.0,f)); rng=s.max_val-s.min_val
        s.value=round((s.min_val+f*rng)/s.step)*s.step; s.value=max(s.min_val,min(s.max_val,s.value))

class TBBtn:
    __slots__=('label','sc','action','rect','toggle','get_active')
    def __init__(s,label,sc,action,toggle=False,get_active=None):
        s.label=label;s.sc=sc;s.action=action;s.rect=pygame.Rect(0,0,0,0);s.toggle=toggle;s.get_active=get_active

# ═══════════════════════════════════════════════════════════════════
# EDITOR STATE
# ═══════════════════════════════════════════════════════════════════
class Editor:
    def __init__(self, level, gc):
        self.level=level; self.gc=gc; self.game=gc.game; self.game_label=gc.game_label
        self.tile_entries=tile_entries_for(self.game)
        self.undo_stack=[]; self.redo_stack=[]
        self.selected_tile='#'; self.show_grid=True; self.fill_mode=False; self.eraser_mode=False
        self.cam_x=0.0; self.cam_y=0.0; self.zoom=1.0
        self.painting=False; self.paint_char=' '; self.panning=False
        self.pan_start=(0,0); self.pan_cam=(0.0,0.0)
        self.last_cell=None; self.hover_cell=None; self.dirty=False
        self.plat_placing=False; self.plat_start_world=None; self.plat_ghost_end=None
        self.plat_default_spd=PLAT_DEFAULT_SPD; self.sel_plat_idx=None
        self.drag_handle=None; self.drag_offset=(0.0,0.0)
        self.right_scroll=0; self.right_max_scroll=0; self.browser_level_id=None
        self._sb_thumb=pygame.Rect(0,0,0,0); self._sb_drag=False; self._sb_dy=0; self._sb_ds=0
        self._hs_thumb=pygame.Rect(0,0,0,0); self._hs_drag=False; self._hs_dx=0; self._hs_dc=0.0
        self._spd_m=pygame.Rect(0,0,0,0); self._spd_p=pygame.Rect(0,0,0,0)
        self._tile_rects={}; self._tool_rects={}
        self._lv_btn={}; self._ul_btn={}; self._ul_del={}
        self._tog_btn={}; self._ren_btn={}; self._del_btn={}; self._dh_rects={}
        self._assign_r=pygame.Rect(0,0,0,0)
        self._drag_lid=None; self._drag_y=0; self._drag_si=-1; self._drag_di=-1
        self._lv_list_top=0; self._lv_row_h=36
        ph=gc.physics or {}; fric=ph.get('friction',{}) or {}
        self.phys_sliders=[
            PhysicsSlider("Gravity","gravity",ph.get('gravity',1300),200,3000,50),
            PhysicsSlider("Fast Fall","fast_fall",ph.get('fast_fall_gravity',2500),500,5000,50),
            PhysicsSlider("Gnd Frict","ground_fric",fric.get('ground',1300),200,3000,50),
            PhysicsSlider("Air Frict","air_fric",fric.get('air',250),50,1000,25),
            PhysicsSlider("Max Run","max_run",150,50,500,10),
            PhysicsSlider("Jump Vel","jump_vel",800,200,1500,25),
            PhysicsSlider("Max Fall","max_fall",550,200,1200,25),
        ]
        ep=(ph.get('enemies',{}) or {}) if ph else {}
        self.enemy_sliders=[
            PhysicsSlider("Walk Spd","enemy_speed",ep.get('walk_speed',60),10,300,5),
            PhysicsSlider("Grav Mult","enemy_grav_pct",ep.get('gravity_mult',1.0)*100,10,300,10),
            PhysicsSlider("Max Fall","enemy_max_fall",ep.get('max_fall_speed',550),100,1200,25),
            PhysicsSlider("Patrol W","enemy_patrol_w",ep.get('patrol_width',96),16,512,16),
        ]
        self._center()
    def _center(self):
        vw=WINDOW_W-LEFT_PANEL_W-RIGHT_PANEL_W; vh=WINDOW_H-TOOLBAR_H-STATUS_H-HSB_H
        self.cam_x=(self.level.cols*TILE_SIZE*self.zoom-vw)/2
        self.cam_y=(self.level.rows*TILE_SIZE*self.zoom-vh)/2
    def reset_view(self): self.cam_x=0.0;self.cam_y=0.0;self.zoom=1.0
    def push_undo(self):
        self.undo_stack.append(self.level.clone())
        if len(self.undo_stack)>UNDO_LIMIT: self.undo_stack.pop(0)
        self.redo_stack.clear()
    def undo(self):
        if self.undo_stack: self.redo_stack.append(self.level.clone()); self.level=self.undo_stack.pop(); self.dirty=True
    def redo(self):
        if self.redo_stack: self.undo_stack.append(self.level.clone()); self.level=self.redo_stack.pop(); self.dirty=True
    def clear_all(self): self.push_undo(); self.level.clear_all(); self.level.platforms.clear(); self.dirty=True
    def resize_grid(self,nr,nc): self.push_undo(); self.level.resize(nr,nc); self.dirty=True
    def w2s(self,wx,wy):
        ts=TILE_SIZE*self.zoom; return (int(wx*ts-self.cam_x+LEFT_PANEL_W),int(wy*ts-self.cam_y+TOOLBAR_H))
    def wp2s(self,px,py): return (int(px*self.zoom-self.cam_x+LEFT_PANEL_W),int(py*self.zoom-self.cam_y+TOOLBAR_H))
    def s2wp(self,sx,sy,snap=True):
        wx=(sx-LEFT_PANEL_W+self.cam_x)/self.zoom; wy=(sy-TOOLBAR_H+self.cam_y)/self.zoom
        if snap: wx=round(wx/TILE_SIZE)*TILE_SIZE; wy=round(wy/TILE_SIZE)*TILE_SIZE
        return wx,wy
    def s2c(self,sx,sy):
        ts=TILE_SIZE*self.zoom; return int((sy-TOOLBAR_H+self.cam_y)/ts),int((sx-LEFT_PANEL_W+self.cam_x)/ts)
    def paint_cell(self,r,c,ch):
        if (r,c)==self.last_cell: return
        if self.level.get(r,c)!=ch: self.level.set(r,c,ch); self.last_cell=(r,c); self.dirty=True
    def flood_fill(self,r,c,ch):
        tgt=self.level.get(r,c)
        if tgt is None or tgt==ch: return
        self.push_undo(); stk=[(r,c)]; vis=set()
        while stk:
            cr,cc=stk.pop()
            if (cr,cc) in vis or self.level.get(cr,cc)!=tgt: continue
            vis.add((cr,cc)); self.level.set(cr,cc,ch)
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr2,nc2=cr+dr,cc+dc
                if (nr2,nc2) not in vis and self.level.get(nr2,nc2)==tgt: stk.append((nr2,nc2))
        self.dirty=True
    def ht_plat(self,sx,sy):
        for i,p in enumerate(self.level.platforms):
            for h in ('start','end'):
                wp=p.start if h=='start' else p.end
                hsx,hsy=self.wp2s(wp[0]+p.width/2,wp[1]+p.height/2)
                if math.hypot(sx-hsx,sy-hsy)<=HANDLE_HIT_R: return i,h
        return None,None
    def load_by_id(self,lid):
        path=self.gc.get_level_file_path(lid)
        if not path: return False
        try:
            self.level=Level.load(path); self.level.level_id=lid
            self.undo_stack.clear(); self.redo_stack.clear()
            self.dirty=False; self._center(); self.browser_level_id=lid; return True
        except: return False
    def brush(self): return ' ' if self.eraser_mode or self.selected_tile==MOVING_PLATFORM_BRUSH else self.selected_tile
    def start_lv_drag(self,lid,si,my): self._drag_lid=lid;self._drag_si=si;self._drag_y=my;self._drag_di=si
    def finish_lv_drag(self):
        if not self._drag_lid: return
        s,d=self._drag_si,self._drag_di
        if s!=d:
            ids=list(self.gc.level_ids); ids.insert(d,ids.pop(s)); self.gc.reorder_levels(ids)
        self._drag_lid=None;self._drag_si=-1;self._drag_di=-1
    def cancel_lv_drag(self): self._drag_lid=None
    def check_unsaved(self,scr,sf):
        if not self.dirty: return True
        r=dlg_unsaved(scr,sf)
        if r=="yes":
            if self.level.filename: self.level.save(self.level.filename); self.dirty=False
            else:
                p=dlg_save(scr,sf)
                if p: self.level.save(p); self.dirty=False
                else: return False
            return True
        return r=="no"

# ═══════════════════════════════════════════════════════════════════
# DRAW FUNCTIONS (viewport, panels, toolbar, status)
# ═══════════════════════════════════════════════════════════════════
def draw_platforms(surf,ed,vp):
    def bs(p,wp): sx,sy=ed.wp2s(wp[0],wp[1]); return sx,sy,int(p.width*ed.zoom),int(p.height*ed.zoom)
    def dash(s,col,p1,p2,d=8,g=5):
        dx=p2[0]-p1[0];dy=p2[1]-p1[1];L=math.hypot(dx,dy)
        if L<1:return
        ux,uy=dx/L,dy/L;pos=0.0;on=True
        while pos<L:
            seg=d if on else g;end=min(pos+seg,L)
            if on: pygame.draw.line(s,col,(int(p1[0]+ux*pos),int(p1[1]+uy*pos)),(int(p1[0]+ux*end),int(p1[1]+uy*end)),1)
            pos+=seg;on=not on
    for i,p in enumerate(ed.level.platforms):
        sel=i==ed.sel_plat_idx
        bx,by,bw,bh=bs(p,p.start); br=pygame.Rect(bx,by,bw,bh)
        if br.colliderect(vp):
            pygame.draw.rect(surf,PLAT_BODY_COL,br); pygame.draw.rect(surf,PLAT_HIGH_COL,(bx,by,bw,max(2,bh//4)))
            pygame.draw.rect(surf,PLAT_SEL_COL if sel else (160,90,30),br,2)
        ex,ey,ew,eh=bs(p,p.end)
        gs=pygame.Surface((max(1,ew),max(1,eh)),pygame.SRCALPHA); gs.fill((*PLAT_BODY_COL,100))
        pygame.draw.rect(gs,(*PLAT_SEL_COL,220) if sel else (160,90,30,120),(0,0,ew,eh),2); surf.blit(gs,(ex,ey))
        dash(surf,PLAT_PATH_COL,(bx+bw//2,by+bh//2),(ex+ew//2,ey+eh//2))
        for cx2,cy2,cl in [(bx+bw//2,by+bh//2,HANDLE_START),(ex+ew//2,ey+eh//2,HANDLE_END)]:
            pygame.draw.circle(surf,cl,(cx2,cy2),HANDLE_R); pygame.draw.circle(surf,(255,255,255),(cx2,cy2),HANDLE_R,2)
    if ed.selected_tile==MOVING_PLATFORM_BRUSH and ed.plat_placing and ed.plat_ghost_end:
        sx0,sy0=ed.wp2s(*ed.plat_start_world); sw=int(PLAT_DEFAULT_W*ed.zoom);sh=int(PLAT_DEFAULT_H*ed.zoom)
        ex0,ey0=ed.wp2s(*ed.plat_ghost_end)
        g1=pygame.Surface((max(1,sw),max(1,sh)),pygame.SRCALPHA);g1.fill((*PLAT_BODY_COL,160));surf.blit(g1,(sx0,sy0))
        g2=pygame.Surface((max(1,sw),max(1,sh)),pygame.SRCALPHA);g2.fill((*PLAT_BODY_COL,80));surf.blit(g2,(ex0,ey0))
        dash(surf,PLAT_PATH_COL,(sx0+sw//2,sy0+sh//2),(ex0+sw//2,ey0+sh//2))
        pygame.draw.circle(surf,HANDLE_START,(sx0+sw//2,sy0+sh//2),HANDLE_R)
        pygame.draw.circle(surf,HANDLE_END,(ex0+sw//2,ey0+sh//2),HANDLE_R)

def draw_viewport(surf,ed,font,sfont):
    vx=LEFT_PANEL_W;vy=TOOLBAR_H;vw=WINDOW_W-LEFT_PANEL_W-RIGHT_PANEL_W;vh=WINDOW_H-TOOLBAR_H-STATUS_H-HSB_H
    vp=pygame.Rect(vx,vy,vw,vh); surf.set_clip(vp); pygame.draw.rect(surf,UI_BG,vp)
    ts=TILE_SIZE*ed.zoom; lv=ed.level
    cs=max(0,int(ed.cam_x/ts));rs=max(0,int(ed.cam_y/ts))
    ce=min(lv.cols,cs+int(vw/ts)+2);re=min(lv.rows,rs+int(vh/ts)+2)
    for r in range(rs,re):
        for c in range(cs,ce):
            sx,sy=ed.w2s(c,r);ch=lv.get(r,c) or ' '
            tr=pygame.Rect(sx,sy,int(ts),int(ts))
            if tr.colliderect(vp): render_tile_object(surf,ch,sx,sy,int(ts))
    bx,by=ed.w2s(0,0); pygame.draw.rect(surf,(70,88,140),(bx,by,int(lv.cols*ts),int(lv.rows*ts)),2)
    draw_platforms(surf,ed,vp)
    if ed.show_grid and ts>=6:
        for c in range(cs,ce+1):
            sx=int(c*ts-ed.cam_x+vx);bold=c%10==0
            pygame.draw.line(surf,GRID_BOLD if bold else GRID_COLOR,(sx,vy),(sx,vy+vh),2 if bold else 1)
        for r in range(rs,re+1):
            sy=int(r*ts-ed.cam_y+vy);bold=r%10==0
            pygame.draw.line(surf,GRID_BOLD if bold else GRID_COLOR,(vx,sy),(vx+vw,sy),2 if bold else 1)
    if ed.hover_cell:
        hr,hc=ed.hover_cell;sx,sy=ed.w2s(hc,hr)
        if ed.eraser_mode:
            pygame.draw.rect(surf,(255,80,80),(sx,sy,int(ts),int(ts)),2)
            pygame.draw.line(surf,(255,80,80),(sx+2,sy+2),(sx+int(ts)-2,sy+int(ts)-2),2)
            pygame.draw.line(surf,(255,80,80),(sx+int(ts)-2,sy+2),(sx+2,sy+int(ts)-2),2)
        else:
            pygame.draw.rect(surf,(255,255,255),(sx,sy,int(ts),int(ts)),2)
            gh=pygame.Surface((int(ts),int(ts)),pygame.SRCALPHA)
            draw_tile_rect(gh,ed.selected_tile,0,0,int(ts),int(ts),alpha=100); surf.blit(gh,(sx,sy))
    surf.set_clip(None)
    # H-scrollbar
    hy=vy+vh; pygame.draw.rect(surf,UI_PANEL2,(vx,hy,vw,HSB_H))
    pygame.draw.line(surf,UI_BORDER,(vx,hy),(vx+vw,hy),1)
    lw=lv.cols*ts
    if lw>vw:
        rat=vw/lw;tw=max(32,int(vw*rat));sr=max(1,vw-tw);mc=lw-vw
        fr=max(0.0,min(1.0,ed.cam_x/mc)) if mc>0 else 0.0
        tx=vx+int(fr*sr);ht=pygame.Rect(tx,hy+2,tw,HSB_H-4)
        hov=ht.collidepoint(pygame.mouse.get_pos()) or ed._hs_drag
        pygame.draw.rect(surf,UI_ACCENT if hov else UI_BORDER,ht,border_radius=3); ed._hs_thumb=ht
    else: ed._hs_thumb=pygame.Rect(0,0,0,0)

# (Left panel, right panel, toolbar, status — using same logic as v3
#  but with the critical toolbar click fix applied in main loop)

def draw_left_panel(surf,ed,font,sfont):
    px=LEFT_PANEL_W;pad=8
    pygame.draw.rect(surf,UI_PANEL,(0,TOOLBAR_H,px,WINDOW_H-TOOLBAR_H-STATUS_H))
    pygame.draw.line(surf,UI_BORDER,(px-1,TOOLBAR_H),(px-1,WINDOW_H-STATUS_H),1)
    y=TOOLBAR_H+10
    try:sfont.render_to(surf,(pad+2,y),f"TILES  {ed.game.upper()}",fgcolor=UI_SUBTEXT,size=11)
    except:pass
    y+=16;cw=(px-pad*2-4)//2;th=34;ics=24;gap=3;ed._tile_rects.clear()
    for idx,tile in enumerate(ed.tile_entries):
        char,label,color,tcol=tile;ci=idx%2;ri=idx//2
        tx=pad+ci*(cw+gap);ty=y+ri*(th+gap);rect=pygame.Rect(tx,ty,cw,th);ed._tile_rects[char]=rect
        sel=char==ed.selected_tile and not ed.eraser_mode
        pygame.draw.rect(surf,UI_SELECT if sel else UI_PANEL2,rect,border_radius=4)
        draw_palette_icon(surf,char,pygame.Rect(tx+3,ty+2,ics,ics))
        try:sfont.render_to(surf,(tx+ics+6,ty+th//2-5),label,fgcolor=UI_TEXT if sel else UI_SUBTEXT,size=10)
        except:pass
        if sel:pygame.draw.rect(surf,UI_ACCENT,rect,2,border_radius=4)
    rn=math.ceil(len(ed.tile_entries)/2);y+=rn*(th+gap)+8
    pygame.draw.line(surf,UI_BORDER,(pad,y),(px-pad,y),1);y+=8
    try:sfont.render_to(surf,(pad+2,y),"TOOLS",fgcolor=UI_SUBTEXT,size=11)
    except:pass
    y+=14;ed._tool_rects.clear();tbw=(px-pad*2-4)//3
    for ti2,(key,label,active) in enumerate([("eraser","Eraser",ed.eraser_mode),("fill","Fill",ed.fill_mode),("grid","Grid",ed.show_grid)]):
        tx=pad+ti2*(tbw+2);br=pygame.Rect(tx,y,tbw,26);ed._tool_rects[key]=br
        if key=="eraser" and active:pygame.draw.rect(surf,UI_DANGER,br,border_radius=4)
        elif active:pygame.draw.rect(surf,UI_BTN_ACT,br,border_radius=4)
        else:pygame.draw.rect(surf,UI_BTN,br,border_radius=4)
        try:
            b=sfont.get_rect(label,size=10);fc=UI_TEXT if active else UI_SUBTEXT
            sfont.render_to(surf,(br.x+(br.w-b.width)//2,y+7),label,fgcolor=fc,size=10)
        except:pass
        if active:pygame.draw.rect(surf,UI_ACCENT if key!="eraser" else UI_DANGER,br,1,border_radius=4)
    y+=32;pygame.draw.line(surf,UI_BORDER,(pad,y),(px-pad,y),1);y+=8
    try:sfont.render_to(surf,(pad+2,y),"PLATFORM",fgcolor=UI_SUBTEXT,size=11)
    except:pass
    y+=14;pa=ed.selected_tile==MOVING_PLATFORM_BRUSH;pr=pygame.Rect(pad,y,px-pad*2,26);ed._tool_rects['platform']=pr
    pygame.draw.rect(surf,UI_BTN_ACT if pa else UI_BTN,pr,border_radius=4)
    ico=pygame.Rect(pad+6,y+10,18,5);pygame.draw.rect(surf,PLAT_BODY_COL,ico,border_radius=1)
    pygame.draw.circle(surf,HANDLE_START,(ico.x+3,ico.centery),3);pygame.draw.circle(surf,HANDLE_END,(ico.x+ico.w-3,ico.centery),3)
    try:sfont.render_to(surf,(pad+28,y+7),"V  Platform",fgcolor=UI_TEXT if pa else UI_SUBTEXT,size=11)
    except:pass
    if pa:pygame.draw.rect(surf,UI_ACCENT,pr,1,border_radius=4)
    y+=30;sr=pygame.Rect(pad,y,px-pad*2,24)
    pygame.draw.rect(surf,UI_PANEL2,sr,border_radius=3);pygame.draw.rect(surf,UI_BORDER,sr,1,border_radius=3)
    try:sfont.render_to(surf,(pad+6,y+6),f"Spd: {ed.plat_default_spd:.0f}",fgcolor=UI_TEXT,size=10)
    except:pass
    bw=18;mr=pygame.Rect(px-pad-bw*2-4,y+3,bw,18);pr2=pygame.Rect(px-pad-bw,y+3,bw,18)
    for btn,sym in [(mr,'-'),(pr2,'+')]:
        pygame.draw.rect(surf,UI_BORDER,btn,border_radius=3)
        try:
            b=sfont.get_rect(sym,size=12);sfont.render_to(surf,(btn.x+(btn.w-b.width)//2,btn.y+(btn.h-b.height)//2),sym,fgcolor=UI_TEXT,size=12)
        except:pass
    ed._spd_m=mr;ed._spd_p=pr2;y+=28
    if pa:
        hint="Click END" if ed.plat_placing else "Click START";hcol=UI_WARN if ed.plat_placing else UI_ACCENT2
        try:sfont.render_to(surf,(pad+4,y),hint,fgcolor=hcol,size=10)
        except:pass
        y+=16
    y+=8;pygame.draw.line(surf,UI_BORDER,(pad,y),(px-pad,y),1);y+=8
    try:sfont.render_to(surf,(pad+2,y),"CONTROLS",fgcolor=UI_SUBTEXT,size=10)
    except:pass
    y+=14
    for kt,dt in [("LClick","Paint"),("RClick","Erase"),("MDrag","Pan"),("Scroll","Zoom"),
                   ("Ctrl+S","Save"),("Ctrl+O","Open"),("Ctrl+N","New"),("Ctrl+R","Resize"),
                   ("Ctrl+P","Play"),("Ctrl+Z","Undo"),("Ctrl+Y","Redo"),("Ctrl+X","Clear"),
                   ("G","Grid"),("F","Fill"),("X","Eraser"),("V","Platform"),("Del","Del plat"),
                   ("Home","Reset"),("1-0","Tiles"),("Esc","Cancel")]:
        if y+12>WINDOW_H-STATUS_H-4:break
        try:
            sfont.render_to(surf,(pad+2,y),kt,fgcolor=UI_ACCENT2,size=9)
            sfont.render_to(surf,(pad+56,y),dt,fgcolor=(110,78,92),size=9)
        except:pass
        y+=12

def draw_right_panel(surf,ed,sfont,mpos):
    rx=WINDOW_W-RIGHT_PANEL_W;ry=TOOLBAR_H;rw=RIGHT_PANEL_W;rh=WINDOW_H-TOOLBAR_H-STATUS_H
    cw=rw-SB_W;sbx=rx+cw;pygame.draw.rect(surf,UI_PANEL,(rx,ry,rw,rh))
    pygame.draw.line(surf,UI_BORDER,(rx,ry),(rx,ry+rh),1)
    clip=pygame.Rect(rx,ry,cw,rh);surf.set_clip(clip)
    pad=8;y=ry+8-ed.right_scroll;DW=18;gc=ed.gc
    ed._lv_btn.clear();ed._ul_btn.clear();ed._ul_del.clear()
    ed._tog_btn.clear();ed._ren_btn.clear();ed._del_btn.clear();ed._dh_rects.clear()
    def reg(d,k,r):
        if clip.colliderect(r):d[k]=r
    cnt=len(gc.level_ids)
    try:
        sfont.render_to(surf,(rx+pad,y),"LEVELS",fgcolor=UI_ACCENT,size=12)
        sfont.render_to(surf,(rx+cw-30,y),f"({cnt})",fgcolor=UI_SUBTEXT,size=11)
    except:pass
    y+=18
    if ed._drag_lid:
        try:sfont.render_to(surf,(rx+pad,y),f"drag {ed._drag_lid}",fgcolor=UI_WARN,size=10)
        except:pass
        y+=14
    ed._lv_list_top=y;rh2=36;ed._lv_row_h=rh2
    for idx,lid in enumerate(gc.level_ids):
        lcfg=gc.levels.get(lid,{});fn=lcfg.get('file','???');cur=lid==ed.browser_level_id;isDrag=lid==ed._drag_lid
        if ed._drag_lid and ed._drag_di==idx and not isDrag:
            pygame.draw.line(surf,UI_ACCENT2,(rx+pad,y-1),(rx+cw-pad,y-1),2)
        dhr=pygame.Rect(rx+pad,y+10,DW,rh2-18);reg(ed._dh_rects,lid,dhr)
        dhc=UI_ACCENT if dhr.collidepoint(mpos) or isDrag else UI_SUBTEXT
        for li in range(3):pygame.draw.line(surf,dhc,(dhr.x+2,dhr.y+li*5),(dhr.x+dhr.w-2,dhr.y+li*5),2)
        tw2=36;th2=18;cb=pygame.Rect(rx+pad+DW+4,y+9,tw2,th2);reg(ed._tog_btn,lid,cb)
        tbg=(30,110,40) if not isDrag else UI_BTN
        pygame.draw.rect(surf,UI_BTN_HOVER if cb.collidepoint(mpos) else tbg,cb,border_radius=4)
        pygame.draw.rect(surf,UI_ACCENT2,cb,1,border_radius=4)
        try:sfont.render_to(surf,(cb.x+4,cb.y+4),"ON",fgcolor=UI_TEXT,size=9)
        except:pass
        rbw=42;rbh=16;rby=y+10
        renr=pygame.Rect(rx+cw-rbw*2-6,rby,rbw,rbh);delr=pygame.Rect(rx+cw-rbw-2,rby,rbw,rbh)
        reg(ed._ren_btn,lid,renr);reg(ed._del_btn,lid,delr)
        bx2=rx+pad+DW+4+tw2+4;bw2=(renr.x-4)-bx2;btnr=pygame.Rect(bx2,y+2,bw2,rh2-4)
        reg(ed._lv_btn,lid,btnr)
        hov=btnr.collidepoint(mpos);bg=UI_SELECT if cur else (UI_BTN_HOVER if hov else UI_BTN)
        pygame.draw.rect(surf,bg,btnr,border_radius=4)
        mc_id=max(1,(bw2-10)//8);mc_fn=max(1,(bw2-10)//7)
        lt=lid if len(lid)<=mc_id else lid[:mc_id-1]+'..';ft=fn if len(fn)<=mc_fn else fn[:mc_fn-1]+'..'
        try:
            sfont.render_to(surf,(btnr.x+6,y+7),lt,fgcolor=UI_TEXT,size=12)
            sfont.render_to(surf,(btnr.x+6,y+22),ft,fgcolor=UI_SUBTEXT,size=9)
        except:pass
        if cur:pygame.draw.rect(surf,UI_ACCENT,btnr,2,border_radius=4)
        hr2=renr.collidepoint(mpos);pygame.draw.rect(surf,UI_BTN_HOVER if hr2 else UI_BTN,renr,border_radius=3)
        pygame.draw.rect(surf,UI_ACCENT if hr2 else UI_BORDER,renr,1,border_radius=3)
        try:sfont.render_to(surf,(renr.x+4,renr.y+3),"Rename",fgcolor=UI_ACCENT if hr2 else UI_SUBTEXT,size=8)
        except:pass
        hd2=delr.collidepoint(mpos);pygame.draw.rect(surf,UI_DANGER if hd2 else UI_BTN,delr,border_radius=3)
        pygame.draw.rect(surf,UI_DANGER if hd2 else UI_BORDER,delr,1,border_radius=3)
        try:sfont.render_to(surf,(delr.x+5,delr.y+3),"Delete",fgcolor=UI_TEXT if hd2 else UI_SUBTEXT,size=8)
        except:pass
        y+=rh2+2
    # Commented levels
    cm=gc.get_commented_levels()
    if cm:
        y+=4
        try:sfont.render_to(surf,(rx+pad,y),"DISABLED",fgcolor=UI_DANGER,size=10)
        except:pass
        y+=16
        for lid in sorted(cm.keys()):
            dr2=pygame.Rect(rx+cw-50,y+5,48,16);reg(ed._del_btn,f"off:{lid}",dr2)
            br2=pygame.Rect(rx+pad+DW+4+38,y,cw-pad-(DW+4+38)-52,30)
            pygame.draw.rect(surf,UI_BTN_HOVER if br2.collidepoint(mpos) else UI_BTN,br2,border_radius=3)
            try:sfont.render_to(surf,(br2.x+8,y+4),lid,fgcolor=UI_SUBTEXT,size=11)
            except:pass
            cb2=pygame.Rect(rx+pad+DW+4,y+7,36,16);reg(ed._tog_btn,f"off:{lid}",cb2)
            pygame.draw.rect(surf,UI_BTN_HOVER if cb2.collidepoint(mpos) else (80,14,22),cb2,border_radius=4)
            pygame.draw.rect(surf,UI_DANGER,cb2,1,border_radius=4)
            try:sfont.render_to(surf,(cb2.x+4,cb2.y+3),"OFF",fgcolor=UI_DANGER,size=9)
            except:pass
            hd3=dr2.collidepoint(mpos);pygame.draw.rect(surf,UI_DANGER if hd3 else UI_BTN,dr2,border_radius=3)
            try:sfont.render_to(surf,(dr2.x+3,dr2.y+3),"Delete",fgcolor=UI_TEXT if hd3 else UI_SUBTEXT,size=8)
            except:pass
            y+=32
    # Unlisted files
    af=gc.get_all_level_files(extra_path=ed.level.filename)
    lf={cfg.get('file','') for cfg in list(gc.levels.values()) + list(gc.disabled_levels.values())}
    ul=[f for f in af if f.name not in lf]
    if ul:
        y+=6;pygame.draw.line(surf,UI_BORDER,(rx+pad,y),(rx+cw-pad,y),1);y+=6
        try:sfont.render_to(surf,(rx+pad,y),"UNLISTED FILES",fgcolor=UI_SUBTEXT,size=11)
        except:pass
        y+=14
        for fp in ul:
            dur=pygame.Rect(rx+cw-22,y+2,22,20);br2=pygame.Rect(rx+pad,y,cw-pad*2-26,24)
            hov=br2.collidepoint(mpos);cur2=str(fp)==ed.level.filename
            pygame.draw.rect(surf,UI_SELECT_DIM if cur2 else (UI_BTN_HOVER if hov else UI_BTN),br2,border_radius=3)
            try:sfont.render_to(surf,(br2.x+6,y+6),fp.name,fgcolor=UI_TEXT if cur2 else UI_SUBTEXT,size=10)
            except:pass
            reg(ed._ul_btn,str(fp),br2)
            hdu=dur.collidepoint(mpos);pygame.draw.rect(surf,UI_DANGER if hdu else UI_BTN,dur,border_radius=3)
            try:sfont.render_to(surf,(dur.x+5,dur.y+3),"x",fgcolor=UI_TEXT if hdu else UI_SUBTEXT,size=10)
            except:pass
            reg(ed._ul_del,str(fp),dur);y+=26
        if ed.level.filename and any(str(fp)==ed.level.filename for fp in ul):
            y+=3;ab=pygame.Rect(rx+pad,y,cw-pad*2,24)
            pygame.draw.rect(surf,UI_ACCENT if ab.collidepoint(mpos) else UI_BTN_ACT,ab,border_radius=4)
            try:sfont.render_to(surf,(rx+pad+8,y+6),"+ Assign to Level ID",fgcolor=UI_BG if ab.collidepoint(mpos) else UI_TEXT,size=10)
            except:pass
            ed._assign_r=ab;y+=28
        else:ed._assign_r=pygame.Rect(0,0,0,0)
    # Level info
    y+=6;pygame.draw.line(surf,UI_BORDER,(rx+pad,y),(rx+cw-pad,y),1);y+=6
    try:sfont.render_to(surf,(rx+pad,y),"LEVEL INFO",fgcolor=UI_ACCENT,size=12)
    except:pass
    y+=14;lv=ed.level;bb=lv.bounding_box()
    for i,(k,v) in enumerate([("Size",f"{lv.rows}r x {lv.cols}c"),("Plats",str(len(lv.platforms))),
        ("Export",f"{bb[2]-bb[0]+3}r x {bb[3]-bb[1]+3}c" if bb else "-"),("ID",lv.level_id or "-"),
        ("File",Path(lv.filename).name if lv.filename else "-"),
        ("Undo",f"{len(ed.undo_stack)}/Redo {len(ed.redo_stack)}")]):
        col2=i%2;row2=i//2;ix=rx+pad+col2*((cw-pad*2)//2);iy=y+row2*18
        try:
            sfont.render_to(surf,(ix,iy),k+":",fgcolor=UI_SUBTEXT,size=9)
            sfont.render_to(surf,(ix+32,iy),v,fgcolor=UI_TEXT,size=9)
        except:pass
    y+=54
    # Physics
    pygame.draw.line(surf,UI_BORDER,(rx+pad,y),(rx+cw-pad,y),1);y+=6
    try:sfont.render_to(surf,(rx+pad,y),"PHYSICS",fgcolor=UI_ACCENT,size=12)
    except:pass
    y+=14;sw=cw-pad*2-4
    for sl in ed.phys_sliders:
        try:
            sfont.render_to(surf,(rx+pad+2,y),sl.label,fgcolor=UI_SUBTEXT,size=9)
            vs=f"{sl.value:.0f}";vb=sfont.get_rect(vs,size=9)
            sfont.render_to(surf,(rx+cw-pad-vb.width-2,y),vs,fgcolor=UI_TEXT,size=9)
        except:pass
        y+=13;tr=pygame.Rect(rx+pad+2,y,sw,8)
        pygame.draw.rect(surf,UI_PANEL2,tr,border_radius=4);pygame.draw.rect(surf,UI_BORDER,tr,1,border_radius=4)
        fr=sl.fraction();fw=int(fr*(sw-4))
        if fw>0:pygame.draw.rect(surf,UI_ACCENT,(tr.x+2,tr.y+2,fw,4),border_radius=2)
        tx2=tr.x+2+int(fr*(sw-4));pygame.draw.rect(surf,UI_TEXT,(tx2-4,tr.y-2,8,12),border_radius=3)
        sl.rect=tr;y+=16
    # Enemy Physics
    pygame.draw.line(surf,UI_BORDER,(rx+pad,y),(rx+cw-pad,y),1);y+=6
    try:sfont.render_to(surf,(rx+pad,y),"ENEMY PHYSICS",fgcolor=(220,140,80),size=12)
    except:pass
    y+=14
    for sl in ed.enemy_sliders:
        val_str=f"{sl.value/100:.2f}x" if sl.key=="enemy_grav_pct" else f"{sl.value:.0f}"
        try:
            sfont.render_to(surf,(rx+pad+2,y),sl.label,fgcolor=UI_SUBTEXT,size=9)
            vb=sfont.get_rect(val_str,size=9)
            sfont.render_to(surf,(rx+cw-pad-vb.width-2,y),val_str,fgcolor=(220,180,120),size=9)
        except:pass
        y+=13;tr=pygame.Rect(rx+pad+2,y,sw,8)
        pygame.draw.rect(surf,UI_PANEL2,tr,border_radius=4);pygame.draw.rect(surf,UI_BORDER,tr,1,border_radius=4)
        fr=sl.fraction();fw=int(fr*(sw-4))
        if fw>0:pygame.draw.rect(surf,(200,110,40),(tr.x+2,tr.y+2,fw,4),border_radius=2)
        tx2=tr.x+2+int(fr*(sw-4));pygame.draw.rect(surf,(220,180,120),(tx2-4,tr.y-2,8,12),border_radius=3)
        sl.rect=tr;y+=16
    ch=(y+ed.right_scroll)-ry;ed.right_max_scroll=max(0,ch-rh+16)
    ed.right_scroll=min(ed.right_scroll,ed.right_max_scroll);surf.set_clip(None)
    # Scrollbar
    pygame.draw.rect(surf,UI_PANEL2,(sbx,ry,SB_W,rh));pygame.draw.line(surf,UI_BORDER,(sbx,ry),(sbx,ry+rh),1)
    if ed.right_max_scroll>0:
        rat=rh/max(1,ch);th2=max(28,int(rh*rat));sr=max(1,rh-th2)
        ty=ry+int(ed.right_scroll/max(1,ed.right_max_scroll)*sr)
        tr=pygame.Rect(sbx+2,ty,SB_W-4,th2)
        hov=tr.collidepoint(mpos) or ed._sb_drag
        pygame.draw.rect(surf,UI_ACCENT if hov else UI_BORDER,tr,border_radius=4);ed._sb_thumb=tr
    else:ed._sb_thumb=pygame.Rect(0,0,0,0)

def build_tb():
    return [
        TBBtn("New","Ctrl+N",lambda e:"NEW"),TBBtn("Open","Ctrl+O",lambda e:"OPEN"),
        TBBtn("Save","Ctrl+S",lambda e:"SAVE"),TBBtn("|","",None),
        TBBtn("Undo","Ctrl+Z",lambda e:e.undo()),TBBtn("Redo","Ctrl+Y",lambda e:e.redo()),
        TBBtn("|","",None),TBBtn("Resize","Ctrl+R",lambda e:"RESIZE"),
        TBBtn("Clear","Ctrl+X",lambda e:e.clear_all()),TBBtn("|","",None),
        TBBtn("Grid","G",None,True,lambda e:e.show_grid),TBBtn("Fill","F",None,True,lambda e:e.fill_mode),
        TBBtn("Eraser","X",None,True,lambda e:e.eraser_mode),TBBtn("|","",None),
        TBBtn("Home","Home",lambda e:e.reset_view()),TBBtn("|","",None),
        TBBtn("Play","Ctrl+P",lambda e:"PLAY"),
    ]

def draw_toolbar(surf,ed,sfont,btns,mpos):
    pygame.draw.rect(surf,UI_TOOLBAR,(0,0,WINDOW_W,TOOLBAR_H))
    pygame.draw.line(surf,UI_BORDER,(0,TOOLBAR_H-1),(WINDOW_W,TOOLBAR_H-1),1)
    try:
        sfont.render_to(surf,(10,8),"PEAK",fgcolor=UI_ACCENT,size=18)
        sfont.render_to(surf,(10,30),f"Level Editor  {ed.game_label}",fgcolor=UI_SUBTEXT,size=11)
    except:pass
    title=""
    if ed.level.filename:title=Path(ed.level.filename).name
    if ed.level.level_id:title=f"[{ed.level.level_id}]  {title}"
    if ed.dirty:title+="  *"
    if title:
        try:sfont.render_to(surf,(115,18),title,fgcolor=UI_TEXT,size=13)
        except:pass
    bx=LEFT_PANEL_W+12;by=8;bh=34
    for btn in btns:
        if btn.label=="|":
            pygame.draw.line(surf,UI_BORDER,(bx+4,by+4),(bx+4,by+bh-4),1);bx+=12;continue
        try:tw=sfont.get_rect(btn.label,size=11).width
        except:tw=30
        bw2=max(44,tw+20);btn.rect=pygame.Rect(bx,by,bw2,bh)
        hov=btn.rect.collidepoint(mpos);act=btn.get_active(ed) if btn.get_active else False
        is_play=btn.label=="Play"
        if is_play:bg=(30,90,50) if hov else (20,65,35)
        elif btn.label=="Eraser" and act:bg=UI_DANGER
        elif act:bg=UI_BTN_ACT
        elif hov:bg=UI_BTN_HOVER
        else:bg=UI_BTN
        pygame.draw.rect(surf,bg,btn.rect,border_radius=5)
        if is_play:pygame.draw.rect(surf,(60,200,100),btn.rect,2,border_radius=5)
        elif act:pygame.draw.rect(surf,UI_ACCENT if btn.label!="Eraser" else UI_DANGER,btn.rect,2,border_radius=5)
        try:
            lb=sfont.get_rect(btn.label,size=11)
            fc=(150,255,170) if is_play else (UI_TEXT if (act or hov) else UI_SUBTEXT)
            sfont.render_to(surf,(bx+(bw2-lb.width)//2,by+(bh-lb.height)//2),btn.label,fgcolor=fc,size=11)
        except:pass
        if btn.sc and hov:
            try:
                sb=sfont.get_rect(btn.sc,size=8)
                sfont.render_to(surf,(bx+(bw2-sb.width)//2,by+bh-10),btn.sc,fgcolor=(80,90,110),size=8)
            except:pass
        bx+=bw2+4

def draw_status(surf,ed,sfont):
    y=WINDOW_H-STATUS_H;pygame.draw.rect(surf,UI_STATUS,(0,y,WINDOW_W,STATUS_H))
    pygame.draw.line(surf,UI_BORDER,(0,y),(WINDOW_W,y),1)
    lv=ed.level
    tl="Eraser" if ed.eraser_mode else ("Platform" if ed.selected_tile==MOVING_PLATFORM_BRUSH else TILE_BY_CHAR.get(ed.selected_tile,TILE_BY_CHAR[' '])[1])
    info=f"  {ed.game_label}  |  {lv.rows}r x {lv.cols}c  |  Zoom {ed.zoom:.2f}x  |  Brush: {tl}  |  Plats: {len(lv.platforms)}"
    if ed.hover_cell and ed.selected_tile!=MOVING_PLATFORM_BRUSH:
        hr,hc=ed.hover_cell;ch=lv.get(hr,hc)
        info+=f"  |  [{hr},{hc}]: {TILE_BY_CHAR.get(ch,TILE_BY_CHAR[' '])[1] if ch else 'Air'}"
    bb=lv.bounding_box()
    if bb:info+=f"  |  Export: {bb[2]-bb[0]+3}r x {bb[3]-bb[1]+3}c"
    try:sfont.render_to(surf,(LEFT_PANEL_W,y+7),info,fgcolor=UI_SUBTEXT,size=11)
    except:pass

# ─── Playtest ─────────────────────────────────────────────────────
def launch_play(ed,scr,sf):
    lv=ed.level
    # Bake current slider values into the level before saving (same as SAVE action)
    lv.enemy_physics={
        'walk_speed':   ed.enemy_sliders[0].value,
        'gravity_mult': round(ed.enemy_sliders[1].value/100.0, 3),
        'max_fall_speed': ed.enemy_sliders[2].value,
        'patrol_width': ed.enemy_sliders[3].value,
    }
    if lv.filename:lv.save(lv.filename);ed.dirty=False
    else:
        p=dlg_save(scr,sf)
        if not p:return
        lv.save(p);ed.dirty=False
    if not lv.filename:return
    def fpr(s):
        c=s.resolve()
        for _ in range(10):
            if (c/'code'/'__init__.py').exists() or (c/'code').is_dir():return c
            p=c.parent
            if p==c:break
            c=p
        return s.resolve()
    sd=Path(__file__).resolve().parent;pr=fpr(sd)
    ps=next((str(p) for p in [sd/'manual_play.py',pr/'code'/'scripts'/'manual_play.py',pr/'manual_play.py'] if Path(p).exists()),None)
    if not ps:print("[Editor] manual_play.py not found");return
    env=os.environ.copy();env['PYTHONPATH']=(str(pr)+os.pathsep+env.get('PYTHONPATH','')).rstrip(os.pathsep)
    # Always pass --file so manual_play loads exactly the file being edited,
    # regardless of whether it's registered in game_config.yaml.
    cmd=[sys.executable,ps,'--game',ed.gc.game,'--file',str(Path(lv.filename).resolve())]
    subprocess.Popen(cmd,env=env,cwd=str(pr))

# ─── Toolbar action dispatcher ───────────────────────────────────
def do_action(result,ed,scr,sf):
    if result=="NEW":
        if not ed.check_unsaved(scr,sf):return
        r,c=dlg_new(scr,sf)
        if r and c:
            ed.level=Level(r,c)
            for r2 in range(r):ed.level.set(r2,0,'#');ed.level.set(r2,c-1,'#')
            for c2 in range(c):ed.level.set(0,c2,'#');ed.level.set(r-1,c2,'#')
            ed.level.set(r-2,2,'P')
            ed.undo_stack.clear();ed.redo_stack.clear();ed.dirty=False;ed._center()
    elif result=="OPEN":
        if not ed.check_unsaved(scr,sf):return
        p=dlg_open(scr,sf)
        if p:
            try:ed.level=Level.load(p);ed.undo_stack.clear();ed.redo_stack.clear();ed.dirty=False;ed._center()
            except Exception as e:print(f"[Editor] {e}")
    elif result=="SAVE":
        p=ed.level.filename or dlg_save(scr,sf)
        if p:
            # Bake slider values back into the level before writing
            ed.level.enemy_physics={
                'walk_speed':   ed.enemy_sliders[0].value,
                'gravity_mult': round(ed.enemy_sliders[1].value/100.0, 3),
                'max_fall_speed': ed.enemy_sliders[2].value,
                'patrol_width': ed.enemy_sliders[3].value,
            }
            ed.level.save(p); ed.dirty=False; print(f"[Editor] Saved: {p}")
    elif result=="RESIZE":
        r,c=dlg_resize(scr,sf,ed.level.rows,ed.level.cols)
        if r and c:ed.resize_grid(r,c);ed._center()
    elif result=="PLAY":
        launch_play(ed,scr,sf)

# ═══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="PEAK level editor")
    ap.add_argument("level", nargs="?", help="Optional level file to open")
    ap.add_argument("--game", choices=sorted(GAME_LABELS.keys()), default=DEFAULT_GAME, help="Game ruleset/palette to edit")
    return ap.parse_args(argv)

def main(argv=None):
    global WINDOW_W,WINDOW_H
    args=parse_args(argv)
    pygame.init();pygame.freetype.init()
    screen=pygame.display.set_mode((WINDOW_W,WINDOW_H),pygame.RESIZABLE)
    pygame.display.set_caption(f"PEAK Level Editor v3.1 - {GAME_LABELS.get(args.game, args.game)}")
    font=pygame.freetype.SysFont("monospace",14);sfont=pygame.freetype.SysFont("monospace",14)
    clock=pygame.time.Clock()
    gc=GameConfig(args.game)
    if args.level:
        lv=Level.load(args.level); lv.level_id=next((lid for lid,cfg in gc.levels.items() if Path(cfg.get('file','')).name==Path(args.level).name), None)
    else:
        lv=Level(DEFAULT_ROWS,DEFAULT_COLS)
        for r in range(DEFAULT_ROWS):lv.set(r,0,'#');lv.set(r,DEFAULT_COLS-1,'#')
        for c in range(DEFAULT_COLS):lv.set(0,c,'#');lv.set(DEFAULT_ROWS-1,c,'#')
        lv.set(DEFAULT_ROWS-2,2,'P')
    ed=Editor(lv,gc);btns=build_tb()
    pstarted=False;running=True
    while running:
        clock.tick(60);WINDOW_W,WINDOW_H=screen.get_size()
        mpos=pygame.mouse.get_pos();mx,my=mpos
        in_vp=LEFT_PANEL_W<mx<WINDOW_W-RIGHT_PANEL_W and TOOLBAR_H<my<WINDOW_H-STATUS_H-HSB_H
        in_rp=mx>WINDOW_W-RIGHT_PANEL_W;in_lp=mx<LEFT_PANEL_W;in_tb=my<TOOLBAR_H
        if in_vp:
            r,c=ed.s2c(mx,my);ed.hover_cell=(r,c) if 0<=r<ed.level.rows and 0<=c<ed.level.cols else None
        else:ed.hover_cell=None
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:
                if ed.check_unsaved(screen,sfont):running=False
            elif ev.type==pygame.VIDEORESIZE:WINDOW_W,WINDOW_H=ev.w,ev.h
            elif ev.type==pygame.MOUSEWHEEL:
                if in_vp:
                    oz=ed.zoom;ed.zoom=max(MIN_ZOOM,min(MAX_ZOOM,ed.zoom*(1.1**ev.y)));s=ed.zoom/oz
                    ed.cam_x=mx-LEFT_PANEL_W+(ed.cam_x-(mx-LEFT_PANEL_W))*s
                    ed.cam_y=my-TOOLBAR_H+(ed.cam_y-(my-TOOLBAR_H))*s
                elif in_rp:ed.right_scroll=max(0,min(ed.right_max_scroll,ed.right_scroll-ev.y*20))
            elif ev.type==pygame.MOUSEBUTTONDOWN:
                ex,ey=ev.pos
                if in_rp and ed._sb_thumb.collidepoint(ex,ey):
                    ed._sb_drag=True;ed._sb_dy=ey;ed._sb_ds=ed.right_scroll;continue
                if not in_rp and not in_lp and ed._hs_thumb.collidepoint(ex,ey):
                    ed._hs_drag=True;ed._hs_dx=ex;ed._hs_dc=ed.cam_x;continue
                if ev.button==1:
                    # TOOLBAR CLICKS (FIX from v3.0)
                    if in_tb:
                        for btn in btns:
                            if btn.label=="|":continue
                            if btn.rect.collidepoint(ex,ey):
                                if btn.toggle:
                                    if btn.label=="Grid":ed.show_grid=not ed.show_grid
                                    elif btn.label=="Fill":ed.fill_mode=not ed.fill_mode
                                    elif btn.label=="Eraser":ed.eraser_mode=not ed.eraser_mode
                                elif btn.action:
                                    r2=btn.action(ed)
                                    if isinstance(r2,str):do_action(r2,ed,screen,sfont)
                                break
                        continue
                    if in_rp:
                        h=False
                        if not h:
                            for lid,rect in ed._dh_rects.items():
                                if rect.collidepoint(ex,ey):ed.start_lv_drag(lid,ed.gc.level_ids.index(lid),ey);h=True;break
                        if not h:
                            for lid,rect in list(ed._tog_btn.items()):
                                if rect.collidepoint(ex,ey):
                                    gc.toggle_level_in_config(lid[4:] if lid.startswith("off:") else lid,lid.startswith("off:"))
                                    h=True;break
                        if not h:
                            for lid,rect in ed._ren_btn.items():
                                if rect.collidepoint(ex,ey):
                                    ni=dlg_assign(screen,sfont,lid,f"Rename '{lid}' to:")
                                    if ni and ni!=lid:gc.assign_stage_to_level(gc.levels.get(lid,{}).get('file',''),ni);gc.delete_level_from_config(lid)
                                    h=True;break
                        if not h:
                            for lid,rect in ed._del_btn.items():
                                if rect.collidepoint(ex,ey):
                                    rl=lid[4:] if lid.startswith("off:") else lid;gc.delete_level_from_config(rl)
                                    if ed.browser_level_id==rl:ed.browser_level_id=None
                                    h=True;break
                        if not h:
                            for lid,rect in ed._lv_btn.items():
                                if rect.collidepoint(ex,ey):
                                    if ed.check_unsaved(screen,sfont):ed.load_by_id(lid)
                                    h=True;break
                        if not h:
                            for fp,rect in ed._ul_btn.items():
                                if rect.collidepoint(ex,ey):
                                    if ed.check_unsaved(screen,sfont):
                                        try:ed.level=Level.load(fp);ed.undo_stack.clear();ed.redo_stack.clear();ed.dirty=False;ed._center()
                                        except:pass
                                    h=True;break
                        if not h:
                            for fp,rect in ed._ul_del.items():
                                if rect.collidepoint(ex,ey):
                                    try:os.remove(fp)
                                    except:pass
                                    h=True;break
                        if not h and ed._assign_r.collidepoint(ex,ey) and ed.level.filename:
                            lid=dlg_assign(screen,sfont,ed.level.filename)
                            if lid:gc.assign_stage_to_level(Path(ed.level.filename).name,lid);ed.level.level_id=lid;ed.browser_level_id=lid
                            h=True
                        if not h:
                            for sl in ed.phys_sliders:
                                if sl.rect.collidepoint(ex,ey):sl.dragging=True;sl.set_from_fraction((ex-sl.rect.x)/max(1,sl.rect.width));h=True;break
                        if not h:
                            for sl in ed.enemy_sliders:
                                if sl.rect.collidepoint(ex,ey):sl.dragging=True;sl.set_from_fraction((ex-sl.rect.x)/max(1,sl.rect.width));h=True;break
                        if not h:
                            if ed._spd_m.collidepoint(ex,ey):ed.plat_default_spd=max(10,ed.plat_default_spd-10)
                            elif ed._spd_p.collidepoint(ex,ey):ed.plat_default_spd=min(500,ed.plat_default_spd+10)
                    elif in_lp:
                        for key,rect in ed._tool_rects.items():
                            if rect.collidepoint(ex,ey):
                                if key=="eraser":ed.eraser_mode=not ed.eraser_mode
                                elif key=="fill":ed.fill_mode=not ed.fill_mode
                                elif key=="grid":ed.show_grid=not ed.show_grid
                                elif key=="platform":ed.selected_tile=MOVING_PLATFORM_BRUSH;ed.plat_placing=False;ed.eraser_mode=False
                                break
                        for ch,rect in ed._tile_rects.items():
                            if rect.collidepoint(ex,ey):ed.selected_tile=ch;ed.plat_placing=False;ed.eraser_mode=False;break
                    elif in_vp:
                        if ed.selected_tile==MOVING_PLATFORM_BRUSH and not ed.eraser_mode:
                            hi,hh=ed.ht_plat(ex,ey)
                            if hi is not None:
                                ed.sel_plat_idx=hi;ed.drag_handle=hh;p=ed.level.platforms[hi]
                                wp=p.start if hh=='start' else p.end
                                hsx,hsy=ed.wp2s(wp[0]+p.width/2,wp[1]+p.height/2);ed.drag_offset=(hsx-ex,hsy-ey)
                            elif not ed.plat_placing:
                                wx,wy=ed.s2wp(ex,ey,snap=True);ed.plat_start_world=(wx,wy);ed.plat_ghost_end=(wx,wy)
                                ed.plat_placing=True;ed.sel_plat_idx=None
                            else:
                                wx,wy=ed.s2wp(ex,ey,snap=True);ed.push_undo()
                                ed.level.platforms.append(PlatformDef(list(ed.plat_start_world),[wx,wy],speed=ed.plat_default_spd))
                                ed.sel_plat_idx=len(ed.level.platforms)-1;ed.plat_placing=False;ed.dirty=True
                        elif ed.hover_cell:
                            r,c=ed.hover_cell;brush=ed.brush()
                            if ed.fill_mode:ed.flood_fill(r,c,brush)
                            else:
                                if not pstarted:ed.push_undo();pstarted=True
                                ed.paint_char=brush;ed.painting=True;ed.last_cell=None;ed.paint_cell(r,c,ed.paint_char)
                elif ev.button==3 and in_vp:
                    if ed.selected_tile==MOVING_PLATFORM_BRUSH:ed.plat_placing=False;ed.sel_plat_idx=None
                    elif ed.hover_cell:
                        if not pstarted:ed.push_undo();pstarted=True
                        ed.paint_char=' ';ed.painting=True;ed.last_cell=None;ed.paint_cell(ed.hover_cell[0],ed.hover_cell[1],' ')
                elif ev.button==2:ed.panning=True;ed.pan_start=ev.pos;ed.pan_cam=(ed.cam_x,ed.cam_y)
            elif ev.type==pygame.MOUSEBUTTONUP:
                if ev.button==1:
                    if ed._drag_lid:ed.finish_lv_drag()
                    ed.painting=False;ed.last_cell=None;pstarted=False;ed.drag_handle=None
                    ed._sb_drag=False;ed._hs_drag=False
                    for sl in ed.phys_sliders:sl.dragging=False
                    for sl in ed.enemy_sliders:sl.dragging=False
                elif ev.button==2:ed.panning=False
            elif ev.type==pygame.MOUSEMOTION:
                if ed._drag_lid:
                    sy2=my+ed.right_scroll-ed._lv_list_top
                    ed._drag_di=max(0,min(len(ed.gc.level_ids)-1,sy2//ed._lv_row_h));ed._drag_y=my
                for sl in ed.phys_sliders:
                    if sl.dragging:sl.set_from_fraction((ev.pos[0]-sl.rect.x)/max(1,sl.rect.width))
                for sl in ed.enemy_sliders:
                    if sl.dragging:sl.set_from_fraction((ev.pos[0]-sl.rect.x)/max(1,sl.rect.width))
                if ed._sb_drag:
                    dy=ev.pos[1]-ed._sb_dy;rh2=WINDOW_H-TOOLBAR_H-STATUS_H
                    th=ed._sb_thumb.height if ed._sb_thumb.height>0 else 40
                    sc=ed.right_max_scroll/max(1,rh2-th)
                    ed.right_scroll=max(0,min(ed.right_max_scroll,ed._sb_ds+int(dy*sc)))
                if ed._hs_drag:
                    dx=ev.pos[0]-ed._hs_dx;vw2=WINDOW_W-LEFT_PANEL_W-RIGHT_PANEL_W
                    tw=ed._hs_thumb.width if ed._hs_thumb.width>0 else 40
                    mc=max(0,ed.level.cols*TILE_SIZE*ed.zoom-vw2)
                    sh2=mc/max(1,vw2-tw);ed.cam_x=max(0.0,min(float(mc),ed._hs_dc+dx*sh2))
                if ed.selected_tile==MOVING_PLATFORM_BRUSH:
                    if ed.drag_handle is not None and ed.sel_plat_idx is not None:
                        wx,wy=ed.s2wp(mpos[0]+ed.drag_offset[0],mpos[1]+ed.drag_offset[1],snap=True)
                        p=ed.level.platforms[ed.sel_plat_idx];wp=[wx-p.width/2,wy-p.height/2]
                        if ed.drag_handle=='start':p.start=wp
                        else:p.end=wp
                        ed.dirty=True
                    elif ed.plat_placing and in_vp:ed.plat_ghost_end=ed.s2wp(mpos[0],mpos[1],snap=True)
                if ed.painting and in_vp and ed.hover_cell:ed.paint_cell(ed.hover_cell[0],ed.hover_cell[1],ed.paint_char)
                if ed.panning:ed.cam_x=ed.pan_cam[0]-(ev.pos[0]-ed.pan_start[0]);ed.cam_y=ed.pan_cam[1]-(ev.pos[1]-ed.pan_start[1])
            elif ev.type==pygame.KEYDOWN:
                mods=pygame.key.get_mods();ctrl=mods&pygame.KMOD_CTRL
                if ctrl and ev.key==pygame.K_s:do_action("SAVE",ed,screen,sfont)
                elif ctrl and ev.key==pygame.K_p:do_action("PLAY",ed,screen,sfont)
                elif ctrl and ev.key==pygame.K_o:do_action("OPEN",ed,screen,sfont)
                elif ctrl and ev.key==pygame.K_z:
                    if mods&pygame.KMOD_SHIFT:ed.redo()
                    else:ed.undo()
                elif ctrl and ev.key==pygame.K_y:ed.redo()
                elif ctrl and ev.key==pygame.K_n:do_action("NEW",ed,screen,sfont)
                elif ctrl and ev.key==pygame.K_r:do_action("RESIZE",ed,screen,sfont)
                elif ctrl and ev.key==pygame.K_x:ed.clear_all()
                elif ev.key==pygame.K_F5:do_action("RESIZE",ed,screen,sfont)
                elif ev.key==pygame.K_HOME:ed.reset_view()
                elif ev.key==pygame.K_g:ed.show_grid=not ed.show_grid
                elif ev.key==pygame.K_f:ed.fill_mode=not ed.fill_mode
                elif ev.key==pygame.K_x and not ctrl:ed.eraser_mode=not ed.eraser_mode
                elif ev.key==pygame.K_v:ed.selected_tile=MOVING_PLATFORM_BRUSH;ed.plat_placing=False;ed.eraser_mode=False
                elif ev.key==pygame.K_ESCAPE:
                    if ed._drag_lid:ed.cancel_lv_drag()
                    elif ed.plat_placing:ed.plat_placing=False
                    elif ed.eraser_mode:ed.eraser_mode=False
                elif ev.key==pygame.K_DELETE:
                    if ed.selected_tile==MOVING_PLATFORM_BRUSH and ed.sel_plat_idx is not None and 0<=ed.sel_plat_idx<len(ed.level.platforms):
                        ed.push_undo();ed.level.platforms.pop(ed.sel_plat_idx);ed.sel_plat_idx=None;ed.dirty=True
                else:
                    for i,k in enumerate([pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,pygame.K_5,
                                          pygame.K_6,pygame.K_7,pygame.K_8,pygame.K_9,pygame.K_0]):
                        if ev.key==k and i<len(ed.tile_entries):ed.selected_tile=ed.tile_entries[i][0];ed.eraser_mode=False
        screen.fill(UI_BG)
        draw_viewport(screen,ed,font,sfont);draw_toolbar(screen,ed,sfont,btns,mpos)
        draw_left_panel(screen,ed,font,sfont);draw_right_panel(screen,ed,sfont,mpos)
        draw_status(screen,ed,sfont);pygame.display.flip()
    pygame.quit()

if __name__=="__main__":
    main()
