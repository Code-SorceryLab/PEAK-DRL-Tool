import pygame
from ...Parameters.Map_parameters import (TILE_SIZE, TILE_AIR, TILE_SPIKE, TILE_GOAL)

# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT  (all values in pixels, relative to debug panel left edge)
#
#  ┌──────────────────────────────────┐  ← debug panel (350px wide)
#  │  BANNER                  36px    │
#  ├──────────────────────────────────┤
#  │  Agent Vision card       ~290px  │
#  ├──────────────┬───────────────────┤
#  │ Player Info  │  Obs Values       │
#  │   (left)     │  (right)          │
#  ├──────────────┴───────────────────┤
#  │  Reward Trace  (rest of height)  │
#  └──────────────────────────────────┘
# ═══════════════════════════════════════════════════════════════════════════════

_PAD       = 7
_BANNER_H  = 36
_GAP       = 6
_HDR_H     = 20        # card section header height

# ── Agent Vision ──────────────────────────────────────────────────────────────
_AV_CELL   = 12        # 21 × 12 = 252px
_HUD_STRIP_H = 40      # height of the lives/score HUD strip added below the banner
_AV_Y      = _BANNER_H + _GAP + _HUD_STRIP_H + _GAP   # shifted down for HUD strip

_AV_LEGEND_W = 54   # left legend column width (sq + label)
_AV_LEGEND_SEP = 3  # gap between legend col and grid

def _av_card_h(core):
    return _HDR_H + getattr(core, 'obs_height', 21) * _AV_CELL + _PAD

# ── Lower two-column zone ─────────────────────────────────────────────────────
_INFO_LINE_H  = 15
_INFO_LINES   = 5
_VEC_LINE_H   = 12
_N_OBS        = 16     # 5 player + 11 tracking

def _lower_y(core):
    return _AV_Y + _av_card_h(core) + _GAP

def _lower_h():
    info_h = _HDR_H + _INFO_LINES * _INFO_LINE_H + _PAD
    vec_h  = _HDR_H + ((_N_OBS + 1) // 2) * _VEC_LINE_H + _PAD
    return max(info_h, vec_h)

# ── Reward strip ──────────────────────────────────────────────────────────────
def _reward_y(core):
    return _lower_y(core) + _lower_h() + _GAP

# ═══════════════════════════════════════════════════════════════════════════════
# Colour palette
# ═══════════════════════════════════════════════════════════════════════════════
_C_BG      = ( 17,  17,  23)
_C_CARD    = ( 22,  22,  30)
_C_HDR     = ( 30,  30,  42)
_C_BORDER  = ( 50,  52,  70)
_C_SEP     = ( 40,  42,  58)
_C_LBL     = (105, 108, 128)
_C_VAL     = (210, 215, 228)
_C_POS     = ( 85, 190, 255)
_C_NEG     = (255,  85,  75)
_C_ACT     = ( 80, 225, 115)
_C_ACCENT  = ( 65,  95, 200)

# Ray colours — distinct and highly visible against sky/ground
RAY_EMPTY  = (255, 255, 255, 80)    # translucent white
RAY_SOLID  = (255, 120,  30)        # orange  — walls
RAY_HAZARD = (255,  40,  40)        # red     — spikes/enemies
RAY_COIN   = ( 50, 255, 220)        # cyan    — coins (pops against gold sprite)
RAY_GOAL   = (140, 255,  80)        # lime    — goal

# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _game_surf(surface, core):
    if core.render_mode == "human":
        return surface.subsurface(pygame.Rect(0, 0, core.WIDTH, core.HEIGHT))
    return surface


def _card(surface, x, y, w, h):
    bg = pygame.Surface((w, h))
    bg.fill(_C_CARD)
    bg.set_alpha(245)
    surface.blit(bg, (x, y))
    pygame.draw.rect(surface, _C_BORDER, (x, y, w, h), 1)


def _section_hdr(surface, font, title, x, y, w, accent=None):
    """Draw a section header bar. Returns y for content start."""
    hdr = pygame.Surface((w, _HDR_H))
    hdr.fill(_C_HDR)
    surface.blit(hdr, (x, y))
    if accent:
        pygame.draw.rect(surface, accent, (x, y, 3, _HDR_H))
    pygame.draw.line(surface, _C_SEP, (x, y + _HDR_H - 1), (x + w, y + _HDR_H - 1))
    t = font.render(title, True, (170, 172, 205))
    surface.blit(t, (x + (8 if not accent else 10), y + (_HDR_H - t.get_height()) // 2))
    return y + _HDR_H


# ═══════════════════════════════════════════════════════════════════════════════
# Overlay classes
# ═══════════════════════════════════════════════════════════════════════════════

class DebugOverlay:
    def render(self, surface: pygame.Surface, core):
        raise NotImplementedError


class HitboxOverlay(DebugOverlay):
    def render(self, surface, core):
        gs = _game_surf(surface, core)
        player = core.player
        if player:
            px, py, _ = core._world_to_screen(player.gObj)
            r = pygame.Rect(px, py, player.gObj.width, player.gObj.height).inflate(6, 6)
            # Dashed-style: draw 4 corner brackets
            blen = 7
            col = (60, 235, 60)
            for bx, by, dx, dy in [(r.left, r.top, 1, 1), (r.right, r.top, -1, 1),
                                    (r.left, r.bottom, 1, -1), (r.right, r.bottom, -1, -1)]:
                pygame.draw.line(gs, col, (bx, by), (bx + dx*blen, by), 2)
                pygame.draw.line(gs, col, (bx, by), (bx, by + dy*blen), 2)

        pm = core.physics_manager
        cx, cy, cw, ch = core.camera_x, core.camera_y, core.WIDTH, core.HEIGHT
        for entity in (pm.hazard_hash.query_rect(cx, cy, cw, ch) +
                       pm.collectible_hash.query_rect(cx, cy, cw, ch)):
            gObj = entity.gObj if hasattr(entity, 'gObj') else entity
            sx, sy, _ = core._world_to_screen(gObj)
            # Coins = cyan outline, hazards = red
            color = ( 50, 235, 200) if hasattr(entity, 'kind') else (255, 55, 55)
            pygame.draw.rect(gs, color, (sx, sy, gObj.width, gObj.height), 1)


class GridOverlay(DebugOverlay):
    def render(self, surface, core):
        gs    = _game_surf(surface, core)
        sc    = int(core.camera_x // TILE_SIZE)
        ec    = int((core.camera_x + core.WIDTH)  // TILE_SIZE) + 1
        sr    = int(core.camera_y // TILE_SIZE)
        er    = int((core.camera_y + core.HEIGHT) // TILE_SIZE) + 1
        alpha = pygame.Surface((core.WIDTH, core.HEIGHT), pygame.SRCALPHA)
        for col in range(sc, ec):
            x = col * TILE_SIZE - core.camera_x
            pygame.draw.line(alpha, (200, 210, 255, 22), (x, 0), (x, core.HEIGHT))
        for row in range(sr, er):
            y = row * TILE_SIZE - core.camera_y
            pygame.draw.line(alpha, (200, 210, 255, 22), (0, y), (core.WIDTH, y))
        gs.blit(alpha, (0, 0))


class AgentViewOverlay(DebugOverlay):
    _TC = {
        'wall':   ( 70,  76,  90), 'spike': (160,  36,  36),
        'goal':   ( 35, 160,  65), 'empty': (  8,   8,  14),
        'oob':    (  4,   4,   6), 'enemy': (215,  45,  45),
        'coin':   (215, 175,  18), 'player':( 35, 135, 255),
    }
    _BC = {
        'enemy':  (255, 100, 100), 'coin':  (255, 225,  70),
        'goal':   ( 70, 255, 110), 'player':(100, 190, 255),
        'default':( 32,  34,  44),
    }
    # Legend items shown as vertical strip on LEFT of grid
    _LEGEND = [
        ('player', "You"),
        ('wall',   "Wall"),
        ('coin',   "Coin"),
        ('enemy',  "Enemy"),
        ('goal',   "Goal"),
        ('spike',  "Spike"),
    ]
    _tiny_font = None  # lazily initialised 8px font for cost numbers

    def _get_tiny_font(self):
        if self._tiny_font is None:
            AgentViewOverlay._tiny_font = pygame.font.SysFont("arial", 8)
        return self._tiny_font

    def render(self, surface, core):
        cols = getattr(core, 'obs_width',  21)
        rows = getattr(core, 'obs_height', 21)
        cw = ch = _AV_CELL

        # Card width = legend col + sep + grid + padding
        grid_w  = cols * cw
        panel_w = _AV_LEGEND_W + _AV_LEGEND_SEP + grid_w + _PAD * 2
        panel_h = _av_card_h(core)
        px = core.DEBUG_PANEL_X + _PAD
        py = _AV_Y

        _card(surface, px, py, panel_w, panel_h)
        cy = _section_hdr(surface, core.debug_manager.font,
                          "Agent Vision", px, py, panel_w, accent=_C_ACCENT)

        if not core.player:
            return

        p_cx = int(core.player.gObj.x // TILE_SIZE)
        p_cy = int(core.player.gObj.y // TILE_SIZE)
        ox = p_cx - core.obs_pad_x
        oy = p_cy - core.obs_pad_y

        enemy_locs = {(int(e.gObj.x//TILE_SIZE), int(e.gObj.y//TILE_SIZE))
                      for e in core.level_data.enemies if e.gObj.active}
        coin_locs  = {(int(c.gObj.x//TILE_SIZE), int(c.gObj.y//TILE_SIZE))
                      for c in core.level_data.coins
                      if c.gObj.active and not getattr(c, 'collected', False)}
        goal_locs  = {(int(g.gObj.x//TILE_SIZE), int(g.gObj.y//TILE_SIZE))
                      for g in core.level_data.goals if g.gObj.active}

        tc, bc = self._TC, self._BC

        # ── LEFT: vertical legend strip ───────────────────────────────────────
        legend_x = px + _PAD
        lf = core.debug_manager.small_font
        sq = 9
        gap = 4
        total_legend_h = len(self._LEGEND) * (sq + gap)
        ly_start = cy + (rows * ch - total_legend_h) // 2   # vertically centre

        for i, (key, label) in enumerate(self._LEGEND):
            lx = legend_x
            ly = ly_start + i * (sq + gap)
            pygame.draw.rect(surface, tc[key], (lx, ly, sq, sq))
            pygame.draw.rect(surface, bc.get(key, bc['default']), (lx, ly, sq, sq), 1)
            lt = lf.render(label, True, _C_LBL)
            surface.blit(lt, (lx + sq + 4, ly + (sq - lt.get_height()) // 2))

        # Thin separator between legend and grid
        sep_x = px + _PAD + _AV_LEGEND_W + 1
        pygame.draw.line(surface, _C_SEP,
                         (sep_x, cy), (sep_x, cy + rows * ch))

        # ── RIGHT: the observation grid ───────────────────────────────────────
        sx0 = sep_x + _AV_LEGEND_SEP
        sy0 = cy

        tiny = self._get_tiny_font()

        for r in range(rows):
            for c in range(cols):
                tx, ty = ox + c, oy + r
                dx = sx0 + c * cw
                dy = sy0 + r * ch

                if 0 <= ty < core.level_data.rows and 0 <= tx < core.level_data.cols:
                    t = core.level_data.grid[ty][tx]
                    if   t == TILE_AIR:   col, bcol = tc['empty'],  bc['default']
                    elif t == TILE_SPIKE: col, bcol = tc['spike'],  bc['default']
                    elif t == TILE_GOAL:  col, bcol = tc['goal'],   bc['goal']
                    else:                 col, bcol = tc['wall'],   bc['default']
                else:
                    col, bcol = tc['oob'], bc['default']

                if   (tx, ty) in enemy_locs: col, bcol = tc['enemy'],  bc['enemy']
                elif (tx, ty) in coin_locs:  col, bcol = tc['coin'],   bc['coin']
                elif (tx, ty) in goal_locs:  col, bcol = tc['goal'],   bc['goal']
                if   tx == p_cx and ty == p_cy: col, bcol = tc['player'], bc['player']

                pygame.draw.rect(surface, col,  (dx, dy, cw - 1, ch - 1))
                pygame.draw.rect(surface, bcol, (dx, dy, cw - 1, ch - 1), 1)

                # Dijkstra cost number (tiny font, centred in cell)
                if core.dijkstra:
                    cost = core.dijkstra.get_dist(tx, ty)
                    if 0 < cost < float('inf'):
                        # Heat tint
                        alpha = min(50, int(cost * 0.7))
                        tint = pygame.Surface((cw - 1, ch - 1), pygame.SRCALPHA)
                        tint.fill((255, 130, 0, alpha))
                        surface.blit(tint, (dx, dy))
                        # Number
                        ns = tiny.render(str(int(cost)), True, (200, 200, 200))
                        surface.blit(ns, (dx + (cw - ns.get_width()) // 2,
                                          dy + (ch - ns.get_height()) // 2))


class InfoPanelOverlay(DebugOverlay):
    def render(self, surface, core):
        if not core.player:
            return
        p = core.player

        panel_w_total = core.TOTAL_WIDTH - core.DEBUG_PANEL_X - _PAD * 2
        half_w = panel_w_total // 2 - 2   # left half
        px = core.DEBUG_PANEL_X + _PAD
        py = _lower_y(core)
        ph = _lower_h()

        _card(surface, px, py, half_w, ph)
        cy = _section_hdr(surface, core.debug_manager.font,
                          "Player Info", px, py, half_w, accent=(80, 180, 80))

        sf = core.debug_manager.small_font
        rows = [
            ("Pos",    f"{int(p.gObj.x)}, {int(p.gObj.y)}"),
            ("Vel",    f"{p.vx:.1f}, {p.vy:.1f}"),
            ("Stall",  f"{core.stall_timer:.2f}s ×{core.stall_windows_count}"),
            ("Best X", f"{int(core.progress_x_best)}"),
            ("Frame",  f"{core.frame}"),
        ]
        rx = px + half_w - 5
        for label, val in rows:
            ls = sf.render(label + ":", True, _C_LBL)
            vs = sf.render(val,         True, _C_VAL)
            surface.blit(ls, (px + 8, cy))
            surface.blit(vs, (rx - vs.get_width(), cy))
            cy += _INFO_LINE_H


class ObsValuesOverlay(DebugOverlay):
    def render(self, surface, core):
        try:
            p_vals = core._player_obs()
            t_vals = core._tracking_obs()
        except Exception:
            return

        p_labels = ["Px", "Py", "Vx", "Vy", "Grnd"]
        t_labels = ["EnmDst", "CoinDst", "GoalDst",
                    "#Enm", "#Coin", "Score",
                    "Time", "Lives", "DirX", "GoalY", "Dijkstra"]
        if len(t_vals) > len(t_labels):
            t_labels.extend([f"V{i}" for i in range(len(t_labels), len(t_vals))])
        data = list(zip(p_labels, p_vals)) + list(zip(t_labels, t_vals))

        panel_w_total = core.TOTAL_WIDTH - core.DEBUG_PANEL_X - _PAD * 2
        half_w = panel_w_total // 2 - 2
        px = core.DEBUG_PANEL_X + _PAD + half_w + 4
        py = _lower_y(core)
        ph = _lower_h()

        _card(surface, px, py, half_w, ph)
        cy = _section_hdr(surface, core.debug_manager.font,
                          "Obs Values", px, py, half_w, accent=(160, 90, 200))

        sf   = core.debug_manager.small_font
        rx   = px + half_w - 5

        for label, val in data:
            if cy + _VEC_LINE_H > py + ph - 2:
                break
            if "Dst" in label and 0 < val < 0.15:
                vc = _C_NEG
            elif "Grnd" in label and val > 0.5:
                vc = _C_ACT
            elif val > 0:
                vc = _C_VAL
            else:
                vc = _C_LBL

            ls = sf.render(label + ":", True, _C_LBL)
            vs = sf.render(f"{val:.2f}",  True, vc)
            surface.blit(ls, (px + 5, cy))
            surface.blit(vs, (rx - vs.get_width(), cy))
            cy += _VEC_LINE_H
