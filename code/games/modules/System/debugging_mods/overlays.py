import pygame
import math
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
_N_OBS        = 20     # 13 player + 7 tracking (full scalar vector)

def _lower_y(core):
    return _AV_Y + _av_card_h(core) + _GAP

def _lower_h():
    info_h = _HDR_H + _INFO_LINES * _INFO_LINE_H + _PAD
    # obs panel: enough rows for all 20 scalars split across two columns
    vec_h  = _HDR_H + ((_N_OBS + 1) // 2) * _VEC_LINE_H + _PAD
    return max(info_h, vec_h)

# ── Reward strip — sits directly below lower panels (no arch strip) ───────────
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
            if not all(math.isfinite(v) for v in (player.gObj.x, player.gObj.y, player.gObj.width, player.gObj.height)):
                return
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
            if not all(math.isfinite(v) for v in (gObj.x, gObj.y, gObj.width, gObj.height)):
                continue
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
    """
    Renders the 21×21 agent observation grid in the debug panel.

    Max-View mode (toggle with F5 or the on-panel button):
      • Expands to a full-screen overlay so every cell is readable.
      • Shows enlarged cells with numeric Dijkstra values and per-cell
        visit-count heatmap from the exploration memory system.
      • Press F5 or Esc to return to compact mode.
    """

    _TC = {
        'wall':   ( 70,  76,  90), 'spike': (160,  36,  36),
        'goal':   ( 35, 160,  65), 'empty': (  8,   8,  14),
        'oob':    (  4,   4,   6), 'enemy': (215,  45,  45),
        'coin':   (215, 175,  18), 'player':( 35, 135, 255),
        'visited':(  30,  60, 90), 'pit':   ( 90,  30, 120),
    }
    _BC = {
        'enemy':  (255, 100, 100), 'coin':  (255, 225,  70),
        'goal':   ( 70, 255, 110), 'player':(100, 190, 255),
        'pit':    (190,  80, 255),
        'default':( 32,  34,  44),
    }
    _LEGEND = [
        ('player', "You"),
        ('wall',   "Wall"),
        ('coin',   "Coin"),
        ('enemy',  "Enemy"),
        ('goal',   "Goal"),
        ('spike',  "Spike"),
        ('pit',    "Pit"),
    ]

    # Lazily initialised fonts
    _tiny_font    = None   # 8px  — compact Dijkstra labels
    _medium_font  = None   # 14px — max-view cell values
    _label_font   = None   # 18px — max-view legend

    def __init__(self):
        self.max_view = False   # toggle state

    # ── Font helpers ──────────────────────────────────────────────────────────
    def _get_tiny_font(self):
        if AgentViewOverlay._tiny_font is None:
            AgentViewOverlay._tiny_font = pygame.font.SysFont("arial", 8)
        return AgentViewOverlay._tiny_font

    def _get_medium_font(self):
        if AgentViewOverlay._medium_font is None:
            AgentViewOverlay._medium_font = pygame.font.SysFont("arial", 14, bold=True)
        return AgentViewOverlay._medium_font

    def _get_label_font(self):
        if AgentViewOverlay._label_font is None:
            AgentViewOverlay._label_font = pygame.font.SysFont("arial", 18, bold=True)
        return AgentViewOverlay._label_font

    # ── Shared data extraction ────────────────────────────────────────────────
    def _extract_locs(self, core):
        if not core.player:
            return None
        if not all(math.isfinite(v) for v in (core.player.gObj.x, core.player.gObj.y)):
            return None
        p_cx = int(core.player.gObj.x // TILE_SIZE)
        p_cy = int(core.player.gObj.y // TILE_SIZE)
        ox   = p_cx - core.obs_pad_x
        oy   = p_cy - core.obs_pad_y
        enemy_src = getattr(core, "enemies", None)
        if not enemy_src:
            enemy_src = getattr(core.level_data, "enemies", [])
        enemy_locs = {
            (int(e.gObj.x // TILE_SIZE), int(e.gObj.y // TILE_SIZE))
            for e in enemy_src
            if getattr(e, "gObj", None) is not None and e.gObj.active
        }

        coin_src = list(getattr(core.level_data, "coins", []))
        coin_src.extend(getattr(core, "rings", []) or [])
        coin_locs  = {
            (int(c.gObj.x // TILE_SIZE), int(c.gObj.y // TILE_SIZE))
            for c in coin_src
            if getattr(c, "gObj", None) is not None
            and c.gObj.active
            and not getattr(c, "collected", False)
        }
        goal_locs  = {(int(g.gObj.x//TILE_SIZE), int(g.gObj.y//TILE_SIZE))
                      for g in core.level_data.goals if g.gObj.active}
        return p_cx, p_cy, ox, oy, enemy_locs, coin_locs, goal_locs

    def _tile_colors(self, tx, ty, p_cx, p_cy, enemy_locs, coin_locs, goal_locs, core,
                     r=None, c=None):
        """
        Returns (fill_color, border_color) for a tile at map position (tx, ty).
        r, c are the local grid row/col — used to read the pit marker from the
        hazard window cache. Pass None to skip pit detection (e.g. no cache yet).
        """
        tc, bc = self._TC, self._BC
        if 0 <= ty < core.level_data.rows and 0 <= tx < core.level_data.cols:
            t = core.level_data.grid[ty][tx]
            if   t == TILE_AIR:   col, bcol = tc['empty'], bc['default']
            elif t == TILE_SPIKE: col, bcol = tc['spike'], bc['default']
            elif t == TILE_GOAL:  col, bcol = tc['goal'],  bc['goal']
            else:                 col, bcol = tc['wall'],  bc['default']
        else:
            col, bcol = tc['oob'], bc['default']

        # ── Pit override: check hazard cache for -0.5 marker ─────────────────
        if r is not None and c is not None:
            haz_cache = getattr(core, '_hazard_window_cache', None)
            if (haz_cache is not None and
                    0 <= r < haz_cache.shape[0] and 0 <= c < haz_cache.shape[1]):
                if abs(float(haz_cache[r, c]) + 0.5) < 0.01:   # == -0.5
                    col, bcol = tc['pit'], bc['pit']

        if   (tx, ty) in enemy_locs: col, bcol = tc['enemy'],  bc['enemy']
        elif (tx, ty) in coin_locs:  col, bcol = tc['coin'],   bc['coin']
        elif (tx, ty) in goal_locs:  col, bcol = tc['goal'],   bc['goal']
        if   tx == p_cx and ty == p_cy: col, bcol = tc['player'], bc['player']
        return col, bcol

    # ── Main render dispatcher ────────────────────────────────────────────────
    def render(self, surface, core):
        if self.max_view:
            self._render_max(surface, core)
        else:
            self._render_compact(surface, core)
            self._draw_max_view_button(surface, core)

    # ── Compact (original) view ───────────────────────────────────────────────
    def _render_compact(self, surface, core):
        cols = getattr(core, 'obs_width',  21)
        rows = getattr(core, 'obs_height', 21)
        cw = ch = _AV_CELL

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

        extracted = self._extract_locs(core)
        if extracted is None:
            return
        p_cx, p_cy, ox, oy, enemy_locs, coin_locs, goal_locs = extracted

        # ── Legend strip (left) ───────────────────────────────────────────────
        legend_x = px + _PAD
        lf  = core.debug_manager.small_font
        sq  = 9
        gap = 4
        total_legend_h = len(self._LEGEND) * (sq + gap)
        ly_start = cy + (rows * ch - total_legend_h) // 2
        for i, (key, label) in enumerate(self._LEGEND):
            lx = legend_x
            ly = ly_start + i * (sq + gap)
            pygame.draw.rect(surface, self._TC[key], (lx, ly, sq, sq))
            pygame.draw.rect(surface, self._BC.get(key, self._BC['default']), (lx, ly, sq, sq), 1)
            lt = lf.render(label, True, _C_LBL)
            surface.blit(lt, (lx + sq + 4, ly + (sq - lt.get_height()) // 2))

        sep_x = px + _PAD + _AV_LEGEND_W + 1
        pygame.draw.line(surface, _C_SEP, (sep_x, cy), (sep_x, cy + rows * ch))

        # ── Grid ──────────────────────────────────────────────────────────────
        sx0 = sep_x + _AV_LEGEND_SEP
        sy0 = cy
        tiny       = self._get_tiny_font()
        cost_cache = getattr(core, '_dijkstra_window_cache', None)

        for r in range(rows):
            for c in range(cols):
                tx, ty = ox + c, oy + r
                dx     = sx0 + c * cw
                dy     = sy0 + r * ch
                col, bcol = self._tile_colors(tx, ty, p_cx, p_cy,
                                               enemy_locs, coin_locs, goal_locs, core,
                                               r=r, c=c)
                pygame.draw.rect(surface, col,  (dx, dy, cw - 1, ch - 1))
                pygame.draw.rect(surface, bcol, (dx, dy, cw - 1, ch - 1), 1)

                # Dijkstra tint only — no numbers in compact view (12px cells unreadable)
                if (cost_cache is not None and
                        0 <= r < cost_cache.shape[0] and 0 <= c < cost_cache.shape[1]):
                    val = float(cost_cache[r, c]) * 10
                    if abs(val) > 0.01:
                        intensity = min(90, int(abs(val) * 90))
                        tint = pygame.Surface((cw - 1, ch - 1), pygame.SRCALPHA)
                        tint.fill((0, 200, 80, intensity) if val > 0 else (255, 60, 30, intensity))
                        surface.blit(tint, (dx, dy))

    # ── Max-View overlay ──────────────────────────────────────────────────────
    def _render_max(self, surface, core):
        """
        Full-screen Agent Vision overlay — clean redesign.

        Layout
        ------
        ┌──────────────────────────────────────────┐
        │  Header bar  (title + F5/Esc hint)        │
        ├──────────────────────────────────────────┤
        │  Legend strip  (horizontal, 7 items)      │
        ├──────────────────────────────────────────┤
        │                                           │
        │          Colour-coded grid                │
        │   (visit heatmap tint, no number spam)    │
        │                                           │
        ├──────────────────────────────────────────┤
        │  Footer stats                             │
        └──────────────────────────────────────────┘

        Design rules
        ─────────────
        • Cell = colour fill only by default (no text).
        • Dijkstra value shown as a tiny centred label ONLY in the
          7×7 window centred on the player (where it matters most).
        • Visit heatmap: gentle blue tint, scales with log(visit_count).
        • Legend is a compact horizontal strip — no side column eating space.
        • Numbers only rendered when cell ≥ 20 px (never below that).
        """
        W, H = surface.get_width(), surface.get_height()
        cols = getattr(core, 'obs_width',  21)
        rows = getattr(core, 'obs_height', 21)

        # ── 1. Full-screen dim ────────────────────────────────────────────────
        dim = pygame.Surface((W, H), pygame.SRCALPHA)
        dim.fill((0, 0, 6, 210))
        surface.blit(dim, (0, 0))

        # ── 2. Layout geometry ────────────────────────────────────────────────
        MARGIN_X   = 40
        MARGIN_Y   = 28
        HDR_H      = 40
        LEGEND_H   = 28
        FOOTER_H   = 24
        INNER_PAD  = 10

        # Total vertical budget for the grid
        v_budget = H - MARGIN_Y * 2 - HDR_H - LEGEND_H - FOOTER_H - INNER_PAD * 3
        h_budget = W - MARGIN_X * 2

        # Cell size: fill as much space as possible, but cap at 26 for readability
        cell = min(26, h_budget // cols, v_budget // rows)
        cell = max(10, cell)   # never below 10

        gw = cols * cell
        gh = rows * cell

        # Centre the grid horizontally
        gx0 = MARGIN_X + (h_budget - gw) // 2
        gy0 = MARGIN_Y + HDR_H + LEGEND_H + INNER_PAD * 2

        card_x = MARGIN_X - 8
        card_y = MARGIN_Y - 6
        card_w = W - MARGIN_X * 2 + 16
        card_h = H - MARGIN_Y * 2 + 12

        # ── 3. Card background ────────────────────────────────────────────────
        card_bg = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        card_bg.fill((11, 12, 20, 250))
        surface.blit(card_bg, (card_x, card_y))
        # Accent top bar
        pygame.draw.rect(surface, (55, 85, 190), (card_x, card_y, card_w, 3))
        # Border
        pygame.draw.rect(surface, (40, 48, 80), (card_x, card_y, card_w, card_h), 1)

        # ── 4. Header ─────────────────────────────────────────────────────────
        hdr_font  = self._get_label_font()
        tiny_font = core.debug_manager.small_font

        title_surf = hdr_font.render("AGENT VISION", True, (200, 205, 240))
        sub_surf   = hdr_font.render("MAX VIEW", True, (80, 110, 210))
        ty_title   = MARGIN_Y + (HDR_H - title_surf.get_height()) // 2
        surface.blit(title_surf, (MARGIN_X, ty_title))
        surface.blit(sub_surf,   (MARGIN_X + title_surf.get_width() + 10, ty_title))

        hint_surf = tiny_font.render("F5 / Esc to close", True, (55, 62, 95))
        surface.blit(hint_surf, (W - MARGIN_X - hint_surf.get_width(),
                                  MARGIN_Y + (HDR_H - hint_surf.get_height()) // 2))

        hdr_sep_y = MARGIN_Y + HDR_H
        pygame.draw.line(surface, (30, 36, 60),
                         (MARGIN_X, hdr_sep_y), (W - MARGIN_X, hdr_sep_y))

        # ── 5. Legend strip (horizontal) ──────────────────────────────────────
        legend_items = list(self._LEGEND) + [('visited', "Visited")]
        sq = 12
        spacing = 14
        sm = core.debug_manager.small_font

        # Measure total legend width to centre it
        total_legend_w = 0
        for key, lbl in legend_items:
            total_legend_w += sq + 4 + sm.size(lbl)[0] + spacing
        total_legend_w -= spacing

        lx = MARGIN_X + (h_budget - total_legend_w) // 2
        ly = MARGIN_Y + HDR_H + (LEGEND_H - sq) // 2 + 2

        for key, lbl in legend_items:
            # Swatch
            pygame.draw.rect(surface, self._TC[key],            (lx, ly, sq, sq))
            pygame.draw.rect(surface, self._BC.get(key, (45, 48, 62)), (lx, ly, sq, sq), 1)
            # Label
            lt = sm.render(lbl, True, (150, 155, 185))
            surface.blit(lt, (lx + sq + 4, ly + (sq - lt.get_height()) // 2))
            lx += sq + 4 + lt.get_width() + spacing

        legend_sep_y = MARGIN_Y + HDR_H + LEGEND_H + INNER_PAD
        pygame.draw.line(surface, (25, 30, 52),
                         (MARGIN_X, legend_sep_y), (W - MARGIN_X, legend_sep_y))

        if not core.player:
            return

        extracted = self._extract_locs(core)
        if extracted is None:
            return
        p_cx, p_cy, ox, oy, enemy_locs, coin_locs, goal_locs = extracted
        cost_cache = getattr(core, '_dijkstra_window_cache', None)
        visit_map  = getattr(core, '_visit_map', None)
        med_font   = self._get_medium_font()
        tiny_lbl   = self._get_tiny_font()

        # Neighbourhood radius for showing Dijkstra numbers
        LABEL_RADIUS = 3 if cell >= 20 else 99  # 99 = disabled

        # ── 6. Grid ───────────────────────────────────────────────────────────
        import math as _math
        for r in range(rows):
            for c in range(cols):
                tx, ty = ox + c, oy + r
                dx     = gx0 + c * cell
                dy     = gy0 + r * cell
                cw     = cell - 1

                col, bcol = self._tile_colors(tx, ty, p_cx, p_cy,
                                               enemy_locs, coin_locs, goal_locs, core,
                                               r=r, c=c)
                pygame.draw.rect(surface, col,  (dx, dy, cw, cw))

                # ── Visit heatmap: blue tint, log-scaled intensity ─────────
                if (visit_map is not None and
                        0 <= ty < visit_map.shape[0] and 0 <= tx < visit_map.shape[1]):
                    vc = int(visit_map[ty, tx])
                    if vc > 0:
                        log_vc = _math.log2(vc + 1)
                        alpha  = min(160, int(log_vc * 38))
                        vt = pygame.Surface((cw, cw), pygame.SRCALPHA)
                        vt.fill((35, 105, 215, alpha))
                        surface.blit(vt, (dx, dy))

                # ── Dijkstra tint — only near player ─────────────────────
                in_nbhd = (abs(c - cols // 2) <= LABEL_RADIUS and
                           abs(r - rows // 2) <= LABEL_RADIUS)

                if (cost_cache is not None and in_nbhd and
                        0 <= r < cost_cache.shape[0] and 0 <= c < cost_cache.shape[1]):
                    val = float(cost_cache[r, c])
                    if abs(val) > 0.01:
                        intensity = min(70, int(abs(val) * 10 * 70))
                        tint = pygame.Surface((cw, cw), pygame.SRCALPHA)
                        tint.fill((0, 210, 90, intensity) if val > 0
                                  else (255, 55, 30, intensity))
                        surface.blit(tint, (dx, dy))

                    # Number label (only in neighbourhood and only if cell is big)
                    if cell >= 20:
                        lbl_str = f"{val:+.1f}"
                        if tx == p_cx and ty == p_cy:
                            lbl_str = "YOU"
                        ns = tiny_lbl.render(lbl_str, True, (210, 215, 235))
                        surface.blit(ns, (dx + (cw - ns.get_width())  // 2,
                                          dy + (cw - ns.get_height()) // 2))
                elif tx == p_cx and ty == p_cy and cell >= 20:
                    ns = tiny_lbl.render("YOU", True, (255, 255, 255))
                    surface.blit(ns, (dx + (cw - ns.get_width())  // 2,
                                      dy + (cw - ns.get_height()) // 2))

                # ── Border: bright on player row/col, dim elsewhere ────────
                if tx == p_cx or ty == p_cy:
                    pygame.draw.rect(surface, (50, 58, 85), (dx, dy, cw, cw), 1)
                # else: no border — cleaner at small sizes

        # ── Player crosshair ──────────────────────────────────────────────────
        pr = rows // 2
        pc = cols // 2
        cross_x = gx0 + pc * cell + cell // 2
        cross_y = gy0 + pr * cell + cell // 2
        cx_len  = max(6, cell // 2)
        pygame.draw.line(surface, (100, 190, 255),
                         (cross_x - cx_len, cross_y), (cross_x + cx_len, cross_y), 1)
        pygame.draw.line(surface, (100, 190, 255),
                         (cross_x, cross_y - cx_len), (cross_x, cross_y + cx_len), 1)

        # ── Grid border ───────────────────────────────────────────────────────
        pygame.draw.rect(surface, (38, 44, 72), (gx0 - 1, gy0 - 1, gw + 2, gh + 2), 1)

        # ── 7. Footer stats ───────────────────────────────────────────────────
        fy      = gy0 + gh + INNER_PAD
        stats   = []
        if visit_map is not None:
            total_visited = int((visit_map > 0).sum())
            total_tiles   = visit_map.size
            pct           = 100 * total_visited / max(1, total_tiles)
            stats.append(f"Coverage: {total_visited}/{total_tiles}  ({pct:.0f}%)")
        if cost_cache is not None:
            lo, hi = float(cost_cache.min()), float(cost_cache.max())
            stats.append(f"Dijkstra  [{lo:+.2f} → {hi:+.2f}]  (7×7 neighbourhood shown)")
        stats.append(f"Cell {cell}px  ·  Grid {cols}×{rows}")

        stat_surf = tiny_font.render("    ".join(stats), True, (52, 58, 90))
        surface.blit(stat_surf, (MARGIN_X,
                                  fy + (FOOTER_H - stat_surf.get_height()) // 2))

    # ── "Max View" toggle button rendered in compact panel ───────────────────
    def _draw_max_view_button(self, surface, core):
        """Draws a small [MAX] button at the top-right of the Agent Vision card."""
        cols    = getattr(core, 'obs_width',  21)
        grid_w  = cols * _AV_CELL
        panel_w = _AV_LEGEND_W + _AV_LEGEND_SEP + grid_w + _PAD * 2
        px      = core.DEBUG_PANEL_X + _PAD
        py      = _AV_Y

        btn_w, btn_h = 36, 14
        bx = px + panel_w - btn_w - 4
        by = py + 3

        col_bg  = (40, 65, 140) if not self.max_view else (65, 95, 200)
        col_txt = (140, 160, 230)
        pygame.draw.rect(surface, col_bg, (bx, by, btn_w, btn_h), border_radius=3)
        pygame.draw.rect(surface, col_txt, (bx, by, btn_w, btn_h), 1, border_radius=3)
        sf  = core.debug_manager.small_font
        lbl = sf.render("F5 MAX", True, col_txt)
        surface.blit(lbl, (bx + (btn_w - lbl.get_width()) // 2,
                            by + (btn_h - lbl.get_height()) // 2))


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
        px_val = int(p.gObj.x) if math.isfinite(p.gObj.x) else 0
        py_val = int(p.gObj.y) if math.isfinite(p.gObj.y) else 0
        rows = [
            ("Pos",    f"{px_val}, {py_val}"),
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
    # Scalar labels — must match the exact layout produced by _player_obs (13)
    # followed by _tracking_obs (7) = 20 total scalars.
    _P_LABELS = [
        "Px",        # [0]  x pos (tiles)
        "Py",        # [1]  y pos (tiles)
        "Vx",        # [2]  vel x (norm)
        "Vy",        # [3]  vel y (norm)
        "Grnd",      # [4]  on_ground
        "PwrUp",     # [5]  powered_up
        "Fire",      # [6]  can_fire
        "Invinc",    # [7]  invincible
        "FaceR",     # [8]  facing_right
        "FirCD",     # [9]  fire_cooldown
        "InvTmr",    # [10] invincible_timer
        "Coyote",    # [11] coyote_active
        "JmpExt",    # [12] jump_extendable
    ]
    _T_LABELS = [
        "EnmDst",    # [0]  enemy distance (norm)
        "GoalDst",   # [1]  goal distance (norm)
        "Timer",     # [2]  time remaining (norm)
        "GoalΔY",    # [3]  goal delta-Y (norm)
        "Dijkstra",  # [4]  dijkstra dist (norm)
        "StepX",     # [5]  best step direction X
        "StepY",     # [6]  best step direction Y
    ]

    def render(self, surface, core):
        try:
            p_vals = core._player_obs()
            t_vals = core._tracking_obs()
        except Exception:
            return

        p_labels = self._P_LABELS
        t_labels = self._T_LABELS
        label_fn = getattr(core, "get_obs_value_labels", None)
        if callable(label_fn):
            try:
                custom = label_fn()
                if custom and len(custom) == 2:
                    p_custom, t_custom = custom
                    if len(p_custom) == len(p_vals):
                        p_labels = list(p_custom)
                    if len(t_custom) == len(t_vals):
                        t_labels = list(t_custom)
            except Exception:
                pass

        data = (list(zip(p_labels, p_vals)) +
                list(zip(t_labels, t_vals)))

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

            # Colour-code by semantic meaning
            if label in ("GoalDst", "EnmDst") and 0 < val < 0.15:
                vc = _C_NEG   # danger-close
            elif label in ("Grnd", "Coyote", "JmpExt", "Laddr", "Ball", "Rings") and val > 0.5:
                vc = _C_ACT   # active ground/jump state
            elif label == "Climb" and abs(val) > 0.01:
                vc = _C_ACT
            elif label in ("PwrUp", "Fire", "Invinc") and val > 0.5:
                vc = (255, 195, 60)  # power state — gold
            elif val > 0:
                vc = _C_VAL
            else:
                vc = _C_LBL

            ls = sf.render(label + ":", True, _C_LBL)
            vs = sf.render(f"{val:.2f}",  True, vc)
            surface.blit(ls, (px + 5, cy))
            surface.blit(vs, (rx - vs.get_width(), cy))
            cy += _VEC_LINE_H

# ═══════════════════════════════════════════════════════════════════════════════
# ArchOverlay — compact pill strip showing the active feature extractor
# Sits between Player Info / Obs Values and the Reward Trace card.
# ═══════════════════════════════════════════════════════════════════════════════

_ARCH_PILLS = {
    # tag  → list of (pill_color_rgb, short_label, detail_text)
    "light":    [
        ((55,  120, 190), "Grids CNN",    "Conv×2+DW"),
        ((80,  160,  90), "Scalars",      "Lin(20→64)"),
    ],
    "slim":     [
        ((55,  120, 190), "Semantic CNN", "Ch0-1 +Attn"),
        ((160,  90, 200), "Jump CNN",     "Ch2-3 5×1→1×5"),
        ((80,  160,  90), "Scalars",      "Lin(20→64)"),
    ],
    "balanced": [
        ((55,  120, 190), "Semantic CNN", "Ch0-1 SE+Conv×2"),
        ((160,  90, 200), "Jump CNN",     "Ch2-3 5×1→1×5"),
        ((80,  160,  90), "Scalars",      "Lin(20→64)×2"),
    ],
    "peak":     [
        ((55,  120, 190), "Semantic CNN", "Ch0-1 SE+Conv×3"),
        ((160,  90, 200), "Jump CNN",     "Ch2-3 5×1→1×5"),
        ((80,  160,  90), "Scalars",      "Lin(20→64)×2"),
    ],
}

_ARCH_META = {
    "light":    ("~18K",   192),
    "slim":     ("~77K",   128),
    "balanced": ("~230K",  192),
    "peak":     ("~922K",  256),
}


class ArchOverlay(DebugOverlay):
    """
    Architecture overlay — disabled.  The ARCH strip has been removed from the
    debug panel to give more vertical space to the Reward Trace sparkline.
    The class is kept so existing import statements don't break.
    """
    def render(self, surface, core):
        pass  # intentionally empty — arch info removed from debug panel


# ═══════════════════════════════════════════════════════════════════════════════
# JumpArcOverlay — draw a predicted jump trajectory on the GAME viewport
# ═══════════════════════════════════════════════════════════════════════════════

class JumpArcOverlay(DebugOverlay):
    """
    Simulates the agent's jump arc from its current world position and draws
    a dotted parabola on the game viewport.

    Physics model (mirrors PhysicsManager):
      - initial vy = -jump_force  (upward, so negative)
      - each frame: vy += gravity * dt  then  y += vy * dt
      - arc is clipped when y > player_y (landed) or off screen, max 120 frames

    Coordinate conversion:
      screen_x = world_x - cam_x
      screen_y = world_y - cam_y
    """

    _COLOR_GROUND = ( 80, 220,  80)   # green — arc when on ground (preview)
    _COLOR_AIR    = ( 80, 190, 255)   # blue  — arc when airborne (actual)
    _DOT_GAP      = 5                 # pixels between dots

    def render(self, surface, core):
        if not core.player:
            return

        p = core.player

        # ── Gather physics constants ──────────────────────────────────────────
        # Try to read from the physics manager or fall back to common defaults
        phys = getattr(core, 'physics_manager', None)
        if phys is not None and hasattr(phys, 'context'):
            ctx = phys.context
            gravity = float(getattr(ctx, 'GRAVITY', 1800.0))
            jump_force = abs(float(getattr(ctx, 'JUMP_VEL_MIN', -620.0)))
        elif phys is not None:
            gravity    = float(getattr(phys, 'gravity',    1800.0))
            jump_force = float(getattr(phys, 'jump_force',  620.0))
        else:
            gravity    = float(getattr(core, 'gravity',    1800.0))
            jump_force = float(getattr(core, 'jump_force',  620.0))

        # ── Player state ──────────────────────────────────────────────────────
        state_fn = getattr(core, "get_jump_arc_debug_state", None)
        if callable(state_fn):
            state = state_fn() or {}
        else:
            state = {}

        if state:
            wx = float(state.get("x", p.gObj.x))
            wy = float(state.get("y", p.gObj.y))
            grounded = bool(state.get("grounded", getattr(p, 'grounded', True)))
            can_jump = bool(state.get("can_jump", getattr(p, 'can_jump', True)))
            arc_vx = float(state.get("vx", getattr(p, 'vx', 0.0)))
            arc_vy = float(state.get("vy", -jump_force if grounded and can_jump else getattr(p, 'vy', 0.0)))
            col = state.get("color", self._COLOR_GROUND if grounded and can_jump else self._COLOR_AIR)
            if grounded and not can_jump:
                return
        else:
            wx  = float(p.gObj.x)
            wy  = float(p.gObj.y)
            vx  = float(getattr(p, 'vx', 0.0))
            vy  = float(getattr(p, 'vy', 0.0))

            grounded  = getattr(p, 'grounded', True)
            can_jump  = getattr(p, 'can_jump', True)

            # If on the ground, preview a jump with the current vx (or a default vx)
            # If airborne, show the actual trajectory from current vy
            if grounded and can_jump:
                arc_vy = -jump_force
                arc_vx = vx if abs(vx) > 10 else 200.0
                col = self._COLOR_GROUND
            elif not grounded:
                arc_vy = vy
                arc_vx = vx
                col = self._COLOR_AIR
            else:
                return

        # ── Camera offset ─────────────────────────────────────────────────────
        cam_x = float(getattr(core, 'camera_x', 0.0))
        cam_y = float(getattr(core, 'camera_y', 0.0))
        sw    = core.WIDTH    # game viewport width (not total width)
        sh    = core.HEIGHT

        # ── Simulate arc ──────────────────────────────────────────────────────
        dt         = 1.0 / 60.0
        sim_x, sim_y = wx, wy
        sim_vy     = arc_vy
        sim_vx     = arc_vx
        start_y    = wy

        points = []
        for _ in range(150):          # max ~2.5 s of flight
            sim_x  += sim_vx * dt
            sim_vy += gravity * dt
            sim_y  += sim_vy * dt

            sx = int(sim_x - cam_x)
            sy = int(sim_y - cam_y)

            # Stop if the arc lands back at or below the launch y (parabola peak done)
            if sim_y > start_y + 4 and sim_vy > 0:
                points.append((sx, sy))
                break

            # Stop if out of game viewport (x)
            if sx < -20 or sx > sw + 20:
                break

            points.append((sx, sy))

        # ── Draw dotted arc ───────────────────────────────────────────────────
        if len(points) < 2:
            return

        # Accumulate distance to space dots evenly
        accumulated = 0.0
        dot_on = True
        prev = points[0]

        for pt in points[1:]:
            dx = pt[0] - prev[0]
            dy = pt[1] - prev[1]
            seg_len = (dx * dx + dy * dy) ** 0.5
            accumulated += seg_len

            if accumulated >= self._DOT_GAP:
                accumulated = 0.0
                dot_on = not dot_on
                if 0 <= pt[0] < sw and 0 <= pt[1] < sh:
                    if dot_on:
                        pygame.draw.circle(surface, col, pt, 2)
                    else:
                        # Dimmer gap dot
                        gap_col = tuple(max(0, c - 80) for c in col)
                        pygame.draw.circle(surface, gap_col, pt, 1)

            prev = pt

        # Landing dot (slightly larger)
        lp = points[-1]
        if 0 <= lp[0] < sw and 0 <= lp[1] < sh:
            pygame.draw.circle(surface, col, lp, 4)
            pygame.draw.circle(surface, (255, 255, 255), lp, 4, 1)
