import pygame
from .....wrappers.RewardHub import RewardHub
from .overlays import (
    HitboxOverlay, GridOverlay, AgentViewOverlay, InfoPanelOverlay, ObsValuesOverlay,
    ArchOverlay, JumpArcOverlay,
    _PAD, _GAP, _BANNER_H, _HDR_H,
    _card, _section_hdr,
    _C_BG, _C_CARD, _C_HDR, _C_BORDER, _C_SEP,
    _C_LBL, _C_VAL, _C_POS, _C_NEG, _C_ACT, _C_ACCENT,
    _reward_y,
    RAY_EMPTY, RAY_SOLID, RAY_HAZARD, RAY_COIN, RAY_GOAL,
)


class DebugManager:
    def __init__(self, default_active=True, print_help=True):
        self.active = True
        self.hub    = None  # injected by GameEnv wrapper

        # ── Always-on ──────────────────────────────────────
        self.show_agent_view = True
        self.show_obs_panel  = True
        self.show_grid       = True
        self.show_reward_log = True
        self.show_obs_values = True

        # ── Toggleable (F1–F4, F5) ─────────────────────────────
        self.show_sensors    = False   # F1 — sensor rays
        self.free_cam_active = False   # F2 — free camera
        self.slow_motion     = False   # F3 — slow motion
        self.show_hitboxes   = True    # F4 — hitboxes (default on)
        # F5 max-view is owned by agent_view_overlay.max_view (toggled below)

        self.cam_move_speed   = 600.0
        self.current_cam_move = [0.0, 0.0]
        self.last_action_name = "None"

        self.font       = pygame.font.SysFont("segoeui", 13, bold=True)
        self.small_font = pygame.font.SysFont("segoeui", 11)
        self._prev_keys = None

        if pygame.display.get_init():
            try:
                self._prev_keys = pygame.key.get_pressed()
                self.font       = pygame.font.SysFont("segoeui", 13, bold=True)
                self.small_font = pygame.font.SysFont("segoeui", 11)
            except pygame.error:
                print("[DebugManager] Warning: Video system not initialized.")
                self._prev_keys = None
        else:
            if pygame.font.get_init():
                self.font       = pygame.font.SysFont("segoeui", 13, bold=True)
                self.small_font = pygame.font.SysFont("segoeui", 11)

        self.hitbox_overlay     = HitboxOverlay()
        self.grid_overlay       = GridOverlay()
        self.agent_view_overlay = AgentViewOverlay()
        self.info_overlay       = InfoPanelOverlay()
        self.obs_values_overlay = ObsValuesOverlay()
        self.arch_overlay       = ArchOverlay()
        self.jump_arc_overlay   = JumpArcOverlay()

        if print_help:
            self._print_help_to_terminal()

    # ─────────────────────────────────────────────────────────
    def _print_help_to_terminal(self):
        # Lazy import — colour helpers live in menu.py which imports manager.py,
        # so we define minimal equivalents inline to avoid circular imports.
        def _c(code, t): return f"\033[{code}m{t}\033[0m"
        dim  = lambda t: _c("2",  t)
        bold = lambda t: _c("1",  t)
        cyan = lambda t: _c("96", t)
        red  = lambda t: _c("91", t)
        yel  = lambda t: _c("93", t)
        wht  = lambda t: _c("97", t)
        grn  = lambda t: _c("92", t)

        W = 44
        print()
        print(dim("    ╔" + "═" * W + "╗"))
        print(dim("    ║") + red(f"  PEAK ENGINE  ·  DRL INSPECTOR".center(W)) + dim("║"))
        print(dim("    ╠" + "═" * W + "╣"))

        def row(key, desc, active=None):
            k = yel(f"  {key:<4}")
            d = wht(f"{desc}")
            state = ""
            if active is not None:
                state = grn(" [ON]") if active else dim(" [off]")
            inner = f"{k}  {d}{state}"
            # pad to W chars (strip ANSI for length)
            import re
            visible = re.sub(r'\033\[[^m]*m', '', inner)
            pad = " " * max(0, W - len(visible))
            print(dim("    ║") + inner + pad + dim("║"))

        row("F1", "Rays + Jump Arc   (toggle)")
        row("F2", "Free camera       (IJKL to move)")
        row("F3", "Slow motion       (0.5×)")
        row("F4", "Hitboxes          (toggle)")
        row("F5", "Agent Vision      (Max View toggle)")
        row("Esc", "Max View off      (if open)")
        print(dim("    ╚" + "═" * W + "╝"))
        print()

    # ─────────────────────────────────────────────────────────
    def update_input(self):
        if self._prev_keys is None:
            return
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        def _c(code, t): return f"\033[{code}m{t}\033[0m"
        dim = lambda t: _c("2",  t)
        grn = lambda t: _c("92", t)
        red = lambda t: _c("91", t)

        def rising(k):
            return keys[k] and not self._prev_keys[k]

        if rising(pygame.K_F1):
            self.show_sensors = not self.show_sensors
            state = grn("ON") if self.show_sensors else dim("off")
            print(f"  {dim('F1')}  Rays + Jump Arc  →  {state}")
        if rising(pygame.K_F2):
            self.free_cam_active = not self.free_cam_active
            state = grn("ON") if self.free_cam_active else dim("off")
            print(f"  {dim('F2')}  Free Camera      →  {state}")
        if rising(pygame.K_F3):
            self.slow_motion = not self.slow_motion
            state = grn("ON") if self.slow_motion else dim("off")
            print(f"  {dim('F3')}  Slow Motion      →  {state}")
        if rising(pygame.K_F4):
            self.show_hitboxes = not self.show_hitboxes
            state = grn("ON") if self.show_hitboxes else dim("off")
            print(f"  {dim('F4')}  Hitboxes         →  {state}")
        if rising(pygame.K_F5):
            self.agent_view_overlay.max_view = not self.agent_view_overlay.max_view
            state = grn("ON") if self.agent_view_overlay.max_view else dim("off")
            print(f"  {dim('F5')}  Agent Max View   →  {state}")
        if rising(pygame.K_ESCAPE) and self.agent_view_overlay.max_view:
            self.agent_view_overlay.max_view = False
            print(f"  {dim('Esc')} Agent Max View   →  {dim('closed')}")

        self.current_cam_move = [0.0, 0.0]
        if self.free_cam_active:
            if keys[pygame.K_j]: self.current_cam_move[0] = -self.cam_move_speed
            if keys[pygame.K_l]: self.current_cam_move[0] =  self.cam_move_speed
            if keys[pygame.K_i]: self.current_cam_move[1] = -self.cam_move_speed
            if keys[pygame.K_k]: self.current_cam_move[1] =  self.cam_move_speed

        self._prev_keys = keys

    # ─────────────────────────────────────────────────────────
    def log_step(self, reward, action_name):
        self.last_action_name = action_name

    # ─────────────────────────────────────────────────────────
    def render_overlays(self, surface: pygame.Surface, core):
        if core.render_mode != "human":
            return

        debug_x = core.DEBUG_PANEL_X
        panel_w = core.TOTAL_WIDTH - debug_x
        H       = core.HEIGHT

        # ── Panel background ──────────────────────────────────
        bg = pygame.Surface((panel_w, H))
        bg.fill(_C_BG)
        surface.blit(bg, (debug_x, 0))
        pygame.draw.line(surface, (42, 44, 62), (debug_x, 0), (debug_x, H), 2)

        # ── Banner ────────────────────────────────────────────
        banner = pygame.Surface((panel_w, _BANNER_H))
        banner.fill((20, 20, 30))
        surface.blit(banner, (debug_x, 0))
        pygame.draw.line(surface, _C_SEP,
                         (debug_x, _BANNER_H), (debug_x + panel_w, _BANNER_H))
        # accent strip
        pygame.draw.rect(surface, _C_ACCENT, (debug_x, 0, 3, _BANNER_H))

        title = self.font.render("DEBUG", True, (130, 135, 170))
        surface.blit(title, (debug_x + 10, (_BANNER_H - title.get_height()) // 2))

        # F-key hint chips (right-aligned in banner)
        toggles = [
            ("F1", "rays+arc", self.show_sensors),
            ("F2", "cam",      self.free_cam_active),
            ("F3", "slow",     self.slow_motion),
            ("F4", "hbox",     self.show_hitboxes),
            ("F5", "max",      self.agent_view_overlay.max_view),
        ]
        hx = debug_x + panel_w - 6
        sf = self.small_font
        for key, desc, active in reversed(toggles):
            # value
            ds = sf.render(desc, True, (85, 210, 120) if active else (65, 68, 85))
            ks = sf.render(key,  True, (120, 145, 220))
            hx -= ds.get_width() + 3
            surface.blit(ds, (hx, (_BANNER_H - ds.get_height()) // 2 + 1))
            hx -= ks.get_width() + 3
            surface.blit(ks, (hx, (_BANNER_H - ks.get_height()) // 2))
            hx -= 8

        # ── HUD strip (lives / score / coins / status / time) ─────────────
        self._render_hud_strip(surface, core, debug_x, panel_w)

        # ── Game-area overlays ────────────────────────────────
        if self.show_grid:
            self.grid_overlay.render(surface, core)
        if self.show_hitboxes:
            self.hitbox_overlay.render(surface, core)

        # ── Jump arc — toggled with F1 (rays+arc) ──────────────
        if self.show_sensors:
            self.jump_arc_overlay.render(surface, core)

        # ── Panel overlays ────────────────────────────────────
        self.agent_view_overlay.render(surface, core)
        self.info_overlay.render(surface, core)
        self.obs_values_overlay.render(surface, core)
        # NOTE: arch_overlay not rendered — ARCH strip removed for more reward trace space
        self._render_reward_strip(surface, core, debug_x, panel_w)

        # ── Status badges (game area, top-centre) ─────────────
        by = 6
        if self.free_cam_active:
            by = self._badge(surface, core, "FREE CAM  (IJKL)", (40, 80, 200), y=by) + 4
        if self.slow_motion:
            by = self._badge(surface, core, "SLOW MOTION",       (185, 110, 0), y=by) + 4
        if self.show_sensors:
            self._badge(surface, core, "RAYS + JUMP ARC  (F1)", (0, 145, 80), y=by)

        # ── Max-view draws last so it covers everything ────────
        # (AgentViewOverlay.render already dispatches; this badge confirms state)
        if self.agent_view_overlay.max_view:
            self._badge(surface, core, "AGENT MAX VIEW  (F5/Esc)", (30, 55, 150), y=6)

    # ─────────────────────────────────────────────────────────
    def _render_hud_strip(self, surface, core, debug_x, panel_w):
        """Render Lives / Score / Coins / Status / Time in the debug panel."""
        from .overlays import _C_BG, _C_CARD, _C_HDR, _C_BORDER, _C_SEP, _C_LBL, _C_VAL, _C_ACT, _C_NEG, _BANNER_H, _GAP, _PAD, _HDR_H, _HUD_STRIP_H

        # Position: just below the banner
        px = debug_x + _PAD
        py = _BANNER_H + _GAP
        pw = panel_w - _PAD * 2
        ph = _HUD_STRIP_H  # matches the constant in overlays.py

        # Card background
        bg = pygame.Surface((pw, ph))
        bg.fill(_C_CARD)
        surface.blit(bg, (px, py))
        pygame.draw.rect(surface, _C_BORDER, (px, py, pw, ph), 1)
        # left accent bar (cyan-ish)
        pygame.draw.rect(surface, (60, 190, 210), (px, py, 3, ph))

        sf = self.small_font

        # Pull values from core
        lives  = max(0, getattr(core, 'lives', 0))
        score  = getattr(core, 'score', 0)
        coins  = getattr(core, 'coins_total', 0)
        timer  = int(getattr(core, 'timer', 0))
        player = getattr(core, 'player', None)
        if player:
            status = "STAR" if player.invincible_timer > 0 else ("SUPER" if player.powered_up else "SMALL")
        else:
            status = "—"

        # Warn colours
        lives_col = _C_NEG if lives <= 1 else _C_VAL
        time_col  = _C_NEG if timer < 60  else _C_VAL

        # Build label/value pairs and space them evenly
        items = [
            ("LIVES",  str(lives),  lives_col),
            ("SCORE",  str(score),  _C_VAL),
            ("COINS",  str(coins),  (220, 190, 60)),
            ("STATUS", status,      _C_ACT),
            ("TIME",   str(timer),  time_col),
        ]

        slot_w = pw // len(items)

        # Vertically centre the two-row block (label ~11px + 2px gap + value ~13px = 26px total)
        block_h = 11 + 2 + 13
        block_top = py + (ph - block_h) // 2
        cy_lbl = block_top
        cy_val = block_top + 13  # label height + gap

        for i, (label, value, vcol) in enumerate(items):
            sx = px + i * slot_w + slot_w // 2

            ls = sf.render(label, True, _C_LBL)
            vs = self.font.render(value, True, vcol)

            surface.blit(ls, (sx - ls.get_width() // 2, cy_lbl))
            surface.blit(vs, (sx - vs.get_width() // 2, cy_val))

            # Thin vertical divider between items
            if i > 0:
                pygame.draw.line(surface, _C_SEP,
                                 (px + i * slot_w, py + 4),
                                 (px + i * slot_w, py + ph - 4))

    # ─────────────────────────────────────────────────────────
    def _render_reward_strip(self, surface, core, debug_x, panel_w):
        py = _reward_y(core)
        if py + 32 > core.HEIGHT:
            return

        px = debug_x + _PAD
        pw = panel_w - _PAD * 2
        ph = core.HEIGHT - py - _GAP

        _card(surface, px, py, pw, ph)
        cy = _section_hdr(surface, self.font, "Reward Trace", px, py, pw, accent=(200, 80, 60))

        hub = self.hub if self.hub else RewardHub.get_instance()
        sf  = self.small_font

        persona = getattr(core, "persona", "?").replace("_", " ").title()
        row1 = sf.render(f"Action:  {hub.last_action_name}", True, _C_VAL)
        row2 = sf.render(f"Persona: {persona}",              True, _C_LBL)
        surface.blit(row1, (px + 6, cy));   cy += 13
        surface.blit(row2, (px + 6, cy));   cy += 13

        rh = hub.reward_history
        if len(rh) < 2:
            return

        curr  = rh[-1]
        r_col = _C_ACT if curr >= 0 else _C_NEG
        r_lbl = sf.render(f"R: {curr:+.4f}", True, r_col)
        surface.blit(r_lbl, (px + pw - r_lbl.get_width() - 4, cy - 13))

        # Sparkline
        aw = pw - 10
        ah = core.HEIGHT - cy - _GAP - 4
        if ah < 12 or aw < 20:
            return

        max_r   = max(max(rh),  0.01)
        min_r   = min(min(rh), -0.01)
        r_range = (max_r - min_r) or 1.0

        zero_y = cy + ah - int((0 - min_r) / r_range * ah)
        pygame.draw.line(surface, _C_SEP,
                         (px + 5, zero_y), (px + 5 + aw, zero_y))

        pts = [(px + 5 + int(i / (len(rh)-1) * aw),
                cy + ah - int((r - min_r) / r_range * ah))
               for i, r in enumerate(rh)]

        if len(pts) > 1:
            pygame.draw.lines(surface, (55, 200, 100), False, pts, 1)
        if pts:
            pygame.draw.circle(surface, r_col, pts[-1], 3)

    # ─────────────────────────────────────────────────────────
    def _badge(self, surface, core, text, color, y=6):
        """Draw a status badge centred on the game area. Returns bottom y."""
        t  = self.font.render(text, True, (230, 235, 245))
        w  = t.get_width() + 20
        h  = t.get_height() + 8
        bg = pygame.Surface((w, h))
        bg.fill(color)
        bg.set_alpha(210)
        x = core.WIDTH // 2 - w // 2
        surface.blit(bg, (x, y))
        surface.blit(t,  (x + 10, y + 4))
        return y + h
