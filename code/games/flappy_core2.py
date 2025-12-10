"""Flappy-Bird v2: wind + small QoL improvements (API compatible with FlappyCore).

Public API is unchanged:
  - reset() -> obs
  - step(flap: bool) -> (obs, base_reward, terminated, info)
  - render(surface, blit_only=False)
  - get_observation_space(), get_action_space()

Obs layout is identical to v1:
  [bird_y, bird_v, dx_to_pipe, pipe_top, pipe_bottom]
"""
from __future__ import annotations
import math
import random
from typing import Tuple, Dict, Any
import pygame
import numpy as np
import gymnasium as gym


class FlappyCoreV2:
    # Base gameplay (same defaults as v1 where relevant)
    WIDTH, HEIGHT = 400, 600
    PIPE_GAP, PIPE_SPEED = 150, 4
    GRAVITY, FLAP_STRENGTH = 1.0, 12.0
    BIRD_SIZE, MAX_VEL, PIPE_WIDTH = 20, 15.0, 50

    # --- New knobs ---------------------------------------------------------
    # Wind system
    WIND_MAX = 1.75           # max additional accel magnitude (+down / -up)
    WIND_PERIOD = 360         # frames per slow sinusoidal cycle
    GUST_CHANCE = 0.005       # per-frame chance to start a gust
    GUST_DURATION = (45, 90)  # frames
    GUST_STRENGTH = (0.8, 1.6)  # multiplier on WIND_MAX during gust
    TURBULENCE = 0.15         # small per-frame noise on wind accel

    # Moving pipes
    PIPE_OSC_AMP = 12         # pixels
    PIPE_OSC_SPEED = 0.035    # radians per frame

    # Adaptive difficulty
    GAP_DECAY_PER_SCORE = 1.5  # every point reduces gap a bit
    GAP_MIN = 110

    # Fairness
    GRACE_PIXELS = 3          # loosen collision bounds slightly

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------ helpers
    def _spawn_pipe(self, x: float) -> Dict[str, float]:
        gap = self._current_gap()
        top = self.rng.randint(50, self.HEIGHT - gap - 50)
        phase = self.rng.random() * math.tau
        return {"x": float(x), "top": float(top), "phase": phase}

    def _current_gap(self) -> int:
        gap = int(self.PIPE_GAP - self.score * self.GAP_DECAY_PER_SCORE)
        return max(self.GAP_MIN, gap)

    def _wind_accel(self) -> float:
        """Signed vertical accel added to gravity: +downwards, -updraft."""
        # base sinusoid
        w = math.sin((self.timestep % self.WIND_PERIOD) * (2 * math.pi / self.WIND_PERIOD))
        accel = w * self.WIND_MAX

        # active gust?
        if self.gust_ticks > 0:
            accel += self.gust_dir * (self.gust_strength * self.WIND_MAX)
            self.gust_ticks -= 1
        else:
            # maybe start a gust
            if self.rng.random() < self.GUST_CHANCE:
                self.gust_ticks = self.rng.randint(*self.GUST_DURATION)
                self.gust_dir = self.rng.choice([-1.0, 1.0])
                self.gust_strength = self.rng.uniform(*self.GUST_STRENGTH)

        # turbulence
        accel += self.rng.uniform(-self.TURBULENCE, self.TURBULENCE)
        return accel

    # ------------------------------------------------ public API
    def reset(self) -> np.ndarray:
        """Reset game state and return first observation."""
        self.bird_x, self.bird_y, self.bird_v = 50.0, self.HEIGHT / 2, 0.0
        self.score, self.dead = 0, False
        self.pipes = []
        start_x = self.WIDTH + 10
        for i in range(3):
            self.pipes.append(self._spawn_pipe(start_x + i * (self.WIDTH / 2)))

        self.next_pipe = 0
        self.last_action = False  # True if flap
        self.timestep = 0

        # wind state
        self.gust_ticks = 0
        self.gust_dir = 0.0
        self.gust_strength = 0.0

        return self._state()

    def step(self, flap: bool) -> Tuple[np.ndarray, float | None, bool, Dict[str, Any]]:
        """Advance one frame.  Returns (obs, base_reward, terminated, info)."""
        self.timestep += 1
        self.last_action = flap

        if self.dead:
            return self._state(), 0.0, True, {"score": self.score, "wind": 0.0}

        # action → velocity impulse
        if flap:
            self.bird_v = -self.FLAP_STRENGTH

        # physics: gravity + wind (clamped)
        eff_gravity = self.GRAVITY + self._wind_accel()
        self.bird_v = min(self.bird_v + eff_gravity, self.MAX_VEL)
        self.bird_y += self.bird_v

        # move pipes & oscillate
        for p in self.pipes:
            p["x"] -= self.PIPE_SPEED
        if self.pipes[0]["x"] < -self.PIPE_WIDTH:
            self.pipes.pop(0)
            self.pipes.append(self._spawn_pipe(self.pipes[-1]["x"] + self.WIDTH / 2))
            self.next_pipe = max(0, self.next_pipe - 1)

        # per-frame oscillation of current pipe (visual + gameplay)
        # note: we do not permanently write top to avoid drift—compute on the fly in getters
        # (but for collision we use the instantaneous oscillated top)
        # Score if passed pipe
        p = self.pipes[self.next_pipe]
        if p["x"] + self.PIPE_WIDTH < self.bird_x:
            self.score += 1
            self.next_pipe = min(self.next_pipe + 1, len(self.pipes) - 1)

        # collisions (use current oscillated top)
        self.dead = self._crashed()
        info = {
            "score": self.score,
            "wind": eff_gravity - self.GRAVITY,  # net wind accel applied this frame
            "gap": self._current_gap(),
        }
        return self._state(), 0.0, self.dead, info

    # alias for legacy code
    update = step

    # ------------------------------------------------ drawing (no flip)
    def render(self, surface: pygame.Surface, *, blit_only: bool = False) -> None:
        surface.fill((135, 206, 235))  # sky
        # ground
        pygame.draw.rect(surface, (255, 223, 0), (0, self.HEIGHT - 30, self.WIDTH, 30))

        # wind HUD (arrow at top)
        wind = self._last_wind_for_render if hasattr(self, "_last_wind_for_render") else 0.0
        cx, cy = self.WIDTH - 40, 30
        L = int(20 + 25 * min(1.0, abs(wind) / (self.WIND_MAX * 2)))
        color = (50, 50, 50) if abs(wind) < 0.3 else ((0, 120, 255) if wind < 0 else (255, 80, 0))
        # updraft: arrow up, downdraft: arrow down
        end_y = cy - L if wind < 0 else cy + L
        pygame.draw.line(surface, color, (cx, cy), (cx, end_y), 3)
        pygame.draw.polygon(
            surface,
            color,
            [(cx - 6, end_y - ( -6 if wind < 0 else 6)),
             (cx + 6, end_y - ( -6 if wind < 0 else 6)),
             (cx, end_y + ( -10 if wind < 0 else 10))],
        )

        # bird (tilt by velocity)
        tilt = max(-30, min(30, -self.bird_v * 2))  # negative v = nose up
        bird_rect = pygame.Rect(self.bird_x, int(self.bird_y), self.BIRD_SIZE, self.BIRD_SIZE)
        bird_surf = pygame.Surface((self.BIRD_SIZE, self.BIRD_SIZE), pygame.SRCALPHA)
        pygame.draw.ellipse(bird_surf, (255, 0, 0), bird_surf.get_rect())
        bird_surf = pygame.transform.rotate(bird_surf, tilt)
        surface.blit(bird_surf, bird_rect)

        # pipes (draw with instantaneous oscillation)
        for p in self.pipes:
            x = int(p["x"])
            top = int(self._osc_top(p))
            bottom = top + self._current_gap()
            pygame.draw.rect(surface, (0, 200, 0), (x, 0, self.PIPE_WIDTH, top))
            pygame.draw.rect(surface, (0, 200, 0), (x, bottom, self.PIPE_WIDTH, self.HEIGHT - bottom))

        if not blit_only:
            pygame.display.flip()

    # ------------------------------------------------ helpers
    def _osc_top(self, p: Dict[str, float]) -> float:
        return p["top"] + self.PIPE_OSC_AMP * math.sin(self.PIPE_OSC_SPEED * self.timestep + p["phase"])

    def _crashed(self) -> bool:
        # out of vertical bounds?
        if self.bird_y < -self.GRACE_PIXELS or self.bird_y + self.BIRD_SIZE > self.HEIGHT + self.GRACE_PIXELS:
            return True

        gap = self._current_gap()
        for p in self.pipes:
            x_in = (self.bird_x + self.BIRD_SIZE - self.GRACE_PIXELS > p["x"]) and \
                   (self.bird_x + self.GRACE_PIXELS < p["x"] + self.PIPE_WIDTH)
            if not x_in:
                continue
            gap_top = self._osc_top(p)
            gap_bottom = gap_top + gap
            # allow grace near edges
            bird_top = self.bird_y + self.GRACE_PIXELS
            bird_bottom = self.bird_y + self.BIRD_SIZE - self.GRACE_PIXELS
            if not (gap_top < bird_top and bird_bottom < gap_bottom):
                return True
        return False

    def _state(self) -> np.ndarray:
        p = self.pipes[self.next_pipe]
        top = self._osc_top(p)
        gap = self._current_gap()
        # stash wind for HUD (render step)
        # Approximate last wind as dv - GRAVITY; safe if step() called every frame
        self._last_wind_for_render = getattr(self, "_last_wind_for_render", 0.0)
        # (We don't compute exact here; render() uses last recorded value in info.)
        return np.array([
            self.bird_y,
            self.bird_v,
            p["x"] - self.bird_x,
            top,
            top + gap,
        ], dtype=np.float32)

    # ------------------------------------------------ spaces for adapter convenience
    def get_observation_space(self):
        low = np.full(self._state().shape, -np.inf, dtype=np.float32)
        high = np.full(self._state().shape, np.inf, dtype=np.float32)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def get_action_space(self):
        return gym.spaces.Discrete(2)  # 0 idle, 1 flap
