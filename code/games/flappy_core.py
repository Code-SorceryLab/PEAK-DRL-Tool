"""Pure Flappy‑Bird physics & drawing – no RL or HUD here."""
import random
from typing import Tuple, Dict, Any
import pygame
import numpy as np
import gymnasium as gym


class FlappyCore:
    WIDTH, HEIGHT = 400, 600
    PIPE_GAP, PIPE_SPEED = 150, 4
    GRAVITY, FLAP_STRENGTH = 1.0, 12.0
    BIRD_SIZE, MAX_VEL, PIPE_WIDTH = 20, 15.0, 50

    def __init__(self, seed: int | None = None, render_mode: str | None = None, **kwargs):
        # render_mode/kwargs are accepted for compatibility, even if unused
        self.rng = random.Random(seed)
        self.reset()


    # ------------------------------------------------ helpers
    def _spawn_pipe(self, x: float) -> Dict[str, float]:
        top = self.rng.randint(50, self.HEIGHT - self.PIPE_GAP - 50)
        return {"x": float(x), "top": float(top)}

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
        self.last_action = False  # bool: True if flap
        self.timestep = 0
        return self._state()

    def step(self, flap: bool) -> Tuple[np.ndarray, float | None, bool, Dict[str, Any]]:
        """Advance one frame.  Returns (obs, base_reward, terminated, info)."""
        self.timestep += 1
        self.last_action = flap

        if self.dead:
            return self._state(), 0.0, True, {"score": self.score}

        # action → velocity impulse
        if flap:
            self.bird_v = -self.FLAP_STRENGTH

        # physics
        self.bird_v = min(self.bird_v + self.GRAVITY, self.MAX_VEL)
        self.bird_y += self.bird_v

        # move pipes
        for p in self.pipes:
            p["x"] -= self.PIPE_SPEED
        if self.pipes[0]["x"] < -self.PIPE_WIDTH:
            self.pipes.pop(0)
            self.pipes.append(self._spawn_pipe(self.pipes[-1]["x"] + self.WIDTH / 2))
            self.next_pipe = max(0, self.next_pipe - 1)

        # Score if passed pipe
        p = self.pipes[self.next_pipe]
        if p["x"] + self.PIPE_WIDTH < self.bird_x:
            self.score += 1
            self.next_pipe += 1
            self.next_pipe = min(self.next_pipe, len(self.pipes) - 1)

        # collisions
        self.dead = self._crashed()
        return self._state(), 0.0, self.dead, {"score": self.score}

    # alias for legacy code
    update = step

    # ------------------------------------------------ drawing (no flip)
    def render(self, surface: pygame.Surface, *, blit_only: bool = False) -> None:
        surface.fill((135, 206, 235))  # sky
        pygame.draw.rect(surface, (255, 223, 0), (0, self.HEIGHT - 30, self.WIDTH, 30))
        # bird
        pygame.draw.ellipse(surface, (255, 0, 0), (self.bird_x, int(self.bird_y), self.BIRD_SIZE, self.BIRD_SIZE))
        # pipes
        for p in self.pipes:
            x = int(p["x"])
            top = int(p["top"])
            bottom = top + self.PIPE_GAP
            pygame.draw.rect(surface, (0, 200, 0), (x, 0, self.PIPE_WIDTH, top))
            pygame.draw.rect(surface, (0, 200, 0), (x, bottom, self.PIPE_WIDTH, self.HEIGHT - bottom))
        if not blit_only:
            pygame.display.flip()

    # ------------------------------------------------ helpers
    def _crashed(self) -> bool:
        if self.bird_y < 0 or self.bird_y + self.BIRD_SIZE > self.HEIGHT:
            return True
        for p in self.pipes:
            in_x = self.bird_x + self.BIRD_SIZE > p["x"] and self.bird_x < p["x"] + self.PIPE_WIDTH
            if not in_x:
                continue
            gap_top = p["top"]
            gap_bottom = gap_top + self.PIPE_GAP
            if not (gap_top < self.bird_y < gap_bottom - self.BIRD_SIZE):
                return True
        return False

    def _state(self) -> np.ndarray:
        p = self.pipes[self.next_pipe]
        return np.array([
            self.bird_y,
            self.bird_v,
            p["x"] - self.bird_x,
            p["top"],
            p["top"] + self.PIPE_GAP,
        ], dtype=np.float32)

    # ------------------------------------------------ spaces for adapter convenience
    def get_observation_space(self):
        low = np.full(self._state().shape, -np.inf, dtype=np.float32)
        high = np.full(self._state().shape, np.inf, dtype=np.float32)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def get_action_space(self):
        return gym.spaces.Discrete(2)  # 0 idle, 1 flap