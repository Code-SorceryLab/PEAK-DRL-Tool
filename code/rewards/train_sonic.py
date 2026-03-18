from __future__ import annotations

import math
from typing import Callable, Dict, Any

Info = Dict[str, Any]


class _Tracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_dist = None
        self.last_dijkstra = None
        self.max_x = 0.0
        self.last_coins = 0
        self.last_kills = 0
        self.last_top_speed = 0.0
        self.last_lives = None

    def step(self, info: Info):
        lives = int(info.get("lives", 3))
        if self.last_lives is None:
            self.last_lives = lives
        life_lost = lives < self.last_lives
        self.last_lives = lives
        info["life_lost"] = life_lost

        current_dist = float(info.get("goal_dist", 0.0))
        if math.isinf(current_dist):
            current_dist = 0.0
        if self.last_dist is None or life_lost:
            progress = 0.0
        else:
            progress = self.last_dist - current_dist
        self.last_dist = current_dist
        info["progress"] = progress

        raw_dijkstra = float(info.get("dijkstra_dist", -1.0))
        dijkstra_valid = raw_dijkstra >= 0.0
        info["dijkstra_valid"] = dijkstra_valid
        if self.last_dijkstra is None or life_lost:
            dijkstra_progress = 0.0
        elif not dijkstra_valid:
            dijkstra_progress = 0.0
        else:
            dijkstra_progress = self.last_dijkstra - raw_dijkstra
        if dijkstra_valid:
            self.last_dijkstra = raw_dijkstra
        info["dijkstra_progress"] = dijkstra_progress

        max_x_seen = float(info.get("max_x_seen", info.get("x_position", 0.0)))
        frontier_dx = max(0.0, max_x_seen - self.max_x)
        self.max_x = max(self.max_x, max_x_seen)
        if life_lost:
            self.max_x = max_x_seen
        info["frontier_dx"] = frontier_dx

        coins = int(info.get("coins_collected", 0))
        info["coins_delta"] = max(0, coins - self.last_coins)
        self.last_coins = coins

        kills = int(info.get("badniks_destroyed", info.get("enemies_killed_step", 0)))
        info["kills_delta"] = max(0, kills - self.last_kills)
        self.last_kills = kills

        top_speed = float(info.get("top_speed", 0.0))
        info["top_speed_delta"] = max(0.0, top_speed - self.last_top_speed)
        self.last_top_speed = max(self.last_top_speed, top_speed)
        if life_lost:
            self.last_top_speed = top_speed

        if info.get("won", False):
            self.last_dist = None
            self.last_dijkstra = None


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _wrap_with_tracker(core_fn) -> Callable:
    def make_reward_fn():
        tracker = _Tracker()

        def reward(obs, base, terminated: bool, info: Info) -> float:
            info = info or {}
            tracker.step(info)
            components = core_fn(terminated, info)
            info["reward_components"] = components
            total = float(sum(components.values()))
            if terminated or info.get("terminated", False):
                tracker.reset()
            return total

        return reward

    make_reward_fn._core_fn = core_fn
    make_reward_fn._is_factory = True
    return make_reward_fn


def _template(terminated: bool, info: Info, cfg: Dict[str, float]) -> Dict[str, float]:
    progress = float(info.get("progress", 0.0))
    dijkstra_progress = float(info.get("dijkstra_progress", 0.0))
    frontier = float(info.get("frontier_dx", 0.0))
    coins = int(info.get("coins_delta", 0))
    kills = int(info.get("kills_delta", 0))
    top_speed_delta = float(info.get("top_speed_delta", 0.0))
    won = bool(info.get("won", False))
    life_lost = bool(info.get("life_lost", False))
    stalled = bool(info.get("stalled", False))
    cause = str(info.get("cause", "") or "").lower()
    is_ball = bool(info.get("is_ball", False))
    state = str(info.get("sonic_state", "") or "")
    spin_dash_charge = float(info.get("spin_dash_charge", 0.0))

    route = _clip(progress * cfg["route_scale"], -cfg["route_clip"], cfg["route_clip"])
    dijkstra = _clip(dijkstra_progress * cfg["dijkstra_scale"], -cfg["dijkstra_clip"], cfg["dijkstra_clip"])
    frontier_r = _clip(frontier * cfg["frontier_scale"], 0.0, cfg["frontier_clip"])
    ring_r = coins * cfg["ring_reward"]
    combat = kills * cfg["kill_reward"]
    speed = _clip(top_speed_delta * cfg["speed_scale"], 0.0, cfg["speed_clip"])
    alive = cfg["alive_reward"]
    stall = -cfg["stall_penalty"] if stalled else 0.0
    ball = cfg["ball_reward"] if is_ball else 0.0
    dash = min(spin_dash_charge, 8.0) * cfg["dash_scale"] if state == "SPIN_DASH" else 0.0

    death = 0.0
    pit = 0.0
    spike = 0.0
    if terminated or life_lost:
        death = -cfg["death_penalty"]
        if cause == "pit":
            pit = -cfg["pit_penalty"]
        elif cause == "spike":
            spike = -cfg["spike_penalty"]

    win = cfg["win_reward"] if won else 0.0
    time = -cfg["time_penalty"]

    return {
        "route": route,
        "dijkstra": dijkstra,
        "frontier": frontier_r,
        "rings": ring_r,
        "combat": combat,
        "speed": speed,
        "alive": alive,
        "ball": ball,
        "dash": dash,
        "stall": stall,
        "time": time,
        "win": win,
        "death": death,
        "pit": pit,
        "spike": spike,
    }


@_wrap_with_tracker
def simple(terminated: bool, info: Info) -> Dict[str, float]:
    return _template(terminated, info, {
        "route_scale": 0.008,
        "route_clip": 0.20,
        "dijkstra_scale": 0.50,
        "dijkstra_clip": 0.20,
        "frontier_scale": 0.050,
        "frontier_clip": 0.20,
        "ring_reward": 0.05,
        "kill_reward": 0.12,
        "speed_scale": 0.0005,
        "speed_clip": 0.08,
        "alive_reward": 0.0005,
        "ball_reward": 0.0008,
        "dash_scale": 0.0005,
        "stall_penalty": 0.003,
        "time_penalty": 0.0002,
        "death_penalty": 0.50,
        "pit_penalty": 0.80,
        "spike_penalty": 0.40,
        "win_reward": 5.0,
    })


@_wrap_with_tracker
def speedrunner(terminated: bool, info: Info) -> Dict[str, float]:
    return _template(terminated, info, {
        "route_scale": 0.006,
        "route_clip": 0.18,
        "dijkstra_scale": 0.35,
        "dijkstra_clip": 0.14,
        "frontier_scale": 0.060,
        "frontier_clip": 0.24,
        "ring_reward": 0.03,
        "kill_reward": 0.08,
        "speed_scale": 0.0010,
        "speed_clip": 0.14,
        "alive_reward": 0.0003,
        "ball_reward": 0.0012,
        "dash_scale": 0.0012,
        "stall_penalty": 0.004,
        "time_penalty": 0.00025,
        "death_penalty": 0.60,
        "pit_penalty": 1.00,
        "spike_penalty": 0.50,
        "win_reward": 5.0,
    })


default = simple
sonic_simple = simple
sonic_speedrunner = speedrunner
