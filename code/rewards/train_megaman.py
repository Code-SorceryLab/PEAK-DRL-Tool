from __future__ import annotations

import math
from typing import Any, Callable, Dict

Info = Dict[str, Any]


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _action_flags(action_name: str) -> Dict[str, bool]:
    return {
        "jump": "JUMP" in action_name,
        "fire": "FIRE" in action_name,
        "up": "+UP" in action_name,
        "down": "+DOWN" in action_name,
        "left": "LEFT" in action_name,
        "right": "RIGHT" in action_name,
    }


class _MegaTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_goal_dist = None
        self.last_dijkstra = None
        self.last_x = None
        self.max_x = None
        self.last_boss_hp = None
        self.last_lives = None

    def step(self, info: Info) -> Info:
        current_lives = int(info.get("lives", 3))
        life_lost = self.last_lives is not None and current_lives < self.last_lives
        self.last_lives = current_lives
        info["life_lost"] = life_lost

        goal_dist = float(info.get("goal_dist", 0.0))
        if math.isinf(goal_dist):
            goal_dist = 0.0
        if self.last_goal_dist is None:
            self.last_goal_dist = goal_dist
        if life_lost:
            goal_progress = 0.0
            self.last_goal_dist = goal_dist
        else:
            goal_progress = self.last_goal_dist - goal_dist
            self.last_goal_dist = goal_dist
        info["goal_progress"] = goal_progress

        raw_dijkstra = float(info.get("dijkstra_dist", -1.0))
        dijkstra_valid = raw_dijkstra >= 0.0 and not math.isinf(raw_dijkstra)
        info["dijkstra_valid"] = dijkstra_valid

        if self.last_dijkstra is None and dijkstra_valid:
            self.last_dijkstra = raw_dijkstra

        if life_lost:
            dijkstra_progress = 0.0
            self.last_dijkstra = raw_dijkstra if dijkstra_valid else None
        elif not dijkstra_valid or self.last_dijkstra is None:
            dijkstra_progress = 0.0
        else:
            dijkstra_progress = self.last_dijkstra - raw_dijkstra
            self.last_dijkstra = raw_dijkstra
        info["dijkstra_progress"] = dijkstra_progress

        x_pos = float(info.get("x_position", 0.0))
        if self.last_x is None:
            self.last_x = x_pos
        info["x_progress"] = x_pos - self.last_x
        self.last_x = x_pos

        frontier_src = float(info.get("max_x_seen", x_pos))
        if self.max_x is None:
            self.max_x = frontier_src
        if life_lost:
            frontier_dx = 0.0
            self.max_x = frontier_src
        else:
            frontier_dx = max(0.0, frontier_src - self.max_x)
            self.max_x = max(self.max_x, frontier_src)
        info["frontier_dx"] = frontier_dx

        boss_level = bool(info.get("boss_level", False))
        boss_active = bool(info.get("boss_active", False))
        boss_hp = float(info.get("boss_hp_ratio", 0.0))
        if self.last_boss_hp is None or not boss_level:
            self.last_boss_hp = boss_hp
        info["boss_damage_progress"] = (
            max(0.0, self.last_boss_hp - boss_hp) if boss_level and boss_active else 0.0
        )
        self.last_boss_hp = boss_hp

        if bool(info.get("won", False)) or bool(info.get("terminated", False)):
            info["goal_progress"] = 0.0
            info["dijkstra_progress"] = 0.0
            info["frontier_dx"] = 0.0

        return info


def _reward_template(terminated: bool, info: Info, cfg: Dict[str, float]) -> Dict[str, float]:
    won = bool(info.get("won", False))
    cause = str(info.get("cause", "")).lower()
    action_name = str(info.get("action_name", ""))
    flags = _action_flags(action_name)

    goal_dy = float(info.get("goal_dy", 0.0))
    step_dx = float(info.get("step_dx", info.get("x_progress", 0.0)))
    step_dy = float(info.get("step_dy", 0.0))
    goal_progress = float(info.get("goal_progress", 0.0))
    dijkstra_progress = float(info.get("dijkstra_progress", 0.0))
    frontier_dx = float(info.get("frontier_dx", 0.0))
    boss_chip = float(info.get("boss_damage_progress", 0.0))

    kills = int(info.get("enemies_killed_step", 0))
    damage_taken = float(info.get("damage_taken_step", 0.0))
    on_ladder = bool(info.get("on_ladder", False))
    on_ground = bool(info.get("on_ground", False))
    boss_level = bool(info.get("boss_level", False))
    boss_active = bool(info.get("boss_active", False))
    life_lost = bool(info.get("life_lost", False))
    stalled = bool(info.get("stalled", False))

    route = _clip(
        goal_progress * cfg["goal_scale"] + dijkstra_progress * cfg["dijkstra_scale"],
        -cfg["route_neg_cap"],
        cfg["route_pos_cap"],
    )
    frontier = min(cfg["frontier_cap"], frontier_dx * cfg["frontier_scale"])
    forward = max(0.0, step_dx) * cfg["forward_scale"]
    backtrack = min(0.0, step_dx) * cfg["backtrack_scale"]

    climb = 0.0
    if on_ladder and flags["up"] and goal_dy < -6.0:
        climb += cfg["climb_up"]
    if on_ladder and flags["down"] and goal_dy > 6.0:
        climb += cfg["climb_down"]

    climb_wrong = 0.0
    if on_ladder and flags["down"] and goal_dy < -8.0:
        climb_wrong += cfg["climb_wrong"]
    if on_ladder and flags["up"] and goal_dy > 8.0:
        climb_wrong += cfg["climb_wrong"] * 0.75

    vertical = 0.0
    if goal_dy < -10.0 and step_dy < -0.25 and (flags["jump"] or flags["up"] or on_ladder):
        vertical += min(cfg["vertical_cap"], (-step_dy) * cfg["vertical_scale"])
    elif goal_dy > 10.0 and step_dy > 0.25 and (flags["down"] or on_ladder):
        vertical += min(cfg["vertical_cap"], step_dy * cfg["vertical_scale"] * 0.65)

    combat = kills * cfg["kill_scale"]
    boss = boss_chip * cfg["boss_scale"]
    if boss_level and boss_active and flags["fire"]:
        boss += cfg["boss_fire_bonus"]

    damage = damage_taken * cfg["damage_scale"]
    idle = cfg["idle_penalty"] if on_ground and not on_ladder and abs(step_dx) < 0.05 else 0.0
    stall = cfg["stall_penalty"] if stalled else 0.0

    spam = 0.0
    if flags["jump"] and abs(step_dx) < 0.05 and goal_dy >= -6.0:
        spam += cfg["jump_spam_penalty"]
    if flags["fire"] and abs(step_dx) < 0.05 and kills == 0 and boss_chip <= 0.0 and not boss_active:
        spam += cfg["fire_spam_penalty"]

    survive = cfg["alive_bonus"]
    time = cfg["time_penalty"]
    life = cfg["life_lost_penalty"] if life_lost and not terminated else 0.0

    win = 0.0
    death = 0.0
    pit = 0.0
    spike = 0.0
    if won:
        win = cfg["win_reward"]
    elif terminated:
        death = cfg["death_penalty"]
        if cause == "pit":
            pit = cfg["pit_penalty"]
        elif cause == "spike":
            spike = cfg["spike_penalty"]

    return {
        "route": route,
        "frontier": frontier,
        "forward": forward,
        "backtrack": backtrack,
        "climb": climb,
        "climb_wrong": climb_wrong,
        "vertical": vertical,
        "combat": combat,
        "boss": boss,
        "damage": damage,
        "idle": idle,
        "stall": stall,
        "spam": spam,
        "alive": survive,
        "time": time,
        "life": life,
        "win": win,
        "death": death,
        "pit": pit,
        "spike": spike,
    }


def _wrap_with_tracker(core_fn) -> Callable:
    def make_reward_fn():
        tracker = _MegaTracker()

        def reward(obs, base, terminated: bool, info: Info) -> float:
            info = tracker.step(info or {})
            components = core_fn(terminated, info)
            info["reward_components"] = components
            total = float(sum(components.values()))
            if terminated or bool(info.get("terminated", False)) or bool(info.get("won", False)):
                tracker.reset()
            return total

        return reward

    make_reward_fn._core_fn = core_fn
    make_reward_fn._is_factory = True
    return make_reward_fn


@_wrap_with_tracker
def simple(terminated: bool, info: Info) -> Dict[str, float]:
    return _reward_template(
        terminated,
        info,
        {
            "goal_scale": 0.16,
            "dijkstra_scale": 0.26,
            "route_neg_cap": 0.030,
            "route_pos_cap": 0.050,
            "frontier_scale": 0.0013,
            "frontier_cap": 0.018,
            "forward_scale": 0.0015,
            "backtrack_scale": 0.0013,
            "climb_up": 0.0032,
            "climb_down": 0.0020,
            "climb_wrong": -0.0022,
            "vertical_scale": 0.0010,
            "vertical_cap": 0.0032,
            "kill_scale": 0.28,
            "boss_scale": 2.6,
            "boss_fire_bonus": 0.0009,
            "damage_scale": -0.012,
            "idle_penalty": -0.0005,
            "stall_penalty": -0.0045,
            "jump_spam_penalty": -0.0010,
            "fire_spam_penalty": -0.0006,
            "alive_bonus": 0.0002,
            "time_penalty": -0.0002,
            "life_lost_penalty": -0.8,
            "win_reward": 7.0,
            "death_penalty": -3.8,
            "pit_penalty": -2.8,
            "spike_penalty": -1.6,
        },
    )


@_wrap_with_tracker
def mega(terminated: bool, info: Info) -> Dict[str, float]:
    return _reward_template(
        terminated,
        info,
        {
            "goal_scale": 0.12,
            "dijkstra_scale": 0.34,
            "route_neg_cap": 0.028,
            "route_pos_cap": 0.052,
            "frontier_scale": 0.0011,
            "frontier_cap": 0.015,
            "forward_scale": 0.0013,
            "backtrack_scale": 0.0014,
            "climb_up": 0.0038,
            "climb_down": 0.0022,
            "climb_wrong": -0.0028,
            "vertical_scale": 0.0012,
            "vertical_cap": 0.0036,
            "kill_scale": 0.42,
            "boss_scale": 3.3,
            "boss_fire_bonus": 0.0012,
            "damage_scale": -0.016,
            "idle_penalty": -0.0007,
            "stall_penalty": -0.0052,
            "jump_spam_penalty": -0.0011,
            "fire_spam_penalty": -0.0008,
            "alive_bonus": 0.0001,
            "time_penalty": -0.0002,
            "life_lost_penalty": -1.2,
            "win_reward": 8.0,
            "death_penalty": -4.4,
            "pit_penalty": -3.2,
            "spike_penalty": -1.8,
        },
    )


default = simple
megaman_simple = simple
megaman_mega = mega
