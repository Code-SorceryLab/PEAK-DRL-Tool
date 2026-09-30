# -*- coding: utf-8 -*-
"""Data loading utilities and path constants for the PEAK dashboard."""

import os
import glob
import ast

import streamlit as st
import pandas as pd
import numpy as np
import yaml

from code.stats import episode_log, registry


# Path constants

CONFIG_PATH = os.path.join(registry.THRESHOLDS_DIR, "default.yaml")
GAME_CONFIG_PATH = "code/games/game_config.yaml"
MEATBOY_CONFIG_PATH = "code/games/meatboy_config.yaml"
LEVELS_ROOT = "code/games/levels"
TILE_SIZE = 32
SOLID_CHARS = set("#=?<>F")


# Loaders

@st.cache_data
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_game_config(path):
    """world -> level-file mapping across every game section (mario top-level,
    nested megaman/sonic, and meatboy's indexed level list)."""
    mapping = {}
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        sections = [cfg] + [v for v in cfg.values()
                            if isinstance(v, dict) and ("levels" in v or "disabled_levels" in v)]
        for sec in sections:
            for section_key in ("levels", "disabled_levels"):
                section = sec.get(section_key, {})
                if isinstance(section, dict):
                    for world_id, world_cfg in section.items():
                        if isinstance(world_cfg, dict) and "file" in world_cfg:
                            mapping[world_id] = world_cfg["file"]
    except Exception:
        pass
    try:  # meatboy: a flat list, worlds are index strings
        with open(MEATBOY_CONFIG_PATH, "r") as f:
            mb = yaml.safe_load(f) or {}
        for i, rel in enumerate(mb.get("levels", [])):
            mapping.setdefault(str(i), rel)
    except Exception:
        pass
    return mapping


@st.cache_data
def load_level_grid(level_file_path):
    """Parse a level .txt file into a 2D numpy array.
    Returns (grid, rows, cols) where grid values:
      1 = solid, -1 = pit/void, 0 = air/empty
    """
    try:
        with open(level_file_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None, 0, 0
    rows_raw = [line.rstrip("\n") for line in lines]
    if not rows_raw:
        return None, 0, 0
    max_cols = max(len(r) for r in rows_raw)
    grid = np.zeros((len(rows_raw), max_cols), dtype=np.int8)
    for r, row_str in enumerate(rows_raw):
        for c, ch in enumerate(row_str):
            if ch in SOLID_CHARS:
                grid[r, c] = 1
            elif ch == "O":
                grid[r, c] = -1
    return grid, len(rows_raw), max_cols


def parse_route(route):
    """A route as a list of (x, y): episode logs already hold lists; older CSVs held strings."""
    if isinstance(route, list):
        return route
    if not isinstance(route, str) or not route.strip():
        return []
    try:
        return ast.literal_eval(route.strip())
    except Exception:
        return []


@st.cache_data
def load_all_csvs(patterns):
    """Every episode log matching the glob patterns (thresholds default.yaml: dashboard_paths)."""
    patterns = patterns if isinstance(patterns, list) else [patterns]
    files = sorted({f for p in patterns for f in glob.glob(p, recursive=True)})
    rows = [r for fp in files for r in episode_log.read(fp)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # level ids are text ("0".."14" for the indexed games); "world" is the dashboard's name for it
    for col in ("level", "persona", "game", "source"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["world"] = df["level"]
    return df


def episodes(df, game, world, persona=None):
    """Episode rows (dicts, as the registry takes them) for one game + level [+ persona]."""
    sub = df[(df["game"] == game) & (df["world"] == world)]
    if persona is not None:
        sub = sub[sub["persona"] == persona]
    return [{k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))}
            for r in sub.to_dict("records")]


# Helpers

def win_rate(df, game, world, persona):
    """Completion rate of one persona on one level (the registry's completion_rate)."""
    eps = episodes(df, game, world, persona)
    return registry.completion_rate(eps, None) if eps else None
