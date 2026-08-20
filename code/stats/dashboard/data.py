# -*- coding: utf-8 -*-
"""Data loading utilities and path constants for the PEAK dashboard."""

import os
import glob
import ast

import streamlit as st
import pandas as pd
import numpy as np
import yaml


# Path constants

CONFIG_PATH = "code/stats/MarioThresholds.yaml"
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


def parse_route(route_str):
    """Parse a route string like '[(x,y), ...]' into a list of (x,y) floats."""
    if not isinstance(route_str, str) or not route_str.strip():
        return []
    try:
        return ast.literal_eval(route_str.strip())
    except Exception:
        return []


@st.cache_data
def load_all_csvs(data_path):
    """Read all CSVs (recursively) from one path or a list of paths."""
    paths = data_path if isinstance(data_path, list) else [data_path]
    files = []
    for p in paths:
        files += glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)
    if not files:
        return pd.DataFrame()
    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # meatboy worlds are numeric strings ("0".."10") — pandas infers int per-file,
    # mario infers str, and sorted() on the mix crashes. Normalize once here.
    for col in ("world", "persona", "game"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


# Helpers

def win_rate(df, world, persona):
    """Win rate = fraction of runs where cause_of_death == 'Success'."""
    sub = df[(df["world"] == world) & (df["persona"] == persona)]
    if len(sub) == 0:
        return None
    successes = sub["cause_of_death"].str.lower()
    wins = (successes == "success").sum()
    return wins / len(sub)
