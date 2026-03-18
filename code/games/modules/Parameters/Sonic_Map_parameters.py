"""
Sonic_Map_parameters.py
-----------------------
Tile constants and color palette for the Sonic NES clone.
Green Hill Zone inspired — bright greens, blues, and checkered brown earth.
"""

TILE_SIZE = 32

# Tile type IDs
TILE_AIR       = 0
TILE_GROUND    = 1
TILE_PLATFORM  = 2
TILE_GOAL      = 3
TILE_SPIKE     = 4
TILE_SPRING    = 5
TILE_LOOP      = 6   # Slope / loop terrain
TILE_CHECKPOINT = 7
TILE_PIT       = 8

# ── Green Hill Zone Palette ──────────────────────────────────────────────────
COLOR_SKY           = (  0, 128, 255)    # Bright Sonic blue sky
COLOR_GROUND        = (139,  90,  43)    # Brown earth
COLOR_GROUND_CHECK  = (115,  66,  23)    # Darker checkered earth pattern
COLOR_GRASS_TOP     = (  0, 200,  50)    # Bright green grass cap
COLOR_PLATFORM      = (180, 140,  80)    # Sandy platform
COLOR_GOAL          = (255, 215,   0)    # Gold signpost
COLOR_SPIKE         = (180, 180, 180)    # Metallic spike
COLOR_SPRING_RED    = (255,  50,  50)    # Red spring top
COLOR_SPRING_YELLOW = (255, 220,   0)    # Yellow spring base
COLOR_CHECKPOINT    = (  0, 100, 255)    # Blue signpost

# Entity colours
COLOR_WHITE      = (255, 255, 255)
COLOR_BLACK      = (  0,   0,   0)
COLOR_RING       = (255, 215,   0)    # Gold ring
COLOR_RING_INNER = (200, 170,   0)    # Darker ring inner
COLOR_BADNIK     = (100, 100, 200)    # Blueish enemy body
COLOR_BADNIK_EYE = (255,   0,   0)    # Red enemy eyes
COLOR_SONIC_BLUE = ( 30,  70, 220)    # Sonic blue
COLOR_SONIC_SKIN = (255, 200, 150)    # Sonic skin/belly
COLOR_SONIC_SHOE = (255,  50,  50)    # Sonic red shoes
COLOR_SONIC_BALL = ( 20,  40, 180)   # Spin ball colour

# UI / Debug
COLOR_HITBOX     = (255,  64,  64)
COLOR_SENSOR     = ( 64, 255, 128)
COLOR_AGENT_PANEL= ( 30,  30,  30)
COLOR_STREAK     = (200, 200, 255)
COLOR_EMPTY      = (  0, 100, 200)    # Sky fill for empty tiles
COLOR_HUD_BG     = (  0,   0,   0)

# Legacy compat aliases (used by shared code like Tile, SpatialHash, etc.)
COLOR_QBLOCK       = COLOR_SPRING_YELLOW
COLOR_ENEMY        = COLOR_BADNIK
COLOR_POWERUP_MUSH = (255, 100, 0)
COLOR_POWERUP_STAR = COLOR_RING
COLOR_COIN         = COLOR_RING
