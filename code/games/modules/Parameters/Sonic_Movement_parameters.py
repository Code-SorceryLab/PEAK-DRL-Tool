"""
Sonic_Movement_parameters.py
-----------------------------
Movement constants tuned for Sonic-style gameplay.

Key differences from the platformer:
  - Much higher top speed (Sonic is FAST)
  - Stronger acceleration, especially when rolling
  - Lower friction so momentum carries
  - Gravity tuned for floaty Sonic arcs
  - Spin dash adds a burst velocity
"""

# ── Base Movement ────────────────────────────────────────────────────────────
RUN_ACCEL        = 300.0     # Ground acceleration (holding right)
WALK_ACCEL       = 200.0     # Walk acceleration (no run held)
MAX_WALK_SPEED   = 180.0     # Walk cap
MAX_RUN_SPEED    = 400.0     # Sprint cap (holding shift / sprint)
TOP_SPEED        = 600.0     # Absolute max (spin dash / downhill)

# ── Friction ─────────────────────────────────────────────────────────────────
GROUND_FRICTION  = 600.0     # Lower than platformer → momentum carries
AIR_FRICTION     = 100.0     # Very low air friction (Sonic floats)
ROLL_FRICTION    = 200.0     # Rolling friction (slower decel while balled up)

# ── Air Control ──────────────────────────────────────────────────────────────
AIR_CONTROL      = 0.40      # More air control than Mario

# ── Skidding ─────────────────────────────────────────────────────────────────
SKID_DECEL       = 1800.0    # Quick direction change

# ── Gravity ──────────────────────────────────────────────────────────────────
GRAVITY          = 1100.0    # Slightly lower than platformer (floatier arcs)
FAST_FALL_GRAV   = 2000.0    # Fast fall gravity
MAX_FALL_SPEED   = 600.0     # Terminal velocity

# ── Spin Dash ────────────────────────────────────────────────────────────────
SPIN_DASH_MIN    = 300.0     # Minimum spin dash release speed
SPIN_DASH_MAX    = 560.0     # Maximum spin dash speed (fully charged)
SPIN_DASH_CHARGE_RATE = 60.0 # Speed added per charge tap
SPIN_DASH_CHARGES_MAX = 8    # Max charge taps

# ── Rolling ──────────────────────────────────────────────────────────────────
ROLL_SPEED_THRESHOLD = 60.0  # Min speed to initiate a roll
ROLL_DECEL       = 150.0     # Deceleration while rolling on flat ground
ROLL_UPHILL_DECEL = 300.0    # Extra decel going uphill while rolling
ROLL_DOWNHILL_ACCEL = 200.0  # Bonus accel going downhill while rolling

# ── Spring Bounce ────────────────────────────────────────────────────────────
SPRING_BOUNCE_VEL = -700.0   # Vertical spring launch velocity
SPRING_BOUNCE_WEAK = -450.0  # Yellow spring (weaker)
