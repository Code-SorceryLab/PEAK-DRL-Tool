"""
Sonic_Jump_parameters.py
------------------------
Jump constants for Sonic gameplay.

Sonic's jump:
  - Higher arc than Mario
  - Variable height (tap vs hold)
  - Jump velocity scales with ground speed (running jump goes higher)
  - Generous coyote time and jump buffering
"""

# Jump velocities (negative = upward in screen coords)
JUMP_VEL_MIN       = -520.0    # Minimum jump (tap)
JUMP_VEL_MAX       = -700.0    # Maximum jump (full hold)

# Variable jump hold
JUMP_HOLD_FRAMES   = 20        # Longer hold window than platformer

# Speed-dependent jump bonus
SPEED_JUMP_BONUS   = 0.12      # Faster = higher jump

# Coyote time & buffer
COYOTE_FRAMES      = 6         # Generous coyote frames
JUMP_BUFFER_FRAMES = 8         # Generous buffer

# Ball bounce (when hitting enemy from above while rolling)
BOUNCE_VEL         = -400.0    # Bounce off enemy
